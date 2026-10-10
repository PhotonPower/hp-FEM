/// Bindings of `physics`: analytic incident fields, the scattering problem (setup, solve,
/// field evaluation, errors, estimator), parameter sweeps and waveguide modes.
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "common.hpp"
#include "hpfem/adaptivity/residual_estimator.hpp"
#include "hpfem/physics/axisymmetric.hpp"
#include "hpfem/physics/band_sensitivity.hpp"
#include "hpfem/physics/band_structure.hpp"
#include "hpfem/physics/conical_resonance.hpp"
#include "hpfem/physics/conical_scattering.hpp"
#include "hpfem/physics/conical_sweep.hpp"
#include "hpfem/physics/dipole_emission.hpp"
#include "hpfem/physics/field_sampling.hpp"
#include "hpfem/physics/kept_factorisation.hpp"
#include "hpfem/physics/propagating_mode.hpp"
#include "hpfem/physics/resonance.hpp"
#include "hpfem/physics/riesz_projection.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"
#include "hpfem/physics/sweep.hpp"
#include "hpfem/physics/thermal.hpp"
#include "hpfem/physics/thermo_optical.hpp"
#include "hpfem/physics/time_domain.hpp"

namespace hpfem::python {

namespace {

/// Sampling options from keyword arguments.
[[nodiscard]] inline physics::SamplingOptions sampling_options(bool scattered, bool bloch_wrap,
                                                               int interface_side,
                                                               const std::string& quantity) {
  physics::SamplingOptions o;
  o.scattered = scattered;
  o.bloch_wrap = bloch_wrap;
  o.interface_side = interface_side;
  if (quantity == "E") {
    o.quantity = physics::SampledQuantity::kElectric;
  } else if (quantity == "H") {
    o.quantity = physics::SampledQuantity::kMagnetic;
  } else if (quantity == "S") {
    o.quantity = physics::SampledQuantity::kPoynting;
  } else {
    throw InvalidArgument("quantity must be 'E', 'H' or 'S'");
  }
  return o;
}

/// (values (n, c) complex, cells (n,) int64) of a sampled field.
[[nodiscard]] inline py::tuple sampled_to_python(const physics::SampledField& field) {
  const auto rows = static_cast<py::ssize_t>(field.values.rows());
  const auto cols = static_cast<py::ssize_t>(field.values.cols());
  py::array_t<Complex> values({rows, cols});
  auto r = values.mutable_unchecked<2>();
  for (py::ssize_t i = 0; i < rows; ++i) {
    for (py::ssize_t j = 0; j < cols; ++j) r(i, j) = field.values(i, j);
  }
  return py::make_tuple(values, to_array(field.cells));
}

template <int Dim>
void bind_triangulated_field(py::module_& m) {
  using T = physics::TriangulatedField<Dim>;
  py::class_<T>(m, named("TriangulatedField", Dim).c_str(),
                "A field on the subdivided mesh: points (n, dim), simplices (m, dim + 1) by "
                "point index, values (n, components) complex, cell and tag (m,) per simplex; "
                "points are not shared between parent cells, so discontinuities are kept. "
                "2D: matplotlib.tri.Triangulation(points[:, 0], points[:, 1], simplices)")
      .def_property_readonly(
          "points",
          [](const T& t) { return points_to_array<Dim>(std::span<const Point<Dim>>(t.points)); })
      .def_property_readonly(
          "simplices",
          [](const T& t) {
            return tuples_to_array<Index, static_cast<std::size_t>(Dim + 1)>(
                std::span<const std::array<Index, static_cast<std::size_t>(Dim + 1)>>(t.simplices));
          })
      .def_property_readonly("values",
                             [](const T& t) {
                               const auto rows = static_cast<py::ssize_t>(t.values.rows());
                               const auto cols = static_cast<py::ssize_t>(t.values.cols());
                               py::array_t<Complex> values({rows, cols});
                               auto r = values.mutable_unchecked<2>();
                               for (py::ssize_t i = 0; i < rows; ++i) {
                                 for (py::ssize_t j = 0; j < cols; ++j) r(i, j) = t.values(i, j);
                               }
                               return values;
                             })
      .def_property_readonly("cell", [](const T& t) { return to_array(t.cell); })
      .def_property_readonly("tag", [](const T& t) { return to_array(t.tag); })
      .def("__len__", [](const T& t) { return t.simplices.size(); });
}

template <int Dim>
void bind_physics_dim(py::module_& m) {
  using assembly::ComplexVector;
  using physics::IncidentField;
  using physics::Scattering;
  using physics::ScatteringOperator;
  using physics::ScatteringSetup;
  using physics::ScatteringSolution;
  using ND = fespace::NedelecDofMap<Dim>;

  py::class_<IncidentField<Dim>>(m, named("IncidentField", Dim).c_str(),
                                 "Analytic field given as value(x) and curl(x) callables "
                                 "(Dirichlet data, scattered-field source, reference solution)")
      .def(py::init(
               [](assembly::ComplexVectorField<Dim> value, assembly::ComplexCurlField<Dim> curl) {
                 IncidentField<Dim> f;
                 f.value = std::move(value);
                 f.curl = std::move(curl);
                 return f;
               }),
           py::arg("value") = py::none(), py::arg("curl") = py::none())
      .def_readwrite("value", &IncidentField<Dim>::value)
      .def_readwrite("curl", &IncidentField<Dim>::curl)
      .def("__bool__", [](const IncidentField<Dim>& f) { return static_cast<bool>(f); });
  m.def(
      "plane_wave",
      [](const ComplexVector<Dim>& amplitude, const Point<Dim>& k) {
        return physics::plane_wave<Dim>(amplitude, k);
      },
      py::arg("amplitude"), py::arg("wave_vector"),
      "E0 exp(i k . x) with real wave vector k [1/m] and complex amplitude E0 perpendicular to k");
  m.def(
      "gaussian_current",
      [](const Point<Dim>& x0, const ComplexVector<Dim>& moment, Real sigma, Real omega) {
        return physics::gaussian_current<Dim>(x0, moment, sigma, omega);
      },
      py::arg("position"), py::arg("moment"), py::arg("sigma"), py::arg("omega"),
      "Volume source f = i omega mu0 J of a dipole smeared over a normalised Gaussian of "
      "width sigma, evaluated in C++ (for ScatteringSetup.current in the total-field "
      "formulation: an emitter inside a structure)");
  m.def(
      "dipole_field",
      [](const Point<Dim>& x0, const ComplexVector<Dim>& moment, Real k) {
        return physics::dipole_field<Dim>(x0, moment, k);
      },
      py::arg("position"), py::arg("moment"), py::arg("k"),
      "Outgoing field of a point (3D) / line (2D) dipole at x0 in a background of wavenumber k");

  py::class_<ScatteringSetup<Dim>>(
      m, named("ScatteringSetup", Dim).c_str(),
      "Description of a scattering problem at angular frequency omega [rad/s]: materials by "
      "cell tag, analytic incident field, formulation (total / scattered field), PEC and "
      "incident-trace facet tags, current source f = i omega mu0 J (callable), PML box, "
      "Bloch-periodic directions, solver backend and quadrature extras")
      .def(py::init<>())
      .def_readwrite("omega", &ScatteringSetup<Dim>::omega)
      .def_readwrite("materials", &ScatteringSetup<Dim>::materials)
      .def_readwrite("incident", &ScatteringSetup<Dim>::incident)
      .def_readwrite("incident_wave", &ScatteringSetup<Dim>::incident_wave,
                     "the incident plane wave alone on a layered background "
                     "(LayeredPlaneWave.incident_wave); empty means incident")
      .def_readwrite("background", &ScatteringSetup<Dim>::background,
                     "optional LayerStack: layered background of the scattered-field "
                     "formulation (ADR-0009); set incident to its plane_wave(...).field")
      .def_readwrite("formulation", &ScatteringSetup<Dim>::formulation)
      .def_readwrite("pec_tags", &ScatteringSetup<Dim>::pec_tags)
      .def_readwrite("incident_tags", &ScatteringSetup<Dim>::incident_tags,
                     "facets with n x E = n x E_inc (test domains)")
      .def_readwrite("current", &ScatteringSetup<Dim>::current)
      .def_readwrite("pml", &ScatteringSetup<Dim>::pml)
      .def_readwrite("periodic", &ScatteringSetup<Dim>::periodic)
      .def_readwrite("ports", &ScatteringSetup<Dim>::ports,
                     "waveguide ports (modal absorption and excitation; 2D)")
      .def_readwrite("solver", &ScatteringSetup<Dim>::solver)
      .def_readwrite("condense", &ScatteringSetup<Dim>::condense)
      .def_readwrite(
          "keep_factorisation", &ScatteringSetup<Dim>::keep_factorisation,
          "keep the factorised system in the solution (solution.factorisation) for adjoint and "
          "tangent solves of the sensitivities; holds the factors while the solution lives")
      .def_readwrite("progress", &ScatteringSetup<Dim>::progress,
                     "callback(event) -> bool at the start of every phase of solve and when "
                     "done; False cancels (hpfem.Cancelled)")
      .def_readwrite("extra_quadrature_order", &ScatteringSetup<Dim>::extra_quadrature_order)
      .def_readwrite("pml_extra_quadrature_order",
                     &ScatteringSetup<Dim>::pml_extra_quadrature_order);
  py::class_<ScatteringSolution<Dim>>(m, named("ScatteringSolution", Dim).c_str(),
                                      "Coefficients of the unknown field (E or E_sc)")
      .def(py::init([](physics::Formulation f, Vector unknown) {
             return ScatteringSolution<Dim>{f, std::move(unknown), {}, nullptr};
           }),
           py::arg("formulation"), py::arg("unknown"))
      .def_readwrite("formulation", &ScatteringSolution<Dim>::formulation)
      .def_readwrite("unknown", &ScatteringSolution<Dim>::unknown)
      .def_readonly("timing", &ScatteringSolution<Dim>::timing,
                    "seconds per phase of solve and 'total'")
      .def_property_readonly(
          "factorisation",
          [](const ScatteringSolution<Dim>& s) {
            return std::const_pointer_cast<physics::KeptFactorisation>(s.factorisation);
          },
          "the kept factorised system (KeptFactorisation) or None");

  py::class_<Scattering<Dim>>(
      m, named("Scattering", Dim).c_str(),
      "Time-harmonic scattering on a Nédélec space: curl(mu_r^-1 curl E) - k0^2 eps_r E = f; "
      "assembles with static condensation, applies PEC / incident traces, hanging-node and "
      "Bloch constraints, PML as stretched tensors, and solves with a direct solver")
      .def(py::init<const ND&, ScatteringSetup<Dim>>(), py::arg("dofs"), py::arg("setup"),
           py::keep_alive<1, 2>())
      .def_property_readonly("dofs", &Scattering<Dim>::dofs,
                             py::return_value_policy::reference_internal)
      .def("background_material", &Scattering<Dim>::background_material, py::arg("cell"),
           "material of the (layered) background at the cell centroid")
      .def_property_readonly("incidence_material", &Scattering<Dim>::incidence_material)
      .def_property_readonly("setup", &Scattering<Dim>::setup,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("wavenumber", &Scattering<Dim>::wavenumber, "k0 = omega / c0 [1/m]")
      .def("material", &Scattering<Dim>::material, py::arg("cell"))
      .def("form_of_cell", &Scattering<Dim>::form_of_cell, py::arg("cell"),
           "MaxwellForm of the cell (relative tensors, PML-stretched, sources of the formulation)")
      .def("dirichlet", &Scattering<Dim>::dirichlet, Release(),
           "Dirichlet data on the PEC and incident facets")
      .def("assemble_raw", &Scattering<Dim>::assemble_raw, Release(),
           "A = S - k0^2 M and the load without boundary conditions")
      .def("assemble", &Scattering<Dim>::assemble, Release(),
           "A = S - k0^2 M and the load with the Dirichlet data applied")
      .def("constraints", &Scattering<Dim>::constraints, Release(),
           "hanging-node followed by Bloch constraints")
      .def("solve", &Scattering<Dim>::solve, Release())
      .def("port_modes", &Scattering<Dim>::port_modes, py::arg("port"),
           py::return_value_policy::reference_internal, "modes of port p of the setup")
      .def("port_coefficients", &Scattering<Dim>::port_coefficients, py::arg("solution"), Release(),
           "incoming and outgoing modal amplitudes on every port")
      .def(
          "total_field",
          [](const Scattering<Dim>& p, const ScatteringSolution<Dim>& s, Index c,
             const Point<Dim>& xi) { return p.total_field(s, c, xi); },
          py::arg("solution"), py::arg("cell"), py::arg("xi"))
      .def(
          "total_field",
          [](const Scattering<Dim>& p, const ScatteringSolution<Dim>& s,
             const mesh::PointLocator<Dim>& locator,
             const Point<Dim>& x) { return p.total_field(s, locator, x); },
          py::arg("solution"), py::arg("locator"), py::arg("x"), "None outside the mesh")
      .def(
          "scattered_field",
          [](const Scattering<Dim>& p, const ScatteringSolution<Dim>& s, Index c,
             const Point<Dim>& xi) { return p.scattered_field(s, c, xi); },
          py::arg("solution"), py::arg("cell"), py::arg("xi"))
      .def(
          "scattered_field",
          [](const Scattering<Dim>& p, const ScatteringSolution<Dim>& s,
             const mesh::PointLocator<Dim>& locator,
             const Point<Dim>& x) { return p.scattered_field(s, locator, x); },
          py::arg("solution"), py::arg("locator"), py::arg("x"), "None outside the mesh")
      .def(
          "sample",
          [](const Scattering<Dim>& p, const ScatteringSolution<Dim>& s,
             const mesh::PointLocator<Dim>& locator, const RealArray& points, bool scattered,
             bool bloch_wrap, int interface_side, const std::string& quantity) {
            const auto pts = array_to_points<Dim>(points);
            const auto options = sampling_options(scattered, bloch_wrap, interface_side, quantity);
            physics::SampledField field;
            {
              py::gil_scoped_release release;
              field = physics::sample_field<Dim>(p, s, locator, pts, options);
            }
            return sampled_to_python(field);
          },
          py::arg("solution"), py::arg("locator"), py::arg("points"), py::arg("scattered") = false,
          py::arg("bloch_wrap") = true, py::arg("interface_side") = 0, py::arg("quantity") = "E",
          "total (or scattered) field at points (n, dim) in parallel: (values (n, dim) complex, "
          "cells (n,)); points outside a Bloch-periodic direction are wrapped back with the "
          "Bloch phase, points outside the mesh give NaN and cell -1; interface_side +1 / -1 "
          "evaluates a point on a facet in the cell above / below (last coordinate)")
      .def(
          "triangulate",
          [](const Scattering<Dim>& p, const ScatteringSolution<Dim>& s, int subdivisions,
             bool scattered, const std::string& quantity) {
            return physics::triangulate_field<Dim>(p, s, subdivisions,
                                                   sampling_options(scattered, true, 0, quantity));
          },
          py::arg("solution"), py::arg("subdivisions") = 2, py::arg("scattered") = false,
          py::arg("quantity") = "E", Release(),
          "the field on the subdivisions-fold subdivided mesh as a TriangulatedField")
      .def(
          "error",
          [](const Scattering<Dim>& p, const ScatteringSolution<Dim>& s,
             const IncidentField<Dim>& exact) { return p.error(s, exact); },
          py::arg("solution"), py::arg("exact"), Release(),
          "Error norms of the unknown field against an exact field of the same kind")
      .def(
          "error",
          [](const Scattering<Dim>& p, const ScatteringSolution<Dim>& s,
             const IncidentField<Dim>& exact,
             const std::vector<Index>& cells) { return p.error(s, exact, cells); },
          py::arg("solution"), py::arg("exact"), py::arg("cells"), Release())
      .def("incident_wave", &Scattering<Dim>::incident_wave, py::arg("x"),
           "the incident wave at x: setup.incident_wave if given, else setup.incident")
      .def(
          "interior_cells", [](const Scattering<Dim>& p) { return to_array(p.interior_cells()); },
          "cells inside the PML box (all cells without PML)")
      .def("estimate", &Scattering<Dim>::estimate, py::arg("solution"),
           py::arg("options") = adaptivity::EstimatorOptions{}, Release(),
           "Residual-based element indicators of the solution");

  py::class_<physics::PortModes<Dim>>(m, named("PortModes", Dim).c_str(),
                                      "Modes, functionals and normalisations of one port")
      .def(py::init<const ND&, mesh::Tag, const materials::MaterialMap&, Real, Index, int>(),
           py::arg("dofs"), py::arg("facet_tag"), py::arg("materials"), py::arg("omega"),
           py::arg("num_modes"), py::arg("extra_order") = 2, py::keep_alive<1, 2>(), Release())
      .def_property_readonly("modes", &physics::PortModes<Dim>::modes)
      .def_property_readonly("num_modes", &physics::PortModes<Dim>::num_modes)
      .def_property_readonly("length", &physics::PortModes<Dim>::length)
      .def_property_readonly("origin", &physics::PortModes<Dim>::origin)
      .def_property_readonly("tangent", &physics::PortModes<Dim>::tangent)
      .def_property_readonly("normal", &physics::PortModes<Dim>::normal)
      .def("functional", &physics::PortModes<Dim>::functional, py::arg("m"),
           py::return_value_policy::copy, "q_m on the DoF map")
      .def("normalisation", &physics::PortModes<Dim>::normalisation, py::arg("m"))
      .def("coefficients", &physics::PortModes<Dim>::coefficients, py::arg("e"),
           "c_m = q_m^T e / N_m of a discrete field")
      .def("trace", &physics::PortModes<Dim>::trace, py::arg("m"), py::arg("s"),
           "tangential trace E . t' of mode m at the port coordinate s")
      .def("profile", &physics::PortModes<Dim>::profile, py::arg("m"), py::arg("s"),
           "H_z profile of mode m at the port coordinate s")
      .def("transverse_field", &physics::PortModes<Dim>::transverse_field, py::arg("m"),
           py::arg("x"), "tangential electric field of mode m at the point x on the port")
      .def_property_readonly("tangent2", &physics::PortModes<Dim>::tangent2)
      .def_property_readonly(
          "section",
          [](const physics::PortModes<Dim>& p) -> std::optional<mesh::Mesh<2>> {
            if (p.section() == nullptr) return std::nullopt;
            return *p.section();
          },
          "3D: the extracted cross-section mesh in frame coordinates (None in 2D)");
  m.def(
      "s_parameters",
      [](const ND& dofs, ScatteringSetup<Dim> setup) {
        return physics::s_parameters<Dim>(dofs, std::move(setup));
      },
      py::arg("dofs"), py::arg("setup"), Release(),
      "Power-normalised S-matrix of the ports of the setup (one solve per propagating channel)");
  py::class_<ScatteringOperator<Dim>>(
      m, named("ScatteringOperator", Dim).c_str(),
      "The operator of a scattering problem assembled, constrained and factorised once; "
      "solves for any incident field / current at the same frequency with a new load only")
      .def(py::init<const Scattering<Dim>&>(), py::arg("problem"), py::keep_alive<1, 2>(),
           Release())
      .def_property_readonly("problem", &ScatteringOperator<Dim>::problem,
                             py::return_value_policy::reference_internal)
      .def(
          "solve", [](const ScatteringOperator<Dim>& op) { return op.solve(); }, Release(),
          "solution of the problem's own setup")
      .def(
          "solve",
          [](const ScatteringOperator<Dim>& op, const IncidentField<Dim>& incident,
             const assembly::ComplexVectorField<Dim>& current) {
            return op.solve(incident, current);
          },
          py::arg("incident"), py::arg("current") = py::none(), Release(),
          "solution for another incident field (and current)")
      .def("solve_port", &ScatteringOperator<Dim>::solve_port, py::arg("port"), py::arg("mode"),
           Release(),
           "solution for the unit excitation of one port mode on the factorised operator (the "
           "port terms are part of the operator); what s_parameters uses per channel")
      .def(
          "solve_many",
          [](const ScatteringOperator<Dim>& op, const std::vector<IncidentField<Dim>>& incidents,
             const assembly::ComplexVectorField<Dim>& current) {
            return op.solve_many(incidents, current);
          },
          py::arg("incidents"), py::arg("current") = py::none(), Release(),
          "solutions for several incident fields in one batched solve");
  using physics::Resonance;
  using physics::ResonanceSetup;
  py::class_<ResonanceSetup<Dim>>(
      m, named("ResonanceSetup", Dim).c_str(),
      "Resonance (quasi-normal mode) problem: target angular frequency (search centre and "
      "PML design frequency), materials by tag (may be lossy), PEC facet tags, PML box, "
      "number of modes, Arnoldi settings")
      .def(py::init<>())
      .def_readwrite("target_omega", &ResonanceSetup<Dim>::target_omega)
      .def_readwrite("materials", &ResonanceSetup<Dim>::materials)
      .def_readwrite("pec_tags", &ResonanceSetup<Dim>::pec_tags)
      .def_readwrite("pml", &ResonanceSetup<Dim>::pml)
      .def_readwrite("num_modes", &ResonanceSetup<Dim>::num_modes)
      .def_readwrite("krylov_dimension", &ResonanceSetup<Dim>::krylov_dimension)
      .def_readwrite("tolerance", &ResonanceSetup<Dim>::tolerance)
      .def_readwrite("max_iterations", &ResonanceSetup<Dim>::max_iterations)
      .def_readwrite("solver", &ResonanceSetup<Dim>::solver)
      .def_readwrite("extra_quadrature_order", &ResonanceSetup<Dim>::extra_quadrature_order)
      .def_readwrite("pml_extra_quadrature_order",
                     &ResonanceSetup<Dim>::pml_extra_quadrature_order);
  py::class_<Resonance<Dim>>(m, named("Resonance", Dim).c_str(),
                             "Assembles the pencil (S, M) with PEC and PML and finds the "
                             "complex eigenfrequencies closest to the target")
      .def(py::init<const ND&, ResonanceSetup<Dim>>(), py::arg("dofs"), py::arg("setup"),
           py::keep_alive<1, 2>())
      .def_property_readonly("dofs", &Resonance<Dim>::dofs,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("setup", &Resonance<Dim>::setup,
                             py::return_value_policy::reference_internal)
      .def("form_of_cell", &Resonance<Dim>::form_of_cell, py::arg("cell"))
      .def("solve", &Resonance<Dim>::solve, Release(),
           "modes ordered by the distance of omega to the target");
  using physics::Thermal;
  using H1 = fespace::DofMap<Dim>;
  m.def(
      "absorbed_power_density",
      [](const ND& dofs, const Vector& e, Real omega, const materials::MaterialMap& materials,
         const H1& h1) {
        return physics::absorbed_power_density<Dim>(dofs, e, omega, materials, h1);
      },
      py::arg("dofs"), py::arg("e"), py::arg("omega"), py::arg("materials"), py::arg("h1"),
      Release(),
      "Absorbed power density omega eps0 / 2 Im(eps_r) |E|^2 [W/m^3] as the H1 interpolant on "
      "the same mesh (for export; jumps at interfaces are smeared over a cell)");
  m.def(
      "absorbed_power_load",
      [](const ND& dofs, const Vector& e, Real omega, const materials::MaterialMap& materials,
         const H1& h1, int extra_order) {
        return physics::absorbed_power_load<Dim>(dofs, e, omega, materials, h1, extra_order);
      },
      py::arg("dofs"), py::arg("e"), py::arg("omega"), py::arg("materials"), py::arg("h1"),
      py::arg("extra_order") = 2, Release(),
      "Load vector of the absorbed power on the H1 map (cell-wise exact), the right-hand "
      "side of Thermal.solve_load");
  py::class_<Thermal<Dim>>(m, named("Thermal", Dim).c_str(),
                           "Steady heat conduction -div(kappa grad T) = q on an H1 space with "
                           "conductivities by cell tag, fixed temperatures on tagged facets "
                           "and adiabatic walls elsewhere")
      .def(py::init<const H1&, physics::ThermalSetup>(), py::arg("dofs"), py::arg("setup"),
           py::keep_alive<1, 2>())
      .def_property_readonly("dofs", &Thermal<Dim>::dofs,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("setup", &Thermal<Dim>::setup,
                             py::return_value_policy::reference_internal)
      .def("conductivity", &Thermal<Dim>::conductivity, py::arg("cell"))
      .def("stiffness", &Thermal<Dim>::stiffness, Release())
      .def("mass", &Thermal<Dim>::mass, Release())
      .def("solve", &Thermal<Dim>::solve, py::arg("q"), Release(),
           "temperature for source coefficients q on the map (K T = M q)")
      .def("solve_load", &Thermal<Dim>::solve_load, py::arg("load"), Release(),
           "temperature for an assembled load, e.g. absorbed_power_load")
      .def("total_power", &Thermal<Dim>::total_power, py::arg("q"), Release(),
           "integral of source coefficients q [W] (3D) or [W/m] (2D)");
  using physics::ThermoOptical;
  using physics::ThermoOpticalSetup;
  using physics::ThermoOpticalState;
  py::class_<ThermoOpticalSetup<Dim>>(
      m, named("ThermoOpticalSetup", Dim).c_str(),
      "Coupled optical-thermal problem: the scattering setup (materials at the reference "
      "temperature), the thermal setup, complex thermo-optic coefficients d eps_r / dT by "
      "cell tag, reference temperature, iteration limit, tolerance [K] and relaxation")
      .def(py::init<>())
      .def_readwrite("optical", &ThermoOpticalSetup<Dim>::optical)
      .def_readwrite("thermal", &ThermoOpticalSetup<Dim>::thermal)
      .def_readwrite("thermo_optic", &ThermoOpticalSetup<Dim>::thermo_optic)
      .def_readwrite("reference_temperature", &ThermoOpticalSetup<Dim>::reference_temperature)
      .def_readwrite("max_iterations", &ThermoOpticalSetup<Dim>::max_iterations)
      .def_readwrite("tolerance", &ThermoOpticalSetup<Dim>::tolerance)
      .def_readwrite("relaxation", &ThermoOpticalSetup<Dim>::relaxation);
  py::class_<ThermoOpticalState<Dim>>(m, named("ThermoOpticalState", Dim).c_str(),
                                      "Converged (or last) state of the feedback loop")
      .def_readonly("solution", &ThermoOpticalState<Dim>::solution)
      .def_readonly("temperature", &ThermoOpticalState<Dim>::temperature)
      .def_readonly("materials", &ThermoOpticalState<Dim>::materials)
      .def_readonly("absorbed_power", &ThermoOpticalState<Dim>::absorbed_power)
      .def_readonly("history", &ThermoOpticalState<Dim>::history, "max |dT| per iteration")
      .def_readonly("iterations", &ThermoOpticalState<Dim>::iterations)
      .def_readonly("converged", &ThermoOpticalState<Dim>::converged);
  py::class_<ThermoOptical<Dim>>(m, named("ThermoOptical", Dim).c_str(),
                                 "Optical-thermal feedback loop (fixed-point iteration) on a "
                                 "Nédélec and an H1 space of the same mesh")
      .def(py::init<const ND&, const H1&, ThermoOpticalSetup<Dim>>(), py::arg("optical_dofs"),
           py::arg("thermal_dofs"), py::arg("setup"), py::keep_alive<1, 2>(),
           py::keep_alive<1, 3>())
      .def_property_readonly("setup", &ThermoOptical<Dim>::setup,
                             py::return_value_policy::reference_internal)
      .def("materials_at", &ThermoOptical<Dim>::materials_at, py::arg("temperature"),
           "materials with the permittivities of a temperature field (per-cell overrides)")
      .def("total_field", &ThermoOptical<Dim>::total_field, py::arg("solution"), Release(),
           "coefficients of the total field of a solution on the optical map")
      .def("solve", &ThermoOptical<Dim>::solve, Release());
  m.def(
      "solve_many",
      [](const Scattering<Dim>& problem, const std::vector<IncidentField<Dim>>& incidents) {
        return physics::solve_many<Dim>(problem, incidents);
      },
      py::arg("problem"), py::arg("incidents"), Release(),
      "Solutions for several incident fields with one factorisation");
  m.def(
      "plane_wave_sweep",
      [](const Scattering<Dim>& problem, const std::vector<Point<Dim>>& wave_vectors,
         const std::function<ComplexVector<Dim>(const Point<Dim>&)>& polarisation) {
        return physics::plane_wave_sweep<Dim>(problem, wave_vectors, polarisation);
      },
      py::arg("problem"), py::arg("wave_vectors"), py::arg("polarisation"), Release(),
      "Angle sweep: plane waves of the given wave vectors with polarisation(k)");
  using physics::Bands;
  using physics::BandStructure;
  using physics::BandStructureSetup;
  py::class_<BandStructureSetup<Dim>>(
      m, named("BandStructureSetup", Dim).c_str(),
      "Floquet-Bloch band-structure problem: lossless materials by tag, the lattice as one "
      "PeriodicPair per lattice vector (phases are set per wave vector), optional PEC walls, "
      "the number of bands and the shift-invert shift in units of (2 pi / a)^2")
      .def(py::init<>())
      .def_readwrite("materials", &BandStructureSetup<Dim>::materials)
      .def_readwrite("lattice", &BandStructureSetup<Dim>::lattice)
      .def_readwrite("pec_tags", &BandStructureSetup<Dim>::pec_tags)
      .def_readwrite("num_bands", &BandStructureSetup<Dim>::num_bands)
      .def_readwrite("shift", &BandStructureSetup<Dim>::shift)
      .def_readwrite("krylov_dimension", &BandStructureSetup<Dim>::krylov_dimension)
      .def_readwrite("tolerance", &BandStructureSetup<Dim>::tolerance)
      .def_readwrite("max_iterations", &BandStructureSetup<Dim>::max_iterations)
      .def_readwrite("solver", &BandStructureSetup<Dim>::solver)
      .def_readwrite("extra_quadrature_order", &BandStructureSetup<Dim>::extra_quadrature_order)
      .def_readwrite("keep_modes", &BandStructureSetup<Dim>::keep_modes,
                     "keep the eigenvectors in Bands.modes (needed by the band derivatives)");
  py::class_<Bands<Dim>>(m, named("Bands", Dim).c_str(),
                         "The bands at one Bloch wave vector: wavenumbers k0 ascending [1/m]")
      .def_readonly("wave_vector", &Bands<Dim>::wave_vector)
      .def_readonly("wavenumber", &Bands<Dim>::wavenumber)
      .def_readonly("residual", &Bands<Dim>::residual)
      .def_readonly("modes", &Bands<Dim>::modes,
                    "full-size eigenvectors (num_dofs, num_bands), w^H M w = 1; empty unless "
                    "setup.keep_modes")
      .def("normalised", &Bands<Dim>::normalised, py::arg("lattice_constant"),
           "omega a / (2 pi c0) = k0 a / (2 pi)");
  py::class_<BandStructure<Dim>>(
      m, named("BandStructure", Dim).c_str(),
      "Assembles the lossless pencil once and solves the Bloch eigenproblem per wave vector "
      "with the gradient kernel projected out (no spurious modes)")
      .def(py::init<const ND&, const H1&, BandStructureSetup<Dim>>(), py::arg("dofs"),
           py::arg("h1"), py::arg("setup"), py::keep_alive<1, 2>(), py::keep_alive<1, 3>())
      .def_property_readonly("setup", &BandStructure<Dim>::setup,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("lattice_constant", &BandStructure<Dim>::lattice_constant)
      .def("bands", &BandStructure<Dim>::bands, py::arg("wave_vector"), Release(),
           "bands at the Bloch wave vector k [1/m]")
      .def("path", &BandStructure<Dim>::path, py::arg("corners"), py::arg("segments"), Release(),
           "bands along the polyline of wave vectors, corners included");
  m.def("band_permittivity_derivative", &physics::band_permittivity_derivative<Dim>,
        py::arg("problem"), py::arg("bands"), py::arg("tag"),
        py::arg("degeneracy_tolerance") = 1e-6, Release(),
        "BandDerivative of the bands (kept modes) with respect to eps_r of the cells with the tag "
        "(Hellmann-Feynman; degenerate clusters resolved)");
  m.def("band_permeability_derivative", &physics::band_permeability_derivative<Dim>,
        py::arg("problem"), py::arg("bands"), py::arg("tag"),
        py::arg("degeneracy_tolerance") = 1e-6, Release(),
        "BandDerivative with respect to mu_r of the cells with the tag");
  m.def("band_shape_derivative", &physics::band_shape_derivative<Dim>, py::arg("problem"),
        py::arg("bands"), py::arg("velocity"), py::arg("relative_step") = 1e-6,
        py::arg("degeneracy_tolerance") = 1e-6, Release(),
        "BandDerivative for the mesh velocity V (num_nodes, Dim) of a geometry parameter; V must "
        "vanish on the periodic faces");
  m.def("band_wave_vector_derivative", &physics::band_wave_vector_derivative<Dim>,
        py::arg("problem"), py::arg("bands"), py::arg("direction"),
        py::arg("degeneracy_tolerance") = 1e-6, Release(),
        "BandDerivative along the wave vector, d/dt k0(k + t d); angular_frequency is the group "
        "velocity along d [m/s] for d in 1/m");
  m.def(
      "group_velocity",
      [](const BandStructure<Dim>& problem, const Bands<Dim>& bands, Real degeneracy_tolerance) {
        std::vector<Point<Dim>> v;
        {
          py::gil_scoped_release release;
          v = physics::group_velocity<Dim>(problem, bands, degeneracy_tolerance);
        }
        py::array_t<Real> out({static_cast<py::ssize_t>(v.size()), static_cast<py::ssize_t>(Dim)});
        auto r = out.mutable_unchecked<2>();
        for (std::size_t i = 0; i < v.size(); ++i) {
          for (int j = 0; j < Dim; ++j) r(static_cast<py::ssize_t>(i), j) = v[i](j);
        }
        return out;
      },
      py::arg("problem"), py::arg("bands"), py::arg("degeneracy_tolerance") = 1e-6,
      "group velocity d omega / dk [m/s] of every band, array (num_bands, Dim); NaN at k0 = 0");
  using physics::TimeDomain;
  using physics::TimeDomainSetup;
  using physics::TimeState;
  py::class_<TimeDomainSetup<Dim>>(
      m, named("TimeDomainSetup", Dim).c_str(),
      "Transient problem: real materials by tag, conductivity [S/m] by tag, PEC and "
      "absorbing (first-order Silver-Mueller) facet tags, a current density J(x) with a "
      "TimeSignal g(t) (source -J g'(t)), the time step and the Newmark parameters")
      .def(py::init<>())
      .def_readwrite("materials", &TimeDomainSetup<Dim>::materials)
      .def_readwrite("conductivity", &TimeDomainSetup<Dim>::conductivity)
      .def_readwrite("pec_tags", &TimeDomainSetup<Dim>::pec_tags)
      .def_readwrite("absorbing_tags", &TimeDomainSetup<Dim>::absorbing_tags)
      .def_readwrite("current", &TimeDomainSetup<Dim>::current)
      .def_readwrite("signal", &TimeDomainSetup<Dim>::signal)
      .def_readwrite("dt", &TimeDomainSetup<Dim>::dt)
      .def_readwrite("beta", &TimeDomainSetup<Dim>::beta)
      .def_readwrite("gamma", &TimeDomainSetup<Dim>::gamma)
      .def_readwrite("solver", &TimeDomainSetup<Dim>::solver)
      .def_readwrite("extra_quadrature_order", &TimeDomainSetup<Dim>::extra_quadrature_order);
  py::class_<TimeState<Dim>>(m, named("TimeState", Dim).c_str(),
                             "Field u = E, velocity v and acceleration a at a time")
      .def_readwrite("time", &TimeState<Dim>::time)
      .def_readwrite("u", &TimeState<Dim>::u)
      .def_readwrite("v", &TimeState<Dim>::v)
      .def_readwrite("a", &TimeState<Dim>::a)
      .def_readonly("step", &TimeState<Dim>::step);
  py::class_<TimeDomain<Dim>>(
      m, named("TimeDomain", Dim).c_str(),
      "Assembles the matrices once, factorises the Newmark operator and advances a state")
      // assembles in the constructor: release the GIL so that Python callbacks (the
      // current density) can run from the assembly threads
      .def(py::init<const ND&, TimeDomainSetup<Dim>>(), py::arg("dofs"), py::arg("setup"),
           py::keep_alive<1, 2>(), Release())
      .def_property_readonly("setup", &TimeDomain<Dim>::setup,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("num_free_dofs", &TimeDomain<Dim>::num_free_dofs)
      .def(
          "initialize",
          [](const TimeDomain<Dim>& self, const Vector& u0, const Vector& v0, Real t0) {
            return self.initialize(u0, v0, t0);
          },
          py::arg("u0"), py::arg("v0"), py::arg("t0") = 0.0, Release(),
          "state at t0 with the consistent acceleration")
      .def(
          "initialize", [](const TimeDomain<Dim>& self, Real t0) { return self.initialize(t0); },
          py::arg("t0") = 0.0, Release(), "zero state")
      .def("step", &TimeDomain<Dim>::step, py::arg("state"), Release(), "one Newmark step")
      .def(
          "run",
          [](const TimeDomain<Dim>& self, TimeState<Dim>& state, int steps,
             const std::function<void(const TimeState<Dim>&)>& observer) {
            self.run(state, steps, observer);
          },
          py::arg("state"), py::arg("steps"), py::arg("observer") = nullptr, Release(),
          "steps Newmark steps, observer(state) after each one")
      .def("energy", &TimeDomain<Dim>::energy, py::arg("state"),
           "discrete energy (v^T M v + u^T S u) / 2 [J]")
      .def("load", &TimeDomain<Dim>::load, py::arg("t"), "load on the free DoFs at time t")
      .def_property_readonly("stiffness", &TimeDomain<Dim>::stiffness)
      .def_property_readonly("mass", &TimeDomain<Dim>::mass)
      .def_property_readonly("damping", &TimeDomain<Dim>::damping);

  using physics::RieszProjection;
  using physics::RieszProjectionBase;
  using physics::RieszSetup;
  py::class_<RieszProjection<Dim>, RieszProjectionBase>(
      m, named("RieszProjection", Dim).c_str(),
      "Modal expansion of a source problem on the pencil of a Resonance problem: pole terms "
      "R_n / (omega - omega_n) of the listed quasi-normal modes plus a background integral "
      "(Riesz projections, contour integrals by the trapezoidal rule)")
      .def(py::init<const Resonance<Dim>&, RieszSetup>(), py::arg("resonance"), py::arg("setup"),
           py::keep_alive<1, 2>(), Release())
      .def(
          "add_current",
          [](RieszProjection<Dim>& self, const assembly::ComplexVectorField<Dim>& current,
             int extra_order) { return self.add_current(current, extra_order); },
          py::arg("current"), py::arg("extra_order") = 2, Release(),
          "current density J(x) (A/m^Dim, without i omega mu0); returns the source index")
      .def("add_point_value", &RieszProjection<Dim>::add_point_value, py::arg("x"),
           py::arg("weight"), "Q(E) = E(x) . w (w not conjugated); returns the functional index")
      .def_property_readonly("dofs", &RieszProjection<Dim>::dofs,
                             py::return_value_policy::reference_internal);
}

/// The setup, contour and base class of the Riesz projections (shared by the dimensions).
void bind_riesz_common(py::module_& m) {
  using physics::RieszContour;
  using physics::RieszProjectionBase;
  using physics::RieszSetup;
  py::class_<RieszSetup>(
      m, "RieszSetup",
      "Contours of a Riesz projection: the poles to expand around (every pole inside the "
      "background contour must be listed), the frequency range, the points per pole circle "
      "and on the background ellipse, radius and aspect factors, what to store")
      .def(py::init<>())
      .def_readwrite("poles", &RieszSetup::poles)
      .def_readwrite("omega_min", &RieszSetup::omega_min)
      .def_readwrite("omega_max", &RieszSetup::omega_max)
      .def_readwrite("points_per_pole", &RieszSetup::points_per_pole)
      .def_readwrite("background_points", &RieszSetup::background_points)
      .def_readwrite("radius_factor", &RieszSetup::radius_factor)
      .def_readwrite("max_radius", &RieszSetup::max_radius)
      .def_readwrite("min_radius_factor", &RieszSetup::min_radius_factor)
      .def_readwrite("background_margin", &RieszSetup::background_margin)
      .def_readwrite("background_aspect", &RieszSetup::background_aspect)
      .def_readwrite("convergence_warning", &RieszSetup::convergence_warning)
      .def_readwrite("store_residues", &RieszSetup::store_residues)
      .def_readwrite("store_fields", &RieszSetup::store_fields)
      .def_readwrite("solver", &RieszSetup::solver);
  py::class_<RieszContour> contour(m, "RieszContour",
                                   "One integration contour: a pole circle, a group circle "
                                   "or the background ellipse, with its points and samples");
  py::enum_<RieszContour::Kind>(contour, "Kind")
      .value("POLE", RieszContour::Kind::kPole)
      .value("GROUP", RieszContour::Kind::kGroup)
      .value("BACKGROUND", RieszContour::Kind::kBackground);
  contour.def_readonly("kind", &RieszContour::kind)
      .def_readonly("centre", &RieszContour::centre)
      .def_readonly("radius", &RieszContour::radius, "real semi-axis")
      .def_readonly("aspect", &RieszContour::aspect)
      .def_readonly("poles", &RieszContour::poles)
      .def_readonly("points", &RieszContour::points)
      .def_readonly("weights", &RieszContour::weights)
      .def_readonly("residue", &RieszContour::residue, "sources x functionals (pole contours)")
      .def_readonly("convergence", &RieszContour::convergence,
                    "relative difference between the N- and the N/2-point rule")
      .def("encloses", &RieszContour::encloses, py::arg("omega"));
  py::class_<RieszProjectionBase>(m, "RieszProjectionBase",
                                  "Sources, functionals, contour solves and the expansion")
      .def(
          "add_load",
          [](RieszProjectionBase& self, const Vector& load, std::function<Complex(Complex)> scale) {
            return self.add_load(load, std::move(scale));
          },
          py::arg("load"), py::arg("scale") = py::none(),
          "load b(omega) = scale(omega) b (full size; scale None means 1); returns the source "
          "index")
      .def(
          "add_load_function",
          [](RieszProjectionBase& self, std::function<Vector(Complex)> load) {
            return self.add_load(std::move(load));
          },
          py::arg("load"), "load given as a function of omega (full size)")
      .def(
          "add_current_load",
          [](RieszProjectionBase& self, const Vector& load) { return self.add_current(load); },
          py::arg("current_load"), "current source from the assembled b_J (load i omega mu0 b_J)")
      .def(
          "add_functional",
          [](RieszProjectionBase& self, const Vector& q) { return self.add_functional(q); },
          py::arg("q"), "functional Q(x) = q^T x (full size); returns the functional index")
      .def("add_emitted_power", &RieszProjectionBase::add_emitted_power, py::arg("source"),
           py::arg("factor") = 1.0,
           "Q(x) = -1/2 factor conj(b_J)^T x, the power the current emits (real part)")
      .def("run", &RieszProjectionBase::run, Release(),
           "chooses the contours and performs the contour solves")
      .def_property_readonly("contours", &RieszProjectionBase::contours)
      .def_property_readonly("background", &RieszProjectionBase::background,
                             "index of the background contour")
      .def_property_readonly("num_sources", &RieszProjectionBase::num_sources)
      .def_property_readonly("num_functionals", &RieszProjectionBase::num_functionals)
      .def_property_readonly("setup", &RieszProjectionBase::setup,
                             py::return_value_policy::reference_internal)
      .def("contribution", &RieszProjectionBase::contribution, py::arg("contour"),
           py::arg("source"), py::arg("functional"), py::arg("omega"))
      .def("total", &RieszProjectionBase::total, py::arg("source"), py::arg("functional"),
           py::arg("omega"), "sum of all contributions")
      .def(
          "spectrum",
          [](const RieszProjectionBase& self, Index source, Index functional,
             const std::vector<Complex>& omegas) {
            return self.spectrum(source, functional, omegas);
          },
          py::arg("source"), py::arg("functional"), py::arg("omegas"),
          "contributions of every contour (rows, background last) at the frequencies (columns)")
      .def("direct", &RieszProjectionBase::direct, py::arg("omega"), Release(),
           "direct solve: Q_f(x_s(omega)) for all sources and functionals")
      .def("field", &RieszProjectionBase::field, py::arg("contour"), py::arg("source"),
           py::arg("omega"), "field contribution of a contour (full size)")
      .def("expand", &RieszProjectionBase::expand, py::arg("source"), py::arg("omega"),
           "the expanded field, sum of all contour fields (needs store_fields)")
      .def("direct_field", &RieszProjectionBase::direct_field, py::arg("source"), py::arg("omega"),
           Release());
}

}  // namespace

void bind_physics(py::module_& m) {
  py::class_<physics::KeptFactorisation, std::shared_ptr<physics::KeptFactorisation>>(
      m, "KeptFactorisation",
      "Factorised system of a solve kept for tangent (A s = r) and adjoint (A^T z = q) solves "
      "on full-size vectors (ADR-0012); the adjoint is the exact transpose of the solve")
      .def("solve", &physics::KeptFactorisation::solve, py::arg("load"), Release(),
           "s = A^-1 r for a full-size load, homogeneous Dirichlet data")
      .def("solve_many",
           py::overload_cast<const Matrix&>(&physics::KeptFactorisation::solve_many, py::const_),
           py::arg("loads"), Release(), "several loads at once (n x m)")
      .def("solve_adjoint", &physics::KeptFactorisation::solve_adjoint, py::arg("functional"),
           Release(), "z = A^-T q for a full-size functional vector (adjoint of Q = q^T e)")
      .def("solve_adjoint_many", &physics::KeptFactorisation::solve_adjoint_many,
           py::arg("functionals"), Release(), "several functionals at once (n x m)")
      .def_property_readonly("num_dofs", &physics::KeptFactorisation::num_dofs,
                             "size of the full vectors")
      .def_property_readonly("size", &physics::KeptFactorisation::size,
                             "unknowns of the factorised system")
      .def_property_readonly(
          "solver_name", [](const physics::KeptFactorisation& k) { return k.solver().name(); },
          "backend of the factorisation");
  {
    using physics::ConicalScattering;
    using physics::ConicalScatteringSetup;
    using physics::ConicalSolution;
    using physics::ConicalVector;
    using ND = fespace::NedelecDofMap<2>;
    using H1 = fespace::DofMap<2>;
    m.def("conical_gaussian_dipole", &physics::conical_gaussian_dipole, py::arg("position"),
          py::arg("moment"), py::arg("sigma"), py::arg("omega"), py::arg("beta"),
          "f = i omega mu0 J of the cell problem of a Gaussian dipole (M17, ADR-0013): current "
          "moment (p_x, p_y, p_z) [A m] at (x0, y0), smearing sigma, the z-smearing as "
          "exp(-sigma^2 beta^2 / 2); scaled components (f_x, f_y, -i f_z) for "
          "ConicalScatteringSetup.current");
    m.def("conical_dipole_responses", &physics::conical_dipole_responses, py::arg("problem"),
          py::arg("solution"), py::arg("position"), py::arg("sigma"), py::arg("moments"), Release(),
          "solutions of the cell problem for the Gaussian dipoles with the moments (k x 3) on the "
          "factorisation the solve kept: one load assembly near the dipole and one solve each");
    m.def("conical_dipole_power_matrix", &physics::conical_dipole_power_matrix, py::arg("problem"),
          py::arg("unit_responses"), py::arg("position"), py::arg("sigma"),
          py::arg("extra_order") = 4, Release(),
          "3 x 3 complex A with P_cell(p) = Re(p^H A p) from the responses to the unit moments");
    m.def("conical_source_power", &physics::conical_source_power, py::arg("problem"),
          py::arg("solution"), py::arg("extra_order") = 4, Release(),
          "-1/2 Re integral conj(J).E dA of the current of a total-field conical problem [W/m]");
    m.def("conical_plane_wave", &physics::conical_plane_wave, py::arg("amplitude"),
          py::arg("wave_vector"),
          "Scaled components (E0x, E0y, -i E0z) e^{i(kx x + ky y)} of the plane wave E0 e^{i k.x}; "
          "beta = k_z is the setup's beta");
    m.def("conical_plane_wave_curl", &physics::conical_plane_wave_curl, py::arg("amplitude"),
          py::arg("wave_vector"),
          "Physical curl i k x E0 e^{i(kx x + ky y)} of the same plane wave (setup.incident_curl)");
    m.def("conical_polarisation", &physics::conical_polarisation, py::arg("wave_vector"),
          py::arg("normal"), py::arg("polarisation"),
          "Unit amplitude of the s (E perpendicular to the plane of incidence spanned by k and the "
          "normal) or p polarisation");
    m.def("conical_pml_form", &physics::conical_pml_form, py::arg("box"), py::arg("material"),
          py::arg("quadrature_order") = std::optional<int>{},
          "PML tensors Lambda = diag(s_y/s_x, s_x/s_y, s_x s_y) as a ConicalForm");
    py::class_<physics::LayeredConicalWave>(
        m, "LayeredConicalWave",
        "Plane wave on a LayerStack2D (normal along y) for the conical solver: the stack field, "
        "the downward incident wave alone, beta, kx, ky and the bare stack's R and T")
        .def_readonly("field", &physics::LayeredConicalWave::field)
        .def_readonly("incident", &physics::LayeredConicalWave::incident)
        .def_readonly("field_curl", &physics::LayeredConicalWave::field_curl,
                      "physical curl of field (for setup.incident_curl)")
        .def_readonly("incident_curl", &physics::LayeredConicalWave::incident_curl,
                      "physical curl of incident")
        .def_readonly("beta", &physics::LayeredConicalWave::beta)
        .def_readonly("kx", &physics::LayeredConicalWave::kx)
        .def_readonly("ky", &physics::LayeredConicalWave::ky)
        .def_readonly("reflectance", &physics::LayeredConicalWave::reflectance)
        .def_readonly("transmittance", &physics::LayeredConicalWave::transmittance);
    m.def(
        "layered_conical_wave", &physics::layered_conical_wave, py::arg("stack"), py::arg("k0"),
        py::arg("angle"), py::arg("azimuth"), py::arg("polarisation"), py::arg("amplitude") = 1.0,
        "Stack wave at the angle from the normal and the azimuth about it; beta = k0 n sin(angle) "
        "sin(azimuth), azimuth 0 is in-plane incidence (E_z polarisation for s)");
    py::class_<ConicalScatteringSetup>(
        m, "ConicalScatteringSetup",
        "Conical scattering problem (2.5D): omega, beta, materials, PEC tags, PML, periodic pairs, "
        "layered background, incident field (scaled components) or current")
        .def(py::init<>())
        .def_readwrite("omega", &ConicalScatteringSetup::omega)
        .def_readwrite("beta", &ConicalScatteringSetup::beta, "longitudinal wavenumber k_z [1/m]")
        .def_readwrite("materials", &ConicalScatteringSetup::materials)
        .def_readwrite("pec_tags", &ConicalScatteringSetup::pec_tags)
        .def_readwrite("pml", &ConicalScatteringSetup::pml)
        .def_readwrite("periodic", &ConicalScatteringSetup::periodic)
        .def_readwrite("background", &ConicalScatteringSetup::background,
                       "LayerStack2D (normal along y) or None")
        .def_readwrite("incident_curl", &ConicalScatteringSetup::incident_curl,
                       "physical curl of the incident field for h_field / poynting "
                       "(conical_plane_wave_curl, LayeredConicalWave.field_curl); empty: central "
                       "differences")
        .def_readwrite("incident", &ConicalScatteringSetup::incident,
                       "incident field x -> (E_x, E_y, -i E_z) (scattered-field formulation)")
        .def_readwrite("current", &ConicalScatteringSetup::current,
                       "f = i omega mu0 J as (f_x, f_y, -i f_z) (total-field formulation)")
        .def_readwrite("solver", &ConicalScatteringSetup::solver)
        .def_readwrite("progress", &ConicalScatteringSetup::progress,
                       "callback(event) -> bool at the start of every phase of solve and when "
                       "done; False cancels (hpfem.Cancelled)")
        .def_readwrite("extra_quadrature_order", &ConicalScatteringSetup::extra_quadrature_order)
        .def_readwrite("pml_extra_quadrature_order",
                       &ConicalScatteringSetup::pml_extra_quadrature_order)
        .def_readwrite("scalar_ez", &ConicalScatteringSetup::scalar_ez,
                       "solve only the H1 block (E_z) at beta = 0 with an E_z-only excitation; "
                       "the in-plane coefficients of the solution are zero")
        .def_readwrite(
            "keep_factorisation", &ConicalScatteringSetup::keep_factorisation,
            "keep the factorised system in the solution (solution.factorisation) for adjoint and "
            "tangent solves of the sensitivities; holds the factors while the solution lives");
    py::class_<ConicalSolution>(m, "ConicalSolution", "Coefficients of the unknown conical field")
        .def_readonly("beta", &ConicalSolution::beta)
        .def_readonly("scattered", &ConicalSolution::scattered)
        .def_readonly("transverse", &ConicalSolution::transverse)
        .def_readonly("longitudinal", &ConicalSolution::longitudinal, "v = -i E_z coefficients")
        .def_readonly("timing", &ConicalSolution::timing, "seconds per phase of solve and 'total'")
        .def_property_readonly(
            "factorisation",
            [](const ConicalSolution& s) {
              return std::const_pointer_cast<physics::KeptFactorisation>(s.factorisation);
            },
            "the kept factorised system on (E_x, E_y | v) (KeptFactorisation) or None");
    py::class_<ConicalScattering>(m, "ConicalScattering",
                                  "Assembles S(beta) - k0^2 M with the constraints and solves; at "
                                  "beta = 0 with an E_z incident field this is the E_z (TE) solver")
        .def(py::init<const ND&, const H1&, ConicalScatteringSetup>(), py::arg("transverse"),
             py::arg("longitudinal"), py::arg("setup"), py::keep_alive<1, 2>(),
             py::keep_alive<1, 3>(), Release())
        .def_property_readonly("setup", &ConicalScattering::setup,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("wavenumber", &ConicalScattering::wavenumber)
        .def_property_readonly("beta", &ConicalScattering::beta)
        .def_property_readonly("transverse_dofs", &ConicalScattering::transverse_dofs,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("longitudinal_dofs", &ConicalScattering::longitudinal_dofs,
                               py::return_value_policy::reference_internal)
        .def("background_material", &ConicalScattering::background_material, py::arg("cell"))
        .def("form_of_cell", &ConicalScattering::form_of_cell, py::arg("cell"))
        .def_property_readonly("free_dofs",
                               [](const ConicalScattering& p) { return to_array(p.free_dofs()); })
        .def("solve", &ConicalScattering::solve, Release())
        .def("estimate", &ConicalScattering::estimate, py::arg("solution"),
             py::arg("options") = adaptivity::EstimatorOptions{}, Release(),
             "residual indicators of the coupled system (adaptivity.Estimate), the same "
             "quantities as Scattering2D.estimate so that marking and hp decisions apply unchanged")
        .def("error", &ConicalScattering::error, py::arg("solution"), py::arg("exact"),
             py::arg("exact_curl") = physics::ConicalField{}, py::arg("extra_order") = 4, Release(),
             "L2 errors of the unknown (E_x, E_y, E_z) and of its conical curl against an exact "
             "physical field and curl (callables of the point)")
        .def(
            "field",
            [](const ConicalScattering& p, const ConicalSolution& s, Index c, const Point<2>& xi) {
              return p.field(s, c, xi);
            },
            py::arg("solution"), py::arg("cell"), py::arg("xi"), "(E_x, E_y, E_z) of the unknown")
        .def(
            "total_field",
            [](const ConicalScattering& p, const ConicalSolution& s, const mesh::PointLocator<2>& l,
               const Point<2>& x) { return p.total_field(s, l, x); },
            py::arg("solution"), py::arg("locator"), py::arg("x"))
        .def(
            "scattered_field",
            [](const ConicalScattering& p, const ConicalSolution& s, const mesh::PointLocator<2>& l,
               const Point<2>& x) { return p.scattered_field(s, l, x); },
            py::arg("solution"), py::arg("locator"), py::arg("x"))
        .def("incident_field", &ConicalScattering::incident_field, py::arg("x"),
             "(E_x, E_y, E_z) of the incident field")
        .def(
            "sample",
            [](const ConicalScattering& p, const ConicalSolution& s,
               const mesh::PointLocator<2>& locator, const RealArray& points, bool scattered,
               bool bloch_wrap, int interface_side, const std::string& quantity) {
              const auto pts = array_to_points<2>(points);
              const auto options =
                  sampling_options(scattered, bloch_wrap, interface_side, quantity);
              physics::SampledField field;
              {
                py::gil_scoped_release release;
                field = physics::sample_field(p, s, locator, pts, options);
              }
              return sampled_to_python(field);
            },
            py::arg("solution"), py::arg("locator"), py::arg("points"),
            py::arg("scattered") = false, py::arg("bloch_wrap") = true,
            py::arg("interface_side") = 0, py::arg("quantity") = "E",
            "total (or scattered) physical field (E_x, E_y, E_z) at points (n, 2) in parallel: "
            "(values (n, 3) complex, cells (n,)); Bloch wrapping, NaN outside, interface_side "
            "as Scattering2D.sample")
        .def(
            "triangulate",
            [](const ConicalScattering& p, const ConicalSolution& s, int subdivisions,
               bool scattered, const std::string& quantity) {
              return physics::triangulate_field(p, s, subdivisions,
                                                sampling_options(scattered, true, 0, quantity));
            },
            py::arg("solution"), py::arg("subdivisions") = 2, py::arg("scattered") = false,
            py::arg("quantity") = "E", Release(),
            "the field on the subdivided mesh as a TriangulatedField2D")
        .def("incident_curl", &ConicalScattering::incident_curl, py::arg("x"),
             "physical curl of the incident field (setup.incident_curl or central differences)")
        .def("incident_h_field", &ConicalScattering::incident_h_field, py::arg("x"),
             "H of the incident wave in the background medium [A/m]")
        .def(
            "curl_field",
            [](const ConicalScattering& p, const ConicalSolution& s, Index c, const Point<2>& xi) {
              return p.curl_field(s, c, xi);
            },
            py::arg("solution"), py::arg("cell"), py::arg("xi"), "physical curl of the unknown")
        .def(
            "h_field",
            [](const ConicalScattering& p, const ConicalSolution& s, Index c, const Point<2>& xi) {
              return p.h_field(s, c, xi);
            },
            py::arg("solution"), py::arg("cell"), py::arg("xi"),
            "H = curl E / (i omega mu0 mu_r) [A/m] of the total field")
        .def(
            "h_field",
            [](const ConicalScattering& p, const ConicalSolution& s, const mesh::PointLocator<2>& l,
               const Point<2>& x) { return p.h_field(s, l, x); },
            py::arg("solution"), py::arg("locator"), py::arg("x"), "None outside the mesh")
        .def(
            "poynting",
            [](const ConicalScattering& p, const ConicalSolution& s, Index c, const Point<2>& xi) {
              return p.poynting(s, c, xi);
            },
            py::arg("solution"), py::arg("cell"), py::arg("xi"),
            "time-averaged Poynting vector Re(E x conj(H)) / 2 [W/m^2] of the total field")
        .def(
            "poynting",
            [](const ConicalScattering& p, const ConicalSolution& s, const mesh::PointLocator<2>& l,
               const Point<2>& x) { return p.poynting(s, l, x); },
            py::arg("solution"), py::arg("locator"), py::arg("x"), "None outside the mesh");
    py::class_<physics::SweepTimings>(m, "SweepTimings",
                                      "Accumulated seconds of the phases of a ConicalSweep")
        .def_readonly("setup", &physics::SweepTimings::setup)
        .def_readonly("combine", &physics::SweepTimings::combine)
        .def_readonly("assemble", &physics::SweepTimings::assemble)
        .def_readonly("reduce", &physics::SweepTimings::reduce)
        .def_readonly("factorize", &physics::SweepTimings::factorize)
        .def_readonly("solve", &physics::SweepTimings::solve)
        .def_readonly("points", &physics::SweepTimings::points)
        .def("total", &physics::SweepTimings::total);
    py::class_<physics::ConicalSweep>(
        m, "ConicalSweep",
        "Frequency / angle sweep of the conical solver with an affine operator: S0 + beta S1 + "
        "beta^2 S2 - k0^2 sum_g eps_g M_g assembled once for the cells outside the PML, PML "
        "and source cells assembled per point, numerical refactorisation on the first "
        "analysis; results agree with ConicalScattering.solve to rounding. Every point must "
        "keep the mesh, orders, PEC tags, periodic pairs, PML box geometry, the cell-to-material "
        "grouping and the formulation; omega, beta, the materials, the incident field and the "
        "Bloch phases may change")
        .def(py::init<const ND&, const H1&, const ConicalScatteringSetup&>(), py::arg("transverse"),
             py::arg("longitudinal"), py::arg("base"), py::keep_alive<1, 2>(),
             py::keep_alive<1, 3>(), Release())
        .def("solve", &physics::ConicalSweep::solve, py::arg("setup"), Release(),
             "the solution at a point of the sweep")
        .def_property_readonly("timings", &physics::ConicalSweep::timings)
        .def_property_readonly("cached", &physics::ConicalSweep::cached,
                               "whether the points run through the pattern cache")
        .def_property_readonly("num_groups", &physics::ConicalSweep::num_groups)
        .def_property_readonly("num_pml_cells", &physics::ConicalSweep::num_pml_cells)
        .def_property_readonly("num_source_cells", &physics::ConicalSweep::num_source_cells)
        .def_property_readonly("solver", &physics::ConicalSweep::solver,
                               py::return_value_policy::reference_internal,
                               "the solver after the first point");
    py::class_<physics::ConicalError>(m, "ConicalError",
                                      "L2 norms of the field and conical-curl errors")
        .def_readonly("l2", &physics::ConicalError::l2)
        .def_readonly("curl", &physics::ConicalError::curl);
    // --- conical resonances (M15 F14) -----------------------------------------------------
    using physics::ConicalResonance;
    using physics::ConicalResonanceResult;
    using physics::ConicalResonanceSetup;
    using physics::ConicalResonantMode;
    py::class_<ConicalResonanceSetup>(
        m, "ConicalResonanceSetup",
        "Conical resonance (quasi-normal mode) problem: target angular frequency (search centre "
        "and PML design frequency), beta, materials, PEC facets, PML, Bloch pairs with phases")
        .def(py::init<>())
        .def_readwrite("target_omega", &ConicalResonanceSetup::target_omega)
        .def_readwrite("beta", &ConicalResonanceSetup::beta)
        .def_readwrite("materials", &ConicalResonanceSetup::materials)
        .def_readwrite("pec_tags", &ConicalResonanceSetup::pec_tags)
        .def_readwrite("pml", &ConicalResonanceSetup::pml)
        .def_readwrite("periodic", &ConicalResonanceSetup::periodic)
        .def_readwrite("num_modes", &ConicalResonanceSetup::num_modes)
        .def_readwrite("krylov_dimension", &ConicalResonanceSetup::krylov_dimension)
        .def_readwrite("tolerance", &ConicalResonanceSetup::tolerance)
        .def_readwrite("max_iterations", &ConicalResonanceSetup::max_iterations)
        .def_readwrite("remove_gradients", &ConicalResonanceSetup::remove_gradients,
                       "project the gradient kernel out of the Krylov space (default)")
        .def_readwrite("solver", &ConicalResonanceSetup::solver)
        .def_readwrite("extra_quadrature_order", &ConicalResonanceSetup::extra_quadrature_order)
        .def_readwrite("pml_extra_quadrature_order",
                       &ConicalResonanceSetup::pml_extra_quadrature_order)
        .def_readwrite("progress", &ConicalResonanceSetup::progress,
                       "callback(event) -> bool for the phases assembly, constraints, "
                       "eigensolve, post; False cancels (hpfem.Cancelled)");
    py::class_<ConicalResonantMode>(m, "ConicalResonantMode",
                                    "One conical resonance: complex omega (Im < 0 decaying), "
                                    "wavelength, Q, residual, beta and the block coefficients")
        .def_readonly("omega", &ConicalResonantMode::omega)
        .def_readonly("wavelength", &ConicalResonantMode::wavelength)
        .def_readonly("quality", &ConicalResonantMode::quality)
        .def_readonly("residual", &ConicalResonantMode::residual)
        .def_readonly("beta", &ConicalResonantMode::beta)
        .def_readonly("transverse", &ConicalResonantMode::transverse)
        .def_readonly("longitudinal", &ConicalResonantMode::longitudinal)
        .def("solution", &ConicalResonantMode::solution,
             "the mode as a ConicalSolution for the coefficient-based post-processing");
    py::class_<ConicalResonanceResult>(m, "ConicalResonanceResult")
        .def_readonly("modes", &ConicalResonanceResult::modes)
        .def_readonly("timing", &ConicalResonanceResult::timing);
    py::class_<ConicalResonance>(
        m, "ConicalResonance",
        "Assembles S(beta) - k0^2 M with PEC, PML and Bloch constraints and finds the complex "
        "eigenfrequencies closest to the target (both polarisations at beta = 0)")
        .def(py::init<const ND&, const H1&, ConicalResonanceSetup>(), py::arg("transverse"),
             py::arg("longitudinal"), py::arg("setup"), py::keep_alive<1, 2>(),
             py::keep_alive<1, 3>(), Release())
        .def_property_readonly("setup", &ConicalResonance::setup,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("wavenumber", &ConicalResonance::wavenumber, "target k0")
        .def_property_readonly("free_dofs",
                               [](const ConicalResonance& p) { return to_array(p.free_dofs()); })
        .def_property_readonly("transverse_dofs", &ConicalResonance::transverse_dofs,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("longitudinal_dofs", &ConicalResonance::longitudinal_dofs,
                               py::return_value_policy::reference_internal)
        .def("form_of_cell", &ConicalResonance::form_of_cell, py::arg("cell"))
        .def("solve", &ConicalResonance::solve, Release(),
             "ConicalResonanceResult: modes ordered by the distance of omega to the target")
        .def(
            "field",
            [](const ConicalResonance& p, const ConicalResonantMode& mode, Index c,
               const Point<2>& xi) { return p.field(mode, c, xi); },
            py::arg("mode"), py::arg("cell"), py::arg("xi"), "physical (E_x, E_y, E_z)")
        .def(
            "h_field",
            [](const ConicalResonance& p, const ConicalResonantMode& mode, Index c,
               const Point<2>& xi) { return p.h_field(mode, c, xi); },
            py::arg("mode"), py::arg("cell"), py::arg("xi"), "H with the mode's complex omega")
        .def(
            "poynting",
            [](const ConicalResonance& p, const ConicalResonantMode& mode, Index c,
               const Point<2>& xi) { return p.poynting(mode, c, xi); },
            py::arg("mode"), py::arg("cell"), py::arg("xi"))
        .def(
            "sample",
            [](const ConicalResonance& p, const ConicalResonantMode& mode,
               const mesh::PointLocator<2>& locator, const RealArray& points, bool bloch_wrap,
               int interface_side, const std::string& quantity) {
              const auto pts = array_to_points<2>(points);
              const auto options = sampling_options(false, bloch_wrap, interface_side, quantity);
              physics::SampledField field;
              {
                py::gil_scoped_release release;
                field = physics::sample_field(p, mode, locator, pts, options);
              }
              return sampled_to_python(field);
            },
            py::arg("mode"), py::arg("locator"), py::arg("points"), py::arg("bloch_wrap") = true,
            py::arg("interface_side") = 0, py::arg("quantity") = "E",
            "the mode (E, H or S) at points (n, 2): (values (n, 3) complex, cells (n,)); Bloch "
            "wrapping with the phases of the setup, NaN outside")
        .def(
            "triangulate",
            [](const ConicalResonance& p, const ConicalResonantMode& mode, int subdivisions,
               const std::string& quantity) {
              return physics::triangulate_field(p, mode, subdivisions,
                                                sampling_options(false, true, 0, quantity));
            },
            py::arg("mode"), py::arg("subdivisions") = 2, py::arg("quantity") = "E", Release(),
            "the mode on the subdivided mesh as a TriangulatedField2D");
    m.def("conical_poynting_flux", &physics::conical_poynting_flux, py::arg("transverse"),
          py::arg("longitudinal"), py::arg("e"), py::arg("v"), py::arg("beta"), py::arg("omega"),
          py::arg("materials"), py::arg("surface"), py::arg("order") = 8, Release(),
          "Power per unit length [W/m] through the surface (take it through homogeneous cells: the "
          "normal derivative of E_z at a material interface converges one order slower)");
    m.def(
        "conical_fourier_coefficients",
        [](const std::function<ConicalVector(const Point<2>&)>& field, const Point<2>& origin,
           const Point<2>& tangent, Real period, Real kt0, int max_order, int num_points) {
          return physics::conical_fourier_coefficients(field, origin, tangent, period, kt0,
                                                       max_order, num_points);
        },
        py::arg("field"), py::arg("origin"), py::arg("tangent"), py::arg("period"), py::arg("kt0"),
        py::arg("max_order"), py::arg("num_points"),
        "A_m of a 3-vector field along the line origin + s tangent, m = -max_order..max_order");
    py::class_<physics::ConicalDiffractionOrder>(m, "ConicalDiffractionOrder")
        .def_readonly("order", &physics::ConicalDiffractionOrder::order)
        .def_readonly("ky", &physics::ConicalDiffractionOrder::ky, "tangential wavenumber")
        .def_readonly("kx", &physics::ConicalDiffractionOrder::kx, "normal wavenumber")
        .def_readonly("propagating", &physics::ConicalDiffractionOrder::propagating)
        .def_readonly("efficiency", &physics::ConicalDiffractionOrder::efficiency)
        .def_readonly("amplitude", &physics::ConicalDiffractionOrder::amplitude);
    m.def("conical_diffraction_efficiencies", &physics::conical_diffraction_efficiencies,
          py::arg("coefficients"), py::arg("k0"), py::arg("index_line"), py::arg("period"),
          py::arg("kt0"), py::arg("beta"), py::arg("kn_incident"), py::arg("incident_amplitude"),
          "eta_m = Re(k_n,m) |A_m|^2 / (k_n^inc |E0|^2) with k_n,m = sqrt(k0^2 n^2 - k_t,m^2 - "
          "beta^2)");
  }
  py::class_<physics::WaveguidePort>(m, "WaveguidePort",
                                     "A modal port on boundary facets (2D: a straight line)")
      .def(py::init<>())
      .def(py::init([](mesh::Tag facet_tag, Index num_modes, std::vector<Complex> incident) {
             return physics::WaveguidePort{facet_tag, num_modes, std::move(incident)};
           }),
           py::arg("facet_tag"), py::arg("num_modes") = 1,
           py::arg("incident") = std::vector<Complex>{})
      .def_readwrite("facet_tag", &physics::WaveguidePort::facet_tag)
      .def_readwrite("num_modes", &physics::WaveguidePort::num_modes,
                     "modes of the expansion: guided first, then by decay")
      .def_readwrite("incident", &physics::WaveguidePort::incident, "incoming amplitudes per mode");
  py::class_<physics::PortMode>(m, "PortMode", "One mode of a port cross-section")
      .def_readonly("beta", &physics::PortMode::beta)
      .def_readonly("effective_index", &physics::PortMode::effective_index)
      .def_readonly("power", &physics::PortMode::power, "power [W] of the unit-amplitude mode")
      .def_readonly("propagating", &physics::PortMode::propagating);
  py::class_<physics::PortCoefficients>(m, "PortCoefficients",
                                        "Incoming and outgoing modal amplitudes on one port")
      .def_readonly("incoming", &physics::PortCoefficients::incoming)
      .def_readonly("outgoing", &physics::PortCoefficients::outgoing);
  py::class_<physics::PortChannel>(m, "PortChannel", "A propagating (port, mode) channel")
      .def_readonly("port", &physics::PortChannel::port)
      .def_readonly("mode", &physics::PortChannel::mode)
      .def_readonly("beta", &physics::PortChannel::beta)
      .def_readonly("power", &physics::PortChannel::power);
  py::class_<physics::SParameters>(m, "SParameters",
                                   "Power-normalised S-matrix over the propagating channels")
      .def_readonly("channels", &physics::SParameters::channels)
      .def_readonly("s", &physics::SParameters::s, "channels x channels, column = excitation");
  py::class_<physics::TimeSignal>(m, "TimeSignal", "Time signal g(t) with its derivative")
      .def(py::init([](std::function<Real(Real)> value, std::function<Real(Real)> derivative) {
             return physics::TimeSignal{std::move(value), std::move(derivative)};
           }),
           py::arg("value"), py::arg("derivative"))
      .def(
          "__call__", [](const physics::TimeSignal& s, Real t) { return s.value(t); }, py::arg("t"))
      .def(
          "derivative", [](const physics::TimeSignal& s, Real t) { return s.derivative(t); },
          py::arg("t"));
  m.def("gaussian_pulse", &physics::gaussian_pulse, py::arg("t0"), py::arg("width"),
        "exp(-(t - t0)^2 / (2 width^2))");
  m.def("modulated_gaussian", &physics::modulated_gaussian, py::arg("omega"), py::arg("t0"),
        py::arg("width"), "sin(omega (t - t0)) exp(-(t - t0)^2 / (2 width^2))");
  py::enum_<physics::Formulation>(m, "Formulation")
      .value("TOTAL_FIELD", physics::Formulation::kTotalField,
             "unknown E; the incident field enters through boundary data / currents")
      .value("SCATTERED_FIELD", physics::Formulation::kScatteredField,
             "unknown E - E_inc; the incident field enters as a volume source");
  m.def("vacuum_wavenumber", &physics::vacuum_wavenumber, py::arg("omega"), "k0 = omega / c0");
  py::class_<physics::ThermalSetup>(m, "ThermalSetup",
                                    "Heat-conduction problem: conductivities kappa [W/(m K)] by "
                                    "cell tag and for the background, fixed temperatures [K] "
                                    "as (facet tag, T) pairs, solver backend")
      .def(py::init<>())
      .def_readwrite("background_conductivity", &physics::ThermalSetup::background_conductivity)
      .def_readwrite("conductivity", &physics::ThermalSetup::conductivity)
      .def_readwrite("fixed_temperature", &physics::ThermalSetup::fixed_temperature)
      .def_readwrite("solver", &physics::ThermalSetup::solver)
      .def_readwrite("extra_quadrature_order", &physics::ThermalSetup::extra_quadrature_order);
  py::class_<physics::ResonantMode>(m, "ResonantMode",
                                    "Complex angular frequency (Im < 0: decaying), vacuum "
                                    "wavelength of the real part, quality factor "
                                    "Re / (-2 Im), Arnoldi residual and field coefficients")
      .def_readonly("omega", &physics::ResonantMode::omega)
      .def_readonly("wavelength", &physics::ResonantMode::wavelength)
      .def_readonly("quality", &physics::ResonantMode::quality)
      .def_readonly("residual", &physics::ResonantMode::residual)
      .def_readonly("field", &physics::ResonantMode::field);
  py::class_<physics::BandDerivative>(
      m, "BandDerivative",
      "Derivatives of the bands with respect to one parameter p, in band order: eigenvalue "
      "d(k0^2)/dp, wavenumber dk0/dp (NaN at k0 = 0), angular_frequency c0 dk0/dp and the size "
      "of each band's degenerate cluster")
      .def_readonly("eigenvalue", &physics::BandDerivative::eigenvalue)
      .def_readonly("wavenumber", &physics::BandDerivative::wavenumber)
      .def_readonly("angular_frequency", &physics::BandDerivative::angular_frequency)
      .def_readonly("multiplicity", &physics::BandDerivative::multiplicity);
  bind_riesz_common(m);
  bind_triangulated_field<2>(m);
  bind_triangulated_field<3>(m);
  bind_physics_dim<2>(m);
  bind_physics_dim<3>(m);
  // --- axisymmetric (2.5D) problems: meridian mesh, one problem per azimuthal order -------
  {
    using fespace::DofMap;
    using fespace::NedelecDofMap;
    using physics::AxisymmetricCavity;
    using physics::AxisymmetricCavitySetup;
    using physics::AxisymmetricMode;
    using physics::AxisymmetricResonance;
    using physics::AxisymmetricResonanceSetup;
    using physics::AxisymmetricResonantMode;
    using physics::AxisymmetricScatteredField;
    using physics::AxisymmetricScattering;
    using physics::AxisymmetricScatteringSetup;
    py::class_<AxisymmetricCavitySetup>(
        m, "AxisymmetricCavitySetup",
        "Closed axisymmetric cavity on the meridian mesh (x = r, y = z): lossless materials, "
        "PEC facet tags, the axis facet tag, the azimuthal order m and the number of modes")
        .def(py::init<>())
        .def_readwrite("materials", &AxisymmetricCavitySetup::materials)
        .def_readwrite("pec_tags", &AxisymmetricCavitySetup::pec_tags)
        .def_readwrite("axis_tag", &AxisymmetricCavitySetup::axis_tag)
        .def_readwrite("azimuthal_order", &AxisymmetricCavitySetup::azimuthal_order)
        .def_readwrite("num_modes", &AxisymmetricCavitySetup::num_modes)
        .def_readwrite("krylov_dimension", &AxisymmetricCavitySetup::krylov_dimension)
        .def_readwrite("tolerance", &AxisymmetricCavitySetup::tolerance)
        .def_readwrite("max_iterations", &AxisymmetricCavitySetup::max_iterations)
        .def_readwrite("extra_quadrature_order", &AxisymmetricCavitySetup::extra_quadrature_order);
    py::class_<AxisymmetricMode>(m, "AxisymmetricMode", "k0 and the coefficients of one mode")
        .def_readonly("wavenumber", &AxisymmetricMode::wavenumber)
        .def_readonly("meridian", &AxisymmetricMode::meridian, "(E_r, E_z) on the Nedelec map")
        .def_readonly("azimuthal", &AxisymmetricMode::azimuthal, "v = -i r E_phi on the H1 map");
    py::class_<AxisymmetricCavity>(m, "AxisymmetricCavity",
                                   "Order-m eigenmodes with the gauged real eigensolver")
        .def(py::init<const NedelecDofMap<2>&, const DofMap<2>&, AxisymmetricCavitySetup>(),
             py::arg("meridian"), py::arg("azimuthal"), py::arg("setup"), py::keep_alive<1, 2>(),
             py::keep_alive<1, 3>(), Release())
        .def_property_readonly("setup", &AxisymmetricCavity::setup,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("num_free_dofs",
                               [](const AxisymmetricCavity& self) {
                                 return static_cast<Index>(self.free_dofs().size());
                               })
        .def("solve", &AxisymmetricCavity::solve, Release(), "lowest modes, ascending in k0");
    py::class_<AxisymmetricResonanceSetup>(
        m, "AxisymmetricResonanceSetup",
        "Open axisymmetric resonator: target omega, materials, PEC and axis tags, order m, "
        "cylindrical PML box (no layer on the axis side)")
        .def(py::init<>())
        .def_readwrite("target_omega", &AxisymmetricResonanceSetup::target_omega)
        .def_readwrite("materials", &AxisymmetricResonanceSetup::materials)
        .def_readwrite("pec_tags", &AxisymmetricResonanceSetup::pec_tags)
        .def_readwrite("axis_tag", &AxisymmetricResonanceSetup::axis_tag)
        .def_readwrite("azimuthal_order", &AxisymmetricResonanceSetup::azimuthal_order)
        .def_readwrite("pml", &AxisymmetricResonanceSetup::pml)
        .def_readwrite("num_modes", &AxisymmetricResonanceSetup::num_modes)
        .def_readwrite("krylov_dimension", &AxisymmetricResonanceSetup::krylov_dimension)
        .def_readwrite("tolerance", &AxisymmetricResonanceSetup::tolerance)
        .def_readwrite("max_iterations", &AxisymmetricResonanceSetup::max_iterations)
        .def_readwrite("solver", &AxisymmetricResonanceSetup::solver)
        .def_readwrite("extra_quadrature_order",
                       &AxisymmetricResonanceSetup::extra_quadrature_order)
        .def_readwrite("pml_extra_quadrature_order",
                       &AxisymmetricResonanceSetup::pml_extra_quadrature_order);
    py::class_<AxisymmetricResonantMode>(m, "AxisymmetricResonantMode",
                                         "Quasi-normal mode of order m")
        .def_readonly("omega", &AxisymmetricResonantMode::omega)
        .def_readonly("wavelength", &AxisymmetricResonantMode::wavelength)
        .def_readonly("quality", &AxisymmetricResonantMode::quality)
        .def_readonly("residual", &AxisymmetricResonantMode::residual)
        .def_readonly("meridian", &AxisymmetricResonantMode::meridian)
        .def_readonly("azimuthal", &AxisymmetricResonantMode::azimuthal);
    py::class_<AxisymmetricResonance>(m, "AxisymmetricResonance",
                                      "Order-m quasi-normal modes with the cylindrical PML")
        .def(py::init<const NedelecDofMap<2>&, const DofMap<2>&, AxisymmetricResonanceSetup>(),
             py::arg("meridian"), py::arg("azimuthal"), py::arg("setup"), py::keep_alive<1, 2>(),
             py::keep_alive<1, 3>())
        .def_property_readonly("setup", &AxisymmetricResonance::setup,
                               py::return_value_policy::reference_internal)
        .def("solve", &AxisymmetricResonance::solve, Release(),
             "modes ordered by the distance of omega to the target");
    py::enum_<physics::AxisDipole>(m, "AxisDipole", "Orientation of a dipole on the axis")
        .value("AXIAL", physics::AxisDipole::kAxial, "moment along z: order m = 0")
        .value("TRANSVERSE", physics::AxisDipole::kTransverse, "moment along x: orders m = +-1");
    m.def("axisymmetric_gaussian_dipole", &physics::axisymmetric_gaussian_dipole,
          py::arg("position"), py::arg("moment"), py::arg("orientation"), py::arg("sigma"),
          py::arg("omega"), py::arg("m"),
          "f = i omega mu0 J of a Gaussian-smeared point dipole on the axis at z = position "
          "(current moment [A m]) in the scaled components of order m");
    m.def("dipole_vacuum_power", &physics::dipole_vacuum_power, py::arg("moment"), py::arg("omega"),
          "Larmor power Z0 k0^2 |p|^2 / (12 pi) [W]");
    m.def("axial_plane_wave", &physics::axial_plane_wave, py::arg("amplitude"), py::arg("k"),
          py::arg("m"),
          "Order m = +-1 of the x-polarised plane wave E0 x e^{ikz} along the axis in the scaled "
          "components (E_r, v = -i r E_phi, E_z)");
    py::class_<AxisymmetricScatteringSetup>(
        m, "AxisymmetricScatteringSetup",
        "Scattered-field problem of one azimuthal order: omega, materials (background for "
        "unlisted tags), PEC and axis tags, order m, PML box, the m-th component of the "
        "incident field as a callable x -> (E_r, v, E_z)")
        .def(py::init<>())
        .def_readwrite("omega", &AxisymmetricScatteringSetup::omega)
        .def_readwrite("materials", &AxisymmetricScatteringSetup::materials)
        .def_readwrite("pec_tags", &AxisymmetricScatteringSetup::pec_tags)
        .def_readwrite("axis_tag", &AxisymmetricScatteringSetup::axis_tag)
        .def_readwrite("azimuthal_order", &AxisymmetricScatteringSetup::azimuthal_order)
        .def_readwrite("pml", &AxisymmetricScatteringSetup::pml)
        .def_readwrite("background", &AxisymmetricScatteringSetup::background,
                       "optional LayerStack3D normal to the axis (z of the stack = y of the "
                       "meridian mesh): layered background (ADR-0014); set incident to "
                       "layered_axisymmetric_wave(...).value")
        .def_readwrite("incident", &AxisymmetricScatteringSetup::incident)
        .def_readwrite("current", &AxisymmetricScatteringSetup::current,
                       "volume source f = i omega mu0 J of order m (total-field formulation)")
        .def_readwrite("solver", &AxisymmetricScatteringSetup::solver)
        .def_readwrite("extra_quadrature_order",
                       &AxisymmetricScatteringSetup::extra_quadrature_order)
        .def_readwrite("pml_extra_quadrature_order",
                       &AxisymmetricScatteringSetup::pml_extra_quadrature_order);
    py::class_<AxisymmetricScatteredField>(m, "AxisymmetricScatteredField",
                                           "Scattered field of one order")
        .def_readonly("azimuthal_order", &AxisymmetricScatteredField::azimuthal_order)
        .def_readonly("meridian", &AxisymmetricScatteredField::meridian)
        .def_readonly("azimuthal", &AxisymmetricScatteredField::azimuthal);
    py::class_<AxisymmetricScattering>(m, "AxisymmetricScattering",
                                       "Assembles S - k0^2 M with PML and source and solves")
        .def(py::init<const NedelecDofMap<2>&, const DofMap<2>&, AxisymmetricScatteringSetup>(),
             py::arg("meridian"), py::arg("azimuthal"), py::arg("setup"), py::keep_alive<1, 2>(),
             py::keep_alive<1, 3>())
        .def_property_readonly("setup", &AxisymmetricScattering::setup,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("wavenumber", &AxisymmetricScattering::wavenumber)
        .def("background_material", &AxisymmetricScattering::background_material, py::arg("cell"),
             py::return_value_policy::copy,
             "the stack layer at the cell centroid (layered background), otherwise "
             "materials.background")
        .def("solve", &AxisymmetricScattering::solve, Release(), "scattered field of order m")
        .def("absorbed_power", &AxisymmetricScattering::absorbed_power, py::arg("field"),
             py::arg("extra_order") = 4, Release(),
             "absorbed power of the total field (incident field added), PML cells left out")
        .def("incident_absorbed_power", &AxisymmetricScattering::incident_absorbed_power,
             py::arg("extra_order") = 4, Release(),
             "absorbed power of the incident (stack) field alone in the same cells")
        .def("scatterer_cells", &AxisymmetricScattering::scatterer_cells,
             "cells outside the PML whose material deviates from the background (the body)")
        .def("estimate", &AxisymmetricScattering::estimate, py::arg("field"),
             py::arg("options") = adaptivity::EstimatorOptions{}, Release(),
             "r-weighted residual indicators of a solution (adaptivity.Estimate)")
        .def("error", &AxisymmetricScattering::error, py::arg("field"), py::arg("exact"),
             py::arg("exact_curl") = physics::AxisymmetricField{}, Release(),
             "weighted L2 and H(curl) errors against an exact field (E_r, v, E_z) and its "
             "cylindrical curl");
    py::class_<physics::AxisymmetricError>(m, "AxisymmetricError",
                                           "Weighted L2 norms of the field and curl errors")
        .def_readonly("l2", &physics::AxisymmetricError::l2)
        .def_readonly("curl", &physics::AxisymmetricError::curl);
    m.def("axisymmetric_error", &physics::axisymmetric_error, py::arg("meridian"),
          py::arg("azimuthal"), py::arg("meridian_coefficients"), py::arg("azimuthal_coefficients"),
          py::arg("azimuthal_order"), py::arg("exact"),
          py::arg("exact_curl") = physics::AxisymmetricField{}, py::arg("extra_order") = 4,
          Release(),
          "Error of an order-m field against the exact (E_r, v, E_z) and its cylindrical curl "
          "in the norms of the body of revolution");
    py::class_<physics::AxisymmetricFarField>(
        m, "AxisymmetricFarField",
        "Far-field pattern F(theta) e^{im phi} e^{ikR}/R of one order at phi = 0")
        .def_readonly("azimuthal_order", &physics::AxisymmetricFarField::azimuthal_order)
        .def_readonly("wavenumber", &physics::AxisymmetricFarField::wavenumber)
        .def_readonly("impedance", &physics::AxisymmetricFarField::impedance)
        .def_readonly("theta", &physics::AxisymmetricFarField::theta)
        .def_readonly("f_theta", &physics::AxisymmetricFarField::f_theta)
        .def_readonly("f_phi", &physics::AxisymmetricFarField::f_phi)
        .def("radiated_power", &physics::AxisymmetricFarField::radiated_power,
             "int |F|^2 dOmega / (2 Z) [W] over the sampled angles")
        .def("power_between", &physics::AxisymmetricFarField::power_between, py::arg("theta_min"),
             py::arg("theta_max"),
             "the same over the sampled angles in [theta_min, theta_max] (a collection cone: "
             "[0, asin(NA / n)] above, [pi - asin(NA / n), pi] below)");
    m.def("axisymmetric_far_field", &physics::axisymmetric_far_field, py::arg("meridian"),
          py::arg("azimuthal"), py::arg("meridian_coefficients"), py::arg("azimuthal_coefficients"),
          py::arg("azimuthal_order"), py::arg("omega"), py::arg("materials"), py::arg("surface"),
          py::arg("theta"), py::arg("order") = 8, Release(),
          "Near-to-far transform of the order-m field on a closed surface of revolution in the "
          "background medium, sampled at the polar angles theta");
    py::class_<physics::AxisymmetricLayeredFarField>(
        m, "AxisymmetricLayeredFarField",
        "Far field of one order on a layer stack: up (cover, theta < pi/2) and down (substrate, "
        "theta > pi/2), each an AxisymmetricFarField")
        .def_readonly("up", &physics::AxisymmetricLayeredFarField::up)
        .def_readonly("down", &physics::AxisymmetricLayeredFarField::down);
    m.def("axisymmetric_layered_far_field", &physics::axisymmetric_layered_far_field,
          py::arg("meridian"), py::arg("azimuthal"), py::arg("meridian_coefficients"),
          py::arg("azimuthal_coefficients"), py::arg("azimuthal_order"), py::arg("omega"),
          py::arg("materials"), py::arg("surface"), py::arg("stack"), py::arg("theta_up"),
          py::arg("theta_down"), py::arg("order") = 8, Release(),
          "Far field of the order-m field in the cover (theta_up in [0, pi/2)) and the lossless "
          "substrate (theta_down in (pi/2, pi]) by reciprocity with the layered plane waves "
          "(ADR-0014); the surface encloses every source and scatterer");
    py::enum_<physics::PlanePolarisation>(m, "PlanePolarisation",
                                          "Polarisation relative to the plane of incidence")
        .value("S", physics::PlanePolarisation::kS, "E along y")
        .value("P", physics::PlanePolarisation::kP, "E in the x-z plane");
    m.def("oblique_plane_wave", &physics::oblique_plane_wave, py::arg("amplitude"), py::arg("k"),
          py::arg("theta_i"), py::arg("polarisation"), py::arg("m"),
          "Order m of the plane wave at the angle theta_i to the axis (Jacobi-Anger expansion) "
          "in the scaled components (E_r, v = -i r E_phi, E_z)");
    py::class_<physics::AxisymmetricLayeredWave>(
        m, "AxisymmetricLayeredWave",
        "Order m of a plane wave on a layer stack: value(x) -> (E_r, v, E_z), curl(x) -> "
        "cylindrical curl, and R, T, A of the bare stack for the side of incidence")
        .def_readonly("value", &physics::AxisymmetricLayeredWave::value)
        .def_readonly("curl", &physics::AxisymmetricLayeredWave::curl)
        .def_readonly("reflectance", &physics::AxisymmetricLayeredWave::reflectance)
        .def_readonly("transmittance", &physics::AxisymmetricLayeredWave::transmittance)
        .def_readonly("absorptance", &physics::AxisymmetricLayeredWave::absorptance);
    m.def(
        "layered_axisymmetric_wave",
        [](const physics::LayerStack<3>& stack, Real k0, Real theta, const py::object& pol,
           int order, const std::string& side, Complex amplitude) {
          physics::Polarisation polarisation = physics::Polarisation::kP;
          if (py::isinstance<py::str>(pol)) {
            const auto name = pol.cast<std::string>();
            if (name == "s" || name == "S") {
              polarisation = physics::Polarisation::kS;
            } else if (name != "p" && name != "P") {
              throw InvalidArgument("layered_axisymmetric_wave: pol must be 's' or 'p', not '" +
                                    name + "'");
            }
          } else {
            polarisation = pol.cast<physics::Polarisation>();
          }
          if (side != "top" && side != "bottom") {
            throw InvalidArgument(
                "layered_axisymmetric_wave: side must be 'top' or 'bottom', "
                "not '" +
                side + "'");
          }
          return physics::layered_axisymmetric_wave(
              stack, k0, theta, polarisation, order,
              side == "top" ? physics::StackSide::kTop : physics::StackSide::kBottom, amplitude);
        },
        py::arg("stack"), py::arg("k0"), py::arg("theta"), py::arg("pol"), py::arg("m"),
        py::arg("side") = "top", py::arg("amplitude") = Complex{1.0, 0.0},
        "Order m of the plane wave of LayerStack3D.plane_wave (ADR-0014): angle theta in [0, "
        "pi/2) from the normal, in-plane wave vector along +x, pol 's' / 'p' (or Polarisation), "
        "from side 'top' (incidence medium, towards -z) or 'bottom' (lossless substrate, towards "
        "+z); returns AxisymmetricLayeredWave with value(x) = (E_r, v = -i r E_phi, E_z) and "
        "curl(x) on the meridian point x = (r, z)");
    py::class_<physics::AxisymmetricOrders>(m, "AxisymmetricOrders",
                                            "Fields of several orders and their powers")
        .def_readonly("orders", &physics::AxisymmetricOrders::orders)
        .def_readonly("fields", &physics::AxisymmetricOrders::fields)
        .def_readonly("power", &physics::AxisymmetricOrders::power)
        .def("total_power", &physics::AxisymmetricOrders::total_power, "sum over the orders [W]");
    m.def("scatter_orders", &physics::scatter_orders, py::arg("meridian"), py::arg("azimuthal"),
          py::arg("setup"), py::arg("incident_of_order"), py::arg("max_order"), py::arg("surface"),
          py::arg("tolerance") = 1e-6, Release(),
          "Solves the orders m = 0, +-1, +-2, ... with the incident component incident_of_order(m) "
          "until the pair +-m carries less than tolerance times the total power through the "
          "surface (|m| >= 2) or max_order is reached");
    m.def("superpose_far_field", &physics::superpose_far_field, py::arg("patterns"),
          py::arg("orders"), py::arg("phi"),
          "Far-field pattern sum_m F_m(theta) e^{i m phi} of the orders at the azimuth phi");
    m.def("axisymmetric_poynting_flux", &physics::axisymmetric_poynting_flux, py::arg("meridian"),
          py::arg("azimuthal"), py::arg("meridian_coefficients"), py::arg("azimuthal_coefficients"),
          py::arg("azimuthal_order"), py::arg("omega"), py::arg("materials"), py::arg("surface"),
          py::arg("order") = 8, py::arg("added_value") = physics::AxisymmetricField{},
          py::arg("added_curl") = physics::AxisymmetricField{}, Release(),
          "Power [W] of the order-m field through the surface of revolution of the meridian "
          "surface (2 pi r Re(E x H*) . n / 2 integrated); added_value / added_curl: an analytic "
          "order-m field (e.g. layered_axisymmetric_wave(...).value / .curl) added for the flux "
          "of the total field");
    py::class_<physics::AxisymmetricFluxChannels>(
        m, "AxisymmetricFluxChannels",
        "Power [W] through a surface split by a layer stack: up (above the top interface), down "
        "(below the bottom interface), lateral (between them, along the layers)")
        .def_readonly("up", &physics::AxisymmetricFluxChannels::up)
        .def_readonly("down", &physics::AxisymmetricFluxChannels::down)
        .def_readonly("lateral", &physics::AxisymmetricFluxChannels::lateral)
        .def("total", &physics::AxisymmetricFluxChannels::total);
    m.def("axisymmetric_flux_channels", &physics::axisymmetric_flux_channels, py::arg("meridian"),
          py::arg("azimuthal"), py::arg("meridian_coefficients"), py::arg("azimuthal_coefficients"),
          py::arg("azimuthal_order"), py::arg("omega"), py::arg("materials"), py::arg("surface"),
          py::arg("stack"), py::arg("order") = 8,
          py::arg("added_value") = physics::AxisymmetricField{},
          py::arg("added_curl") = physics::AxisymmetricField{}, Release(),
          "axisymmetric_poynting_flux split into the channels up / down / lateral of the stack "
          "(each quadrature point by its height)");
    py::class_<physics::AxisymmetricDiscFlux>(
        m, "AxisymmetricDiscFlux",
        "Power [W] through a disc r <= R: total (field + added field), background (added field "
        "alone), change() = total - background")
        .def_readonly("total", &physics::AxisymmetricDiscFlux::total)
        .def_readonly("background", &physics::AxisymmetricDiscFlux::background)
        .def("change", &physics::AxisymmetricDiscFlux::change);
    m.def("axisymmetric_disc_flux", &physics::axisymmetric_disc_flux, py::arg("meridian"),
          py::arg("azimuthal"), py::arg("meridian_coefficients"), py::arg("azimuthal_coefficients"),
          py::arg("azimuthal_order"), py::arg("omega"), py::arg("materials"), py::arg("z"),
          py::arg("radius"), py::arg("direction") = -1,
          py::arg("added_value") = physics::AxisymmetricField{},
          py::arg("added_curl") = physics::AxisymmetricField{}, py::arg("order") = 8, Release(),
          "Power of the order-m field through the disc r <= radius on the mesh line y = z along "
          "direction (-1 downwards, +1 upwards), with and without the added analytic field "
          "(aperture transmission, ADR-0014)");
    m.def("axisymmetric_absorbed_power", &physics::axisymmetric_absorbed_power, py::arg("meridian"),
          py::arg("azimuthal"), py::arg("meridian_coefficients"), py::arg("azimuthal_coefficients"),
          py::arg("azimuthal_order"), py::arg("omega"), py::arg("materials"),
          py::arg("added") = physics::AxisymmetricField{},
          py::arg("pml") = std::optional<pml::PmlBox<2>>{}, py::arg("extra_order") = 4, Release(),
          "Absorbed power (AbsorbedPower: total, by_tag, per_cell) [W] of the order-m field with "
          "the analytic field `added` (scaled components) added, PML cells left out; the orders "
          "add up");
  }

  using physics::PropagatingMode;
  using physics::WaveguideMode;
  using physics::WaveguideSetup;
  py::class_<WaveguideSetup>(m, "WaveguideSetup",
                             "Waveguide cross-section problem: omega, lossless materials by "
                             "tag, PEC facet tags, number of guided modes wanted")
      .def(py::init<>())
      .def_readwrite("omega", &WaveguideSetup::omega)
      .def_readwrite("solver", &WaveguideSetup::solver,
                     "direct solver of the shifted pencil (AUTO: real SparseLU)")
      .def_readwrite("materials", &WaveguideSetup::materials)
      .def_readwrite("pec_tags", &WaveguideSetup::pec_tags)
      .def_readwrite("num_modes", &WaveguideSetup::num_modes)
      .def_readwrite("max_index", &WaveguideSetup::max_index,
                     "largest refractive index (0: maximum over the materials)")
      .def_readwrite("krylov_dimension", &WaveguideSetup::krylov_dimension)
      .def_readwrite("tolerance", &WaveguideSetup::tolerance)
      .def_readwrite("max_iterations", &WaveguideSetup::max_iterations);
  py::class_<WaveguideMode>(m, "WaveguideMode",
                            "Guided mode E = (E_t + z E_z) exp(i beta z): E_t on the Nédélec "
                            "space, E_z = -i beta e_z on the H1 space")
      .def_readonly("beta", &WaveguideMode::beta)
      .def_readonly("effective_index", &WaveguideMode::effective_index)
      .def_readonly("transverse", &WaveguideMode::transverse)
      .def_readonly("longitudinal", &WaveguideMode::longitudinal);
  py::class_<PropagatingMode<2>>(m, "PropagatingMode",
                                 "Propagating modes of a waveguide with a 2D cross-section "
                                 "(Lee–Sun–Cendes pencil, no spurious modes)")
      .def(py::init<const fespace::NedelecDofMap<2>&, const fespace::DofMap<2>&, WaveguideSetup>(),
           py::arg("transverse"), py::arg("longitudinal"), py::arg("setup"), py::keep_alive<1, 2>(),
           py::keep_alive<1, 3>())
      .def_property_readonly("wavenumber", &PropagatingMode<2>::wavenumber)
      .def_property_readonly("max_index", &PropagatingMode<2>::max_index)
      .def("solve", &PropagatingMode<2>::solve, Release(),
           "guided modes with 0 < beta <= k0 n_max, largest beta first");

  using physics::AxisymmetricRieszProjection;
  py::class_<AxisymmetricRieszProjection, physics::RieszProjectionBase>(
      m, "AxisymmetricRieszProjection",
      "Modal expansion on the order-m block pencil of an AxisymmetricResonance problem; full "
      "vectors are the block vectors (meridian, azimuthal), the emitted power carries the "
      "azimuthal factor 2 pi")
      .def(py::init<const physics::AxisymmetricResonance&, physics::RieszSetup>(),
           py::arg("resonance"), py::arg("setup"), py::keep_alive<1, 2>(), Release())
      .def(
          "add_current",
          [](AxisymmetricRieszProjection& self, const physics::AxisymmetricField& current,
             int extra_order) { return self.add_current(current, extra_order); },
          py::arg("current"), py::arg("extra_order") = 4, Release(),
          "order-m current density (J_r, -i r J_phi, J_z) without i omega mu0, e.g. "
          "axisymmetric_gaussian_dipole(...) / (1j * omega * mu0); returns the source index")
      .def(
          "add_emitted_power",
          [](AxisymmetricRieszProjection& self, Index source) {
            return self.add_emitted_power(source);
          },
          py::arg("source"), "the emitted power of the order with the azimuthal factor 2 pi")
      .def_property_readonly("azimuthal_order", &AxisymmetricRieszProjection::azimuthal_order);
}

}  // namespace hpfem::python

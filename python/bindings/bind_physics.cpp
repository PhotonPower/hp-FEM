/// Bindings of `physics`: analytic incident fields, the scattering problem (setup, solve,
/// field evaluation, errors, estimator), parameter sweeps and waveguide modes.
#include <functional>
#include <optional>
#include <vector>

#include "common.hpp"
#include "hpfem/adaptivity/residual_estimator.hpp"
#include "hpfem/physics/propagating_mode.hpp"
#include "hpfem/physics/resonance.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"
#include "hpfem/physics/sweep.hpp"
#include "hpfem/physics/thermal.hpp"

namespace hpfem::python {

namespace {

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
      .def_readwrite("formulation", &ScatteringSetup<Dim>::formulation)
      .def_readwrite("pec_tags", &ScatteringSetup<Dim>::pec_tags)
      .def_readwrite("incident_tags", &ScatteringSetup<Dim>::incident_tags,
                     "facets with n x E = n x E_inc (test domains)")
      .def_readwrite("current", &ScatteringSetup<Dim>::current)
      .def_readwrite("pml", &ScatteringSetup<Dim>::pml)
      .def_readwrite("periodic", &ScatteringSetup<Dim>::periodic)
      .def_readwrite("solver", &ScatteringSetup<Dim>::solver)
      .def_readwrite("condense", &ScatteringSetup<Dim>::condense)
      .def_readwrite("extra_quadrature_order", &ScatteringSetup<Dim>::extra_quadrature_order)
      .def_readwrite("pml_extra_quadrature_order",
                     &ScatteringSetup<Dim>::pml_extra_quadrature_order);
  py::class_<ScatteringSolution<Dim>>(m, named("ScatteringSolution", Dim).c_str(),
                                      "Coefficients of the unknown field (E or E_sc)")
      .def(py::init([](physics::Formulation f, Vector unknown) {
             return ScatteringSolution<Dim>{f, std::move(unknown)};
           }),
           py::arg("formulation"), py::arg("unknown"))
      .def_readwrite("formulation", &ScatteringSolution<Dim>::formulation)
      .def_readwrite("unknown", &ScatteringSolution<Dim>::unknown);

  py::class_<Scattering<Dim>>(
      m, named("Scattering", Dim).c_str(),
      "Time-harmonic scattering on a Nédélec space: curl(mu_r^-1 curl E) - k0^2 eps_r E = f; "
      "assembles with static condensation, applies PEC / incident traces, hanging-node and "
      "Bloch constraints, PML as stretched tensors, and solves with a direct solver")
      .def(py::init<const ND&, ScatteringSetup<Dim>>(), py::arg("dofs"), py::arg("setup"),
           py::keep_alive<1, 2>())
      .def_property_readonly("dofs", &Scattering<Dim>::dofs,
                             py::return_value_policy::reference_internal)
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
      .def(
          "interior_cells", [](const Scattering<Dim>& p) { return to_array(p.interior_cells()); },
          "cells inside the PML box (all cells without PML)")
      .def("estimate", &Scattering<Dim>::estimate, py::arg("solution"),
           py::arg("options") = adaptivity::EstimatorOptions{}, Release(),
           "Residual-based element indicators of the solution");

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
          "solution for another incident field (and current)");
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
}

}  // namespace

void bind_physics(py::module_& m) {
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
  bind_physics_dim<2>(m);
  bind_physics_dim<3>(m);

  using physics::PropagatingMode;
  using physics::WaveguideMode;
  using physics::WaveguideSetup;
  py::class_<WaveguideSetup>(m, "WaveguideSetup",
                             "Waveguide cross-section problem: omega, lossless materials by "
                             "tag, PEC facet tags, number of guided modes wanted")
      .def(py::init<>())
      .def_readwrite("omega", &WaveguideSetup::omega)
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
}

}  // namespace hpfem::python

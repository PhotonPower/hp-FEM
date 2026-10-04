/// Bindings of `physics`: analytic incident fields, the scattering problem (setup, solve,
/// field evaluation, errors, estimator), parameter sweeps and waveguide modes.
#include <functional>
#include <optional>
#include <vector>

#include "common.hpp"
#include "hpfem/adaptivity/residual_estimator.hpp"
#include "hpfem/physics/axisymmetric.hpp"
#include "hpfem/physics/band_structure.hpp"
#include "hpfem/physics/propagating_mode.hpp"
#include "hpfem/physics/resonance.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"
#include "hpfem/physics/sweep.hpp"
#include "hpfem/physics/thermal.hpp"
#include "hpfem/physics/thermo_optical.hpp"
#include "hpfem/physics/time_domain.hpp"

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
          "solution for another incident field (and current)")
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
      .def_readwrite("extra_quadrature_order", &BandStructureSetup<Dim>::extra_quadrature_order);
  py::class_<Bands<Dim>>(m, named("Bands", Dim).c_str(),
                         "The bands at one Bloch wave vector: wavenumbers k0 ascending [1/m]")
      .def_readonly("wave_vector", &Bands<Dim>::wave_vector)
      .def_readonly("wavenumber", &Bands<Dim>::wavenumber)
      .def_readonly("residual", &Bands<Dim>::residual)
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
}

}  // namespace

void bind_physics(py::module_& m) {
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
        .def("solve", &AxisymmetricScattering::solve, Release(), "scattered field of order m");
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
             "int |F|^2 dOmega / (2 Z) [W] over the sampled angles");
    m.def("axisymmetric_far_field", &physics::axisymmetric_far_field, py::arg("meridian"),
          py::arg("azimuthal"), py::arg("meridian_coefficients"), py::arg("azimuthal_coefficients"),
          py::arg("azimuthal_order"), py::arg("omega"), py::arg("materials"), py::arg("surface"),
          py::arg("theta"), py::arg("order") = 8, Release(),
          "Near-to-far transform of the order-m field on a closed surface of revolution in the "
          "background medium, sampled at the polar angles theta");
    py::enum_<physics::PlanePolarisation>(m, "PlanePolarisation",
                                          "Polarisation relative to the plane of incidence")
        .value("S", physics::PlanePolarisation::kS, "E along y")
        .value("P", physics::PlanePolarisation::kP, "E in the x-z plane");
    m.def("oblique_plane_wave", &physics::oblique_plane_wave, py::arg("amplitude"), py::arg("k"),
          py::arg("theta_i"), py::arg("polarisation"), py::arg("m"),
          "Order m of the plane wave at the angle theta_i to the axis (Jacobi-Anger expansion) "
          "in the scaled components (E_r, v = -i r E_phi, E_z)");
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
          py::arg("order") = 8, Release(),
          "Power [W] of the order-m field through the surface of revolution of the meridian "
          "surface (2 pi r Re(E x H*) . n / 2 integrated)");
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
}

}  // namespace hpfem::python

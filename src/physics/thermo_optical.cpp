#include "hpfem/physics/thermo_optical.hpp"

#include <cmath>

#include <fmt/format.h>

#include "hpfem/assembly/h1_forms.hpp"
#include "hpfem/assembly/interpolation.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/physics/postprocess.hpp"

namespace hpfem::physics {

template <int Dim>
ThermoOptical<Dim>::ThermoOptical(const fespace::NedelecDofMap<Dim>& optical_dofs,
                                  const fespace::DofMap<Dim>& thermal_dofs,
                                  ThermoOpticalSetup<Dim> setup)
    : optical_(&optical_dofs), thermal_(&thermal_dofs), setup_(std::move(setup)) {
  if (&optical_dofs.mesh() != &thermal_dofs.mesh()) {
    throw InvalidArgument("ThermoOptical: the optical and thermal maps must share the mesh");
  }
  if (!(setup_.tolerance > 0)) throw InvalidArgument("ThermoOptical: tolerance must be positive");
  if (!(setup_.relaxation > 0) || setup_.relaxation > 1) {
    throw InvalidArgument("ThermoOptical: relaxation must lie in (0, 1]");
  }
  if (setup_.max_iterations < 1) {
    throw InvalidArgument("ThermoOptical: max_iterations must be at least 1");
  }
}

template <int Dim>
materials::MaterialMap ThermoOptical<Dim>::materials_at(const Vector& temperature) const {
  if (temperature.size() != thermal_->num_dofs()) {
    throw InvalidArgument(fmt::format("ThermoOptical: {} temperature coefficients for {} DoFs",
                                      temperature.size(), thermal_->num_dofs()));
  }
  const auto& mesh = thermal_->mesh();
  materials::MaterialMap materials = setup_.optical.materials;
  materials.clear_cells();
  const Point<Dim> centroid = Point<Dim>::Constant(1.0 / (Dim + 1));
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const auto it = setup_.thermo_optic.find(mesh.cell_tag(c));
    if (it == setup_.thermo_optic.end() || it->second == Complex{0.0, 0.0}) continue;
    const Real t = assembly::evaluate_h1<Dim>(*thermal_, temperature, c, centroid).real();
    materials::Material material = setup_.optical.materials.at(mesh.cell_tag(c));
    material.eps_r += it->second * (t - setup_.reference_temperature);
    materials.set_cell(c, material);
  }
  return materials;
}

template <int Dim>
Vector ThermoOptical<Dim>::total_field(const ScatteringSolution<Dim>& solution) const {
  if (solution.formulation == Formulation::kTotalField || !setup_.optical.incident) {
    return solution.unknown;
  }
  return solution.unknown +
         assembly::interpolate<Dim>(*optical_,
                                    assembly::physical_sampler<Dim>(setup_.optical.incident.value));
}

template <int Dim>
ThermoOpticalState<Dim> ThermoOptical<Dim>::solve() const {
  ThermoOpticalState<Dim> state;
  state.temperature = Vector::Constant(thermal_->num_dofs(), setup_.reference_temperature);
  const Thermal<Dim> thermal(*thermal_, setup_.thermal);
  const Real omega = setup_.optical.omega;
  for (int iteration = 1; iteration <= setup_.max_iterations; ++iteration) {
    state.materials = materials_at(state.temperature);
    ScatteringSetup<Dim> optical = setup_.optical;
    optical.materials = state.materials;
    const Scattering<Dim> problem(*optical_, optical);
    state.solution = problem.solve();
    const Vector field = total_field(state.solution);
    const Vector load =
        absorbed_power_load<Dim>(*optical_, field, omega, state.materials, *thermal_);
    state.absorbed_power = absorbed_power<Dim>(*optical_, field, omega, state.materials);
    const Vector updated = thermal.solve_load(load);
    const Real change = (updated - state.temperature).cwiseAbs().maxCoeff();
    state.temperature = (1 - setup_.relaxation) * state.temperature + setup_.relaxation * updated;
    state.history.push_back(change);
    state.iterations = iteration;
    log().info(
        "ThermoOptical<{}>: iteration {}, absorbed {:.6g} W, max |dT| = {:.4g} K, T_max = {:.4g} K",
        Dim, iteration, state.absorbed_power, change, state.temperature.real().maxCoeff());
    if (change < setup_.tolerance || setup_.thermo_optic.empty()) {
      state.converged = true;  // no feedback at all: the first pass is the answer
      break;
    }
  }
  if (!state.converged) {
    log().warn("ThermoOptical<{}>: not converged after {} iterations (max |dT| = {:.4g} K)", Dim,
               state.iterations, state.history.back());
  }
  return state;
}

template struct ThermoOpticalSetup<2>;
template struct ThermoOpticalSetup<3>;
template struct ThermoOpticalState<2>;
template struct ThermoOpticalState<3>;
template class ThermoOptical<2>;
template class ThermoOptical<3>;

}  // namespace hpfem::physics

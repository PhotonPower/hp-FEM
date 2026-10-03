#pragma once
/// @file thermo_optical.hpp
/// Temperature-dependent permittivity, the feedback loop of the optical heating: the
/// absorbed power of the scattering solution heats the structure (`Thermal`), the
/// permittivity of every cell follows its temperature,
/// @f$ \varepsilon_r(T) = \varepsilon_r(T_0) + \frac{d\varepsilon_r}{dT}\,(T - T_0) @f$
/// (complex thermo-optic coefficient per material tag, evaluated at the cell centroid and
/// stored as a per-cell override of the `MaterialMap`), and the scattering problem is
/// solved again until the temperature settles (fixed-point iteration with optional
/// under-relaxation). Convention exp(−iωt), SI units. See docs/theory/multiphysics.md.
#include <map>
#include <vector>

#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/mesh.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/thermal.hpp"

namespace hpfem::physics {

/// Description of the coupled problem.
template <int Dim>
struct ThermoOpticalSetup {
  ScatteringSetup<Dim> optical;  ///< materials give εr(T0) by tag
  ThermalSetup thermal;
  std::map<mesh::Tag, Complex> thermo_optic;  ///< dεr/dT [1/K] by cell tag (others: 0)
  Real reference_temperature = 300.0;         ///< T0 [K] of the optical materials
  int max_iterations = 20;
  Real tolerance = 1e-3;  ///< converged when max |T_new − T_old| < tolerance [K]
  Real relaxation = 1.0;  ///< T ← (1 − r) T_old + r T_new
};

/// Converged (or last) state of the loop.
template <int Dim>
struct ThermoOpticalState {
  ScatteringSolution<Dim> solution;  ///< optical solution with the final permittivities
  Vector temperature;                ///< on the H1 map [K]
  materials::MaterialMap materials;  ///< with the per-cell permittivities of the last step
  Real absorbed_power = 0;           ///< [W] (3D) or [W/m] (2D)
  std::vector<Real> history;         ///< max |ΔT| per iteration
  int iterations = 0;
  bool converged = false;
};

/// Runs the optical–thermal feedback loop on a Nédélec and an H1 space of the same mesh.
template <int Dim>
class ThermoOptical {
 public:
  /// @throws InvalidArgument if the maps live on different meshes, or for a non-positive
  ///         tolerance or relaxation.
  ThermoOptical(const fespace::NedelecDofMap<Dim>& optical_dofs,
                const fespace::DofMap<Dim>& thermal_dofs, ThermoOpticalSetup<Dim> setup);
  [[nodiscard]] const ThermoOpticalSetup<Dim>& setup() const noexcept { return setup_; }
  /// Materials with the permittivities of the given temperature field (cell overrides for
  /// every cell whose tag has a thermo-optic coefficient).
  [[nodiscard]] materials::MaterialMap materials_at(const Vector& temperature) const;
  /// Coefficients of the total field of a solution on the optical map (the scattered
  /// field plus the interpolated incident field, or the total field itself).
  [[nodiscard]] Vector total_field(const ScatteringSolution<Dim>& solution) const;
  /// Iterates until the temperature change falls below the tolerance or the iteration
  /// limit is reached (`converged` tells which).
  [[nodiscard]] ThermoOpticalState<Dim> solve() const;

 private:
  const fespace::NedelecDofMap<Dim>* optical_;
  const fespace::DofMap<Dim>* thermal_;
  ThermoOpticalSetup<Dim> setup_;
};

extern template struct ThermoOpticalSetup<2>;
extern template struct ThermoOpticalSetup<3>;
extern template struct ThermoOpticalState<2>;
extern template struct ThermoOpticalState<3>;
extern template class ThermoOptical<2>;
extern template class ThermoOptical<3>;

}  // namespace hpfem::physics

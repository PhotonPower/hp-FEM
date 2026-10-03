#pragma once
/// @file resonance.hpp
/// Resonances (quasi-normal modes) of open structures: the eigenproblem
/// @f$ \nabla\times(\mu_r^{-1}\nabla\times E) = k^2\varepsilon_r E @f$ with PEC walls and a PML
/// that turns the outgoing-wave condition into complex-symmetric absorbing layers, so the
/// eigenvalues @f$ k^2 = \omega^2/c_0^2 @f$ are complex. With the exp(−iωt) convention a
/// decaying mode has @f$ \mathrm{Im}\,\omega < 0 @f$ and the quality factor is
/// @f$ Q = \mathrm{Re}\,\omega / (-2\,\mathrm{Im}\,\omega) @f$. The pencil (S, M) with the
/// stretched material tensors is solved by `solvers::complex_eigenpairs_near` around the
/// target frequency. See docs/theory/maxwell.md#resonances.
#include <optional>
#include <vector>

#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/mesh.hpp"
#include "hpfem/pml/pml.hpp"
#include "hpfem/solvers/eigen_solver.hpp"
#include "hpfem/solvers/linear_solver.hpp"

namespace hpfem::physics {

/// Description of a resonance problem.
template <int Dim>
struct ResonanceSetup {
  Real target_omega = 0;                ///< search centre [rad/s] (also the PML design frequency)
  materials::MaterialMap materials;     ///< by cell tag; may be lossy
  std::vector<mesh::Tag> pec_tags;      ///< facets with n × E = 0
  std::optional<pml::PmlBox<Dim>> pml;  ///< absorbing layers (open directions)
  Index num_modes = 4;                  ///< eigenvalues wanted around the target
  Index krylov_dimension = 0;           ///< 0: 2 num_modes + 10
  Real tolerance = 1e-10;
  int max_iterations = 100;  ///< Arnoldi restarts
  solvers::DirectSolverBackend solver = solvers::DirectSolverBackend::kAuto;
  int extra_quadrature_order = 2;
  int pml_extra_quadrature_order = 6;
};

/// One resonance: complex angular frequency, vacuum wavelength of the real part, quality
/// factor and the field coefficients (unit 2-norm, zero on PEC DoFs).
struct ResonantMode {
  Complex omega;    ///< [rad/s], Im < 0 for a decaying mode
  Real wavelength;  ///< 2π c0 / Re ω [m]
  Real quality;     ///< Re ω / (−2 Im ω)
  Real residual;    ///< relative Arnoldi residual
  Vector field;     ///< E coefficients on the DoF map
};

/// Assembles and solves the resonance problem on a Nédélec space.
template <int Dim>
class Resonance {
 public:
  /// @throws InvalidArgument if the target frequency is not positive.
  Resonance(const fespace::NedelecDofMap<Dim>& dofs, ResonanceSetup<Dim> setup);
  [[nodiscard]] const fespace::NedelecDofMap<Dim>& dofs() const noexcept { return *dofs_; }
  [[nodiscard]] const ResonanceSetup<Dim>& setup() const noexcept { return setup_; }
  /// Relative tensors of cell c (PML-stretched in the layers), no sources.
  [[nodiscard]] assembly::MaxwellForm<Dim> form_of_cell(Index cell) const;
  /// Modes ordered by the distance of ω to the target; fewer than `num_modes` if the Arnoldi
  /// iteration converged fewer. @throws Error on non-convergence.
  [[nodiscard]] std::vector<ResonantMode> solve() const;

 private:
  const fespace::NedelecDofMap<Dim>* dofs_;
  ResonanceSetup<Dim> setup_;
};

extern template struct ResonanceSetup<2>;
extern template struct ResonanceSetup<3>;
extern template class Resonance<2>;
extern template class Resonance<3>;

}  // namespace hpfem::physics

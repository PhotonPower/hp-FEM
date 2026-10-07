#pragma once
/// @file conical_resonance.hpp
/// Resonances (quasi-normal modes) of z-invariant structures in the conical (2.5D) setting
/// (M15 F14): the source-free pencil of `assembly/conical_forms.hpp`,
/// @f$ S(\beta)\,(e, v) = k_0^2\,M\,(e, v) @f$ with the longitudinal wavenumber β, PEC walls,
/// Bloch-periodic constraints (the phases of the pairs give the Bloch wavenumber along the
/// period) and a PML designed at the target frequency, solved by
/// `solvers::complex_eigenpairs_near_gauged` around @f$ k_{\text{target}}^2 @f$ with the
/// gradient kernel @f$ K_\beta = [G;\ \beta I] @f$ projected out. At β = 0 the pencil
/// decouples into the in-plane (H_z) and the E_z family, so one call returns the resonances
/// of both polarisations of a grating unit cell; at β ≠ 0 the modes are conical. Convention
/// exp(−iωt): a decaying mode has Im ω < 0, @f$ Q = \mathrm{Re}\,\omega / (-2\,\mathrm{Im}\,\omega)
/// @f$. See docs/theory/maxwell.md#conical-resonances.
#include <optional>
#include <vector>

#include "hpfem/assembly/conical_forms.hpp"
#include "hpfem/assembly/periodic.hpp"
#include "hpfem/core/progress.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/fespace/constraints.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/mesh.hpp"
#include "hpfem/physics/conical_scattering.hpp"
#include "hpfem/pml/pml.hpp"
#include "hpfem/solvers/linear_solver.hpp"

namespace hpfem::physics {

/// Description of a conical resonance problem.
struct ConicalResonanceSetup {
  Real target_omega = 0;             ///< search centre [rad/s] (also the PML design frequency)
  Real beta = 0;                     ///< longitudinal wavenumber k_z [1/m]
  materials::MaterialMap materials;  ///< by cell tag (per-cell overrides allowed); may be lossy
  std::vector<mesh::Tag> pec_tags;   ///< facets with n × E = 0
  std::optional<pml::PmlBox<2>> pml;
  std::vector<assembly::PeriodicPair<2>> periodic;  ///< Bloch-periodic directions with phases
  Index num_modes = 4;                              ///< eigenvalues wanted around the target
  Index krylov_dimension = 0;                       ///< 0: 2 num_modes + 10
  Real tolerance = 1e-10;
  int max_iterations = 100;  ///< Arnoldi restarts
  /// Project the gradient kernel (eigenvalue 0, as many as H1 DoFs) out of the Krylov space;
  /// off, the kernel is only kept away by the distance of the target from zero.
  bool remove_gradients = true;
  solvers::DirectSolverBackend solver = solvers::DirectSolverBackend::kAuto;
  int extra_quadrature_order = 2;
  int pml_extra_quadrature_order = 6;
  /// Phases of `solve`: assembly, constraints, eigensolve, post (see core/progress.hpp).
  ProgressCallback progress;
};

/// One conical resonance: complex angular frequency, vacuum wavelength of the real part,
/// quality factor, Arnoldi residual and the block coefficients (unit 2-norm of (e, v), zero
/// on PEC DoFs).
struct ConicalResonantMode {
  Complex omega;        ///< [rad/s], Im < 0 for a decaying mode
  Real wavelength = 0;  ///< 2π c0 / Re ω [m]
  Real quality = 0;     ///< Re ω / (−2 Im ω)
  Real residual = 0;    ///< relative Arnoldi residual
  Real beta = 0;        ///< the longitudinal wavenumber of the setup
  Vector transverse;    ///< (E_x, E_y) coefficients (full size)
  Vector longitudinal;  ///< v = −i E_z coefficients (full size)
  /// The mode as a (total-field) conical solution, for the post-processing of
  /// `conical_scattering.hpp` that takes coefficient vectors.
  [[nodiscard]] ConicalSolution solution() const;
};

/// Modes ordered by the distance of ω to the target, and the timing of the solve.
struct ConicalResonanceResult {
  std::vector<ConicalResonantMode> modes;
  Timing timing;
};

/// Assembles and solves the conical resonance problem on the block space.
class ConicalResonance {
 public:
  /// @throws InvalidArgument for a non-positive target frequency, maps on different meshes
  ///         or `num_modes` < 1.
  ConicalResonance(const fespace::NedelecDofMap<2>& transverse,
                   const fespace::DofMap<2>& longitudinal, ConicalResonanceSetup setup);

  [[nodiscard]] const ConicalResonanceSetup& setup() const noexcept { return setup_; }
  [[nodiscard]] const fespace::NedelecDofMap<2>& transverse_dofs() const noexcept {
    return *transverse_;
  }
  [[nodiscard]] const fespace::DofMap<2>& longitudinal_dofs() const noexcept {
    return *longitudinal_;
  }
  /// Target vacuum wavenumber k0 [1/m].
  [[nodiscard]] Real wavenumber() const noexcept { return k0_; }
  /// Free block DoFs (not on PEC facets), in-plane first.
  [[nodiscard]] const std::vector<Index>& free_dofs() const noexcept { return free_; }
  /// Relative tensors of cell c (PML-stretched in the layers), no sources.
  [[nodiscard]] assembly::ConicalForm form_of_cell(Index cell) const;
  /// Modes closest to the target; fewer than `num_modes` if fewer converged.
  /// @throws Error on non-convergence, Cancelled from the progress callback.
  [[nodiscard]] ConicalResonanceResult solve() const;

  /// Physical field (E_x, E_y, E_z) of a mode at reference point ξ of cell c.
  [[nodiscard]] ConicalVector field(const ConicalResonantMode& mode, Index cell,
                                    const Point<2>& xi) const;
  /// Physical curl of the mode.
  [[nodiscard]] ConicalVector curl_field(const ConicalResonantMode& mode, Index cell,
                                         const Point<2>& xi) const;
  /// @f$ H = \nabla\times E/(i\omega\mu_0\mu_r) @f$ with the mode's complex ω [A/m per unit of E].
  [[nodiscard]] ConicalVector h_field(const ConicalResonantMode& mode, Index cell,
                                      const Point<2>& xi) const;
  /// Time-averaged Poynting vector @f$ \tfrac12\mathrm{Re}(E\times\bar H) @f$ of the mode.
  [[nodiscard]] Point<3> poynting(const ConicalResonantMode& mode, Index cell,
                                  const Point<2>& xi) const;

 private:
  const fespace::NedelecDofMap<2>* transverse_;
  const fespace::DofMap<2>* longitudinal_;
  ConicalResonanceSetup setup_;
  Real k0_ = 0;
  std::vector<Index> free_;
  std::vector<Index> free_nd_;
  std::vector<Index> free_h1_;
  std::optional<fespace::Constraints> constraints_;
  std::optional<fespace::Constraints> h1_constraints_;
};

}  // namespace hpfem::physics

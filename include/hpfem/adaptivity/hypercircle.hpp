#pragma once
/// @file hypercircle.hpp
/// Dual (complementary) formulation of the curl–curl problem and the hypercircle
/// (Prager–Synge) error bound. For the primal problem
/// @f$ \nabla\times(\mu^{-1}\nabla\times E) - k^2\varepsilon E = f + \nabla\times g @f$ the
/// dual unknown is the (scaled) magnetic field @f$ \sigma = \mu^{-1}\nabla\times E - g @f$,
/// which satisfies @f$ \nabla\times\sigma = f + k^2\varepsilon E @f$ and therefore the dual
/// curl–curl problem
/// @f$ \nabla\times((k^2\varepsilon)^{-1}\nabla\times\sigma) - \mu\sigma =
///     \nabla\times((k^2\varepsilon)^{-1} f) + \mu g @f$.
/// In 3D @f$ \sigma @f$ lives in the same Nédélec space as E; in 2D the curl of a vector is
/// a scalar, so @f$ \sigma @f$ is an H1 function with @f$ \nabla\times\sigma =
/// (\partial_y\sigma, -\partial_x\sigma) @f$. For a pair @f$ (E_h, \sigma_h) @f$ of
/// discrete fields the constitutive-relation error
/// @f$ \eta_K^2 = \|\sigma_h - (\mu^{-1}\nabla\times E_h - g)\|^2_{\mu,K}
///              + \|(k^2\varepsilon)^{-1}(\nabla\times\sigma_h - f) - E_h\|^2_{-k^2\varepsilon,K}
///              @f$
/// is computable without any constant, and for the coercive problem (k² < 0, real
/// symmetric positive μ and ε) it is a **guaranteed upper bound** of the energy error:
/// @f$ \|\mu^{-1/2}\nabla\times(E - E_h)\|^2 - k^2\|\varepsilon^{1/2}(E - E_h)\|^2
///     \le \sum_K \eta_K^2 @f$ for *any* @f$ \sigma_h @f$ with the dual essential boundary
/// condition (the Galerkin solution of the dual problem gives the sharpest bound). For the
/// indefinite time-harmonic problem the same quantity is the constitutive-relation error
/// estimator (reliable up to the inf-sup constant). See
/// docs/theory/error-estimation.md#dual-formulation-and-guaranteed-bounds.
#include <span>
#include <type_traits>
#include <vector>

#include "hpfem/assembly/h1_forms.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/solvers/linear_solver.hpp"

namespace hpfem::adaptivity {

/// Space of the dual unknown: H1 in 2D (scalar curl), Nédélec in 3D.
template <int Dim>
using DualDofMap = std::conditional_t<Dim == 2, fespace::DofMap<2>, fespace::NedelecDofMap<3>>;
/// Per-cell form of the dual problem in the units of the respective assembler.
template <int Dim>
using DualForm = std::conditional_t<Dim == 2, assembly::ScalarForm<2>, assembly::MaxwellForm<3>>;

/// The dual form of one cell: coefficients @f$ (k^2\varepsilon)^{-1} @f$ (stiffness),
/// @f$ \mu @f$ (mass, entering as `permittivity` with the mass coefficient 1 in 3D and as the
/// reaction −μ in 2D), the sources @f$ \mu g @f$ and @f$ (k^2\varepsilon)^{-1} f @f$ (paired
/// with the curl of the test function). Empty coefficient functions of `primal` mean the
/// identity (μ⁻¹, ε) or zero (f, g).
/// @throws InvalidArgument if k² is zero.
template <int Dim>
[[nodiscard]] DualForm<Dim> dual_form(const assembly::MaxwellForm<Dim>& primal, Real k_squared);

/// Galerkin solution of the dual problem on `dual_dofs` for the per-cell primal forms and the
/// mass coefficient k² of the primal problem. `essential_facets` are the facets where the
/// primal problem has the natural (PMC) condition: there the dual field has the essential one
/// (tangential trace zero in 3D, value zero in 2D). Where the primal field is PEC nothing is
/// imposed on the dual field. Constrained meshes (hanging nodes, Bloch) are not supported.
/// @throws InvalidArgument if k² is zero.
template <int Dim>
[[nodiscard]] Vector dual_solution(
    const DualDofMap<Dim>& dual_dofs,
    const std::type_identity_t<assembly::CellFormFactory<Dim>>& form_of_cell, Real k_squared,
    std::span<const Index> essential_facets, int extra_order = 2,
    solvers::DirectSolverBackend backend = solvers::DirectSolverBackend::kAuto);

/// Squared contributions of the two constitutive relations of one cell.
struct HypercircleParts {
  Real constitutive = 0;  ///< ‖σ_h − (μ⁻¹ curl E_h − g)‖²_μ
  Real equilibrium = 0;   ///< ‖(k²ε)⁻¹(curl σ_h − f) − E_h‖²_{−k²ε}
  [[nodiscard]] Real sum() const noexcept { return constitutive + equilibrium; }
};

/// Element indicators of a primal/dual pair.
struct HypercircleEstimate {
  std::vector<Real> indicators;         ///< η_K per cell
  std::vector<HypercircleParts> parts;  ///< the squared contributions behind η_K²
  /// Global bound @f$ \eta = (\sum_K \eta_K^2)^{1/2} @f$.
  [[nodiscard]] Real total() const;
  /// Cell with the largest indicator (`kInvalidIndex` if there are none).
  [[nodiscard]] Index argmax() const;
};

/// Indicators of the primal coefficients `e_h` on `dofs` and the dual coefficients `sigma_h`
/// on `dual_dofs` (same mesh) for the per-cell forms and the mass coefficient k². The
/// weighted norms are @f$ |w^H T w| @f$ with the energy weights @f$ T = \mu @f$ and
/// @f$ T = -k^2\varepsilon @f$, which are the exact energy norms for real coefficients and
/// k² < 0 and their moduli otherwise (lossy materials, PML). Quadrature of degree
/// 2p + `extra_order` (+2 on curved cells).
/// @throws InvalidArgument if a coefficient vector does not match its map or the maps have
/// different meshes.
template <int Dim>
[[nodiscard]] HypercircleEstimate hypercircle_estimate(
    const fespace::NedelecDofMap<Dim>& dofs, const Vector& e_h, const DualDofMap<Dim>& dual_dofs,
    const Vector& sigma_h, const std::type_identity_t<assembly::CellFormFactory<Dim>>& form_of_cell,
    Real k_squared, int extra_order = 2);

extern template DualForm<2> dual_form<2>(const assembly::MaxwellForm<2>&, Real);
extern template DualForm<3> dual_form<3>(const assembly::MaxwellForm<3>&, Real);
extern template Vector dual_solution<2>(const DualDofMap<2>&, const assembly::CellFormFactory<2>&,
                                        Real, std::span<const Index>, int,
                                        solvers::DirectSolverBackend);
extern template Vector dual_solution<3>(const DualDofMap<3>&, const assembly::CellFormFactory<3>&,
                                        Real, std::span<const Index>, int,
                                        solvers::DirectSolverBackend);
extern template HypercircleEstimate hypercircle_estimate<2>(const fespace::NedelecDofMap<2>&,
                                                            const Vector&, const DualDofMap<2>&,
                                                            const Vector&,
                                                            const assembly::CellFormFactory<2>&,
                                                            Real, int);
extern template HypercircleEstimate hypercircle_estimate<3>(const fespace::NedelecDofMap<3>&,
                                                            const Vector&, const DualDofMap<3>&,
                                                            const Vector&,
                                                            const assembly::CellFormFactory<3>&,
                                                            Real, int);

}  // namespace hpfem::adaptivity

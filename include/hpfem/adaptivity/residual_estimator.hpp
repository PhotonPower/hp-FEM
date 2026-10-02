#pragma once
/// @file residual_estimator.hpp
/// Residual-based a-posteriori error estimator for the time-harmonic curl–curl problem
/// @f$ \nabla\times(\mu^{-1}\nabla\times E) - k^2\varepsilon E = f + \nabla\times g @f$ on a
/// Nédélec space (Beck, Hiptmair, Hoppe & Wohlmuth 2000; hp weights after Schöberl 2008).
/// Per cell K with diameter h_K and order p_K,
/// @f[
///   \eta_K^2 = \frac{h_K^2}{p_K^2}\,\|R_K\|^2_{L^2(K)}
///            + \frac{h_K^2}{p_K^2}\,\|\nabla\cdot d\|^2_{L^2(K)}
///            + \sum_{F\subset\partial K}\frac{h_F}{2p_F}\Big(\|[\![n\times w]\!]\|^2_{L^2(F)}
///            + \|[\![n\cdot d]\!]\|^2_{L^2(F)}\Big),
/// @f]
/// with @f$ w = \mu^{-1}\nabla\times E_{hp} - g @f$, @f$ d = f + k^2\varepsilon E_{hp} @f$ and the
/// element residual @f$ R_K = d - \nabla\times w @f$. Interior facets only (hanging child
/// facets against the cell of their parent): PEC facets carry no residual, natural (PMC)
/// and periodic facets are not yet accounted for. Complex
/// coefficients and PML cells enter through the per-cell form, so the estimator measures
/// the residual of the equation actually solved. Convention exp(-iωt) as everywhere.
/// See docs/theory/error-estimation.md.

#include <type_traits>
#include <vector>

#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"

namespace hpfem::adaptivity {

struct EstimatorOptions {
  /// Added to 2p for the quadrature degree on cells and facets (+2 on curved cells).
  int extra_order = 2;
  /// Include the Gauss-law terms (divergence residual and normal-flux jumps).
  bool divergence_terms = true;
  /// Step of the central differences, in reference coordinates, that supply the
  /// derivatives of @f$ w @f$ and @f$ d @f$ inside a cell (both are polynomial in ξ on
  /// affine cells with constant coefficients, so the differences are exact up to rounding).
  Real difference_step = 1e-4;
};

/// Squared contributions of the four residual terms of one cell.
struct ResidualParts {
  Real element = 0;          ///< (h/p)² ‖R_K‖²
  Real divergence = 0;       ///< (h/p)² ‖∇·d‖²
  Real tangential_jump = 0;  ///< Σ_F h_F/(2p_F) ‖[n × w]‖²
  Real normal_jump = 0;      ///< Σ_F h_F/(2p_F) ‖[n · d]‖²
  [[nodiscard]] Real sum() const noexcept {
    return element + divergence + tangential_jump + normal_jump;
  }
};

/// Element indicators of one discrete solution.
struct Estimate {
  std::vector<Real> indicators;      ///< η_K per cell
  std::vector<ResidualParts> parts;  ///< the squared contributions behind η_K²
  /// Global estimate @f$ \eta = (\sum_K \eta_K^2)^{1/2} @f$.
  [[nodiscard]] Real total() const;
  /// Cell with the largest indicator (`kInvalidIndex` if there are none).
  [[nodiscard]] Index argmax() const;
};

/// Indicators of the coefficients `e_h` on `dofs` for the problem given by the per-cell
/// forms (coefficients μ⁻¹, ε, f, g of `assembly::MaxwellForm`) and the mass coefficient
/// `k_squared` (k0² for `physics::Scattering`, ω² with SI tensors). Cost: a few basis
/// evaluations per quadrature point on every cell and interior facet.
/// @throws InvalidArgument if `e_h` does not match the DoF map.
template <int Dim>
[[nodiscard]] Estimate residual_estimate(
    const fespace::NedelecDofMap<Dim>& dofs, const Vector& e_h, Real k_squared,
    const std::type_identity_t<assembly::CellFormFactory<Dim>>& form_of_cell,
    const EstimatorOptions& options = {});

extern template Estimate residual_estimate<2>(const fespace::NedelecDofMap<2>&, const Vector&, Real,
                                              const assembly::CellFormFactory<2>&,
                                              const EstimatorOptions&);
extern template Estimate residual_estimate<3>(const fespace::NedelecDofMap<3>&, const Vector&, Real,
                                              const assembly::CellFormFactory<3>&,
                                              const EstimatorOptions&);

}  // namespace hpfem::adaptivity

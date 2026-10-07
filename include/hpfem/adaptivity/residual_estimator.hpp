#pragma once
/// @file residual_estimator.hpp
/// Residual-based a-posteriori error estimator for the time-harmonic curl–curl problem
/// @f$ \nabla\times(\mu^{-1}\nabla\times E) - k^2\varepsilon E = f + \nabla\times g @f$ on a
/// Nédélec space (Beck, Hiptmair, Hoppe & Wohlmuth 2000; hp weights after Schöberl 2008).
/// Per cell K with diameter h_K and order p_K,
/// @f[
///   \eta_K^2 = \frac{h_K^2}{p_K^2}\,\|R_K\|^2_{L^2(K)}
///            + \frac{h_K^2}{p_K^2}\,\ell^2\|\nabla\cdot d\|^2_{L^2(K)}
///            + \sum_{F\subset\partial K}\frac{h_F}{2p_F}\Big(\|[\![n\times w]\!]\|^2_{L^2(F)}
///            + \ell^2\|[\![n\cdot d]\!]\|^2_{L^2(F)}\Big),
/// @f]
/// with @f$ w = \mu^{-1}\nabla\times E_{hp} - g @f$, @f$ d = f + k^2\varepsilon E_{hp} @f$, the
/// element residual @f$ R_K = d - \nabla\times w @f$ and the length scale @f$ \ell = 1/k @f$ of the
/// Gauss-law terms (`EstimatorOptions::length_scale`): @f$ \nabla\cdot d @f$ carries one inverse
/// length more than @f$ R_K @f$, so without @f$ \ell @f$ the published form holds only for lengths
/// of order one and the Gauss-law terms dominate SI-scale problems by @f$ 1/(kh)^2 @f$. Interior
/// facets (hanging child facets against the cell of their parent) and, with `periodic`, the
/// Bloch slave facets, whose jump is taken against the phase-shifted master cells
/// (`assembly::PeriodicLocator`) and credited to both sides; PEC facets carry no residual,
/// natural (PMC) facets are not accounted for. Complex coefficients and
/// PML cells enter through the per-cell form, so the estimator measures the residual of the
/// equation actually solved. Convention exp(-iωt) as everywhere. See
/// docs/theory/error-estimation.md.

#include <type_traits>
#include <vector>

#include <span>

#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/assembly/periodic.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"

namespace hpfem::adaptivity {

struct EstimatorOptions {
  /// Added to 2p for the quadrature degree on cells and facets (+2 on curved cells).
  int extra_order = 2;
  /// Include the Gauss-law terms (divergence residual and normal-flux jumps).
  bool divergence_terms = true;
  /// Length scale ℓ [m] multiplying the Gauss-law terms (squared), which makes them
  /// dimensionally consistent with the curl–curl residual. 0 (default): ℓ = 1/k from the
  /// `k_squared` argument, the wavelength scale (ℓ = 1 if k_squared is 0); set ℓ = c0/ω when
  /// `k_squared` is ω² with SI tensors.
  Real length_scale = 0;
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
    const EstimatorOptions& options = {},
    std::span<const assembly::PeriodicPair<Dim>> periodic = {});

/// Weighted residual for goal-oriented estimation: the cell contributions
/// @f$ r_K(w) = \int_K R_K\cdot w + \tfrac12\sum_{F\subset\partial K}\int_F
/// (n\times[\![w_h]\!])\cdot w @f$ (no conjugation, signed) of a weight @f$ w @f$ given as
/// coefficients on `weight_dofs` (same mesh, e.g. the orders raised by one); boundary facets
/// contribute their one-sided term
/// @f$ \int_F (n	imes w_h)\cdot w @f$ to their cell (the residual of a natural condition),
/// so that @f$ \sum_K r_K(w) = \ell(w) - a(E_h, w) @f$ for every tangentially continuous
/// @f$ w @f$.
template <int Dim>
[[nodiscard]] std::vector<Complex> weighted_residual(
    const fespace::NedelecDofMap<Dim>& dofs, const Vector& e_h, Real k_squared,
    const std::type_identity_t<assembly::CellFormFactory<Dim>>& form_of_cell,
    const fespace::NedelecDofMap<Dim>& weight_dofs, const Vector& weight,
    const EstimatorOptions& options = {});

extern template std::vector<Complex> weighted_residual<2>(const fespace::NedelecDofMap<2>&,
                                                          const Vector&, Real,
                                                          const assembly::CellFormFactory<2>&,
                                                          const fespace::NedelecDofMap<2>&,
                                                          const Vector&, const EstimatorOptions&);
extern template std::vector<Complex> weighted_residual<3>(const fespace::NedelecDofMap<3>&,
                                                          const Vector&, Real,
                                                          const assembly::CellFormFactory<3>&,
                                                          const fespace::NedelecDofMap<3>&,
                                                          const Vector&, const EstimatorOptions&);

extern template Estimate residual_estimate<2>(const fespace::NedelecDofMap<2>&, const Vector&, Real,
                                              const assembly::CellFormFactory<2>&,
                                              const EstimatorOptions&,
                                              std::span<const assembly::PeriodicPair<2>>);
extern template Estimate residual_estimate<3>(const fespace::NedelecDofMap<3>&, const Vector&, Real,
                                              const assembly::CellFormFactory<3>&,
                                              const EstimatorOptions&,
                                              std::span<const assembly::PeriodicPair<3>>);

}  // namespace hpfem::adaptivity

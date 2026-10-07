#pragma once
/// @file axisymmetric_estimator.hpp
/// Residual-based a-posteriori error estimator of the axisymmetric (2.5D) curl–curl problem
/// of one azimuthal order m on the meridian mesh (`assembly/axisymmetric_forms.hpp`): the
/// residual of the three-dimensional equation
/// @f$ \nabla\times(\mu^{-1}\nabla\times E) - k^2\varepsilon E = f @f$ for the mode
/// @f$ E(r, z)e^{im\varphi} @f$, measured in the norms of the body of revolution, i.e. with the
/// weight r of the volume element @f$ r\,dr\,dz @f$ (the factor 2π dropped as in the forms).
/// Per cell K of the meridian mesh with diameter h_K and order p_K,
/// @f[
///   \eta_K^2 = \frac{h_K^2}{p_K^2}\int_K\big(|R_K|^2 + \ell^2|\nabla\cdot d|^2\big)\,r\,dr\,dz
///            + \sum_{F\subset\partial K}\frac{h_F}{2p_F}\int_F\big(|[\![n\times w]\!]|^2
///            + \ell^2|[\![n\cdot d]\!]|^2\big)\,r\,ds,
/// @f]
/// with @f$ w = \mu^{-1}\nabla\times E_{hp} @f$, @f$ d = f + k^2\varepsilon E_{hp} @f$ and
/// @f$ R_K = d - \nabla\times w @f$, all in cylindrical components: for the scaled unknowns
/// @f$ (E_r, v = -irE_\varphi, E_z) @f$ the curl of the mode is
/// @f$ \nabla\times E = \big(\tfrac{i}{r}(mE_z - \partial_z v),\ \partial_zE_r - \partial_rE_z,
/// \ \tfrac{i}{r}(\partial_r v - mE_r)\big) @f$, and for an unscaled vector field
/// @f$ (\nabla\times w)_r = \tfrac{im}{r}w_z - \partial_z w_\varphi @f$,
/// @f$ (\nabla\times w)_\varphi = \partial_z w_r - \partial_r w_z @f$,
/// @f$ (\nabla\times w)_z = \tfrac1r\partial_r(rw_\varphi) - \tfrac{im}{r}w_r @f$,
/// @f$ \nabla\cdot d = \tfrac1r\partial_r(rd_r) + \tfrac{im}{r}d_\varphi + \partial_z d_z @f$.
/// The facet normal lies in the meridian plane, so @f$ |n\times w|^2 = |w_\varphi|^2 +
/// |n_zw_r - n_rw_z|^2 @f$ and @f$ n\cdot d = n_rd_r + n_zd_z @f$. Interior facets only
/// (hanging child facets against the cell of their parent); facets on the axis and on PEC
/// walls carry no term, the axis conditions being essential. On cells touching the axis the
/// 1/r terms of the residual are rational; the combinations that vanish for the regular
/// solution (e.g. @f$ w_\varphi - imw_r @f$ at r = 0 for |m| = 1) do so only approximately
/// for the discrete one, which adds a logarithmic factor to the indicator of those cells
/// (the quadrature avoids r = 0). Convention exp(-iωt). See
/// docs/theory/axisymmetric.md#hp-adaptivity-on-the-meridian-plane.

#include "hpfem/adaptivity/residual_estimator.hpp"
#include "hpfem/assembly/axisymmetric_forms.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"

namespace hpfem::adaptivity {

/// Indicators of the order-m field with the coefficients `meridian` (E_r, E_z on the Nédélec
/// map) and `azimuthal` (v on the H1 map) for the problem given by the per-cell forms
/// (@f$ \mu^{-1}, \varepsilon, f @f$ of `assembly::AxisymmetricForm`) and the mass
/// coefficient `k_squared` (k0² with relative tensors). Quadrature of degree 2p +
/// `options.extra_order` (+2 on cells touching the axis and on curved cells) or the cell's
/// own `quadrature_order`; derivatives by central differences in reference coordinates
/// (`options.difference_step`) as in `residual_estimate`. Cost: a few basis evaluations per
/// quadrature point on every cell and interior facet, parallel over cells and facets.
/// @throws InvalidArgument if the maps differ in mesh, a coefficient vector does not match
///         its map, or the difference step is not positive.
[[nodiscard]] Estimate axisymmetric_residual_estimate(
    const fespace::NedelecDofMap<2>& nedelec, const fespace::DofMap<2>& h1, const Vector& meridian,
    const Vector& azimuthal, int m, Real k_squared,
    const assembly::AxisymmetricFormFactory& form_of_cell, const EstimatorOptions& options = {});

}  // namespace hpfem::adaptivity

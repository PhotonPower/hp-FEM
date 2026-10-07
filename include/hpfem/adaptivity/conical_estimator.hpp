#pragma once
/// @file conical_estimator.hpp
/// Residual-based a-posteriori error estimator of the conical (2.5D) curl–curl problem
/// (`assembly/conical_forms.hpp`): the residual of the three-dimensional equation
/// @f$ \nabla\times(\mu^{-1}\nabla\times E) - k^2\varepsilon E = f @f$ for the field
/// @f$ E(x, y)e^{i\beta z} @f$, so that @f$ \partial_z = i\beta @f$. Per cell K of the 2D mesh with
/// diameter h_K and order p_K,
/// @f[
///   \eta_K^2 = \frac{h_K^2}{p_K^2}\int_K\big(|R_K|^2 + \ell^2|\nabla\cdot d|^2\big)\,dx\,dy
///            + \sum_{F\subset\partial K}\frac{h_F}{2p_F}\int_F\big(|[\![n\times w]\!]|^2
///            + \ell^2|[\![n\cdot d]\!]|^2\big)\,ds,
/// @f]
/// (@f$ \ell = 1/k @f$ the length scale of the Gauss-law terms, `EstimatorOptions::length_scale`)
/// with @f$ w = \mu^{-1}\nabla\times E_{hp} @f$, @f$ d = f + k^2\varepsilon E_{hp} @f$ and
/// @f$ R_K = d - \nabla\times w @f$ in Cartesian components: for the scaled unknowns
/// @f$ (E_x, E_y, v = -iE_z) @f$ the curl of the mode is @f$ \nabla\times E = (i(\partial_yv -
/// \beta E_y),\ -i(\partial_xv - \beta E_x),\ \partial_xE_y - \partial_yE_x) @f$, and for an
/// unscaled vector @f$ (\nabla\times w)_x = \partial_yw_z - i\beta w_y @f$,
/// @f$ (\nabla\times w)_y = i\beta w_x - \partial_xw_z @f$, @f$ (\nabla\times w)_z = \partial_xw_y
/// - \partial_yw_x @f$, @f$ \nabla\cdot d = \partial_xd_x + \partial_yd_y + i\beta d_z @f$. The
/// facet normal lies in the plane, so @f$ |n\times w|^2 = |w_z|^2 + |n_xw_y - n_yw_x|^2 @f$ and
/// @f$ n\cdot d = n_xd_x + n_yd_y @f$. Interior facets only (hanging child facets against the
/// cell of their parent); PEC and Bloch-periodic facets carry no term. At β = 0 the indicators
/// are those of the in-plane and the E_z block side by side. Convention exp(-iωt). See
/// docs/theory/maxwell.md#conical-incidence-and-the-e_z-polarisation.

#include <vector>

#include "hpfem/adaptivity/residual_estimator.hpp"
#include "hpfem/assembly/conical_forms.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"

namespace hpfem::adaptivity {

/// Indicators of the conical field with the coefficients `transverse` (E_x, E_y on the
/// Nédélec map) and `longitudinal` (v on the H1 map) for the problem given by the per-cell
/// forms (@f$ \mu^{-1}, \varepsilon, f @f$ of `assembly::ConicalForm`), the longitudinal
/// wavenumber β and the mass coefficient `k_squared` (k0² with relative tensors). Quadrature
/// of degree 2p + `options.extra_order` (+2 on curved cells) or the cell's own
/// `quadrature_order`; derivatives by central differences in reference coordinates
/// (`options.difference_step`) as in `residual_estimate`. Parallel over cells and facets.
/// @throws InvalidArgument if the maps differ in mesh, a coefficient vector does not match
///         its map, or the difference step is not positive.
[[nodiscard]] Estimate conical_residual_estimate(const fespace::NedelecDofMap<2>& transverse,
                                                 const fespace::DofMap<2>& longitudinal,
                                                 const Vector& e, const Vector& v, Real beta,
                                                 Real k_squared,
                                                 const assembly::ConicalFormFactory& form_of_cell,
                                                 const EstimatorOptions& options = {});

/// Weighted residual of the conical system for goal-oriented estimation: the cell
/// contributions @f$ r_K(W) = \int_K R_K\cdot W + \tfrac12\sum_{F\subset\partial K}\int_F
/// (n\times[\![w_h]\!])\cdot W @f$ (no conjugation) of a weight given as coefficients
/// (`weight_e` on `weight_transverse`, `weight_v` on `weight_longitudinal`, both on the same
/// mesh, e.g. the orders raised by one), with the physical test vector
/// @f$ W = (W_x, W_y, -i\,w) @f$ of the scaled test function w (the pairing of the conical forms);
/// boundary facets contribute their one-sided term to their cell, so that
/// @f$ \sum_K r_K(W) = \ell(W) - a(E_h, W) @f$ for every tangentially continuous W.
/// @throws InvalidArgument for mismatched maps or vectors.
[[nodiscard]] std::vector<Complex> conical_weighted_residual(
    const fespace::NedelecDofMap<2>& transverse, const fespace::DofMap<2>& longitudinal,
    const Vector& e, const Vector& v, Real beta, Real k_squared,
    const assembly::ConicalFormFactory& form_of_cell,
    const fespace::NedelecDofMap<2>& weight_transverse,
    const fespace::DofMap<2>& weight_longitudinal, const Vector& weight_e, const Vector& weight_v,
    const EstimatorOptions& options = {});

}  // namespace hpfem::adaptivity

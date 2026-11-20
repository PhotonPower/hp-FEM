#pragma once
/// @file elasticity_estimator.hpp
/// Residual-based a-posteriori error estimator of small-strain linear elasticity
/// @f$ -\nabla\cdot\sigma = f @f$, @f$ \sigma = C : (\varepsilon(u) - \varepsilon^*) @f$ on the
/// vector H1 space (`assembly/elasticity_forms.hpp`, ADR-0016), in the energy norm
/// @f$ |||e|||^2 = \int \varepsilon(e) : C : \varepsilon(e) @f$ (Verfürth 2013, ch. 4.3; hp
/// weights after Melenk & Wohlmuth 2001). Per cell K with diameter h_K, order p_K and the modulus
/// scale @f$ 2\mu_K @f$ (twice the shear entry of the Voigt stiffness at the centroid),
/// @f[
///   \eta_K^2 = \frac{1}{2\mu_K}\frac{h_K^2}{p_K^2}\|f + \nabla\cdot\sigma_h\|^2_{L^2(K)}
///            + \sum_{F\subset\partial K\setminus\Gamma}\frac{1}{2\mu_F}\frac{h_F}{2p_F}
///              \|[\![\sigma_h n]\!]\|^2_{L^2(F)}
///            + \sum_{F\subset\partial K\cap\Gamma_N}\frac{1}{2\mu_K}\frac{h_F}{p_F}
///              \|t - \sigma_h n\|^2_{L^2(F)},
/// @f]
/// with @f$ \mu_F @f$ the larger shear modulus of the two cells (robust for stiff / soft
/// interfaces), the interior facets including hanging child facets against the cell of their
/// parent, and on the natural boundary @f$ \Gamma_N @f$ only the traction components that are
/// not prescribed as displacements (symmetry planes: the tangential ones). The eigenstrain
/// enters through σ_h, so its divergence and its jumps across material interfaces are part of
/// the residual. In the `Estimate`, `parts.element` holds the volume term, `tangential_jump` the
/// interior traction jumps and `normal_jump` the boundary traction residual (`divergence` is
/// zero). The axisymmetric variant measures the residual of the three-dimensional equilibrium of
/// the body of revolution with the weight r of the volume element,
/// @f$ R_r = f_r + \partial_r\sigma_{rr} + \partial_z\sigma_{rz} + (\sigma_{rr} -
/// \sigma_{\theta\theta})/r @f$, @f$ R_z = f_z + \partial_r\sigma_{rz} + \partial_z\sigma_{zz} +
/// \sigma_{rz}/r @f$; the axis carries no term. See
/// docs/theory/elasticity.md#error-estimation-and-hp-adaptivity-m20-s5b.

#include <array>
#include <cstddef>
#include <functional>
#include <type_traits>

#include <Eigen/Core>

#include "hpfem/adaptivity/residual_estimator.hpp"
#include "hpfem/assembly/elasticity_forms.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"

namespace hpfem::adaptivity {

/// The natural condition on one boundary facet.
template <int Dim>
struct ElasticFacetCondition {
  /// Components whose traction is natural (prescribed or zero); the others are prescribed
  /// displacements and carry no residual.
  std::array<bool, static_cast<std::size_t>(Dim)> natural{};
  /// Prescribed traction t(x) = σ n [Pa]; empty means traction-free.
  std::function<Eigen::Matrix<Complex, Dim, 1>(const Point<Dim>&)> traction;
};
/// The condition of a boundary facet by its id.
template <int Dim>
using ElasticBoundaryFactory = std::function<ElasticFacetCondition<Dim>(Index facet)>;

/// Indicators of the block displacement coefficients `u` on `dofs` (plane strain in 2D) for the
/// per-cell forms (stiffness, body force, eigenstrains of `assembly::ElasticForm`) and the
/// boundary conditions `boundary` (called once per boundary facet). Quadrature of degree 2p +
/// `options.extra_order` (+2 on curved cells); the divergence of σ_h by central differences in
/// reference coordinates (`options.difference_step`), exact up to rounding on affine cells with
/// constant data. Parallel over cells and facets. `divergence_terms` and `length_scale` of the
/// options do not apply.
/// @throws InvalidArgument if `u` does not match the map, the difference step is not positive,
///         or a cell has no stiffness.
template <int Dim>
[[nodiscard]] Estimate elasticity_residual_estimate(
    const fespace::DofMap<Dim>& dofs, const Vector& u,
    const std::type_identity_t<assembly::ElasticFormFactory<Dim>>& form_of_cell,
    const std::type_identity_t<ElasticBoundaryFactory<Dim>>& boundary,
    const EstimatorOptions& options = {});

/// The same for the axisymmetric problem on the meridian half-plane x = r ≥ 0, y = z (the block
/// (u_r | u_z), `assembly::assemble_axisymmetric_elasticity`): all integrals with the weight r,
/// the hoop stress in the radial equilibrium, boundary facets on the axis r = 0 skipped (their
/// weight vanishes; give them no natural component). Quadrature +2 on cells touching the axis.
/// @throws InvalidArgument as `elasticity_residual_estimate`.
[[nodiscard]] Estimate axisymmetric_elasticity_residual_estimate(
    const fespace::DofMap<2>& dofs, const Vector& u,
    const assembly::AxisymmetricElasticFormFactory& form_of_cell,
    const ElasticBoundaryFactory<2>& boundary, const EstimatorOptions& options = {});

extern template Estimate elasticity_residual_estimate<2>(const fespace::DofMap<2>&, const Vector&,
                                                         const assembly::ElasticFormFactory<2>&,
                                                         const ElasticBoundaryFactory<2>&,
                                                         const EstimatorOptions&);
extern template Estimate elasticity_residual_estimate<3>(const fespace::DofMap<3>&, const Vector&,
                                                         const assembly::ElasticFormFactory<3>&,
                                                         const ElasticBoundaryFactory<3>&,
                                                         const EstimatorOptions&);

}  // namespace hpfem::adaptivity

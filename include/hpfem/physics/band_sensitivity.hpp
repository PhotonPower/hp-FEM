#pragma once
/// @file band_sensitivity.hpp
/// Derivatives of photonic bands (`BandStructure`) with respect to material and shape
/// parameters and the Bloch wave vector (group velocity), without a further eigensolve.
/// Convention exp(−iωt), SI units, lossless media. At the wave vector k the reduced pencil
/// @f$ A(p)\,u = \lambda\,B(p)\,u @f$ with @f$ A = P^H S P @f$, @f$ B = P^H M P @f$,
/// @f$ \lambda = k_0^2 @f$ is Hermitian, so the left eigenvector of a simple eigenvalue is
/// the right one and (Hellmann–Feynman)
/// @f[ \frac{d\lambda}{dp} = \frac{u^H(\partial_p A - \lambda\,\partial_p B)\,u}{u^H B u},
///     \qquad \frac{dk_0}{dp} = \frac{1}{2k_0}\frac{d\lambda}{dp},
///     \qquad \frac{d\omega}{dp} = c_0\,\frac{dk_0}{dp} . @f]
/// With the full-size mode @f$ w = P u @f$ (`Bands::modes`, kept with
/// `BandStructureSetup::keep_modes`):
/// - permittivity of the cells with a tag: @f$ \partial B = P^H M_{\text{tag}} P @f$, so
///   @f$ d\lambda = -\lambda\,w^H M_{\text{tag}} w @f$ (@f$ M_{\text{tag}} @f$: mass matrix of
///   the tagged cells with @f$ \varepsilon_r = 1 @f$);
/// - permeability: @f$ \partial A = -\mu_r^{-2}\,P^H S_{\text{tag}} P @f$;
/// - shape (mesh velocity V, ADR-0011): @f$ \partial S, \partial M @f$ by one central
///   directional difference of the element matrices per moving cell; the constraints do not
///   depend on the geometry because V must vanish on the periodic faces;
/// - wave vector along a direction d: only P depends on k,
///   @f$ \partial A - \lambda\,\partial B = \partial P^H (S - \lambda M) P
///   + P^H (S - \lambda M)\,\partial P @f$, with @f$ \partial P @f$ by a central difference of
///   the Bloch prolongation in k (no solve; the phases are smooth, truncation
///   @f$ O(10^{-10}) @f$ relative).
///
/// **Degenerate bands.** Consecutive bands whose eigenvalues differ by at most
/// `degeneracy_tolerance` relative (and bands at @f$ k_0 = 0 @f$) form a cluster with
/// modes W. Hellmann–Feynman does not apply to a multiple eigenvalue; its derivatives are the
/// eigenvalues of the small Hermitian pencil
/// @f[ W^H(\partial A - \tfrac12(\Lambda\,\partial B + \partial B\,\Lambda))W\,y
///     = \mu\,W^H B W\,y , @f]
/// assigned in ascending order to the bands of the cluster: they are the one-sided
/// derivatives of the branches that leave the degenerate point with increasing p (the
/// eigenvalues of @f$ \lambda(p + h) @f$ sorted, @f$ h \to 0^+ @f$). For a simple band this
/// is Hellmann–Feynman. For the group velocity at a degenerate point each component is the
/// sorted spectrum of its own cluster matrix; branch velocities along a path need the
/// directional derivative `band_wave_vector_derivative` along that path.
/// See docs/theory/maxwell.md#band-derivatives-and-group-velocity.
#include <vector>

#include "hpfem/core/types.hpp"
#include "hpfem/mesh/mesh.hpp"
#include "hpfem/physics/band_structure.hpp"
#include "hpfem/physics/shape_sensitivity.hpp"

namespace hpfem::physics {

/// Derivatives of the bands at one wave vector with respect to one parameter p, in the band
/// order of `Bands::wavenumber`.
struct BandDerivative {
  std::vector<Real> eigenvalue;  ///< @f$ d\lambda/dp @f$, @f$ \lambda = k_0^2 @f$ [1/m² per unit p]
  /// @f$ dk_0/dp @f$ [1/m per unit p]; NaN for a band at @f$ k_0 = 0 @f$ (below
  /// @f$ 10^{-6}\cdot 2\pi/a @f$), where @f$ k_0 = \sqrt\lambda @f$ is not differentiable.
  std::vector<Real> wavenumber;
  std::vector<Real> angular_frequency;  ///< @f$ d\omega/dp = c_0\,dk_0/dp @f$ [rad/s per unit p]
  std::vector<Index> multiplicity;      ///< size of the band's degenerate cluster (1: simple)
};

/// Derivative of the bands with respect to the relative permittivity @f$ \varepsilon_r @f$ of
/// the cells with `tag` (real, the bands are lossless).
/// @throws InvalidArgument if `bands` carries no modes (`keep_modes`), the modes do not
///         match the problem, no cell carries the tag or the tolerance is negative.
template <int Dim>
[[nodiscard]] BandDerivative band_permittivity_derivative(const BandStructure<Dim>& problem,
                                                          const Bands<Dim>& bands, mesh::Tag tag,
                                                          Real degeneracy_tolerance = 1e-6);

/// Derivative of the bands with respect to the relative permeability @f$ \mu_r @f$ of the
/// cells with `tag`. @throws InvalidArgument as `band_permittivity_derivative`.
template <int Dim>
[[nodiscard]] BandDerivative band_permeability_derivative(const BandStructure<Dim>& problem,
                                                          const Bands<Dim>& bands, mesh::Tag tag,
                                                          Real degeneracy_tolerance = 1e-6);

/// Derivative of the bands with respect to a geometry parameter with mesh velocity
/// @f$ V = \partial x/\partial p @f$ (one row per geometry node, `num_geometry_nodes`), e.g.
/// `region_normal_velocity` for the radius of a rod. The element matrices of every cell with
/// a moving node are differenced along V (largest node displacement `relative_step` times
/// the cell diameter).
/// @throws InvalidArgument as `band_permittivity_derivative`, if V does not match the
///         geometry nodes, V does not vanish on the periodic faces (the master and slave
///         facets of the lattice), the step is not positive or a perturbed cell degenerates.
template <int Dim>
[[nodiscard]] BandDerivative band_shape_derivative(const BandStructure<Dim>& problem,
                                                   const Bands<Dim>& bands,
                                                   const NodeField& velocity,
                                                   Real relative_step = 1e-6,
                                                   Real degeneracy_tolerance = 1e-6);

/// Directional derivative of the bands along the wave vector, @f$ d/dt\,k_{0,n}(k + t\,d) @f$
/// at t = 0 (linear in d; @f$ dk_0/dt @f$ is dimensionless for a d in 1/m, `angular_frequency`
/// is then the group velocity along d in m/s). Branch slopes through a degenerate point
/// (folded bands, high-symmetry points) are the cluster values.
/// @throws InvalidArgument as `band_permittivity_derivative` (without the tag) or for d = 0.
template <int Dim>
[[nodiscard]] BandDerivative band_wave_vector_derivative(const BandStructure<Dim>& problem,
                                                         const Bands<Dim>& bands,
                                                         const Point<Dim>& direction,
                                                         Real degeneracy_tolerance = 1e-6);

/// Group velocity @f$ v_g = \nabla_k\omega = c_0\,\nabla_k k_0 @f$ [m/s] of every band, one
/// `band_wave_vector_derivative` per Cartesian direction (Dim cheap differences of the
/// prolongation, no solve). Components of a band at @f$ k_0 = 0 @f$ are NaN; at a degenerate
/// point every component is the sorted cluster spectrum of its own direction.
/// @throws InvalidArgument as `band_wave_vector_derivative`.
template <int Dim>
[[nodiscard]] std::vector<Point<Dim>> group_velocity(const BandStructure<Dim>& problem,
                                                     const Bands<Dim>& bands,
                                                     Real degeneracy_tolerance = 1e-6);

extern template BandDerivative band_permittivity_derivative<2>(const BandStructure<2>&,
                                                               const Bands<2>&, mesh::Tag, Real);
extern template BandDerivative band_permittivity_derivative<3>(const BandStructure<3>&,
                                                               const Bands<3>&, mesh::Tag, Real);
extern template BandDerivative band_permeability_derivative<2>(const BandStructure<2>&,
                                                               const Bands<2>&, mesh::Tag, Real);
extern template BandDerivative band_permeability_derivative<3>(const BandStructure<3>&,
                                                               const Bands<3>&, mesh::Tag, Real);
extern template BandDerivative band_shape_derivative<2>(const BandStructure<2>&, const Bands<2>&,
                                                        const NodeField&, Real, Real);
extern template BandDerivative band_shape_derivative<3>(const BandStructure<3>&, const Bands<3>&,
                                                        const NodeField&, Real, Real);
extern template BandDerivative band_wave_vector_derivative<2>(const BandStructure<2>&,
                                                              const Bands<2>&, const Point<2>&,
                                                              Real);
extern template BandDerivative band_wave_vector_derivative<3>(const BandStructure<3>&,
                                                              const Bands<3>&, const Point<3>&,
                                                              Real);
extern template std::vector<Point<2>> group_velocity<2>(const BandStructure<2>&, const Bands<2>&,
                                                        Real);
extern template std::vector<Point<3>> group_velocity<3>(const BandStructure<3>&, const Bands<3>&,
                                                        Real);

}  // namespace hpfem::physics

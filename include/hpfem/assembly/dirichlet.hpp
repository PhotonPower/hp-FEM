#pragma once
/// @file dirichlet.hpp
/// Dirichlet boundary conditions on the hierarchical H1 and H(curl) spaces: boundary values
/// by hierarchical interpolation (H1: vertex values, then L2 projections onto the edge and
/// face functions; H(curl): L2 projections of the tangential trace onto the edge functions,
/// then the face functions) and symmetric elimination of the constrained DoFs from an
/// assembled system. See docs/theory/scalar-fem.md#dirichlet-conditions and
/// docs/theory/maxwell.md#boundary-conditions.

#include <span>
#include <type_traits>
#include <vector>

#include "hpfem/assembly/h1_forms.hpp"
#include "hpfem/assembly/interpolation.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/mesh.hpp"

namespace hpfem::assembly {

/// Constrained DoFs (sorted, unique) and their prescribed values.
using DirichletData = DofValues;

/// Boundary values of g on the given facets as coefficients of the trace space
/// (`interpolate` on the vertices, edges and faces of the facets): vertex DoFs take g(x_v);
/// the edge DoFs of every edge of these facets take the L2 projection (along the edge) of
/// the remainder onto the edge functions; in 3D the face DoFs take the projection of the
/// remainder onto the face functions. For g in the trace space the result is exact;
/// otherwise it is the hierarchical interpolant, accurate to order p + 1.
template <int Dim>
[[nodiscard]] DirichletData dirichlet_values(const fespace::DofMap<Dim>& dofs,
                                             std::span<const Index> facets,
                                             const std::type_identity_t<ScalarField<Dim>>& g);
/// Same for all facets carrying `tag`.
template <int Dim>
[[nodiscard]] DirichletData dirichlet_values(const fespace::DofMap<Dim>& dofs, mesh::Tag tag,
                                             const std::type_identity_t<ScalarField<Dim>>& g);
/// Prescribed tangential trace @f$ n\times E = n\times g @f$ on the given facets as
/// coefficients of the Nédélec trace space: for every edge of these facets the L2
/// projection (along the edge) of @f$ g\cdot t @f$ onto the edge functions; in 3D the face
/// DoFs take the projection of the remaining tangential part onto the face functions. For
/// @f$ g @f$ with a trace in the discrete trace space the result is exact (the DoF values
/// of the discrete field), otherwise the hierarchical interpolant, accurate to order p.
template <int Dim>
[[nodiscard]] DirichletData tangential_dirichlet_values(
    const fespace::NedelecDofMap<Dim>& dofs, std::span<const Index> facets,
    const std::type_identity_t<ComplexVectorField<Dim>>& g);
/// Same for all facets carrying `tag`.
template <int Dim>
[[nodiscard]] DirichletData tangential_dirichlet_values(
    const fespace::NedelecDofMap<Dim>& dofs, mesh::Tag tag,
    const std::type_identity_t<ComplexVectorField<Dim>>& g);
/// All DoFs on the facets, value 0: homogeneous Dirichlet for H1, PEC (vanishing
/// tangential trace) for the Nédélec space.
template <int Dim, class Counts>
[[nodiscard]] DirichletData homogeneous_dirichlet(const fespace::EntityDofMap<Dim, Counts>& dofs,
                                                  std::span<const Index> facets);

/// Union of several constraint sets; a DoF listed twice keeps its first value.
[[nodiscard]] DirichletData merge_dirichlet(std::span<const DirichletData> parts);

/// Eliminates the constrained DoFs symmetrically, keeping the system size:
/// @f$ b \leftarrow b - A_{:,c}\,g_c @f$, rows and columns c cleared, @f$ A_{cc} = 1 @f$,
/// @f$ b_c = g_c @f$. The solution of the modified system equals g on the constrained DoFs
/// and solves the reduced problem on the free ones. O(nnz).
void apply_dirichlet(SparseMatrix& matrix, Vector& rhs, const DirichletData& data);

extern template DirichletData dirichlet_values<2>(const fespace::DofMap<2>&, std::span<const Index>,
                                                  const ScalarField<2>&);
extern template DirichletData dirichlet_values<3>(const fespace::DofMap<3>&, std::span<const Index>,
                                                  const ScalarField<3>&);
extern template DirichletData dirichlet_values<2>(const fespace::DofMap<2>&, mesh::Tag,
                                                  const ScalarField<2>&);
extern template DirichletData dirichlet_values<3>(const fespace::DofMap<3>&, mesh::Tag,
                                                  const ScalarField<3>&);
extern template DirichletData tangential_dirichlet_values<2>(const fespace::NedelecDofMap<2>&,
                                                             std::span<const Index>,
                                                             const ComplexVectorField<2>&);
extern template DirichletData tangential_dirichlet_values<3>(const fespace::NedelecDofMap<3>&,
                                                             std::span<const Index>,
                                                             const ComplexVectorField<3>&);
extern template DirichletData tangential_dirichlet_values<2>(const fespace::NedelecDofMap<2>&,
                                                             mesh::Tag,
                                                             const ComplexVectorField<2>&);
extern template DirichletData tangential_dirichlet_values<3>(const fespace::NedelecDofMap<3>&,
                                                             mesh::Tag,
                                                             const ComplexVectorField<3>&);
extern template DirichletData homogeneous_dirichlet<2, fespace::H1Counts>(const fespace::DofMap<2>&,
                                                                          std::span<const Index>);
extern template DirichletData homogeneous_dirichlet<3, fespace::H1Counts>(const fespace::DofMap<3>&,
                                                                          std::span<const Index>);
extern template DirichletData homogeneous_dirichlet<2, fespace::NedelecCounts>(
    const fespace::NedelecDofMap<2>&, std::span<const Index>);
extern template DirichletData homogeneous_dirichlet<3, fespace::NedelecCounts>(
    const fespace::NedelecDofMap<3>&, std::span<const Index>);

}  // namespace hpfem::assembly

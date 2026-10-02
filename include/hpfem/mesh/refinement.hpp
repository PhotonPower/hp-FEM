#pragma once
/// @file refinement.hpp
/// Uniform (red / regular) refinement of simplicial meshes. See docs/theory/mesh.md#refinement.

#include <vector>

#include "hpfem/core/types.hpp"
#include "hpfem/mesh/mesh.hpp"

namespace hpfem::mesh {

/// Result of one uniform refinement step. Parent vertices keep their ids; the vertex
/// created on parent edge e has id `parent.num_vertices() + e`. Child cells are stored
/// parent by parent: children of parent cell c are `kChildren * c + i`.
template <int Dim>
struct Refined {
  static constexpr Index kChildren = Dim == 2 ? 4 : 8;  ///< children per cell
  Mesh<Dim> mesh;
  std::vector<Index> parent_cell;  ///< parent of every child cell
  std::vector<Index> edge_vertex;  ///< vertex created on every parent edge
};

/// Red refinement of every cell: a triangle into four congruent triangles through the
/// edge midpoints; a tetrahedron into four corner tetrahedra and four from the inner
/// octahedron, split along the diagonal between the midpoints of edges (0,2) and (1,3)
/// (Bey's rule, which keeps the number of congruence classes bounded under repetition).
///
/// Transferred to the children: cell tags, facet tags (to the child facets lying on a
/// tagged parent facet), tag names, and second-order geometry: new vertices are placed at
/// the parent edge nodes and child edge nodes are evaluated with the parent cell map, so
/// the refined mesh represents the same curved geometry.
/// O(N log N) in the number of cells.
template <int Dim>
[[nodiscard]] Refined<Dim> refine_uniform(const Mesh<Dim>& parent);

extern template Refined<2> refine_uniform<2>(const Mesh<2>&);
extern template Refined<3> refine_uniform<3>(const Mesh<3>&);

}  // namespace hpfem::mesh

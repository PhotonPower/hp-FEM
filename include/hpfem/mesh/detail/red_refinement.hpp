#pragma once
/// @file red_refinement.hpp
/// Child patterns of the red (regular) refinement of a simplex, shared by
/// `refine_uniform` and `AdaptiveMesh`. Local nodes of a refined cell: the parent vertices
/// 0..Dim, then the midpoint of parent edge k at Dim + 1 + k (`SimplexTopology<Dim>::kEdgeVertices`
/// order). See docs/theory/mesh.md#refinement.

#include <array>

#include "hpfem/core/types.hpp"
#include "hpfem/mesh/simplex_topology.hpp"

namespace hpfem::mesh::detail {

template <int Dim>
struct RedRefinement;

template <>
struct RedRefinement<2> {
  static constexpr Index kChildren = 4;
  // edges: 3 = (0,1), 4 = (1,2), 5 = (2,0)
  static constexpr std::array<std::array<LocalIndex, 3>, 4> kChildrenNodes{
      {{0, 3, 5}, {3, 1, 4}, {5, 4, 2}, {3, 4, 5}}};
};

template <>
struct RedRefinement<3> {
  static constexpr Index kChildren = 8;
  // edges: 4 = (0,1), 5 = (0,2), 6 = (0,3), 7 = (1,2), 8 = (1,3), 9 = (2,3)
  // Bey (1995): four corner tetrahedra, then the octahedron cut along the diagonal 5-8.
  static constexpr std::array<std::array<LocalIndex, 4>, 8> kChildrenNodes{{{0, 4, 5, 6},
                                                                            {4, 1, 7, 8},
                                                                            {5, 7, 2, 9},
                                                                            {6, 8, 9, 3},
                                                                            {4, 5, 6, 8},
                                                                            {4, 5, 7, 8},
                                                                            {5, 6, 8, 9},
                                                                            {5, 7, 8, 9}}};
};

/// Reference vertex i of the simplex (0 = origin).
template <int Dim>
[[nodiscard]] inline Point<Dim> reference_vertex(LocalIndex i) {
  Point<Dim> xi = Point<Dim>::Zero();
  if (i > 0) xi(i - 1) = 1.0;
  return xi;
}

/// Reference coordinates of local node n of the refined cell (vertex or edge midpoint).
template <int Dim>
[[nodiscard]] inline Point<Dim> reference_node(LocalIndex n) {
  if (n <= Dim) return reference_vertex<Dim>(n);
  const auto& e = SimplexTopology<Dim>::kEdgeVertices[as_size(n - Dim - 1)];
  return 0.5 * (reference_vertex<Dim>(e[0]) + reference_vertex<Dim>(e[1]));
}

/// Reference coordinates in the parent of the reference point ξ of child i (affine map
/// through the child's nodes).
template <int Dim>
[[nodiscard]] inline Point<Dim> parent_reference(LocalIndex child, const Point<Dim>& xi) {
  const auto& nodes = RedRefinement<Dim>::kChildrenNodes[as_size(child)];
  Point<Dim> x = reference_node<Dim>(nodes[0]);
  for (int j = 1; j <= Dim; ++j) {
    x += xi(j - 1) * (reference_node<Dim>(nodes[as_size(j)]) - reference_node<Dim>(nodes[0]));
  }
  return x;
}

}  // namespace hpfem::mesh::detail

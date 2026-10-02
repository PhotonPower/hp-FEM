#pragma once
/// @file simplex_topology.hpp
/// Local (reference-element) numbering of vertices, edges and faces of the simplex:
/// triangle (Dim = 2) and tetrahedron (Dim = 3).
///
/// These tables are **binding** for the whole code base (CLAUDE.md §6,
/// docs/theory/nedelec.md, docs/theory/mesh.md). `fespace::ReferenceElement<Dim>` re-uses
/// them; nothing may redefine them.
///
/// Reference triangle: v0 = (0,0), v1 = (1,0), v2 = (0,1); edges counter-clockwise
///   e0 = (v0,v1), e1 = (v1,v2), e2 = (v2,v0).
/// Reference tetrahedron: v0 = (0,0,0), v1 = (1,0,0), v2 = (0,1,0), v3 = (0,0,1); edges in
///   lexicographic order e0 = (v0,v1), e1 = (v0,v2), e2 = (v0,v3), e3 = (v1,v2),
///   e4 = (v1,v3), e5 = (v2,v3); face i is opposite vertex i, vertices in increasing order.
///
/// A *facet* is the codimension-1 entity: an edge in 2D, a face in 3D. Facet numbering
/// coincides with the edge numbering in 2D and with the face numbering in 3D.

#include <array>

#include "hpfem/core/types.hpp"

namespace hpfem::mesh {

/// Local topology tables of the reference simplex in `Dim` dimensions.
template <int Dim>
struct SimplexTopology;

/// Reference triangle.
template <>
struct SimplexTopology<2> {
  static constexpr int kDim = 2;
  static constexpr LocalIndex kNumVertices = 3;
  static constexpr LocalIndex kNumEdges = 3;
  static constexpr LocalIndex kNumFacets = 3;
  static constexpr LocalIndex kVerticesPerFacet = 2;

  /// Local vertices of local edge k, k = 0..2: (0,1), (1,2), (2,0).
  static constexpr std::array<std::array<LocalIndex, 2>, 3> kEdgeVertices{{{0, 1}, {1, 2}, {2, 0}}};
  /// Facets of a triangle are its edges, with the same numbering.
  static constexpr std::array<std::array<LocalIndex, 2>, 3> kFacetVertices = kEdgeVertices;
};

/// Reference tetrahedron.
template <>
struct SimplexTopology<3> {
  static constexpr int kDim = 3;
  static constexpr LocalIndex kNumVertices = 4;
  static constexpr LocalIndex kNumEdges = 6;
  static constexpr LocalIndex kNumFaces = 4;
  static constexpr LocalIndex kNumFacets = 4;
  static constexpr LocalIndex kVerticesPerFacet = 3;

  /// Local vertices of local edge k, k = 0..5, lexicographic order.
  static constexpr std::array<std::array<LocalIndex, 2>, 6> kEdgeVertices{
      {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}};
  /// Local vertices of local face i (opposite vertex i), increasing order.
  static constexpr std::array<std::array<LocalIndex, 3>, 4> kFaceVertices{
      {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}}};
  /// Facets of a tetrahedron are its faces, with the same numbering.
  static constexpr std::array<std::array<LocalIndex, 3>, 4> kFacetVertices = kFaceVertices;
  /// Local edges of local face i = (f0,f1,f2), in the order (f0,f1), (f1,f2), (f0,f2).
  static constexpr std::array<std::array<LocalIndex, 3>, 4> kFaceEdges{
      {{3, 5, 4}, {1, 5, 2}, {0, 4, 2}, {0, 3, 1}}};
};

/// The six permutations of three items in lexicographic order. `Mesh<3>` stores, per cell
/// and local face, a code k such that local face vertex j sits at position
/// `kFacePermutations[k][j]` of the globally sorted face (docs/theory/mesh.md).
/// Code 0 is the identity: local and global order agree.
inline constexpr std::array<std::array<LocalIndex, 3>, 6> kFacePermutations{
    {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}}};

}  // namespace hpfem::mesh

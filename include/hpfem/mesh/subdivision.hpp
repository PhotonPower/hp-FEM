#pragma once
/// @file subdivision.hpp
/// Uniform subdivision of every cell into @f$ n^{Dim} @f$ sub-simplices in reference space,
/// the piecewise-linear carrier for visualising high-order fields (`io::FieldExporter`).
/// See docs/theory/mesh.md#subdivision.

#include <vector>

#include "hpfem/core/types.hpp"
#include "hpfem/mesh/mesh.hpp"

namespace hpfem::mesh {

/// Result of `subdivide`. The sub-mesh duplicates its vertices per parent cell, so fields
/// that jump across facets (normal components of E, curls) stay discontinuous in the
/// output; cell tags are inherited from the parents. Sub-vertices of parent cell c are the
/// consecutive ids `c * vertices_per_cell(n) + k`, sub-cells likewise with
/// `cells_per_cell(n)`.
template <int Dim>
struct Subdivided {
  Mesh<Dim> mesh;
  std::vector<Index> parent_cell;     ///< parent of every sub-cell
  std::vector<Index> vertex_parent;   ///< parent cell of every sub-vertex
  std::vector<Point<Dim>> vertex_xi;  ///< reference coordinates of every sub-vertex in its parent

  /// Lattice points of the n-fold subdivided reference simplex: @f$ (n+1)(n+2)/2 @f$ in 2D,
  /// @f$ (n+1)(n+2)(n+3)/6 @f$ in 3D.
  [[nodiscard]] static constexpr Index vertices_per_cell(int n) noexcept {
    const Index m = n;
    return Dim == 2 ? (m + 1) * (m + 2) / 2 : (m + 1) * (m + 2) * (m + 3) / 6;
  }
  /// Sub-simplices per parent cell: @f$ n^{Dim} @f$.
  [[nodiscard]] static constexpr Index cells_per_cell(int n) noexcept {
    const Index m = n;
    return Dim == 2 ? m * m : m * m * m;
  }
};

/// Splits every cell into @f$ n^{Dim} @f$ sub-simplices of equal reference volume: the
/// lattice @f$ \xi = (i, j[, k]) / n @f$, @f$ i + j [+ k] \le n @f$, with upward and
/// downward triangles (2D) and, per lattice cube, one corner tetrahedron, an octahedron
/// split into four and an inverted tetrahedron (3D, Freudenthal–Bey). Sub-vertices are
/// placed with the parent cell map, so curved cells become piecewise straight. All
/// sub-cells have the orientation (sign of det J) of their parent. O(n^Dim · #cells).
/// @throws InvalidArgument if n < 1.
template <int Dim>
[[nodiscard]] Subdivided<Dim> subdivide(const Mesh<Dim>& mesh, int n);

extern template struct Subdivided<2>;
extern template struct Subdivided<3>;
extern template Subdivided<2> subdivide<2>(const Mesh<2>&, int);
extern template Subdivided<3> subdivide<3>(const Mesh<3>&, int);

}  // namespace hpfem::mesh

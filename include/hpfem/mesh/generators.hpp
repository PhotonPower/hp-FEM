#pragma once
/// @file generators.hpp
/// Structured simplicial meshes of axis-aligned boxes, for tests and simple studies.
/// Numbering and tagging conventions: docs/theory/mesh.md#generators.

#include "hpfem/core/types.hpp"
#include "hpfem/mesh/mesh.hpp"

namespace hpfem::mesh {

/// Facet tags that `rectangle()` and `box()` assign to the sides of the domain
/// (tag = 1 + 2·axis for the lower side, 2 + 2·axis for the upper side).
namespace box_tag {
inline constexpr Tag kXMin = 1;
inline constexpr Tag kXMax = 2;
inline constexpr Tag kYMin = 3;
inline constexpr Tag kYMax = 4;
inline constexpr Tag kZMin = 5;
inline constexpr Tag kZMax = 6;
}  // namespace box_tag

/// Rectangle [lower, upper] split into nx × ny squares, each into two triangles along the
/// diagonal from (i, j) to (i+1, j+1). Vertices are numbered x-fastest, cells square by
/// square. Counts: V = (nx+1)(ny+1), E = 3 nx ny + nx + ny, C = 2 nx ny.
/// With `tag_sides`, boundary edges carry the `box_tag` of their side; cells stay untagged.
/// @throws InvalidArgument for nx, ny < 1 or upper <= lower in any coordinate.
[[nodiscard]] Mesh<2> rectangle(Index nx, Index ny, const Point<2>& lower = Point<2>::Zero(),
                                const Point<2>& upper = Point<2>::Ones(), bool tag_sides = true);

/// Box [lower, upper] split into nx × ny × nz cubes, each into six tetrahedra (Kuhn /
/// Freudenthal subdivision, all sharing the cube's body diagonal). Vertices are numbered
/// x-fastest. Counts: V = (nx+1)(ny+1)(nz+1), C = 6 nx ny nz.
/// With `tag_sides`, boundary faces carry the `box_tag` of their side; cells stay untagged.
/// @throws InvalidArgument for nx, ny, nz < 1 or upper <= lower in any coordinate.
[[nodiscard]] Mesh<3> box(Index nx, Index ny, Index nz, const Point<3>& lower = Point<3>::Zero(),
                          const Point<3>& upper = Point<3>::Ones(), bool tag_sides = true);

/// Facet tag of the boundary of `disc` and `ball`.
inline constexpr Tag kDiscBoundary = 1;

/// Structured mesh of a disc: the square [-1, 1]^2 with n × n cells mapped by
/// @f$ (u, v) \mapsto (u\sqrt{1 - v^2/2},\; v\sqrt{1 - u^2/2}) @f$ onto the unit disc, then
/// scaled and shifted. With `curved` the nodes of the boundary edges are projected onto the
/// circle (`curve_boundary`, geometry order 2); otherwise the boundary stays polygonal. All
/// boundary facets carry `kDiscBoundary`. @throws InvalidArgument for n < 1 or radius ≤ 0.
[[nodiscard]] Mesh<2> disc(Index n, const Point<2>& center = Point<2>::Zero(), Real radius = 1.0,
                           bool curved = true);
/// Structured mesh of a ball: the cube [-1, 1]^3 with n^3 cubes mapped by
/// @f$ x = u\sqrt{1 - v^2/2 - w^2/2 + v^2w^2/3} @f$ (and cyclically) onto the unit ball,
/// scaled and shifted; boundary edge nodes projected onto the sphere with `curved`.
[[nodiscard]] Mesh<3> ball(Index n, const Point<3>& center = Point<3>::Zero(), Real radius = 1.0,
                           bool curved = true);

}  // namespace hpfem::mesh

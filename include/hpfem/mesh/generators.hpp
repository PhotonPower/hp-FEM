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

}  // namespace hpfem::mesh

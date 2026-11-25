#pragma once
/// @file generators.hpp
/// Structured simplicial meshes of axis-aligned boxes, for tests and simple studies.
/// Numbering and tagging conventions: docs/theory/mesh.md#generators.

#include <functional>
#include <span>

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
/// scaled and shifted. Every square is split along the diagonal pointing away from the centre
/// (unlike `rectangle`), so that no sliver lines the circle. With `curved` the nodes of the
/// boundary edges are projected onto the circle (`curve_boundary`, geometry order 2); otherwise
/// the boundary stays polygonal. All boundary facets carry `kDiscBoundary`.
/// @throws InvalidArgument for n < 1 or radius ≤ 0.
[[nodiscard]] Mesh<2> disc(Index n, const Point<2>& center = Point<2>::Zero(), Real radius = 1.0,
                           bool curved = true);
/// Structured mesh of a ball: the cube [-1, 1]^3 with n^3 cubes mapped by
/// @f$ x = u\sqrt{1 - v^2/2 - w^2/2 + v^2w^2/3} @f$ (and cyclically) onto the unit ball,
/// scaled and shifted; every cube split into the Kuhn tetrahedra around its body diagonal from
/// the corner nearest the centre (unlike `box`); boundary edge nodes projected onto the sphere
/// with `curved`.
[[nodiscard]] Mesh<3> ball(Index n, const Point<3>& center = Point<3>::Zero(), Real radius = 1.0,
                           bool curved = true);

/// Regular hexagon of circumradius `radius` around `center` (corners on the x axis at ±radius)
/// split into the equilateral triangles of a triangular lattice with n cells per side
/// (spacing radius / n): the mesh is invariant under the rotations by multiples of 60° and the
/// reflections of the hexagon, so it serves as the full domain of C_6, C_3 and C_2 symmetric
/// problems (issue #244). Cells within the concentric hexagon of `inner` rings (circumradius
/// inner·radius/n) carry `inner_tag`, the others are untagged; all boundary facets carry
/// `kDiscBoundary`. Counts: V = 3n(n+1) + 1, C = 6n². @throws InvalidArgument for n < 1,
/// radius ≤ 0, inner outside [0, n] or `inner_tag == kNoTag` with inner > 0.
[[nodiscard]] Mesh<2> hexagon(Index n, Real radius = 1.0, Index inner = 0, Tag inner_tag = 2,
                              const Point<2>& center = Point<2>::Zero());

/// Disc of concentric annuli around `center` meshed by rings (issue #244 S4): the circles of
/// `radii` (increasing, the last is the outer boundary) are mesh lines, the annulus k between
/// radii[k-1] and radii[k] (k = 0: the disc of radii[0]) carries the cell tag tags[k] (empty:
/// all untagged). The triangulation is that of the regular `sides`-gon split into `sides`
/// triangles around the centre, each refined uniformly, with ring j (j·sides vertices at
/// uniform angles, the first on the positive x axis) moved onto the circle of radius
/// @f$ \rho_j @f$: n rings inside radii[0], and max(1, round(n (r_k − r_{k−1}) / r_0)) rings in
/// annulus k, equally spaced, so the cells stay about r_0 / n in size. The mesh is invariant
/// under the rotations by multiples of 2π / sides and the reflections about the rays through
/// the polygon's corners and edge midpoints, so it is the full domain of C_n symmetric problems
/// for every divisor n of `sides`, and the circles carry a circular `pml::RadialPml` or a
/// material interface exactly: the edges on every circle of `radii` are curved onto it
/// (geometry order 2). Boundary facets carry `kDiscBoundary`.
/// @throws InvalidArgument for sides < 3, n < 1, empty or non-increasing positive radii, or a
/// tag list of the wrong length.
[[nodiscard]] Mesh<2> polar_disc(Index sides, Index n, const std::vector<Real>& radii,
                                 const std::vector<Tag>& tags = {},
                                 const Point<2>& center = Point<2>::Zero());

/// Structured mesh of the square [-outer, outer]^2 with a circular inclusion of the given
/// radius at the origin whose interface is resolved by (curved) cell edges: the grid of the
/// inner square is mapped onto the disc as in `disc`, the ring up to `half_width` blends
/// linearly between the circle and the square (grid coordinate s = max(|u|, |v|) from 1 to
/// half_width / radius: @f$ x = R[(1 - t)\,s\,\mathrm{disc}(q/s) + t\,q] @f$,
/// @f$ t = (s - 1)/(l - 1) @f$, so the rings keep the cell size), and beyond `half_width` the
/// grid continues uniformly (space for a PML). Squares are split along the diagonal pointing
/// away from the centre, as in `disc`. n is the number of cells per radius; the cell size is about
/// radius / n everywhere, so (outer / radius) n must be an integer. The inclusion cells carry
/// `inclusion_tag`, the outer sides the `box_tag`s. @throws InvalidArgument for inconsistent
/// sizes.
[[nodiscard]] Mesh<2> square_with_disc(Index n, Real radius, Real half_width, Real outer,
                                       Tag inclusion_tag = 2);

/// 3D counterpart of `square_with_disc`: the cube [-outer, outer]^3 with a spherical inclusion
/// of the given radius at the origin resolved by (curved) cell faces. The grid of the inner cube
/// is mapped onto the ball as in `ball`, the shell up to `half_width` blends linearly between
/// the sphere and the cube (as in `square_with_disc`), beyond `half_width` the grid is uniform
/// (space for a PML). Cubes are split as in `ball`. n is the number of
/// cells per radius; (outer / radius) n must be an integer. Cells: 6 (2 (outer / radius) n)^3
/// Kuhn tetrahedra. The inclusion cells carry `inclusion_tag`, the outer sides the `box_tag`s,
/// the interface faces become quadratic (geometry order 2). @throws InvalidArgument for
/// inconsistent sizes.
[[nodiscard]] Mesh<3> box_with_ball(Index n, Real radius, Real half_width, Real outer,
                                    Tag inclusion_tag = 2);

/// Sub-mesh of the given cells (in the given order) of a conforming mesh: vertices are
/// renumbered ascending, cell tags, the facet tags of facets kept, tag names and curved edge
/// nodes transfer. New boundary facets (cut through the interior) stay untagged, e.g. for
/// an L-shaped domain cut from a `rectangle`. @throws InvalidArgument for a cell out of
/// range or a cell listed twice.
template <int Dim>
[[nodiscard]] Mesh<Dim> extract(const Mesh<Dim>& mesh, std::span<const Index> cells);
/// Sub-mesh of the cells for which `keep(centroid)` is true.
template <int Dim>
[[nodiscard]] Mesh<Dim> extract(const Mesh<Dim>& mesh,
                                const std::function<bool(const Point<Dim>&)>& keep);

extern template Mesh<2> extract<2>(const Mesh<2>&, std::span<const Index>);
extern template Mesh<3> extract<3>(const Mesh<3>&, std::span<const Index>);
extern template Mesh<2> extract<2>(const Mesh<2>&, const std::function<bool(const Point<2>&)>&);
extern template Mesh<3> extract<3>(const Mesh<3>&, const std::function<bool(const Point<3>&)>&);

}  // namespace hpfem::mesh

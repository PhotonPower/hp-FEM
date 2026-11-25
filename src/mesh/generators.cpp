#include "hpfem/mesh/generators.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numbers>
#include <span>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "hpfem/core/error.hpp"
#include "hpfem/mesh/geometry.hpp"

namespace hpfem::mesh {

namespace {

/// Coordinate of grid line i of n on [lo, hi]; the end points are reproduced exactly so that
/// side detection can compare with ==.
Real grid_coordinate(Index i, Index n, Real lo, Real hi) {
  if (i == 0) return lo;
  if (i == n) return hi;
  return lo + (hi - lo) * static_cast<Real>(i) / static_cast<Real>(n);
}

template <int Dim>
void check_box(const char* name, const std::array<Index, static_cast<std::size_t>(Dim)>& n,
               const Point<Dim>& lower, const Point<Dim>& upper) {
  for (int d = 0; d < Dim; ++d) {
    if (n[static_cast<std::size_t>(d)] < 1) {
      throw InvalidArgument(fmt::format("{}: need at least one cell per axis, got {} on axis {}",
                                        name, n[static_cast<std::size_t>(d)], d));
    }
    if (!(upper(d) > lower(d))) {
      throw InvalidArgument(fmt::format("{}: upper({}) = {} must exceed lower({}) = {}", name, d,
                                        upper(d), d, lower(d)));
    }
  }
}

/// Triangles of an nx × ny grid (vertices x-fastest) centred on the origin, every square split
/// along the diagonal that points away from the centre: (i, j)–(i+1, j+1) in the first and
/// third quadrant, (i+1, j)–(i, j+1) in the second and fourth. Under the maps of `disc` and
/// `square_with_disc`, which compress the grid towards the diagonals of the square, the other
/// choice leaves slivers along the circle in two quadrants whose curved edges fold the cell
/// (issue #234).
std::vector<Mesh<2>::CellVertices> radially_split_cells(Index nx, Index ny) {
  const auto id = [nx](Index i, Index j) { return j * (nx + 1) + i; };
  std::vector<Mesh<2>::CellVertices> cells;
  cells.reserve(static_cast<std::size_t>(2 * nx * ny));
  for (Index j = 0; j < ny; ++j) {
    for (Index i = 0; i < nx; ++i) {
      const Index cx = 2 * i + 1 - nx;  // twice the square centre in grid units
      const Index cy = 2 * j + 1 - ny;
      if (cx * cy >= 0) {
        cells.push_back({id(i, j), id(i + 1, j), id(i + 1, j + 1)});
        cells.push_back({id(i, j), id(i + 1, j + 1), id(i, j + 1)});
      } else {
        cells.push_back({id(i, j), id(i + 1, j), id(i, j + 1)});
        cells.push_back({id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)});
      }
    }
  }
  return cells;
}

/// Tetrahedra of an nx × ny × nz grid (vertices x-fastest) centred on the origin: every cube
/// split into the six Kuhn tetrahedra around the body diagonal from its corner nearest the
/// centre to the opposite one, the 3D counterpart of `radially_split_cells` (with one diagonal
/// for all cubes, `ball` and `box_with_ball` get slivers along the sphere in the octants the
/// diagonal crosses tangentially).
std::vector<Mesh<3>::CellVertices> radially_split_tets(Index nx, Index ny, Index nz) {
  const auto id = [nx, ny](Index i, Index j, Index k) { return (k * (ny + 1) + j) * (nx + 1) + i; };
  std::vector<Mesh<3>::CellVertices> cells;
  cells.reserve(static_cast<std::size_t>(6 * nx * ny * nz));
  for (Index k = 0; k < nz; ++k) {
    for (Index j = 0; j < ny; ++j) {
      for (Index i = 0; i < nx; ++i) {
        // start corner: offset 1 along the axes on which the cube centre is negative
        const std::array<Index, 3> start{2 * i + 1 - nx < 0 ? 1 : 0, 2 * j + 1 - ny < 0 ? 1 : 0,
                                         2 * k + 1 - nz < 0 ? 1 : 0};
        std::array<std::size_t, 3> axes{0, 1, 2};
        do {
          std::array<Index, 3> off = start;
          Mesh<3>::CellVertices tet{};
          tet[0] = id(i + off[0], j + off[1], k + off[2]);
          for (std::size_t s = 0; s < 3; ++s) {
            off[axes[s]] = 1 - start[axes[s]];
            tet[s + 1] = id(i + off[0], j + off[1], k + off[2]);
          }
          cells.push_back(tet);
        } while (std::next_permutation(axes.begin(), axes.end()));
      }
    }
  }
  return cells;
}

/// Tags every boundary facet with the box side all its vertices lie on.
template <int Dim>
void tag_box_sides(Mesh<Dim>& m, const Point<Dim>& lower, const Point<Dim>& upper) {
  for (const Index f : m.boundary_facets()) {
    const auto& fv = m.facet_vertices(f);
    for (int d = 0; d < Dim; ++d) {
      const auto on = [&](Real value) {
        return std::all_of(fv.begin(), fv.end(), [&](Index v) { return m.vertex(v)(d) == value; });
      };
      if (on(lower(d))) {
        m.set_facet_tag(f, static_cast<Tag>(1 + 2 * d));
      } else if (on(upper(d))) {
        m.set_facet_tag(f, static_cast<Tag>(2 + 2 * d));
      }
    }
  }
}

}  // namespace

Mesh<2> rectangle(Index nx, Index ny, const Point<2>& lower, const Point<2>& upper,
                  bool tag_sides) {
  check_box<2>("rectangle", {nx, ny}, lower, upper);

  std::vector<Point<2>> vertices;
  vertices.reserve(static_cast<std::size_t>((nx + 1) * (ny + 1)));
  for (Index j = 0; j <= ny; ++j) {
    for (Index i = 0; i <= nx; ++i) {
      vertices.emplace_back(grid_coordinate(i, nx, lower(0), upper(0)),
                            grid_coordinate(j, ny, lower(1), upper(1)));
    }
  }
  const auto id = [nx](Index i, Index j) { return j * (nx + 1) + i; };
  std::vector<Mesh<2>::CellVertices> cells;
  cells.reserve(static_cast<std::size_t>(2 * nx * ny));
  for (Index j = 0; j < ny; ++j) {
    for (Index i = 0; i < nx; ++i) {
      cells.push_back({id(i, j), id(i + 1, j), id(i + 1, j + 1)});
      cells.push_back({id(i, j), id(i + 1, j + 1), id(i, j + 1)});
    }
  }
  Mesh<2> m(std::move(vertices), std::move(cells));
  if (tag_sides) tag_box_sides(m, lower, upper);
  return m;
}

Mesh<3> box(Index nx, Index ny, Index nz, const Point<3>& lower, const Point<3>& upper,
            bool tag_sides) {
  check_box<3>("box", {nx, ny, nz}, lower, upper);

  std::vector<Point<3>> vertices;
  vertices.reserve(static_cast<std::size_t>((nx + 1) * (ny + 1) * (nz + 1)));
  for (Index k = 0; k <= nz; ++k) {
    for (Index j = 0; j <= ny; ++j) {
      for (Index i = 0; i <= nx; ++i) {
        vertices.emplace_back(grid_coordinate(i, nx, lower(0), upper(0)),
                              grid_coordinate(j, ny, lower(1), upper(1)),
                              grid_coordinate(k, nz, lower(2), upper(2)));
      }
    }
  }
  const auto id = [nx, ny](Index i, Index j, Index k) { return (k * (ny + 1) + j) * (nx + 1) + i; };
  std::vector<Mesh<3>::CellVertices> cells;
  cells.reserve(static_cast<std::size_t>(6 * nx * ny * nz));
  for (Index k = 0; k < nz; ++k) {
    for (Index j = 0; j < ny; ++j) {
      for (Index i = 0; i < nx; ++i) {
        // Kuhn: one tetrahedron per monotone path from (i,j,k) to (i+1,j+1,k+1), i.e. per
        // permutation of the three axes; all six share the body diagonal.
        std::array<std::size_t, 3> axes{0, 1, 2};
        do {
          std::array<Index, 3> off{0, 0, 0};
          Mesh<3>::CellVertices tet{};
          tet[0] = id(i, j, k);
          for (std::size_t s = 0; s < 3; ++s) {
            off[axes[s]] = 1;
            tet[s + 1] = id(i + off[0], j + off[1], k + off[2]);
          }
          cells.push_back(tet);
        } while (std::next_permutation(axes.begin(), axes.end()));
      }
    }
  }
  Mesh<3> m(std::move(vertices), std::move(cells));
  if (tag_sides) tag_box_sides(m, lower, upper);
  return m;
}

}  // namespace hpfem::mesh

namespace hpfem::mesh {

Mesh<2> disc(Index n, const Point<2>& center, Real radius, bool curved) {
  if (n < 1) throw InvalidArgument(fmt::format("disc: need at least one cell per axis, got {}", n));
  if (!(radius > 0)) throw InvalidArgument(fmt::format("disc: radius {} must be positive", radius));
  const Mesh<2> square = rectangle(n, n, Point<2>(-1.0, -1.0), Point<2>(1.0, 1.0), false);
  std::vector<Point<2>> vertices;
  vertices.reserve(as_size(square.num_vertices()));
  for (const auto& q : square.vertices()) {
    const Real u = q(0);
    const Real v = q(1);
    vertices.push_back(center + radius * Point<2>(u * std::sqrt(1.0 - 0.5 * v * v),
                                                  v * std::sqrt(1.0 - 0.5 * u * u)));
  }
  Mesh<2> m(std::move(vertices), radially_split_cells(n, n));
  m.tag_boundary(kDiscBoundary);
  if (curved) {
    curve_boundary<2>(m, kDiscBoundary, [center, radius](const Point<2>& x) {
      return Point<2>(center + radius * (x - center).normalized());
    });
  }
  return m;
}

Mesh<2> hexagon(Index n, Real radius, Index inner, Tag inner_tag, const Point<2>& center) {
  if (n < 1 || !(radius > 0) || inner < 0 || inner > n || (inner > 0 && inner_tag == kNoTag)) {
    throw InvalidArgument(fmt::format(
        "hexagon: need n >= 1, radius > 0, 0 <= inner <= n and a tag for the inner cells (got n "
        "= {}, radius = {}, inner = {}, inner_tag = {})",
        n, radius, inner, inner_tag));
  }
  const Real h = radius / static_cast<Real>(n);
  const Point<2> a1(h, 0.0);
  const Point<2> a2(0.5 * h, 0.5 * std::numbers::sqrt3 * h);
  // lattice point (i, j) = i a1 + j a2 lies in the hexagon iff max(|i|, |j|, |i + j|) <= n
  const auto ring = [](Index i, Index j) {
    return std::max({std::abs(i), std::abs(j), std::abs(i + j)});
  };
  const Index width = 2 * n + 1;
  std::vector<Index> id(as_size(width * width), kInvalidIndex);
  const auto slot = [&](Index i, Index j) -> Index& {
    return id[as_size((i + n) * width + j + n)];
  };
  std::vector<Point<2>> vertices;
  for (Index j = -n; j <= n; ++j) {
    for (Index i = -n; i <= n; ++i) {
      if (ring(i, j) > n) continue;
      slot(i, j) = static_cast<Index>(vertices.size());
      vertices.emplace_back(center + static_cast<Real>(i) * a1 + static_cast<Real>(j) * a2);
    }
  }
  std::vector<Mesh<2>::CellVertices> cells;
  std::vector<Tag> tags;
  const auto add = [&](std::array<std::array<Index, 2>, 3> corners) {
    Index sum_i = 0;
    Index sum_j = 0;
    Mesh<2>::CellVertices c{};
    for (std::size_t k = 0; k < 3; ++k) {
      const auto [i, j] = corners[k];
      if (ring(i, j) > n) return;
      c[k] = slot(i, j);
      sum_i += i;
      sum_j += j;
    }
    cells.push_back(c);
    // the centroid (sum_i / 3, sum_j / 3) lies inside the hexagon of `inner` rings iff
    // max(|sum_i|, |sum_j|, |sum_i + sum_j|) < 3 inner (it never lies on a lattice line)
    tags.push_back(ring(sum_i, sum_j) < 3 * inner ? inner_tag : kNoTag);
  };
  for (Index j = -n; j < n; ++j) {
    for (Index i = -n; i < n; ++i) {
      add({{{i, j}, {i + 1, j}, {i, j + 1}}});          // pointing up
      add({{{i + 1, j}, {i + 1, j + 1}, {i, j + 1}}});  // pointing down
    }
  }
  Mesh<2> m(std::move(vertices), std::move(cells), std::move(tags));
  m.tag_boundary(kDiscBoundary);
  return m;
}

Mesh<3> ball(Index n, const Point<3>& center, Real radius, bool curved) {
  if (n < 1) throw InvalidArgument(fmt::format("ball: need at least one cell per axis, got {}", n));
  if (!(radius > 0)) throw InvalidArgument(fmt::format("ball: radius {} must be positive", radius));
  const Mesh<3> cube = box(n, n, n, Point<3>(-1.0, -1.0, -1.0), Point<3>(1.0, 1.0, 1.0), false);
  std::vector<Point<3>> vertices;
  vertices.reserve(as_size(cube.num_vertices()));
  for (const auto& q : cube.vertices()) {
    const Real u = q(0);
    const Real v = q(1);
    const Real w = q(2);
    vertices.push_back(
        center +
        radius * Point<3>(u * std::sqrt(1.0 - 0.5 * v * v - 0.5 * w * w + v * v * w * w / 3.0),
                          v * std::sqrt(1.0 - 0.5 * w * w - 0.5 * u * u + w * w * u * u / 3.0),
                          w * std::sqrt(1.0 - 0.5 * u * u - 0.5 * v * v + u * u * v * v / 3.0)));
  }
  Mesh<3> m(std::move(vertices), radially_split_tets(n, n, n));
  m.tag_boundary(kDiscBoundary);
  if (curved) {
    curve_boundary<3>(m, kDiscBoundary, [center, radius](const Point<3>& x) {
      return Point<3>(center + radius * (x - center).normalized());
    });
  }
  return m;
}

}  // namespace hpfem::mesh

namespace hpfem::mesh {

Mesh<2> polar_disc(Index sides, Index n, const std::vector<Real>& radii,
                   const std::vector<Tag>& tags, const Point<2>& center) {
  if (sides < 3 || n < 1 || radii.empty() || !(radii[0] > 0) ||
      (!tags.empty() && tags.size() != radii.size())) {
    throw InvalidArgument(fmt::format(
        "polar_disc: need sides >= 3, n >= 1, positive strictly increasing radii and one tag per "
        "annulus or none (got sides = {}, n = {}, {} radii, {} tags)",
        sides, n, radii.size(), tags.size()));
  }
  for (std::size_t k = 1; k < radii.size(); ++k) {
    if (!(radii[k] > radii[k - 1])) {
      throw InvalidArgument("polar_disc: the radii must increase strictly");
    }
  }
  // ring radii rho_j, the annulus of each ring band and the ring index of each circle
  std::vector<Real> rho{0.0};
  std::vector<std::size_t> band;  // band[j - 1]: annulus of the cells between rings j-1 and j
  std::vector<Index> circle_ring;
  for (std::size_t k = 0; k < radii.size(); ++k) {
    const Real lower = k == 0 ? 0.0 : radii[k - 1];
    const Index rings =
        k == 0
            ? n
            : std::max<Index>(1, std::lround(static_cast<Real>(n) * (radii[k] - lower) / radii[0]));
    for (Index i = 1; i <= rings; ++i) {
      rho.push_back(lower + (radii[k] - lower) * static_cast<Real>(i) / static_cast<Real>(rings));
      band.push_back(k);
    }
    circle_ring.push_back(static_cast<Index>(rho.size()) - 1);
  }
  const Index num_rings = static_cast<Index>(rho.size()) - 1;
  // vertex (j, i), i = 0 .. sides j - 1 counter-clockwise from the x axis; the centre is (0, 0)
  const auto first = [sides](Index j) { return j == 0 ? Index{0} : 1 + sides * j * (j - 1) / 2; };
  const auto id = [&](Index j, Index i) {
    return j == 0 ? Index{0} : first(j) + ((i % (sides * j)) + sides * j) % (sides * j);
  };
  std::vector<Point<2>> vertices{center};
  for (Index j = 1; j <= num_rings; ++j) {
    for (Index i = 0; i < sides * j; ++i) {
      const Real angle =
          2.0 * std::numbers::pi * static_cast<Real>(i) / static_cast<Real>(sides * j);
      vertices.emplace_back(center + rho[as_size(j)] * Point<2>(std::cos(angle), std::sin(angle)));
    }
  }
  std::vector<Mesh<2>::CellVertices> cells;
  std::vector<Tag> cell_tags;
  const auto add = [&](Index a, Index b, Index c, std::size_t k) {
    cells.push_back({a, b, c});
    cell_tags.push_back(tags.empty() ? kNoTag : tags[k]);
  };
  // sector s between rings j-1 and j: the strip of the uniformly refined corner triangle
  for (Index j = 1; j <= num_rings; ++j) {
    const std::size_t k = band[as_size(j - 1)];
    for (Index s = 0; s < sides; ++s) {
      for (Index t = 0; t < j; ++t) {
        add(id(j - 1, s * (j - 1) + t), id(j, s * j + t), id(j, s * j + t + 1), k);
        if (t + 1 < j) {
          add(id(j - 1, s * (j - 1) + t), id(j, s * j + t + 1), id(j - 1, s * (j - 1) + t + 1), k);
        }
      }
    }
  }
  Mesh<2> m(std::move(vertices), std::move(cells), std::move(cell_tags));
  m.tag_boundary(kDiscBoundary);
  // curve the edges on every circle of `radii` onto it
  for (std::size_t k = 0; k < radii.size(); ++k) {
    const Index ring = circle_ring[k];
    std::vector<Index> on_circle;
    for (Index f = 0; f < m.num_facets(); ++f) {
      const auto& fv = m.facet_vertices(f);
      const auto on = [&](Index v) { return v >= first(ring) && v < first(ring) + sides * ring; };
      if (on(fv[0]) && on(fv[1])) on_circle.push_back(f);
    }
    const Real radius = radii[k];
    curve_boundary<2>(m, std::span<const Index>(on_circle), [center, radius](const Point<2>& x) {
      return Point<2>(center + radius * (x - center).normalized());
    });
  }
  return m;
}

Mesh<2> square_with_disc(Index n, Real radius, Real half_width, Real outer, Tag inclusion_tag) {
  if (n < 1 || !(radius > 0) || !(half_width > radius) || !(outer >= half_width)) {
    throw InvalidArgument(fmt::format(
        "square_with_disc: need n >= 1, 0 < radius < half_width <= outer (got n = {}, radius = "
        "{}, half_width = {}, outer = {})",
        n, radius, half_width, outer));
  }
  if (inclusion_tag == kNoTag)
    throw InvalidArgument("square_with_disc: inclusion tag must be non-zero");
  const Real l = half_width / radius;  // grid coordinate of the square
  const Real l_out = outer / radius;   // grid coordinate of the outer boundary
  const Real cells_real = static_cast<Real>(n) * l_out;
  const Index cells = static_cast<Index>(std::lround(cells_real));
  if (std::abs(cells_real - static_cast<Real>(cells)) > 1e-9) {
    throw InvalidArgument(
        fmt::format("square_with_disc: (outer / radius) n = {} must be an integer number of cells",
                    cells_real));
  }
  const Mesh<2> grid =
      rectangle(2 * cells, 2 * cells, Point<2>(-l_out, -l_out), Point<2>(l_out, l_out), false);
  const auto ellipse = [](const Point<2>& q) {
    return Point<2>(q(0) * std::sqrt(1.0 - 0.5 * q(1) * q(1)),
                    q(1) * std::sqrt(1.0 - 0.5 * q(0) * q(0)));
  };
  std::vector<Point<2>> vertices;
  vertices.reserve(as_size(grid.num_vertices()));
  for (const auto& q : grid.vertices()) {
    const Real s = std::max(std::abs(q(0)), std::abs(q(1)));
    Point<2> x;
    if (s <= 1.0 + 1e-12) {
      x = radius * ellipse(q);
    } else if (s <= l + 1e-12) {
      const Real t = (s - 1.0) / (l - 1.0);
      x = (1.0 - t) * s * radius * ellipse(q / s) + t * radius * q;
    } else {
      x = radius * q;
    }
    vertices.push_back(x);
  }
  std::vector<Mesh<2>::CellVertices> cells_list = radially_split_cells(2 * cells, 2 * cells);
  std::vector<Tag> tags;
  tags.reserve(cells_list.size());
  for (const auto& cv : cells_list) {
    bool inside = true;
    for (const Index v : cv) {
      const auto& q = grid.vertex(v);
      if (std::max(std::abs(q(0)), std::abs(q(1))) > 1.0 + 1e-12) inside = false;
    }
    tags.push_back(inside ? inclusion_tag : kNoTag);
  }
  Mesh<2> m(std::move(vertices), std::move(cells_list), std::move(tags));
  tag_box_sides(m, Point<2>(-outer, -outer), Point<2>(outer, outer));
  // curve the interface: facets between inclusion cells and the rest
  std::vector<Index> interface;
  for (Index f = 0; f < m.num_facets(); ++f) {
    const auto& fc = m.facet_cells(f);
    if (fc[1] == kInvalidIndex) continue;
    if ((m.cell_tag(fc[0]) == inclusion_tag) != (m.cell_tag(fc[1]) == inclusion_tag))
      interface.push_back(f);
  }
  curve_boundary<2>(m, std::span<const Index>(interface),
                    [radius](const Point<2>& x) { return Point<2>(radius * x.normalized()); });
  return m;
}

Mesh<3> box_with_ball(Index n, Real radius, Real half_width, Real outer, Tag inclusion_tag) {
  if (n < 1 || !(radius > 0) || !(half_width > radius) || !(outer >= half_width)) {
    throw InvalidArgument(fmt::format(
        "box_with_ball: need n >= 1, 0 < radius < half_width <= outer (got n = {}, radius = {}, "
        "half_width = {}, outer = {})",
        n, radius, half_width, outer));
  }
  if (inclusion_tag == kNoTag)
    throw InvalidArgument("box_with_ball: inclusion tag must be non-zero");
  const Real l = half_width / radius;  // grid coordinate of the inner cube
  const Real l_out = outer / radius;   // grid coordinate of the outer boundary
  const Real cells_real = static_cast<Real>(n) * l_out;
  const Index cells = static_cast<Index>(std::lround(cells_real));
  if (std::abs(cells_real - static_cast<Real>(cells)) > 1e-9) {
    throw InvalidArgument(fmt::format(
        "box_with_ball: (outer / radius) n = {} must be an integer number of cells", cells_real));
  }
  const Mesh<3> grid = box(2 * cells, 2 * cells, 2 * cells, Point<3>(-l_out, -l_out, -l_out),
                           Point<3>(l_out, l_out, l_out), false);
  // the cube [-1, 1]^3 onto the unit ball (same map as `ball`)
  const auto to_ball = [](const Point<3>& q) {
    const Real u = q(0);
    const Real v = q(1);
    const Real w = q(2);
    return Point<3>(u * std::sqrt(1.0 - 0.5 * v * v - 0.5 * w * w + v * v * w * w / 3.0),
                    v * std::sqrt(1.0 - 0.5 * w * w - 0.5 * u * u + w * w * u * u / 3.0),
                    w * std::sqrt(1.0 - 0.5 * u * u - 0.5 * v * v + u * u * v * v / 3.0));
  };
  const auto sup = [](const Point<3>& q) {
    return std::max({std::abs(q(0)), std::abs(q(1)), std::abs(q(2))});
  };
  std::vector<Point<3>> vertices;
  vertices.reserve(as_size(grid.num_vertices()));
  for (const auto& q : grid.vertices()) {
    const Real s = sup(q);
    Point<3> x;
    if (s <= 1.0 + 1e-12) {
      x = radius * to_ball(q);
    } else if (s <= l + 1e-12) {
      const Real t = (s - 1.0) / (l - 1.0);
      x = (1.0 - t) * s * radius * to_ball(q / s) + t * radius * q;
    } else {
      x = radius * q;
    }
    vertices.push_back(x);
  }
  std::vector<Mesh<3>::CellVertices> cells_list =
      radially_split_tets(2 * cells, 2 * cells, 2 * cells);
  std::vector<Tag> tags;
  tags.reserve(cells_list.size());
  for (const auto& cv : cells_list) {
    bool inside = true;
    for (const Index v : cv) {
      if (sup(grid.vertex(v)) > 1.0 + 1e-12) inside = false;
    }
    tags.push_back(inside ? inclusion_tag : kNoTag);
  }
  Mesh<3> m(std::move(vertices), std::move(cells_list), std::move(tags));
  tag_box_sides(m, Point<3>(-outer, -outer, -outer), Point<3>(outer, outer, outer));
  std::vector<Index> interface;
  for (Index f = 0; f < m.num_facets(); ++f) {
    const auto& fc = m.facet_cells(f);
    if (fc[1] == kInvalidIndex) continue;
    if ((m.cell_tag(fc[0]) == inclusion_tag) != (m.cell_tag(fc[1]) == inclusion_tag))
      interface.push_back(f);
  }
  curve_boundary<3>(m, std::span<const Index>(interface),
                    [radius](const Point<3>& x) { return Point<3>(radius * x.normalized()); });
  return m;
}

template <int Dim>
Mesh<Dim> extract(const Mesh<Dim>& mesh, std::span<const Index> cells) {
  std::vector<Index> new_vertex(as_size(mesh.num_vertices()), kInvalidIndex);
  std::vector<char> seen(as_size(mesh.num_cells()), 0);
  for (const Index c : cells) {
    if (c < 0 || c >= mesh.num_cells()) {
      throw InvalidArgument(fmt::format("extract: cell {} outside 0..{}", c, mesh.num_cells() - 1));
    }
    if (seen[as_size(c)] != 0)
      throw InvalidArgument(fmt::format("extract: cell {} listed twice", c));
    seen[as_size(c)] = 1;
    for (const Index v : mesh.cell_vertices(c)) new_vertex[as_size(v)] = 0;
  }
  std::vector<Point<Dim>> vertices;
  std::vector<Index> old_vertex;
  for (Index v = 0; v < mesh.num_vertices(); ++v) {
    if (new_vertex[as_size(v)] == kInvalidIndex) continue;
    new_vertex[as_size(v)] = static_cast<Index>(vertices.size());
    vertices.push_back(mesh.vertex(v));
    old_vertex.push_back(v);
  }
  std::vector<typename Mesh<Dim>::CellVertices> new_cells;
  std::vector<Tag> tags;
  new_cells.reserve(cells.size());
  for (const Index c : cells) {
    typename Mesh<Dim>::CellVertices cv = mesh.cell_vertices(c);
    for (Index& v : cv) v = new_vertex[as_size(v)];
    new_cells.push_back(cv);
    tags.push_back(mesh.cell_tag(c));
  }
  Mesh<Dim> out(std::move(vertices), std::move(new_cells), std::move(tags));
  for (Index f = 0; f < out.num_facets(); ++f) {
    typename Mesh<Dim>::FacetVertices fv = out.facet_vertices(f);
    for (Index& v : fv) v = old_vertex[as_size(v)];
    const Index old = mesh.facet_id(fv);
    if (old != kInvalidIndex) out.set_facet_tag(f, mesh.facet_tag(old));
  }
  for (const int dim : {Dim - 1, Dim}) {
    for (const auto& [tag, name] : mesh.tag_names(dim)) out.set_tag_name(dim, tag, name);
  }
  if (mesh.geometry_order() == 2) {
    std::vector<Point<Dim>> nodes(as_size(out.num_edges()));
    for (Index e = 0; e < out.num_edges(); ++e) {
      const auto& ev = out.edge_vertices(e);
      nodes[as_size(e)] =
          mesh.edge_node(mesh.edge_id(old_vertex[as_size(ev[0])], old_vertex[as_size(ev[1])]));
    }
    out.set_edge_nodes(std::move(nodes));
  }
  return out;
}

template <int Dim>
Mesh<Dim> extract(const Mesh<Dim>& mesh, const std::function<bool(const Point<Dim>&)>& keep) {
  std::vector<Index> cells;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if (keep(affine_map(mesh, c).centroid())) cells.push_back(c);
  }
  return extract(mesh, std::span<const Index>(cells));
}

template Mesh<2> extract<2>(const Mesh<2>&, std::span<const Index>);
template Mesh<3> extract<3>(const Mesh<3>&, std::span<const Index>);
template Mesh<2> extract<2>(const Mesh<2>&, const std::function<bool(const Point<2>&)>&);
template Mesh<3> extract<3>(const Mesh<3>&, const std::function<bool(const Point<3>&)>&);

}  // namespace hpfem::mesh

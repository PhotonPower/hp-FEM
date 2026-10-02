#include "hpfem/mesh/generators.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
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
  std::vector<Mesh<2>::CellVertices> cells(square.cells().begin(), square.cells().end());
  Mesh<2> m(std::move(vertices), std::move(cells));
  m.tag_boundary(kDiscBoundary);
  if (curved) {
    curve_boundary<2>(m, kDiscBoundary, [center, radius](const Point<2>& x) {
      return Point<2>(center + radius * (x - center).normalized());
    });
  }
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
  std::vector<Mesh<3>::CellVertices> cells(cube.cells().begin(), cube.cells().end());
  Mesh<3> m(std::move(vertices), std::move(cells));
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
      x = (1.0 - t) * radius * ellipse(q / s) + t * radius * q;
    } else {
      x = radius * q;
    }
    vertices.push_back(x);
  }
  std::vector<Mesh<2>::CellVertices> cells_list(grid.cells().begin(), grid.cells().end());
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

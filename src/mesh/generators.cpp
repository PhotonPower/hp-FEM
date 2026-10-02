#include "hpfem/mesh/generators.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "hpfem/core/error.hpp"

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

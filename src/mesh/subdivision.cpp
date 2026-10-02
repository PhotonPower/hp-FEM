#include "hpfem/mesh/subdivision.hpp"

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <fmt/format.h>

#include "hpfem/core/error.hpp"
#include "hpfem/mesh/geometry.hpp"

namespace hpfem::mesh {

namespace {

/// Lattice points and sub-simplices of the n-fold subdivided reference simplex.
template <int Dim>
struct Lattice {
  using Cell = std::array<Index, static_cast<std::size_t>(Dim + 1)>;
  std::vector<Point<Dim>> points;
  std::vector<Cell> cells;
};

template <int Dim>
Lattice<Dim> build_lattice(int n) {
  Lattice<Dim> lattice;
  const Index stride = n + 1;
  std::vector<Index> ids(as_size(stride * stride * stride), kInvalidIndex);
  const auto key = [stride](Index i, Index j, Index k) { return (i * stride + j) * stride + k; };
  const auto add_point = [&](Index i, Index j, Index k) {
    ids[as_size(key(i, j, k))] = static_cast<Index>(lattice.points.size());
    Point<Dim> xi;
    xi(0) = static_cast<Real>(i) / static_cast<Real>(n);
    xi(1) = static_cast<Real>(j) / static_cast<Real>(n);
    if constexpr (Dim == 3) xi(2) = static_cast<Real>(k) / static_cast<Real>(n);
    lattice.points.push_back(xi);
  };
  // sub-cells are stored with the orientation of the reference simplex
  const auto add_cell = [&](typename Lattice<Dim>::Cell v) {
    Eigen::Matrix<Real, Dim, Dim> jacobian;
    for (int d = 0; d < Dim; ++d) {
      jacobian.col(d) = lattice.points[as_size(v[as_size(d + 1)])] - lattice.points[as_size(v[0])];
    }
    if (jacobian.determinant() < 0) std::swap(v[Dim - 1], v[Dim]);
    lattice.cells.push_back(v);
  };
  const auto id = [&](Index i, Index j, Index k) { return ids[as_size(key(i, j, k))]; };

  if constexpr (Dim == 2) {
    for (Index i = 0; i <= n; ++i) {
      for (Index j = 0; i + j <= n; ++j) add_point(i, j, 0);
    }
    for (Index i = 0; i < n; ++i) {
      for (Index j = 0; i + j < n; ++j) {
        add_cell({id(i, j, 0), id(i + 1, j, 0), id(i, j + 1, 0)});
        if (i + j + 2 <= n) add_cell({id(i + 1, j, 0), id(i + 1, j + 1, 0), id(i, j + 1, 0)});
      }
    }
  } else {
    for (Index i = 0; i <= n; ++i) {
      for (Index j = 0; i + j <= n; ++j) {
        for (Index k = 0; i + j + k <= n; ++k) add_point(i, j, k);
      }
    }
    for (Index i = 0; i < n; ++i) {
      for (Index j = 0; i + j < n; ++j) {
        for (Index k = 0; i + j + k < n; ++k) {
          // corner tetrahedron
          add_cell({id(i, j, k), id(i + 1, j, k), id(i, j + 1, k), id(i, j, k + 1)});
          if (i + j + k + 2 <= n) {
            // octahedron between the corner and the inverted tetrahedron, cut along the
            // diagonal (i+1,j,k) - (i,j+1,k+1)
            const Index a = id(i + 1, j, k);
            const Index b = id(i, j + 1, k + 1);
            const std::array<Index, 4> ring{id(i, j + 1, k), id(i + 1, j + 1, k),
                                            id(i + 1, j, k + 1), id(i, j, k + 1)};
            for (std::size_t r = 0; r < 4; ++r) add_cell({a, b, ring[r], ring[(r + 1) % 4]});
          }
          if (i + j + k + 3 <= n) {
            add_cell({id(i + 1, j + 1, k), id(i + 1, j, k + 1), id(i, j + 1, k + 1),
                      id(i + 1, j + 1, k + 1)});
          }
        }
      }
    }
  }
  HPFEM_ASSERT(static_cast<Index>(lattice.points.size()) == Subdivided<Dim>::vertices_per_cell(n),
               "lattice point count");
  HPFEM_ASSERT(static_cast<Index>(lattice.cells.size()) == Subdivided<Dim>::cells_per_cell(n),
               "lattice cell count");
  return lattice;
}

}  // namespace

template <int Dim>
Subdivided<Dim> subdivide(const Mesh<Dim>& parent, int n) {
  if (n < 1) throw InvalidArgument(fmt::format("subdivide: n = {} must be at least 1", n));
  const Lattice<Dim> lattice = build_lattice<Dim>(n);
  const Index num_cells = parent.num_cells();

  std::vector<typename Mesh<Dim>::Vertex> vertices;
  std::vector<typename Mesh<Dim>::CellVertices> cells;
  std::vector<Tag> tags;
  std::vector<Index> parent_cell;
  std::vector<Index> vertex_parent;
  std::vector<Point<Dim>> vertex_xi;
  vertices.reserve(as_size(num_cells) * lattice.points.size());
  cells.reserve(as_size(num_cells) * lattice.cells.size());
  for (Index c = 0; c < num_cells; ++c) {
    const auto geometry = cell_geometry(parent, c);
    const Index base = static_cast<Index>(vertices.size());
    for (const auto& xi : lattice.points) {
      vertices.push_back(geometry->evaluate(xi).x);
      vertex_parent.push_back(c);
      vertex_xi.push_back(xi);
    }
    for (const auto& sub : lattice.cells) {
      typename Mesh<Dim>::CellVertices cv;
      for (std::size_t d = 0; d < cv.size(); ++d) cv[d] = base + sub[d];
      cells.push_back(cv);
      tags.push_back(parent.cell_tag(c));
      parent_cell.push_back(c);
    }
  }
  return Subdivided<Dim>{Mesh<Dim>(std::move(vertices), std::move(cells), std::move(tags)),
                         std::move(parent_cell), std::move(vertex_parent), std::move(vertex_xi)};
}

template struct Subdivided<2>;
template struct Subdivided<3>;
template Subdivided<2> subdivide<2>(const Mesh<2>&, int);
template Subdivided<3> subdivide<3>(const Mesh<3>&, int);

}  // namespace hpfem::mesh

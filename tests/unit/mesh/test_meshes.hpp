#pragma once
// Small structured meshes and helpers shared by the mesh unit tests. Test-local until the
// mesh generators of roadmap M1 land in the library.

#include <algorithm>
#include <array>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "hpfem/mesh/mesh.hpp"

namespace hpfem::mesh::testing {

/// n x n unit square, every square split along its diagonal into two triangles.
inline Mesh<2> structured_triangles(Index n) {
  std::vector<Point<2>> vertices;
  for (Index j = 0; j <= n; ++j) {
    for (Index i = 0; i <= n; ++i) {
      vertices.emplace_back(static_cast<Real>(i) / static_cast<Real>(n),
                            static_cast<Real>(j) / static_cast<Real>(n));
    }
  }
  const auto id = [n](Index i, Index j) { return j * (n + 1) + i; };
  std::vector<Mesh<2>::CellVertices> cells;
  for (Index j = 0; j < n; ++j) {
    for (Index i = 0; i < n; ++i) {
      cells.push_back({id(i, j), id(i + 1, j), id(i + 1, j + 1)});
      cells.push_back({id(i, j), id(i + 1, j + 1), id(i, j + 1)});
    }
  }
  return Mesh<2>(std::move(vertices), std::move(cells));
}

/// n x n x n unit cube, every cube split into six tetrahedra (Kuhn / Freudenthal).
inline Mesh<3> structured_tetrahedra(Index n) {
  std::vector<Point<3>> vertices;
  for (Index k = 0; k <= n; ++k) {
    for (Index j = 0; j <= n; ++j) {
      for (Index i = 0; i <= n; ++i) {
        vertices.emplace_back(static_cast<Real>(i) / static_cast<Real>(n),
                              static_cast<Real>(j) / static_cast<Real>(n),
                              static_cast<Real>(k) / static_cast<Real>(n));
      }
    }
  }
  const auto id = [n](Index i, Index j, Index k) { return (k * (n + 1) + j) * (n + 1) + i; };
  std::vector<Mesh<3>::CellVertices> cells;
  for (Index k = 0; k < n; ++k) {
    for (Index j = 0; j < n; ++j) {
      for (Index i = 0; i < n; ++i) {
        std::array<std::size_t, 3> axes{0, 1, 2};
        do {  // one tetrahedron per path along the cube edges, i.e. per axis permutation
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
  return Mesh<3>(std::move(vertices), std::move(cells));
}

/// Returns a copy of `m` with randomly renumbered vertices, shuffled cells and randomly
/// permuted local vertex order, plus the vertex map old → new.
template <int Dim>
std::pair<Mesh<Dim>, std::vector<Index>> relabel(const Mesh<Dim>& m, unsigned seed) {
  std::mt19937 rng(seed);
  std::vector<Index> perm(as_size(m.num_vertices()));
  std::iota(perm.begin(), perm.end(), Index{0});
  std::shuffle(perm.begin(), perm.end(), rng);

  std::vector<typename Mesh<Dim>::Vertex> vertices(as_size(m.num_vertices()));
  for (Index v = 0; v < m.num_vertices(); ++v) vertices[as_size(perm[as_size(v)])] = m.vertex(v);

  std::vector<typename Mesh<Dim>::CellVertices> cells(m.cells().begin(), m.cells().end());
  std::shuffle(cells.begin(), cells.end(), rng);
  for (auto& cv : cells) {
    for (auto& v : cv) v = perm[as_size(v)];
    std::shuffle(cv.begin(), cv.end(), rng);
  }
  return {Mesh<Dim>(std::move(vertices), std::move(cells)), perm};
}

/// Copies a span into a vector so that Catch2 can compare and print it.
template <class T>
std::vector<T> to_vector(std::span<const T> s) {
  return std::vector<T>(s.begin(), s.end());
}

}  // namespace hpfem::mesh::testing

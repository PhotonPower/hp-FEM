#pragma once
// Helpers shared by the mesh unit tests.

#include <algorithm>
#include <array>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/mesh.hpp"

namespace hpfem::mesh::testing {

/// n x n unit square without side tags (tests that count untagged facets rely on it).
inline Mesh<2> structured_triangles(Index n) {
  return rectangle(n, n, Point<2>::Zero(), Point<2>::Ones(), /*tag_sides=*/false);
}

/// n x n x n unit cube (Kuhn tetrahedra) without side tags.
inline Mesh<3> structured_tetrahedra(Index n) {
  return box(n, n, n, Point<3>::Zero(), Point<3>::Ones(), /*tag_sides=*/false);
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

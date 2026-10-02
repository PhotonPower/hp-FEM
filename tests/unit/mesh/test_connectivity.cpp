#include <algorithm>
#include <array>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "hpfem/core/error.hpp"
#include "hpfem/mesh/mesh.hpp"
#include "test_meshes.hpp"

using hpfem::as_size;
using hpfem::Index;
using hpfem::kInvalidIndex;
using hpfem::LocalIndex;
using hpfem::Point;
using hpfem::mesh::Mesh;
using hpfem::mesh::testing::relabel;
using hpfem::mesh::testing::structured_tetrahedra;
using hpfem::mesh::testing::structured_triangles;
using hpfem::mesh::testing::to_vector;

namespace {

/// Checks that every connectivity table is the exact inverse of cell → entity, that
/// neighbour relations are symmetric, and that boundary facets are those with one cell.
template <int Dim>
void check_connectivity(const Mesh<Dim>& m) {
  Index interior = 0;
  Index boundary = 0;
  for (Index f = 0; f < m.num_facets(); ++f) {
    const auto& fc = m.facet_cells(f);
    const auto& fl = m.facet_local_indices(f);
    REQUIRE(fc[0] != kInvalidIndex);
    for (std::size_t s = 0; s < 2; ++s) {
      if (fc[s] == kInvalidIndex) {
        REQUIRE(fl[s] == -1);
      } else {
        REQUIRE(m.cell_facets(fc[s])[as_size(fl[s])] == f);
      }
    }
    if (fc[1] == kInvalidIndex) {
      REQUIRE(m.is_boundary_facet(f));
      ++boundary;
    } else {
      REQUIRE(fc[0] < fc[1]);
      REQUIRE_FALSE(m.is_boundary_facet(f));
      ++interior;
    }
  }
  REQUIRE(m.num_boundary_facets() == boundary);
  REQUIRE(2 * interior + boundary == Mesh<Dim>::kFacetsPerCell * m.num_cells());

  const auto bf = m.boundary_facets();
  REQUIRE(static_cast<Index>(bf.size()) == boundary);
  for (std::size_t i = 0; i < bf.size(); ++i) {
    REQUIRE(m.is_boundary_facet(bf[i]));
    if (i > 0) REQUIRE(bf[i - 1] < bf[i]);
  }

  for (Index c = 0; c < m.num_cells(); ++c) {
    const auto& facets = m.cell_facets(c);
    const auto& nb = m.cell_neighbors(c);
    for (std::size_t k = 0; k < facets.size(); ++k) {
      if (nb[k] == kInvalidIndex) {
        REQUIRE(m.is_boundary_facet(facets[k]));
      } else {
        REQUIRE(nb[k] != c);
        const auto& other = m.cell_facets(nb[k]);
        const auto pos = std::find(other.begin(), other.end(), facets[k]);
        REQUIRE(pos != other.end());
        REQUIRE(m.cell_neighbors(nb[k])[as_size(pos - other.begin())] == c);
      }
    }
  }

  Index total = 0;
  for (Index e = 0; e < m.num_edges(); ++e) {
    const auto ec = m.edge_cells(e);
    REQUIRE_FALSE(ec.empty());
    for (std::size_t i = 0; i < ec.size(); ++i) {
      if (i > 0) REQUIRE(ec[i - 1] < ec[i]);
      const auto& ce = m.cell_edges(ec[i]);
      REQUIRE(std::find(ce.begin(), ce.end(), e) != ce.end());
    }
    total += static_cast<Index>(ec.size());
    if constexpr (Dim == 2) {  // in 2D the edge ring is the facet pair
      std::vector<Index> expected;
      for (const Index c : m.facet_cells(e)) {
        if (c != kInvalidIndex) expected.push_back(c);
      }
      REQUIRE(to_vector(ec) == expected);
    }
  }
  REQUIRE(total == Mesh<Dim>::kEdgesPerCell * m.num_cells());
}

/// Id of the edge with the given (unordered) vertex pair.
template <int Dim>
Index find_edge(const Mesh<Dim>& m, Index a, Index b) {
  const typename Mesh<Dim>::EdgeVertices key{std::min(a, b), std::max(a, b)};
  const auto edges = m.edges();
  const auto it = std::find(edges.begin(), edges.end(), key);
  REQUIRE(it != edges.end());
  return static_cast<Index>(it - edges.begin());
}

}  // namespace

// --- 2D -------------------------------------------------------------------------------------

TEST_CASE("two triangles: shared edge, neighbours and boundary", "[mesh]") {
  const Mesh<2> m({{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}}, {{0, 1, 2}, {0, 2, 3}});
  const Index diag = find_edge(m, 0, 2);
  REQUIRE(diag == 1);
  REQUIRE(m.facet_cells(diag) == Mesh<2>::FacetCells{0, 1});
  REQUIRE(m.facet_local_indices(diag) == Mesh<2>::FacetLocalIndices{2, 0});
  REQUIRE_FALSE(m.is_boundary_facet(diag));
  REQUIRE(m.num_boundary_facets() == 4);
  REQUIRE(to_vector(m.boundary_facets()) == std::vector<Index>{0, 2, 3, 4});
  REQUIRE(m.cell_neighbors(0) == Mesh<2>::CellNeighbors{kInvalidIndex, kInvalidIndex, 1});
  REQUIRE(m.cell_neighbors(1) == Mesh<2>::CellNeighbors{0, kInvalidIndex, kInvalidIndex});
  REQUIRE(to_vector(m.edge_cells(diag)) == std::vector<Index>{0, 1});
  REQUIRE(to_vector(m.edge_cells(0)) == std::vector<Index>{0});
  check_connectivity(m);
}

TEST_CASE("structured triangles: boundary edge count", "[mesh]") {
  for (const Index n : {1, 2, 5}) {
    const Mesh<2> m = structured_triangles(n);
    REQUIRE(m.num_boundary_facets() == 4 * n);
    check_connectivity(m);
  }
}

// --- 3D -------------------------------------------------------------------------------------

TEST_CASE("two tetrahedra: shared face and edge rings", "[mesh]") {
  const Mesh<3> m(
      {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 1.0, 1.0}},
      {{0, 1, 2, 3}, {4, 3, 1, 2}});
  const Index shared = 3;  // (1,2,3), see test_mesh.cpp
  REQUIRE(m.facet_cells(shared) == Mesh<3>::FacetCells{0, 1});
  REQUIRE(m.facet_local_indices(shared) == Mesh<3>::FacetLocalIndices{0, 0});
  REQUIRE(m.num_boundary_facets() == 6);
  REQUIRE(m.cell_neighbors(0) ==
          Mesh<3>::CellNeighbors{1, kInvalidIndex, kInvalidIndex, kInvalidIndex});
  REQUIRE(m.cell_neighbors(1) ==
          Mesh<3>::CellNeighbors{0, kInvalidIndex, kInvalidIndex, kInvalidIndex});
  // edges of the shared face belong to both cells, edges through vertex 0 or 4 to one
  for (const auto [a, b] : std::array<std::array<Index, 2>, 3>{{{1, 2}, {1, 3}, {2, 3}}}) {
    REQUIRE(to_vector(m.edge_cells(find_edge(m, a, b))) == std::vector<Index>{0, 1});
  }
  REQUIRE(to_vector(m.edge_cells(find_edge(m, 0, 1))) == std::vector<Index>{0});
  REQUIRE(to_vector(m.edge_cells(find_edge(m, 4, 2))) == std::vector<Index>{1});
  check_connectivity(m);
}

TEST_CASE("Kuhn cube: boundary faces and edge rings", "[mesh]") {
  for (const Index n : {1, 2, 3}) {
    const Mesh<3> m = structured_tetrahedra(n);
    REQUIRE(m.num_boundary_facets() == 12 * n * n);
    check_connectivity(m);
  }
  // all six tetrahedra of a single cube share its body diagonal
  const Mesh<3> cube = structured_tetrahedra(1);
  REQUIRE(to_vector(cube.edge_cells(find_edge(cube, 0, 7))) ==
          std::vector<Index>{0, 1, 2, 3, 4, 5});
}

// --- invariance and errors ------------------------------------------------------------------

TEST_CASE("connectivity is invariant under renumbering", "[mesh]") {
  const Mesh<2> m2 = structured_triangles(3);
  const Mesh<3> m3 = structured_tetrahedra(2);
  for (const unsigned seed : {11U, 12U}) {
    const auto r2 = relabel(m2, seed).first;
    REQUIRE(r2.num_boundary_facets() == m2.num_boundary_facets());
    check_connectivity(r2);
    const auto r3 = relabel(m3, seed).first;
    REQUIRE(r3.num_boundary_facets() == m3.num_boundary_facets());
    check_connectivity(r3);
  }
}

TEST_CASE("non-manifold meshes are rejected", "[mesh]") {
  using Catch::Matchers::ContainsSubstring;
  const std::vector<Point<2>> v2{{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {0.0, -1.0}, {1.0, 1.0}};
  REQUIRE_THROWS_AS(Mesh<2>(v2, {{0, 1, 2}, {0, 1, 3}, {0, 1, 4}}), hpfem::InvalidArgument);
  REQUIRE_THROWS_WITH(Mesh<2>(v2, {{0, 1, 2}, {0, 1, 3}, {0, 1, 4}}),
                      ContainsSubstring("not a manifold"));
  const std::vector<Point<3>> v3{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},  {0.0, 1.0, 0.0},
                                 {0.0, 0.0, 1.0}, {0.0, 0.0, -1.0}, {1.0, 1.0, 1.0}};
  REQUIRE_THROWS_WITH(Mesh<3>(v3, {{0, 1, 2, 3}, {0, 1, 2, 4}, {0, 1, 2, 5}}),
                      ContainsSubstring("vertices 0,1,2"));
}

#if defined(HPFEM_ENABLE_ASSERTS)
TEST_CASE("connectivity accessors assert on out-of-range indices", "[mesh]") {
  const Mesh<2> m({{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}}, {{0, 1, 2}});
  REQUIRE_THROWS_AS(m.facet_cells(3), hpfem::Error);
  REQUIRE_THROWS_AS(m.cell_neighbors(1), hpfem::Error);
  REQUIRE_THROWS_AS(m.edge_cells(-1), hpfem::Error);
}
#endif

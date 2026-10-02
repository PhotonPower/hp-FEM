#include <algorithm>
#include <array>
#include <numeric>
#include <set>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "hpfem/core/error.hpp"
#include "hpfem/mesh/mesh.hpp"
#include "test_meshes.hpp"

using hpfem::as_size;
using hpfem::Index;
using hpfem::LocalIndex;
using hpfem::Point;
using hpfem::mesh::kFacePermutations;
using hpfem::mesh::Mesh;
using hpfem::mesh::SimplexTopology;
using hpfem::mesh::testing::relabel;
using hpfem::mesh::testing::structured_tetrahedra;
using hpfem::mesh::testing::structured_triangles;

namespace {

// --- generic invariants --------------------------------------------------------------------

/// Checks every cell→entity reference against ADR-0003: entity vertex tuples are ascending,
/// edge flip flags and face permutation codes reproduce the local vertex order, entity ids
/// are unique and lexicographically ordered, and the facet view matches edges/faces.
template <int Dim>
void check_entity_consistency(const Mesh<Dim>& m) {
  using T = SimplexTopology<Dim>;
  for (Index c = 0; c < m.num_cells(); ++c) {
    const auto& cv = m.cell_vertices(c);

    for (std::size_t k = 0; k < as_size(Mesh<Dim>::kEdgesPerCell); ++k) {
      const Index a = cv[as_size(T::kEdgeVertices[k][0])];
      const Index b = cv[as_size(T::kEdgeVertices[k][1])];
      const auto& ev = m.edge_vertices(m.cell_edges(c)[k]);
      REQUIRE(ev[0] < ev[1]);
      const typename Mesh<Dim>::EdgeVertices expected =
          m.cell_edge_flipped(c)[k] ? typename Mesh<Dim>::EdgeVertices{b, a}
                                    : typename Mesh<Dim>::EdgeVertices{a, b};
      REQUIRE(ev == expected);
    }

    if constexpr (Dim == 2) {
      REQUIRE(m.cell_facets(c) == m.cell_edges(c));
    } else {
      REQUIRE(m.cell_facets(c) == m.cell_faces(c));
      for (std::size_t i = 0; i < 4; ++i) {
        const auto& fv = m.face_vertices(m.cell_faces(c)[i]);
        REQUIRE(fv[0] < fv[1]);
        REQUIRE(fv[1] < fv[2]);
        const auto code = m.cell_face_permutations(c)[i];
        REQUIRE(code < 6);
        const auto& perm = kFacePermutations[code];
        for (std::size_t j = 0; j < 3; ++j) {
          REQUIRE(cv[as_size(T::kFaceVertices[i][j])] == fv[as_size(perm[j])]);
        }
      }
    }
  }

  std::set<typename Mesh<Dim>::EdgeVertices> unique_edges(m.edges().begin(), m.edges().end());
  REQUIRE(static_cast<Index>(unique_edges.size()) == m.num_edges());
  for (Index e = 1; e < m.num_edges(); ++e) REQUIRE(m.edge_vertices(e - 1) < m.edge_vertices(e));
  for (Index f = 0; f < m.num_facets(); ++f) {
    if constexpr (Dim == 2) {
      REQUIRE(m.facet_vertices(f) == m.edge_vertices(f));
    } else {
      REQUIRE(m.facet_vertices(f) == m.face_vertices(f));
    }
  }
  if constexpr (Dim == 3) {
    std::set<typename Mesh<Dim>::FaceVertices> unique_faces(m.faces().begin(), m.faces().end());
    REQUIRE(static_cast<Index>(unique_faces.size()) == m.num_faces());
    for (Index f = 1; f < m.num_faces(); ++f) {
      REQUIRE(m.face_vertices(f - 1) < m.face_vertices(f));
    }
  }
}

template <int Dim>
std::set<std::array<Index, 2>> mapped_edge_set(const Mesh<Dim>& m, const std::vector<Index>& map) {
  std::set<std::array<Index, 2>> out;
  for (const auto& e : m.edges()) {
    std::array<Index, 2> t{map[as_size(e[0])], map[as_size(e[1])]};
    std::sort(t.begin(), t.end());
    out.insert(t);
  }
  return out;
}

}  // namespace

// --- 2D -------------------------------------------------------------------------------------

TEST_CASE("single triangle", "[mesh]") {
  const Mesh<2> m({{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}}, {{0, 1, 2}});
  REQUIRE(m.num_vertices() == 3);
  REQUIRE(m.num_edges() == 3);
  REQUIRE(m.num_facets() == 3);
  REQUIRE(m.num_cells() == 1);
  REQUIRE(m.vertex(1)(0) == 1.0);
  // lexicographic edge ids: (0,1) -> 0, (0,2) -> 1, (1,2) -> 2
  REQUIRE(m.cell_edges(0) == Mesh<2>::CellEdges{0, 2, 1});
  // local e2 = (v2, v0) runs against the global direction 0 -> 2
  REQUIRE(m.cell_edge_flipped(0) == Mesh<2>::CellEdgeFlags{false, false, true});
  check_entity_consistency(m);
}

TEST_CASE("two triangles share one edge", "[mesh]") {
  const Mesh<2> m({{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}}, {{0, 1, 2}, {0, 2, 3}});
  REQUIRE(m.num_vertices() == 4);
  REQUIRE(m.num_edges() == 5);
  REQUIRE(m.num_cells() == 2);
  // the diagonal (0,2) is edge 1 in lexicographic order and must be shared
  const auto& e0 = m.cell_edges(0);
  const auto& e1 = m.cell_edges(1);
  REQUIRE(std::count(e0.begin(), e0.end(), Index{1}) == 1);
  REQUIRE(std::count(e1.begin(), e1.end(), Index{1}) == 1);
  REQUIRE(m.edge_vertices(1) == Mesh<2>::EdgeVertices{0, 2});
  REQUIRE(m.cell_edge_flipped(0)[2]);        // cell 0: local e2 = (v2, v0) = 2 -> 0, flipped
  REQUIRE_FALSE(m.cell_edge_flipped(1)[0]);  // cell 1: local e0 = (v0, v1) = 0 -> 2, as global
  check_entity_consistency(m);
}

TEST_CASE("structured triangle mesh: counts and Euler characteristic", "[mesh]") {
  for (const Index n : {1, 2, 5}) {
    const Mesh<2> m = structured_triangles(n);
    REQUIRE(m.num_vertices() == (n + 1) * (n + 1));
    REQUIRE(m.num_edges() == 3 * n * n + 2 * n);
    REQUIRE(m.num_cells() == 2 * n * n);
    REQUIRE(m.num_vertices() - m.num_edges() + m.num_cells() == 1);  // disk: V - E + F = 1
    check_entity_consistency(m);
  }
}

TEST_CASE("2D topology is invariant under vertex and cell renumbering", "[mesh]") {
  const Mesh<2> m = structured_triangles(4);
  for (const unsigned seed : {1U, 2U, 3U}) {
    const auto [r, perm] = relabel(m, seed);
    REQUIRE(r.num_vertices() == m.num_vertices());
    REQUIRE(r.num_edges() == m.num_edges());
    REQUIRE(r.num_cells() == m.num_cells());
    check_entity_consistency(r);
    std::vector<Index> identity(as_size(r.num_vertices()));
    std::iota(identity.begin(), identity.end(), Index{0});
    REQUIRE(mapped_edge_set(m, perm) == mapped_edge_set(r, identity));
  }
}

// --- 3D -------------------------------------------------------------------------------------

TEST_CASE("single tetrahedron", "[mesh]") {
  const Mesh<3> m({{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}},
                  {{0, 1, 2, 3}});
  REQUIRE(m.num_vertices() == 4);
  REQUIRE(m.num_edges() == 6);
  REQUIRE(m.num_faces() == 4);
  REQUIRE(m.num_facets() == 4);
  REQUIRE(m.num_cells() == 1);
  // local order equals global order: identity everywhere
  REQUIRE(m.cell_edges(0) == Mesh<3>::CellEdges{0, 1, 2, 3, 4, 5});
  REQUIRE(m.cell_edge_flipped(0) ==
          Mesh<3>::CellEdgeFlags{false, false, false, false, false, false});
  REQUIRE(m.cell_face_permutations(0) == Mesh<3>::CellFacePermutations{0, 0, 0, 0});
  for (std::size_t i = 0; i < 4; ++i) {  // face i is opposite vertex i
    const auto& fv = m.face_vertices(m.cell_faces(0)[i]);
    REQUIRE(std::find(fv.begin(), fv.end(), static_cast<Index>(i)) == fv.end());
  }
  check_entity_consistency(m);
}

TEST_CASE("tetrahedron with reversed local order gets flips and permutations", "[mesh]") {
  const Mesh<3> m({{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}},
                  {{3, 2, 1, 0}});
  // every local edge (a,b) has a > b globally
  REQUIRE(m.cell_edge_flipped(0) == Mesh<3>::CellEdgeFlags{true, true, true, true, true, true});
  // local face (l0,l1,l2) maps to globally decreasing ids: full reversal, code 5 = (2,1,0)
  REQUIRE(m.cell_face_permutations(0) == Mesh<3>::CellFacePermutations{5, 5, 5, 5});
  check_entity_consistency(m);
}

TEST_CASE("two tetrahedra share one face", "[mesh]") {
  const Mesh<3> m(
      {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 1.0, 1.0}},
      {{0, 1, 2, 3}, {4, 3, 1, 2}});
  REQUIRE(m.num_vertices() == 5);
  REQUIRE(m.num_edges() == 9);
  REQUIRE(m.num_faces() == 7);
  REQUIRE(m.num_cells() == 2);
  // (1,2,3) is face 3 in lexicographic order; it is local face 0 of cell 0 and of cell 1
  REQUIRE(m.face_vertices(3) == Mesh<3>::FaceVertices{1, 2, 3});
  REQUIRE(m.cell_faces(0)[0] == 3);
  REQUIRE(m.cell_faces(1)[0] == 3);
  REQUIRE(m.cell_face_permutations(0)[0] == 0);  // local (1,2,3): identity
  REQUIRE(m.cell_face_permutations(1)[0] == 4);  // local (3,1,2) sits at positions (2,0,1)
  check_entity_consistency(m);
  REQUIRE(m.num_vertices() - m.num_edges() + m.num_faces() - m.num_cells() == 1);
}

TEST_CASE("Kuhn cube: counts and Euler characteristic", "[mesh]") {
  for (const Index n : {1, 2, 3}) {
    const Mesh<3> m = structured_tetrahedra(n);
    REQUIRE(m.num_vertices() == (n + 1) * (n + 1) * (n + 1));
    // axis-parallel + face-diagonal + body-diagonal edges
    REQUIRE(m.num_edges() == 3 * n * (n + 1) * (n + 1) + 3 * n * n * (n + 1) + n * n * n);
    REQUIRE(m.num_faces() == 12 * n * n * n + 6 * n * n);
    REQUIRE(m.num_cells() == 6 * n * n * n);
    REQUIRE(m.num_vertices() - m.num_edges() + m.num_faces() - m.num_cells() == 1);  // ball
    check_entity_consistency(m);
  }
}

TEST_CASE("3D topology is invariant under vertex and cell renumbering", "[mesh]") {
  const Mesh<3> m = structured_tetrahedra(2);
  for (const unsigned seed : {7U, 8U, 9U}) {
    const auto [r, perm] = relabel(m, seed);
    REQUIRE(r.num_vertices() == m.num_vertices());
    REQUIRE(r.num_edges() == m.num_edges());
    REQUIRE(r.num_faces() == m.num_faces());
    REQUIRE(r.num_cells() == m.num_cells());
    check_entity_consistency(r);
    std::vector<Index> identity(as_size(r.num_vertices()));
    std::iota(identity.begin(), identity.end(), Index{0});
    REQUIRE(mapped_edge_set(m, perm) == mapped_edge_set(r, identity));
  }
}

// --- errors ---------------------------------------------------------------------------------

TEST_CASE("invalid cells are rejected with a message naming the cell", "[mesh]") {
  const std::vector<Point<2>> v{{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
  REQUIRE_THROWS_AS(Mesh<2>(v, {{0, 1, 3}}), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(Mesh<2>(v, {{0, 1, -1}}), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(Mesh<2>(v, {{0, 1, 1}}), hpfem::InvalidArgument);
  REQUIRE_THROWS_WITH(Mesh<2>(v, {{0, 1, 2}, {0, 2, 7}}),
                      Catch::Matchers::ContainsSubstring("cell 1"));
}

#if defined(HPFEM_ENABLE_ASSERTS)
TEST_CASE("accessors assert on out-of-range indices", "[mesh]") {
  const Mesh<2> m({{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}}, {{0, 1, 2}});
  REQUIRE_THROWS_AS(m.vertex(3), hpfem::Error);
  REQUIRE_THROWS_AS(m.cell_vertices(-1), hpfem::Error);
  REQUIRE_THROWS_AS(m.edge_vertices(3), hpfem::Error);
}
#endif

#include <algorithm>
#include <array>
#include <set>

#include <catch2/catch_test_macros.hpp>

#include "hpfem/mesh/simplex_topology.hpp"

using hpfem::LocalIndex;
using hpfem::mesh::kFacePermutations;
using hpfem::mesh::SimplexTopology;

namespace {

template <std::size_t N>
std::array<LocalIndex, N> sorted(std::array<LocalIndex, N> a) {
  std::sort(a.begin(), a.end());
  return a;
}

}  // namespace

TEST_CASE("triangle local numbering matches docs/theory/nedelec.md", "[mesh]") {
  using T = SimplexTopology<2>;
  STATIC_REQUIRE(T::kDim == 2);
  STATIC_REQUIRE(T::kNumVertices == 3);
  STATIC_REQUIRE(T::kNumEdges == 3);
  STATIC_REQUIRE(T::kNumFacets == 3);
  STATIC_REQUIRE(T::kVerticesPerFacet == 2);

  // counter-clockwise: e_k = (v_k, v_{k+1 mod 3})
  for (LocalIndex k = 0; k < 3; ++k) {
    const auto e = T::kEdgeVertices[static_cast<std::size_t>(k)];
    REQUIRE(e[0] == k);
    REQUIRE(e[1] == (k + 1) % 3);
  }
  STATIC_REQUIRE(T::kFacetVertices == T::kEdgeVertices);
}

TEST_CASE("tetrahedron local numbering matches docs/theory/nedelec.md", "[mesh]") {
  using T = SimplexTopology<3>;
  STATIC_REQUIRE(T::kDim == 3);
  STATIC_REQUIRE(T::kNumVertices == 4);
  STATIC_REQUIRE(T::kNumEdges == 6);
  STATIC_REQUIRE(T::kNumFaces == 4);
  STATIC_REQUIRE(T::kNumFacets == 4);
  STATIC_REQUIRE(T::kVerticesPerFacet == 3);

  SECTION("edges: ascending pairs in lexicographic order") {
    const std::array<std::array<LocalIndex, 2>, 6> expected{
        {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}};
    REQUIRE(T::kEdgeVertices == expected);
    for (std::size_t k = 1; k < 6; ++k) REQUIRE(T::kEdgeVertices[k - 1] < T::kEdgeVertices[k]);
  }

  SECTION("faces: face i is opposite vertex i, vertices ascending") {
    const std::array<std::array<LocalIndex, 3>, 4> expected{
        {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}}};
    REQUIRE(T::kFaceVertices == expected);
    for (std::size_t i = 0; i < 4; ++i) {
      const auto f = T::kFaceVertices[i];
      REQUIRE(std::find(f.begin(), f.end(), static_cast<LocalIndex>(i)) == f.end());
      REQUIRE(sorted(f) == f);
    }
    STATIC_REQUIRE(T::kFacetVertices == T::kFaceVertices);
  }

  SECTION("face-edge table: edge j of face (f0,f1,f2) is (f0,f1), (f1,f2), (f0,f2)") {
    std::array<int, 6> edge_use_count{};
    for (std::size_t i = 0; i < 4; ++i) {
      const auto f = T::kFaceVertices[i];
      const std::array<std::array<LocalIndex, 2>, 3> pairs{
          {{f[0], f[1]}, {f[1], f[2]}, {f[0], f[2]}}};
      for (std::size_t j = 0; j < 3; ++j) {
        const auto e = T::kFaceEdges[i][j];
        REQUIRE(e >= 0);
        REQUIRE(e < 6);
        REQUIRE(T::kEdgeVertices[static_cast<std::size_t>(e)] == sorted(pairs[j]));
        ++edge_use_count[static_cast<std::size_t>(e)];
      }
    }
    for (const int n : edge_use_count) REQUIRE(n == 2);  // every edge lies in two faces
  }
}

TEST_CASE("face permutation table lists all six permutations in lexicographic order", "[mesh]") {
  std::set<std::array<LocalIndex, 3>> seen;
  for (std::size_t k = 0; k < 6; ++k) {
    const auto p = kFacePermutations[k];
    REQUIRE(sorted(p) == std::array<LocalIndex, 3>{0, 1, 2});
    seen.insert(p);
    if (k > 0) REQUIRE(kFacePermutations[k - 1] < p);
  }
  REQUIRE(seen.size() == 6);
  REQUIRE(kFacePermutations[0] == std::array<LocalIndex, 3>{0, 1, 2});  // identity is code 0
}

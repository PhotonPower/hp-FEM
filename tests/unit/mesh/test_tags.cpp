#include <algorithm>
#include <array>
#include <optional>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "hpfem/core/error.hpp"
#include "hpfem/mesh/mesh.hpp"
#include "test_meshes.hpp"

using Catch::Matchers::ContainsSubstring;
using hpfem::Index;
using hpfem::kInvalidIndex;
using hpfem::Point;
using hpfem::mesh::kNoTag;
using hpfem::mesh::Mesh;
using hpfem::mesh::Tag;
using hpfem::mesh::testing::structured_tetrahedra;
using hpfem::mesh::testing::structured_triangles;

namespace {

Mesh<2> two_triangles(std::vector<Tag> cell_tags = {}) {
  return Mesh<2>({{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}}, {{0, 1, 2}, {0, 2, 3}},
                 std::move(cell_tags));
}

Mesh<3> two_tetrahedra() {
  return Mesh<3>(
      {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 1.0, 1.0}},
      {{0, 1, 2, 3}, {4, 3, 1, 2}});
}

}  // namespace

TEST_CASE("cell tags: construction, defaults and queries", "[mesh]") {
  const Mesh<2> untagged = two_triangles();
  REQUIRE(untagged.cell_tags().size() == 2);
  REQUIRE(untagged.cell_tag(0) == kNoTag);
  REQUIRE(untagged.cells_with_tag(kNoTag) == std::vector<Index>{0, 1});

  Mesh<2> m = two_triangles({1, 2});
  REQUIRE(m.cell_tag(0) == 1);
  REQUIRE(m.cell_tag(1) == 2);
  REQUIRE(m.cells_with_tag(2) == std::vector<Index>{1});
  REQUIRE(m.cells_with_tag(3).empty());
  m.set_cell_tag(0, 2);
  REQUIRE(m.cells_with_tag(2) == std::vector<Index>{0, 1});

  REQUIRE_THROWS_AS(two_triangles({1}), hpfem::InvalidArgument);
  REQUIRE_THROWS_WITH(two_triangles({1, 2, 3}), ContainsSubstring("3 cell tags"));
}

TEST_CASE("entity lookup by unordered vertex tuples", "[mesh]") {
  const Mesh<2> m = two_triangles();
  // edges in lexicographic order: (0,1) (0,2) (0,3) (1,2) (2,3)
  REQUIRE(m.edge_id(0, 1) == 0);
  REQUIRE(m.edge_id(2, 0) == 1);
  REQUIRE(m.edge_id(3, 2) == 4);
  REQUIRE(m.edge_id(1, 3) == kInvalidIndex);
  REQUIRE(m.edge_id(0, 9) == kInvalidIndex);
  REQUIRE(m.facet_id({3, 0}) == 2);
  REQUIRE(m.facet_id({1, 3}) == kInvalidIndex);

  const Mesh<3> t = two_tetrahedra();
  // faces: (0,1,2) (0,1,3) (0,2,3) (1,2,3) (1,2,4) (1,3,4) (2,3,4)
  REQUIRE(t.face_id(3, 2, 1) == 3);
  REQUIRE(t.face_id(4, 1, 2) == 4);
  REQUIRE(t.face_id(0, 1, 4) == kInvalidIndex);
  REQUIRE(t.facet_id({2, 4, 3}) == 6);
  REQUIRE(t.edge_id(4, 1) == t.edge_id(1, 4));
  REQUIRE(t.edge_id(0, 4) == kInvalidIndex);
  for (Index e = 0; e < t.num_edges(); ++e) {
    const auto& v = t.edge_vertices(e);
    REQUIRE(t.edge_id(v[1], v[0]) == e);
  }
}

TEST_CASE("facet tags from vertex lists, interior facets and tag_boundary (2D)", "[mesh]") {
  Mesh<2> m = two_triangles();
  REQUIRE(
      std::all_of(m.facet_tags().begin(), m.facet_tags().end(), [](Tag t) { return t == kNoTag; }));

  const std::vector<Mesh<2>::FacetVertices> bottom_right{{1, 0}, {1, 2}};
  const std::vector<Tag> tags{10, 10};
  m.set_facet_tags(bottom_right, tags);
  m.set_facet_tags(std::vector<Mesh<2>::FacetVertices>{{3, 0}}, std::vector<Tag>{20});
  REQUIRE(m.facet_tag(m.edge_id(0, 1)) == 10);
  REQUIRE(m.facet_tag(m.edge_id(1, 2)) == 10);
  REQUIRE(m.facet_tag(m.edge_id(0, 3)) == 20);
  REQUIRE(m.facets_with_tag(10) == std::vector<Index>{0, 3});

  // the top edge (2,3) is still untagged: tag_boundary fills exactly that one
  REQUIRE(m.tag_boundary(99) == 1);
  REQUIRE(m.facet_tag(m.edge_id(2, 3)) == 99);
  REQUIRE(m.facet_tag(m.edge_id(0, 1)) == 10);  // existing tags untouched
  REQUIRE(m.tag_boundary(99) == 0);
  const Index diag = m.edge_id(0, 2);
  REQUIRE(m.facet_tag(diag) == kNoTag);  // interior facet never touched by tag_boundary

  m.set_facet_tag(diag, 5);  // interior facets may carry tags (interfaces)
  REQUIRE(m.facets_with_tag(5) == std::vector<Index>{diag});

  REQUIRE_THROWS_WITH(
      m.set_facet_tags(std::vector<Mesh<2>::FacetVertices>{{1, 3}}, std::vector<Tag>{1}),
      ContainsSubstring("(1,3)"));
  REQUIRE_THROWS_AS(m.set_facet_tags(bottom_right, std::vector<Tag>{1}), hpfem::InvalidArgument);
}

TEST_CASE("facet tags and tag_boundary (3D)", "[mesh]") {
  Mesh<3> t = two_tetrahedra();
  t.set_facet_tags(std::vector<Mesh<3>::FacetVertices>{{4, 1, 2}, {0, 2, 1}},
                   std::vector<Tag>{7, 8});
  REQUIRE(t.facet_tag(t.face_id(1, 2, 4)) == 7);
  REQUIRE(t.facet_tag(t.face_id(0, 1, 2)) == 8);
  REQUIRE(t.tag_boundary(1) == 4);  // 6 boundary faces, 2 already tagged
  REQUIRE(t.facets_with_tag(1).size() == 4);
  REQUIRE(t.facet_tag(t.face_id(1, 2, 3)) == kNoTag);  // the shared interior face

  Mesh<3> cube = structured_tetrahedra(2);
  REQUIRE(cube.tag_boundary(1) == 48);  // 12 n^2 faces for n = 2
  for (const Index f : cube.facets_with_tag(1)) REQUIRE(cube.is_boundary_facet(f));
  REQUIRE(static_cast<Index>(cube.facets_with_tag(kNoTag).size()) == cube.num_facets() - 48);

  Mesh<2> square = structured_triangles(3);
  REQUIRE(square.tag_boundary(1) == 12);
}

TEST_CASE("physical names per dimension", "[mesh]") {
  Mesh<2> m = two_triangles({1, 2});
  m.set_tag_name(2, 1, "silicon");
  m.set_tag_name(2, 2, "air");
  m.set_tag_name(1, 10, "pec");
  REQUIRE(m.tag_name(2, 1) == "silicon");
  REQUIRE(m.tag_name(1, 10) == "pec");
  REQUIRE(m.tag_name(2, 10).empty());  // tag 10 is a facet tag, not a cell tag
  REQUIRE(m.tag_by_name(2, "air") == std::optional<Tag>{2});
  REQUIRE(m.tag_by_name(1, "air") == std::nullopt);
  REQUIRE(m.tag_by_name(2, "gold") == std::nullopt);
  m.set_tag_name(2, 1, "Si");  // renaming replaces
  REQUIRE(m.tag_name(2, 1) == "Si");
  REQUIRE_THROWS_AS(m.set_tag_name(0, 1, "vertex group"), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(m.tag_name(3, 1), hpfem::InvalidArgument);

  Mesh<3> t = two_tetrahedra();
  t.set_tag_name(3, 1, "core");
  t.set_tag_name(2, 1, "wall");
  REQUIRE(t.tag_name(3, 1) == "core");
  REQUIRE(t.tag_name(2, 1) == "wall");
  REQUIRE_THROWS_AS(t.set_tag_name(1, 1, "edge group"), hpfem::InvalidArgument);
}

#if defined(HPFEM_ENABLE_ASSERTS)
TEST_CASE("tag accessors assert on out-of-range indices", "[mesh]") {
  Mesh<2> m = two_triangles();
  REQUIRE_THROWS_AS(m.cell_tag(2), hpfem::Error);
  REQUIRE_THROWS_AS(m.facet_tag(5), hpfem::Error);
  REQUIRE_THROWS_AS(m.set_cell_tag(-1, 1), hpfem::Error);
}
#endif

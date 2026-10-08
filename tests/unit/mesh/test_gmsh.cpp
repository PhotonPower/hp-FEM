#include <algorithm>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <fmt/format.h>

#include "hpfem/core/error.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/gmsh.hpp"

using Catch::Approx;
using Catch::Matchers::ContainsSubstring;
using hpfem::as_size;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::mesh::kNoTag;
using hpfem::mesh::Mesh;
using hpfem::mesh::read_gmsh;
using hpfem::mesh::rectangle;
using hpfem::mesh::Tag;

namespace {

/// Unit square, two triangles; four physical curves and one physical surface.
const char* const kSquare = R"($MeshFormat
4.1 0 8
$EndMeshFormat
$PhysicalNames
5
1 1 "bottom"
1 2 "right"
1 3 "top"
1 4 "left"
2 10 "domain"
$EndPhysicalNames
$Entities
4 4 1 0
1 0 0 0 0
2 1 0 0 0
3 1 1 0 0
4 0 1 0 0
1 0 0 0 1 0 0 1 1 2 1 -2
2 1 0 0 1 1 0 1 2 2 2 -3
3 0 1 0 1 1 0 1 3 2 3 -4
4 0 0 0 0 1 0 1 4 2 4 -1
1 0 0 0 1 1 0 1 10 4 1 2 3 4
$EndEntities
$Nodes
1 4 1 4
2 1 0 4
1
2
3
4
0 0 0
1 0 0
1 1 0
0 1 0
$EndNodes
$Elements
5 6 1 6
1 1 1 1
1 1 2
1 2 1 1
2 2 3
1 3 1 1
3 3 4
1 4 1 1
4 4 1
2 1 2 2
5 1 2 3
6 1 3 4
$EndElements
)";

/// Two tetrahedra sharing the face (2,3,4); one tagged surface element, one physical volume.
const char* const kTwoTets = R"($MeshFormat
4.1 0 8
$EndMeshFormat
$PhysicalNames
2
2 7 "wall"
3 20 "core"
$EndPhysicalNames
$Entities
0 0 1 1
1 0 0 0 1 1 1 1 7 0
1 0 0 0 1 1 1 1 20 1 1
$EndEntities
$Nodes
1 5 1 5
3 1 0 5
1
2
3
4
5
0 0 0
1 0 0
0 1 0
0 0 1
1 1 1
$EndNodes
$Elements
2 3 1 3
2 1 2 1
1 2 3 5
3 1 4 2
2 1 2 3 4
3 5 4 2 3
$EndElements
)";

Mesh<2> read2(const std::string& text, Real scale = 1.0) {
  std::istringstream in(text);
  return read_gmsh<2>(in, scale);
}

Mesh<3> read3(const std::string& text) {
  std::istringstream in(text);
  return read_gmsh<3>(in);
}

std::string replace(std::string text, const std::string& from, const std::string& to) {
  const auto pos = text.find(from);
  REQUIRE(pos != std::string::npos);
  return text.replace(pos, from.size(), to);
}

/// Writes a Mesh<2> as MSH 4.1 with sparse node tags (100 + 3 i) and shuffled node order,
/// boundary edges as line elements tagged with the facet tag, triangles on one surface
/// with physical tag 10. Exercises the tag → index mapping of the reader.
std::string write_square_mesh(const Mesh<2>& m) {
  const auto node_tag = [](Index v) { return 100 + 3 * v; };
  std::string s = "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n";
  s += "$Entities\n0 4 1 0\n";
  for (Tag t = 1; t <= 4; ++t) s += fmt::format("{} 0 0 0 1 1 0 1 {} 0\n", t, t);
  s += "1 0 0 0 1 1 0 1 10 0\n$EndEntities\n";

  const Index nv = m.num_vertices();
  s += fmt::format("$Nodes\n1 {} {} {}\n2 1 0 {}\n", nv, node_tag(0), node_tag(nv - 1), nv);
  std::vector<Index> order(as_size(nv));
  for (Index v = 0; v < nv; ++v) order[as_size(v)] = nv - 1 - v;  // reversed file order
  for (const Index v : order) s += fmt::format("{}\n", node_tag(v));
  for (const Index v : order) s += fmt::format("{} {} 0\n", m.vertex(v)(0), m.vertex(v)(1));
  s += "$EndNodes\n";

  std::size_t element_tag = 1;
  std::string blocks;
  std::size_t num_blocks = 0;
  for (Tag t = 1; t <= 4; ++t) {
    const auto facets = m.facets_with_tag(t);
    blocks += fmt::format("1 {} 1 {}\n", t, facets.size());
    for (const Index f : facets) {
      const auto& fv = m.facet_vertices(f);
      blocks += fmt::format("{} {} {}\n", element_tag++, node_tag(fv[1]), node_tag(fv[0]));
    }
    ++num_blocks;
  }
  blocks += fmt::format("2 1 2 {}\n", m.num_cells());
  for (Index c = 0; c < m.num_cells(); ++c) {
    const auto& cv = m.cell_vertices(c);
    blocks += fmt::format("{} {} {} {}\n", element_tag++, node_tag(cv[0]), node_tag(cv[1]),
                          node_tag(cv[2]));
  }
  ++num_blocks;
  s += fmt::format("$Elements\n{} {} 1 {}\n", num_blocks, element_tag - 1, element_tag - 1);
  s += blocks + "$EndElements\n";
  return s;
}

}  // namespace

TEST_CASE("Gmsh 4.1: unit square with physical curves and surface", "[mesh][gmsh]") {
  const Mesh<2> m = read2(kSquare);
  REQUIRE(m.num_vertices() == 4);
  REQUIRE(m.num_cells() == 2);
  REQUIRE(m.num_edges() == 5);
  REQUIRE(m.vertex(2) == Point<2>(1.0, 1.0));
  REQUIRE(m.cell_vertices(0) == Mesh<2>::CellVertices{0, 1, 2});
  REQUIRE(m.cell_vertices(1) == Mesh<2>::CellVertices{0, 2, 3});
  REQUIRE(m.cells_with_tag(10) == std::vector<Index>{0, 1});
  REQUIRE(m.facet_tag(m.edge_id(0, 1)) == 1);
  REQUIRE(m.facet_tag(m.edge_id(1, 2)) == 2);
  REQUIRE(m.facet_tag(m.edge_id(2, 3)) == 3);
  REQUIRE(m.facet_tag(m.edge_id(3, 0)) == 4);
  REQUIRE(m.facet_tag(m.edge_id(0, 2)) == kNoTag);
  REQUIRE(m.tag_name(2, 10) == "domain");
  REQUIRE(m.tag_name(1, 3) == "top");
  REQUIRE(m.tag_by_name(1, "left") == std::optional<Tag>{4});

  const Mesh<2> nm = read2(kSquare, 1e-9);
  REQUIRE(nm.vertex(2)(0) == 1e-9);
}

TEST_CASE("Gmsh 4.1: two tetrahedra with a tagged face and physical volume", "[mesh][gmsh]") {
  const Mesh<3> m = read3(kTwoTets);
  REQUIRE(m.num_vertices() == 5);
  REQUIRE(m.num_cells() == 2);
  REQUIRE(m.num_faces() == 7);
  REQUIRE(m.cells_with_tag(20) == std::vector<Index>{0, 1});
  REQUIRE(m.facet_tag(m.face_id(1, 2, 4)) == 7);
  REQUIRE(m.facets_with_tag(7).size() == 1);
  REQUIRE(m.tag_name(3, 20) == "core");
  REQUIRE(m.tag_name(2, 7) == "wall");
  REQUIRE(m.vertex(4) == Point<3>(1.0, 1.0, 1.0));
}

TEST_CASE("Gmsh 4.1: round trip of a generated rectangle with sparse node tags", "[mesh][gmsh]") {
  const Mesh<2> original = rectangle(4, 3, Point<2>(-1.0, 0.5), Point<2>(2.0, 2.5));
  const Mesh<2> m = read2(write_square_mesh(original));
  REQUIRE(m.num_vertices() == original.num_vertices());
  REQUIRE(m.num_cells() == original.num_cells());
  REQUIRE(m.num_edges() == original.num_edges());
  REQUIRE(m.num_boundary_facets() == original.num_boundary_facets());

  // vertices arrive in (reversed) file order; match them by coordinates
  std::vector<Index> to_original(as_size(m.num_vertices()), hpfem::kInvalidIndex);
  for (Index v = 0; v < m.num_vertices(); ++v) {
    for (Index w = 0; w < original.num_vertices(); ++w) {
      if (m.vertex(v) == original.vertex(w)) to_original[as_size(v)] = w;
    }
    REQUIRE(to_original[as_size(v)] != hpfem::kInvalidIndex);
  }
  std::set<std::set<Index>> cells_read;
  std::set<std::set<Index>> cells_orig;
  for (Index c = 0; c < m.num_cells(); ++c) {
    std::set<Index> mapped;
    for (const Index v : m.cell_vertices(c)) mapped.insert(to_original[as_size(v)]);
    cells_read.insert(mapped);
    cells_orig.insert({original.cell_vertices(c).begin(), original.cell_vertices(c).end()});
  }
  REQUIRE(cells_read == cells_orig);
  REQUIRE(m.cells_with_tag(10).size() == as_size(m.num_cells()));
  for (Tag t = 1; t <= 4; ++t) {
    REQUIRE(m.facets_with_tag(t).size() == original.facets_with_tag(t).size());
  }
  for (const Index f : m.boundary_facets()) REQUIRE(m.facet_tag(f) != kNoTag);
}

TEST_CASE("Gmsh: file overload", "[mesh][gmsh]") {
  const auto path = std::filesystem::temp_directory_path() / "hpfem_test_square.msh";
  {
    std::ofstream out(path);
    out << kSquare;
  }
  const Mesh<2> m = read_gmsh<2>(path);
  REQUIRE(m.num_cells() == 2);
  std::filesystem::remove(path);
  REQUIRE_THROWS_WITH(read_gmsh<2>(path), ContainsSubstring("cannot open"));
}

TEST_CASE("Gmsh: unsupported formats and elements are rejected with a reason", "[mesh][gmsh]") {
  REQUIRE_THROWS_AS(read2(replace(kSquare, "4.1 0 8", "2.2 0 8")), hpfem::NotImplemented);
  REQUIRE_THROWS_WITH(read2(replace(kSquare, "4.1 0 8", "4.1 1 8")), ContainsSubstring("ASCII"));
  // triangles declared as 6-node (order 2) but listed with three nodes: garbage follows
  REQUIRE_THROWS_AS(read2(replace(kSquare, "2 1 2 2\n", "2 1 9 2\n")), hpfem::InvalidArgument);
  // triangles declared as quadrangles: element type 3
  REQUIRE_THROWS_AS(read2(replace(kSquare, "2 1 2 2\n", "2 1 3 2\n")), hpfem::InvalidArgument);
  // a 2D file read as a 3D mesh has no tetrahedra
  REQUIRE_THROWS_WITH(read3(kSquare), ContainsSubstring("no tetrahedra"));
  // a 3D file read as a 2D mesh
  REQUIRE_THROWS_WITH(read2(kTwoTets), ContainsSubstring("3-dimensional"));
  // boundary element that is not an edge of the mesh: nodes 2-4 (the missing diagonal)
  REQUIRE_THROWS_WITH(read2(replace(kSquare, "1 1 2\n", "1 2 4\n")),
                      ContainsSubstring("not part of the mesh"));
  // unknown node tag
  REQUIRE_THROWS_WITH(read2(replace(kSquare, "5 1 2 3\n", "5 1 2 9\n")), ContainsSubstring("node"));
  // truncated file
  REQUIRE_THROWS_AS(read2(std::string(kSquare).substr(0, 200)), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(read2("$Nodes\n"), hpfem::InvalidArgument);
}

TEST_CASE("Gmsh: unknown sections are skipped", "[mesh][gmsh]") {
  const std::string with_extra =
      replace(kSquare, "$Nodes\n",
              "$Periodic\n0\n$EndPeriodic\n$Comment\nfree text 1 2 3\n$EndComment\n$Nodes\n");
  REQUIRE(read2(with_extra).num_cells() == 2);
}

namespace {

// one 6-node triangle whose hypotenuse node lies on the unit circle, plus the 3-node line
// of that hypotenuse on an entity with physical tag 7
constexpr const char* kCurvedTriangle = R"(
$MeshFormat
4.1 0 8
$EndMeshFormat
$Entities
0 1 1 0
7 0 0 0 1 1 0 1 7 0
1 0 0 0 1 1 0 0 0
$EndEntities
$Nodes
1 6 1 6
2 1 0 6
1
2
3
4
5
6
0 0 0
1 0 0
0 1 0
0.5 0 0
0.70710678118654752 0.70710678118654752 0
0 0.5 0
$EndNodes
$Elements
2 2 1 2
1 7 8 1
1 2 3 5
2 1 9 1
2 1 2 3 4 5 6
$EndElements
)";

// one 10-node tetrahedron, edge (1, 2) bulged onto the unit sphere (Gmsh node order:
// corners, then edges (0,1) (1,2) (0,2) (0,3) (2,3) (1,3))
constexpr const char* kCurvedTet = R"(
$MeshFormat
4.1 0 8
$EndMeshFormat
$Nodes
1 10 1 10
3 1 0 10
1
2
3
4
5
6
7
8
9
10
0 0 0
1 0 0
0 1 0
0 0 1
0.5 0 0
0.70710678118654752 0.70710678118654752 0
0 0.5 0
0 0 0.5
0 0.5 0.5
0.5 0 0.5
$EndNodes
$Elements
1 1 1 1
3 1 11 1
1 1 2 3 4 5 6 7 8 9 10
$EndElements
)";

}  // namespace

TEST_CASE("Gmsh 4.1: second-order triangles and tetrahedra become edge nodes", "[mesh][gmsh]") {
  const Real s = std::numbers::sqrt2 / 2;
  const Mesh<2> t = read2(kCurvedTriangle);
  REQUIRE(t.num_vertices() == 3);
  REQUIRE(t.num_cells() == 1);
  REQUIRE(t.geometry_order() == 2);
  REQUIRE((t.edge_node(t.edge_id(1, 2)) - Point<2>(s, s)).norm() < 1e-14);
  REQUIRE((t.edge_node(t.edge_id(0, 1)) - Point<2>(0.5, 0.0)).norm() < 1e-14);
  REQUIRE(t.facet_tag(t.edge_id(1, 2)) == 7);  // the 3-node line carries its entity's tag
  const auto geometry = hpfem::mesh::cell_geometry(t, 0);
  REQUIRE((geometry->evaluate(Point<2>(0.5, 0.5)).x - Point<2>(s, s)).norm() < 1e-14);

  const Mesh<3> k = read3(kCurvedTet);
  REQUIRE(k.num_vertices() == 4);
  REQUIRE(k.num_edges() == 6);
  REQUIRE(k.geometry_order() == 2);
  REQUIRE((k.edge_node(k.edge_id(1, 2)) - Point<3>(s, s, 0.0)).norm() < 1e-14);
  REQUIRE((k.edge_node(k.edge_id(0, 3)) - Point<3>(0.0, 0.0, 0.5)).norm() < 1e-14);
  REQUIRE((k.edge_node(k.edge_id(2, 3)) - Point<3>(0.0, 0.5, 0.5)).norm() < 1e-14);
  REQUIRE((k.edge_node(k.edge_id(1, 3)) - Point<3>(0.5, 0.0, 0.5)).norm() < 1e-14);
  // scale applies to edge nodes as well
  std::istringstream in{std::string(kCurvedTet)};
  const Mesh<3> scaled = hpfem::mesh::read_gmsh<3>(in, 2.0);
  REQUIRE((scaled.edge_node(scaled.edge_id(1, 2)) - Point<3>(2 * s, 2 * s, 0.0)).norm() < 1e-14);
}

TEST_CASE("read_gmsh_with_periodic: the $Periodic section gives the pairs and the shift",
          "[mesh][gmsh][periodic]") {
  // right curve (entity 2, physical 2) = left curve (entity 4, physical 4) + (1, 0): the
  // affine transform carries the translation, the node pairs are listed as Gmsh writes them
  const std::string with_affine =
      std::string(kSquare) +
      "$Periodic\n1\n1 2 4\n16 1 0 0 1 0 1 0 0 0 0 1 0 0 0 0 1\n2\n2 1\n3 4\n"
      "$EndPeriodic\n";
  std::istringstream in(with_affine);
  const auto result = hpfem::mesh::read_gmsh_with_periodic<2>(in, 2.0);
  REQUIRE(result.mesh.num_cells() == 2);
  REQUIRE(result.periodic.size() == 1);
  REQUIRE(result.periodic[0].master == 4);
  REQUIRE(result.periodic[0].slave == 2);
  REQUIRE(result.periodic[0].shift(0) == Approx(2.0));  // scaled
  REQUIRE(result.periodic[0].shift(1) == Approx(0.0));
  // without an affine transform the shift comes from the first node pair; point links are
  // ignored; a merged second curve link with the same tags does not duplicate the pair
  const std::string from_nodes =
      std::string(kSquare) +
      "$Periodic\n3\n0 2 1\n0\n1\n2 1\n1 2 4\n0\n2\n2 1\n3 4\n1 2 4\n0\n1\n3 4\n"
      "$EndPeriodic\n";
  std::istringstream in2(from_nodes);
  const auto result2 = hpfem::mesh::read_gmsh_with_periodic<2>(in2, 1.0);
  REQUIRE(result2.periodic.size() == 1);
  REQUIRE(result2.periodic[0].shift(0) == Approx(1.0));
  // an entity without a physical group is an error
  const std::string bad = std::string(kSquare) + "$Periodic\n1\n1 2 9\n0\n1\n2 1\n$EndPeriodic\n";
  std::istringstream in3(bad);
  REQUIRE_THROWS_AS(hpfem::mesh::read_gmsh_with_periodic<2>(in3, 1.0), hpfem::InvalidArgument);
  // plain read_gmsh still skips the section
  std::istringstream in4(with_affine);
  REQUIRE(read_gmsh<2>(in4, 1.0).num_cells() == 2);
}

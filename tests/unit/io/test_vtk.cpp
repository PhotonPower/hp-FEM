#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <fmt/format.h>

#include "hpfem/core/error.hpp"
#include "hpfem/io/vtk.hpp"
#include "hpfem/mesh/generators.hpp"

using Catch::Matchers::ContainsSubstring;
using hpfem::as_size;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::io::VtkFormat;
using hpfem::io::VtkWriter;
using hpfem::io::write_vtu_facets;
using hpfem::mesh::box;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

/// Text between the DataArray tag with the given name and its closing tag.
std::string array_body(const std::string& xml, const std::string& name) {
  const auto tag = xml.find("Name=\"" + name + "\"");
  REQUIRE(tag != std::string::npos);
  const auto start = xml.find('>', tag) + 1;
  const auto end = xml.find("</DataArray>", start);
  return xml.substr(start, end - start);
}

/// Attribute value `attr="..."` of the DataArray with the given name.
std::string array_attribute(const std::string& xml, const std::string& name,
                            const std::string& attr) {
  const auto tag = xml.find("Name=\"" + name + "\"");
  REQUIRE(tag != std::string::npos);
  const auto line_start = xml.rfind('<', tag);
  const auto key = xml.find(attr + "=\"", line_start);
  const auto value_start = key + attr.size() + 2;
  return xml.substr(value_start, xml.find('"', value_start) - value_start);
}

template <class T>
std::vector<T> parse_ascii(const std::string& body) {
  std::istringstream in(body);
  std::vector<T> values;
  T v;
  while (in >> v) values.push_back(v);
  return values;
}

std::vector<std::uint8_t> base64_decode(const std::string& text) {
  const std::string table = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::vector<std::uint8_t> out;
  std::uint32_t buffer = 0;
  int bits = 0;
  for (const char c : text) {
    if (c == '=' || c == '\n' || c == ' ') continue;
    const auto pos = table.find(c);
    REQUIRE(pos != std::string::npos);
    buffer = (buffer << 6U) | static_cast<std::uint32_t>(pos);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push_back(static_cast<std::uint8_t>((buffer >> static_cast<unsigned>(bits)) & 0xFFU));
    }
  }
  return out;
}

/// Decodes a binary DataArray body (UInt64 byte count + raw little-endian values).
template <class T>
std::vector<T> parse_binary(const std::string& body) {
  const auto bytes = base64_decode(body);
  REQUIRE(bytes.size() >= 8);
  std::uint64_t count = 0;
  std::memcpy(&count, bytes.data(), 8);
  REQUIRE(bytes.size() == 8 + count);
  std::vector<T> values(count / sizeof(T));
  std::memcpy(values.data(), bytes.data() + 8, count);
  return values;
}

std::string write_to_string(const VtkWriter<2>& w) {
  std::ostringstream out;
  w.write(out);
  return out.str();
}

}  // namespace

TEST_CASE("VTK: rectangle as ASCII unstructured grid", "[io][vtk]") {
  Mesh<2> m = rectangle(2, 1);
  m.set_cell_tag(3, 7);
  VtkWriter<2> w(m, VtkFormat::kAscii);
  const std::vector<Real> area(as_size(m.num_cells()), 0.25);
  const std::vector<Complex> field{{1.0, -1.0}, {2.0, 0.5}, {0.0, 0.0}, {-3.0, 4.0}};
  w.cell_scalars("area", area).cell_scalars("field", field);
  std::vector<Point<2>> normals(as_size(m.num_vertices()), Point<2>(1.0, 2.0));
  w.point_vectors("normals", normals);
  const std::string xml = write_to_string(w);

  REQUIRE_THAT(xml, ContainsSubstring("<VTKFile type=\"UnstructuredGrid\""));
  REQUIRE_THAT(xml, ContainsSubstring("NumberOfPoints=\"6\" NumberOfCells=\"4\""));
  REQUIRE(array_attribute(xml, "Points", "NumberOfComponents") == "3");
  REQUIRE(array_attribute(xml, "Points", "format") == "ascii");
  const auto points = parse_ascii<Real>(array_body(xml, "Points"));
  REQUIRE(points.size() == 18);
  REQUIRE(points[3] == 0.5);  // vertex 1 = (0.5, 0, 0)
  REQUIRE(points[5] == 0.0);  // z padded
  const auto connectivity = parse_ascii<Index>(array_body(xml, "connectivity"));
  REQUIRE(connectivity.size() == 12);
  REQUIRE(connectivity[0] == m.cell_vertices(0)[0]);
  REQUIRE(connectivity[11] == m.cell_vertices(3)[2]);
  REQUIRE(parse_ascii<Index>(array_body(xml, "offsets")) == std::vector<Index>{3, 6, 9, 12});
  REQUIRE(parse_ascii<int>(array_body(xml, "types")) == std::vector<int>{5, 5, 5, 5});
  REQUIRE(array_attribute(xml, "types", "type") == "UInt8");

  REQUIRE(array_attribute(xml, "cell_tag", "type") == "Int32");
  REQUIRE(parse_ascii<int>(array_body(xml, "cell_tag")) == std::vector<int>{0, 0, 0, 7});
  REQUIRE(parse_ascii<Real>(array_body(xml, "area")) == area);
  REQUIRE(parse_ascii<Real>(array_body(xml, "field_re")) == std::vector<Real>{1.0, 2.0, 0.0, -3.0});
  REQUIRE(parse_ascii<Real>(array_body(xml, "field_im")) == std::vector<Real>{-1.0, 0.5, 0.0, 4.0});
  REQUIRE(array_attribute(xml, "normals", "NumberOfComponents") == "3");
  const auto n = parse_ascii<Real>(array_body(xml, "normals"));
  REQUIRE(n.size() == 18);
  REQUIRE(n[0] == 1.0);
  REQUIRE(n[1] == 2.0);
  REQUIRE(n[2] == 0.0);
}

TEST_CASE("VTK: binary arrays decode to the same values", "[io][vtk]") {
  const Mesh<3> m = box(1, 1, 1);
  VtkWriter<3> w(m, VtkFormat::kBinary);
  std::vector<Index> ids(as_size(m.num_cells()));
  for (Index c = 0; c < m.num_cells(); ++c) ids[as_size(c)] = 10 * c;
  w.cell_scalars("ids", ids);
  std::ostringstream out;
  w.write(out);
  const std::string xml = out.str();

  REQUIRE(array_attribute(xml, "Points", "format") == "binary");
  const auto points = parse_binary<Real>(array_body(xml, "Points"));
  REQUIRE(points.size() == 24);
  for (Index v = 0; v < 8; ++v) {
    for (int d = 0; d < 3; ++d) REQUIRE(points[as_size(3 * v + d)] == m.vertex(v)(d));
  }
  REQUIRE(parse_binary<std::uint8_t>(array_body(xml, "types")) == std::vector<std::uint8_t>(6, 10));
  REQUIRE(parse_binary<Index>(array_body(xml, "ids")) == ids);
  const auto connectivity = parse_binary<Index>(array_body(xml, "connectivity"));
  REQUIRE(connectivity.size() == 24);
  REQUIRE(connectivity[4] == m.cell_vertices(1)[0]);
}

TEST_CASE("VTK: second-order meshes export quadratic cells with edge nodes", "[io][vtk]") {
  Mesh<2> t({{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}}, {{0, 1, 2}});
  std::vector<Point<2>> nodes;
  for (Index e = 0; e < t.num_edges(); ++e) {
    const auto& ev = t.edge_vertices(e);
    nodes.push_back(0.5 * (t.vertex(ev[0]) + t.vertex(ev[1])));
  }
  const Real s = std::numbers::sqrt2 / 2;
  nodes[as_size(t.edge_id(1, 2))] = Point<2>(s, s);
  t.set_edge_nodes(nodes);
  VtkWriter<2> w(t, VtkFormat::kAscii);
  REQUIRE(w.num_points() == 6);
  std::ostringstream out;
  w.write(out);
  const std::string xml = out.str();
  REQUIRE_THAT(xml, ContainsSubstring("NumberOfPoints=\"6\" NumberOfCells=\"1\""));
  REQUIRE(parse_ascii<int>(array_body(xml, "types")) == std::vector<int>{22});
  // VTK quadratic triangle: vertices, then the nodes of edges (0,1), (1,2), (2,0)
  const auto connectivity = parse_ascii<Index>(array_body(xml, "connectivity"));
  REQUIRE(connectivity == std::vector<Index>{0, 1, 2, 3 + t.edge_id(0, 1), 3 + t.edge_id(1, 2),
                                             3 + t.edge_id(0, 2)});
  const auto points = parse_ascii<Real>(array_body(xml, "Points"));
  const auto arc = as_size(3 * (3 + t.edge_id(1, 2)));
  REQUIRE(points[arc] == s);
  REQUIRE(points[arc + 1] == s);
  // point data must cover vertices and edge nodes
  REQUIRE_THROWS_AS(w.point_scalars("p", std::vector<Real>(3)), hpfem::InvalidArgument);
  w.point_scalars("p", std::vector<Real>(6, 1.0));

  // quadratic tetrahedron: VTK edge order (0,1) (1,2) (0,2) (0,3) (1,3) (2,3)
  Mesh<3> tet({{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}, {{0, 1, 2, 3}});
  std::vector<Point<3>> tet_nodes;
  for (Index e = 0; e < tet.num_edges(); ++e) {
    const auto& ev = tet.edge_vertices(e);
    tet_nodes.push_back(0.5 * (tet.vertex(ev[0]) + tet.vertex(ev[1])));
  }
  tet.set_edge_nodes(tet_nodes);
  std::ostringstream out3;
  VtkWriter<3>(tet, VtkFormat::kAscii).write(out3);
  const std::string xml3 = out3.str();
  REQUIRE(parse_ascii<int>(array_body(xml3, "types")) == std::vector<int>{24});
  const auto c3 = parse_ascii<Index>(array_body(xml3, "connectivity"));
  REQUIRE(c3 == std::vector<Index>{0, 1, 2, 3, 4 + tet.edge_id(0, 1), 4 + tet.edge_id(1, 2),
                                   4 + tet.edge_id(0, 2), 4 + tet.edge_id(0, 3),
                                   4 + tet.edge_id(1, 3), 4 + tet.edge_id(2, 3)});
}

TEST_CASE("VTK: size mismatches are rejected, files are written", "[io][vtk]") {
  const Mesh<2> m = rectangle(2, 2);
  VtkWriter<2> w(m, VtkFormat::kAscii);
  REQUIRE_THROWS_WITH(w.cell_scalars("x", std::vector<Real>(3)), ContainsSubstring("8 cells"));
  REQUIRE_THROWS_AS(w.cell_vectors("v", std::vector<Point<2>>(1)), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(w.point_scalars("p", std::vector<Real>(8)), hpfem::InvalidArgument);

  const auto path = std::filesystem::temp_directory_path() / "hpfem_test_mesh.vtu";
  w.write(path);
  REQUIRE(std::filesystem::file_size(path) > 100);
  std::ifstream in(path);
  std::string first;
  std::getline(in, first);
  REQUIRE(first == "<?xml version=\"1.0\"?>");
  in.close();
  std::filesystem::remove(path);
  REQUIRE_THROWS_AS(w.write(std::filesystem::path("/nonexistent_dir_hpfem/x.vtu")),
                    hpfem::InvalidArgument);
}

TEST_CASE("VTK: facets with tags", "[io][vtk]") {
  const Mesh<2> m = rectangle(3, 2);
  std::ostringstream out;
  write_vtu_facets(m, out, true, VtkFormat::kAscii);
  const std::string xml = out.str();
  REQUIRE_THAT(xml, ContainsSubstring("NumberOfPoints=\"12\" NumberOfCells=\"10\""));
  REQUIRE(parse_ascii<int>(array_body(xml, "types")) == std::vector<int>(10, 3));
  const auto tags = parse_ascii<int>(array_body(xml, "facet_tag"));
  REQUIRE(tags.size() == 10);
  REQUIRE(std::count(tags.begin(), tags.end(), box_tag::kYMin) == 3);
  REQUIRE(std::count(tags.begin(), tags.end(), box_tag::kXMin) == 2);
  REQUIRE(parse_ascii<int>(array_body(xml, "is_boundary")) == std::vector<int>(10, 1));
  const auto ids = parse_ascii<Index>(array_body(xml, "facet_index"));
  for (std::size_t i = 0; i < ids.size(); ++i) REQUIRE(ids[i] == m.boundary_facets()[i]);

  std::ostringstream all;
  write_vtu_facets(m, all, false, VtkFormat::kAscii);
  REQUIRE_THAT(all.str(), ContainsSubstring(fmt::format("NumberOfCells=\"{}\"", m.num_facets())));

  const Mesh<3> b = box(1, 1, 1);
  std::ostringstream out3;
  write_vtu_facets(b, out3, true, VtkFormat::kBinary);
  REQUIRE(parse_binary<std::uint8_t>(array_body(out3.str(), "types")) ==
          std::vector<std::uint8_t>(12, 5));
}

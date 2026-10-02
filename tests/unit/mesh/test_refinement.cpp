#include <array>
#include <cmath>
#include <numbers>
#include <set>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/refinement.hpp"

using Catch::Approx;
using hpfem::as_size;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::mesh::affine_map;
using hpfem::mesh::box;
using hpfem::mesh::cell_geometry;
using hpfem::mesh::kNoTag;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::mesh::refine_uniform;
using hpfem::mesh::Refined;
using hpfem::mesh::Tag;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

template <int Dim>
Real total_volume(const Mesh<Dim>& m) {
  Real v = 0;
  for (Index c = 0; c < m.num_cells(); ++c) v += affine_map(m, c).volume();
  return v;
}

template <int Dim>
Index euler_characteristic(const Mesh<Dim>& m) {
  if constexpr (Dim == 2) {
    return m.num_vertices() - m.num_edges() + m.num_cells();
  } else {
    return m.num_vertices() - m.num_edges() + m.num_faces() - m.num_cells();
  }
}

/// Children of parent c cover exactly the parent's volume and keep its tag.
template <int Dim>
void check_children(const Mesh<Dim>& parent, const Refined<Dim>& r) {
  REQUIRE(r.mesh.num_cells() == Refined<Dim>::kChildren * parent.num_cells());
  REQUIRE(r.mesh.num_vertices() == parent.num_vertices() + parent.num_edges());
  REQUIRE(static_cast<Index>(r.parent_cell.size()) == r.mesh.num_cells());
  REQUIRE(static_cast<Index>(r.edge_vertex.size()) == parent.num_edges());
  for (Index v = 0; v < parent.num_vertices(); ++v) REQUIRE(r.mesh.vertex(v) == parent.vertex(v));
  for (Index c = 0; c < parent.num_cells(); ++c) {
    Real volume = 0;
    for (Index i = 0; i < Refined<Dim>::kChildren; ++i) {
      const Index child = Refined<Dim>::kChildren * c + i;
      REQUIRE(r.parent_cell[as_size(child)] == c);
      REQUIRE(r.mesh.cell_tag(child) == parent.cell_tag(c));
      volume += affine_map(r.mesh, child).volume();
    }
    REQUIRE(volume == Approx(affine_map(parent, c).volume()));
  }
  REQUIRE(euler_characteristic(r.mesh) == euler_characteristic(parent));
}

}  // namespace

TEST_CASE("red refinement of one triangle", "[mesh][refinement]") {
  const Mesh<2> t({{0.0, 0.0}, {2.0, 0.0}, {0.0, 2.0}}, {{0, 1, 2}});
  const Refined<2> r = refine_uniform(t);
  REQUIRE(r.mesh.num_vertices() == 6);
  REQUIRE(r.mesh.num_edges() == 9);
  REQUIRE(r.mesh.num_cells() == 4);
  REQUIRE(r.mesh.num_boundary_facets() == 6);
  for (Index e = 0; e < t.num_edges(); ++e) {
    const auto& ev = t.edge_vertices(e);
    REQUIRE(r.mesh.vertex(r.edge_vertex[as_size(e)]) == 0.5 * (t.vertex(ev[0]) + t.vertex(ev[1])));
  }
  for (Index c = 0; c < 4; ++c) {
    const auto map = affine_map(r.mesh, c);
    REQUIRE(map.volume() == Approx(0.5));  // a quarter of the parent
    REQUIRE(map.det > 0);                  // orientation preserved
    REQUIRE(map.h == Approx(std::numbers::sqrt2));
  }
  check_children(t, r);
}

TEST_CASE("refined rectangle equals the finer rectangle in all counts and tags",
          "[mesh][refinement]") {
  const Mesh<2> coarse = rectangle(3, 2, Point<2>(0.0, 0.0), Point<2>(3.0, 2.0));
  const Mesh<2> fine = rectangle(6, 4, Point<2>(0.0, 0.0), Point<2>(3.0, 2.0));
  const Refined<2> r = refine_uniform(coarse);
  REQUIRE(r.mesh.num_vertices() == fine.num_vertices());
  REQUIRE(r.mesh.num_edges() == fine.num_edges());
  REQUIRE(r.mesh.num_cells() == fine.num_cells());
  REQUIRE(r.mesh.num_boundary_facets() == fine.num_boundary_facets());
  for (const Tag t : {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax}) {
    REQUIRE(r.mesh.facets_with_tag(t).size() == fine.facets_with_tag(t).size());
  }
  REQUIRE(r.mesh.facets_with_tag(kNoTag).size() == fine.facets_with_tag(kNoTag).size());
  // same vertex set
  std::set<std::array<Real, 2>> a;
  std::set<std::array<Real, 2>> b;
  for (Index v = 0; v < fine.num_vertices(); ++v) {
    a.insert({r.mesh.vertex(v)(0), r.mesh.vertex(v)(1)});
    b.insert({fine.vertex(v)(0), fine.vertex(v)(1)});
  }
  REQUIRE(a == b);
  check_children(coarse, r);
}

TEST_CASE("red refinement of one tetrahedron (Bey)", "[mesh][refinement]") {
  const Mesh<3> t({{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}},
                  {{0, 1, 2, 3}});
  const Refined<3> r = refine_uniform(t);
  REQUIRE(r.mesh.num_vertices() == 10);
  REQUIRE(r.mesh.num_cells() == 8);
  REQUIRE(r.mesh.num_boundary_facets() == 16);
  REQUIRE(euler_characteristic(r.mesh) == 1);
  for (Index c = 0; c < 8; ++c) REQUIRE(affine_map(r.mesh, c).volume() == Approx(1.0 / 48.0));
  REQUIRE(total_volume(r.mesh) == Approx(1.0 / 6.0));
  check_children(t, r);
}

TEST_CASE("refined box: counts, side tags, repeated refinement", "[mesh][refinement]") {
  Mesh<3> coarse = box(1, 2, 1, Point<3>(0.0, 0.0, 0.0), Point<3>(1.0, 2.0, 1.0));
  for (Index c = 0; c < coarse.num_cells(); ++c) coarse.set_cell_tag(c, static_cast<Tag>(c % 3));
  coarse.set_tag_name(3, 1, "core");
  coarse.set_tag_name(2, box_tag::kZMax, "top");
  const Refined<3> r = refine_uniform(coarse);
  REQUIRE(r.mesh.num_vertices() == 3 * 5 * 3);
  REQUIRE(r.mesh.num_cells() == 8 * coarse.num_cells());
  REQUIRE(r.mesh.num_boundary_facets() == 4 * coarse.num_boundary_facets());
  REQUIRE(total_volume(r.mesh) == Approx(2.0));
  for (const Tag t : {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax,
                      box_tag::kZMin, box_tag::kZMax}) {
    REQUIRE(r.mesh.facets_with_tag(t).size() == 4 * coarse.facets_with_tag(t).size());
    for (const Index f : r.mesh.facets_with_tag(t)) REQUIRE(r.mesh.is_boundary_facet(f));
  }
  REQUIRE(r.mesh.tag_name(3, 1) == "core");
  REQUIRE(r.mesh.tag_name(2, box_tag::kZMax) == "top");
  check_children(coarse, r);

  const Refined<3> rr = refine_uniform(r.mesh);
  REQUIRE(rr.mesh.num_cells() == 64 * coarse.num_cells());
  REQUIRE(rr.mesh.num_boundary_facets() == 16 * coarse.num_boundary_facets());
  REQUIRE(total_volume(rr.mesh) == Approx(2.0));
  check_children(r.mesh, rr);
}

TEST_CASE("refinement preserves second-order geometry", "[mesh][refinement]") {
  // quarter disc as one curved triangle (see test_curved_geometry.cpp)
  Mesh<2> t({{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}}, {{0, 1, 2}});
  std::vector<Point<2>> nodes;
  for (Index e = 0; e < t.num_edges(); ++e) {
    const auto& ev = t.edge_vertices(e);
    nodes.push_back(0.5 * (t.vertex(ev[0]) + t.vertex(ev[1])));
  }
  const Real s = std::numbers::sqrt2 / 2;
  nodes[as_size(t.edge_id(1, 2))] = Point<2>(s, s);
  t.set_edge_nodes(nodes);
  const Real expected_area = 4.0 * s / 3.0 - 1.0 / 6.0;

  const Refined<2> r = refine_uniform(t);
  REQUIRE(r.mesh.geometry_order() == 2);
  // the new vertex of the hypotenuse is the parent edge node, i.e. on the arc
  REQUIRE((r.mesh.vertex(r.edge_vertex[as_size(t.edge_id(1, 2))]) - Point<2>(s, s)).norm() < 1e-14);
  // the straight legs stay straight: their edge nodes are midpoints
  for (Index e = 0; e < r.mesh.num_edges(); ++e) {
    const auto& ev = r.mesh.edge_vertices(e);
    const Point<2> mid = 0.5 * (r.mesh.vertex(ev[0]) + r.mesh.vertex(ev[1]));
    const bool on_leg = (r.mesh.vertex(ev[0])(0) == 0.0 && r.mesh.vertex(ev[1])(0) == 0.0) ||
                        (r.mesh.vertex(ev[0])(1) == 0.0 && r.mesh.vertex(ev[1])(1) == 0.0);
    if (on_leg) REQUIRE((r.mesh.edge_node(e) - mid).norm() < 1e-14);
  }
  // the four children describe the same P2 region: the degree-2 midpoint rule is exact
  const std::array<Point<2>, 3> quadrature{Point<2>(0.5, 0.0), Point<2>(0.5, 0.5),
                                           Point<2>(0.0, 0.5)};
  Real area = 0;
  for (Index c = 0; c < r.mesh.num_cells(); ++c) {
    const auto geometry = cell_geometry(r.mesh, c);
    for (const auto& q : quadrature) area += geometry->evaluate(q).det / 6.0;
  }
  REQUIRE(area == Approx(expected_area));
  // and again after a second refinement
  const Refined<2> rr = refine_uniform(r.mesh);
  area = 0;
  for (Index c = 0; c < rr.mesh.num_cells(); ++c) {
    const auto geometry = cell_geometry(rr.mesh, c);
    for (const auto& q : quadrature) area += geometry->evaluate(q).det / 6.0;
  }
  REQUIRE(area == Approx(expected_area));
}

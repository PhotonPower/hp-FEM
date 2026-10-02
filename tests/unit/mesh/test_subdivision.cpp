#include <cmath>
#include <numbers>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/core/error.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/subdivision.hpp"

using Catch::Approx;
using hpfem::as_size;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::mesh::affine_map;
using hpfem::mesh::box;
using hpfem::mesh::cell_geometry;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::mesh::subdivide;
using hpfem::mesh::Subdivided;
using hpfem::mesh::Tag;

namespace {

constexpr Real kTol = 1e-13;

/// Counts, equal sub-volumes with the parent's orientation, sub-vertices on the parent map,
/// inherited tags.
template <int Dim>
void check_subdivision(const Mesh<Dim>& parent, int n) {
  const Subdivided<Dim> sub = subdivide(parent, n);
  const Index vpc = Subdivided<Dim>::vertices_per_cell(n);
  const Index cpc = Subdivided<Dim>::cells_per_cell(n);
  REQUIRE(sub.mesh.num_vertices() == parent.num_cells() * vpc);
  REQUIRE(sub.mesh.num_cells() == parent.num_cells() * cpc);
  REQUIRE(static_cast<Index>(sub.parent_cell.size()) == sub.mesh.num_cells());
  REQUIRE(static_cast<Index>(sub.vertex_parent.size()) == sub.mesh.num_vertices());
  REQUIRE(static_cast<Index>(sub.vertex_xi.size()) == sub.mesh.num_vertices());
  REQUIRE(sub.mesh.geometry_order() == 1);
  for (Index c = 0; c < parent.num_cells(); ++c) {
    const auto geometry = cell_geometry(parent, c);
    const Real parent_det = geometry->evaluate(Point<Dim>::Constant(0.25)).det;
    for (Index k = 0; k < cpc; ++k) {
      const Index s = c * cpc + k;
      REQUIRE(sub.parent_cell[as_size(s)] == c);
      REQUIRE(sub.mesh.cell_tag(s) == parent.cell_tag(c));
      const auto map = affine_map(sub.mesh, s);
      if (geometry->is_affine()) {
        REQUIRE(map.volume() == Approx(affine_map(parent, c).volume() / static_cast<Real>(cpc)));
      }
      INFO("n = " << n << ", sub-cell " << k);
      REQUIRE(map.det * parent_det > 0);
    }
    for (Index k = 0; k < vpc; ++k) {
      const Index v = c * vpc + k;
      REQUIRE(sub.vertex_parent[as_size(v)] == c);
      const Point<Dim>& xi = sub.vertex_xi[as_size(v)];
      Real sum = 0;
      for (int d = 0; d < Dim; ++d) {
        REQUIRE(xi(d) >= 0);
        sum += xi(d);
      }
      REQUIRE(sum <= 1 + kTol);
      REQUIRE((geometry->evaluate(xi).x - sub.mesh.vertex(v)).norm() < kTol);
    }
  }
  // every sub-vertex is a lattice point: n xi is integer
  for (const auto& xi : sub.vertex_xi) {
    for (int d = 0; d < Dim; ++d) {
      const Real scaled = xi(d) * n;
      REQUIRE(std::abs(scaled - std::round(scaled)) < kTol);
    }
  }
}

template <int Dim>
Mesh<Dim> with_tags(const Mesh<Dim>& m) {
  std::vector<typename Mesh<Dim>::Vertex> vertices(m.vertices().begin(), m.vertices().end());
  std::vector<typename Mesh<Dim>::CellVertices> cells(m.cells().begin(), m.cells().end());
  std::vector<Tag> tags;
  for (Index c = 0; c < m.num_cells(); ++c) tags.push_back(static_cast<Tag>(c % 3 + 1));
  return Mesh<Dim>(std::move(vertices), std::move(cells), std::move(tags));
}

}  // namespace

TEST_CASE("subdivide: counts, volumes, orientation, tags (2D and 3D)", "[mesh][subdivision]") {
  REQUIRE(Subdivided<2>::vertices_per_cell(1) == 3);
  REQUIRE(Subdivided<2>::vertices_per_cell(3) == 10);
  REQUIRE(Subdivided<2>::cells_per_cell(3) == 9);
  REQUIRE(Subdivided<3>::vertices_per_cell(1) == 4);
  REQUIRE(Subdivided<3>::vertices_per_cell(2) == 10);
  REQUIRE(Subdivided<3>::vertices_per_cell(3) == 20);
  REQUIRE(Subdivided<3>::cells_per_cell(3) == 27);
  const Mesh<2> r = with_tags(rectangle(3, 2, Point<2>(-1.0, 0.0), Point<2>(1.0, 1.5)));
  for (int n = 1; n <= 4; ++n) check_subdivision(r, n);
  const Mesh<3> b = with_tags(box(2, 1, 2, Point<3>(0.0, 0.0, 0.0), Point<3>(2.0, 1.0, 1.0)));
  for (int n = 1; n <= 4; ++n) check_subdivision(b, n);
  REQUIRE_THROWS_AS(subdivide(r, 0), hpfem::InvalidArgument);
}

TEST_CASE("subdivide: n = 1 reproduces the cells with duplicated vertices", "[mesh][subdivision]") {
  const Mesh<2> r = rectangle(2, 2);
  const Subdivided<2> sub = subdivide(r, 1);
  REQUIRE(sub.mesh.num_cells() == r.num_cells());
  REQUIRE(sub.mesh.num_vertices() == 3 * r.num_cells());
  for (Index c = 0; c < r.num_cells(); ++c) {
    for (std::size_t i = 0; i < 3; ++i) {
      const Index v = sub.mesh.cell_vertices(c)[i];
      REQUIRE((sub.mesh.vertex(v) - r.vertex(r.cell_vertices(c)[i])).norm() < kTol);
    }
  }
  // no sub-cell shares a vertex with a sub-cell of another parent
  REQUIRE(sub.mesh.boundary_facets().size() == as_size(3 * r.num_cells()));
}

TEST_CASE("subdivide: curved cell becomes piecewise straight on the parent map",
          "[mesh][subdivision]") {
  Mesh<2> m({{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}}, {{0, 1, 2}});
  const Real s = std::numbers::sqrt2 / 2;
  // global edges are numbered (0,1), (0,2), (1,2): the hypotenuse is edge_id(1, 2)
  std::vector<Point<2>> nodes;
  for (Index e = 0; e < m.num_edges(); ++e) {
    const auto& ev = m.edge_vertices(e);
    nodes.push_back(0.5 * (m.vertex(ev[0]) + m.vertex(ev[1])));
  }
  nodes[as_size(m.edge_id(1, 2))] = Point<2>(s, s);
  m.set_edge_nodes(nodes);
  check_subdivision(m, 2);
  check_subdivision(m, 3);
  const Subdivided<2> sub = subdivide(m, 2);
  // the lattice point xi = (1/2, 1/2) is the hypotenuse node on the arc
  bool found = false;
  for (Index v = 0; v < sub.mesh.num_vertices(); ++v) {
    if ((sub.vertex_xi[as_size(v)] - Point<2>(0.5, 0.5)).norm() < kTol) {
      found = true;
      REQUIRE((sub.mesh.vertex(v) - Point<2>(s, s)).norm() < kTol);
    }
  }
  REQUIRE(found);
  // the sub-mesh area approaches the P2 quarter disc 4 sqrt 2 / 3 - 1/6 from below
  Real area2 = 0;
  for (Index c = 0; c < sub.mesh.num_cells(); ++c) area2 += affine_map(sub.mesh, c).volume();
  Real area8 = 0;
  const Subdivided<2> fine = subdivide(m, 8);
  for (Index c = 0; c < fine.mesh.num_cells(); ++c) area8 += affine_map(fine.mesh, c).volume();
  const Real exact = 4.0 * s / 3.0 - 1.0 / 6.0;
  REQUIRE(area2 < exact);
  REQUIRE(area8 < exact);
  REQUIRE(exact - area8 < exact - area2);
  REQUIRE(area8 == Approx(exact).epsilon(1e-2));
}

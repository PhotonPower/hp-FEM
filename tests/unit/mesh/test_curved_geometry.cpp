#include <array>
#include <cmath>
#include <numbers>
#include <random>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/core/error.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"

using Catch::Approx;
using hpfem::as_size;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::mesh::affine_map;
using hpfem::mesh::AffineGeometry;
using hpfem::mesh::box;
using hpfem::mesh::cell_geometry;
using hpfem::mesh::CellGeometry;
using hpfem::mesh::GeometryPoint;
using hpfem::mesh::Mesh;
using hpfem::mesh::QuadraticGeometry;
using hpfem::mesh::rectangle;

namespace {

constexpr Real kTol = 1e-12;

/// Random points inside the reference simplex.
template <int Dim>
std::vector<Point<Dim>> reference_points(unsigned seed, int count) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<Real> u(0.0, 1.0);
  std::vector<Point<Dim>> points;
  while (static_cast<int>(points.size()) < count) {
    Point<Dim> xi;
    Real sum = 0;
    for (int d = 0; d < Dim; ++d) {
      xi(d) = u(rng);
      sum += xi(d);
    }
    if (sum <= 1.0) points.push_back(xi);
  }
  return points;
}

/// Straight edge midpoints of every edge of the mesh.
template <int Dim>
std::vector<Point<Dim>> midpoints(const Mesh<Dim>& m) {
  std::vector<Point<Dim>> nodes;
  for (Index e = 0; e < m.num_edges(); ++e) {
    const auto& ev = m.edge_vertices(e);
    nodes.push_back(0.5 * (m.vertex(ev[0]) + m.vertex(ev[1])));
  }
  return nodes;
}

template <int Dim>
void check_partition_of_unity(unsigned seed) {
  for (const auto& xi : reference_points<Dim>(seed, 20)) {
    typename QuadraticGeometry<Dim>::ShapeValues n{};
    typename QuadraticGeometry<Dim>::ShapeGradients dn{};
    QuadraticGeometry<Dim>::shape_functions(xi, n, dn);
    Real sum = 0;
    Point<Dim> grad_sum = Point<Dim>::Zero();
    for (std::size_t i = 0; i < n.size(); ++i) {
      sum += n[i];
      grad_sum += dn[i];
    }
    REQUIRE(sum == Approx(1.0).margin(kTol));
    REQUIRE(grad_sum.norm() < kTol);
  }
  // Kronecker property at the nodes: vertices e_i, edge nodes at midpoints
  typename QuadraticGeometry<Dim>::ShapeValues n{};
  typename QuadraticGeometry<Dim>::ShapeGradients dn{};
  QuadraticGeometry<Dim>::shape_functions(Point<Dim>::Zero(), n, dn);
  REQUIRE(n[0] == Approx(1.0));
  for (std::size_t i = 1; i < n.size(); ++i) REQUIRE(n[i] == Approx(0.0).margin(kTol));
}

/// A mesh with midpoint edge nodes must reproduce the affine map exactly.
template <int Dim>
void check_midpoints_reproduce_affine(const Mesh<Dim>& straight, unsigned seed) {
  Mesh<Dim> m = straight;
  m.set_edge_nodes(midpoints(m));
  REQUIRE(m.geometry_order() == 2);
  for (Index c = 0; c < m.num_cells(); ++c) {
    const auto geometry = cell_geometry(m, c);
    REQUIRE(geometry->order() == 2);
    REQUIRE_FALSE(geometry->is_affine());
    const auto affine = affine_map(straight, c);
    REQUIRE(geometry->h() == Approx(affine.h));
    for (const auto& xi : reference_points<Dim>(seed, 5)) {
      const GeometryPoint<Dim> g = geometry->evaluate(xi);
      REQUIRE((g.x - affine.to_physical(xi)).norm() < kTol * affine.h);
      REQUIRE((g.jacobian - affine.jacobian).norm() < kTol * affine.h);
      REQUIRE(g.det == Approx(affine.det));
      REQUIRE((g.inverse_transpose - affine.inverse_transpose).norm() < kTol / affine.h);
      REQUIRE((geometry->to_reference(g.x) - xi).norm() < 1e-10);
    }
  }
}

}  // namespace

TEST_CASE("quadratic shape functions: partition of unity and Kronecker property",
          "[mesh][geometry]") {
  check_partition_of_unity<2>(1);
  check_partition_of_unity<3>(2);
}

TEST_CASE("cell_geometry dispatches on the mesh geometry order", "[mesh][geometry]") {
  Mesh<2> m = rectangle(2, 2);
  REQUIRE(m.geometry_order() == 1);
  const auto affine = cell_geometry(m, 3);
  REQUIRE(affine->order() == 1);
  REQUIRE(affine->is_affine());
  REQUIRE(dynamic_cast<const AffineGeometry<2>*>(affine.get()) != nullptr);
  const auto map = affine_map(m, 3);
  const GeometryPoint<2> g = affine->evaluate(Point<2>(0.25, 0.25));
  REQUIRE((g.x - map.to_physical(Point<2>(0.25, 0.25))).norm() < kTol);
  REQUIRE(g.det == map.det);
  REQUIRE(affine->h() == map.h);

  REQUIRE_THROWS_AS(m.set_edge_nodes(std::vector<Point<2>>(3)), hpfem::InvalidArgument);
  m.set_edge_nodes(midpoints(m));
  REQUIRE(m.geometry_order() == 2);
  REQUIRE(m.edge_node(0) ==
          0.5 * (m.vertex(m.edge_vertices(0)[0]) + m.vertex(m.edge_vertices(0)[1])));
  REQUIRE(dynamic_cast<const QuadraticGeometry<2>*>(cell_geometry(m, 3).get()) != nullptr);
  m.set_edge_nodes({});
  REQUIRE(m.geometry_order() == 1);
}

TEST_CASE("midpoint edge nodes reproduce the affine map (2D and 3D)", "[mesh][geometry]") {
  check_midpoints_reproduce_affine(rectangle(2, 3, Point<2>(-1.0, 0.0), Point<2>(1.0, 3.0)), 3);
  check_midpoints_reproduce_affine(box(1, 2, 1, Point<3>(0.0, 0.0, 0.0), Point<3>(2.0, 1.0, 0.5)),
                                   4);
}

TEST_CASE("one curved triangle approximates a quarter disc", "[mesh][geometry]") {
  // vertices (0,0), (1,0), (0,1); the node of the hypotenuse is moved onto the unit circle
  Mesh<2> m({{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}}, {{0, 1, 2}});
  auto nodes = midpoints(m);
  const Real s = std::numbers::sqrt2 / 2;
  nodes[as_size(m.edge_id(1, 2))] = Point<2>(s, s);
  m.set_edge_nodes(nodes);
  const auto geometry = cell_geometry(m, 0);
  REQUIRE(geometry->order() == 2);

  // the mapped midpoint of the hypotenuse lies on the arc
  const GeometryPoint<2> mid = geometry->evaluate(Point<2>(0.5, 0.5));
  REQUIRE((mid.x - Point<2>(s, s)).norm() < kTol);
  REQUIRE(mid.x.norm() == Approx(1.0));
  // the straight legs are unchanged
  REQUIRE((geometry->evaluate(Point<2>(0.3, 0.0)).x - Point<2>(0.3, 0.0)).norm() < kTol);
  REQUIRE((geometry->evaluate(Point<2>(0.0, 0.7)).x - Point<2>(0.0, 0.7)).norm() < kTol);

  // det J is a quadratic polynomial, so the 3-point edge-midpoint rule integrates it
  // exactly: the area of the P2 quarter disc is 4 sqrt(2)/3 - 1/6 (Green's theorem along
  // the parabolic arc), slightly below pi/4
  const std::array<Point<2>, 3> quadrature{Point<2>(0.5, 0.0), Point<2>(0.5, 0.5),
                                           Point<2>(0.0, 0.5)};
  Real area = 0;
  for (const auto& q : quadrature) area += geometry->evaluate(q).det / 6.0;
  REQUIRE(area == Approx(4.0 * s / 3.0 - 1.0 / 6.0));
  REQUIRE(area < std::numbers::pi / 4);
  REQUIRE(area > 0.5);

  // positive Jacobian everywhere and Newton inversion round trips
  for (const auto& xi : reference_points<2>(5, 50)) {
    const GeometryPoint<2> g = geometry->evaluate(xi);
    REQUIRE(g.det > 0);
    REQUIRE((g.inverse_transpose * g.jacobian.transpose() - Eigen::Matrix2d::Identity()).norm() <
            kTol);
    REQUIRE((geometry->to_reference(g.x) - xi).norm() < 1e-10);
  }
  REQUIRE_THROWS_AS(geometry->to_reference(Point<2>(1e6, -1e6)), hpfem::Error);
}

TEST_CASE("one curved tetrahedron with a bulged edge", "[mesh][geometry]") {
  Mesh<3> m({{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}, {{0, 1, 2, 3}});
  auto nodes = midpoints(m);
  const Real s = std::numbers::sqrt2 / 2;
  nodes[as_size(m.edge_id(1, 2))] = Point<3>(s, s, 0.0);  // onto the unit sphere
  m.set_edge_nodes(nodes);
  const auto geometry = cell_geometry(m, 0);

  const GeometryPoint<3> mid = geometry->evaluate(Point<3>(0.5, 0.5, 0.0));
  REQUIRE((mid.x - Point<3>(s, s, 0.0)).norm() < kTol);
  // the face z = 0 is the curved triangle of the 2D test; det J is cubic in 3D, so instead
  // of an exact rule check positivity, inversion and that the cell grew
  for (const auto& xi : reference_points<3>(6, 50)) {
    const GeometryPoint<3> g = geometry->evaluate(xi);
    REQUIRE(g.det > 0);
    REQUIRE((geometry->to_reference(g.x) - xi).norm() < 1e-10);
  }
  // the volume grows compared with the straight tetrahedron (1/6): sample the mean det
  Real mean_det = 0;
  const auto sample = reference_points<3>(7, 2000);
  for (const auto& xi : sample) mean_det += geometry->evaluate(xi).det;
  mean_det /= static_cast<Real>(sample.size());
  REQUIRE(mean_det > 1.0);
}

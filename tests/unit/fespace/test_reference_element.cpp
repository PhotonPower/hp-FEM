#include <cmath>
#include <numbers>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/core/error.hpp"
#include "hpfem/fespace/reference_element.hpp"
#include "hpfem/mesh/simplex_topology.hpp"

using Catch::Approx;
using hpfem::LocalIndex;
using hpfem::Point;
using hpfem::Real;
using hpfem::fespace::ReferenceElement;
using hpfem::mesh::SimplexTopology;

namespace {

constexpr Real kTol = 1e-14;

template <int Dim>
void check_common() {
  using R = ReferenceElement<Dim>;
  STATIC_REQUIRE(R::kNumVertices == Dim + 1);
  STATIC_REQUIRE(R::kNumFacets == Dim + 1);
  STATIC_REQUIRE(R::kNumEdges == SimplexTopology<Dim>::kNumEdges);
  REQUIRE(&R::kEdgeVertices == &SimplexTopology<Dim>::kEdgeVertices);  // same table, not a copy
  REQUIRE(&R::kFacetVertices == &SimplexTopology<Dim>::kFacetVertices);

  // vertices and barycentric coordinates
  REQUIRE(R::vertex(0) == Point<Dim>::Zero());
  for (LocalIndex i = 0; i <= Dim; ++i) {
    const auto lambda = R::barycentric(R::vertex(i));
    for (LocalIndex j = 0; j <= Dim; ++j) {
      REQUIRE(lambda[static_cast<std::size_t>(j)] == (i == j ? 1.0 : 0.0));
    }
    REQUIRE(R::contains(R::vertex(i)));
  }
  const auto lambda = R::barycentric(R::centroid());
  for (const Real l : lambda) REQUIRE(l == Approx(1.0 / (Dim + 1)));
  const auto grad = R::barycentric_gradients();
  Point<Dim> sum = Point<Dim>::Zero();
  for (const auto& g : grad) sum += g;
  REQUIRE(sum.norm() < kTol);
  for (LocalIndex i = 1; i <= Dim; ++i) {
    REQUIRE(grad[static_cast<std::size_t>(i)](i - 1) == 1.0);  // d lambda_i / d xi_i = 1
  }
  REQUIRE_FALSE(R::contains(Point<Dim>::Constant(1.0)));
  REQUIRE_FALSE(R::contains(-1e-6 * Point<Dim>::Ones()));
  REQUIRE(R::contains(-1e-6 * Point<Dim>::Ones(), 1e-5));

  // facets: outward unit normals, measures summing to the surface, opposite vertices
  Real surface = 0;
  for (LocalIndex k = 0; k < R::kNumFacets; ++k) {
    const Point<Dim> n = R::facet_normal(k);
    REQUIRE(n.norm() == Approx(1.0));
    const auto& fv = R::kFacetVertices[static_cast<std::size_t>(k)];
    for (const LocalIndex v : fv) {
      REQUIRE(std::find(fv.begin(), fv.end(), R::opposite_vertex(k)) == fv.end());
      // the normal is orthogonal to the facet and points away from the opposite vertex
      REQUIRE(std::abs(n.dot(R::vertex(v) - R::vertex(fv[0]))) < kTol);
    }
    REQUIRE(n.dot(R::vertex(fv[0]) - R::vertex(R::opposite_vertex(k))) > 0);
    surface += R::facet_measure(k);
    // facet_point maps the reference facet vertices onto the facet vertices in order
    for (int j = 0; j < Dim - 1; ++j) {
      Point<Dim - 1> eta = Point<Dim - 1>::Zero();
      eta(j) = 1.0;
      REQUIRE((R::facet_point(k, eta) - R::vertex(fv[static_cast<std::size_t>(j + 1)])).norm() <
              kTol);
    }
    REQUIRE((R::facet_point(k, Point<Dim - 1>::Zero()) - R::vertex(fv[0])).norm() < kTol);
  }
  const Real expected_surface = Dim == 2 ? 2.0 + std::numbers::sqrt2 : 1.5 + std::sqrt(3.0) / 2.0;
  REQUIRE(surface == Approx(expected_surface));

  // edges
  for (LocalIndex k = 0; k < R::kNumEdges; ++k) {
    const auto& ev = R::kEdgeVertices[static_cast<std::size_t>(k)];
    REQUIRE((R::edge_point(k, 0.0) - R::vertex(ev[0])).norm() < kTol);
    REQUIRE((R::edge_point(k, 1.0) - R::vertex(ev[1])).norm() < kTol);
    REQUIRE(R::contains(R::edge_point(k, 0.3)));
  }
}

}  // namespace

TEST_CASE("reference triangle", "[fespace]") {
  using R = ReferenceElement<2>;
  check_common<2>();
  STATIC_REQUIRE(R::volume() == 0.5);
  REQUIRE(R::vertex(1) == Point<2>(1.0, 0.0));
  REQUIRE(R::vertex(2) == Point<2>(0.0, 1.0));
  REQUIRE(R::facet_normal(0) == Point<2>(0.0, -1.0));  // edge (0,1): y = 0
  REQUIRE((R::facet_normal(1) - Point<2>(1.0, 1.0) / std::numbers::sqrt2).norm() < kTol);
  REQUIRE(R::facet_normal(2) == Point<2>(-1.0, 0.0));  // edge (2,0): x = 0
  REQUIRE(R::facet_measure(1) == Approx(std::numbers::sqrt2));
  REQUIRE(R::opposite_vertex(0) == 2);
  REQUIRE(R::opposite_vertex(1) == 0);
  REQUIRE(R::opposite_vertex(2) == 1);
  REQUIRE((R::facet_point(1, Point<1>::Constant(0.5)) - Point<2>(0.5, 0.5)).norm() < kTol);
}

TEST_CASE("reference tetrahedron", "[fespace]") {
  using R = ReferenceElement<3>;
  check_common<3>();
  REQUIRE(R::volume() == Approx(1.0 / 6.0));
  REQUIRE(R::vertex(3) == Point<3>(0.0, 0.0, 1.0));
  for (LocalIndex k = 0; k < 4; ++k)
    REQUIRE(R::opposite_vertex(k) == k);  // face i opposite vertex i
  REQUIRE((R::facet_normal(0) - Point<3>::Constant(1.0 / std::sqrt(3.0))).norm() < kTol);
  REQUIRE(R::facet_normal(1) == Point<3>(-1.0, 0.0, 0.0));
  REQUIRE(R::facet_normal(2) == Point<3>(0.0, -1.0, 0.0));
  REQUIRE(R::facet_normal(3) == Point<3>(0.0, 0.0, -1.0));
  REQUIRE(R::facet_measure(0) == Approx(std::sqrt(3.0) / 2.0));
  REQUIRE(R::facet_measure(3) == 0.5);
  // the centroid of face 3 = (0,1,2) is reached from the reference-triangle centroid
  REQUIRE((R::facet_point(3, Point<2>(1.0 / 3.0, 1.0 / 3.0)) - Point<3>(1.0 / 3.0, 1.0 / 3.0, 0.0))
              .norm() < kTol);
}

#if defined(HPFEM_ENABLE_ASSERTS)
TEST_CASE("reference element asserts on bad local indices", "[fespace]") {
  REQUIRE_THROWS_AS(ReferenceElement<2>::vertex(3), hpfem::Error);
  REQUIRE_THROWS_AS(ReferenceElement<3>::facet_normal(4), hpfem::Error);
  REQUIRE_THROWS_AS(ReferenceElement<3>::edge_point(6, 0.5), hpfem::Error);
}
#endif

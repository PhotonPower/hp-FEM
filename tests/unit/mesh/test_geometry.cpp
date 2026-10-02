#include <cmath>
#include <numbers>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "hpfem/core/error.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"

using Catch::Approx;
using hpfem::as_size;
using hpfem::Index;
using hpfem::LocalIndex;
using hpfem::Point;
using hpfem::Real;
using hpfem::mesh::affine_map;
using hpfem::mesh::affine_maps;
using hpfem::mesh::AffineMap;
using hpfem::mesh::box;
using hpfem::mesh::facet_measure;
using hpfem::mesh::Mesh;
using hpfem::mesh::outward_normal;
using hpfem::mesh::rectangle;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kTol = 1e-13;

template <int Dim>
void check_map_consistency(const Mesh<Dim>& m, const AffineMap<Dim>& map, Index c) {
  // J^{-T} J^T = I
  const auto identity = map.inverse_transpose * map.jacobian.transpose();
  REQUIRE((identity - AffineMap<Dim>::Matrix::Identity()).norm() < kTol);
  // vertices map to the reference vertices and back
  for (LocalIndex i = 0; i <= Dim; ++i) {
    Point<Dim> xi = Point<Dim>::Zero();
    if (i > 0) xi(i - 1) = 1.0;
    const Point<Dim>& x = m.vertex(m.cell_vertices(c)[as_size(i)]);
    REQUIRE((map.to_physical(xi) - x).norm() < kTol);
    REQUIRE((map.to_reference(x) - xi).norm() < kTol);
  }
  REQUIRE(map.volume() > 0);
  REQUIRE(map.h > 0);
}

/// Normals of the two cells sharing an interior facet are opposite; boundary normals point
/// out of the domain (checked against the side tag of the generated meshes).
template <int Dim>
void check_normals(const Mesh<Dim>& m) {
  for (Index f = 0; f < m.num_facets(); ++f) {
    const auto& fc = m.facet_cells(f);
    const auto& fl = m.facet_local_indices(f);
    const Point<Dim> n0 = outward_normal(m, fc[0], fl[0]);
    REQUIRE(n0.norm() == Approx(1.0).margin(kTol));
    if (m.is_boundary_facet(f)) {
      const auto tag = m.facet_tag(f);
      const int axis = (tag - 1) / 2;
      const Real sign = (tag % 2 == 1) ? -1.0 : 1.0;
      Point<Dim> expected = Point<Dim>::Zero();
      expected(axis) = sign;
      REQUIRE((n0 - expected).norm() < kTol);
    } else {
      const Point<Dim> n1 = outward_normal(m, fc[1], fl[1]);
      REQUIRE((n0 + n1).norm() < kTol);
    }
  }
}

}  // namespace

TEST_CASE("affine map of the reference triangle is the identity", "[mesh][geometry]") {
  const Mesh<2> m({{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}}, {{0, 1, 2}});
  const AffineMap<2> map = affine_map(m, 0);
  REQUIRE(map.origin == Point<2>::Zero());
  REQUIRE(map.jacobian == AffineMap<2>::Matrix::Identity());
  REQUIRE(map.inverse_transpose == AffineMap<2>::Matrix::Identity());
  REQUIRE(map.det == 1.0);
  REQUIRE(map.volume() == 0.5);
  REQUIRE(map.h == Approx(std::numbers::sqrt2));
  REQUIRE((map.centroid() - Point<2>(1.0 / 3.0, 1.0 / 3.0)).norm() < kTol);
  check_map_consistency(m, map, 0);
}

TEST_CASE("affine map of a scaled, rotated and translated triangle", "[mesh][geometry]") {
  const Real s = 2.0;
  const Real phi = 0.7;
  const Point<2> t(3.0, -1.0);
  const auto place = [&](Real x, Real y) {
    return Point<2>(t(0) + s * (std::cos(phi) * x - std::sin(phi) * y),
                    t(1) + s * (std::sin(phi) * x + std::cos(phi) * y));
  };
  // local order (v2, v0, v1) of the placed reference triangle: a non-identity local map
  const Mesh<2> m({place(0.0, 0.0), place(1.0, 0.0), place(0.0, 1.0)}, {{2, 0, 1}});
  const AffineMap<2> map = affine_map(m, 0);
  REQUIRE(map.origin == m.vertex(2));
  REQUIRE((map.jacobian.col(0) - (m.vertex(0) - m.vertex(2))).norm() < kTol);
  REQUIRE((map.jacobian.col(1) - (m.vertex(1) - m.vertex(2))).norm() < kTol);
  REQUIRE(map.volume() == Approx(0.5 * s * s));
  REQUIRE(map.h == Approx(s * std::numbers::sqrt2));
  check_map_consistency(m, map, 0);
  // a point inside stays inside after the round trip
  const Point<2> x = map.to_physical(Point<2>(0.2, 0.3));
  REQUIRE((map.to_reference(x) - Point<2>(0.2, 0.3)).norm() < kTol);
}

TEST_CASE("reversed vertex order flips the sign of det but not the volume", "[mesh][geometry]") {
  const Mesh<2> m({{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}}, {{0, 1, 2}, {0, 2, 1}});
  REQUIRE(affine_map(m, 0).det == 1.0);
  REQUIRE(affine_map(m, 1).det == -1.0);
  REQUIRE(affine_map(m, 1).volume() == 0.5);
  check_map_consistency(m, affine_map(m, 1), 1);
}

TEST_CASE("Kuhn cube: volumes sum to the box volume, both orientations occur", "[mesh][geometry]") {
  const Point<3> lower(0.0, 0.0, 0.0);
  const Point<3> upper(2.0, 1.0, 0.5);
  const Mesh<3> m = box(2, 3, 1, lower, upper);
  const auto maps = affine_maps(m);
  REQUIRE(maps.size() == as_size(m.num_cells()));
  Real total = 0;
  Index positive = 0;
  for (Index c = 0; c < m.num_cells(); ++c) {
    total += maps[as_size(c)].volume();
    if (maps[as_size(c)].det > 0) ++positive;
    check_map_consistency(m, maps[as_size(c)], c);
    REQUIRE(maps[as_size(c)].h == Approx(std::sqrt(1.0 + 1.0 / 9.0 + 0.25)));  // body diagonal
  }
  REQUIRE(total == Approx(2.0 * 1.0 * 0.5));
  REQUIRE(positive == m.num_cells() / 2);  // three of the six Kuhn tetrahedra per cube
}

TEST_CASE("facet measures and outward normals on generated meshes", "[mesh][geometry]") {
  const Mesh<2> r = rectangle(3, 2, Point<2>(0.0, 0.0), Point<2>(3.0, 1.0));
  for (const Index f : r.facets_with_tag(box_tag::kYMin))
    REQUIRE(facet_measure(r, f) == Approx(1.0));
  for (const Index f : r.facets_with_tag(box_tag::kXMin))
    REQUIRE(facet_measure(r, f) == Approx(0.5));
  check_normals(r);

  const Mesh<3> b = box(2, 1, 1, Point<3>(0.0, 0.0, 0.0), Point<3>(2.0, 2.0, 1.0));
  for (const Index f : b.facets_with_tag(box_tag::kZMin))
    REQUIRE(facet_measure(b, f) == Approx(1.0));
  for (const Index f : b.facets_with_tag(box_tag::kXMin))
    REQUIRE(facet_measure(b, f) == Approx(1.0));
  check_normals(b);

  // the sum of outward normals weighted by facet measure vanishes for every cell
  for (Index c = 0; c < b.num_cells(); ++c) {
    Point<3> sum = Point<3>::Zero();
    for (LocalIndex k = 0; k < 4; ++k) {
      sum += facet_measure(b, b.cell_facets(c)[as_size(k)]) * outward_normal(b, c, k);
    }
    REQUIRE(sum.norm() < 1e-12);
  }
}

TEST_CASE("degenerate cells are rejected", "[mesh][geometry]") {
  using Catch::Matchers::ContainsSubstring;
  const Mesh<2> collinear({{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}}, {{0, 1, 2}});
  REQUIRE_THROWS_WITH(affine_map(collinear, 0), ContainsSubstring("collinear"));
  const Mesh<3> coplanar({{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {1.0, 1.0, 0.0}},
                         {{0, 1, 2, 3}});
  REQUIRE_THROWS_AS(affine_map(coplanar, 0), hpfem::InvalidArgument);
  // nearly degenerate but valid: tiny cell far from the origin
  const Real eps = 1e-6;
  const Mesh<2> tiny({{1e3, 1e3}, {1e3 + eps, 1e3}, {1e3, 1e3 + eps}}, {{0, 1, 2}});
  REQUIRE(affine_map(tiny, 0).volume() == Approx(0.5 * eps * eps));
}

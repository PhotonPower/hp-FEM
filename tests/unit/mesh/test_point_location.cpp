#include <cmath>
#include <numbers>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "hpfem/core/error.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/mesh.hpp"
#include "hpfem/mesh/point_location.hpp"
#include "test_meshes.hpp"

using hpfem::as_size;
using hpfem::Index;
using hpfem::kInvalidIndex;
using hpfem::Point;
using hpfem::Real;
using hpfem::mesh::affine_map;
using hpfem::mesh::box;
using hpfem::mesh::cell_geometry;
using hpfem::mesh::LocatedPoint;
using hpfem::mesh::Mesh;
using hpfem::mesh::PointLocator;
using hpfem::mesh::rectangle;

namespace {

constexpr Real kTol = 1e-12;

template <int Dim>
std::vector<Point<Dim>> random_points(const Point<Dim>& lower, const Point<Dim>& upper, int count,
                                      unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<Real> u(0.0, 1.0);
  std::vector<Point<Dim>> points;
  for (int i = 0; i < count; ++i) {
    Point<Dim> x;
    for (int d = 0; d < Dim; ++d) x(d) = lower(d) + u(rng) * (upper(d) - lower(d));
    points.push_back(x);
  }
  return points;
}

/// The located cell really contains x: xi lies in the closed reference simplex and maps
/// back to x.
template <int Dim>
void check_located(const Mesh<Dim>& m, const LocatedPoint<Dim>& located, const Point<Dim>& x,
                   Real tolerance) {
  REQUIRE(located.cell >= 0);
  REQUIRE(located.cell < m.num_cells());
  Real sum = 0;
  for (int d = 0; d < Dim; ++d) {
    REQUIRE(located.xi(d) >= -tolerance);
    sum += located.xi(d);
  }
  REQUIRE(1.0 - sum >= -tolerance);
  REQUIRE((cell_geometry(m, located.cell)->evaluate(located.xi).x - x).norm() < kTol);
}

/// Random points of the bounding box are located consistently by the grid search, the
/// hinted search and a brute-force scan; vertices and centroids are found; outside points
/// are rejected.
template <int Dim>
void check_locator(const Mesh<Dim>& m, const Point<Dim>& lower, const Point<Dim>& upper,
                   unsigned seed) {
  const PointLocator<Dim> locator(m);
  std::mt19937 rng(seed);
  std::uniform_int_distribution<Index> any_cell(0, m.num_cells() - 1);
  for (const auto& x : random_points<Dim>(lower, upper, 200, seed)) {
    const auto located = locator.locate(x);
    REQUIRE(located.has_value());
    check_located(m, *located, x, locator.tolerance());
    // brute force: the lowest cell id containing x
    Index lowest = kInvalidIndex;
    for (Index c = 0; c < m.num_cells() && lowest == kInvalidIndex; ++c) {
      if (locator.reference_coordinates(c, x)) lowest = c;
    }
    REQUIRE(located->cell == lowest);
    // the hinted search finds a cell containing x as well (random points are interior)
    const auto hinted = locator.locate(x, any_cell(rng));
    REQUIRE(hinted.has_value());
    REQUIRE(hinted->cell == located->cell);
    REQUIRE((hinted->xi - located->xi).norm() < kTol);
  }
  for (Index v = 0; v < m.num_vertices(); ++v) {
    const auto located = locator.locate(m.vertex(v));
    REQUIRE(located.has_value());
    check_located(m, *located, m.vertex(v), locator.tolerance());
  }
  for (Index c = 0; c < m.num_cells(); ++c) {
    const Point<Dim> x = affine_map(m, c).centroid();
    const auto located = locator.locate(x);
    REQUIRE(located.has_value());
    REQUIRE(located->cell == c);
    // the hint itself is used when it contains the point
    REQUIRE(locator.locate(x, c)->cell == c);
    // an invalid hint falls back to the grid search
    REQUIRE(locator.locate(x, kInvalidIndex)->cell == c);
  }
  const Point<Dim> extent = upper - lower;
  REQUIRE_FALSE(locator.locate(upper + 0.01 * extent).has_value());
  REQUIRE_FALSE(locator.locate(lower - 0.01 * extent).has_value());
  REQUIRE_FALSE(locator.locate(Point<Dim>::Constant(1e6)).has_value());
  // about one cell per bucket
  Index buckets = 1;
  for (const Index n : locator.grid_divisions()) buckets *= n;
  REQUIRE(buckets >= 1);
  REQUIRE(buckets <= 4 * m.num_cells());
}

}  // namespace

TEST_CASE("PointLocator: structured rectangle and box", "[mesh][point-location]") {
  const Point<2> lo2(-1.0, 0.0);
  const Point<2> hi2(2.0, 1.5);
  check_locator(rectangle(5, 4, lo2, hi2), lo2, hi2, 1);
  const Point<3> lo3(0.0, -1.0, 2.0);
  const Point<3> hi3(1.0, 1.0, 3.5);
  check_locator(box(3, 2, 4, lo3, hi3), lo3, hi3, 2);
}

TEST_CASE("PointLocator: renumbered mesh with mixed local orientations", "[mesh][point-location]") {
  const auto [m2, perm2] = hpfem::mesh::testing::relabel(rectangle(4, 4), 3);
  check_locator<2>(m2, Point<2>::Zero(), Point<2>::Ones(), 4);
  const auto [m3, perm3] = hpfem::mesh::testing::relabel(box(2, 2, 2), 5);
  check_locator<3>(m3, Point<3>::Zero(), Point<3>::Ones(), 6);
}

TEST_CASE("PointLocator: grid follows the aspect ratio of the bounding box",
          "[mesh][point-location]") {
  const PointLocator<2> locator(rectangle(8, 2, Point<2>::Zero(), Point<2>(4.0, 1.0)));
  REQUIRE(locator.grid_divisions()[0] > locator.grid_divisions()[1]);
  REQUIRE(locator.grid_divisions()[1] >= 1);
}

TEST_CASE("PointLocator: tolerance decides about points just outside", "[mesh][point-location]") {
  const Mesh<2> m = rectangle(2, 2);
  const Point<2> x(0.5, -1e-11);  // 1e-11 below the bottom edge
  REQUIRE(PointLocator<2>(m, 1e-10).locate(x).has_value());
  REQUIRE_FALSE(PointLocator<2>(m, 0.0).locate(x).has_value());
  REQUIRE_FALSE(PointLocator<2>(m, 1e-10).locate(Point<2>(0.5, -1e-3)).has_value());
  REQUIRE_THROWS_AS(PointLocator<2>(m, -1.0), hpfem::InvalidArgument);
}

TEST_CASE("PointLocator: points on shared facets go to the lowest cell id",
          "[mesh][point-location]") {
  const Mesh<2> m = rectangle(3, 3);
  const PointLocator<2> locator(m);
  for (Index f = 0; f < m.num_facets(); ++f) {
    const auto& ev = m.edge_vertices(f);
    const Point<2> x = 0.5 * (m.vertex(ev[0]) + m.vertex(ev[1]));
    const auto located = locator.locate(x);
    REQUIRE(located.has_value());
    const auto& fc = m.facet_cells(f);
    const Index expected = fc[1] == kInvalidIndex ? fc[0] : std::min(fc[0], fc[1]);
    REQUIRE(located->cell == expected);
  }
}

TEST_CASE("PointLocator: curved cell (quarter disc)", "[mesh][point-location]") {
  // one triangle whose hypotenuse node lies on the unit circle: the cell bulges beyond the
  // straight triangle x + y <= 1
  Mesh<2> m({{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}}, {{0, 1, 2}});
  const Real s = std::numbers::sqrt2 / 2;
  m.set_edge_nodes({Point<2>(0.5, 0.0), Point<2>(s, s), Point<2>(0.0, 0.5)});
  REQUIRE(m.geometry_order() == 2);
  const PointLocator<2> locator(m);
  const auto geometry = cell_geometry(m, 0);

  const Point<2> bulge(0.68, 0.68);  // |x| = 0.96 < 1 but x + y = 1.36 > 1
  const auto located = locator.locate(bulge);
  REQUIRE(located.has_value());
  REQUIRE(located->cell == 0);
  REQUIRE((geometry->evaluate(located->xi).x - bulge).norm() < 1e-10);
  REQUIRE(locator.locate(Point<2>(0.3, 0.2)).has_value());
  REQUIRE(locator.locate(Point<2>(0.0, 0.0)).has_value());
  REQUIRE_FALSE(locator.locate(Point<2>(0.75, 0.75)).has_value());  // |x| = 1.06
  REQUIRE_FALSE(locator.locate(Point<2>(-0.1, 0.5)).has_value());
  REQUIRE_FALSE(locator.locate(Point<2>(50.0, 50.0)).has_value());  // Newton diverges
  // the bounding box covers the bulge (Bézier control point 2 m - (a + b) / 2)
  REQUIRE(locator.locate(Point<2>(0.7, 0.7)).has_value());
}

TEST_CASE("PointLocator: single cell and empty-ish meshes", "[mesh][point-location]") {
  const Mesh<3> m({{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}},
                  {{0, 1, 2, 3}});
  const PointLocator<3> locator(m);
  REQUIRE(locator.locate(Point<3>(0.2, 0.2, 0.2))->cell == 0);
  REQUIRE_FALSE(locator.locate(Point<3>(0.4, 0.4, 0.4)).has_value());
  REQUIRE(&locator.mesh() == &m);
}

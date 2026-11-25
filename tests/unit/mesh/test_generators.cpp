#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"

using Catch::Approx;

using hpfem::Index;
using hpfem::kInvalidIndex;
using hpfem::Point;
using hpfem::Real;
using hpfem::mesh::box;
using hpfem::mesh::kNoTag;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::mesh::Tag;
namespace box_tag = hpfem::mesh::box_tag;

TEST_CASE("rectangle: counts, coordinates and side tags", "[mesh]") {
  const Index nx = 3;
  const Index ny = 2;
  const Point<2> lower(-1.0, 2.0);
  const Point<2> upper(3.0, 5.0);
  const Mesh<2> m = rectangle(nx, ny, lower, upper);

  REQUIRE(m.num_vertices() == (nx + 1) * (ny + 1));
  REQUIRE(m.num_edges() == 3 * nx * ny + nx + ny);
  REQUIRE(m.num_cells() == 2 * nx * ny);
  REQUIRE(m.vertex(0) == lower);
  REQUIRE(m.vertex(m.num_vertices() - 1) == upper);
  REQUIRE(m.vertex(1)(0) == lower(0) + (upper(0) - lower(0)) / static_cast<Real>(nx));
  REQUIRE(m.vertex(nx + 1)(1) == lower(1) + (upper(1) - lower(1)) / static_cast<Real>(ny));

  REQUIRE(static_cast<Index>(m.facets_with_tag(box_tag::kXMin).size()) == ny);
  REQUIRE(static_cast<Index>(m.facets_with_tag(box_tag::kXMax).size()) == ny);
  REQUIRE(static_cast<Index>(m.facets_with_tag(box_tag::kYMin).size()) == nx);
  REQUIRE(static_cast<Index>(m.facets_with_tag(box_tag::kYMax).size()) == nx);
  for (const Index f : m.boundary_facets()) {
    const Tag t = m.facet_tag(f);
    REQUIRE(t >= box_tag::kXMin);
    REQUIRE(t <= box_tag::kYMax);
    for (const Index v : m.facet_vertices(f)) {  // every vertex of the facet lies on that side
      const int axis = (t - 1) / 2;
      const Real expected = (t % 2 == 1) ? lower(axis) : upper(axis);
      REQUIRE(m.vertex(v)(axis) == expected);
    }
  }
  REQUIRE(static_cast<Index>(m.facets_with_tag(kNoTag).size()) ==
          m.num_facets() - m.num_boundary_facets());
  REQUIRE(
      std::all_of(m.cell_tags().begin(), m.cell_tags().end(), [](Tag t) { return t == kNoTag; }));

  const Mesh<2> untagged = rectangle(nx, ny, lower, upper, false);
  REQUIRE(static_cast<Index>(untagged.facets_with_tag(kNoTag).size()) == untagged.num_facets());
}

TEST_CASE("box: counts, coordinates and side tags", "[mesh]") {
  const Index nx = 2;
  const Index ny = 3;
  const Index nz = 1;
  const Point<3> lower(0.0, 0.0, -0.5);
  const Point<3> upper(2.0, 3.0, 0.5);
  const Mesh<3> m = box(nx, ny, nz, lower, upper);

  REQUIRE(m.num_vertices() == (nx + 1) * (ny + 1) * (nz + 1));
  REQUIRE(m.num_cells() == 6 * nx * ny * nz);
  REQUIRE(m.num_boundary_facets() == 4 * (nx * ny + ny * nz + nx * nz));
  REQUIRE(m.vertex(0) == lower);
  REQUIRE(m.vertex(m.num_vertices() - 1) == upper);
  REQUIRE(m.num_vertices() - m.num_edges() + m.num_faces() - m.num_cells() == 1);

  REQUIRE(static_cast<Index>(m.facets_with_tag(box_tag::kXMin).size()) == 2 * ny * nz);
  REQUIRE(static_cast<Index>(m.facets_with_tag(box_tag::kXMax).size()) == 2 * ny * nz);
  REQUIRE(static_cast<Index>(m.facets_with_tag(box_tag::kYMin).size()) == 2 * nx * nz);
  REQUIRE(static_cast<Index>(m.facets_with_tag(box_tag::kYMax).size()) == 2 * nx * nz);
  REQUIRE(static_cast<Index>(m.facets_with_tag(box_tag::kZMin).size()) == 2 * nx * ny);
  REQUIRE(static_cast<Index>(m.facets_with_tag(box_tag::kZMax).size()) == 2 * nx * ny);
  for (const Index f : m.boundary_facets()) {
    const Tag t = m.facet_tag(f);
    REQUIRE(t >= box_tag::kXMin);
    REQUIRE(t <= box_tag::kZMax);
    for (const Index v : m.facet_vertices(f)) {
      const int axis = (t - 1) / 2;
      const Real expected = (t % 2 == 1) ? lower(axis) : upper(axis);
      REQUIRE(m.vertex(v)(axis) == expected);
    }
  }
  REQUIRE(static_cast<Index>(m.facets_with_tag(kNoTag).size()) ==
          m.num_facets() - m.num_boundary_facets());
}

TEST_CASE("generators reject empty or inverted boxes", "[mesh]") {
  REQUIRE_THROWS_AS(rectangle(0, 1), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(rectangle(1, 1, Point<2>(0.0, 0.0), Point<2>(1.0, 0.0)),
                    hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(box(1, 1, 0), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(box(1, 1, 1, Point<3>(0.0, 0.0, 0.0), Point<3>(1.0, -1.0, 1.0)),
                    hpfem::InvalidArgument);
}

TEST_CASE("disc and ball: boundary on the circle / sphere, tags, curved area and volume",
          "[mesh][generators]") {
  const Point<2> c2(0.5, -1.0);
  const Real r = 1.5;
  for (const bool curved : {false, true}) {
    const Mesh<2> d = hpfem::mesh::disc(6, c2, r, curved);
    REQUIRE(d.num_cells() == 2 * 36);
    REQUIRE(d.geometry_order() == (curved ? 2 : 1));
    for (const Index f : d.boundary_facets()) {
      REQUIRE(d.facet_tag(f) == hpfem::mesh::kDiscBoundary);
      for (const Index v : d.facet_vertices(f)) {
        REQUIRE(std::abs((d.vertex(v) - c2).norm() - r) < 1e-12);
      }
      if (curved) REQUIRE(std::abs((d.edge_node(f) - c2).norm() - r) < 1e-12);
    }
    for (Index c = 0; c < d.num_cells(); ++c) {
      REQUIRE(hpfem::mesh::affine_map(d, c).det > 0);
    }
  }
  // the curved mesh approximates the area pi r^2 much better than the polygon
  const auto area = [](const Mesh<2>& m) {
    Real a = 0;
    const std::array<Point<2>, 3> midpoints{Point<2>(0.5, 0.0), Point<2>(0.5, 0.5),
                                            Point<2>(0.0, 0.5)};
    for (Index c = 0; c < m.num_cells(); ++c) {
      const auto g = hpfem::mesh::cell_geometry(m, c);
      for (const auto& q : midpoints)
        a += std::abs(g->evaluate(q).det) / 6.0;  // exact for det quadratic
    }
    return a;
  };
  const Real exact = std::numbers::pi * r * r;
  const Real straight = area(hpfem::mesh::disc(8, c2, r, false));
  const Real bent = area(hpfem::mesh::disc(8, c2, r, true));
  REQUIRE(straight < exact);
  REQUIRE(std::abs(bent - exact) < 0.1 * std::abs(straight - exact));
  REQUIRE(std::abs(bent - exact) < 1e-2 * exact);

  const Point<3> c3(1.0, 0.0, 2.0);
  const Mesh<3> b = hpfem::mesh::ball(3, c3, 0.5);
  REQUIRE(b.num_cells() == 6 * 27);
  REQUIRE(b.geometry_order() == 2);
  for (const Index f : b.boundary_facets()) {
    REQUIRE(b.facet_tag(f) == hpfem::mesh::kDiscBoundary);
    for (const Index v : b.facet_vertices(f))
      REQUIRE(std::abs((b.vertex(v) - c3).norm() - 0.5) < 1e-12);
  }
  // the map preserves the orientation of every Kuhn tetrahedron (both signs occur): the same
  // tetrahedra on the unmapped grid (same vertex numbering)
  const Mesh<3> grid = box(3, 3, 3, Point<3>(-1.0, -1.0, -1.0), Point<3>(1.0, 1.0, 1.0), false);
  const Mesh<3> cube(std::vector<Point<3>>(grid.vertices().begin(), grid.vertices().end()),
                     std::vector<Mesh<3>::CellVertices>(b.cells().begin(), b.cells().end()));
  for (Index c = 0; c < b.num_cells(); ++c) {
    REQUIRE(hpfem::mesh::affine_map(b, c).det * hpfem::mesh::affine_map(cube, c).det > 0);
  }
  const auto volume = [](const Mesh<3>& m) {
    Real v = 0;
    const auto rule = hpfem::assembly::simplex_quadrature<3>(3);  // det cubic
    for (Index c = 0; c < m.num_cells(); ++c) {
      const auto g = hpfem::mesh::cell_geometry(m, c);
      for (std::size_t q = 0; q < rule.size(); ++q)
        v += rule.weights[q] * std::abs(g->evaluate(rule.points[q]).det);
    }
    return v;
  };
  const Real exact3 = 4.0 / 3.0 * std::numbers::pi * 0.125;
  const Real straight3 = volume(hpfem::mesh::ball(4, c3, 0.5, false));
  const Real bent3 = volume(hpfem::mesh::ball(4, c3, 0.5, true));
  REQUIRE(straight3 < exact3);
  REQUIRE(std::abs(bent3 - exact3) < 0.2 * std::abs(straight3 - exact3));
  REQUIRE_THROWS_AS(hpfem::mesh::disc(0), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(hpfem::mesh::ball(2, c3, -1.0), hpfem::InvalidArgument);
}

TEST_CASE("curve_boundary moves only the nodes of the tagged facets", "[mesh][generators]") {
  Mesh<2> m = rectangle(3, 2);
  hpfem::mesh::curve_boundary<2>(m, box_tag::kXMax, [](const Point<2>& x) {
    return Point<2>(1.0 + 0.1 * std::sin(std::numbers::pi * x(1)), x(1));
  });
  REQUIRE(m.geometry_order() == 2);
  for (Index e = 0; e < m.num_edges(); ++e) {
    const auto& ev = m.edge_vertices(e);
    const Point<2> mid = 0.5 * (m.vertex(ev[0]) + m.vertex(ev[1]));
    if (m.is_boundary_facet(e) && m.facet_tag(e) == box_tag::kXMax) {
      REQUIRE(m.edge_node(e)(0) == Approx(1.0 + 0.1 * std::sin(std::numbers::pi * mid(1))));
    } else {
      REQUIRE((m.edge_node(e) - mid).norm() < 1e-15);
    }
  }
}

TEST_CASE("square_with_disc: inclusion cells, curved interface on the circle, outer tags",
          "[mesh][generators]") {
  const Real r = 0.25;
  const Mesh<2> m = hpfem::mesh::square_with_disc(2, r, 1.0, 1.5, 2);
  REQUIRE(m.num_cells() == 2 * 24 * 24);  // (1.5 / 0.25) n = 12 cells per half axis
  REQUIRE(m.geometry_order() == 2);
  Index inclusion = 0;
  Index interface = 0;
  for (Index c = 0; c < m.num_cells(); ++c) {
    if (m.cell_tag(c) == 2) ++inclusion;
    REQUIRE(hpfem::mesh::affine_map(m, c).det > 0);
  }
  REQUIRE(inclusion == 2 * 4 * 4);  // the inner square has 2n x 2n = 4 x 4 squares
  for (Index f = 0; f < m.num_facets(); ++f) {
    const auto& fc = m.facet_cells(f);
    if (fc[1] == kInvalidIndex) {
      REQUIRE(m.facet_tag(f) != kNoTag);  // every outer side is tagged
      continue;
    }
    if ((m.cell_tag(fc[0]) == 2) != (m.cell_tag(fc[1]) == 2)) {
      ++interface;
      for (const Index v : m.facet_vertices(f)) REQUIRE(m.vertex(v).norm() == Approx(r));
      REQUIRE(m.edge_node(f).norm() == Approx(r));
    } else {
      const auto& ev = m.edge_vertices(f);
      REQUIRE((m.edge_node(f) - 0.5 * (m.vertex(ev[0]) + m.vertex(ev[1]))).norm() < 1e-15);
    }
  }
  REQUIRE(interface == 4 * 4);  // perimeter of the 4 x 4 inner square in grid edges
  REQUIRE(m.facets_with_tag(box_tag::kXMax).size() == 24);
  REQUIRE(m.vertex(0) == Point<2>(-1.5, -1.5));
  REQUIRE_THROWS_AS(hpfem::mesh::square_with_disc(3, r, 1.0, 1.1), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(hpfem::mesh::square_with_disc(2, 0.5, 0.4, 1.0), hpfem::InvalidArgument);
}

namespace {

/// Smallest ratio of det J of the curved map to det J of the cell's straight (affine) map over a
/// lattice of points in all cells: positive means no cell folds, small means a nearly singular
/// map (estimator residuals with derivatives of the field blow up there).
template <int Dim>
Real min_curved_jacobian_ratio(const Mesh<Dim>& m) {
  constexpr int kSteps = 12;
  Real worst = 1e300;
  for (Index c = 0; c < m.num_cells(); ++c) {
    const auto geometry = hpfem::mesh::cell_geometry(m, c);
    const Real affine = hpfem::mesh::affine_map(m, c).det;
    for (int i = 0; i <= kSteps; ++i) {
      for (int j = 0; i + j <= kSteps; ++j) {
        for (int k = 0; i + j + k <= (Dim == 3 ? kSteps : 0); ++k) {
          Point<Dim> xi;
          xi(0) = Real(i) / kSteps;
          xi(1) = Real(j) / kSteps;
          if constexpr (Dim == 3) xi(2) = Real(k) / kSteps;
          worst = std::min(worst, geometry->evaluate(xi).det / affine);
        }
      }
    }
  }
  return worst;
}

}  // namespace

TEST_CASE("disc, ball and the inclusion meshes: no curved cell folds or degenerates (#234)",
          "[mesh][generators]") {
  // Before: the ring between circle and square grew quadratically away from the circle (first
  // ring of cells ~ 1 / (n (l - 1)) times thinner than wide), and with one diagonal for all
  // squares / cubes the cells along the circle in two quadrants were slivers; their curved edges
  // folded them (square_with_disc n = 4: ratio -1.1; box_with_ball n = 3: 5e-4)
  for (const Index n : {1, 2, 3, 4, 6, 8}) {
    INFO("n = " << n);
    const Real disc = min_curved_jacobian_ratio(hpfem::mesh::disc(n));
    const Real square = min_curved_jacobian_ratio(hpfem::mesh::square_with_disc(n, 0.25, 1.0, 2.0));
    INFO("disc " << disc << ", square_with_disc " << square);
    CHECK(disc > 0.5);
    CHECK(square > 0.5);
  }
  for (const Index n : {1, 2, 3}) {
    INFO("n = " << n);
    const Real ball = min_curved_jacobian_ratio(hpfem::mesh::ball(n));
    const Real box = min_curved_jacobian_ratio(hpfem::mesh::box_with_ball(n, 0.5, 1.0, 1.5));
    INFO("ball " << ball << ", box_with_ball " << box);
    CHECK(ball > (n == 1 ? 0.45 : 0.5));  // n = 1: a single cube, its corners on the sphere
    CHECK(box > 0.5);
  }
}

TEST_CASE("box_with_ball: inclusion cells, curved interface on the sphere, outer tags",
          "[mesh][generators]") {
  const Real r = 0.5;
  const Mesh<3> m = hpfem::mesh::box_with_ball(1, r, 1.0, 1.5, 2);
  REQUIRE(m.num_cells() == 6 * 6 * 6 * 6);  // (1.5 / 0.5) n = 3 cubes per half axis, 6 tets each
  REQUIRE(m.geometry_order() == 2);
  Index inclusion = 0;
  Index curved_edges = 0;
  for (Index c = 0; c < m.num_cells(); ++c) {
    const bool tagged = m.cell_tag(c) == 2;
    if (tagged) ++inclusion;
    // inclusion cells lie in the ball, all other cells outside (vertices on or beyond the sphere)
    for (const Index v : m.cell_vertices(c)) {
      if (tagged) REQUIRE(m.vertex(v).norm() <= r * (1 + 1e-12));
      if (!tagged) REQUIRE(m.vertex(v).norm() >= r * (1 - 1e-12));
    }
  }
  // the edges of the interface faces are curved: their nodes lie on the sphere (a chord edge
  // inside the ball whose end points happen to lie on the sphere stays straight)
  for (Index f = 0; f < m.num_facets(); ++f) {
    const auto& fc = m.facet_cells(f);
    if (fc[1] == kInvalidIndex) continue;
    if ((m.cell_tag(fc[0]) == 2) == (m.cell_tag(fc[1]) == 2)) continue;
    const auto& fv = m.facet_vertices(f);
    for (const auto& [a, b] :
         {std::pair{fv[0], fv[1]}, std::pair{fv[1], fv[2]}, std::pair{fv[0], fv[2]}}) {
      REQUIRE(m.vertex(a).norm() == Approx(r).epsilon(1e-12));
      REQUIRE(m.edge_node(m.edge_id(a, b)).norm() == Approx(r).epsilon(1e-12));
      ++curved_edges;
    }
  }
  REQUIRE(inclusion == 6 * 2 * 2 * 2);  // the inner cube [-1, 1]^3 has 2n = 2 cubes per axis
  REQUIRE(curved_edges > 0);
  for (const Index f : m.boundary_facets()) REQUIRE(m.facet_tag(f) != kNoTag);
  REQUIRE(m.facets_with_tag(box_tag::kZMax).size() == 2 * 6 * 6);
  Real volume = 0;
  for (Index c = 0; c < m.num_cells(); ++c) volume += hpfem::mesh::affine_map(m, c).volume();
  REQUIRE(volume == Approx(27.0).epsilon(1e-12));
  REQUIRE_THROWS_AS(hpfem::mesh::box_with_ball(3, r, 1.0, 1.1), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(hpfem::mesh::box_with_ball(1, 0.5, 0.4, 1.0), hpfem::InvalidArgument);
}

TEST_CASE("hexagon: counts, positive cells, boundary on the hexagon, inner tag, C_6 symmetry",
          "[mesh][generators]") {
  const Index n = 4;
  const Real radius = 2.0;
  const Mesh<2> m = hpfem::mesh::hexagon(n, radius, 2, 7);
  CHECK(m.num_vertices() == 3 * n * (n + 1) + 1);
  CHECK(m.num_cells() == 6 * n * n);
  Real area = 0;
  Real inner_area = 0;
  for (Index c = 0; c < m.num_cells(); ++c) {
    const auto& v = m.cell_vertices(c);
    const Point<2> a = m.vertex(v[1]) - m.vertex(v[0]);
    const Point<2> b = m.vertex(v[2]) - m.vertex(v[0]);
    const Real det = a(0) * b(1) - a(1) * b(0);
    REQUIRE(det > 0);
    area += 0.5 * det;
    if (m.cell_tag(c) == 7) inner_area += 0.5 * det;
  }
  const Real hex = 1.5 * std::sqrt(3.0);  // area of the unit hexagon
  CHECK(area == Approx(hex * radius * radius));
  CHECK(inner_area == Approx(hex * 1.0));  // inner circumradius 2 * radius / n = 1
  for (const Index f : m.boundary_facets()) {
    CHECK(m.facet_tag(f) == hpfem::mesh::kDiscBoundary);
    for (const Index v : m.facet_vertices(f)) {
      // on the hexagon: the largest projection onto the six edge normals is the apothem
      Real proj = 0;
      for (int k = 0; k < 6; ++k) {
        const Real t = std::numbers::pi / 6 + k * std::numbers::pi / 3;
        proj = std::max(proj, m.vertex(v).dot(Point<2>(std::cos(t), std::sin(t))));
      }
      CHECK(proj == Approx(radius * std::sqrt(3.0) / 2));
    }
  }
  // the vertex set is invariant under the rotation by 60 degrees
  const Real c = 0.5;
  const Real s = 0.5 * std::sqrt(3.0);
  for (Index v = 0; v < m.num_vertices(); ++v) {
    const Point<2> x = m.vertex(v);
    const Point<2> r(c * x(0) - s * x(1), s * x(0) + c * x(1));
    bool found = false;
    for (Index w = 0; w < m.num_vertices() && !found; ++w) found = (m.vertex(w) - r).norm() < 1e-12;
    CHECK(found);
  }
  CHECK_THROWS_AS(hpfem::mesh::hexagon(0), hpfem::InvalidArgument);
  CHECK_THROWS_AS(hpfem::mesh::hexagon(2, 1.0, 3), hpfem::InvalidArgument);
  CHECK_THROWS_AS(hpfem::mesh::hexagon(2, 1.0, 1, kNoTag), hpfem::InvalidArgument);
}

TEST_CASE("polar_disc: counts, curved circles, annulus tags, C_6 and mirror symmetry",
          "[mesh][generators]") {
  const Index sides = 6;
  const std::vector<Real> radii{0.5, 1.0, 1.5};
  const Mesh<2> m = hpfem::mesh::polar_disc(sides, 2, radii, {7, kNoTag, 8});
  const Index rings = 6;  // 2 per annulus: the spacing r_0 / n everywhere
  CHECK(m.num_vertices() == 1 + sides * rings * (rings + 1) / 2);
  CHECK(m.num_cells() == sides * rings * rings);
  REQUIRE(m.geometry_order() == 2);
  // positive affine orientation and the curved areas of the annuli
  const std::array<Point<2>, 3> midpoints{Point<2>(0.5, 0.0), Point<2>(0.5, 0.5),
                                          Point<2>(0.0, 0.5)};
  std::array<Real, 3> area{};
  for (Index c = 0; c < m.num_cells(); ++c) {
    const auto& v = m.cell_vertices(c);
    const Point<2> a = m.vertex(v[1]) - m.vertex(v[0]);
    const Point<2> b = m.vertex(v[2]) - m.vertex(v[0]);
    REQUIRE(a(0) * b(1) - a(1) * b(0) > 0);
    const auto g = hpfem::mesh::cell_geometry(m, c);
    Real cell_area = 0;
    for (const auto& q : midpoints) cell_area += std::abs(g->evaluate(q).det) / 6.0;
    const Tag tag = m.cell_tag(c);
    area[tag == 7 ? 0 : (tag == kNoTag ? 1 : 2)] += cell_area;
  }
  const Real pi = std::numbers::pi;
  CHECK(area[0] == Approx(pi * 0.25).epsilon(1e-3));
  CHECK(area[1] == Approx(pi * (1.0 - 0.25)).epsilon(1e-3));
  CHECK(area[2] == Approx(pi * (2.25 - 1.0)).epsilon(1e-3));
  // the boundary and the edge nodes of every circle lie on it
  for (const Index f : m.boundary_facets()) {
    CHECK(m.facet_tag(f) == hpfem::mesh::kDiscBoundary);
    for (const Index v : m.facet_vertices(f)) CHECK(m.vertex(v).norm() == Approx(1.5));
  }
  Index on_circles = 0;
  for (Index e = 0; e < m.num_edges(); ++e) {
    const auto& ev = m.edge_vertices(e);
    for (const Real r : radii) {
      if (std::abs(m.vertex(ev[0]).norm() - r) < 1e-12 &&
          std::abs(m.vertex(ev[1]).norm() - r) < 1e-12) {
        CHECK(m.edge_node(e).norm() == Approx(r));
        ++on_circles;
      }
    }
  }
  CHECK(on_circles == sides * (2 + 4 + 6));
  // vertices and edge nodes invariant under the rotation by 60 degrees and the x-axis mirror
  const auto invariant = [](const std::vector<Point<2>>& points, auto&& map) {
    for (const Point<2>& x : points) {
      const Point<2> y = map(x);
      if (std::none_of(points.begin(), points.end(),
                       [&](const Point<2>& z) { return (z - y).norm() < 1e-12; })) {
        return false;
      }
    }
    return true;
  };
  std::vector<Point<2>> points(m.vertices().begin(), m.vertices().end());
  for (Index e = 0; e < m.num_edges(); ++e) points.push_back(m.edge_node(e));
  const Real c60 = 0.5;
  const Real s60 = 0.5 * std::sqrt(3.0);
  CHECK(invariant(points, [&](const Point<2>& x) {
    return Point<2>(c60 * x(0) - s60 * x(1), s60 * x(0) + c60 * x(1));
  }));
  CHECK(invariant(points, [](const Point<2>& x) { return Point<2>(x(0), -x(1)); }));
  CHECK_THROWS_AS(hpfem::mesh::polar_disc(2, 2, {1.0}), hpfem::InvalidArgument);
  CHECK_THROWS_AS(hpfem::mesh::polar_disc(6, 0, {1.0}), hpfem::InvalidArgument);
  CHECK_THROWS_AS(hpfem::mesh::polar_disc(6, 2, {1.0, 1.0}), hpfem::InvalidArgument);
  CHECK_THROWS_AS(hpfem::mesh::polar_disc(6, 2, {1.0, 2.0}, {7}), hpfem::InvalidArgument);
  CHECK_THROWS_AS(hpfem::mesh::polar_disc(6, 2, {}), hpfem::InvalidArgument);
}

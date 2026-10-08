// Mesh report (M15 F6): counts, angles, aspect ratio, edge lengths, curved-cell validity and
// tags of generated meshes, and the periodic-pair check.
#include <cmath>
#include <numbers>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/core/error.hpp"
#include "hpfem/mesh/adaptive_mesh.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/report.hpp"

using Catch::Approx;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
namespace box_tag = hpfem::mesh::box_tag;

TEST_CASE("mesh report of a structured rectangle", "[mesh][report]") {
  hpfem::mesh::Mesh<2> mesh = hpfem::mesh::rectangle(2, 3, Point<2>(0.0, 0.0), Point<2>(2.0, 3.0));
  mesh.set_cell_tag(0, 5);
  const auto r = hpfem::mesh::report(mesh);
  REQUIRE(r.num_cells == 12);
  REQUIRE(r.num_vertices == 12);
  REQUIRE(r.num_facets == mesh.num_facets());
  REQUIRE(r.num_boundary_facets == 10);
  // right isosceles triangles: angles 45, 45, 90 degrees, mean 60
  REQUIRE(r.min_angle == Approx(std::numbers::pi / 4));
  REQUIRE(r.mean_angle == Approx(std::numbers::pi / 3));
  // circumradius / inradius = 1 + sqrt(2), normalised by 2
  REQUIRE(r.max_aspect_ratio == Approx((1 + std::sqrt(2.0)) / 2));
  REQUIRE(r.min_edge == Approx(1.0));
  REQUIRE(r.max_edge == Approx(std::sqrt(2.0)));
  REQUIRE(r.mean_edge > 1.0);
  REQUIRE(r.mean_edge < std::sqrt(2.0));
  REQUIRE(r.num_curved == 0);
  REQUIRE(r.num_invalid == 0);
  REQUIRE(r.num_untagged == 11);
  REQUIRE(r.cell_tags == std::vector<hpfem::mesh::Tag>{5});
  REQUIRE(r.facet_tags.size() == 4);
  REQUIRE(r.num_hanging == 0);
  hpfem::mesh::AdaptiveMesh<2> adaptive(mesh);
  adaptive.refine(std::vector<Index>{0});
  REQUIRE(hpfem::mesh::report(adaptive.mesh()).num_hanging > 0);
}

TEST_CASE("mesh report counts curved cells and finds their Jacobians valid", "[mesh][report]") {
  const auto r2 = hpfem::mesh::report(hpfem::mesh::disc(2));
  REQUIRE(r2.num_curved > 0);
  REQUIRE(r2.num_invalid == 0);
  REQUIRE(r2.invalid_cells.empty());
  const auto r3 = hpfem::mesh::report(hpfem::mesh::box(2, 2, 2));
  REQUIRE(r3.num_cells == 48);
  REQUIRE(r3.min_angle > 0.0);
  REQUIRE(r3.min_angle < r3.mean_angle);
  REQUIRE(r3.max_aspect_ratio >= 1.0);
  REQUIRE(r3.facet_tags.size() == 6);
}

TEST_CASE("check_periodic of a rectangle's sides", "[mesh][report][periodic]") {
  hpfem::mesh::Mesh<2> mesh = hpfem::mesh::rectangle(3, 4);
  const auto ok =
      hpfem::mesh::check_periodic(mesh, box_tag::kXMin, box_tag::kXMax, Point<2>(1.0, 0.0));
  REQUIRE(ok.identical());
  REQUIRE(ok.matched == 4);
  REQUIRE(ok.max_mismatch < 1e-14);
  const auto wrong =
      hpfem::mesh::check_periodic(mesh, box_tag::kXMin, box_tag::kXMax, Point<2>(1.0, 0.1));
  REQUIRE_FALSE(wrong.identical());
  REQUIRE(wrong.unmatched_slave == 4);
  REQUIRE(wrong.max_mismatch == Approx(0.1));
  // one side refined: the counts differ, the coarse facets match nothing exactly
  hpfem::mesh::AdaptiveMesh<2> adaptive(mesh);
  std::vector<Index> marked;
  for (Index c = 0; c < adaptive.mesh().num_cells(); ++c) {
    if (hpfem::mesh::affine_map(adaptive.mesh(), c).centroid()(0) > 2.0 / 3.0) marked.push_back(c);
  }
  adaptive.refine(marked);
  const auto refined = hpfem::mesh::check_periodic(adaptive.mesh(), box_tag::kXMin, box_tag::kXMax,
                                                   Point<2>(1.0, 0.0));
  REQUIRE_FALSE(refined.identical());
  REQUIRE(refined.num_slave == 8);
  REQUIRE(refined.num_master == 4);
  REQUIRE_THROWS_AS(hpfem::mesh::check_periodic(mesh, 77, box_tag::kXMax, Point<2>(1.0, 0.0)),
                    hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(hpfem::mesh::report(hpfem::mesh::Mesh<2>({}, {})), hpfem::InvalidArgument);
}

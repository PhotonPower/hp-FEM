#include <algorithm>

#include <catch2/catch_test_macros.hpp>

#include "hpfem/core/error.hpp"
#include "hpfem/mesh/generators.hpp"

using hpfem::Index;
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

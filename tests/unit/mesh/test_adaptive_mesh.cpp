#include <algorithm>
#include <cmath>
#include <set>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/core/error.hpp"
#include "hpfem/mesh/adaptive_mesh.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/refinement.hpp"

using Catch::Approx;
using hpfem::as_size;
using hpfem::Index;
using hpfem::kInvalidIndex;
using hpfem::LocalIndex;
using hpfem::Point;
using hpfem::Real;
using hpfem::mesh::AdaptiveMesh;
using hpfem::mesh::affine_map;
using hpfem::mesh::box;
using hpfem::mesh::disc;
using hpfem::mesh::extract;
using hpfem::mesh::kDiscBoundary;
using hpfem::mesh::kNoTag;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::mesh::refine_uniform;
using hpfem::mesh::RefinementStep;
using hpfem::mesh::Tag;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

template <int Dim>
Mesh<Dim> unit_mesh(Index n) {
  if constexpr (Dim == 2) {
    return rectangle(n, n);
  } else {
    return box(n, n, n);
  }
}

template <int Dim>
Real total_volume(const Mesh<Dim>& m) {
  Real v = 0;
  for (Index c = 0; c < m.num_cells(); ++c) v += affine_map(m, c).volume();
  return v;
}

/// Every pair of leaf cells sharing a vertex differs by at most one level.
template <int Dim>
void check_balance(const AdaptiveMesh<Dim>& adaptive) {
  const auto& m = adaptive.mesh();
  std::vector<std::vector<Index>> vertex_cells(as_size(m.num_vertices()));
  for (Index c = 0; c < m.num_cells(); ++c) {
    for (const Index v : m.cell_vertices(c)) vertex_cells[as_size(v)].push_back(c);
  }
  for (const auto& cells : vertex_cells) {
    for (const Index a : cells) {
      for (const Index b : cells) REQUIRE(std::abs(adaptive.level(a) - adaptive.level(b)) <= 1);
    }
  }
}

/// Hanging tables are consistent with the mesh: one cell per parent / child facet, the
/// hanging vertex at the midpoint of the parent edge, the children covering the parent.
template <int Dim>
void check_hanging(const Mesh<Dim>& m) {
  using Role = typename Mesh<Dim>::HangingRole;
  for (const auto& h : m.hanging_edges()) {
    const auto& pv = m.edge_vertices(h.parent);
    const Point<Dim> mid = 0.5 * (m.vertex(pv[0]) + m.vertex(pv[1]));
    REQUIRE((m.vertex(h.vertex) - mid).norm() < 1e-12);
    REQUIRE(m.edge_vertices(h.children[0])[0] == std::min(pv[0], h.vertex));
    REQUIRE(m.edge_vertices(h.children[0])[1] == std::max(pv[0], h.vertex));
    REQUIRE(m.edge_vertices(h.children[1])[0] == std::min(pv[1], h.vertex));
    REQUIRE(m.edge_vertices(h.children[1])[1] == std::max(pv[1], h.vertex));
    if constexpr (Dim == 2) {
      REQUIRE(m.facet_hanging_role(h.parent) == Role::kParent);
      REQUIRE(m.facet_cells(h.parent)[1] == kInvalidIndex);
      REQUIRE(!m.is_boundary_facet(h.parent));
      for (const Index c : h.children) {
        REQUIRE(m.facet_hanging_role(c) == Role::kChild);
        REQUIRE(m.hanging_parent_facet(c) == h.parent);
        REQUIRE(m.facet_cells(c)[1] == kInvalidIndex);
        REQUIRE(!m.is_boundary_facet(c));
        REQUIRE(m.facet_cells(c)[0] != m.facet_cells(h.parent)[0]);
      }
      const auto children = m.hanging_child_facets(h.parent);
      REQUIRE(children.size() == 2);
    }
  }
  if constexpr (Dim == 3) {
    for (const auto& h : m.hanging_faces()) {
      REQUIRE(m.facet_hanging_role(h.parent) == Role::kParent);
      REQUIRE(m.facet_cells(h.parent)[1] == kInvalidIndex);
      Real area = 0;
      for (const Index c : h.children) {
        REQUIRE(m.facet_hanging_role(c) == Role::kChild);
        REQUIRE(m.hanging_parent_facet(c) == h.parent);
        REQUIRE(m.facet_cells(c)[1] == kInvalidIndex);
        area += hpfem::mesh::facet_measure(m, c);
      }
      REQUIRE(area == Approx(hpfem::mesh::facet_measure(m, h.parent)));
      // the three edges of a hanging face hang as well
      const auto& fv = m.face_vertices(h.parent);
      for (const auto [a, b] :
           {std::pair{fv[0], fv[1]}, std::pair{fv[1], fv[2]}, std::pair{fv[0], fv[2]}}) {
        const Index e = m.edge_id(a, b);
        REQUIRE(std::any_of(m.hanging_edges().begin(), m.hanging_edges().end(),
                            [e](const auto& he) { return he.parent == e; }));
      }
    }
  }
  // no facet with two cells may hang; boundary facets are exactly the unhung one-cell facets
  Index boundary = 0;
  for (Index f = 0; f < m.num_facets(); ++f) {
    if (m.facet_cells(f)[1] != kInvalidIndex) REQUIRE(m.facet_hanging_role(f) == Role::kNone);
    if (m.is_boundary_facet(f)) ++boundary;
  }
  REQUIRE(boundary == m.num_boundary_facets());
}

template <int Dim>
Index count_tag(const Mesh<Dim>& m, Tag tag) {
  return static_cast<Index>(m.facets_with_tag(tag).size());
}

/// The step maps every new cell onto its old cell: the physical centroid of a child agrees
/// with the parent map of its reference centroid.
template <int Dim>
void check_step(const Mesh<Dim>& old_mesh, const Mesh<Dim>& new_mesh, const RefinementStep& step) {
  REQUIRE(step.num_cells() == new_mesh.num_cells());
  REQUIRE(step.num_old_cells == old_mesh.num_cells());
  const Point<Dim> centre = Point<Dim>::Constant(1.0 / (Dim + 1));
  for (Index c = 0; c < new_mesh.num_cells(); ++c) {
    const Index parent = step.parent[as_size(c)];
    REQUIRE(parent >= 0);
    REQUIRE(parent < old_mesh.num_cells());
    const LocalIndex child = step.child[as_size(c)];
    const Point<Dim> xi_old =
        child < 0 ? centre : hpfem::mesh::detail::parent_reference<Dim>(child, centre);
    const Point<Dim> x_old = affine_map(old_mesh, parent).to_physical(xi_old);
    REQUIRE((x_old - affine_map(new_mesh, c).centroid()).norm() < 1e-12);
    if (child < 0) REQUIRE(new_mesh.cell_vertices(c) == old_mesh.cell_vertices(parent));
  }
}

}  // namespace

TEST_CASE("AdaptiveMesh refines one triangle with hanging edges", "[mesh][adaptive]") {
  AdaptiveMesh<2> adaptive(rectangle(2, 2));
  const Mesh<2> before = adaptive.mesh();
  REQUIRE(adaptive.mesh().is_conforming());
  const std::vector<Index> marked{0};
  const RefinementStep step = adaptive.refine(marked);
  const Mesh<2>& m = adaptive.mesh();
  CHECK(m.num_cells() == before.num_cells() - 1 + 4);
  CHECK(m.num_vertices() == before.num_vertices() + 3);
  CHECK(total_volume(m) == Approx(1.0));
  CHECK(adaptive.max_level() == 1);
  Index interior_edges_of_cell = 0;
  Index boundary_edges_of_cell = 0;
  for (const Index e : before.cell_edges(0)) {
    (before.is_boundary_facet(e) ? boundary_edges_of_cell : interior_edges_of_cell)++;
  }
  CHECK(static_cast<Index>(m.hanging_edges().size()) == interior_edges_of_cell);
  CHECK(m.num_boundary_facets() == before.num_boundary_facets() + boundary_edges_of_cell);
  CHECK(!m.is_conforming());
  check_hanging(m);
  check_balance(adaptive);
  check_step(before, m, step);
  // boundary tags transfer to the split boundary edges, interior cuts stay untagged
  Index tagged = 0;
  for (const Index f : m.boundary_facets()) {
    if (m.facet_tag(f) != kNoTag) ++tagged;
  }
  CHECK(tagged == m.num_boundary_facets());
  for (Index f = 0; f < m.num_facets(); ++f) {
    if (!m.is_boundary_facet(f)) CHECK(m.facet_tag(f) == kNoTag);
  }
  CHECK(count_tag(m, box_tag::kXMin) + count_tag(m, box_tag::kXMax) + count_tag(m, box_tag::kYMin) +
            count_tag(m, box_tag::kYMax) ==
        m.num_boundary_facets());
}

TEST_CASE("AdaptiveMesh keeps the 2:1 balance under repeated corner refinement",
          "[mesh][adaptive]") {
  AdaptiveMesh<2> adaptive(rectangle(2, 2));
  for (int step = 0; step < 5; ++step) {
    // refine every cell touching the origin
    std::vector<Index> marked;
    const Mesh<2>& m = adaptive.mesh();
    for (Index c = 0; c < m.num_cells(); ++c) {
      for (const Index v : m.cell_vertices(c)) {
        if (m.vertex(v).norm() < 1e-12) marked.push_back(c);
      }
    }
    const Mesh<2> before = m;
    const RefinementStep r = adaptive.refine(marked);
    check_step(before, adaptive.mesh(), r);
    check_balance(adaptive);
    check_hanging(adaptive.mesh());
    CHECK(total_volume(adaptive.mesh()) == Approx(1.0));
  }
  CHECK(adaptive.max_level() == 5);
  // graded: far from the corner the mesh is still coarse
  const Mesh<2>& m = adaptive.mesh();
  Index coarse = 0;
  for (Index c = 0; c < m.num_cells(); ++c) {
    if (adaptive.level(c) == 0) ++coarse;
  }
  CHECK(coarse >= 1);
  CHECK(m.num_cells() < 8 * 32);
}

TEST_CASE("AdaptiveMesh::refine_all equals uniform refinement", "[mesh][adaptive]") {
  AdaptiveMesh<2> adaptive(rectangle(3, 2));
  const auto uniform = refine_uniform(adaptive.mesh());
  adaptive.refine_all();
  const Mesh<2>& m = adaptive.mesh();
  CHECK(m.is_conforming());
  CHECK(m.num_cells() == uniform.mesh.num_cells());
  CHECK(m.num_vertices() == uniform.mesh.num_vertices());
  CHECK(m.num_edges() == uniform.mesh.num_edges());
  CHECK(m.num_boundary_facets() == uniform.mesh.num_boundary_facets());
  for (const Tag tag : {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax}) {
    CHECK(count_tag(m, tag) == count_tag(uniform.mesh, tag));
  }
  std::set<std::pair<Real, Real>> a;
  std::set<std::pair<Real, Real>> b;
  for (Index v = 0; v < m.num_vertices(); ++v) {
    a.emplace(m.vertex(v)(0), m.vertex(v)(1));
    b.emplace(uniform.mesh.vertex(v)(0), uniform.mesh.vertex(v)(1));
  }
  CHECK(a == b);
}

TEST_CASE("AdaptiveMesh refines tetrahedra with hanging faces and edges", "[mesh][adaptive]") {
  AdaptiveMesh<3> adaptive(box(2, 2, 2));
  const Mesh<3> before = adaptive.mesh();
  const std::vector<Index> marked{0};
  const RefinementStep step = adaptive.refine(marked);
  const Mesh<3>& m = adaptive.mesh();
  CHECK(m.num_cells() == before.num_cells() - 1 + 8);
  CHECK(total_volume(m) == Approx(1.0));
  CHECK(!m.hanging_faces().empty());
  CHECK(!m.hanging_edges().empty());
  // every interior edge of the refined cell hangs (shared with unrefined cells)
  Index interior_edges = 0;
  for (const Index e : before.cell_edges(0)) {
    if (before.edge_cells(e).size() > 1) ++interior_edges;
  }
  CHECK(static_cast<Index>(m.hanging_edges().size()) == interior_edges);
  check_hanging(m);
  check_balance(adaptive);
  check_step(before, m, step);
  for (int i = 0; i < 3; ++i) {
    std::vector<Index> again;
    for (Index c = 0; c < adaptive.mesh().num_cells(); ++c) {
      if (adaptive.level(c) == adaptive.max_level()) {
        again.push_back(c);
        break;
      }
    }
    const Mesh<3> old = adaptive.mesh();
    const RefinementStep r = adaptive.refine(again);
    check_step(old, adaptive.mesh(), r);
    check_balance(adaptive);
    check_hanging(adaptive.mesh());
    CHECK(total_volume(adaptive.mesh()) == Approx(1.0));
  }
  Index tagged = 0;
  for (const Index f : adaptive.mesh().boundary_facets()) {
    if (adaptive.mesh().facet_tag(f) != kNoTag) ++tagged;
  }
  CHECK(tagged == adaptive.mesh().num_boundary_facets());
}

TEST_CASE("AdaptiveMesh keeps faces one-irregular across the inner child faces (3D)",
          "[mesh][adaptive]") {
  // the inner child of a face shares no vertex with the coarse neighbour, so vertex balance
  // alone would let the coarse face meet level-2 faces; every hanging face must be registered
  AdaptiveMesh<3> adaptive(box(2, 2, 2));
  for (int step = 0; step < 2; ++step) {
    std::vector<Index> marked;
    const Mesh<3>& m = adaptive.mesh();
    for (Index c = 0; c < m.num_cells(); ++c) {
      for (const Index v : m.cell_vertices(c)) {
        if (m.vertex(v).norm() < 1e-12) marked.push_back(c);
      }
    }
    adaptive.refine(marked);
  }
  for (int step = 0; step < 3; ++step) {
    const std::vector<Index> last{adaptive.mesh().num_cells() - 1 - 7 * step};
    adaptive.refine(last);
    const Mesh<3>& m = adaptive.mesh();
    check_hanging(m);
    check_balance(adaptive);
    for (const Index f : m.boundary_facets()) {
      Point<3> centroid = Point<3>::Zero();
      for (const Index v : m.facet_vertices(f)) centroid += m.vertex(v) / 3.0;
      bool on_box = false;
      for (int d = 0; d < 3; ++d) {
        on_box = on_box || std::abs(centroid(d)) < 1e-12 || std::abs(centroid(d) - 1.0) < 1e-12;
      }
      CHECK(on_box);
      CHECK(m.facet_tag(f) != kNoTag);
    }
    // every leaf facet with one cell that is not on the boundary is a registered hanging facet
    for (Index f = 0; f < m.num_facets(); ++f) {
      if (m.facet_cells(f)[1] == kInvalidIndex && !m.is_boundary_facet(f)) {
        CHECK(m.facet_hanging_role(f) != Mesh<3>::HangingRole::kNone);
      }
    }
  }
}

TEST_CASE("AdaptiveMesh::refine_all on a box equals uniform refinement", "[mesh][adaptive]") {
  AdaptiveMesh<3> adaptive(box(2, 1, 1));
  const auto uniform = refine_uniform(adaptive.mesh());
  adaptive.refine_all();
  CHECK(adaptive.mesh().is_conforming());
  CHECK(adaptive.mesh().num_cells() == uniform.mesh.num_cells());
  CHECK(adaptive.mesh().num_faces() == uniform.mesh.num_faces());
  CHECK(adaptive.mesh().num_boundary_facets() == uniform.mesh.num_boundary_facets());
  CHECK(count_tag(adaptive.mesh(), box_tag::kZMax) == count_tag(uniform.mesh, box_tag::kZMax));
}

TEST_CASE("AdaptiveMesh keeps curved boundaries", "[mesh][adaptive]") {
  AdaptiveMesh<2> adaptive(disc(2));
  std::vector<Index> marked;
  const Mesh<2>& root = adaptive.mesh();
  for (const Index f : root.boundary_facets()) marked.push_back(root.facet_cells(f)[0]);
  std::sort(marked.begin(), marked.end());
  marked.erase(std::unique(marked.begin(), marked.end()), marked.end());
  adaptive.refine(marked);
  const Mesh<2>& m = adaptive.mesh();
  REQUIRE(m.geometry_order() == 2);
  for (const Index f : m.boundary_facets()) {
    CHECK(m.facet_tag(f) == kDiscBoundary);
    for (const Index v : m.facet_vertices(f)) CHECK(m.vertex(v).norm() == Approx(1.0));
    // child edge nodes follow the quadratic root surface (as refine_uniform), not the circle
    CHECK(m.edge_node(f).norm() == Approx(1.0).margin(2e-3));
  }
  check_hanging(m);
  check_balance(adaptive);
}

TEST_CASE("AdaptiveMesh rejects bad input", "[mesh][adaptive]") {
  AdaptiveMesh<2> adaptive(rectangle(1, 1));
  const std::vector<Index> bad{5};
  CHECK_THROWS_AS(adaptive.refine(bad), hpfem::InvalidArgument);
  Mesh<2> m = rectangle(1, 1);
  Index shared = kInvalidIndex;
  Index boundary = kInvalidIndex;
  for (Index e = 0; e < m.num_edges(); ++e) (m.is_boundary_facet(e) ? boundary : shared) = e;
  CHECK_THROWS_AS(m.set_hanging({{shared, {boundary, boundary}, 0}}, {}), hpfem::InvalidArgument);
  CHECK_THROWS_AS(m.set_hanging({{boundary, {boundary, boundary}, 0}}, {}), hpfem::InvalidArgument);
  CHECK_THROWS_AS(m.set_hanging({{boundary, {boundary, boundary}, 99}}, {}),
                  hpfem::InvalidArgument);
}

TEST_CASE("extract cuts an L-shaped domain out of a rectangle", "[mesh][generators]") {
  const Mesh<2> square = rectangle(2, 2, Point<2>(-1.0, -1.0), Point<2>(1.0, 1.0));
  const Mesh<2> l = extract<2>(square, [](const Point<2>& x) { return !(x(0) > 0 && x(1) < 0); });
  CHECK(l.num_cells() == 6);
  CHECK(l.num_vertices() == 8);
  CHECK(l.num_boundary_facets() == 8);
  CHECK(total_volume(l) == Approx(3.0));
  CHECK(count_tag(l, box_tag::kXMin) == 2);
  CHECK(count_tag(l, box_tag::kXMax) == 1);
  CHECK(count_tag(l, box_tag::kYMin) == 1);
  CHECK(count_tag(l, box_tag::kYMax) == 2);
  Index untagged = 0;
  for (const Index f : l.boundary_facets()) {
    if (l.facet_tag(f) == kNoTag) ++untagged;
  }
  CHECK(untagged == 2);  // the two cut edges
  const std::vector<Index> twice{0, 0};
  CHECK_THROWS_AS(extract<2>(square, twice), hpfem::InvalidArgument);
}

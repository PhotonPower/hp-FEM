// p- and hp-refinement steps: order bookkeeping, inheritance by children, the minimum rule
// on shared entities, and exact transfer of solutions across a combined step.
#include <algorithm>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "hpfem/adaptivity/refinement.hpp"
#include "hpfem/assembly/h1_forms.hpp"
#include "hpfem/assembly/hanging_constraints.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/assembly/prolongation.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/adaptive_mesh.hpp"
#include "hpfem/mesh/detail/red_refinement.hpp"
#include "hpfem/mesh/generators.hpp"

using hpfem::as_size;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::adaptivity::hp_refine;
using hpfem::adaptivity::HpStep;
using hpfem::adaptivity::identity_step;
using hpfem::adaptivity::p_refine;
using hpfem::assembly::evaluate_h1;
using hpfem::assembly::evaluate_hcurl;
using hpfem::assembly::prolongate;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::AdaptiveMesh;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;

namespace {

Vector random_vector(Index n, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<Real> dist(-1.0, 1.0);
  Vector v(n);
  for (Index i = 0; i < n; ++i) v(i) = Complex{dist(gen), dist(gen)};
  return v;
}

/// The transferred function equals the old one at interior points of every new cell.
template <class Map>
void check_transfer(const Map& old_dofs, const Vector& u, const Map& new_dofs, const Vector& v,
                    const hpfem::mesh::RefinementStep& step) {
  REQUIRE(v.size() == new_dofs.num_dofs());
  const std::vector<Point<2>> points{Point<2>(0.2, 0.3), Point<2>(0.6, 0.1), Point<2>(0.1, 0.7)};
  for (Index c = 0; c < new_dofs.mesh().num_cells(); ++c) {
    for (const auto& xi : points) {
      const Point<2> xi_old = step.old_reference<2>(c, xi);
      if constexpr (std::is_same_v<Map, DofMap<2>>) {
        REQUIRE(std::abs(evaluate_h1(new_dofs, v, c, xi) -
                         evaluate_h1(old_dofs, u, step.parent[as_size(c)], xi_old)) < 1e-10);
      } else {
        REQUIRE((evaluate_hcurl(new_dofs, v, c, xi) -
                 evaluate_hcurl(old_dofs, u, step.parent[as_size(c)], xi_old))
                    .norm() < 1e-10);
      }
    }
  }
  const auto constraints = hpfem::assembly::hanging_constraints(new_dofs);
  for (Index s = 0; s < new_dofs.num_dofs(); ++s) {
    if (!constraints.is_constrained(s)) continue;
    Complex sum = 0;
    for (const auto& t : constraints.terms(s)) sum += t.coefficient * v(t.master);
    REQUIRE(std::abs(sum - v(s)) < 1e-10);
  }
}

}  // namespace

TEST_CASE("p_refine raises marked cells and caps at the maximum", "[adaptivity][refinement]") {
  const std::vector<int> orders{1, 2, 3, 4};
  const std::vector<Index> marked{0, 3};
  CHECK(p_refine(orders, marked) == std::vector<int>{2, 2, 3, 5});
  CHECK(p_refine(orders, marked, 2) == std::vector<int>{3, 2, 3, 6});
  CHECK(p_refine(orders, marked, 2, 4) == std::vector<int>{3, 2, 3, 4});
  CHECK(p_refine(orders, {}) == orders);
  const std::vector<Index> bad{4};
  CHECK_THROWS_AS(p_refine(orders, bad), hpfem::InvalidArgument);
  CHECK_THROWS_AS(p_refine(orders, marked, 0), hpfem::InvalidArgument);
}

TEST_CASE("hp_refine: children inherit, p-marked cells are raised, min rule on entities",
          "[adaptivity][refinement]") {
  AdaptiveMesh<2> adaptive(rectangle(2, 2));
  const std::vector<int> orders(8, 2);
  const std::vector<Index> h_marked{0};
  const std::vector<Index> p_marked{1, 7, 0};  // 0 is h-refined too: its children get 3
  const HpStep hp = hp_refine<2>(adaptive, orders, h_marked, p_marked, 1, 0, false);
  const Mesh<2>& m = adaptive.mesh();
  REQUIRE(hp.step.num_cells() == m.num_cells());
  REQUIRE(static_cast<Index>(hp.orders.size()) == m.num_cells());
  for (Index c = 0; c < m.num_cells(); ++c) {
    const Index parent = hp.step.parent[as_size(c)];
    const bool raised = parent == 0 || parent == 1 || parent == 7;
    CHECK(hp.orders[as_size(c)] == (raised ? 3 : 2));
  }
  const NedelecDofMap<2> dofs(m, hp.orders);
  for (Index e = 0; e < m.num_edges(); ++e) {
    int expected = 99;
    for (const Index c : m.edge_cells(e)) expected = std::min(expected, hp.orders[as_size(c)]);
    CHECK(dofs.edge_order(e) <= expected);
  }
  CHECK(dofs.max_order() == 3);
  const std::vector<int> wrong(3, 1);
  CHECK_THROWS_AS(hp_refine<2>(adaptive, wrong, {}, {}), hpfem::InvalidArgument);
}

TEST_CASE("hp_refine spreads p-refinement to lower-order facet neighbours",
          "[adaptivity][refinement]") {
  AdaptiveMesh<2> adaptive(rectangle(2, 2));
  const Mesh<2>& m = adaptive.mesh();
  std::vector<int> orders(8, 2);
  const std::vector<Index> p_marked{3};
  orders[3] = 4;  // raised to 5: neighbours at 2 go to 3 (one increment), not to 5
  std::vector<Index> neighbours;
  for (const Index n : m.cell_neighbors(3)) {
    if (n != hpfem::kInvalidIndex) neighbours.push_back(n);
  }
  REQUIRE(!neighbours.empty());
  const HpStep hp = hp_refine<2>(adaptive, orders, {}, p_marked);
  CHECK(hp.orders[3] == 5);
  for (Index c = 0; c < 8; ++c) {
    const bool neighbour = std::find(neighbours.begin(), neighbours.end(), c) != neighbours.end();
    CHECK(hp.orders[as_size(c)] == (c == 3 ? 5 : (neighbour ? 3 : 2)));
  }
  // an h-marked neighbour is split instead and its children keep their order
  AdaptiveMesh<2> other(rectangle(2, 2));
  const std::vector<Index> h_marked{neighbours.front()};
  const HpStep hp2 = hp_refine<2>(other, orders, h_marked, p_marked);
  for (Index c = 0; c < hp2.step.num_cells(); ++c) {
    if (hp2.step.parent[as_size(c)] == neighbours.front()) CHECK(hp2.orders[as_size(c)] == 2);
  }
}

TEST_CASE("solutions transfer exactly across p- and hp-steps", "[adaptivity][refinement]") {
  AdaptiveMesh<2> adaptive(rectangle(2, 2));
  const Mesh<2> old_mesh = adaptive.mesh();
  std::vector<int> orders(8, 1);
  orders[2] = 3;
  orders[5] = 2;
  const DofMap<2> old_h1(old_mesh, orders);
  const NedelecDofMap<2> old_nd(old_mesh, orders);
  const Vector u1 = random_vector(old_h1.num_dofs(), 11);
  const Vector u2 = random_vector(old_nd.num_dofs(), 12);

  SECTION("pure p-refinement on the same mesh") {
    const std::vector<Index> p_marked{0, 2, 4};
    const std::vector<int> raised = p_refine(orders, p_marked);
    const DofMap<2> new_h1(old_mesh, raised);
    const NedelecDofMap<2> new_nd(old_mesh, raised);
    const auto step = identity_step(old_mesh.num_cells());
    check_transfer(old_h1, u1, new_h1, prolongate(old_h1, u1, new_h1, step), step);
    check_transfer(old_nd, u2, new_nd, prolongate(old_nd, u2, new_nd, step), step);
    CHECK(new_nd.num_dofs() > old_nd.num_dofs());
  }
  SECTION("combined hp-step with hanging nodes") {
    const std::vector<Index> h_marked{2, 6};
    const std::vector<Index> p_marked{1, 2};
    const HpStep hp = hp_refine<2>(adaptive, orders, h_marked, p_marked);
    const DofMap<2> new_h1(adaptive.mesh(), hp.orders);
    const NedelecDofMap<2> new_nd(adaptive.mesh(), hp.orders);
    REQUIRE(!adaptive.mesh().is_conforming());
    check_transfer(old_h1, u1, new_h1, prolongate(old_h1, u1, new_h1, hp.step), hp.step);
    check_transfer(old_nd, u2, new_nd, prolongate(old_nd, u2, new_nd, hp.step), hp.step);
  }
}

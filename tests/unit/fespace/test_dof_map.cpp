#include <algorithm>
#include <random>
#include <set>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/fespace/h1_basis.hpp"
#include "hpfem/fespace/reference_element.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"

using Catch::Approx;
using hpfem::as_size;
using hpfem::Index;
using hpfem::kInvalidIndex;
using hpfem::LocalIndex;
using hpfem::Point;
using hpfem::Real;
using hpfem::fespace::DofMap;
using hpfem::fespace::H1Basis;
using hpfem::fespace::ReferenceElement;
using hpfem::mesh::affine_map;
using hpfem::mesh::box;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

template <int Dim>
Index expected_dofs(const Mesh<Dim>& m, int p) {
  Index n = m.num_vertices() + (p - 1) * m.num_edges();
  if constexpr (Dim == 2) {
    n += (p - 1) * (p - 2) / 2 * m.num_cells();
  } else {
    n += (p - 1) * (p - 2) / 2 * m.num_faces() + (p - 1) * (p - 2) * (p - 3) / 6 * m.num_cells();
  }
  return n;
}

/// Global function values at a physical point x seen from cell c, scattered by DoF id.
template <int Dim>
std::vector<Real> global_values(const DofMap<Dim>& dofs, Index c, const Point<Dim>& x) {
  const H1Basis<Dim> basis(dofs.cell_layout(c));
  std::vector<Real> local(as_size(basis.size()));
  basis.evaluate(affine_map(dofs.mesh(), c).to_reference(x), local, {});
  std::vector<Real> global(as_size(dofs.num_dofs()), 0.0);
  const auto ids = dofs.cell_dofs(c);
  REQUIRE(ids.size() == local.size());
  for (std::size_t i = 0; i < ids.size(); ++i) global[as_size(ids[i])] = local[i];
  return global;
}

/// On every interior facet the basis functions seen from both cells coincide for the facet
/// DoFs and vanish for all other DoFs of the two cells (H1 conformity).
template <int Dim>
void check_conformity(const DofMap<Dim>& dofs, unsigned seed) {
  const auto& m = dofs.mesh();
  std::mt19937 rng(seed);
  std::uniform_real_distribution<Real> u(0.05, 0.95);
  for (Index f = 0; f < m.num_facets(); ++f) {
    if (m.is_boundary_facet(f)) continue;
    const auto& fc = m.facet_cells(f);
    const auto& fl = m.facet_local_indices(f);
    const auto on_facet = dofs.facet_dofs(f);
    for (int sample = 0; sample < 3; ++sample) {
      // random point on the facet, generated through cell 0's local facet
      Point<Dim - 1> eta;
      Real sum = 0;
      for (int d = 0; d < Dim - 1; ++d) {
        eta(d) = u(rng);
        sum += eta(d);
      }
      if (Dim == 3 && sum > 1.0) eta *= 0.5 / sum;
      const Point<Dim> x =
          affine_map(m, fc[0]).to_physical(ReferenceElement<Dim>::facet_point(fl[0], eta));
      const auto a = global_values(dofs, fc[0], x);
      const auto b = global_values(dofs, fc[1], x);
      for (Index i = 0; i < dofs.num_dofs(); ++i) {
        if (std::binary_search(on_facet.begin(), on_facet.end(), i)) {
          REQUIRE(a[as_size(i)] == Approx(b[as_size(i)]).margin(1e-12));
        } else {
          REQUIRE(std::abs(a[as_size(i)]) < 1e-12);
          REQUIRE(std::abs(b[as_size(i)]) < 1e-12);
        }
      }
    }
  }
}

template <int Dim>
std::vector<int> random_orders(const Mesh<Dim>& m, int pmax, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(1, pmax);
  std::vector<int> orders(as_size(m.num_cells()));
  for (auto& p : orders) p = dist(rng);
  return orders;
}

}  // namespace

TEST_CASE("DofMap: uniform order counts", "[fespace][dofmap]") {
  const Mesh<2> r = rectangle(2, 2);
  const Mesh<3> b = box(1, 1, 1);
  for (int p = 1; p <= 5; ++p) {
    const DofMap<2> d2(r, p);
    REQUIRE(d2.num_dofs() == expected_dofs(r, p));
    REQUIRE(d2.max_order() == p);
    const DofMap<3> d3(b, p);
    REQUIRE(d3.num_dofs() == expected_dofs(b, p));
    for (Index c = 0; c < r.num_cells(); ++c) {
      REQUIRE(static_cast<Index>(d2.cell_dofs(c).size()) == H1Basis<2>(d2.cell_layout(c)).size());
    }
    for (Index c = 0; c < b.num_cells(); ++c) {
      REQUIRE(static_cast<Index>(d3.cell_dofs(c).size()) == H1Basis<3>(d3.cell_layout(c)).size());
    }
  }
  // all DoF ids 0..n-1 are used exactly by the union of cells
  const DofMap<3> d(b, 4);
  std::set<Index> used;
  for (Index c = 0; c < b.num_cells(); ++c) {
    for (const Index i : d.cell_dofs(c)) used.insert(i);
  }
  REQUIRE(static_cast<Index>(used.size()) == d.num_dofs());
  REQUIRE(*used.rbegin() == d.num_dofs() - 1);
}

TEST_CASE("DofMap: minimum rule on shared entities", "[fespace][dofmap]") {
  const Mesh<2> m({{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}}, {{0, 1, 2}, {0, 2, 3}});
  const DofMap<2> d(m, std::vector<int>{1, 3});
  REQUIRE(d.cell_order(0) == 1);
  REQUIRE(d.cell_order(1) == 3);
  const Index diagonal = m.edge_id(0, 2);
  REQUIRE(d.edge_order(diagonal) == 1);
  REQUIRE(d.edge_dofs(diagonal).empty());
  REQUIRE(d.edge_dofs(m.edge_id(2, 3)).size() == 2);
  REQUIRE(d.edge_dofs(m.edge_id(0, 1)).empty());
  REQUIRE(d.interior_dofs(0).empty());
  REQUIRE(d.interior_dofs(1).size() == 1);
  REQUIRE(d.num_dofs() == 4 + 2 * 2 + 1);
  REQUIRE(d.cell_dofs(0).size() == 3);
  REQUIRE(d.cell_dofs(1).size() == 3 + 4 + 1);
  const auto layout = d.cell_layout(1);
  REQUIRE(layout.cell_order == 3);
  REQUIRE(layout.edge_orders[0] == 1);  // local edge 0 of cell 1 = (0,2), the diagonal
  REQUIRE(layout.edge_orders[1] == 3);
}

TEST_CASE("DofMap: H1 conformity across interior facets (uniform and random orders)",
          "[fespace][dofmap]") {
  const Mesh<2> r = rectangle(2, 2, Point<2>(-1.0, 0.0), Point<2>(1.0, 1.5));
  check_conformity(DofMap<2>(r, 4), 1);
  check_conformity(DofMap<2>(r, random_orders(r, 5, 2)), 3);
  const Mesh<3> b = box(1, 1, 1, Point<3>(0.0, 0.0, 0.0), Point<3>(2.0, 1.0, 1.0));
  check_conformity(DofMap<3>(b, 4), 4);
  check_conformity(DofMap<3>(b, random_orders(b, 5, 5)), 6);
}

TEST_CASE("DofMap: facet DoFs and tagged boundaries", "[fespace][dofmap]") {
  const int p = 3;
  const Mesh<2> r = rectangle(2, 2);
  const DofMap<2> d2(r, p);
  REQUIRE(d2.facet_dofs(0).size() == 2 + (p - 1));
  // boundary of the 2 x 2 square: 8 vertices and 8 edges
  const auto bottom = d2.dofs_on_tag(box_tag::kYMin);
  REQUIRE(bottom.size() == 3 + 2 * (p - 1));
  REQUIRE(std::is_sorted(bottom.begin(), bottom.end()));
  std::vector<Index> all;
  for (const auto t : {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax}) {
    const auto on = d2.dofs_on_tag(t);
    all.insert(all.end(), on.begin(), on.end());
  }
  std::sort(all.begin(), all.end());
  all.erase(std::unique(all.begin(), all.end()), all.end());
  REQUIRE(all.size() == 8 + 8 * (p - 1));

  const Mesh<3> b = box(1, 1, 1);
  const DofMap<3> d3(b, p);
  REQUIRE(d3.facet_dofs(0).size() == 3 + 3 * (p - 1) + (p - 1) * (p - 2) / 2);
  const auto top = d3.dofs_on_tag(box_tag::kZMax);  // 4 vertices, 5 edges, 2 faces
  REQUIRE(top.size() == 4 + 5 * (p - 1) + 2 * (p - 1) * (p - 2) / 2);
}

TEST_CASE("DofMap: invalid orders are rejected", "[fespace][dofmap]") {
  const Mesh<2> r = rectangle(1, 1);
  REQUIRE_THROWS_AS(DofMap<2>(r, std::vector<int>{1}), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(DofMap<2>(r, std::vector<int>{1, 0}), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(DofMap<2>(r, 0), hpfem::InvalidArgument);
}

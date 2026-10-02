#include <algorithm>
#include <random>
#include <set>
#include <vector>

#include <Eigen/Geometry>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/fespace/nedelec_basis.hpp"
#include "hpfem/fespace/reference_element.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"

using Catch::Approx;
using hpfem::as_size;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::fespace::NedelecBasis;
using hpfem::fespace::NedelecDofMap;
using hpfem::fespace::ReferenceElement;
using hpfem::mesh::affine_map;
using hpfem::mesh::box;
using hpfem::mesh::Mesh;
using hpfem::mesh::outward_normal;
using hpfem::mesh::rectangle;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

template <int Dim>
Index expected_dofs(const Mesh<Dim>& m, int p) {
  Index n = p * m.num_edges();
  if constexpr (Dim == 2) {
    n += p * (p - 1) * m.num_cells();
  } else {
    n += p * (p - 1) * m.num_faces() + p * (p - 1) * (p - 2) / 2 * m.num_cells();
  }
  return n;
}

/// Physical values (covariant Piola: J^-T phi_hat) of all global functions of cell c at x.
template <int Dim>
std::vector<Point<Dim>> global_values(const NedelecDofMap<Dim>& dofs, Index c,
                                      const Point<Dim>& x) {
  const NedelecBasis<Dim> basis(dofs.cell_layout(c));
  const auto map = affine_map(dofs.mesh(), c);
  std::vector<Point<Dim>> local(as_size(basis.size()));
  basis.evaluate(map.to_reference(x), local, {});
  std::vector<Point<Dim>> global(as_size(dofs.num_dofs()), Point<Dim>::Zero());
  const auto ids = dofs.cell_dofs(c);
  REQUIRE(ids.size() == local.size());
  for (std::size_t i = 0; i < ids.size(); ++i)
    global[as_size(ids[i])] = map.inverse_transpose * local[i];
  return global;
}

/// Tangential component(s) w.r.t. the facet: t . v in 2D (edge tangent), n x v in 3D.
template <int Dim>
Point<3> tangential(const Point<Dim>& v, const Point<Dim>& normal) {
  if constexpr (Dim == 2) {
    return Point<3>(-normal(1) * v(0) + normal(0) * v(1), 0.0, 0.0);
  } else {
    return normal.cross(v);
  }
}

/// On every interior facet the tangential traces seen from both cells coincide for the
/// facet DoFs and vanish for all other DoFs of the two cells (H(curl) conformity).
template <int Dim>
void check_tangential_continuity(const NedelecDofMap<Dim>& dofs, unsigned seed) {
  const auto& m = dofs.mesh();
  std::mt19937 rng(seed);
  std::uniform_real_distribution<Real> u(0.05, 0.95);
  for (Index f = 0; f < m.num_facets(); ++f) {
    if (m.is_boundary_facet(f)) continue;
    const auto& fc = m.facet_cells(f);
    const auto& fl = m.facet_local_indices(f);
    const auto on_facet = dofs.facet_dofs(f);
    const Point<Dim> normal = outward_normal(m, fc[0], fl[0]);
    for (int sample = 0; sample < 3; ++sample) {
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
        const Point<3> ta = tangential<Dim>(a[as_size(i)], normal);
        const Point<3> tb = tangential<Dim>(b[as_size(i)], normal);
        if (std::binary_search(on_facet.begin(), on_facet.end(), i)) {
          REQUIRE((ta - tb).norm() < 1e-11);
        } else {
          REQUIRE(ta.norm() < 1e-11);
          REQUIRE(tb.norm() < 1e-11);
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

TEST_CASE("NedelecDofMap: counts and cell lists", "[fespace][nedelec][dofmap]") {
  const Mesh<2> r = rectangle(2, 2);
  const Mesh<3> b = box(1, 1, 1);
  for (int p = 1; p <= 4; ++p) {
    const NedelecDofMap<2> d2(r, p);
    REQUIRE(d2.num_dofs() == expected_dofs(r, p));
    for (Index c = 0; c < r.num_cells(); ++c) {
      REQUIRE(static_cast<Index>(d2.cell_dofs(c).size()) ==
              NedelecBasis<2>(d2.cell_layout(c)).size());
    }
    const NedelecDofMap<3> d3(b, p);
    REQUIRE(d3.num_dofs() == expected_dofs(b, p));
    for (Index c = 0; c < b.num_cells(); ++c) {
      REQUIRE(static_cast<Index>(d3.cell_dofs(c).size()) ==
              NedelecBasis<3>(d3.cell_layout(c)).size());
    }
  }
  const NedelecDofMap<3> d(b, 3);
  std::set<Index> used;
  for (Index c = 0; c < b.num_cells(); ++c) {
    for (const Index i : d.cell_dofs(c)) used.insert(i);
  }
  REQUIRE(static_cast<Index>(used.size()) == d.num_dofs());
  // no vertex DoFs: facet DoFs of a boundary face are its 3 edges x p plus p(p-1) face functions
  REQUIRE(d.facet_dofs(0).size() == 3 * 3 + 3 * 2);
  REQUIRE(d.dofs_on_tag(box_tag::kZMax).size() == 5 * 3 + 2 * 6);  // 5 edges, 2 faces on the top
}

TEST_CASE("NedelecDofMap: minimum rule", "[fespace][nedelec][dofmap]") {
  const Mesh<2> m({{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}}, {{0, 1, 2}, {0, 2, 3}});
  const NedelecDofMap<2> d(m, std::vector<int>{1, 3});
  REQUIRE(d.edge_dofs(m.edge_id(0, 2)).size() == 1);  // shared diagonal: order 1
  REQUIRE(d.edge_dofs(m.edge_id(2, 3)).size() == 3);
  REQUIRE(d.interior_dofs(0).empty());
  REQUIRE(d.interior_dofs(1).size() == 6);
  REQUIRE(d.num_dofs() == 3 * 1 + 2 * 3 + 6);
}

TEST_CASE("NedelecDofMap: tangential continuity across interior facets",
          "[fespace][nedelec][dofmap]") {
  const Mesh<2> r = rectangle(2, 2, Point<2>(-1.0, 0.0), Point<2>(1.0, 1.5));
  check_tangential_continuity(NedelecDofMap<2>(r, 3), 1);
  check_tangential_continuity(NedelecDofMap<2>(r, random_orders(r, 6, 2)), 3);
  const Mesh<3> b = box(1, 1, 1, Point<3>(0.0, 0.0, 0.0), Point<3>(2.0, 1.0, 1.0));
  check_tangential_continuity(NedelecDofMap<3>(b, 3), 4);
  check_tangential_continuity(NedelecDofMap<3>(b, random_orders(b, 3, 5)), 6);
}

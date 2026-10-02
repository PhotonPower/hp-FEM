// Prescribed tangential traces on the Nédélec space and the curl-source load.
#include <algorithm>
#include <random>
#include <vector>

#include <Eigen/SparseLU>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/dirichlet.hpp"
#include "hpfem/assembly/discrete_gradient.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/fespace/nedelec_basis.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"

using Catch::Approx;
using hpfem::as_size;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::SparseMatrix;
using hpfem::Vector;
using hpfem::assembly::assemble_maxwell;
using hpfem::assembly::ComplexCurl;
using hpfem::assembly::ComplexVector;
using hpfem::assembly::DirichletData;
using hpfem::assembly::discrete_gradient;
using hpfem::assembly::element_maxwell;
using hpfem::assembly::MaxwellForm;
using hpfem::assembly::tangential_dirichlet_values;
using hpfem::fespace::CellLayout;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecBasis;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::box;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kTol = 1e-9;

Vector solve(const SparseMatrix& a, const Vector& b) {
  const Eigen::SparseMatrix<Complex> column_major(a);
  Eigen::SparseLU<Eigen::SparseMatrix<Complex>> lu(column_major);
  REQUIRE(lu.info() == Eigen::Success);
  return lu.solve(b);
}

template <int Dim>
std::vector<Index> all_boundary(const Mesh<Dim>& m) {
  return {m.boundary_facets().begin(), m.boundary_facets().end()};
}

/// For a field in the discrete space the trace projection must return the boundary
/// coefficients of its global L2 projection, and constrain exactly the facet DoFs.
template <int Dim>
void check_trace_exact(const NedelecDofMap<Dim>& dofs, const std::vector<Index>& facets,
                       const std::type_identity_t<hpfem::assembly::ComplexVectorField<Dim>>& e) {
  MaxwellForm<Dim> form;
  form.source = e;
  const auto system = assemble_maxwell(dofs, form);
  const Vector e_h = solve(system.mass, system.rhs);
  const DirichletData data = tangential_dirichlet_values(dofs, std::span<const Index>(facets), e);
  REQUIRE(std::is_sorted(data.dofs.begin(), data.dofs.end()));
  std::vector<Index> expected;
  for (const Index f : facets) {
    const auto d = dofs.facet_dofs(f);
    expected.insert(expected.end(), d.begin(), d.end());
  }
  std::sort(expected.begin(), expected.end());
  expected.erase(std::unique(expected.begin(), expected.end()), expected.end());
  REQUIRE(data.dofs == expected);
  for (Index i = 0; i < data.size(); ++i) {
    REQUIRE(std::abs(data.values(i) - e_h(data.dofs[as_size(i)])) < kTol);
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

// fields in ND_1 (constant + x^perp / x cross e) and ND_2
ComplexVector<2> nd1_2d(const Point<2>& x) {
  return ComplexVector<2>(Complex{1.0 - 2.0 * x(1), 0.5}, Complex{3.0 + 2.0 * x(0), -1.0});
}
ComplexVector<2> nd2_2d(const Point<2>& x) {
  return ComplexVector<2>(Complex{1.0 + 2.0 * x(0) - x(1) - x(0) * x(1), x(1)},
                          Complex{3.0 + x(0) + x(1) + x(0) * x(0), -x(0)});
}
ComplexVector<3> nd1_3d(const Point<3>& x) {
  return ComplexVector<3>(Complex{1.0, 0.0}, Complex{2.0 + x(2), 1.0}, Complex{-x(1), 0.0});
}
ComplexVector<3> nd2_3d(const Point<3>& x) {
  // P1 field + x cross e_1 * x(2) (degree 2 with x . F = 0)
  return ComplexVector<3>(Complex{1.0 + x(1) - x(2), 0.0},
                          Complex{2.0 + 2.0 * x(2) + x(2) * x(2), 0.5},
                          Complex{x(0) - x(1) - x(1) * x(2), 0.0});
}

}  // namespace

TEST_CASE("tangential_dirichlet_values is exact for fields in the discrete space (2D)",
          "[assembly][dirichlet][maxwell]") {
  const Mesh<2> m = rectangle(3, 2, Point<2>(-1.0, 0.0), Point<2>(1.0, 1.5));
  for (int p = 1; p <= 3; ++p) {
    const NedelecDofMap<2> dofs(m, p);
    check_trace_exact<2>(dofs, all_boundary(m), nd1_2d);
    if (p >= 2) check_trace_exact<2>(dofs, all_boundary(m), nd2_2d);
  }
  const NedelecDofMap<2> mixed(m, random_orders(m, 3, 1));
  check_trace_exact<2>(mixed, all_boundary(m), nd1_2d);
  // the tag overload equals the facet overload
  const NedelecDofMap<2> dofs(m, 2);
  const auto facets = m.facets_with_tag(box_tag::kXMin);
  const DirichletData a = tangential_dirichlet_values(dofs, box_tag::kXMin, nd2_2d);
  const DirichletData b = tangential_dirichlet_values(dofs, std::span<const Index>(facets), nd2_2d);
  REQUIRE(a.dofs == b.dofs);
  REQUIRE((a.values - b.values).norm() < 1e-14);
  REQUIRE(a.size() == 2 * 2);  // two boundary edges with p = 2 functions each
}

TEST_CASE("tangential_dirichlet_values is exact for fields in the discrete space (3D)",
          "[assembly][dirichlet][maxwell]") {
  const Mesh<3> m = box(2, 1, 2, Point<3>(0.0, 0.0, 0.0), Point<3>(2.0, 1.0, 1.0));
  for (int p = 1; p <= 3; ++p) {
    const NedelecDofMap<3> dofs(m, p);
    check_trace_exact<3>(dofs, all_boundary(m), nd1_3d);
    if (p >= 2) check_trace_exact<3>(dofs, all_boundary(m), nd2_3d);
  }
  const NedelecDofMap<3> mixed(m, random_orders(m, 3, 2));
  check_trace_exact<3>(mixed, all_boundary(m), nd1_3d);
  const NedelecDofMap<3> dofs(m, 3);
  const auto facets = m.facets_with_tag(box_tag::kZMax);
  check_trace_exact<3>(dofs, {facets.begin(), facets.end()}, nd2_3d);
}

TEST_CASE("curl_source load: exact on the Whitney functions, zero on gradients",
          "[assembly][maxwell]") {
  // reference triangle, g constant: load_i = g * curl w_i * |K| = g for every Whitney function
  const Mesh<2> one({{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}}, {{0, 1, 2}});
  const NedelecBasis<2> basis(CellLayout<2>::uniform(1));
  MaxwellForm<2> form;
  form.curl_source = [](const Point<2>&) { return ComplexCurl<2>(Complex{0.7, -0.2}); };
  const auto geometry = hpfem::mesh::cell_geometry(one, 0);
  const auto local =
      element_maxwell(basis, *geometry, hpfem::assembly::simplex_quadrature<2>(2), form);
  for (Index i = 0; i < 3; ++i) REQUIRE(std::abs(local.load(i) - Complex{0.7, -0.2}) < 1e-14);

  // the load of a curl source vanishes on every discrete gradient: G^T b = 0
  const Mesh<2> r = rectangle(3, 2);
  const DofMap<2> h1(r, 3);
  const NedelecDofMap<2> nd(r, 3);
  MaxwellForm<2> form2;
  form2.curl_source = [](const Point<2>& x) {
    return ComplexCurl<2>(Complex{x(0) * x(1), std::sin(x(0))});
  };
  const auto system = assemble_maxwell(nd, form2);
  const Vector gtb = SparseMatrix(discrete_gradient(h1, nd).transpose()) * system.rhs;
  REQUIRE(gtb.norm() < 1e-13 * system.rhs.norm());

  const Mesh<3> b = box(1, 2, 1);
  const DofMap<3> h1_3(b, 2);
  const NedelecDofMap<3> nd_3(b, 2);
  MaxwellForm<3> form3;
  form3.curl_source = [](const Point<3>& x) {
    return ComplexCurl<3>(Complex{x(1), 0.0}, Complex{0.0, x(2)}, Complex{1.0, x(0)});
  };
  const auto system3 = assemble_maxwell(nd_3, form3);
  const Vector gtb3 = SparseMatrix(discrete_gradient(h1_3, nd_3).transpose()) * system3.rhs;
  REQUIRE(gtb3.norm() < 1e-13 * system3.rhs.norm());
}

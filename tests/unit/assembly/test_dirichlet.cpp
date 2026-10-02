#include <cmath>
#include <random>
#include <type_traits>
#include <vector>

#include <Eigen/Dense>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/dirichlet.hpp"
#include "hpfem/assembly/h1_forms.hpp"
#include "hpfem/assembly/sparse_assembler.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/fespace/reference_element.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"

using Catch::Approx;
using hpfem::as_size;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Matrix;
using hpfem::Point;
using hpfem::Real;
using hpfem::SparseMatrix;
using hpfem::Vector;
using hpfem::assembly::apply_dirichlet;
using hpfem::assembly::dirichlet_values;
using hpfem::assembly::DirichletData;
using hpfem::assembly::evaluate_h1;
using hpfem::assembly::homogeneous_dirichlet;
using hpfem::assembly::merge_dirichlet;
using hpfem::assembly::SparseAssembler;
using hpfem::fespace::DofMap;
using hpfem::fespace::ReferenceElement;
using hpfem::mesh::affine_map;
using hpfem::mesh::box;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

/// The boundary interpolant of g (zero interior DoFs) must equal g at random points on the
/// tagged facets when g lies in the trace space.
template <int Dim>
void check_trace_exact(const DofMap<Dim>& dofs, const std::vector<Index>& facets,
                       const std::type_identity_t<hpfem::assembly::ScalarField<Dim>>& g,
                       unsigned seed) {
  const DirichletData data = dirichlet_values(dofs, std::span<const Index>(facets), g);
  REQUIRE(std::is_sorted(data.dofs.begin(), data.dofs.end()));
  Vector u = Vector::Zero(dofs.num_dofs());
  for (Index i = 0; i < data.size(); ++i) u(data.dofs[as_size(i)]) = data.values(i);
  const auto& m = dofs.mesh();
  std::mt19937 rng(seed);
  std::uniform_real_distribution<Real> dist(0.0, 1.0);
  for (const Index f : facets) {
    const Index c = m.facet_cells(f)[0];
    const auto k = m.facet_local_indices(f)[0];
    for (int s = 0; s < 4; ++s) {
      Point<Dim - 1> eta;
      Real sum = 0;
      for (int d = 0; d < Dim - 1; ++d) {
        eta(d) = dist(rng);
        sum += eta(d);
      }
      if (Dim == 3 && sum > 1.0) eta *= 0.9 / sum;
      const Point<Dim> xi = ReferenceElement<Dim>::facet_point(k, eta);
      const Point<Dim> x = affine_map(m, c).to_physical(xi);
      REQUIRE(std::abs(evaluate_h1(dofs, u, c, xi) - g(x)) < 1e-10);
    }
  }
}

}  // namespace

TEST_CASE("apply_dirichlet eliminates symmetrically and keeps the size", "[assembly][dirichlet]") {
  // SPD 4 x 4 system
  Eigen::Matrix4cd dense;
  dense << 4, -1, 0, -1, -1, 4, -1, 0, 0, -1, 4, -1, -1, 0, -1, 4;
  SparseAssembler assembler(4, 4);
  for (Index i = 0; i < 4; ++i) {
    for (Index j = 0; j < 4; ++j) {
      if (dense(i, j) != Complex{0.0, 0.0}) assembler.add(i, j, dense(i, j));
    }
  }
  SparseMatrix a = assembler.finalize();
  Vector b(4);
  b << 1.0, 2.0, 3.0, 4.0;
  DirichletData data;
  data.dofs = {1, 3};
  data.values.resize(2);
  data.values << Complex{0.5, 0.0}, Complex{-2.0, 1.0};
  const Vector b_original = b;
  apply_dirichlet(a, b, data);

  REQUIRE(a.rows() == 4);
  REQUIRE(a.coeff(1, 1) == Complex{1.0, 0.0});
  REQUIRE(a.coeff(3, 3) == Complex{1.0, 0.0});
  REQUIRE(a.coeff(1, 0) == Complex{0.0, 0.0});
  REQUIRE(a.coeff(0, 1) == Complex{0.0, 0.0});
  REQUIRE(a.coeff(0, 0) == dense(0, 0));
  REQUIRE(a.coeff(0, 2) == dense(0, 2));
  REQUIRE(b(1) == data.values(0));
  REQUIRE(b(3) == data.values(1));
  // the free rows carry the moved columns: b_f - A_fc g_c
  REQUIRE(b(0) == b_original(0) - dense(0, 1) * data.values(0) - dense(0, 3) * data.values(1));
  REQUIRE(b(2) == b_original(2) - dense(2, 1) * data.values(0) - dense(2, 3) * data.values(1));
  // solving the reduced system equals the dense solution with the constraints enforced
  const Eigen::Vector2cd gc(data.values(0), data.values(1));
  Eigen::Matrix2cd aff;
  aff << dense(0, 0), dense(0, 2), dense(2, 0), dense(2, 2);
  Eigen::Matrix2cd afc;
  afc << dense(0, 1), dense(0, 3), dense(2, 1), dense(2, 3);
  const Eigen::Vector2cd bf(b_original(0), b_original(2));
  const Eigen::Vector2cd uf = aff.partialPivLu().solve(bf - afc * gc);
  const Eigen::MatrixXcd a_dense = Eigen::MatrixXcd(a);
  const Vector u = a_dense.partialPivLu().solve(b);
  REQUIRE(std::abs(u(0) - uf(0)) < 1e-14);
  REQUIRE(std::abs(u(2) - uf(1)) < 1e-14);
  REQUIRE(u(1) == data.values(0));

  DirichletData bad;
  bad.dofs = {7};
  bad.values = Vector::Zero(1);
  REQUIRE_THROWS_AS(apply_dirichlet(a, b, bad), hpfem::InvalidArgument);
}

TEST_CASE("dirichlet_values: hierarchical interpolation is exact for trace-space data",
          "[assembly][dirichlet]") {
  const Mesh<2> r = rectangle(3, 2, Point<2>(0.0, 0.0), Point<2>(1.5, 1.0));
  const DofMap<2> d2(r, 3);
  const std::vector<Index> boundary2(r.boundary_facets().begin(), r.boundary_facets().end());
  // linear data: vertex values only, every edge coefficient must vanish
  const DirichletData lin =
      dirichlet_values(d2, std::span<const Index>(boundary2),
                       [](const Point<2>& x) { return Complex{1.0 + 2.0 * x(0) - x(1), 0.0}; });
  for (Index i = 0; i < lin.size(); ++i) {
    if (lin.dofs[as_size(i)] >= r.num_vertices()) REQUIRE(std::abs(lin.values(i)) < 1e-13);
  }
  // cubic data lies in the p = 3 trace space: exact on every boundary edge
  check_trace_exact(
      d2, boundary2,
      [](const Point<2>& x) {
        return Complex{x(0) * x(0) * x(0) - 2.0 * x(0) * x(1) + x(1) * x(1), 0.5 * x(1)};
      },
      1);
  // a single tag
  const DirichletData bottom =
      dirichlet_values(d2, box_tag::kYMin, [](const Point<2>& x) { return Complex{x(0), 0.0}; });
  REQUIRE(bottom.size() == 4 + 3 * 2);

  const Mesh<3> b = box(1, 1, 1, Point<3>(0.0, 0.0, 0.0), Point<3>(1.0, 2.0, 1.0));
  const DofMap<3> d3(b, 3);
  const std::vector<Index> boundary3(b.boundary_facets().begin(), b.boundary_facets().end());
  check_trace_exact(
      d3, boundary3,
      [](const Point<3>& x) {
        return Complex{x(0) * x(1) * x(2) + x(2) * x(2) - x(0), x(1) * x(1) * x(1)};
      },
      2);
}

TEST_CASE("homogeneous_dirichlet and merge_dirichlet", "[assembly][dirichlet]") {
  const Mesh<2> r = rectangle(2, 2);
  const DofMap<2> d(r, 2);
  const auto left = r.facets_with_tag(box_tag::kXMin);
  const auto bottom = r.facets_with_tag(box_tag::kYMin);
  const DirichletData a = homogeneous_dirichlet(d, std::span<const Index>(left));
  REQUIRE(a.size() == 3 + 2);  // 3 vertices, 2 edges with one function each
  REQUIRE(a.values.norm() == 0.0);
  const DirichletData bv = dirichlet_values(d, std::span<const Index>(bottom),
                                            [](const Point<2>&) { return Complex{1.0, 0.0}; });
  const std::vector<DirichletData> parts{a, bv};
  const DirichletData merged = merge_dirichlet(parts);
  REQUIRE(merged.size() ==
          a.size() + bv.size() - 1);  // the corner (0,0) is shared: first value wins
  REQUIRE(std::is_sorted(merged.dofs.begin(), merged.dofs.end()));
  REQUIRE(merged.values(0) == Complex{0.0, 0.0});  // vertex 0 = corner from `a`
}

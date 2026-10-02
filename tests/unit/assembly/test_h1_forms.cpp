#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/dirichlet.hpp"
#include "hpfem/assembly/h1_forms.hpp"
#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/fespace/h1_basis.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/solvers/linear_solver.hpp"

using Catch::Approx;
using hpfem::as_size;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::assembly::apply_dirichlet;
using hpfem::assembly::assemble_h1;
using hpfem::assembly::dirichlet_values;
using hpfem::assembly::element_h1;
using hpfem::assembly::h1_error;
using hpfem::assembly::ScalarForm;
using hpfem::assembly::simplex_quadrature;
using hpfem::fespace::DofMap;
using hpfem::fespace::H1Basis;
using hpfem::fespace::H1Layout;
using hpfem::mesh::box;
using hpfem::mesh::cell_geometry;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::solvers::solve_direct;

namespace {

template <int Dim>
Eigen::Matrix<Complex, Dim, 1> constant_gradient(const Eigen::Matrix<Real, Dim, 1>& g) {
  return g.template cast<Complex>();
}

/// Patch test: with f = 0 and Dirichlet data from a linear u the discrete solution is u.
template <int Dim>
void patch_test(const Mesh<Dim>& m, int p, const Eigen::Matrix<Real, Dim, 1>& slope) {
  const DofMap<Dim> dofs(m, p);
  const auto u = [slope](const Point<Dim>& x) { return Complex{1.5 + slope.dot(x), 0.0}; };
  ScalarForm<Dim> form;
  form.diffusion = [](const Point<Dim>&) { return Complex{2.0, 0.0}; };
  auto system = assemble_h1(dofs, form);
  const std::vector<Index> boundary(m.boundary_facets().begin(), m.boundary_facets().end());
  apply_dirichlet(system.matrix, system.rhs,
                  dirichlet_values(dofs, std::span<const Index>(boundary), u));
  const Vector uh = solve_direct(system.matrix, system.rhs);
  const auto err =
      h1_error(dofs, uh, u, [slope](const Point<Dim>&) { return constant_gradient<Dim>(slope); });
  REQUIRE(err.l2 < 1e-11 * err.l2_norm);
  REQUIRE(err.h1_semi < 1e-10 * err.h1_norm);
  for (Index v = 0; v < m.num_vertices(); ++v) REQUIRE(std::abs(uh(v) - u(m.vertex(v))) < 1e-11);
}

}  // namespace

TEST_CASE("element matrices: mass sums to the volume, stiffness annihilates constants",
          "[assembly][h1]") {
  const Mesh<2> m = rectangle(1, 1, Point<2>(0.0, 0.0), Point<2>(2.0, 1.0));
  for (int p = 1; p <= 4; ++p) {
    const H1Basis<2> basis(H1Layout<2>::uniform(p));
    const auto geometry = cell_geometry(m, 0);
    ScalarForm<2> form;
    form.diffusion = [](const Point<2>&) { return Complex{1.0, 0.0}; };
    form.reaction = [](const Point<2>&) { return Complex{1.0, 0.0}; };
    form.source = [](const Point<2>&) { return Complex{1.0, 0.0}; };
    ScalarForm<2> mass_only;
    mass_only.reaction = form.reaction;
    const auto rule = simplex_quadrature<2>(2 * p + 1);
    const auto both = element_h1(basis, *geometry, rule, form);
    const auto mass = element_h1(basis, *geometry, rule, mass_only);
    REQUIRE(both.matrix.rows() == basis.size());
    // sum of all mass entries = int 1 = area (the vertex functions sum to one, higher
    // functions have zero-sum... use the load vector: int phi_i summed over the vertex
    // functions = area)
    Complex vertex_load = 0;
    for (Index i = 0; i < 3; ++i) vertex_load += both.vector(i);
    REQUIRE(std::abs(vertex_load - Complex{1.0, 0.0}) < 1e-13);  // triangle area 1
    const hpfem::Matrix stiffness = both.matrix - mass.matrix;
    // constants (coefficient 1 on the vertex functions, 0 elsewhere) are in the kernel
    // three leading ones (vertex functions), zeros elsewhere; built without indexed writes
    // because GCC's -Wnull-dereference misfires on them for vectors of unknown size
    std::vector<Complex> ones(3, Complex{1.0, 0.0});
    ones.resize(as_size(basis.size()), Complex{0.0, 0.0});
    Real residual = 0;
    for (Index i = 0; i < basis.size(); ++i) {
      Complex row_sum = 0;
      for (Index j = 0; j < basis.size(); ++j) row_sum += stiffness(i, j) * ones[as_size(j)];
      residual += std::norm(row_sum);
    }
    REQUIRE(std::sqrt(residual) < 1e-12);
    const hpfem::Matrix skew = stiffness - stiffness.transpose();
    REQUIRE(skew.norm() < 1e-12);
    const hpfem::Matrix mass_skew = mass.matrix - mass.matrix.transpose();
    REQUIRE(mass_skew.norm() < 1e-13);
  }
}

TEST_CASE("patch test: linear solutions are reproduced exactly (2D and 3D, p = 1..3)",
          "[assembly][h1]") {
  const Mesh<2> r = rectangle(3, 2, Point<2>(-1.0, 0.0), Point<2>(1.0, 1.0));
  for (int p = 1; p <= 3; ++p) patch_test(r, p, Eigen::Vector2d(0.7, -1.3));
  const Mesh<3> b = box(2, 1, 1);
  for (int p = 1; p <= 2; ++p) patch_test(b, p, Eigen::Vector3d(0.5, 1.0, -0.25));
}

TEST_CASE("assembled matrix is complex symmetric and the mass form integrates constants",
          "[assembly][h1]") {
  const Mesh<2> r = rectangle(2, 2, Point<2>(0.0, 0.0), Point<2>(3.0, 1.0));
  const DofMap<2> dofs(r, 3);
  ScalarForm<2> form;
  form.reaction = [](const Point<2>&) { return Complex{1.0, 0.5}; };
  form.source = [](const Point<2>&) { return Complex{1.0, 0.0}; };
  const auto system = assemble_h1(dofs, form);
  const hpfem::SparseMatrix diff = system.matrix - hpfem::SparseMatrix(system.matrix.transpose());
  REQUIRE(diff.norm() < 1e-12);
  // int 1 = sum of the loads of the vertex functions = area 3
  Complex total = 0;
  for (Index v = 0; v < r.num_vertices(); ++v) total += system.rhs(v);
  REQUIRE(std::abs(total - Complex{3.0, 0.0}) < 1e-12);
  // u = 1 (vertex coefficients 1): M u = load scaled by beta
  Vector one = Vector::Zero(dofs.num_dofs());
  for (Index v = 0; v < r.num_vertices(); ++v) one(v) = 1.0;
  const Vector mu = system.matrix * one;
  const Vector mass_residual = mu - Complex{1.0, 0.5} * system.rhs;
  REQUIRE(mass_residual.norm() < 1e-12);
}

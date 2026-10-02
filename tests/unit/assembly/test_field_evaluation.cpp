// Field evaluation at arbitrary physical points: L2 projections of polynomial fields that
// lie exactly in the discrete spaces must be reproduced (value and curl) wherever the point
// locator finds the point.
#include <random>
#include <vector>

#include <Eigen/SparseLU>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/h1_forms.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/point_location.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::SparseMatrix;
using hpfem::Vector;
using hpfem::assembly::assemble_h1;
using hpfem::assembly::assemble_maxwell;
using hpfem::assembly::ComplexCurl;
using hpfem::assembly::ComplexVector;
using hpfem::assembly::evaluate_h1;
using hpfem::assembly::evaluate_hcurl;
using hpfem::assembly::evaluate_hcurl_curl;
using hpfem::assembly::MaxwellForm;
using hpfem::assembly::ScalarForm;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::box;
using hpfem::mesh::Mesh;
using hpfem::mesh::PointLocator;
using hpfem::mesh::rectangle;

namespace {

constexpr Real kTol = 1e-10;

Vector solve(const SparseMatrix& a, const Vector& b) {
  const Eigen::SparseMatrix<Complex> column_major(a);
  Eigen::SparseLU<Eigen::SparseMatrix<Complex>> lu(column_major);
  REQUIRE(lu.info() == Eigen::Success);
  return lu.solve(b);
}

template <int Dim>
std::vector<Point<Dim>> random_points(int count, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<Real> u(0.0, 1.0);
  std::vector<Point<Dim>> points;
  for (int i = 0; i < count; ++i) {
    Point<Dim> x;
    for (int d = 0; d < Dim; ++d) x(d) = u(rng);
    points.push_back(x);
  }
  return points;
}

}  // namespace

TEST_CASE("evaluate_h1 at physical points reproduces an L2-projected cubic", "[assembly][h1]") {
  const Mesh<2> m = rectangle(4, 3);
  const DofMap<2> dofs(m, 3);
  const auto u = [](const Point<2>& x) {
    return Complex{x(0) * x(0) * x(1) + 2.0 * x(0) - x(1) * x(1) * x(1), 0.5 * x(0) * x(1)};
  };
  ScalarForm<2> form;
  form.reaction = [](const Point<2>&) { return Complex{1.0, 0.0}; };
  form.source = u;
  const auto system = assemble_h1(dofs, form);
  const Vector u_h = solve(system.matrix, system.rhs);
  const PointLocator<2> locator(m);
  for (const auto& x : random_points<2>(50, 7)) {
    const auto value = evaluate_h1(dofs, u_h, locator, x);
    REQUIRE(value.has_value());
    REQUIRE(std::abs(*value - u(x)) < kTol);
  }
  REQUIRE_FALSE(evaluate_h1(dofs, u_h, locator, Point<2>(1.5, 0.5)).has_value());
}

TEST_CASE("evaluate_hcurl / evaluate_hcurl_curl at physical points (2D, p = 2)",
          "[assembly][maxwell]") {
  // E = (1 + 2x - y - xy, 3 + x + y + x^2) = P1 field + x^perp (x) lies in ND_2, curl = 2 + 3x
  const Mesh<2> m = rectangle(3, 4);
  const NedelecDofMap<2> dofs(m, 2);
  const auto field = [](const Point<2>& x) {
    return ComplexVector<2>(Complex{1.0 + 2.0 * x(0) - x(1) - x(0) * x(1), 0.0},
                            Complex{3.0 + x(0) + x(1) + x(0) * x(0), 0.0});
  };
  MaxwellForm<2> form;
  form.source = field;
  const auto system = assemble_maxwell(dofs, form);
  const Vector e_h = solve(system.mass, system.rhs);
  const PointLocator<2> locator(m);
  for (const auto& x : random_points<2>(50, 8)) {
    const auto value = evaluate_hcurl(dofs, e_h, locator, x);
    REQUIRE(value.has_value());
    REQUIRE((*value - field(x)).norm() < kTol);
    const auto curl = evaluate_hcurl_curl(dofs, e_h, locator, x);
    REQUIRE(curl.has_value());
    REQUIRE(std::abs((*curl)(0) - Complex{2.0 + 3.0 * x(0), 0.0}) < kTol);
    // the per-cell overloads agree with the located point
    const auto located = locator.locate(x);
    REQUIRE((evaluate_hcurl(dofs, e_h, located->cell, located->xi) - *value).norm() < 1e-14);
    REQUIRE((evaluate_hcurl_curl(dofs, e_h, located->cell, located->xi) - *curl).norm() < 1e-14);
  }
  REQUIRE_FALSE(evaluate_hcurl(dofs, e_h, locator, Point<2>(-0.2, 0.5)).has_value());
  REQUIRE_FALSE(evaluate_hcurl_curl(dofs, e_h, locator, Point<2>(0.5, 1.2)).has_value());
}

TEST_CASE("evaluate_hcurl / evaluate_hcurl_curl at physical points (3D, p = 2)",
          "[assembly][maxwell]") {
  // E = (1 + y - z, 2 + 2z, x - y) = P1 field + x cross e_1 lies in ND_2, curl = (-3, -2, -1)
  const Mesh<3> m = box(2, 2, 2);
  const NedelecDofMap<3> dofs(m, 2);
  const auto field = [](const Point<3>& x) {
    return ComplexVector<3>(Complex{1.0 + x(1) - x(2), 0.0}, Complex{2.0 + 2.0 * x(2), 0.0},
                            Complex{x(0) - x(1), 0.0});
  };
  const ComplexCurl<3> exact_curl(Complex{-3.0, 0.0}, Complex{-2.0, 0.0}, Complex{-1.0, 0.0});
  MaxwellForm<3> form;
  form.source = field;
  const auto system = assemble_maxwell(dofs, form);
  const Vector e_h = solve(system.mass, system.rhs);
  const PointLocator<3> locator(m);
  for (const auto& x : random_points<3>(30, 9)) {
    const auto value = evaluate_hcurl(dofs, e_h, locator, x);
    REQUIRE(value.has_value());
    REQUIRE((*value - field(x)).norm() < kTol);
    const auto curl = evaluate_hcurl_curl(dofs, e_h, locator, x);
    REQUIRE(curl.has_value());
    REQUIRE((*curl - exact_curl).norm() < kTol);
  }
  REQUIRE_FALSE(evaluate_hcurl(dofs, e_h, locator, Point<3>(0.5, 0.5, 1.1)).has_value());
}

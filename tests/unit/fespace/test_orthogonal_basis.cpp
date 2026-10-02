#include <cmath>
#include <vector>

#include <Eigen/Dense>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/orthogonal_basis.hpp"
#include "hpfem/fespace/polynomials.hpp"

using Catch::Approx;
using hpfem::as_size;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::fespace::DubinerBasis;
using hpfem::fespace::jacobi;
using hpfem::fespace::legendre;

TEST_CASE("Jacobi polynomials: Legendre for (0,0), explicit values, value at 1", "[fespace]") {
  const int n = 6;
  std::vector<Real> p(n + 1);
  std::vector<Real> l(n + 1);
  std::vector<Real> dl(n + 1);
  for (const Real x : {-0.9, -0.2, 0.3, 0.8}) {
    jacobi(n, 0.0, 0.0, x, p);
    legendre(n, x, l, dl);
    for (int i = 0; i <= n; ++i) REQUIRE(p[as_size(i)] == Approx(l[as_size(i)]));
    // P_1^{(a,b)} = (a + 1) + (a + b + 2)(x - 1)/2, P_2^{(1,0)} explicit
    jacobi(2, 1.0, 0.0, x, p);
    REQUIRE(p[1] == Approx(2.0 + 1.5 * (x - 1.0)));
    REQUIRE(p[2] == Approx(0.5 * (5.0 * x * x + 2.0 * x - 1.0) * 1.0 + 0.0 * x).margin(1e-12));
    jacobi(3, 2.0, 0.5, x, p);
    REQUIRE(p[1] == Approx(3.0 + 2.25 * (x - 1.0)));
  }
  // P_n^{(a,b)}(1) = binom(n + a, n)
  jacobi(4, 3.0, 1.0, 1.0, p);
  REQUIRE(p[1] == Approx(4.0));
  REQUIRE(p[2] == Approx(10.0));
  REQUIRE(p[3] == Approx(20.0));
  REQUIRE(p[4] == Approx(35.0));
}

TEST_CASE("Dubiner basis is orthonormal on the unit triangle and ordered by degree", "[fespace]") {
  for (int p = 0; p <= 5; ++p) {
    const DubinerBasis<2> basis(p);
    REQUIRE(basis.size() == (p + 1) * (p + 2) / 2);
    const auto rule = hpfem::assembly::simplex_quadrature<2>(2 * p);
    Eigen::MatrixXd gram = Eigen::MatrixXd::Zero(basis.size(), basis.size());
    std::vector<Real> values(as_size(basis.size()));
    for (std::size_t q = 0; q < rule.size(); ++q) {
      basis.evaluate(rule.points[q], values);
      const Eigen::Map<const Eigen::VectorXd> v(values.data(), basis.size());
      gram += rule.weights[q] * v * v.transpose();
    }
    REQUIRE((gram - Eigen::MatrixXd::Identity(basis.size(), basis.size())).norm() < 1e-11);
    int previous = 0;
    for (Index i = 0; i < basis.size(); ++i) {
      REQUIRE(basis.degree(i) >= previous);
      REQUIRE(basis.degree(i) <= p);
      previous = basis.degree(i);
    }
    REQUIRE(basis.degree(basis.size() - 1) == p);
  }
  CHECK_THROWS_AS(DubinerBasis<2>(-1), hpfem::InvalidArgument);
}

TEST_CASE("Dubiner basis is orthonormal on the unit tetrahedron", "[fespace]") {
  for (int p = 0; p <= 4; ++p) {
    const DubinerBasis<3> basis(p);
    REQUIRE(basis.size() == (p + 1) * (p + 2) * (p + 3) / 6);
    const auto rule = hpfem::assembly::simplex_quadrature<3>(2 * p);
    Eigen::MatrixXd gram = Eigen::MatrixXd::Zero(basis.size(), basis.size());
    std::vector<Real> values(as_size(basis.size()));
    for (std::size_t q = 0; q < rule.size(); ++q) {
      basis.evaluate(rule.points[q], values);
      const Eigen::Map<const Eigen::VectorXd> v(values.data(), basis.size());
      gram += rule.weights[q] * v * v.transpose();
    }
    REQUIRE((gram - Eigen::MatrixXd::Identity(basis.size(), basis.size())).norm() < 1e-11);
    REQUIRE(basis.degree(basis.size() - 1) == p);
  }
}

TEST_CASE("Dubiner functions of degree n span the polynomials of degree n", "[fespace]") {
  // x^2 y on the triangle has an exact expansion with no coefficient beyond degree 3
  const DubinerBasis<2> basis(5);
  const auto rule = hpfem::assembly::simplex_quadrature<2>(12);
  Eigen::VectorXd coefficients = Eigen::VectorXd::Zero(basis.size());
  std::vector<Real> values(as_size(basis.size()));
  for (std::size_t q = 0; q < rule.size(); ++q) {
    basis.evaluate(rule.points[q], values);
    const Real f = rule.points[q](0) * rule.points[q](0) * rule.points[q](1);
    for (Index i = 0; i < basis.size(); ++i)
      coefficients(i) += rule.weights[q] * f * values[as_size(i)];
  }
  for (Index i = 0; i < basis.size(); ++i) {
    if (basis.degree(i) > 3) CHECK(std::abs(coefficients(i)) < 1e-13);
  }
  CHECK(coefficients.norm() > 0.01);
}

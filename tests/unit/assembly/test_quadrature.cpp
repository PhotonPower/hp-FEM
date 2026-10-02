#include <cmath>
#include <numbers>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/reference_element.hpp"

using Catch::Approx;
using hpfem::Point;
using hpfem::Real;
using hpfem::assembly::gauss_jacobi;
using hpfem::assembly::gauss_legendre;
using hpfem::assembly::QuadratureRule;
using hpfem::assembly::simplex_quadrature;
using hpfem::fespace::ReferenceElement;

namespace {

Real factorial(int n) {
  return std::tgamma(static_cast<Real>(n) + 1.0);
}

/// Exact integral of x^a y^b (z^c) over the reference simplex: a! b! (c!) / (a+b(+c)+Dim)!
Real exact_monomial(int a, int b, int c = 0) {
  if (c < 0) return factorial(a) * factorial(b) / factorial(a + b + 2);
  return factorial(a) * factorial(b) * factorial(c) / factorial(a + b + c + 3);
}

template <int Dim>
Real integrate_monomial(const QuadratureRule<Dim>& rule, int a, int b, int c) {
  Real sum = 0;
  for (std::size_t q = 0; q < rule.size(); ++q) {
    const auto& p = rule.points[q];
    Real m = std::pow(p(0), a) * std::pow(p(1), b);
    if constexpr (Dim == 3) m *= std::pow(p(2), c);
    sum += rule.weights[q] * m;
  }
  return sum;
}

template <int Dim>
void check_simplex_rule(int order) {
  const auto rule = simplex_quadrature<Dim>(order);
  const int n = (order + 2) / 2;
  REQUIRE(rule.order >= order);
  REQUIRE(rule.size() == static_cast<std::size_t>(std::pow(n, Dim)));
  Real total = 0;
  for (std::size_t q = 0; q < rule.size(); ++q) {
    REQUIRE(rule.weights[q] > 0);
    REQUIRE(ReferenceElement<Dim>::contains(rule.points[q], 0.0));  // strictly inside
    total += rule.weights[q];
  }
  REQUIRE(total == Approx(ReferenceElement<Dim>::volume()).epsilon(1e-14));
  // every monomial of total degree <= order is integrated to relative precision 1e-12
  for (int a = 0; a <= order; ++a) {
    for (int b = 0; a + b <= order; ++b) {
      if constexpr (Dim == 2) {
        const Real exact = exact_monomial(a, b, -1);
        REQUIRE(integrate_monomial<2>(rule, a, b, 0) == Approx(exact).epsilon(1e-12));
      } else {
        for (int c = 0; a + b + c <= order; ++c) {
          const Real exact = exact_monomial(a, b, c);
          REQUIRE(integrate_monomial<3>(rule, a, b, c) == Approx(exact).epsilon(1e-12));
        }
      }
    }
  }
}

}  // namespace

TEST_CASE("Gauss-Legendre on [0,1]", "[assembly][quadrature]") {
  const auto two = gauss_legendre(2);
  REQUIRE(two.size() == 2);
  REQUIRE(two.order == 3);
  REQUIRE(two.points[0](0) == Approx(0.5 - 0.5 / std::sqrt(3.0)));
  REQUIRE(two.points[1](0) == Approx(0.5 + 0.5 / std::sqrt(3.0)));
  REQUIRE(two.weights[0] == Approx(0.5));
  REQUIRE(two.weights[1] == Approx(0.5));
  for (int n = 1; n <= 12; ++n) {
    const auto rule = gauss_legendre(n);
    REQUIRE(rule.size() == static_cast<std::size_t>(n));
    for (int k = 0; k <= 2 * n - 1; ++k) {  // int_0^1 s^k ds = 1/(k+1)
      Real sum = 0;
      for (std::size_t q = 0; q < rule.size(); ++q)
        sum += rule.weights[q] * std::pow(rule.points[q](0), k);
      REQUIRE(sum == Approx(1.0 / (k + 1)).epsilon(1e-13));
    }
    // degree 2n is not exact (sanity check that the order claim is sharp)
    if (n <= 4) {
      Real sum = 0;
      for (std::size_t q = 0; q < rule.size(); ++q)
        sum += rule.weights[q] * std::pow(rule.points[q](0), 2 * n);
      REQUIRE(std::abs(sum - 1.0 / (2 * n + 1)) > 1e-6);
    }
  }
  REQUIRE_THROWS_AS(gauss_legendre(0), hpfem::InvalidArgument);
}

TEST_CASE("Gauss-Jacobi on [0,1] integrates s^k (1-s)^alpha s^beta exactly",
          "[assembly][quadrature]") {
  // int_0^1 s^k (1-s)^a s^b ds = B(k+b+1, a+1)
  const auto beta_fn = [](Real x, Real y) {
    return std::tgamma(x) * std::tgamma(y) / std::tgamma(x + y);
  };
  for (const Real alpha : {0.0, 1.0, 2.0, 0.5}) {
    for (const Real beta : {0.0, 1.0, -0.5}) {
      for (int n = 1; n <= 8; ++n) {
        const auto rule = gauss_jacobi(n, alpha, beta);
        REQUIRE(rule.order == 2 * n - 1);
        for (int k = 0; k <= 2 * n - 1; ++k) {
          Real sum = 0;
          for (std::size_t q = 0; q < rule.size(); ++q) {
            REQUIRE(rule.points[q](0) > 0.0);
            REQUIRE(rule.points[q](0) < 1.0);
            sum += rule.weights[q] * std::pow(rule.points[q](0), k);
          }
          REQUIRE(sum == Approx(beta_fn(k + beta + 1.0, alpha + 1.0)).epsilon(1e-12));
        }
      }
    }
  }
  REQUIRE_THROWS_AS(gauss_jacobi(3, -1.0, 0.0), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(gauss_jacobi(3, 0.0, -1.5), hpfem::InvalidArgument);
}

TEST_CASE("triangle rules are exact up to order 20", "[assembly][quadrature]") {
  for (const int order : {0, 1, 2, 3, 4, 5, 7, 10, 13, 16, 20}) check_simplex_rule<2>(order);
  REQUIRE(simplex_quadrature<2>(1).size() == 1);  // the centroid rule
  REQUIRE((simplex_quadrature<2>(1).points[0] - Point<2>(1.0 / 3.0, 1.0 / 3.0)).norm() < 1e-14);
  REQUIRE_THROWS_AS(simplex_quadrature<2>(-1), hpfem::InvalidArgument);
}

TEST_CASE("tetrahedron rules are exact up to order 20", "[assembly][quadrature]") {
  for (const int order : {0, 1, 2, 3, 4, 6, 9, 12, 16, 20}) check_simplex_rule<3>(order);
  REQUIRE((simplex_quadrature<3>(1).points[0] - Point<3>::Constant(0.25)).norm() < 1e-14);
}

TEST_CASE("1D simplex rule is Gauss-Legendre", "[assembly][quadrature]") {
  const auto a = simplex_quadrature<1>(7);
  const auto b = gauss_legendre(4);
  REQUIRE(a.size() == b.size());
  for (std::size_t q = 0; q < a.size(); ++q) {
    REQUIRE(a.points[q](0) == b.points[q](0));
    REQUIRE(a.weights[q] == b.weights[q]);
  }
}

TEST_CASE("a transcendental integrand converges with the order", "[assembly][quadrature]") {
  // int_T exp(x + y) dA = int_0^1 (e - e^x) dx = e - (e - 1) = 1
  const Real exact = 1.0;
  Real previous = 1.0;
  for (const int order : {1, 3, 5, 9, 13}) {
    const auto rule = simplex_quadrature<2>(order);
    Real sum = 0;
    for (std::size_t q = 0; q < rule.size(); ++q) {
      sum += rule.weights[q] * std::exp(rule.points[q](0) + rule.points[q](1));
    }
    const Real error = std::abs(sum - exact);
    REQUIRE(error < previous);
    previous = error;
  }
  REQUIRE(previous < 1e-14);
}

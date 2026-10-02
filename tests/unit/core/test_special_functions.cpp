#include <cmath>
#include <numbers>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/core/special_functions.hpp"

using Catch::Approx;
using hpfem::bessel_j;
using hpfem::bessel_y;
using hpfem::Complex;
using hpfem::hankel1;
using hpfem::hankel1_derivative;
using hpfem::Real;

TEST_CASE("Bessel functions: known values, Wronskian, negative orders", "[core][special]") {
  REQUIRE(bessel_j(0, 0.0) == Approx(1.0));
  REQUIRE(bessel_j(1, 2.0) == Approx(0.576724807756873));
  REQUIRE(bessel_y(0, 2.0) == Approx(0.510375672649745));
  REQUIRE(bessel_y(1, 0.5) == Approx(-1.471472392670243));
  // Wronskian J_n Y_{n+1} - J_{n+1} Y_n = -2 / (pi x)
  for (const Real x : {0.3, 1.0, 2.5, 7.0}) {
    for (int n = 0; n <= 3; ++n) {
      const Real w = bessel_j(n, x) * bessel_y(n + 1, x) - bessel_j(n + 1, x) * bessel_y(n, x);
      REQUIRE(w == Approx(-2.0 / (std::numbers::pi * x)).epsilon(1e-12));
    }
  }
  // J_{-n} = (-1)^n J_n, Y_{-n} = (-1)^n Y_n
  REQUIRE(bessel_j(-1, 1.5) == Approx(-bessel_j(1, 1.5)));
  REQUIRE(bessel_j(-2, 1.5) == Approx(bessel_j(2, 1.5)));
  REQUIRE(bessel_y(-3, 1.5) == Approx(-bessel_y(3, 1.5)));
}

TEST_CASE("Hankel function and its derivative", "[core][special]") {
  const Real x = 1.7;
  REQUIRE(hankel1(0, x).real() == Approx(bessel_j(0, x)));
  REQUIRE(hankel1(0, x).imag() == Approx(bessel_y(0, x)));
  // derivative by central differences; H_0' = -H_1
  const Real h = 1e-6;
  for (int n = 0; n <= 3; ++n) {
    const Complex fd = (hankel1(n, x + h) - hankel1(n, x - h)) / (2 * h);
    REQUIRE(std::abs(fd - hankel1_derivative(n, x)) < 1e-8);
  }
  REQUIRE(std::abs(hankel1_derivative(0, x) + hankel1(1, x)) < 1e-14);
  // large-argument asymptotics: |H_0| ~ sqrt(2 / (pi x))
  const Real big = 200.0;
  REQUIRE(std::abs(hankel1(0, big)) ==
          Approx(std::sqrt(2.0 / (std::numbers::pi * big))).epsilon(1e-4));
}

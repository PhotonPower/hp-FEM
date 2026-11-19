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

TEST_CASE("bessel_i_scaled: values, the recurrence and the normalisation", "[core][special]") {
  using hpfem::bessel_i_scaled;
  // reference values e^{-x} I_n(x) (scipy.special.ive)
  const struct {
    int n;
    double x;
    double value;
  } cases[] = {{0, 1.0, 0.4657596075936404},       {1, 1.0, 0.20791041534970842},
               {5, 10.0, 0.03528429361493396},     {0, 1e4, 0.0039894726746047314},
               {80, 1e4, 0.0028969103306616},      {150, 1e4, 0.001295146612430127},
               {3, 1e-3, 2.0812511713977277e-11},  {40, 2.0, 1.6996341282984332e-49},
               {100, 250.0, 6.467102394401546e-11}};
  for (const auto& c : cases) {
    const auto values = bessel_i_scaled(c.n, c.x);
    REQUIRE(values.size() == static_cast<std::size_t>(c.n) + 1);
    REQUIRE(values.back() == Catch::Approx(c.value).epsilon(1e-12));
  }
  // the recurrence I_{n-1} - I_{n+1} = (2n / x) I_n and e^{-x}(I_0 + 2 Σ I_k) = 1
  for (const double x : {0.3, 7.0, 500.0}) {
    const auto v = bessel_i_scaled(60, x);
    for (int n = 1; n < 60; ++n) {
      const auto k = static_cast<std::size_t>(n);
      REQUIRE(v[k - 1] - v[k + 1] ==
              Catch::Approx(2.0 * n / x * v[k]).epsilon(1e-12).margin(1e-300));
    }
  }
  const auto zero = bessel_i_scaled(3, 0.0);
  REQUIRE(zero[0] == 1.0);
  REQUIRE(zero[3] == 0.0);
  REQUIRE_THROWS_AS(bessel_i_scaled(-1, 1.0), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(bessel_i_scaled(1, -1.0), hpfem::InvalidArgument);
}

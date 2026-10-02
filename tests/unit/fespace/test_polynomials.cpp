#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/fespace/polynomials.hpp"

using Catch::Approx;
using hpfem::Real;
using hpfem::fespace::legendre;
using hpfem::fespace::scaled_integrated_legendre;

namespace {

/// Plain integrated Legendre L_i(x) = int_{-1}^x P_{i-1}, i >= 2; L_1 = x.
Real integrated_legendre_explicit(int i, Real x) {
  switch (i) {
    case 1:
      return x;
    case 2:
      return (x * x - 1.0) / 2.0;
    case 3:
      return x * (x * x - 1.0) / 2.0;
    case 4:
      return (5.0 * x * x - 1.0) * (x * x - 1.0) / 8.0;
    default:
      return 0.0;
  }
}

}  // namespace

TEST_CASE("Legendre polynomials: explicit values, endpoints, derivatives", "[fespace]") {
  const int n = 6;
  std::vector<Real> p(n + 1);
  std::vector<Real> dp(n + 1);
  for (const Real x : {-1.0, -0.3, 0.0, 0.7, 1.0}) {
    legendre(n, x, p, dp);
    REQUIRE(p[0] == 1.0);
    REQUIRE(p[1] == x);
    REQUIRE(p[2] == Approx((3.0 * x * x - 1.0) / 2.0));
    REQUIRE(p[3] == Approx((5.0 * x * x * x - 3.0 * x) / 2.0));
    REQUIRE(p[4] == Approx((35.0 * std::pow(x, 4) - 30.0 * x * x + 3.0) / 8.0));
    // finite-difference derivative
    const Real h = 1e-6;
    std::vector<Real> plus(n + 1);
    std::vector<Real> minus(n + 1);
    std::vector<Real> dummy(n + 1);
    legendre(n, x + h, plus, dummy);
    legendre(n, x - h, minus, dummy);
    for (int i = 0; i <= n; ++i) {
      REQUIRE(
          dp[static_cast<std::size_t>(i)] ==
          Approx((plus[static_cast<std::size_t>(i)] - minus[static_cast<std::size_t>(i)]) / (2 * h))
              .margin(1e-7));
    }
  }
  legendre(n, 1.0, p, dp);
  for (const Real v : p) REQUIRE(v == Approx(1.0));
  legendre(n, -1.0, p, dp);
  for (int i = 0; i <= n; ++i)
    REQUIRE(p[static_cast<std::size_t>(i)] == Approx(i % 2 == 0 ? 1.0 : -1.0));
}

TEST_CASE("scaled integrated Legendre: explicit values, homogeneity, zeros, derivatives",
          "[fespace]") {
  const int n = 7;
  std::vector<Real> v(n + 1);
  std::vector<Real> dx(n + 1);
  std::vector<Real> dt(n + 1);

  SECTION("t = 1 gives the plain integrated Legendre polynomials") {
    for (const Real x : {-0.9, -0.2, 0.4, 1.0}) {
      scaled_integrated_legendre(n, x, 1.0, v, dx, dt);
      for (int i = 1; i <= 4; ++i) {
        REQUIRE(v[static_cast<std::size_t>(i)] == Approx(integrated_legendre_explicit(i, x)));
      }
    }
  }

  SECTION("homogeneity L_i^S(x, t) = t^i L_i(x / t) and zeros at x = +-t") {
    std::vector<Real> ref(n + 1);
    std::vector<Real> d1(n + 1);
    std::vector<Real> d2(n + 1);
    for (const Real t : {0.3, 1.0, 2.5}) {
      for (const Real r : {-1.0, -0.5, 0.1, 1.0}) {
        const Real x = r * t;
        scaled_integrated_legendre(n, x, t, v, dx, dt);
        scaled_integrated_legendre(n, r, 1.0, ref, d1, d2);
        for (int i = 1; i <= n; ++i) {
          REQUIRE(v[static_cast<std::size_t>(i)] ==
                  Approx(std::pow(t, i) * ref[static_cast<std::size_t>(i)]).margin(1e-14));
          if (i >= 2 && std::abs(r) == 1.0)
            REQUIRE(std::abs(v[static_cast<std::size_t>(i)]) < 1e-14);
        }
      }
    }
    // the collapsed vertex t = 0: all functions of degree >= 2 and their derivatives vanish
    scaled_integrated_legendre(n, 0.0, 0.0, v, dx, dt);
    for (int i = 2; i <= n; ++i) {
      REQUIRE(v[static_cast<std::size_t>(i)] == 0.0);
      REQUIRE(dx[static_cast<std::size_t>(i)] == 0.0);
      REQUIRE(dt[static_cast<std::size_t>(i)] == 0.0);
    }
  }

  SECTION("partial derivatives by finite differences") {
    const Real h = 1e-6;
    std::vector<Real> a(n + 1);
    std::vector<Real> b(n + 1);
    std::vector<Real> dummy(n + 1);
    for (const auto [x, t] :
         std::vector<std::pair<Real, Real>>{{0.2, 0.9}, {-0.4, 0.5}, {0.0, 0.0}, {0.3, 0.3}}) {
      scaled_integrated_legendre(n, x, t, v, dx, dt);
      scaled_integrated_legendre(n, x + h, t, a, dummy, dummy);
      scaled_integrated_legendre(n, x - h, t, b, dummy, dummy);
      for (int i = 1; i <= n; ++i) {
        REQUIRE(dx[static_cast<std::size_t>(i)] ==
                Approx((a[static_cast<std::size_t>(i)] - b[static_cast<std::size_t>(i)]) / (2 * h))
                    .margin(1e-7));
      }
      scaled_integrated_legendre(n, x, t + h, a, dummy, dummy);
      scaled_integrated_legendre(n, x, t - h, b, dummy, dummy);
      for (int i = 1; i <= n; ++i) {
        REQUIRE(dt[static_cast<std::size_t>(i)] ==
                Approx((a[static_cast<std::size_t>(i)] - b[static_cast<std::size_t>(i)]) / (2 * h))
                    .margin(1e-7));
      }
    }
  }
}

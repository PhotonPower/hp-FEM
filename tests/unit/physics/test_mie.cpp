#include <cmath>
#include <numbers>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/core/error.hpp"
#include "hpfem/core/special_functions.hpp"
#include "hpfem/physics/mie.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::Real;
using hpfem::physics::mie_cylinder_coefficients;
using hpfem::physics::mie_cylinder_scattering_width;

TEST_CASE("Mie cylinder: coefficients vanish without contrast, series converges, limits",
          "[physics][mie]") {
  // no contrast: nothing is scattered
  for (const Complex& c : mie_cylinder_coefficients(5.0, 0.3, 1.0, 8)) REQUIRE(std::abs(c) < 1e-14);
  REQUIRE(mie_cylinder_scattering_width(5.0, 0.3, 1.0) < 1e-13);
  // the truncated series converges: the default cut-off agrees with a much longer one
  const Real k = 6.0;
  const Real r = 0.25;
  const Real n = 1.5;
  const Real width = mie_cylinder_scattering_width(k, r, n);
  REQUIRE(width == Approx(mie_cylinder_scattering_width(k, r, n, 40)).epsilon(1e-12));
  REQUIRE(width > 0);
  // the coefficients decay fast beyond the size parameter
  const auto c = mie_cylinder_coefficients(k, r, n, 12);
  REQUIRE(std::abs(c[12]) < 1e-10 * std::abs(c[0]));
  // the derivative identity used inside: J_n' = J_{n-1} - n/x J_n by finite differences
  const Real x = 2.3;
  const Real h = 1e-6;
  for (int order = 0; order <= 3; ++order) {
    const Real fd = (hpfem::bessel_j(order, x + h) - hpfem::bessel_j(order, x - h)) / (2 * h);
    REQUIRE(hpfem::bessel_j_derivative(order, x) == Approx(fd).margin(1e-8));
  }
  // Rayleigh limit (kR -> 0, H_z polarisation): sigma ~ (pi^2 / 4) k^3 R^4 ((eps-1)/(eps+1))^2
  // ... up to the n = 0 term; check only the scaling with k^3 between two small sizes
  const Real small = mie_cylinder_scattering_width(0.02, r, n);
  const Real smaller = mie_cylinder_scattering_width(0.01, r, n);
  REQUIRE(small / smaller == Approx(8.0).epsilon(0.05));
  REQUIRE_THROWS_AS(mie_cylinder_coefficients(0.0, r, n, 3), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(mie_cylinder_coefficients(k, r, n, -2), hpfem::InvalidArgument);
}

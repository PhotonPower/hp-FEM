// Mie series of the sphere (Bohren & Huffman ch. 4) without any FEM: optical theorem,
// Rayleigh limit, convergence in the order, lossless limit, and the boundary conditions of the
// field expansions (tangential E and normal ε E continuous across the surface), which checks
// a_n, b_n, c_n, d_n and the vector harmonics together.
#include <cmath>
#include <numbers>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/core/error.hpp"
#include "hpfem/core/special_functions.hpp"
#include "hpfem/physics/mie.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::Point;
using hpfem::Real;
using hpfem::physics::mie_sphere;
using hpfem::physics::MieSphere;

TEST_CASE("Spherical Bessel functions: complex downward recurrence matches the real functions",
          "[core][special]") {
  for (const Real x : {0.3, 2.0, 7.5}) {
    const auto j = hpfem::spherical_bessel_j(12, Complex{x, 0.0});
    for (int n = 0; n <= 12; ++n) {
      REQUIRE(std::real(j[static_cast<std::size_t>(n)]) ==
              Approx(hpfem::spherical_bessel_j(n, x)).margin(1e-14));
      REQUIRE(std::abs(std::imag(j[static_cast<std::size_t>(n)])) < 1e-14);
    }
  }
  // complex argument: j_0 = sin z / z and j_1 = sin z / z^2 - cos z / z
  const Complex z{1.3, 2.1};
  const auto j = hpfem::spherical_bessel_j(5, z);
  REQUIRE(std::abs(j[0] - std::sin(z) / z) < 1e-14 * std::abs(j[0]));
  REQUIRE(std::abs(j[1] - (std::sin(z) / (z * z) - std::cos(z) / z)) < 1e-13 * std::abs(j[1]));
  REQUIRE_THROWS_AS(hpfem::spherical_bessel_j(-1, z), hpfem::InvalidArgument);
  // known values (closed forms j_n = combinations of sin, cos; Abramowitz & Stegun 10.1.11):
  // j_0(1) = sin 1, j_1(1) = sin 1 - cos 1, j_2(1) = (3 - 1) sin 1 - 3 cos 1,
  // y_0(1) = -cos 1, y_1(1) = -cos 1 - sin 1
  REQUIRE(hpfem::spherical_bessel_j(0, 1.0) == Approx(0.8414709848078965).epsilon(1e-14));
  REQUIRE(hpfem::spherical_bessel_j(1, 1.0) == Approx(0.3011686789397568).epsilon(1e-13));
  REQUIRE(hpfem::spherical_bessel_j(2, 1.0) == Approx(0.0620350520113738).epsilon(1e-12));
  REQUIRE(hpfem::spherical_bessel_y(0, 1.0) == Approx(-0.5403023058681398).epsilon(1e-14));
  REQUIRE(hpfem::spherical_bessel_y(1, 1.0) == Approx(-1.3817732906760363).epsilon(1e-13));
  REQUIRE(std::abs(hpfem::spherical_hankel1(1, 1.0) -
                   Complex{0.3011686789397568, -1.3817732906760363}) < 1e-13);
  // the complex recurrence at a complex argument against the closed form of j_2
  const Complex w{0.7, -0.4};
  const auto jw = hpfem::spherical_bessel_j(2, w);
  const Complex j2 = (3.0 / (w * w * w) - 1.0 / w) * std::sin(w) - 3.0 / (w * w) * std::cos(w);
  REQUIRE(std::abs(jw[2] - j2) < 1e-13 * std::abs(j2));
  // high order at small argument: the downward recurrence stays accurate where the upward
  // one would lose all digits (j_20(0.5) ~ 1e-33)
  const auto small = hpfem::spherical_bessel_j(20, Complex{0.5, 0.0});
  REQUIRE(std::real(small[20]) == Approx(hpfem::spherical_bessel_j(20, 0.5)).epsilon(1e-10));
}

TEST_CASE("Mie sphere: optical theorem, Rayleigh limit, convergence, lossless limit",
          "[physics][mie]") {
  const Real k = 2 * std::numbers::pi / 500e-9;
  const Real a = 100e-9;  // x = 1.257
  // lossless: Q_abs = 0 (a_n, b_n on the circle Re a = |a|^2)
  const MieSphere lossless = mie_sphere(k, a, Complex{4.0, 0.0});
  REQUIRE(lossless.scattering_efficiency() > 0.1);
  REQUIRE(std::abs(lossless.absorption_efficiency()) < 1e-12 * lossless.scattering_efficiency());
  for (std::size_t n = 0; n < lossless.a.size(); ++n) {
    REQUIRE(std::real(lossless.a[n]) == Approx(std::norm(lossless.a[n])).margin(1e-14));
    REQUIRE(std::real(lossless.b[n]) == Approx(std::norm(lossless.b[n])).margin(1e-14));
  }
  // absorbing: 0 < Q_abs < Q_ext and the series is converged at the default order
  const MieSphere lossy = mie_sphere(k, a, Complex{-10.0, 1.0});
  REQUIRE(lossy.absorption_efficiency() > 0);
  REQUIRE(lossy.absorption_efficiency() < lossy.extinction_efficiency());
  const MieSphere longer = mie_sphere(k, a, Complex{-10.0, 1.0}, 1.0, 60);
  REQUIRE(lossy.extinction_efficiency() == Approx(longer.extinction_efficiency()).epsilon(1e-12));
  REQUIRE(lossy.scattering_efficiency() == Approx(longer.scattering_efficiency()).epsilon(1e-12));
  // coefficients decay fast beyond the size parameter
  REQUIRE(std::abs(lossy.a[10]) < 1e-10 * std::abs(lossy.a[0]));
  // background index: the same size parameter and relative index give the same efficiencies
  const MieSphere in_glass = mie_sphere(k, a, Complex{1.5 * 1.5 * 4.0, 0.0}, 1.5);
  REQUIRE(in_glass.scattering_efficiency() ==
          Approx(lossless.scattering_efficiency()).epsilon(1e-12));
  // Rayleigh limit: Q_sca -> (8/3) x^4 |(eps-1)/(eps+2)|^2, Q_abs -> 4 x Im((eps-1)/(eps+2))
  const Real small_x = 0.01;
  const Complex eps{2.5, 0.3};
  const MieSphere rayleigh = mie_sphere(small_x / a, a, eps);
  const Complex alpha = (eps - 1.0) / (eps + 2.0);
  REQUIRE(rayleigh.scattering_efficiency() ==
          Approx(8.0 / 3.0 * std::pow(small_x, 4) * std::norm(alpha)).epsilon(1e-3));
  REQUIRE(rayleigh.absorption_efficiency() ==
          Approx(4.0 * small_x * std::imag(alpha)).epsilon(1e-3));
  // cross-sections
  REQUIRE(lossy.scattering_cross_section() ==
          Approx(lossy.scattering_efficiency() * std::numbers::pi * a * a));
  // no contrast: nothing scattered
  const MieSphere none = mie_sphere(k, a, Complex{1.0, 0.0});
  REQUIRE(none.scattering_efficiency() < 1e-20);
  REQUIRE_THROWS_AS(mie_sphere(0.0, a, eps), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(mie_sphere(k, a, Complex{2.0, -0.1}), hpfem::InvalidArgument);
}

TEST_CASE("Mie sphere: fields satisfy the interface conditions and the incident wave",
          "[physics][mie]") {
  const Real k = 2 * std::numbers::pi / 500e-9;
  const Real a = 150e-9;
  for (const Complex eps : {Complex{4.0, 0.0}, Complex{-10.0, 1.0}, Complex{2.25, 0.5}}) {
    const MieSphere s = mie_sphere(k, a, eps, 1.0, 40);
    // incident wave
    const Point<3> p(30e-9, -70e-9, 120e-9);
    const auto inc = s.incident_field(p);
    REQUIRE(std::abs(inc(0) - std::exp(Complex{0.0, k * p(2)})) < 1e-15);
    REQUIRE(std::abs(inc(1)) == 0.0);
    // interface conditions on a set of directions: tangential E continuous, normal eps E continuous
    for (const auto& dir :
         {Point<3>(0.3, 0.5, 0.81), Point<3>(-0.7, 0.1, -0.4), Point<3>(0.0, 0.0, 1.0),
          Point<3>(1.0, 0.0, 0.0), Point<3>(0.2, -0.9, 0.05)}) {
      const Point<3> n = dir.normalized();
      const Point<3> outside = (1.0 + 1e-9) * a * n;
      const Point<3> inside = (1.0 - 1e-9) * a * n;
      const auto e_out = s.incident_field(outside) + s.scattered_field(outside);
      const auto e_in = s.internal_field(inside);
      const Complex normal_out = n.cast<Complex>().dot(e_out);
      const Complex normal_in = n.cast<Complex>().dot(e_in);
      const auto tangential_out = e_out - normal_out * n.cast<Complex>();
      const auto tangential_in = e_in - normal_in * n.cast<Complex>();
      const Real scale = e_out.norm() + 1e-300;
      REQUIRE((tangential_out - tangential_in).norm() < 1e-7 * scale);
      REQUIRE(std::abs(normal_out - eps * normal_in) < 1e-7 * scale);
    }
    // total field picks the right expansion
    REQUIRE((s.total_field(0.5 * a * Point<3>(0.0, 1.0, 0.0)) -
             s.internal_field(0.5 * a * Point<3>(0.0, 1.0, 0.0)))
                .norm() == 0.0);
    // far field: |E_sca| ~ 1/r
    const Point<3> far(0.0, 0.0, 200 * a);
    const Point<3> farther(0.0, 0.0, 400 * a);
    REQUIRE(s.scattered_field(far).norm() / s.scattered_field(farther).norm() ==
            Approx(2.0).epsilon(2e-2));
  }
}

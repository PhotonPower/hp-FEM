// Oblique incidence on a body of revolution: the plane-wave orders reduce to the axial case
// at theta = 0, the sum over orders reproduces the Mie cross-section of a sphere at oblique
// incidence (a sphere scatters the same at every angle), the +-m powers are equal, the
// orders converge with |m| and the superposed far field carries the total power.
#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "axisymmetric_sphere.hpp"
#include "hpfem/core/error.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::Point;
using hpfem::Real;
using hpfem::physics::axial_plane_wave;
using hpfem::physics::oblique_plane_wave;
using hpfem::physics::PlanePolarisation;
using hpfem::physics::scatter_orders;

TEST_CASE("oblique plane wave orders: axial limit and Jacobi-Anger sum",
          "[physics][axisymmetric]") {
  const Real k = 2.0;
  const Point<2> x(0.6, 0.3);
  // theta = 0, p polarisation = x-polarised: only m = +-1, equal to the axial wave
  for (const int m : {-1, 1}) {
    const auto a = axial_plane_wave(Complex{1.0, 0.0}, k, m)(x);
    const auto o = oblique_plane_wave(Complex{1.0, 0.0}, k, 0.0, PlanePolarisation::kP, m)(x);
    REQUIRE((a - o).norm() < 1e-14);
  }
  REQUIRE(oblique_plane_wave(Complex{1.0, 0.0}, k, 0.0, PlanePolarisation::kP, 0)(x).norm() <
          1e-14);
  REQUIRE(oblique_plane_wave(Complex{1.0, 0.0}, k, 0.0, PlanePolarisation::kP, 2)(x).norm() <
          1e-14);
  // summing the orders over m at a point restores the plane wave E0 p e^{ik.x} (phi = 0.7)
  const Real theta = 0.9;
  const Real phi = 0.7;
  const Real rho = x(0);
  const Real z = x(1);
  const Complex plane =
      std::exp(hpfem::kI * k * (std::sin(theta) * rho * std::cos(phi) + std::cos(theta) * z));
  for (const PlanePolarisation pol : {PlanePolarisation::kS, PlanePolarisation::kP}) {
    Complex e_r = 0, e_phi = 0, e_z = 0;
    for (int m = -25; m <= 25; ++m) {
      const auto c = oblique_plane_wave(Complex{1.0, 0.0}, k, theta, pol, m)(x);
      const Complex rotation = std::exp(hpfem::kI * static_cast<Real>(m) * phi);
      e_r += rotation * c(0);
      e_phi += rotation * (hpfem::kI * c(1) / rho);  // E_phi = i v / rho
      e_z += rotation * c(2);
    }
    const Real p_x = pol == PlanePolarisation::kP ? std::cos(theta) : 0.0;
    const Real p_y = pol == PlanePolarisation::kS ? 1.0 : 0.0;
    const Real p_z = pol == PlanePolarisation::kP ? -std::sin(theta) : 0.0;
    REQUIRE(std::abs(e_r - plane * (p_x * std::cos(phi) + p_y * std::sin(phi))) < 1e-12);
    REQUIRE(std::abs(e_phi - plane * (-p_x * std::sin(phi) + p_y * std::cos(phi))) < 1e-12);
    REQUIRE(std::abs(e_z - plane * p_z) < 1e-12);
  }
}

TEST_CASE("sphere at oblique incidence: Mie cross-section from the sum over orders",
          "[physics][axisymmetric]") {
  using namespace hpfem::physics::test;
  const Real n = 2.0;
  const Real x = 1.5;
  const Real theta_i = 50.0 * std::numbers::pi / 180.0;
  const auto base = sphere_scattering(n, 4, 2, x, 1);
  const auto& setup = base.scattering->setup();
  const hpfem::physics::Surface<2> interface =
      hpfem::physics::Surface<2>::around_cells(*base.mesh, base.sphere_tag);
  const Real mie = mie_scattering_efficiency(x, n) * hpfem::constants::pi;
  const Real intensity = 1.0 / (2.0 * hpfem::constants::Z0);
  for (const PlanePolarisation pol : {PlanePolarisation::kS, PlanePolarisation::kP}) {
    const auto result = scatter_orders(
        *base.meridian, *base.azimuthal, setup,
        [&](int m) { return oblique_plane_wave(Complex{1.0, 0.0}, x, theta_i, pol, m); }, 8,
        interface, 1e-5);
    REQUIRE(result.orders.size() >= 7);  // 0, +-1, +-2, +-3 at least
    REQUIRE(result.orders.size() == result.fields.size());
    // +-m carry the same power; higher orders fade
    for (std::size_t i = 1; i + 1 < result.orders.size(); i += 2) {
      REQUIRE(result.power[i] == Approx(result.power[i + 1]).epsilon(1e-6));
    }
    REQUIRE(result.power.back() < 1e-2 * result.total_power());
    const Real sigma = result.total_power() / intensity;
    REQUIRE(sigma == Approx(mie).epsilon(5e-2));
    // the superposed far field at any azimuth radiates the total power
    std::vector<Real> theta(91);
    for (std::size_t i = 0; i < theta.size(); ++i)
      theta[i] = std::numbers::pi * static_cast<Real>(i) / 90.0;
    std::vector<hpfem::physics::AxisymmetricFarField> patterns;
    for (std::size_t i = 0; i < result.orders.size(); ++i) {
      patterns.push_back(hpfem::physics::axisymmetric_far_field(
          *base.meridian, *base.azimuthal, result.fields[i].meridian, result.fields[i].azimuthal,
          result.orders[i], setup.omega, setup.materials, interface, theta));
    }
    // integrate |F(theta, phi)|^2 over the sphere with 36 azimuths
    Real power = 0;
    for (int j = 0; j < 36; ++j) {
      const Real phi = 2 * std::numbers::pi * j / 36.0;
      const auto total = hpfem::physics::superpose_far_field(patterns, result.orders, phi);
      Real integral = 0;
      for (std::size_t t = 1; t < theta.size(); ++t) {
        const auto density = [&](std::size_t s) {
          return (std::norm(total.f_theta[s]) + std::norm(total.f_phi[s])) * std::sin(theta[s]);
        };
        integral += 0.5 * (density(t - 1) + density(t)) * (theta[t] - theta[t - 1]);
      }
      power += integral * (2 * std::numbers::pi / 36.0) / (2 * total.impedance);
    }
    REQUIRE(power == Approx(result.total_power()).epsilon(1e-2));
  }
  REQUIRE_THROWS_AS(scatter_orders(
                        *base.meridian, *base.azimuthal, setup,
                        [&](int m) {
                          return oblique_plane_wave(Complex{1.0, 0.0}, x, theta_i,
                                                    PlanePolarisation::kS, m);
                        },
                        -1, interface),
                    hpfem::InvalidArgument);
}

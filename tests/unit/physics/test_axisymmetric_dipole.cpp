// Dipole sources on the axis: the Gaussian dipole components per order, the vacuum power
// near the Larmor value, the transverse orders m = +-1 equal, the Purcell enhancement of
// the axial dipole at the centre of a dielectric sphere peaks at the TM_1 Mie pole found
// by the resonance solver, and the total-field setup is validated.
#include <cmath>
#include <complex>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "axisymmetric_dipole.hpp"
#include "hpfem/core/error.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::Point;
using hpfem::Real;
using hpfem::physics::AxisDipole;
using hpfem::physics::axisymmetric_gaussian_dipole;
using hpfem::physics::AxisymmetricScattering;
using hpfem::physics::AxisymmetricScatteringSetup;

TEST_CASE("Gaussian dipole components and the Larmor power", "[physics][axisymmetric]") {
  const Real omega = 2.0 * hpfem::constants::c0;
  const Real sigma = 0.1;
  const auto axial =
      axisymmetric_gaussian_dipole(0.3, Complex{1.0, 0.0}, AxisDipole::kAxial, sigma, omega, 0);
  const auto plus = axisymmetric_gaussian_dipole(0.3, Complex{1.0, 0.0}, AxisDipole::kTransverse,
                                                 sigma, omega, 1);
  const auto minus = axisymmetric_gaussian_dipole(0.3, Complex{1.0, 0.0}, AxisDipole::kTransverse,
                                                  sigma, omega, -1);
  const Point<2> x(0.05, 0.35);
  const Real g = std::exp(-(0.05 * 0.05 + 0.05 * 0.05) / (2 * sigma * sigma)) /
                 (std::pow(2 * hpfem::constants::pi, 1.5) * sigma * sigma * sigma);
  const Complex factor = hpfem::kI * omega * hpfem::constants::mu0;
  REQUIRE(std::abs(axial(x)(2) - factor * g) < 1e-12 * std::abs(factor * g));
  REQUIRE(std::abs(axial(x)(0)) == 0.0);
  REQUIRE(std::abs(plus(x)(0) - 0.5 * factor * g) < 1e-12 * std::abs(factor * g));
  REQUIRE(std::abs(plus(x)(1) - 0.5 * 0.05 * factor * g) < 1e-12 * std::abs(factor * g));
  REQUIRE(std::abs(minus(x)(1) + 0.5 * 0.05 * factor * g) < 1e-12 * std::abs(factor * g));
  REQUIRE_THROWS_AS(
      axisymmetric_gaussian_dipole(0.0, Complex{1.0, 0.0}, AxisDipole::kAxial, sigma, omega, 1),
      hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(axisymmetric_gaussian_dipole(0.0, Complex{1.0, 0.0}, AxisDipole::kTransverse,
                                                 sigma, omega, 0),
                    hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(
      axisymmetric_gaussian_dipole(0.0, Complex{1.0, 0.0}, AxisDipole::kAxial, 0.0, omega, 0),
      hpfem::InvalidArgument);
  // Larmor: Z0 k0^2 |p|^2 / (12 pi)
  const Real k0 = 2.0;
  REQUIRE(hpfem::physics::dipole_vacuum_power(Complex{3.0, 0.0}, omega) ==
          Approx(hpfem::constants::Z0 * k0 * k0 * 9.0 / (12 * hpfem::constants::pi)));
}

TEST_CASE("dipole in vacuum and at the centre of a sphere", "[physics][axisymmetric]") {
  using namespace hpfem::physics::test;
  const Real x = 1.5;
  const Real sigma = 0.08;
  const Real reference = smeared_vacuum_power(x, sigma);
  const auto axial = dipole_problem(1.0, 4, 2, x, AxisDipole::kAxial, 0, sigma);
  const Real p_axial = dipole_power(axial, axial.scattering->solve());
  REQUIRE(p_axial == Approx(reference).epsilon(3e-2));
  const auto plus = dipole_problem(1.0, 4, 2, x, AxisDipole::kTransverse, 1, sigma);
  const auto minus = dipole_problem(1.0, 4, 2, x, AxisDipole::kTransverse, -1, sigma);
  const Real p_plus = dipole_power(plus, plus.scattering->solve());
  const Real p_minus = dipole_power(minus, minus.scattering->solve());
  REQUIRE(p_plus == Approx(p_minus).epsilon(1e-8));
  REQUIRE(2 * p_plus == Approx(reference).epsilon(3e-2));
  // Purcell factor of the axial dipole at the centre of the sphere n = 3: the electric
  // dipole couples to the TM_1 modes, so the enhancement peaks at Re x_TM of the Mie pole
  // (stage 2) and is small off resonance
  const Real n = 3.0;
  const Complex x_tm = mie_pole(n, Polarisation::kTM);
  const auto purcell = [&](Real size) {
    const auto problem = dipole_problem(n, 4, 2, size, AxisDipole::kAxial, 0, sigma);
    return dipole_power(problem, problem.scattering->solve()) / smeared_vacuum_power(size, sigma);
  };
  const Real on = purcell(x_tm.real());
  const Real below = purcell(x_tm.real() - 0.3);
  const Real above = purcell(x_tm.real() + 0.3);
  REQUIRE(on > 2.0 * below);
  REQUIRE(on > 2.0 * above);
  REQUIRE(on > 1.0);
  // validation: neither or both sources
  AxisymmetricScatteringSetup bad = axial.scattering->setup();
  bad.incident = hpfem::physics::axial_plane_wave(Complex{1.0, 0.0}, x, 1);
  REQUIRE_THROWS_AS(AxisymmetricScattering(*axial.meridian, *axial.azimuthal, bad),
                    hpfem::InvalidArgument);
  bad.current = {};
  bad.incident = {};
  REQUIRE_THROWS_AS(AxisymmetricScattering(*axial.meridian, *axial.azimuthal, bad),
                    hpfem::InvalidArgument);
}

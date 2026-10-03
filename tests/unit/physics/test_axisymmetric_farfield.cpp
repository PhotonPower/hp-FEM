// Far field of axisymmetric fields: the axial dipole radiates the Larmor pattern sin(theta)
// with no azimuthal component, the far-field power equals the near-field Poynting flux for
// the axial and the transverse dipole, and for the sphere scattering the far-field power
// agrees with the flux (and the Mie cross-section); input validation.
#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "axisymmetric_dipole.hpp"
#include "hpfem/core/error.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::Real;
using hpfem::physics::AxisDipole;
using hpfem::physics::axisymmetric_far_field;

namespace {

std::vector<Real> angles(int count) {
  std::vector<Real> theta(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    theta[static_cast<std::size_t>(i)] = std::numbers::pi * i / (count - 1);
  }
  return theta;
}

}  // namespace

TEST_CASE("far field of the dipoles: Larmor pattern and power balance", "[physics][axisymmetric]") {
  using namespace hpfem::physics::test;
  const Real x = 1.5;
  const Real sigma = 0.08;
  const auto theta = angles(181);
  // axial dipole, m = 0: F_phi = 0, |F_theta| ~ sin(theta)
  const auto axial = dipole_problem(1.0, 4, 3, x, AxisDipole::kAxial, 0, sigma);
  const auto field = axial.scattering->solve();
  const auto& setup = axial.scattering->setup();
  const hpfem::physics::Surface<2> interface =
      hpfem::physics::Surface<2>::around_cells(*axial.mesh, axial.sphere_tag);
  const auto far =
      axisymmetric_far_field(*axial.meridian, *axial.azimuthal, field.meridian, field.azimuthal, 0,
                             setup.omega, setup.materials, interface, theta);
  REQUIRE(far.theta.size() == theta.size());
  REQUIRE(far.wavenumber == Approx(x));
  REQUIRE(far.impedance == Approx(hpfem::constants::Z0));
  Real max_phi = 0;
  Real max_theta = 0;
  for (std::size_t i = 0; i < theta.size(); ++i) {
    max_phi = std::max(max_phi, std::abs(far.f_phi[i]));
    max_theta = std::max(max_theta, std::abs(far.f_theta[i]));
  }
  REQUIRE(max_phi < 1e-6 * max_theta);
  const Real at_90 = std::abs(far.f_theta[90]);
  for (const std::size_t i : {30u, 45u, 60u, 120u, 150u}) {
    REQUIRE(std::abs(far.f_theta[i]) / at_90 == Approx(std::sin(theta[i])).epsilon(2e-3));
  }
  REQUIRE(std::abs(far.f_theta[0]) < 1e-3 * at_90);
  const Real flux = dipole_power(axial, field);
  REQUIRE(far.radiated_power() == Approx(flux).epsilon(2e-3));
  REQUIRE(far.radiated_power() == Approx(smeared_vacuum_power(x, sigma)).epsilon(5e-3));
  // transverse dipole, m = 1: power of both orders equals the flux
  const auto transverse = dipole_problem(1.0, 4, 3, x, AxisDipole::kTransverse, 1, sigma);
  const auto field_t = transverse.scattering->solve();
  const auto far_t = axisymmetric_far_field(
      *transverse.meridian, *transverse.azimuthal, field_t.meridian, field_t.azimuthal, 1,
      setup.omega, setup.materials,
      hpfem::physics::Surface<2>::around_cells(*transverse.mesh, transverse.sphere_tag), theta);
  REQUIRE(far_t.radiated_power() == Approx(dipole_power(transverse, field_t)).epsilon(2e-3));
  // the transverse dipole radiates along the axis: F(0) != 0
  REQUIRE(std::abs(far_t.f_theta[0]) + std::abs(far_t.f_phi[0]) > 0.5 * at_90);
  // validation
  REQUIRE_THROWS_AS(
      axisymmetric_far_field(*axial.meridian, *axial.azimuthal, hpfem::Vector::Zero(2),
                             field.azimuthal, 0, setup.omega, setup.materials, interface, theta),
      hpfem::InvalidArgument);
}

TEST_CASE("far field of the sphere scattering agrees with the flux and Mie",
          "[physics][axisymmetric]") {
  using namespace hpfem::physics::test;
  const Real n = 2.0;
  const Real x = 1.5;
  const auto problem = sphere_scattering(n, 4, 3, x, 1);
  const auto field = problem.scattering->solve();
  const auto& setup = problem.scattering->setup();
  const auto far = axisymmetric_far_field(
      *problem.meridian, *problem.azimuthal, field.meridian, field.azimuthal, 1, setup.omega,
      setup.materials, hpfem::physics::Surface<2>::around_cells(*problem.mesh, problem.sphere_tag),
      angles(361));
  const Real flux = sphere_scattered_power(problem, field);
  REQUIRE(far.radiated_power() == Approx(flux).epsilon(3e-3));
  const Real sigma = sphere_cross_section(far.radiated_power());
  REQUIRE(sigma == Approx(mie_scattering_efficiency(x, n) * hpfem::constants::pi).epsilon(5e-3));
}

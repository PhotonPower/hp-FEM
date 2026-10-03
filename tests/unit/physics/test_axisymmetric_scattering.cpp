// Axisymmetric scattering: the axial plane wave has the orders m = +-1 only, the scattered
// power of a lossless sphere is the same through the sphere interface and any enclosing
// surface (no sources in between), equal for m = +1 and m = -1, and the cross-section lies
// near the Mie value already at p = 2; the setup is validated.
#include <cmath>
#include <complex>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "axisymmetric_sphere.hpp"
#include "hpfem/core/error.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::physics::axial_plane_wave;
using hpfem::physics::AxisymmetricScattering;
using hpfem::physics::AxisymmetricScatteringSetup;

TEST_CASE("axial plane wave components and Mie series sanity", "[physics][axisymmetric]") {
  const Real k = 2.0;
  const auto plus = axial_plane_wave(Complex{2.0, 0.0}, k, 1);
  const auto minus = axial_plane_wave(Complex{2.0, 0.0}, k, -1);
  const Point<2> x(0.7, 0.3);
  const Complex phase = std::exp(Complex{0.0, 1.0} * k * x(1));
  REQUIRE(std::abs(plus(x)(0) - phase) < 1e-14);
  REQUIRE(std::abs(plus(x)(1) - 0.7 * phase) < 1e-14);
  REQUIRE(std::abs(plus(x)(2)) == 0.0);
  REQUIRE(std::abs(minus(x)(1) + 0.7 * phase) < 1e-14);
  REQUIRE_THROWS_AS(axial_plane_wave(Complex{1.0, 0.0}, k, 0), hpfem::InvalidArgument);
  // Rayleigh limit: Q_sca -> (8/3) x^4 ((n^2 - 1) / (n^2 + 2))^2
  using hpfem::physics::test::mie_scattering_efficiency;
  const Real n = 2.0;
  const Real small = 0.05;
  const Real rayleigh = 8.0 / 3.0 * std::pow(small, 4) * std::pow((n * n - 1) / (n * n + 2), 2);
  REQUIRE(mie_scattering_efficiency(small, n) == Approx(rayleigh).epsilon(1e-2));
  REQUIRE(mie_scattering_efficiency(1.5, n) > 0.5);
}

TEST_CASE("sphere scattering: power balance, symmetry in m and the Mie cross-section",
          "[physics][axisymmetric]") {
  using namespace hpfem::physics::test;
  const Real n = 2.0;
  const Real x = 1.5;
  const auto plus = sphere_scattering(n, 4, 2, x, 1);
  const auto field = plus.scattering->solve();
  REQUIRE(field.azimuthal_order == 1);
  REQUIRE(field.meridian.size() == plus.meridian->num_dofs());
  const Real power = sphere_scattered_power(plus, field);
  REQUIRE(power > 0);
  // the closed surface around the shell between the sphere and r, |z| < 2 carries no power
  hpfem::mesh::Mesh<2>& mesh = *plus.mesh;
  constexpr hpfem::mesh::Tag kShell = 3;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const Point<2> centre = hpfem::mesh::affine_map(mesh, c).centroid();
    if (mesh.cell_tag(c) != plus.sphere_tag && centre(0) < 2.0 && std::abs(centre(1)) < 2.0) {
      mesh.set_cell_tag(c, kShell);
    }
  }
  const hpfem::physics::Surface<2> shell = hpfem::physics::Surface<2>::around_cells(mesh, kShell);
  const Real balance = hpfem::physics::axisymmetric_poynting_flux(
      *plus.meridian, *plus.azimuthal, field.meridian, field.azimuthal, 1,
      plus.scattering->setup().omega, plus.scattering->setup().materials, shell);
  // the discrete Poynting flux is conserved only up to the discretisation error (p = 2)
  REQUIRE(std::abs(balance) < 5e-2 * power);
  // m = -1 gives the same power
  const auto minus = sphere_scattering(n, 4, 2, x, -1);
  const Real power_minus = sphere_scattered_power(minus, minus.scattering->solve());
  REQUIRE(power_minus == Approx(power).epsilon(1e-6));
  // cross-section against the Mie series (p = 2: a few percent)
  const Real sigma = sphere_cross_section(power);
  const Real mie = mie_scattering_efficiency(x, n) * hpfem::constants::pi;
  REQUIRE(sigma == Approx(mie).epsilon(5e-2));
  // validation
  AxisymmetricScatteringSetup bad = plus.scattering->setup();
  bad.omega = 0;
  REQUIRE_THROWS_AS(AxisymmetricScattering(*plus.meridian, *plus.azimuthal, bad),
                    hpfem::InvalidArgument);
  bad = plus.scattering->setup();
  bad.incident = {};
  REQUIRE_THROWS_AS(AxisymmetricScattering(*plus.meridian, *plus.azimuthal, bad),
                    hpfem::InvalidArgument);
}

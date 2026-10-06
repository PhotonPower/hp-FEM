// Conical incidence on a flat interface with the layered background: the stack wave solves
// the problem, so the scattered field vanishes for s and p polarisation at beta != 0; the
// separated incident wave reproduces the stack reflectance through the reflected zeroth
// order and the transmitted order carries the transmittance.
#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/periodic.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/point_location.hpp"
#include "hpfem/physics/conical_scattering.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::Index;
using hpfem::kI;
using hpfem::Point;
using hpfem::Real;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::materials::Material;
using hpfem::mesh::Mesh;
using hpfem::physics::ConicalScattering;
using hpfem::physics::ConicalScatteringSetup;
using hpfem::physics::ConicalVector;
using hpfem::physics::LayerStack;
using hpfem::physics::Polarisation;
using hpfem::pml::PmlBox;
using hpfem::pml::PmlProfile;
namespace box_tag = hpfem::mesh::box_tag;

TEST_CASE("conical flat interface: the stack wave is the solution, orders give R and T",
          "[physics][conical][layered]") {
  const Real k0 = 2 * std::numbers::pi;  // wavelength 1
  const Real n2 = 1.6;
  const Real period = 0.7;
  const Real angle = 35.0 * std::numbers::pi / 180.0;
  const Real azimuth = 50.0 * std::numbers::pi / 180.0;
  const LayerStack<2> stack(Material::dielectric(1.0), {}, Material::dielectric(n2), 0.0);
  // unit cell x in [0, period], y in [-2, 2] with PML beyond |y| > 1 (thickness 1)
  Mesh<2> mesh = hpfem::mesh::rectangle(4, 24, Point<2>(0.0, -2.0), Point<2>(period, 2.0));
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if (hpfem::mesh::affine_map(mesh, c).centroid()(1) < 0) mesh.set_cell_tag(c, 2);
  }
  const NedelecDofMap<2> nd(mesh, 3);
  const DofMap<2> h1(mesh, 3);
  for (const Polarisation pol : {Polarisation::kS, Polarisation::kP}) {
    const auto wave = hpfem::physics::layered_conical_wave(stack, k0, angle, azimuth, pol);
    REQUIRE(wave.beta == Approx(k0 * std::sin(angle) * std::sin(azimuth)));
    REQUIRE(wave.kx == Approx(k0 * std::sin(angle) * std::cos(azimuth)));
    REQUIRE(wave.reflectance + wave.transmittance == Approx(1.0).epsilon(1e-10));
    // the separated incident wave: a downward plane wave of amplitude 1
    const Point<2> probe(0.2, 0.9);
    const ConicalVector inc = wave.incident(probe);
    const ConicalVector physical(inc(0), inc(1), kI * inc(2));
    REQUIRE(physical.norm() == Approx(1.0).epsilon(1e-8));
    const Complex k_dot_e = physical(0) * wave.kx - physical(1) * wave.ky + physical(2) * wave.beta;
    REQUIRE(std::abs(k_dot_e) < 1e-8 * k0);
    // incident + reflected = the stack field above the interface at a second point
    const ConicalVector full = wave.field(Point<2>(0.4, 0.6));
    const ConicalVector down = wave.incident(Point<2>(0.4, 0.6));
    const ConicalVector up = full - down;
    const Complex ratio = up(0) / down(0);
    (void)ratio;
    REQUIRE(up.norm() > 0.0);
    ConicalScatteringSetup setup;
    setup.omega = k0 * hpfem::constants::c0;
    setup.beta = wave.beta;
    setup.materials.set(2, Material::dielectric(n2));
    setup.background = stack;
    setup.incident = wave.field;
    setup.pml = PmlBox<2>(Point<2>(0.0, -1.0), Point<2>(period, 1.0),
                          PmlBox<2>::Thickness{0.0, 0.0, 1.0, 1.0}, k0, 1.0, PmlProfile{2, 1e-10});
    setup.pec_tags = {box_tag::kYMin, box_tag::kYMax};
    setup.periodic = {hpfem::assembly::PeriodicPair<2>{
        box_tag::kXMin, box_tag::kXMax, Point<2>(period, 0.0),
        hpfem::assembly::bloch_phase<2>(Point<2>(wave.kx, 0.0), Point<2>(period, 0.0))}};
    const ConicalScattering problem(nd, h1, setup);
    const auto solution = problem.solve();
    // no contrast anywhere: the scattered field vanishes
    const Real scale = std::sqrt(solution.transverse.size() + solution.longitudinal.size());
    INFO("polarisation " << (pol == Polarisation::kS ? "s" : "p"));
    CHECK(solution.transverse.norm() / scale < 1e-9);
    CHECK(solution.longitudinal.norm() / scale < 1e-9);
    // reflected orders of (total - incident) above, transmitted orders below
    const hpfem::mesh::PointLocator<2> locator(mesh);
    const auto reflected = hpfem::physics::conical_fourier_coefficients(
        [&](const Point<2>& x) {
          const ConicalVector total = *problem.total_field(solution, locator, x);
          const ConicalVector i = wave.incident(x);
          return ConicalVector(total - ConicalVector(i(0), i(1), kI * i(2)));
        },
        Point<2>(0.0, 0.5), Point<2>(1.0, 0.0), period, wave.kx, 1, 32);
    const auto transmitted = hpfem::physics::conical_fourier_coefficients(
        [&](const Point<2>& x) { return *problem.total_field(solution, locator, x); },
        Point<2>(0.0, -0.5), Point<2>(1.0, 0.0), period, wave.kx, 1, 32);
    const auto r = hpfem::physics::conical_diffraction_efficiencies(
        reflected, k0, 1.0, period, wave.kx, wave.beta, wave.ky, 1.0);
    const auto t = hpfem::physics::conical_diffraction_efficiencies(
        transmitted, k0, n2, period, wave.kx, wave.beta, wave.ky, 1.0);
    CHECK(r[1].efficiency == Approx(wave.reflectance).epsilon(1e-6));
    CHECK(t[1].efficiency == Approx(wave.transmittance).epsilon(1e-6));
    CHECK(r[0].efficiency + r[2].efficiency < 1e-10);
  }
}

TEST_CASE("conical conventions: s and p amplitudes and the phase of the layered wave",
          "[physics][conical][layered]") {
  // s = k x y_hat / |.| = (-sin phi, 0, cos phi), p = k_hat x s = (-cos th cos phi, -sin th, -cos
  // th sin phi), phase 1 at the origin on the top interface; k = k0 n (sin th cos phi, -cos th, sin
  // th sin phi)
  const Real k0 = 2 * std::numbers::pi;
  const LayerStack<2> stack(Material::dielectric(1.0), {}, Material::dielectric(1.5), 0.0);
  for (const auto [theta_deg, phi_deg] :
       {std::pair{35.0, 50.0}, std::pair{50.0, 0.0}, std::pair{0.0, 0.0}, std::pair{30.0, 90.0},
        std::pair{60.0, 200.0}}) {
    const Real th = theta_deg * std::numbers::pi / 180.0;
    const Real ph = phi_deg * std::numbers::pi / 180.0;
    const Point<3> k(k0 * std::sin(th) * std::cos(ph), -k0 * std::cos(th),
                     k0 * std::sin(th) * std::sin(ph));
    const Point<3> s_expected(-std::sin(ph), 0.0, std::cos(ph));
    const Point<3> p_expected(-std::cos(th) * std::cos(ph), -std::sin(th),
                              -std::cos(th) * std::sin(ph));
    for (const Polarisation pol : {Polarisation::kS, Polarisation::kP}) {
      const auto wave = hpfem::physics::layered_conical_wave(stack, k0, th, ph, pol, 2.0);
      REQUIRE(wave.kx == Approx(k(0)).margin(1e-12));
      REQUIRE(wave.beta == Approx(k(2)).margin(1e-12));
      REQUIRE(wave.ky == Approx(-k(1)).margin(1e-12));
      const Point<3>& expected = pol == Polarisation::kS ? s_expected : p_expected;
      // physical amplitude at the origin: 2 * expected, phase 1
      const ConicalVector at_origin = wave.incident(Point<2>(0.0, 0.0));
      const ConicalVector physical(at_origin(0), at_origin(1), kI * at_origin(2));
      for (int c = 0; c < 3; ++c) REQUIRE(std::abs(physical(c) - 2.0 * expected(c)) < 1e-10);
      // plane-wave phase elsewhere: e^{i(kx x + ky_normal (-1) y)} with the downward normal
      // component
      const Point<2> x(0.3, 0.7);
      const ConicalVector there = wave.incident(x);
      const Complex phase = std::exp(kI * (k(0) * x(0) + k(1) * x(1)));
      for (int c = 0; c < 3; ++c) REQUIRE(std::abs(there(c) - at_origin(c) * phase) < 1e-10);
      // conical_polarisation agrees with the stack convention for the same k and normal
      const ConicalVector helper =
          hpfem::physics::conical_polarisation(k, Point<3>(0.0, 1.0, 0.0), pol);
      const Real sign = (helper.real().dot(expected) >= 0) ? 1.0 : -1.0;
      for (int c = 0; c < 3; ++c) REQUIRE(std::abs(helper(c) - sign * expected(c)) < 1e-10);
    }
  }
}

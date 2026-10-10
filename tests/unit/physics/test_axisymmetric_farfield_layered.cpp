// Far field by reciprocity on a layered background (ADR-0014 §4, M18 S3): the homogeneous limit
// against the near-to-far transform (both half-spaces, both polarisations), and an axial dipole
// above glass: the far field in the cover and in the substrate (also beyond the critical angle)
// against the reciprocity value of the point source, the power into both half-spaces against
// the flux through a surface around the dipole, and the collection over a cone.
#include <cmath>
#include <complex>
#include <memory>
#include <numbers>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "axisymmetric_sphere.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/physics/layer_stack.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::materials::Material;
using hpfem::physics::LayerStack;
using hpfem::physics::Surface;
namespace physics = hpfem::physics;

namespace {

constexpr Real kPi = std::numbers::pi;

std::vector<Real> angles(Real from, Real to, int count) {
  std::vector<Real> out;
  for (int i = 0; i < count; ++i) out.push_back(from + (to - from) * i / (count - 1));
  return out;
}

/// Closed surface around the cells whose centroid satisfies `inside`.
template <typename Inside>
Surface<2> surface_around(const hpfem::mesh::Mesh<2>& mesh, Inside inside) {
  hpfem::mesh::Mesh<2> marked = mesh;
  constexpr hpfem::mesh::Tag kMarked = 99;
  for (Index c = 0; c < marked.num_cells(); ++c) {
    marked.set_cell_tag(c, inside(hpfem::mesh::affine_map(mesh, c).centroid()) ? kMarked : 1);
  }
  return Surface<2>::around_cells(marked, kMarked);
}

}  // namespace

TEST_CASE("layered far field: the homogeneous limit is the near-to-far transform",
          "[physics][axisymmetric][layered]") {
  using namespace hpfem::physics::test;
  const Real x = 1.5;
  for (const int m : {1, -1}) {
    const auto problem = sphere_scattering(2.0, 4, 3, x, m);
    const auto field = problem.scattering->solve();
    const auto& setup = problem.scattering->setup();
    const Surface<2> interface = Surface<2>::around_cells(*problem.mesh, problem.sphere_tag);
    const auto up = angles(0.0, 1.5, 7);
    const auto down = angles(1.65, kPi, 7);
    const LayerStack<3> vacuum(Material::vacuum(), {}, Material::vacuum(), 0.3);
    const auto layered = physics::axisymmetric_layered_far_field(
        *problem.meridian, *problem.azimuthal, field.meridian, field.azimuthal, m, setup.omega,
        setup.materials, interface, vacuum, up, down);
    std::vector<Real> all = up;
    all.insert(all.end(), down.begin(), down.end());
    const auto reference = physics::axisymmetric_far_field(
        *problem.meridian, *problem.azimuthal, field.meridian, field.azimuthal, m, setup.omega,
        setup.materials, interface, all);
    Real scale = 0;
    for (std::size_t i = 0; i < all.size(); ++i) {
      scale = std::max(scale, std::abs(reference.f_theta[i]) + std::abs(reference.f_phi[i]));
    }
    Real worst = 0;
    for (std::size_t i = 0; i < all.size(); ++i) {
      const bool is_up = i < up.size();
      const auto& half = is_up ? layered.up : layered.down;
      const std::size_t j = is_up ? i : i - up.size();
      worst = std::max(worst, std::abs(half.f_theta[j] - reference.f_theta[i]) +
                                  std::abs(half.f_phi[j] - reference.f_phi[i]));
    }
    INFO("m = " << m << ": largest deviation " << worst / scale);
    REQUIRE(worst < 1e-9 * scale);
    REQUIRE(layered.up.wavenumber == Approx(x));
    REQUIRE(layered.down.impedance == Approx(hpfem::constants::Z0));
  }
}

TEST_CASE("layered far field: axial dipole above glass, both half-spaces and the power",
          "[physics][axisymmetric][layered]") {
  using namespace hpfem;
  const Real k0 = 2 * kPi;
  const Real omega = k0 * constants::c0;
  const Real height = 0.8;
  const Real sigma = 0.15;
  const Complex moment{1e-9, 0.0};
  constexpr mesh::Tag kGlass = 3;
  mesh::Mesh<2> mesh = mesh::rectangle(30, 60, Point<2>(0.0, -3.0), Point<2>(3.0, 3.0));
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if (mesh::affine_map(mesh, c).centroid()(1) < 0) mesh.set_cell_tag(c, kGlass);
  }
  const fespace::NedelecDofMap<2> meridian(mesh, 3);
  const fespace::DofMap<2> azimuthal(mesh, 3);
  physics::AxisymmetricScatteringSetup setup;
  setup.omega = omega;
  setup.materials.set(kGlass, Material::dielectric(1.5));
  setup.axis_tag = mesh::box_tag::kXMin;
  setup.azimuthal_order = 0;
  setup.pml = pml::PmlBox<2>(Point<2>(0.0, -2.0), Point<2>(2.0, 2.0), {0.0, 1.0, 1.0, 1.0}, k0);
  setup.current = physics::axisymmetric_gaussian_dipole(height, moment, physics::AxisDipole::kAxial,
                                                        sigma, omega, 0);
  setup.extra_quadrature_order = 6;
  const physics::AxisymmetricScattering problem(meridian, azimuthal, setup);
  const auto field = problem.solve();
  // a closed surface around the Gaussian, in the air
  const Surface<2> around = surface_around(
      mesh, [](const Point<2>& c) { return c(0) < 0.8 && c(1) > 0.1 && c(1) < 1.5; });
  const LayerStack<3> stack(Material::vacuum(), {}, Material::dielectric(1.5), 0.0);
  // the substrate directions 1.0 and 1.3 rad from the normal lie beyond the critical angle
  // asin(1 / 1.5) = 0.73 (light from the near field of the dipole)
  const std::vector<Real> up = {0.0, 0.2, 0.6, 1.0, 1.3};
  const std::vector<Real> down = {kPi - 0.2, kPi - 0.6, kPi - 1.0, kPi - 1.3, kPi};
  const auto far =
      physics::axisymmetric_layered_far_field(meridian, azimuthal, field.meridian, field.azimuthal,
                                              0, omega, setup.materials, around, stack, up, down);
  // reciprocity of the point source: F . e = i omega mu0 / (4 pi) p E_pw,z(0, h), Gaussian-averaged
  const Real smearing = std::exp(-0.5 * k0 * k0 * sigma * sigma);
  Real scale = 0;
  std::vector<Complex> expected_up;
  std::vector<Complex> expected_down;
  for (const Real theta : up) {
    const auto wave =
        physics::layered_axisymmetric_wave(stack, k0, theta, physics::Polarisation::kP, 0);
    expected_up.push_back(kI * omega * constants::mu0 / (4 * kPi) * moment *
                          wave.value(Point<2>(0.0, height))(2) * smearing);
    scale = std::max(scale, std::abs(expected_up.back()));
  }
  for (const Real theta : down) {
    const Real theta_inc = kPi - theta;
    const auto wave = physics::layered_axisymmetric_wave(
        stack, k0, theta_inc, physics::Polarisation::kP, 0, physics::StackSide::kBottom);
    expected_down.push_back(kI * omega * constants::mu0 / (4 * kPi) * moment *
                            wave.value(Point<2>(0.0, height))(2) * smearing);
    scale = std::max(scale, std::abs(expected_down.back()));
  }
  for (std::size_t i = 0; i < up.size(); ++i) {
    INFO("up " << up[i] << ": " << far.up.f_theta[i] << " against " << expected_up[i]);
    REQUIRE(std::abs(far.up.f_theta[i] - expected_up[i]) < 1e-2 * scale);
    REQUIRE(std::abs(far.up.f_phi[i]) < 1e-6 * scale);
  }
  for (std::size_t i = 0; i < down.size(); ++i) {
    INFO("down " << down[i] << ": " << far.down.f_theta[i] << " against " << expected_down[i]);
    REQUIRE(std::abs(far.down.f_theta[i] - expected_down[i]) < 1e-2 * scale);
    REQUIRE(std::abs(far.down.f_phi[i]) < 1e-6 * scale);
  }
  // the radiated power splits into the two half-spaces (no guided modes at a single interface)
  const auto dense = physics::axisymmetric_layered_far_field(
      meridian, azimuthal, field.meridian, field.azimuthal, 0, omega, setup.materials, around,
      stack, angles(0.0, kPi / 2 - 1e-6, 301), angles(kPi / 2 + 1e-6, kPi, 301));
  const Real radiated = physics::axisymmetric_poynting_flux(
      meridian, azimuthal, field.meridian, field.azimuthal, 0, omega, setup.materials, around);
  const Real into_air = dense.up.radiated_power();
  const Real into_glass = dense.down.radiated_power();
  INFO("radiated " << radiated << ", into the air " << into_air << ", into the glass "
                   << into_glass);
  REQUIRE(into_air + into_glass == Approx(radiated).epsilon(1e-2));
  // collection: an objective of NA 0.5 above the dipole collects a part of the upper power
  const Real cone = std::asin(0.5);
  const auto collected = physics::axisymmetric_layered_far_field(
      meridian, azimuthal, field.meridian, field.azimuthal, 0, omega, setup.materials, around,
      stack, angles(0.0, cone, 101), {});
  REQUIRE(collected.up.power_between(0.0, cone) ==
          Approx(collected.up.radiated_power()).epsilon(1e-14));
  // the dense samples end one interval below the cone
  REQUIRE(collected.up.radiated_power() == Approx(dense.up.power_between(0.0, cone)).epsilon(3e-2));
  REQUIRE(collected.up.radiated_power() > 0);
  REQUIRE(collected.up.radiated_power() < into_air);
  // validation
  REQUIRE_THROWS_AS(physics::axisymmetric_layered_far_field(
                        meridian, azimuthal, field.meridian, field.azimuthal, 0, omega,
                        setup.materials, around, stack, {kPi / 2}, {}),
                    InvalidArgument);
  REQUIRE_THROWS_AS(
      physics::axisymmetric_layered_far_field(meridian, azimuthal, field.meridian, field.azimuthal,
                                              0, omega, setup.materials, around, stack, {}, {0.3}),
      InvalidArgument);
  const LayerStack<3> lossy(Material::vacuum(), {}, Material{Complex{2.0, 0.1}, Complex{1.0, 0.0}});
  REQUIRE_THROWS_AS(
      physics::axisymmetric_layered_far_field(meridian, azimuthal, field.meridian, field.azimuthal,
                                              0, omega, setup.materials, around, lossy, {}, {2.5}),
      InvalidArgument);
}

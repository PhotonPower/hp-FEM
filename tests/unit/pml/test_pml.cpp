#include <cmath>
#include <numbers>
#include <sstream>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <spdlog/sinks/ostream_sink.h>

#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"
#include "hpfem/pml/pml.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::kI;
using hpfem::Point;
using hpfem::Real;
using hpfem::pml::PmlBox;
using hpfem::pml::PmlProfile;

TEST_CASE("PmlBox: stretch is one inside, continuous at the interface, sigma_max by design",
          "[pml]") {
  const Real k0 = 2 * std::numbers::pi;
  const PmlProfile profile{3, 1e-8};
  const PmlBox<2> pml =
      PmlBox<2>::uniform(Point<2>::Zero(), Point<2>::Ones(), 0.5, k0, 1.0, profile);
  REQUIRE(pml.outer_lower() == Point<2>(-0.5, -0.5));
  REQUIRE(pml.outer_upper() == Point<2>(1.5, 1.5));
  REQUIRE_FALSE(pml.in_layer(Point<2>(0.3, 0.7)));
  REQUIRE(pml.in_layer(Point<2>(1.2, 0.7)));
  REQUIRE(pml.in_layer(Point<2>(0.5, -0.1)));
  REQUIRE(pml.stretch(Point<2>(0.3, 0.7)) == hpfem::pml::ComplexCoordinates<2>::Ones());
  // sigma_max = -(m + 1) ln R0 / (2 k0 n d)
  const Real expected = -4.0 * std::log(1e-8) / (2.0 * k0 * 0.5);
  for (std::size_t side = 0; side < 4; ++side) REQUIRE(pml.sigma_max(side) == Approx(expected));
  // continuity and the polynomial profile along +x
  REQUIRE(std::abs(pml.stretch(Point<2>(1.0 + 1e-9, 0.5))(0) - 1.0) < 1e-20);
  REQUIRE(pml.stretch(Point<2>(1.25, 0.5))(0).imag() == Approx(expected * std::pow(0.5, 3)));
  REQUIRE(pml.stretch(Point<2>(1.5, 0.5))(0).imag() == Approx(expected));
  REQUIRE(pml.stretch(Point<2>(1.25, 0.5))(1) == Complex{1.0, 0.0});
  // the lower side mirrors the upper one
  REQUIRE(pml.stretch(Point<2>(-0.25, 0.5))(0) == pml.stretch(Point<2>(1.25, 0.5))(0));
  // the real part stays 1: no change of the wave speed, only absorption
  REQUIRE(pml.stretch(Point<2>(1.4, 1.3))(0).real() == 1.0);
  REQUIRE(pml.stretch(Point<2>(1.4, 1.3))(1).real() == 1.0);
}

TEST_CASE("PmlBox: stretched coordinates integrate the stretch and absorb outgoing waves",
          "[pml]") {
  const Real k0 = 3.0;
  const PmlProfile profile{2, 1e-6};
  PmlBox<2>::Thickness thickness{0.0, 0.4, 0.3, 0.0};  // layers on x-max and y-min only
  const PmlBox<2> pml(Point<2>(-1.0, 0.0), Point<2>(1.0, 2.0), thickness, k0, 1.5, profile);
  REQUIRE(pml.sigma_max(0) == 0.0);
  REQUIRE(pml.sigma_max(3) == 0.0);
  REQUIRE_FALSE(pml.in_layer(Point<2>(-1.5, 1.0)));  // no layer on x-min
  REQUIRE(pml.in_layer(Point<2>(1.2, 1.0)));
  // d x~ / dx = s by central differences
  const Real h = 1e-6;
  for (const auto& x : {Point<2>(1.1, 1.0), Point<2>(1.37, -0.2), Point<2>(0.0, -0.15)}) {
    for (int d = 0; d < 2; ++d) {
      Point<2> step = Point<2>::Zero();
      step(d) = h;
      const Complex fd =
          (pml.stretched_coordinate(x + step)(d) - pml.stretched_coordinate(x - step)(d)) / (2 * h);
      REQUIRE(std::abs(fd - pml.stretch(x)(d)) < 1e-8);
    }
  }
  // outgoing wave e^{i k n x} through the x-max layer: amplitude sqrt(R0) at the far end,
  // and e^{-i k n y} through the y-min layer likewise
  const Real k = k0 * 1.5;
  const Complex xt = pml.stretched_coordinate(Point<2>(1.4, 1.0))(0);
  REQUIRE(std::abs(std::exp(kI * k * xt)) == Approx(std::sqrt(1e-6)).epsilon(1e-10));
  const Complex yt = pml.stretched_coordinate(Point<2>(0.0, -0.3))(1);
  REQUIRE(std::abs(std::exp(-kI * k * yt)) == Approx(std::sqrt(1e-6)).epsilon(1e-10));
  // the sign is the absorbing one for exp(-i omega t): the wave must not grow
  REQUIRE(std::abs(std::exp(kI * k * xt)) < 1.0);
  // beyond the layer the stretch stays at sigma_max
  REQUIRE(pml.stretch(Point<2>(2.0, 1.0))(0).imag() == Approx(pml.sigma_max(1)));
}

TEST_CASE("PmlBox: recommended thickness follows the wavelength and the cell size", "[pml]") {
  // lambda = 1 m in vacuum: half a wavelength rounded up to whole cells
  const Real k0 = 2 * std::numbers::pi;
  REQUIRE(PmlBox<2>::recommended_thickness(k0, 1.0, 0.125) == Approx(0.5));
  REQUIRE(PmlBox<2>::recommended_thickness(k0, 1.0, 0.3) == Approx(0.6));
  REQUIRE(PmlBox<2>::recommended_thickness(k0, 2.0, 0.125) == Approx(0.25));  // lambda / n
  REQUIRE(PmlBox<2>::recommended_thickness(k0, 1.0, 0.125, 1.0) == Approx(1.0));
  REQUIRE(PmlBox<2>::recommended_thickness(k0, 1.0, 5.0) == Approx(5.0));  // at least one cell
  REQUIRE_THROWS_AS(PmlBox<2>::recommended_thickness(0.0, 1.0, 0.1), hpfem::InvalidArgument);
}

TEST_CASE("PmlBox: effective material tensors in 2D and 3D", "[pml]") {
  const PmlBox<2> p2 = PmlBox<2>::uniform(Point<2>::Zero(), Point<2>::Ones(), 0.5, 4.0);
  const Point<2> corner(1.3, -0.2);
  const auto s = p2.stretch(corner);
  const Complex eps_r{2.0, 0.1};
  const Complex mu_r{1.0, 0.0};
  const auto eps = p2.permittivity(eps_r, corner);
  REQUIRE(std::abs(eps(0, 0) - eps_r * s(1) / s(0)) < 1e-14);
  REQUIRE(std::abs(eps(1, 1) - eps_r * s(0) / s(1)) < 1e-14);
  REQUIRE(eps(0, 1) == Complex{0.0, 0.0});
  const auto inv = p2.inverse_permeability(mu_r, corner);
  REQUIRE(std::abs(inv(0, 0) - 1.0 / (mu_r * s(0) * s(1))) < 1e-14);
  // identity inside the box
  const auto inside = p2.permittivity(eps_r, Point<2>(0.5, 0.5));
  REQUIRE(std::abs(inside(0, 0) - eps_r) < 1e-15);
  REQUIRE(std::abs(p2.inverse_permeability(mu_r, Point<2>(0.5, 0.5))(0, 0) - 1.0) < 1e-15);

  const PmlBox<3> p3 = PmlBox<3>::uniform(Point<3>::Zero(), Point<3>::Ones(), 0.25, 4.0);
  const Point<3> edge(1.1, 0.5, -0.2);
  const auto s3 = p3.stretch(edge);
  REQUIRE(s3(1) == Complex{1.0, 0.0});
  const auto eps3 = p3.permittivity(eps_r, edge);
  const auto inv3 = p3.inverse_permeability(mu_r, edge);
  const Complex det = s3(0) * s3(1) * s3(2);
  for (int d = 0; d < 3; ++d) {
    REQUIRE(std::abs(eps3(d, d) - eps_r * det / (s3(d) * s3(d))) < 1e-14);
    REQUIRE(std::abs(inv3(d, d) - s3(d) * s3(d) / (mu_r * det)) < 1e-14);
  }
  // eps~ mu~ = eps mu det^2 Lambda^-4 ... the product of the diagonal tensors equals
  // eps_r mu_r for every axis once the inverse permeability is inverted: (eps~)(mu~) = eps mu I
  for (int d = 0; d < 3; ++d) {
    REQUIRE(std::abs(eps3(d, d) / inv3(d, d) -
                     eps_r * mu_r * det * det / (s3(d) * s3(d)) / (s3(d) * s3(d))) < 1e-12);
  }

  REQUIRE_THROWS_AS(PmlBox<2>::uniform(Point<2>::Ones(), Point<2>::Zero(), 0.5, 4.0),
                    hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(PmlBox<2>::uniform(Point<2>::Zero(), Point<2>::Ones(), -0.5, 4.0),
                    hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(PmlBox<2>::uniform(Point<2>::Zero(), Point<2>::Ones(), 0.5, 0.0),
                    hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(
      PmlBox<2>::uniform(Point<2>::Zero(), Point<2>::Ones(), 0.5, 4.0, 1.0, PmlProfile{3, 2.0}),
      hpfem::InvalidArgument);
}

TEST_CASE("PmlProfile::for_angle: reflection for the largest incidence angle", "[pml]") {
  using hpfem::pml::PmlProfile;
  constexpr hpfem::Real deg = std::numbers::pi / 180.0;
  // R0 = (target / (2 r))^(2 / cos theta): the values of the specification (target 1e-4)
  struct Row {
    hpfem::Real r, angle_deg, reflection;
  };
  const Row rows[] = {{1.0, 50.0, 4.1e-14},  {1.0, 60.0, 6.3e-18},  {1.0, 70.0, 7.1e-26},
                      {1.0, 72.0, 1.5e-28},  {0.31, 50.0, 1.6e-12}, {0.31, 60.0, 6.8e-16},
                      {0.31, 70.0, 6.7e-23}, {0.31, 72.0, 2.9e-25}};
  for (const Row& row : rows) {
    const PmlProfile profile = PmlProfile::for_angle(row.angle_deg * deg, 1e-4, row.r);
    const hpfem::Real exact = std::pow(1e-4 / (2 * row.r), 2.0 / std::cos(row.angle_deg * deg));
    REQUIRE(profile.reflection == Approx(exact).epsilon(1e-12));
    REQUIRE(profile.reflection == Approx(row.reflection).epsilon(0.03));  // two digits
    REQUIRE(profile.order == 2);
  }
  // monotonic: steeper angles and smaller targets need smaller R0; the normal incidence value
  // is the one-way bound (target / 2 r)^2
  REQUIRE(PmlProfile::for_angle(0.0, 1e-4).reflection == Approx(2.5e-9));
  REQUIRE(PmlProfile::for_angle(70 * deg, 1e-4).reflection <
          PmlProfile::for_angle(50 * deg, 1e-4).reflection);
  REQUIRE(PmlProfile::for_angle(50 * deg, 1e-6).reflection <
          PmlProfile::for_angle(50 * deg, 1e-4).reflection);
  REQUIRE(PmlProfile::for_angle(50 * deg, 1e-4, 1.0, 3).order == 3);
  // clamping: nothing below 1e-300, nothing at or above 1
  REQUIRE(PmlProfile::for_angle(89.9 * deg, 1e-12).reflection == 1e-300);
  REQUIRE(PmlProfile::for_angle(0.0, 10.0).reflection < 1.0);
  // the clamped profiles are usable in a box
  const hpfem::pml::PmlBox<2> box(hpfem::Point<2>(0, 0), hpfem::Point<2>(1, 1), {0, 1, 0, 0}, 1.0,
                                  1.0, PmlProfile::for_angle(0.0, 10.0));
  REQUIRE(box.sigma_max(1) > 0);
  CHECK_THROWS_AS(PmlProfile::for_angle(-0.1, 1e-4), hpfem::InvalidArgument);
  CHECK_THROWS_AS(PmlProfile::for_angle(std::numbers::pi / 2, 1e-4), hpfem::InvalidArgument);
  CHECK_THROWS_AS(PmlProfile::for_angle(0.5, 0.0), hpfem::InvalidArgument);
  CHECK_THROWS_AS(PmlProfile::for_angle(0.5, 1e-4, 0.0), hpfem::InvalidArgument);
  CHECK_THROWS_AS(PmlProfile::for_angle(0.5, 1e-4, 1.0, 0), hpfem::InvalidArgument);
}

TEST_CASE("PmlBox: resolution |k s| h and the thickness that keeps a steep profile resolved",
          "[pml]") {
  using hpfem::pml::PmlBox;
  using hpfem::pml::PmlProfile;
  const hpfem::Real k0 = 2 * std::numbers::pi / 405e-9;
  const PmlProfile profile{2, 1e-16};
  const hpfem::Real thickness = 1215e-9;  // three wavelengths in air
  const PmlBox<2> box(hpfem::Point<2>(0, 0), hpfem::Point<2>(400e-9, 400e-9), {0, thickness, 0, 0},
                      k0, 1.0, profile);
  REQUIRE(box.k0() == k0);
  // by hand: sigma_max = -(m + 1) ln R0 / (2 k0 n d), |s| = sqrt(1 + sigma_max^2)
  const hpfem::Real sigma = -3.0 * std::log(1e-16) / (2 * k0 * thickness);
  REQUIRE(box.sigma_max(1) == Approx(sigma).epsilon(1e-12));
  const hpfem::Real h = 37e-9;
  const hpfem::Real air = k0 * 1.0 * std::sqrt(1 + sigma * sigma) * h;
  REQUIRE(box.max_resolution(h, 1.0) == Approx(air).epsilon(1e-12));
  REQUIRE(air < PmlBox<2>::resolution_limit(5));  // the air PML of the report: 1.5 to 2.8
  // the same layer meshed with silicon (n = 5.44) is not resolved: 12 to 38 in the report
  const hpfem::Real silicon = box.max_resolution(h, 5.44);
  REQUIRE(silicon == Approx(5.44 * air).epsilon(1e-12));
  REQUIRE(silicon > PmlBox<2>::resolution_limit(5));
  // sides without a layer do not count
  REQUIRE(PmlBox<2>(hpfem::Point<2>(0, 0), hpfem::Point<2>(1, 1), {0, 0, 0, 0}, k0)
              .max_resolution(h, 1.0) == 0.0);
  CHECK_THROWS_AS(box.max_resolution(0.0, 1.0), hpfem::InvalidArgument);
  // the limit: 3 from p = 4, 0.75 p below
  REQUIRE(PmlBox<2>::resolution_limit(4) == 3.0);
  REQUIRE(PmlBox<2>::resolution_limit(6) == 3.0);
  REQUIRE(PmlBox<2>::resolution_limit(2) == Approx(1.5));
  // recommended thickness with the profile: the smallest whole-cell thickness whose far end
  // satisfies the limit, never below the half-wavelength rule
  for (const int p : {2, 4, 5}) {
    const hpfem::Real d = PmlBox<2>::recommended_thickness(k0, 1.0, h, profile, p);
    REQUIRE(std::abs(d / h - std::round(d / h)) < 1e-9);
    REQUIRE(d >= PmlBox<2>::recommended_thickness(k0, 1.0, h));
    const PmlBox<2> thick(hpfem::Point<2>(0, 0), hpfem::Point<2>(1e-6, 1e-6), {0, d, 0, 0}, k0, 1.0,
                          profile);
    REQUIRE(thick.max_resolution(h, 1.0) <= PmlBox<2>::resolution_limit(p) * (1 + 1e-12));
    if (d > h) {
      const PmlBox<2> thinner(hpfem::Point<2>(0, 0), hpfem::Point<2>(1e-6, 1e-6), {0, d - h, 0, 0},
                              k0, 1.0, profile);
      const bool half_wave_bound = d - h < PmlBox<2>::recommended_thickness(k0, 1.0, h);
      REQUIRE((half_wave_bound || thinner.max_resolution(h, 1.0) > PmlBox<2>::resolution_limit(p)));
    }
  }
  // a mesh too coarse for the wavelength cannot be rescued by thickness
  CHECK_THROWS_AS(PmlBox<2>::recommended_thickness(k0, 5.44, 200e-9, profile, 2),
                  hpfem::InvalidArgument);
}

TEST_CASE("Scattering warns about an under-resolved PML in a high-index substrate", "[pml]") {
  using hpfem::Complex;
  using hpfem::Point;
  using hpfem::pml::PmlBox;
  using hpfem::pml::PmlProfile;
  // a layer designed for air (background_index 1) that the mesh fills with silicon
  const hpfem::Real k0 = 2 * std::numbers::pi / 405e-9;
  const hpfem::Real h = 50e-9;
  hpfem::mesh::Mesh<2> mesh =
      hpfem::mesh::rectangle(8, 8, Point<2>(0.0, 0.0), Point<2>(8 * h, 8 * h));
  for (hpfem::Index c = 0; c < mesh.num_cells(); ++c) mesh.set_cell_tag(c, 2);
  const hpfem::fespace::NedelecDofMap<2> dofs(mesh, 4);
  const auto run = [&](Complex eps) {
    hpfem::physics::ScatteringSetup<2> setup;
    setup.omega = k0 * hpfem::constants::c0;
    setup.materials.set(2, hpfem::materials::Material{eps, Complex{1.0, 0.0}});
    setup.incident = hpfem::physics::plane_wave<2>(
        hpfem::assembly::ComplexVector<2>(Complex{0.0, 0.0}, Complex{1.0, 0.0}),
        Point<2>(k0 * std::sqrt(eps).real(), 0.0));
    setup.formulation = hpfem::physics::Formulation::kScatteredField;
    // a mild profile on four cells: |k s| h = 2.6 in air (resolved at p = 4), 14 in silicon
    setup.pml = PmlBox<2>(Point<2>(0.0, 0.0), Point<2>(4 * h, 8 * h), {0, 4 * h, 0, 0}, k0, 1.0,
                          PmlProfile{2, 1e-2});
    std::ostringstream stream;
    const auto sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(stream);
    hpfem::log().sinks().push_back(sink);
    const auto level = hpfem::log().level();
    hpfem::log().set_level(spdlog::level::warn);
    const hpfem::physics::Scattering<2> problem(dofs, setup);
    hpfem::log().set_level(level);
    hpfem::log().sinks().pop_back();
    return stream.str();
  };
  const std::string silicon = run(Complex{29.6345, 2.7721});
  REQUIRE(silicon.find("PML is under-resolved") != std::string::npos);
  const std::string air = run(Complex{1.0, 0.0});
  REQUIRE(air.find("PML is under-resolved") == std::string::npos);
}

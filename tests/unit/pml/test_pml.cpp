#include <cmath>
#include <numbers>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/core/error.hpp"
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

// Axisymmetric resonances: the cylindrical PML tensors reduce to the material inside the
// box and follow the Teixeira-Chew formula in the layers, the quasi-normal modes of a
// dielectric sphere (order m = 1) match the zeros of the l = 1 Mie denominators, and the
// setup is validated.
#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "axisymmetric_sphere.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/physics/axisymmetric.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::physics::AxisymmetricResonance;
using hpfem::physics::AxisymmetricResonanceSetup;

TEST_CASE("cylindrical PML tensors", "[physics][axisymmetric][pml]") {
  const Real k0 = 2.0;
  const hpfem::pml::PmlBox<2> box(Point<2>(0.0, -1.0), Point<2>(1.0, 1.0), {0.0, 0.5, 0.5, 0.5},
                                  k0);
  const hpfem::materials::Material material = hpfem::materials::Material::dielectric(1.5);
  const auto form = hpfem::physics::axisymmetric_pml_form(box, material);
  // inside the box: plain material
  const Point<2> inside(0.5, 0.2);
  REQUIRE(form.permittivity(inside)(0).real() == Approx(2.25));
  REQUIRE(form.permittivity(inside)(1).real() == Approx(2.25));
  REQUIRE(form.inverse_permeability(inside)(2).real() == Approx(1.0));
  // in the radial layer: Lambda = diag(s_phi s_z / s_r, s_r s_z / s_phi, s_r s_phi / s_z)
  const Point<2> layer(1.3, 0.2);
  const auto s = box.stretch(layer);
  const Complex s_phi = box.stretched_coordinate(layer)(0) / layer(0);
  REQUIRE(s(0).imag() > 0);
  REQUIRE(s(1) == Complex{1.0, 0.0});
  const auto eps = form.permittivity(layer);
  REQUIRE(std::abs(eps(0) - 2.25 * s_phi * s(1) / s(0)) < 1e-12);
  REQUIRE(std::abs(eps(1) - 2.25 * s(0) * s(1) / s_phi) < 1e-12);
  REQUIRE(std::abs(eps(2) - 2.25 * s(0) * s_phi / s(1)) < 1e-12);
  const auto inv_mu = form.inverse_permeability(layer);
  REQUIRE(std::abs(inv_mu(0) * eps(0) - 2.25) < 1e-12);  // Lambda cancels
  // a layer on the axis side is rejected
  const hpfem::pml::PmlBox<2> bad(Point<2>(0.0, -1.0), Point<2>(1.0, 1.0), {0.5, 0.5, 0.5, 0.5},
                                  k0);
  REQUIRE_THROWS_AS(hpfem::physics::axisymmetric_pml_form(bad, material), hpfem::InvalidArgument);
}

TEST_CASE("sphere resonances of order m = 1 match the Mie poles", "[physics][axisymmetric]") {
  using namespace hpfem::physics::test;
  const Real n = 3.0;
  const Complex x_te = mie_pole(n, Polarisation::kTE);
  const Complex x_tm = mie_pole(n, Polarisation::kTM);
  REQUIRE(x_te.imag() < 0);
  REQUIRE(x_tm.imag() < 0);
  REQUIRE(x_te.real() < x_tm.real());
  const SphereProblem problem = sphere_problem(n, 4, 2, 0.5 * (x_te.real() + x_tm.real()), 8);
  const auto modes = problem.resonance->solve();
  REQUIRE_FALSE(modes.empty());
  for (const auto& mode : modes) REQUIRE(mode.residual < 1e-8);
  const Complex k_te = closest(modes, x_te);
  const Complex k_tm = closest(modes, x_tm);
  REQUIRE(std::abs(k_te - x_te) < 2e-2 * std::abs(x_te));
  REQUIRE(std::abs(k_tm - x_tm) < 2e-2 * std::abs(x_tm));
  const auto& first = modes.front();
  REQUIRE(first.quality == Approx(first.omega.real() / (-2 * first.omega.imag())));
  REQUIRE(first.wavelength ==
          Approx(2 * std::numbers::pi * hpfem::constants::c0 / first.omega.real()));
  REQUIRE(first.meridian.size() == problem.meridian->num_dofs());

  // validation
  AxisymmetricResonanceSetup bad = problem.resonance->setup();
  bad.target_omega = 0;
  REQUIRE_THROWS_AS(AxisymmetricResonance(*problem.meridian, *problem.azimuthal, bad),
                    hpfem::InvalidArgument);
  bad = problem.resonance->setup();
  bad.axis_tag = 99;
  REQUIRE_THROWS_AS(AxisymmetricResonance(*problem.meridian, *problem.azimuthal, bad),
                    hpfem::InvalidArgument);
}

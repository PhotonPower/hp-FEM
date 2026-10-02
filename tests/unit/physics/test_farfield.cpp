// Far field by Stratton-Chu: the analytic dipole fields sampled on a closed curved surface
// must reproduce the dipole's far-field pattern in 2D and 3D, the radiated power must match
// the Poynting flux through the surface, and the far-field cross-section of the Mie
// cylinder must agree with the flux-based one. Diffraction orders of a Bloch plane wave.
#include <cmath>
#include <numbers>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/periodic.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/point_location.hpp"
#include "hpfem/physics/diffraction.hpp"
#include "hpfem/physics/farfield.hpp"
#include "hpfem/physics/mie.hpp"
#include "hpfem/physics/postprocess.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"
#include "hpfem/pml/pml.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::Index;
using hpfem::kI;
using hpfem::Point;
using hpfem::Real;
using hpfem::assembly::bloch_phase;
using hpfem::assembly::ComplexVector;
using hpfem::assembly::PeriodicPair;
using hpfem::fespace::NedelecDofMap;
using hpfem::materials::Material;
using hpfem::materials::MaterialMap;
using hpfem::mesh::ball;
using hpfem::mesh::disc;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::physics::analytic_field;
using hpfem::physics::diffraction_efficiencies;
using hpfem::physics::dipole_field;
using hpfem::physics::FarField;
using hpfem::physics::Formulation;
using hpfem::physics::fourier_coefficients;
using hpfem::physics::plane_wave;
using hpfem::physics::poynting_flux;
using hpfem::physics::Scattering;
using hpfem::physics::ScatteringSetup;
using hpfem::physics::Surface;
namespace box_tag = hpfem::mesh::box_tag;

TEST_CASE("FarField: line dipole pattern and power from a circle (2D)", "[physics][farfield]") {
  const Real k = 5.0;
  const Real omega = k * hpfem::constants::c0;
  const Point<2> x0(0.1, -0.05);
  const ComplexVector<2> p(Complex{1.0, 0.0}, Complex{0.4, -0.3});
  const auto field = dipole_field<2>(x0, p, k);
  const Mesh<2> m = disc(12, x0, 0.6);
  const FarField<2> far(m, Surface<2>::whole_boundary(m), analytic_field<2>(field), omega,
                        Material::vacuum(), 10);
  REQUIRE(far.wavenumber() == Approx(k));
  // F = (i/4) sqrt(2 / (pi k)) e^{-i pi/4} p_t, with p_t the transverse part and the
  // reference point x0: the pattern of a dipole at x0 carries the phase e^{-i k r.x0}
  const Complex prefactor =
      0.25 * kI * std::sqrt(2.0 / (std::numbers::pi * k)) * std::exp(-kI * std::numbers::pi / 4.0);
  for (const Real phi : {0.0, 0.7, 2.0, 3.9, 5.5}) {
    const Point<2> r(std::cos(phi), std::sin(phi));
    const Point<2> t(-std::sin(phi), std::cos(phi));
    const Complex p_t = p(0) * t(0) + p(1) * t(1);
    const ComplexVector<2> expected =
        prefactor * std::exp(-kI * k * r.dot(x0)) * p_t * t.cast<Complex>();
    const ComplexVector<2> f = far.pattern(r);
    REQUIRE((f - expected).norm() < 2e-3 * p.norm() * std::abs(prefactor));
  }
  // radiated power equals the Poynting flux through the surface
  const Real flux = poynting_flux<2>(m, Surface<2>::whole_boundary(m), analytic_field<2>(field),
                                     omega, MaterialMap{}, 10);
  REQUIRE(far.radiated_power(360) == Approx(flux).epsilon(5e-3));
  REQUIRE(far.scattering_cross_section(2.0) ==
          Approx(far.radiated_power() * 2 * hpfem::constants::Z0 / 4.0));
  REQUIRE_THROWS_AS(far.radiated_power(2), hpfem::InvalidArgument);
}

TEST_CASE("FarField: Hertz dipole pattern and power from a sphere (3D)", "[physics][farfield]") {
  const Real k = 4.0;
  const Real omega = k * hpfem::constants::c0;
  const Point<3> x0(0.0, 0.1, 0.0);
  const ComplexVector<3> p(Complex{0.0, 0.0}, Complex{1.0, 0.0}, Complex{0.5, 0.2});
  const auto field = dipole_field<3>(x0, p, k);
  const Mesh<3> m = ball(5, x0, 0.5);
  const FarField<3> far(m, Surface<3>::whole_boundary(m), analytic_field<3>(field), omega,
                        Material::vacuum(), 6);
  // F = p_t / (4 pi) times the reference phase
  for (const auto& dir :
       {Point<3>(1.0, 0.0, 0.0), Point<3>(0.3, -0.5, 0.8), Point<3>(0.0, 0.0, -1.0)}) {
    const Point<3> r = dir.normalized();
    Complex radial = 0;
    for (int d = 0; d < 3; ++d) radial += r(d) * p(d);
    const ComplexVector<3> expected =
        std::exp(-kI * k * r.dot(x0)) / (4.0 * std::numbers::pi) * (p - radial * r.cast<Complex>());
    REQUIRE((far.pattern(r) - expected).norm() < 1e-2 * p.norm() / (4.0 * std::numbers::pi));
  }
  const Real flux = poynting_flux<3>(m, Surface<3>::whole_boundary(m), analytic_field<3>(field),
                                     omega, MaterialMap{}, 6);
  REQUIRE(far.radiated_power(72) == Approx(flux).epsilon(2e-2));
  REQUIRE_THROWS_AS(FarField<3>(m, Surface<3>::whole_boundary(m), analytic_field<3>(field), omega,
                                Material{Complex{2.0, 0.1}, Complex{1.0, 0.0}}, 4),
                    hpfem::InvalidArgument);
}

TEST_CASE("FarField: Mie cylinder cross-section from the far field matches the flux",
          "[physics][farfield]") {
  const Real r0 = 0.25;
  const Real k = 6.0;
  const Mesh<2> mesh = hpfem::mesh::square_with_disc(2, r0, 1.0, 2.0, 2);
  const NedelecDofMap<2> dofs(mesh, 2);
  ScatteringSetup<2> setup;
  setup.omega = k * hpfem::constants::c0;
  setup.materials.set(2, Material::dielectric(1.5));
  setup.incident = plane_wave<2>(ComplexVector<2>(0.0, 1.0), Point<2>(k, 0.0));
  setup.formulation = Formulation::kScatteredField;
  setup.pml = hpfem::pml::PmlBox<2>::uniform(Point<2>(-1.0, -1.0), Point<2>(1.0, 1.0), 1.0, k, 1.0,
                                             hpfem::pml::PmlProfile{2, 1e-10});
  setup.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  const Scattering<2> problem(dofs, setup);
  const auto solution = problem.solve();
  const auto surface = Surface<2>::around_cells(mesh, 2);
  const auto cs = hpfem::physics::cross_sections<2>(problem, solution, surface, 1.0);
  const FarField<2> far(mesh, surface, hpfem::physics::discrete_field<2>(dofs, solution.unknown),
                        setup.omega, Material::vacuum(), 8);
  const Real exact = hpfem::physics::mie_cylinder_scattering_width(k, r0, 1.5);
  REQUIRE(far.scattering_cross_section(1.0) == Approx(cs.scattering).epsilon(2e-2));
  REQUIRE(far.scattering_cross_section(1.0) == Approx(exact).epsilon(3e-2));
}

TEST_CASE("Diffraction orders of a Bloch plane wave: zeroth order only, efficiency one",
          "[physics][diffraction]") {
  const Real k0 = 4.0;
  const Real angle = 0.5;
  const Point<2> k = k0 * Point<2>(std::cos(angle), std::sin(angle));
  const ComplexVector<2> e0(Complex{-std::sin(angle), 0.0}, Complex{std::cos(angle), 0.0});
  const auto wave = plane_wave<2>(2.0 * e0, k);
  // analytic field: A_0 = 2 e0 e^{i kx x0}, other orders vanish
  const auto coefficients = fourier_coefficients(wave.value, 0.3, 0.0, 1.0, k(1), 2, 32);
  REQUIRE(coefficients.size() == 5);
  const ComplexVector<2> expected = 2.0 * std::exp(kI * k(0) * 0.3) * e0;
  REQUIRE((coefficients[2] - expected).norm() < 1e-10);
  for (const std::size_t i : {0U, 1U, 3U, 4U}) REQUIRE(coefficients[i].norm() < 1e-10);
  const auto orders = diffraction_efficiencies(coefficients, k0, 1.0, 1.0, k(1), k(0), 2.0);
  REQUIRE(orders.size() == 5);
  REQUIRE(orders[2].order == 0);
  REQUIRE(orders[2].propagating);
  REQUIRE(orders[2].efficiency == Approx(1.0).epsilon(1e-10));
  REQUIRE(orders[2].kx.real() == Approx(k(0)));
  // orders with |ky| > k are evanescent
  for (const auto& o : orders) {
    if (o.order != 0) {
      REQUIRE(o.efficiency == 0.0);
      REQUIRE_FALSE(o.propagating);
      REQUIRE(o.kx.imag() > 0);
    }
  }
  // the discrete Bloch solution gives the same coefficients up to discretisation error
  const Mesh<2> m = rectangle(6, 6);
  const NedelecDofMap<2> dofs(m, 3);
  ScatteringSetup<2> setup;
  setup.omega = k0 * hpfem::constants::c0;
  setup.incident = wave;
  setup.incident_tags = {box_tag::kXMin, box_tag::kXMax};
  setup.periodic = {PeriodicPair<2>{box_tag::kYMin, box_tag::kYMax, Point<2>(0.0, 1.0),
                                    bloch_phase<2>(k, Point<2>(0.0, 1.0))}};
  const Scattering<2> problem(dofs, setup);
  const auto solution = problem.solve();
  const hpfem::mesh::PointLocator<2> locator(m);
  const auto discrete =
      fourier_coefficients(dofs, solution.unknown, locator, 0.3, 0.0, 1.0, k(1), 2, 48);
  REQUIRE((discrete[2] - expected).norm() < 1e-2);
  REQUIRE(discrete[1].norm() < 1e-2);
  REQUIRE_THROWS_AS(
      fourier_coefficients(dofs, solution.unknown, locator, 1.5, 0.0, 1.0, k(1), 1, 8),
      hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(diffraction_efficiencies(coefficients, 0.0, 1.0, 1.0, k(1), k(0), 2.0),
                    hpfem::InvalidArgument);
}

// Transient solver: time signals, setup validation, energy conservation of the Newmark
// trapezoidal rule on a PEC cavity (2D and 3D), monotone decay with conductivity, and a
// pulse leaving a strip through the first-order absorbing boundary.
#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/interpolation.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/physics/time_domain.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::assembly::ComplexVector;
using hpfem::constants::c0;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::box;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::physics::gaussian_pulse;
using hpfem::physics::modulated_gaussian;
using hpfem::physics::TimeDomain;
using hpfem::physics::TimeDomainSetup;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kPi = std::numbers::pi;

/// TE_11 mode of the unit square, E = rot(cos(pi x) cos(pi y)).
ComplexVector<2> mode(const Point<2>& x) {
  return ComplexVector<2>(-kPi * std::cos(kPi * x(0)) * std::sin(kPi * x(1)),
                          kPi * std::sin(kPi * x(0)) * std::cos(kPi * x(1)));
}

std::vector<hpfem::mesh::Tag> sides_2d() {
  return {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
}

}  // namespace

TEST_CASE("time signals: derivatives match finite differences", "[physics][time]") {
  const auto pulse = gaussian_pulse(2.0, 0.5);
  const auto wave = modulated_gaussian(3.0, 2.0, 0.7);
  REQUIRE(pulse.value(2.0) == Approx(1.0));
  REQUIRE(wave.value(2.0) == Approx(0.0).margin(1e-14));
  const Real h = 1e-5;
  for (const Real t : {1.0, 1.7, 2.3, 3.1}) {
    REQUIRE(pulse.derivative(t) ==
            Approx((pulse.value(t + h) - pulse.value(t - h)) / (2 * h)).epsilon(1e-6));
    REQUIRE(wave.derivative(t) ==
            Approx((wave.value(t + h) - wave.value(t - h)) / (2 * h)).epsilon(1e-6));
  }
  REQUIRE_THROWS_AS(gaussian_pulse(0.0, 0.0), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(modulated_gaussian(-1.0, 0.0, 1.0), hpfem::InvalidArgument);
}

TEST_CASE("time domain: setup validation", "[physics][time]") {
  const Mesh<2> mesh = rectangle(2, 2);
  const NedelecDofMap<2> dofs(mesh, 1);
  TimeDomainSetup<2> setup;
  REQUIRE_THROWS_AS(TimeDomain<2>(dofs, setup), hpfem::InvalidArgument);  // dt = 0
  setup.dt = 1e-12;
  setup.gamma = 0.4;
  REQUIRE_THROWS_AS(TimeDomain<2>(dofs, setup), hpfem::InvalidArgument);
  setup.gamma = 0.5;
  setup.materials = hpfem::materials::MaterialMap{
      hpfem::materials::Material{Complex{2.0, 0.1}, Complex{1.0, 0.0}}};  // lossy background
  REQUIRE_THROWS_AS(TimeDomain<2>(dofs, setup), hpfem::InvalidArgument);  // lossy
  setup.materials = hpfem::materials::MaterialMap{};
  setup.current = [](const Point<2>&) { return ComplexVector<2>(1.0, 0.0); };
  REQUIRE_THROWS_AS(TimeDomain<2>(dofs, setup), hpfem::InvalidArgument);  // no signal
  setup.current = {};
  setup.absorbing_tags = {7};
  for (Index f = 0; f < mesh.num_facets(); ++f) {
    if (!mesh.is_boundary_facet(f)) {
      Mesh<2> tagged = mesh;
      tagged.set_facet_tag(f, 7);
      const NedelecDofMap<2> other(tagged, 1);
      REQUIRE_THROWS_AS(TimeDomain<2>(other, setup), hpfem::InvalidArgument);
      break;
    }
  }
  setup.absorbing_tags = {};
  setup.pec_tags = sides_2d();
  const TimeDomain<2> problem(dofs, setup);
  REQUIRE(problem.num_free_dofs() < dofs.num_dofs());
  REQUIRE_THROWS_AS(problem.initialize(Vector::Zero(2), Vector::Zero(dofs.num_dofs())),
                    hpfem::InvalidArgument);
  auto state = problem.initialize();
  REQUIRE(state.u.size() == dofs.num_dofs());
  REQUIRE(problem.energy(state) == 0.0);
  REQUIRE_THROWS_AS(problem.run(state, -1), hpfem::InvalidArgument);
}

TEST_CASE("time domain: the trapezoidal rule conserves the cavity energy and the mode phase",
          "[physics][time]") {
  const Mesh<2> mesh = rectangle(6, 6);
  const NedelecDofMap<2> dofs(mesh, 3);
  const Real omega = c0 * kPi * std::sqrt(2.0);
  const Real period = 2 * kPi / omega;
  TimeDomainSetup<2> setup;
  setup.pec_tags = sides_2d();
  setup.dt = period / 40;
  const TimeDomain<2> problem(dofs, setup);
  const Vector u0 =
      hpfem::assembly::interpolate<2>(dofs, hpfem::assembly::physical_sampler<2>(mode));
  auto state = problem.initialize(u0, Vector::Zero(dofs.num_dofs()));
  const Real e0 = problem.energy(state);
  REQUIRE(e0 > 0);
  Real max_drift = 0;
  problem.run(state, 80, [&](const hpfem::physics::TimeState<2>& s) {
    max_drift = std::max(max_drift, std::abs(problem.energy(s) - e0) / e0);
  });
  REQUIRE(max_drift < 1e-10);
  REQUIRE(state.step == 80);
  REQUIRE(state.time == Approx(2 * period));
  // after two periods the field is back at the initial one (phase error of the trapezoidal
  // rule: (omega dt)^2 / 12 per radian)
  REQUIRE((state.u - u0).norm() < 0.05 * u0.norm());
  // numerical damping with gamma > 1/2 reduces the energy
  setup.gamma = 0.6;
  setup.beta = 0.3025;
  const TimeDomain<2> damped(dofs, setup);
  auto state2 = damped.initialize(u0, Vector::Zero(dofs.num_dofs()));
  damped.run(state2, 40);
  REQUIRE(damped.energy(state2) < 0.99 * e0);
}

TEST_CASE("time domain: conductivity decays the energy monotonically, 3D cavity conserves",
          "[physics][time]") {
  const Mesh<2> mesh = rectangle(4, 4);
  const NedelecDofMap<2> dofs(mesh, 2);
  const Real omega = c0 * kPi * std::sqrt(2.0);
  TimeDomainSetup<2> setup;
  setup.pec_tags = sides_2d();
  setup.dt = 2 * kPi / omega / 30;
  setup.conductivity[mesh.cell_tag(0)] = 1e-3;  // all cells carry the same tag
  const TimeDomain<2> problem(dofs, setup);
  REQUIRE(problem.damping().norm() > 0);
  const Vector u0 =
      hpfem::assembly::interpolate<2>(dofs, hpfem::assembly::physical_sampler<2>(mode));
  auto state = problem.initialize(u0, Vector::Zero(dofs.num_dofs()));
  Real previous = problem.energy(state);
  problem.run(state, 30, [&](const hpfem::physics::TimeState<2>& s) {
    const Real e = problem.energy(s);
    REQUIRE(e <= previous * (1 + 1e-12));
    previous = e;
  });
  REQUIRE(previous < 0.9 * problem.energy(problem.initialize(u0, Vector::Zero(dofs.num_dofs()))));

  const Mesh<3> cube = box(2, 2, 2);
  const NedelecDofMap<3> dofs3(cube, 1);
  TimeDomainSetup<3> setup3;
  setup3.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin,
                     box_tag::kYMax, box_tag::kZMin, box_tag::kZMax};
  setup3.dt = 2 * kPi / omega / 20;
  const TimeDomain<3> cavity(dofs3, setup3);
  const Vector u3 = hpfem::assembly::interpolate<3>(
      dofs3, hpfem::assembly::physical_sampler<3>([](const Point<3>& x) {
        return ComplexVector<3>(std::sin(kPi * x(1)) * std::sin(kPi * x(2)),
                                std::sin(kPi * x(2)) * std::sin(kPi * x(0)),
                                std::sin(kPi * x(0)) * std::sin(kPi * x(1)));
      }));
  auto state3 = cavity.initialize(u3, Vector::Zero(dofs3.num_dofs()));
  const Real e3 = cavity.energy(state3);
  REQUIRE(e3 > 0);
  cavity.run(state3, 10);
  REQUIRE(cavity.energy(state3) == Approx(e3).epsilon(1e-10));
}

TEST_CASE("time domain: a pulse leaves the strip through the absorbing boundary",
          "[physics][time]") {
  // strip [0, 2] x [0, 0.25] with PEC top and bottom (E_y polarisation, H_z), a current
  // sheet in the middle radiating to both sides, absorbing ends: after the pulse has
  // reached the ends the energy must drop to a small fraction of its peak
  const Real wavelength = 0.5;
  const Real omega = 2 * kPi * c0 / wavelength;
  const Real period = 2 * kPi / omega;
  const Mesh<2> mesh = rectangle(32, 4, Point<2>(0.0, 0.0), Point<2>(2.0, 0.25));
  const NedelecDofMap<2> dofs(mesh, 2);
  TimeDomainSetup<2> setup;
  setup.pec_tags = {box_tag::kYMin, box_tag::kYMax};
  setup.absorbing_tags = {box_tag::kXMin, box_tag::kXMax};
  setup.current = [](const Point<2>& x) {
    const Real s = (x(0) - 1.0) / 0.05;
    return ComplexVector<2>(0.0, std::exp(-0.5 * s * s));
  };
  setup.signal = modulated_gaussian(omega, 2 * period, 0.6 * period);
  setup.dt = period / 20;
  const TimeDomain<2> problem(dofs, setup);
  REQUIRE(problem.damping().norm() > 0);
  auto state = problem.initialize();
  Real peak = 0;
  std::vector<Real> energies;
  problem.run(state, 120, [&](const hpfem::physics::TimeState<2>& s) {
    energies.push_back(problem.energy(s));
    peak = std::max(peak, energies.back());
  });
  // 120 steps = 6 periods = 6 * 0.5 m of travel: the pulse (centred at 2 T) reached the
  // ends (1 m away, 2 T of travel) by 4 T and has left
  REQUIRE(peak > 0);
  REQUIRE(energies.back() < 0.02 * peak);
  // without absorbing ends (PEC) the energy stays
  setup.absorbing_tags = {};
  setup.pec_tags = sides_2d();
  const TimeDomain<2> closed(dofs, setup);
  auto state2 = closed.initialize();
  Real peak2 = 0;
  closed.run(state2, 120, [&](const hpfem::physics::TimeState<2>& s) {
    peak2 = std::max(peak2, closed.energy(s));
  });
  REQUIRE(closed.energy(state2) > 0.5 * peak2);
}

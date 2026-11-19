// Dipole sources on the axis: the Gaussian dipole components per order, the vacuum power
// near the Larmor value, the transverse orders m = +-1 equal, the Purcell enhancement of
// the axial dipole at the centre of a dielectric sphere peaks at the TM_1 Mie pole found
// by the resonance solver, and the total-field setup is validated.
#include <algorithm>
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

TEST_CASE("kept factorisation: other heights and the order -m on one factorisation",
          "[physics][axisymmetric]") {
  using namespace hpfem::physics::test;
  using hpfem::Index;
  using hpfem::Vector;
  const Real x = 1.5;
  const Real sigma = 0.08;
  // a dielectric sphere and the PML: materials, stretch and the e-v coupling all enter
  const auto plus = dipole_problem(2.0, 4, 2, x, AxisDipole::kTransverse, 1, sigma);
  AxisymmetricScatteringSetup kept_setup = plus.scattering->setup();
  kept_setup.keep_factorisation = true;
  const AxisymmetricScattering kept_problem(*plus.meridian, *plus.azimuthal, kept_setup);
  const auto field = kept_problem.solve();
  REQUIRE(field.factorisation != nullptr);
  REQUIRE(plus.scattering->solve().factorisation == nullptr);
  const auto& kept = *field.factorisation;
  const auto block = [](const hpfem::physics::AxisymmetricScatteredField& f) {
    Vector out(f.meridian.size() + f.azimuthal.size());
    out << f.meridian, f.azimuthal;
    return out;
  };
  const Vector reference = block(field);
  const auto relative = [&](const Vector& a, const Vector& b) { return (a - b).norm() / b.norm(); };
  // the load of the problem itself (the current lives in every cell) gives back the solution
  REQUIRE(relative(kept.solve(kept_problem.load()), reference) < 1e-10);

  // another height: the load on the support of the Gaussian only (10 sigma around z0)
  const Real z0 = 0.2;
  AxisymmetricScatteringSetup shifted = plus.scattering->setup();
  shifted.current = axisymmetric_gaussian_dipole(z0, Complex{1.0, 0.0}, AxisDipole::kTransverse,
                                                 sigma, shifted.omega, 1);
  const AxisymmetricScattering shifted_problem(*plus.meridian, *plus.azimuthal, shifted);
  std::vector<Index> support;
  const auto& mesh = *plus.mesh;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    Point<2> centroid = Point<2>::Zero();
    Real diameter = 0;
    const auto& cv = mesh.cell_vertices(c);
    for (const Index v : cv) centroid += mesh.vertex(v) / static_cast<Real>(cv.size());
    for (const Index v : cv) diameter = std::max(diameter, (mesh.vertex(v) - centroid).norm());
    if ((centroid - Point<2>(0.0, z0)).norm() < 10 * sigma + diameter) support.push_back(c);
  }
  REQUIRE(support.size() < static_cast<std::size_t>(mesh.num_cells()) / 2);
  const Vector shifted_direct = block(shifted_problem.solve());
  const Vector shifted_load = shifted_problem.load(support);
  REQUIRE(relative(shifted_problem.solution_from(kept.solve(shifted_load)).meridian,
                   shifted_direct.head(plus.meridian->num_dofs())) < 1e-10);
  REQUIRE(relative(kept.solve(shifted_load), shifted_direct) < 1e-10);

  // the order -1 on the factorisation of +1: x_{-m} = D A_m^{-1} D b_{-m}
  AxisymmetricScatteringSetup minus_setup = plus.scattering->setup();
  minus_setup.azimuthal_order = -1;
  minus_setup.current = axisymmetric_gaussian_dipole(z0, Complex{1.0, 0.0}, AxisDipole::kTransverse,
                                                     sigma, minus_setup.omega, -1);
  const AxisymmetricScattering minus_problem(*plus.meridian, *plus.azimuthal, minus_setup);
  const Vector mirrored =
      minus_problem.mirrored(kept.solve(minus_problem.mirrored(minus_problem.load())));
  REQUIRE(relative(mirrored, block(minus_problem.solve())) < 1e-10);

  // validation
  REQUIRE_THROWS_AS(kept_problem.solution_from(Vector::Zero(3)), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(kept_problem.mirrored(Vector::Zero(3)), hpfem::InvalidArgument);
  const std::vector<Index> outside{mesh.num_cells()};
  REQUIRE_THROWS_AS(kept_problem.load(outside), hpfem::InvalidArgument);
}

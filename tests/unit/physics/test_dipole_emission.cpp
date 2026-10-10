// Source of the dipole-emitter cell problems (M17, ADR-0013): the Gaussian source integrates to
// i omega mu0 p exp(-sigma^2 beta^2 / 2) in the scaled components (f_x, f_y, -i f_z); argument
// checks; conical_source_power refuses a problem without a current; the responses to several
// moments on one kept factorisation equal the direct solves and the power matrix reproduces the
// delivered power.
#include <cmath>
#include <complex>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/physics/conical_scattering.hpp"
#include "hpfem/physics/dipole_emission.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::physics::ConicalVector;

TEST_CASE("conical Gaussian dipole: normalisation, scaled components and checks",
          "[physics][dipole]") {
  const Real sigma = 0.1;
  const Real omega = 2.0e9;
  const Real beta = 7.0;
  const Point<2> x0(0.13, -0.07);
  const ConicalVector p(Complex{1.0, 0.5}, -2.0, Complex{0.0, 0.3});
  const auto f = hpfem::physics::conical_gaussian_dipole(x0, p, sigma, omega, beta);
  // integrate over [-1, 1]^2 (the Gaussian is 1e-20 at the boundary)
  const hpfem::mesh::Mesh<2> mesh =
      hpfem::mesh::rectangle(40, 40, Point<2>(-1.0, -1.0), Point<2>(1.0, 1.0));
  const auto rule = hpfem::assembly::simplex_quadrature<2>(12);
  ConicalVector integral = ConicalVector::Zero();
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const auto geometry = hpfem::mesh::cell_geometry(mesh, c);
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const auto g = geometry->evaluate(rule.points[q]);
      integral += f(g.x) * (rule.weights[q] * std::abs(g.det));
    }
  }
  const Complex scale =
      hpfem::kI * omega * hpfem::constants::mu0 * std::exp(-0.5 * sigma * sigma * beta * beta);
  const ConicalVector expected(scale * p(0), scale * p(1), -hpfem::kI * scale * p(2));
  INFO("integral " << integral.transpose() << ", expected " << expected.transpose());
  REQUIRE((integral - expected).norm() < 1e-8 * expected.norm());
  // the peak sits at the position
  REQUIRE(f(x0).norm() > f(Point<2>(0.2, -0.07)).norm());
  REQUIRE_THROWS_AS(hpfem::physics::conical_gaussian_dipole(x0, p, 0.0, omega, beta),
                    hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(hpfem::physics::conical_gaussian_dipole(x0, p, sigma, -1.0, beta),
                    hpfem::InvalidArgument);
}

TEST_CASE("conical source power needs a total-field problem", "[physics][dipole]") {
  const hpfem::mesh::Mesh<2> mesh = hpfem::mesh::rectangle(2, 2);
  const hpfem::fespace::NedelecDofMap<2> nd(mesh, 1);
  const hpfem::fespace::DofMap<2> h1(mesh, 1);
  hpfem::physics::ConicalScatteringSetup setup;
  setup.omega = 1.0e9;
  setup.incident = [](const Point<2>&) { return ConicalVector(0.0, 0.0, 1.0); };
  const hpfem::physics::ConicalScattering problem(nd, h1, setup);
  hpfem::physics::ConicalSolution solution;
  solution.transverse = hpfem::Vector::Zero(nd.num_dofs());
  solution.longitudinal = hpfem::Vector::Zero(h1.num_dofs());
  REQUIRE_THROWS_AS(hpfem::physics::conical_source_power(problem, solution),
                    hpfem::InvalidArgument);
}

TEST_CASE("dipole responses on the kept factorisation and the power matrix", "[physics][dipole]") {
  // a lossy box (unique solution without PML), a Gaussian dipole off the mesh nodes
  const hpfem::mesh::Mesh<2> mesh = hpfem::mesh::rectangle(10, 10);
  const hpfem::fespace::NedelecDofMap<2> nd(mesh, 3);
  const hpfem::fespace::DofMap<2> h1(mesh, 3);
  const Real omega = 4.0 * hpfem::constants::c0;
  const Real beta = 0.7;
  const Real sigma = 0.08;
  const Point<2> x0(0.47, 0.52);
  hpfem::materials::Material lossy;
  lossy.eps_r = Complex{2.0, 0.4};
  const auto make_setup = [&](const ConicalVector& p, bool keep) {
    hpfem::physics::ConicalScatteringSetup setup;
    setup.omega = omega;
    setup.beta = beta;
    setup.materials = hpfem::materials::MaterialMap(lossy);
    setup.current = hpfem::physics::conical_gaussian_dipole(x0, p, sigma, omega, beta);
    setup.keep_factorisation = keep;
    return setup;
  };
  const ConicalVector p(Complex{1.0, 0.5}, -0.7, Complex{0.2, -0.3});
  // reference: the full solve with the moment p, its delivered power
  const hpfem::physics::ConicalScattering direct(nd, h1, make_setup(p, false));
  const auto reference = direct.solve();
  const Real power = hpfem::physics::conical_source_power(direct, reference);
  REQUIRE(power > 0.0);
  // the responses to e_x, e_y, e_z and p on the factorisation of the e_x problem
  const hpfem::physics::ConicalScattering kept(nd, h1,
                                               make_setup(ConicalVector(1.0, 0.0, 0.0), true));
  const auto solution = kept.solve();
  hpfem::Matrix moments = hpfem::Matrix::Zero(4, 3);
  moments.topRows(3) = hpfem::Matrix::Identity(3, 3);
  moments.row(3) = p.transpose();
  const auto responses =
      hpfem::physics::conical_dipole_responses(kept, solution, x0, sigma, moments);
  REQUIRE(responses.size() == 4);
  REQUIRE((responses[0].transverse - solution.transverse).norm() <
          1e-10 * solution.transverse.norm());
  REQUIRE((responses[3].transverse - reference.transverse).norm() <
          1e-10 * reference.transverse.norm());
  REQUIRE((responses[3].longitudinal - reference.longitudinal).norm() <
          1e-10 * reference.longitudinal.norm());
  // P(p) = Re(p^H A p) from the three unit responses
  const std::vector<hpfem::physics::ConicalSolution> unit(responses.begin(), responses.begin() + 3);
  const Eigen::Matrix3cd a = hpfem::physics::conical_dipole_power_matrix(kept, unit, x0, sigma);
  const Real from_matrix = (p.adjoint() * a * p).value().real();
  INFO("source power " << power << ", from the matrix " << from_matrix);
  REQUIRE(std::abs(from_matrix - power) < 1e-9 * power);
  // checks
  REQUIRE_THROWS_AS(hpfem::physics::conical_dipole_responses(direct, reference, x0, sigma, moments),
                    hpfem::InvalidArgument);  // no kept factorisation
  REQUIRE_THROWS_AS(hpfem::physics::conical_dipole_responses(kept, solution, x0, sigma,
                                                             hpfem::Matrix::Zero(1, 2)),
                    hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(hpfem::physics::conical_dipole_responses(kept, solution, x0, 0.0, moments),
                    hpfem::InvalidArgument);
  const std::vector<hpfem::physics::ConicalSolution> two(responses.begin(), responses.begin() + 2);
  REQUIRE_THROWS_AS(hpfem::physics::conical_dipole_power_matrix(kept, two, x0, sigma),
                    hpfem::InvalidArgument);
}

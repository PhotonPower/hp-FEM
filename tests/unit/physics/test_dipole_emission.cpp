// Source of the dipole-emitter cell problems (M17, ADR-0013): the Gaussian source integrates to
// i omega mu0 p exp(-sigma^2 beta^2 / 2) in the scaled components (f_x, f_y, -i f_z); argument
// checks; conical_source_power refuses a problem without a current.
#include <cmath>
#include <complex>

#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
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

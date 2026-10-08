// Material sensitivities by the adjoint solve (M12): the derivative dQ/d(eps_tag) of a point
// value of the scattered field against central finite differences of the discrete goal, for
// the in-plane solver (disc, PML) and the conical solver (disc at beta != 0; Bloch strip with
// a current), in the real and the imaginary direction; argument checks.
#include <cmath>
#include <complex>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/periodic.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/physics/conical_goal.hpp"
#include "hpfem/physics/conical_scattering.hpp"
#include "hpfem/physics/goal_oriented.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sensitivity.hpp"
#include "hpfem/physics/sources.hpp"
#include "hpfem/pml/pml.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::assembly::ComplexVector;
using hpfem::physics::ConicalVector;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr hpfem::mesh::Tag kDisc = 2;
constexpr Real kWavenumber = 4.0;
constexpr Real kDelta = 1e-5;

hpfem::physics::ScatteringSetup<2> in_plane_setup(Complex eps) {
  hpfem::physics::ScatteringSetup<2> setup;
  setup.omega = kWavenumber * hpfem::constants::c0;
  hpfem::materials::Material disc;
  disc.eps_r = eps;
  setup.materials.set(kDisc, disc);
  setup.incident =
      hpfem::physics::plane_wave<2>(ComplexVector<2>(0.0, 1.0), Point<2>(kWavenumber, 0.0));
  setup.formulation = hpfem::physics::Formulation::kScatteredField;
  setup.pml = hpfem::pml::PmlBox<2>::uniform(Point<2>(-1.0, -1.0), Point<2>(1.0, 1.0), 0.5,
                                             kWavenumber, 1.0, hpfem::pml::PmlProfile{2, 1e-8});
  setup.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  setup.condense = false;
  return setup;
}

/// Central finite difference of a complex-valued goal in the complex direction `direction`.
template <class Goal>
Complex finite_difference(const Goal& goal, Complex eps, Complex direction) {
  return (goal(eps + kDelta * direction) - goal(eps - kDelta * direction)) / (2 * kDelta);
}

}  // namespace

TEST_CASE("material sensitivity of the in-plane solver matches finite differences",
          "[physics][sensitivity]") {
  const hpfem::mesh::Mesh<2> mesh = hpfem::mesh::square_with_disc(2, 0.25, 1.0, 1.5, kDisc);
  const hpfem::fespace::NedelecDofMap<2> dofs(mesh, 2);
  const Complex eps{2.25, 0.3};
  const auto functional =
      hpfem::physics::point_value_functional<2>(Point<2>(0.6, 0.35), ComplexVector<2>(1.0, 0.5));
  const Vector q = functional(dofs);
  const auto goal = [&](Complex e) {
    const hpfem::physics::Scattering<2> problem(dofs, in_plane_setup(e));
    const auto solution = problem.solve();
    return (q.transpose() * solution.unknown).value();  // q^T e
  };
  const hpfem::physics::Scattering<2> problem(dofs, in_plane_setup(eps));
  const auto solution = problem.solve();
  const Vector z = hpfem::physics::adjoint_solution<2>(problem, q);
  REQUIRE(z.size() == dofs.num_dofs());
  const Complex derivative = hpfem::physics::material_sensitivity<2>(problem, solution, z, kDisc);
  REQUIRE(std::abs(derivative) > 0.0);
  // holomorphic: the derivative along i equals i times the derivative along 1
  const Complex fd_real = finite_difference(goal, eps, Complex{1.0, 0.0});
  const Complex fd_imag = finite_difference(goal, eps, Complex{0.0, 1.0});
  INFO("adjoint " << derivative << " FD real " << fd_real << " FD imag " << fd_imag);
  REQUIRE(std::abs(derivative - fd_real) < 1e-6 * std::abs(derivative));
  REQUIRE(std::abs(hpfem::kI * derivative - fd_imag) < 1e-6 * std::abs(derivative));
  // errors
  REQUIRE_THROWS_AS(hpfem::physics::material_sensitivity<2>(problem, solution, z, 77),
                    hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(hpfem::physics::adjoint_solution<2>(problem, Vector::Zero(3)),
                    hpfem::InvalidArgument);
}

TEST_CASE("material sensitivity of the conical solver matches finite differences",
          "[physics][sensitivity][conical]") {
  const hpfem::mesh::Mesh<2> mesh = hpfem::mesh::square_with_disc(2, 0.25, 1.0, 1.5, kDisc);
  const hpfem::fespace::NedelecDofMap<2> nd(mesh, 2);
  const hpfem::fespace::DofMap<2> h1(mesh, 2);
  const Complex eps{2.0, 0.2};
  const Real beta = 0.6 * kWavenumber;
  const Real kt = std::sqrt(kWavenumber * kWavenumber - beta * beta);
  const auto make_setup = [&](Complex e) {
    hpfem::physics::ConicalScatteringSetup setup;
    setup.omega = kWavenumber * hpfem::constants::c0;
    setup.beta = beta;
    hpfem::materials::Material disc;
    disc.eps_r = e;
    setup.materials.set(kDisc, disc);
    const Point<3> k(kt, 0.0, beta);
    setup.incident = hpfem::physics::conical_plane_wave(
        hpfem::physics::conical_polarisation(k, Point<3>(1.0, 0.0, 0.0),
                                             hpfem::physics::Polarisation::kP),
        k);
    setup.pml = hpfem::pml::PmlBox<2>::uniform(Point<2>(-1.0, -1.0), Point<2>(1.0, 1.0), 0.5,
                                               kWavenumber, 1.0, hpfem::pml::PmlProfile{2, 1e-8});
    setup.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
    return setup;
  };
  const auto functional = hpfem::physics::conical_point_functional(
      Point<2>(0.6, 0.35), ConicalVector(1.0, Complex{0.0, 0.5}, 0.7));
  // no structured binding: clang's OpenMP mode cannot capture one in the goal lambda
  const auto q_pair = functional(nd, h1);
  const Vector& q_e = q_pair.first;
  const Vector& q_v = q_pair.second;
  const auto goal = [&](Complex e) {
    const hpfem::physics::ConicalScattering problem(nd, h1, make_setup(e));
    const auto s = problem.solve();
    return (q_e.transpose() * s.transverse).value() + (q_v.transpose() * s.longitudinal).value();
  };
  const hpfem::physics::ConicalScattering problem(nd, h1, make_setup(eps));
  const auto solution = problem.solve();
  const auto z = hpfem::physics::conical_adjoint_solution(problem, q_e, q_v);
  const Complex derivative =
      hpfem::physics::conical_material_sensitivity(problem, solution, z, kDisc);
  const Complex fd_real = finite_difference(goal, eps, Complex{1.0, 0.0});
  const Complex fd_imag = finite_difference(goal, eps, Complex{0.0, 1.0});
  INFO("adjoint " << derivative << " FD real " << fd_real << " FD imag " << fd_imag);
  REQUIRE(std::abs(derivative - fd_real) < 1e-6 * std::abs(derivative));
  REQUIRE(std::abs(hpfem::kI * derivative - fd_imag) < 1e-6 * std::abs(derivative));
  REQUIRE_THROWS_AS(hpfem::physics::conical_material_sensitivity(problem, solution, z, 77),
                    hpfem::InvalidArgument);
}

TEST_CASE("material sensitivity with Bloch constraints and a current source",
          "[physics][sensitivity][conical][periodic]") {
  // strip [0, 1] x [0, 0.75], PEC top and bottom, Bloch phase along x, a z current in the
  // lower half, a lossy block (tag 5) in the middle: total-field formulation (no incident)
  hpfem::mesh::Mesh<2> mesh = hpfem::mesh::rectangle(4, 3, Point<2>(0.0, 0.0), Point<2>(1.0, 0.75));
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const Point<2> x = hpfem::mesh::affine_map(mesh, c).centroid();
    if (x(0) > 0.25 && x(0) < 0.75 && x(1) > 0.25 && x(1) < 0.5) mesh.set_cell_tag(c, 5);
  }
  const hpfem::fespace::NedelecDofMap<2> nd(mesh, 3);
  const hpfem::fespace::DofMap<2> h1(mesh, 3);
  const Complex eps{3.0, 0.5};
  const auto make_setup = [&](Complex e) {
    hpfem::physics::ConicalScatteringSetup setup;
    setup.omega = 4.0 * hpfem::constants::c0;
    setup.beta = 0.4;
    hpfem::materials::Material block;
    block.eps_r = e;
    setup.materials.set(5, block);
    setup.pec_tags = {box_tag::kYMin, box_tag::kYMax};
    setup.periodic = {hpfem::assembly::PeriodicPair<2>{
        box_tag::kXMin, box_tag::kXMax, Point<2>(1.0, 0.0), std::exp(hpfem::kI * 1.1)}};
    setup.current = [](const Point<2>& x) {
      return ConicalVector(0.0, 0.3 * x(0),
                           std::exp(hpfem::kI * 1.1 * x(0)) * x(1) * (0.75 - x(1)));
    };
    return setup;
  };
  const auto functional = hpfem::physics::conical_point_functional(
      Point<2>(0.9, 0.6), ConicalVector(0.2, 1.0, Complex{0.0, -0.4}));
  // no structured binding: clang's OpenMP mode cannot capture one in the goal lambda
  const auto q_pair = functional(nd, h1);
  const Vector& q_e = q_pair.first;
  const Vector& q_v = q_pair.second;
  const auto goal = [&](Complex e) {
    const hpfem::physics::ConicalScattering problem(nd, h1, make_setup(e));
    const auto s = problem.solve();
    return (q_e.transpose() * s.transverse).value() + (q_v.transpose() * s.longitudinal).value();
  };
  const hpfem::physics::ConicalScattering problem(nd, h1, make_setup(eps));
  const auto solution = problem.solve();
  const auto z = hpfem::physics::conical_adjoint_solution(problem, q_e, q_v);
  const Complex derivative = hpfem::physics::conical_material_sensitivity(problem, solution, z, 5);
  const Complex fd_real = finite_difference(goal, eps, Complex{1.0, 0.0});
  INFO("adjoint " << derivative << " FD " << fd_real);
  REQUIRE(std::abs(derivative - fd_real) < 1e-6 * std::abs(derivative));
}

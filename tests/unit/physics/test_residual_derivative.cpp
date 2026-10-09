// Residual derivatives and the direct (tangent) mode of the sensitivities (M16 S1, ADR-0012):
// for a material and a shape parameter (disc permittivity and radius, second-order mesh) the
// direct mode q^T A^{-1} r (+ the functional term of a shape) equals the adjoint mode
// (material_sensitivity, shape_derivative), z^T r_V equals the node-gradient pairing of
// ADR-0011, and a 2 x 2 Jacobian agrees in both modes; in-plane and conical solver; only
// the cells whose nodes move contribute to r_V; argument checks.
#include <cmath>
#include <complex>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/physics/conical_goal.hpp"
#include "hpfem/physics/conical_scattering.hpp"
#include "hpfem/physics/goal_oriented.hpp"
#include "hpfem/physics/kept_factorisation.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sensitivity.hpp"
#include "hpfem/physics/shape_sensitivity.hpp"
#include "hpfem/physics/sources.hpp"
#include "hpfem/pml/pml.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::Matrix;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::assembly::ComplexVector;
using hpfem::physics::ConicalVector;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr hpfem::mesh::Tag kDisc = 2;
constexpr Real kWavenumber = 4.0;

/// Relative difference, guarded against a vanishing reference.
Real relative(Complex a, Complex b) {
  return std::abs(a - b) / std::max(std::abs(b), 1e-300);
}

}  // namespace

TEST_CASE("direct mode of the in-plane solver: material and shape against the adjoint",
          "[physics][sensitivity][direct]") {
  const hpfem::mesh::Mesh<2> mesh = hpfem::mesh::square_with_disc(2, 0.25, 1.0, 1.5, kDisc);
  REQUIRE(mesh.geometry_order() == 2);
  const hpfem::fespace::NedelecDofMap<2> dofs(mesh, 3);
  hpfem::physics::ScatteringSetup<2> setup;
  setup.omega = kWavenumber * hpfem::constants::c0;
  hpfem::materials::Material disc;
  disc.eps_r = Complex{2.25, 0.3};
  setup.materials.set(kDisc, disc);
  setup.incident =
      hpfem::physics::plane_wave<2>(ComplexVector<2>(0.0, 1.0), Point<2>(kWavenumber, 0.0));
  setup.formulation = hpfem::physics::Formulation::kScatteredField;
  setup.pml = hpfem::pml::PmlBox<2>::uniform(Point<2>(-1.0, -1.0), Point<2>(1.0, 1.0), 0.5,
                                             kWavenumber, 1.0, hpfem::pml::PmlProfile{2, 1e-8});
  setup.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  setup.keep_factorisation = true;  // condensation on (default): part of the kept chain
  const hpfem::physics::Scattering<2> problem(dofs, setup);
  const auto solution = problem.solve();
  const auto& kept = *solution.factorisation;
  const auto functional =
      hpfem::physics::point_value_functional<2>(Point<2>(0.6, 0.35), ComplexVector<2>(1.0, 0.5));
  const auto second =
      hpfem::physics::point_value_functional<2>(Point<2>(-0.5, 0.1), ComplexVector<2>(0.3, 1.0));
  const Vector q = functional(dofs);
  const Vector z = hpfem::physics::adjoint_solution<2>(problem, solution, q);

  // material: q^T A^{-1} r_eps == z^T r_eps == material_sensitivity
  const Vector r_eps = hpfem::physics::material_residual_derivative<2>(problem, solution, kDisc);
  const Complex direct_eps = (q.transpose() * kept.solve(r_eps)).value();
  const Complex adjoint_eps = hpfem::physics::material_sensitivity<2>(problem, solution, z, kDisc);
  INFO("material: direct " << direct_eps << ", adjoint " << adjoint_eps);
  REQUIRE(std::abs(adjoint_eps) > 0.0);
  REQUIRE(relative(direct_eps, adjoint_eps) < 1e-9);

  // shape (disc radius): only the cells at the circle move
  const auto velocity = hpfem::physics::region_normal_velocity<2>(mesh, kDisc);
  const Vector r_v = hpfem::physics::shape_residual_derivative<2>(problem, solution, velocity);
  Index nonzero = 0;
  for (Index i = 0; i < r_v.size(); ++i) nonzero += r_v(i) != Complex{0.0, 0.0} ? 1 : 0;
  REQUIRE(nonzero > 0);
  REQUIRE(nonzero < dofs.num_dofs() / 2);
  const Vector dq = hpfem::physics::functional_shape_derivative<2>(dofs, functional, velocity);
  const Complex direct_v =
      (q.transpose() * kept.solve(r_v)).value() + (dq.transpose() * solution.unknown).value();
  const Complex adjoint_v =
      hpfem::physics::shape_derivative<2>(problem, solution, functional, velocity);
  INFO("shape: direct " << direct_v << ", adjoint " << adjoint_v);
  REQUIRE(std::abs(adjoint_v) > 0.0);
  // two different central differences (directional per cell, per node coordinate): O(step^2)
  REQUIRE(relative(direct_v, adjoint_v) < 1e-6);
  const auto gradient = hpfem::physics::shape_gradient<2>(problem, solution, z);
  REQUIRE(relative((z.transpose() * r_v).value(),
                   hpfem::physics::shape_sensitivity(gradient, velocity)) < 1e-6);

  // a 2 x 2 Jacobian (observables x parameters) in both modes
  Matrix observables(dofs.num_dofs(), 2);
  observables.col(0) = q;
  observables.col(1) = second(dofs);
  Matrix residuals(dofs.num_dofs(), 2);
  residuals.col(0) = r_eps;
  residuals.col(1) = r_v;
  const Matrix direct = observables.transpose() * kept.solve_many(residuals);
  const Matrix adjoint = kept.solve_adjoint_many(observables).transpose() * residuals;
  REQUIRE((direct - adjoint).norm() < 1e-9 * adjoint.norm());

  // errors
  REQUIRE_THROWS_AS(hpfem::physics::shape_residual_derivative<2>(
                        problem, solution, hpfem::physics::NodeField::Zero(3, 2)),
                    hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(hpfem::physics::shape_residual_derivative<2>(problem, solution, velocity, 0.0),
                    hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(hpfem::physics::material_residual_derivative<2>(problem, solution, 77),
                    hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(
      hpfem::physics::functional_shape_derivative<2>(dofs, functional, velocity, -1.0),
      hpfem::InvalidArgument);
}

TEST_CASE("direct mode of the conical solver: material and shape against the adjoint",
          "[physics][sensitivity][direct][conical]") {
  const hpfem::mesh::Mesh<2> mesh = hpfem::mesh::square_with_disc(2, 0.25, 1.0, 1.5, kDisc);
  const hpfem::fespace::NedelecDofMap<2> nd(mesh, 2);
  const hpfem::fespace::DofMap<2> h1(mesh, 2);
  const Real beta = 0.6 * kWavenumber;
  const Real kt = std::sqrt(kWavenumber * kWavenumber - beta * beta);
  hpfem::physics::ConicalScatteringSetup setup;
  setup.omega = kWavenumber * hpfem::constants::c0;
  setup.beta = beta;
  hpfem::materials::Material disc;
  disc.eps_r = Complex{2.0, 0.2};
  setup.materials.set(kDisc, disc);
  const Point<3> k(kt, 0.0, beta);
  setup.incident = hpfem::physics::conical_plane_wave(
      hpfem::physics::conical_polarisation(k, Point<3>(1.0, 0.0, 0.0),
                                           hpfem::physics::Polarisation::kP),
      k);
  setup.pml = hpfem::pml::PmlBox<2>::uniform(Point<2>(-1.0, -1.0), Point<2>(1.0, 1.0), 0.5,
                                             kWavenumber, 1.0, hpfem::pml::PmlProfile{2, 1e-8});
  setup.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  setup.keep_factorisation = true;
  const hpfem::physics::ConicalScattering problem(nd, h1, setup);
  const auto solution = problem.solve();
  const auto& kept = *solution.factorisation;
  const auto functional = hpfem::physics::conical_point_functional(
      Point<2>(0.6, 0.35), ConicalVector(1.0, Complex{0.0, 0.5}, 0.7));
  const auto q_pair = functional(nd, h1);
  Vector q(kept.num_dofs());
  q << q_pair.first, q_pair.second;
  Vector e(kept.num_dofs());
  e << solution.transverse, solution.longitudinal;
  const auto z =
      hpfem::physics::conical_adjoint_solution(problem, solution, q_pair.first, q_pair.second);

  const Vector r_eps =
      hpfem::physics::conical_material_residual_derivative(problem, solution, kDisc);
  const Complex direct_eps = (q.transpose() * kept.solve(r_eps)).value();
  const Complex adjoint_eps =
      hpfem::physics::conical_material_sensitivity(problem, solution, z, kDisc);
  INFO("material: direct " << direct_eps << ", adjoint " << adjoint_eps);
  REQUIRE(std::abs(adjoint_eps) > 0.0);
  REQUIRE(relative(direct_eps, adjoint_eps) < 1e-9);

  const auto velocity = hpfem::physics::region_normal_velocity<2>(mesh, kDisc);
  const Vector r_v = hpfem::physics::conical_shape_residual_derivative(problem, solution, velocity);
  const Vector dq =
      hpfem::physics::conical_functional_shape_derivative(nd, h1, functional, velocity);
  const Complex direct_v = (q.transpose() * kept.solve(r_v)).value() + (dq.transpose() * e).value();
  const Complex adjoint_v =
      hpfem::physics::conical_shape_derivative(problem, solution, functional, velocity);
  INFO("shape: direct " << direct_v << ", adjoint " << adjoint_v);
  REQUIRE(std::abs(adjoint_v) > 0.0);
  REQUIRE(relative(direct_v, adjoint_v) < 1e-6);
  REQUIRE_THROWS_AS(hpfem::physics::conical_shape_residual_derivative(
                        problem, solution, hpfem::physics::NodeField::Zero(3, 2)),
                    hpfem::InvalidArgument);
}

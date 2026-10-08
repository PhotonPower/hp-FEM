// Shape derivatives by the discrete adjoint (M12, ADR-0011): the derivative of a point value
// with respect to the radius of a disc (in-plane and conical solver, second-order mesh) and
// of a ball (3D) by shape_gradient / region_normal_velocity against central finite
// differences of the solve on meshes whose nodes were moved by move_nodes; argument checks.
// The point of the functional lies in cells that do not deform (the functional vector q is
// taken as fixed by the adjoint formula).
#include <cmath>
#include <complex>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/physics/conical_goal.hpp"
#include "hpfem/physics/conical_scattering.hpp"
#include "hpfem/physics/goal_oriented.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sensitivity.hpp"
#include "hpfem/physics/shape_sensitivity.hpp"
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
constexpr Real kDelta = 1e-6;
// the finite differences of the solve carry the roundoff of two solves divided by 2 delta
// (about 1e-5 relative) and the second-order term of the discrete goal, which is unusually
// large on the coarse disc mesh (cells of aspect 10 next to the circle: a node motion of
// 1e-4 changes their matrices by a few per cent); the adjoint derivative is the delta -> 0
// limit, which the goal FD approaches as 1e-5 -> 4e-4, 1e-6 -> 3e-6, 1e-7 -> 1e-7 (2D)
constexpr Real kTolerance = 2e-4;

template <int Dim>
hpfem::physics::ScatteringSetup<Dim> in_plane_setup() {
  hpfem::physics::ScatteringSetup<Dim> setup;
  setup.omega = kWavenumber * hpfem::constants::c0;
  setup.materials.set(kDisc, hpfem::materials::Material::dielectric(1.5));
  if constexpr (Dim == 2) {
    setup.incident =
        hpfem::physics::plane_wave<2>(ComplexVector<2>(0.0, 1.0), Point<2>(kWavenumber, 0.0));
    setup.pml = hpfem::pml::PmlBox<2>::uniform(Point<2>(-1.0, -1.0), Point<2>(1.0, 1.0), 0.5,
                                               kWavenumber, 1.0, hpfem::pml::PmlProfile{2, 1e-8});
    setup.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  } else {
    setup.incident = hpfem::physics::plane_wave<3>(ComplexVector<3>(0.0, 0.0, 1.0),
                                                   Point<3>(kWavenumber, 0.0, 0.0));
    setup.pml =
        hpfem::pml::PmlBox<3>::uniform(Point<3>(-1.0, -1.0, -1.0), Point<3>(1.0, 1.0, 1.0), 0.5,
                                       kWavenumber, 1.0, hpfem::pml::PmlProfile{2, 1e-8});
    setup.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin,
                      box_tag::kYMax, box_tag::kZMin, box_tag::kZMax};
  }
  setup.formulation = hpfem::physics::Formulation::kScatteredField;
  setup.condense = false;
  return setup;
}

}  // namespace

TEST_CASE("shape sensitivity of the in-plane solver: disc radius against finite differences",
          "[physics][sensitivity][shape]") {
  const hpfem::mesh::Mesh<2> mesh = hpfem::mesh::square_with_disc(2, 0.25, 1.0, 1.5, kDisc);
  REQUIRE(mesh.geometry_order() == 2);  // curved disc: vertices and edge nodes move
  const auto velocity = hpfem::physics::region_normal_velocity<2>(mesh, kDisc);
  REQUIRE(velocity.rows() == hpfem::physics::num_geometry_nodes(mesh));
  // boundary vertices of the disc move radially, interior ones not at all
  Index moving = 0;
  for (Index v = 0; v < mesh.num_vertices(); ++v) {
    const Real norm = velocity.row(v).norm();
    if (norm == 0.0) continue;
    ++moving;
    REQUIRE(norm == Approx(1.0));
    REQUIRE(std::abs(mesh.vertex(v).norm() - 0.25) < 1e-9);
    REQUIRE(velocity.row(v).transpose().dot(mesh.vertex(v).normalized()) > 0.9);
  }
  REQUIRE(moving > 4);
  const auto functional =
      hpfem::physics::point_value_functional<2>(Point<2>(0.8, 0.75), ComplexVector<2>(1.0, 0.4));
  const auto goal = [&](Real t) {
    hpfem::mesh::Mesh<2> moved = mesh;
    hpfem::physics::move_nodes<2>(moved, velocity, t);
    const hpfem::fespace::NedelecDofMap<2> dofs(moved, 2);
    const hpfem::physics::Scattering<2> problem(dofs, in_plane_setup<2>());
    const auto solution = problem.solve();
    return (functional(dofs).transpose() * solution.unknown).value();
  };
  const hpfem::fespace::NedelecDofMap<2> dofs(mesh, 2);
  const hpfem::physics::Scattering<2> problem(dofs, in_plane_setup<2>());
  const auto solution = problem.solve();
  const Vector z = hpfem::physics::adjoint_solution<2>(problem, functional(dofs));
  const auto gradient = hpfem::physics::shape_gradient<2>(problem, solution, z);
  REQUIRE(gradient.rows() == velocity.rows());
  const Complex matrix_part = hpfem::physics::shape_sensitivity(gradient, velocity);
  const Complex derivative =
      hpfem::physics::shape_derivative<2>(problem, solution, functional, velocity);
  const Complex fd = (goal(kDelta) - goal(-kDelta)) / (2 * kDelta);
  INFO("adjoint " << derivative << " (matrix part " << matrix_part << ") FD " << fd);
  REQUIRE(std::abs(derivative) > 0.0);
  REQUIRE(std::abs(derivative - fd) < kTolerance * std::abs(fd));
  // errors
  REQUIRE_THROWS_AS(hpfem::physics::region_normal_velocity<2>(mesh, 77), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(hpfem::physics::shape_gradient<2>(problem, solution, Vector::Zero(3)),
                    hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(hpfem::physics::shape_sensitivity(gradient, velocity.topRows(2)),
                    hpfem::InvalidArgument);
  hpfem::mesh::Mesh<2> copy = mesh;
  REQUIRE_THROWS_AS(hpfem::physics::move_nodes<2>(copy, velocity.topRows(2), 1.0),
                    hpfem::InvalidArgument);
}

TEST_CASE("shape sensitivity of the conical solver: disc radius at beta != 0",
          "[physics][sensitivity][shape][conical]") {
  const hpfem::mesh::Mesh<2> mesh = hpfem::mesh::square_with_disc(2, 0.25, 1.0, 1.5, kDisc);
  const auto velocity = hpfem::physics::region_normal_velocity<2>(mesh, kDisc);
  const Real beta = 0.5 * kWavenumber;
  const Real kt = std::sqrt(kWavenumber * kWavenumber - beta * beta);
  const auto make_setup = [&]() {
    hpfem::physics::ConicalScatteringSetup setup;
    setup.omega = kWavenumber * hpfem::constants::c0;
    setup.beta = beta;
    hpfem::materials::Material disc;
    disc.eps_r = Complex{2.0, 0.1};
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
      Point<2>(0.8, 0.75), ConicalVector(1.0, Complex{0.0, 0.4}, 0.6));
  const auto goal = [&](Real t) {
    hpfem::mesh::Mesh<2> moved = mesh;
    hpfem::physics::move_nodes<2>(moved, velocity, t);
    const hpfem::fespace::NedelecDofMap<2> nd(moved, 2);
    const hpfem::fespace::DofMap<2> h1(moved, 2);
    const hpfem::physics::ConicalScattering problem(nd, h1, make_setup());
    const auto s = problem.solve();
    const auto [q_e, q_v] = functional(nd, h1);
    return (q_e.transpose() * s.transverse).value() + (q_v.transpose() * s.longitudinal).value();
  };
  const hpfem::fespace::NedelecDofMap<2> nd(mesh, 2);
  const hpfem::fespace::DofMap<2> h1(mesh, 2);
  const hpfem::physics::ConicalScattering problem(nd, h1, make_setup());
  const auto solution = problem.solve();
  const auto [q_e, q_v] = functional(nd, h1);
  const auto z = hpfem::physics::conical_adjoint_solution(problem, q_e, q_v);
  const auto gradient = hpfem::physics::conical_shape_gradient(problem, solution, z);
  const Complex matrix_part = hpfem::physics::shape_sensitivity(gradient, velocity);
  const Complex derivative =
      hpfem::physics::conical_shape_derivative(problem, solution, functional, velocity);
  const Complex fd = (goal(kDelta) - goal(-kDelta)) / (2 * kDelta);
  INFO("adjoint " << derivative << " (matrix part " << matrix_part << ") FD " << fd);
  REQUIRE(std::abs(derivative - fd) < kTolerance * std::abs(fd));
}

TEST_CASE("shape sensitivity in 3D: ball radius against finite differences",
          "[physics][sensitivity][shape][3d]") {
  const hpfem::mesh::Mesh<3> mesh = hpfem::mesh::box_with_ball(1, 0.25, 1.0, 1.5, kDisc);
  const auto velocity = hpfem::physics::region_normal_velocity<3>(mesh, kDisc);
  const auto functional = hpfem::physics::point_value_functional<3>(
      Point<3>(0.8, 0.7, 0.6), ComplexVector<3>(0.3, 1.0, 0.5));
  const auto goal = [&](Real t) {
    hpfem::mesh::Mesh<3> moved = mesh;
    hpfem::physics::move_nodes<3>(moved, velocity, t);
    const hpfem::fespace::NedelecDofMap<3> dofs(moved, 1);
    const hpfem::physics::Scattering<3> problem(dofs, in_plane_setup<3>());
    return (functional(dofs).transpose() * problem.solve().unknown).value();
  };
  const hpfem::fespace::NedelecDofMap<3> dofs(mesh, 1);
  const hpfem::physics::Scattering<3> problem(dofs, in_plane_setup<3>());
  const auto solution = problem.solve();
  const Complex derivative =
      hpfem::physics::shape_derivative<3>(problem, solution, functional, velocity);
  const Complex fd = (goal(kDelta) - goal(-kDelta)) / (2 * kDelta);
  INFO("adjoint " << derivative << " FD " << fd);
  REQUIRE(std::abs(derivative - fd) < kTolerance * std::abs(fd));
}

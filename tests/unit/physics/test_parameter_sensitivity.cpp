// Frequency and angle derivatives of the conical solver (M16 S1, ADR-0012 §4): a parameter of
// the whole setup changes the frequency, beta, the source and the Bloch phases at once; the
// tangent on the kept factorisation (with the derivative of the constraints) matches central
// differences of full solves for the coefficients and for a goal, on the vector and on the
// scalar E_z path; the full residual vanishes in the test space; argument checks.
#include <cmath>
#include <complex>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/periodic.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/physics/conical_goal.hpp"
#include "hpfem/physics/conical_scattering.hpp"
#include "hpfem/physics/parameter_sensitivity.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::physics::ConicalScattering;
using hpfem::physics::ConicalScatteringSetup;
using hpfem::physics::ConicalSolution;
using hpfem::physics::ConicalVector;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr hpfem::mesh::Tag kBlock = 5;

hpfem::mesh::Mesh<2> strip() {
  hpfem::mesh::Mesh<2> mesh = hpfem::mesh::rectangle(4, 3, Point<2>(0.0, 0.0), Point<2>(1.0, 0.75));
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const Point<2> x = hpfem::mesh::affine_map(mesh, c).centroid();
    if (x(0) > 0.25 && x(0) < 0.75 && x(1) > 0.25 && x(1) < 0.5) mesh.set_cell_tag(c, kBlock);
  }
  return mesh;
}

/// The Bloch strip of the kept-factorisation test as a function of one parameter t: the
/// Bloch wavenumber, the frequency, beta and the quasi-periodic source all move with t.
ConicalScatteringSetup strip_setup(Real t, bool scalar) {
  const Real kx = 1.1 + 0.7 * t;
  ConicalScatteringSetup setup;
  setup.omega = (4.0 + 0.5 * t) * hpfem::constants::c0;
  setup.beta = scalar ? 0.0 : 0.4 + 0.3 * t;
  hpfem::materials::Material block;
  block.eps_r = Complex{3.0, 0.5};
  setup.materials.set(kBlock, block);
  setup.pec_tags = {box_tag::kYMin, box_tag::kYMax};
  setup.periodic = {hpfem::assembly::PeriodicPair<2>{box_tag::kXMin, box_tag::kXMax,
                                                     Point<2>(1.0, 0.0), std::exp(hpfem::kI * kx)}};
  setup.current = [kx, scalar](const Point<2>& x) {
    const Complex bloch = std::exp(hpfem::kI * kx * x(0));
    return ConicalVector(0.0, scalar ? 0.0 : 0.3 * x(1) * bloch, bloch * x(1) * (0.75 - x(1)));
  };
  setup.scalar_ez = scalar;
  return setup;
}

Vector stacked(const ConicalSolution& s) {
  Vector e(s.transverse.size() + s.longitudinal.size());
  e << s.transverse, s.longitudinal;
  return e;
}

void check_tangent(bool scalar) {
  const hpfem::mesh::Mesh<2> mesh = strip();
  const hpfem::fespace::NedelecDofMap<2> nd(mesh, 3);
  const hpfem::fespace::DofMap<2> h1(mesh, 3);
  constexpr Real kStep = 1e-5;       // of the tangent's differences
  constexpr Real kReference = 1e-3;  // of the reference solves
  ConicalScatteringSetup setup = strip_setup(0.0, scalar);
  setup.keep_factorisation = true;
  const ConicalScattering problem(nd, h1, setup);
  const ConicalSolution solution = problem.solve();
  REQUIRE(problem.system_constraints().has_value());
  const ConicalScattering minus(nd, h1, strip_setup(-kStep, scalar));
  const ConicalScattering plus(nd, h1, strip_setup(kStep, scalar));
  const auto tangent =
      hpfem::physics::conical_parameter_tangent(problem, solution, minus, plus, kStep);
  Vector de(tangent.transverse.size() + tangent.longitudinal.size());
  de << tangent.transverse, tangent.longitudinal;

  // central difference of full solves (each with its own Bloch phases)
  const Vector e_lo = stacked(ConicalScattering(nd, h1, strip_setup(-kReference, scalar)).solve());
  const Vector e_hi = stacked(ConicalScattering(nd, h1, strip_setup(kReference, scalar)).solve());
  const Vector reference = (e_hi - e_lo) / (2.0 * kReference);
  INFO("|de| = " << de.norm() << ", |de - fd| = " << (de - reference).norm());
  REQUIRE(reference.norm() > 1e-3 * stacked(solution).norm());
  REQUIRE((de - reference).norm() < 1e-5 * reference.norm());

  // a goal Q = q^T e: dQ/dt = q^T de (no explicit dependence)
  const auto functional = hpfem::physics::conical_point_functional(
      Point<2>(0.9, 0.6), ConicalVector(0.2, 1.0, Complex{0.0, -0.4}));
  const auto [q_e, q_v] = functional(nd, h1);
  Vector q(q_e.size() + q_v.size());
  q << q_e, q_v;
  const Complex dq = (q.transpose() * de).value();
  const Complex dq_fd = (q.transpose() * reference).value();
  INFO("dQ/dt = " << dq << ", difference quotient " << dq_fd);
  REQUIRE(std::abs(dq_fd) > 0.0);
  REQUIRE(std::abs(dq - dq_fd) < 1e-5 * std::abs(dq_fd));

  // without the derivative of the constraints the tangent is wrong: the term matters
  const ConicalScattering frozen_minus(nd, h1, [&] {
    ConicalScatteringSetup s = strip_setup(-kStep, scalar);
    s.periodic = setup.periodic;
    return s;
  }());
  const ConicalScattering frozen_plus(nd, h1, [&] {
    ConicalScatteringSetup s = strip_setup(kStep, scalar);
    s.periodic = setup.periodic;
    return s;
  }());
  const auto frozen = hpfem::physics::conical_parameter_tangent(problem, solution, frozen_minus,
                                                                frozen_plus, kStep);
  Vector de_frozen(de.size());
  de_frozen << frozen.transverse, frozen.longitudinal;
  REQUIRE((de_frozen - reference).norm() > 1e-2 * reference.norm());
}

}  // namespace

TEST_CASE("conical parameter tangent: frequency, beta and Bloch phase against full solves",
          "[physics][sensitivity][conical][periodic]") {
  check_tangent(false);
}

TEST_CASE("conical parameter tangent on the scalar E_z path",
          "[physics][sensitivity][conical][periodic]") {
  check_tangent(true);
}

TEST_CASE("conical residual and transported solution", "[physics][sensitivity][conical]") {
  const hpfem::mesh::Mesh<2> mesh = strip();
  const hpfem::fespace::NedelecDofMap<2> nd(mesh, 2);
  const hpfem::fespace::DofMap<2> h1(mesh, 2);
  const ConicalScattering problem(nd, h1, strip_setup(0.0, false));
  const ConicalSolution solution = problem.solve();
  const Vector rho =
      hpfem::physics::conical_residual(problem, solution.transverse, solution.longitudinal);
  const Vector load = hpfem::physics::conical_residual(problem, Vector::Zero(nd.num_dofs()),
                                                       Vector::Zero(h1.num_dofs()));
  const auto& dofs = problem.system_dofs();
  Vector rho_system(static_cast<Index>(dofs.size()));
  Vector load_system(static_cast<Index>(dofs.size()));
  for (std::size_t j = 0; j < dofs.size(); ++j) {
    rho_system(static_cast<Index>(j)) = rho(dofs[j]);
    load_system(static_cast<Index>(j)) = load(dofs[j]);
  }
  const hpfem::SparseMatrix p = problem.system_constraints()->prolongation();
  const Vector tested = p.adjoint() * rho_system;
  INFO("|P^H rho| = " << tested.norm() << ", |rho| = " << rho_system.norm());
  REQUIRE(tested.norm() < 1e-10 * load_system.norm());
  REQUIRE(rho_system.norm() > 1e-6 * load_system.norm());  // only the tested residual vanishes

  // transported to the same problem: unchanged; to another phase: the masters stay
  const ConicalSolution same = hpfem::physics::conical_transported_solution(problem, solution);
  REQUIRE((stacked(same) - stacked(solution)).norm() < 1e-14 * stacked(solution).norm());
  const ConicalScattering other(nd, h1, strip_setup(0.2, false));
  const ConicalSolution moved = hpfem::physics::conical_transported_solution(other, solution);
  REQUIRE(moved.beta == other.beta());
  const auto& constraints = *problem.system_constraints();
  const Vector before = stacked(solution);
  const Vector after = stacked(moved);
  Index changed = 0;
  for (std::size_t j = 0; j < dofs.size(); ++j) {
    const Index i = static_cast<Index>(j);
    const Complex a = before(dofs[j]);
    const Complex b = after(dofs[j]);
    if (!constraints.is_constrained(i)) {
      REQUIRE(a == b);
    } else if (std::abs(a - b) > 1e-12 * std::abs(a)) {
      ++changed;
    }
  }
  REQUIRE(changed > 0);
}

TEST_CASE("conical parameter tangent: argument checks", "[physics][sensitivity][conical]") {
  const hpfem::mesh::Mesh<2> mesh = strip();
  const hpfem::fespace::NedelecDofMap<2> nd(mesh, 2);
  const hpfem::fespace::DofMap<2> h1(mesh, 2);
  ConicalScatteringSetup setup = strip_setup(0.0, false);
  const ConicalScattering unkept(nd, h1, setup);
  const ConicalSolution plain = unkept.solve();
  const ConicalScattering minus(nd, h1, strip_setup(-1e-5, false));
  const ConicalScattering plus(nd, h1, strip_setup(1e-5, false));
  REQUIRE_THROWS_AS(hpfem::physics::conical_parameter_tangent(unkept, plain, minus, plus, 1e-5),
                    hpfem::InvalidArgument);
  setup.keep_factorisation = true;
  const ConicalScattering problem(nd, h1, setup);
  const ConicalSolution solution = problem.solve();
  REQUIRE_THROWS_AS(hpfem::physics::conical_parameter_tangent(problem, solution, minus, plus, 0.0),
                    hpfem::InvalidArgument);
  // another PEC choice factorises other unknowns
  ConicalScatteringSetup open = strip_setup(1e-5, false);
  open.pec_tags = {box_tag::kYMin};
  const ConicalScattering other(nd, h1, open);
  REQUIRE_THROWS_AS(
      hpfem::physics::conical_parameter_tangent(problem, solution, minus, other, 1e-5),
      hpfem::InvalidArgument);
  // without Bloch directions the constraints differ
  ConicalScatteringSetup closed = strip_setup(1e-5, false);
  closed.periodic.clear();
  closed.pec_tags = {box_tag::kYMin, box_tag::kYMax};
  const ConicalScattering unconstrained(nd, h1, closed);
  REQUIRE_THROWS_AS(
      hpfem::physics::conical_parameter_tangent(problem, solution, minus, unconstrained, 1e-5),
      hpfem::InvalidArgument);
  // vectors of another size
  ConicalSolution wrong = solution;
  wrong.transverse = Vector::Zero(3);
  REQUIRE_THROWS_AS(hpfem::physics::conical_parameter_tangent(problem, wrong, minus, plus, 1e-5),
                    hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(
      hpfem::physics::conical_residual(problem, Vector::Zero(3), solution.longitudinal),
      hpfem::InvalidArgument);
}

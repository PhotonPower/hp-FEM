// Resonance derivatives (M16 S4): the eigenvalue derivative of a quasi-normal mode with the
// left eigenvector, for the permittivity of a tag, a mesh velocity (the growth of a block),
// beta and the Bloch wavenumber, against central differences of re-solved eigenproblems; an
// open (PML, lossy) cavity for Resonance<2>, also on a hanging-node mesh (real constraints keep
// the pencil complex symmetric), and a Bloch strip with PML for ConicalResonance (the left
// vector of the non-symmetric reduced pencil); the Q and wavelength derivatives; argument checks.
#include <cmath>
#include <complex>
#include <limits>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/periodic.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/adaptive_mesh.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/physics/conical_resonance.hpp"
#include "hpfem/physics/eigen_sensitivity.hpp"
#include "hpfem/physics/resonance.hpp"
#include "hpfem/physics/shape_sensitivity.hpp"
#include "hpfem/pml/pml.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::physics::ConicalResonance;
using hpfem::physics::ConicalResonanceSetup;
using hpfem::physics::Resonance;
using hpfem::physics::ResonanceSetup;
namespace box_tag = hpfem::mesh::box_tag;
namespace c = hpfem::constants;

namespace {

constexpr hpfem::mesh::Tag kBlock = 2;

Complex lambda_of(Complex omega) {
  const Complex k = omega / c::c0;
  return k * k;
}

/// The eigenvalue of the re-solved modes closest to `reference`.
template <class Modes>
Complex closest(const Modes& modes, Complex reference) {
  Complex best{std::numeric_limits<Real>::infinity(), 0.0};
  for (const auto& mode : modes) {
    const Complex lambda = lambda_of(mode.omega);
    if (std::abs(lambda - reference) < std::abs(best - reference)) best = lambda;
  }
  return best;
}

void tag_block(hpfem::mesh::Mesh<2>& mesh, Real x0, Real x1, Real y0, Real y1) {
  for (Index cell = 0; cell < mesh.num_cells(); ++cell) {
    const Point<2> x = hpfem::mesh::affine_map(mesh, cell).centroid();
    if (x(0) > x0 && x(0) < x1 && x(1) > y0 && x(1) < y1) mesh.set_cell_tag(cell, kBlock);
  }
}

// --- an open cavity: a lossy block in a PML box ------------------------------------------

ResonanceSetup<2> cavity_setup(Complex eps) {
  ResonanceSetup<2> setup;
  setup.target_omega = 2.4 * c::c0;
  hpfem::materials::Material block;
  block.eps_r = eps;
  setup.materials.set(kBlock, block);
  setup.pml = hpfem::pml::PmlBox<2>::uniform(Point<2>(-0.6, -0.6), Point<2>(0.6, 0.6), 0.4, 2.4,
                                             1.0, hpfem::pml::PmlProfile{2, 1e-8});
  setup.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  setup.num_modes = 4;
  setup.krylov_dimension = 40;
  return setup;
}

hpfem::mesh::Mesh<2> cavity_mesh() {
  hpfem::mesh::Mesh<2> mesh =
      hpfem::mesh::rectangle(16, 16, Point<2>(-1.0, -1.0), Point<2>(1.0, 1.0));
  tag_block(mesh, -0.25, 0.25, -0.25, 0.25);
  return mesh;
}

constexpr Complex kCavityEps{8.0, 0.1};

}  // namespace

TEST_CASE("resonance derivatives of an open cavity against re-solved modes",
          "[physics][resonance][sensitivity]") {
  const hpfem::mesh::Mesh<2> mesh = cavity_mesh();
  const hpfem::fespace::NedelecDofMap<2> dofs(mesh, 3);
  const Resonance<2> problem(dofs, cavity_setup(kCavityEps));
  const auto modes = problem.solve();
  REQUIRE(!modes.empty());
  const auto& mode = modes.front();
  REQUIRE(mode.omega.imag() < 0.0);  // leaky
  const Complex lambda = lambda_of(mode.omega);
  const auto adjoint = hpfem::physics::resonance_adjoint(problem, mode);
  REQUIRE(std::abs(adjoint.normalisation) > 0.0);

  // permittivity of the block (holomorphic)
  const auto d_eps = hpfem::physics::resonance_material_derivative(problem, mode, adjoint, kBlock);
  const Complex h{1e-4, 0.0};
  const auto solve_eps = [&](Complex eps) {
    return closest(Resonance<2>(dofs, cavity_setup(eps)).solve(), lambda);
  };
  const Complex fd_eps = (solve_eps(kCavityEps + h) - solve_eps(kCavityEps - h)) / (2.0 * h);
  INFO("d lambda / d eps = " << d_eps.dlambda << ", difference quotient " << fd_eps);
  REQUIRE(std::abs(d_eps.dlambda - fd_eps) < 1e-5 * std::abs(fd_eps));
  // the derivative with respect to Im eps (i times) moves the width
  const Complex fd_im =
      (solve_eps(kCavityEps + Complex{0.0, 1e-4}) - solve_eps(kCavityEps - Complex{0.0, 1e-4})) /
      2e-4;
  REQUIRE(std::abs(hpfem::kI * d_eps.dlambda - fd_im) < 1e-5 * std::abs(fd_im));

  // the uniform growth of the block
  const hpfem::physics::NodeField velocity = hpfem::physics::region_normal_velocity(mesh, kBlock);
  const auto d_shape = hpfem::physics::resonance_shape_derivative(problem, mode, adjoint, velocity);
  const Real t = 1e-5;
  const auto solve_moved = [&](Real s) {
    hpfem::mesh::Mesh<2> moved = mesh;
    hpfem::physics::move_nodes(moved, velocity, s);
    const hpfem::fespace::NedelecDofMap<2> moved_dofs(moved, 3);
    return closest(Resonance<2>(moved_dofs, cavity_setup(kCavityEps)).solve(), lambda);
  };
  const Complex fd_shape = (solve_moved(t) - solve_moved(-t)) / (2 * t);
  INFO("d lambda / d t = " << d_shape.dlambda << ", difference quotient " << fd_shape);
  REQUIRE(std::abs(d_shape.dlambda - fd_shape) < 1e-4 * std::abs(fd_shape));

  // omega, Q and the wavelength follow from d lambda
  const Complex k = mode.omega / c::c0;
  REQUIRE(std::abs(d_eps.domega - c::c0 * d_eps.dlambda / (2.0 * k)) <
          1e-12 * std::abs(d_eps.domega));
  const auto quality_of = [](Complex lam) {
    Complex kk = std::sqrt(lam);
    if (kk.real() < 0) kk = -kk;
    return kk.real() / (-2 * kk.imag());
  };
  const Real fd_q =
      (quality_of(solve_eps(kCavityEps + h)) - quality_of(solve_eps(kCavityEps - h))) /
      (2 * h.real());
  INFO("dQ/d eps = " << d_eps.dquality << ", difference quotient " << fd_q);
  REQUIRE(std::abs(d_eps.dquality - fd_q) < 1e-4 * std::abs(fd_q));
  REQUIRE(d_eps.dwavelength ==
          -2 * c::pi * c::c0 * d_eps.domega.real() / (mode.omega.real() * mode.omega.real()));
}

TEST_CASE("resonance derivative on a hanging-node mesh", "[physics][resonance][sensitivity]") {
  hpfem::mesh::AdaptiveMesh<2> adaptive(cavity_mesh());
  const std::vector<Index> marked{100, 101, 120, 135};
  adaptive.refine(marked);
  const hpfem::mesh::Mesh<2>& mesh = adaptive.mesh();
  REQUIRE_FALSE(mesh.is_conforming());
  const hpfem::fespace::NedelecDofMap<2> dofs(mesh, 2);
  const Resonance<2> problem(dofs, cavity_setup(kCavityEps));
  const auto modes = problem.solve();
  REQUIRE(!modes.empty());
  const Complex lambda = lambda_of(modes.front().omega);
  const auto adjoint = hpfem::physics::resonance_adjoint(problem, modes.front());
  const auto d_eps =
      hpfem::physics::resonance_material_derivative(problem, modes.front(), adjoint, kBlock);
  const Complex h{1e-4, 0.0};
  const auto solve_eps = [&](Complex eps) {
    return closest(Resonance<2>(dofs, cavity_setup(eps)).solve(), lambda);
  };
  const Complex fd = (solve_eps(kCavityEps + h) - solve_eps(kCavityEps - h)) / (2.0 * h);
  INFO("hanging: " << d_eps.dlambda << " against " << fd);
  REQUIRE(std::abs(d_eps.dlambda - fd) < 1e-5 * std::abs(fd));
}

namespace {

// --- a Bloch strip with a lossy block and PML above and below ----------------------------

constexpr Real kKx = 0.7;
constexpr Real kBeta = 0.4;
constexpr Complex kStripEps{6.0, 0.2};

ConicalResonanceSetup strip_setup(Complex eps, Real kx, Real beta) {
  ConicalResonanceSetup setup;
  setup.target_omega = 3.2 * c::c0;
  setup.beta = beta;
  hpfem::materials::Material block;
  block.eps_r = eps;
  setup.materials.set(kBlock, block);
  setup.pml = hpfem::pml::PmlBox<2>(Point<2>(0.0, -1.0), Point<2>(1.0, 1.0),
                                    hpfem::pml::PmlBox<2>::Thickness{0.0, 0.0, 0.5, 0.5}, 3.2, 1.0,
                                    hpfem::pml::PmlProfile{2, 1e-8});
  setup.pec_tags = {box_tag::kYMin, box_tag::kYMax};
  setup.periodic = {hpfem::assembly::PeriodicPair<2>{box_tag::kXMin, box_tag::kXMax,
                                                     Point<2>(1.0, 0.0), std::exp(hpfem::kI * kx)}};
  setup.num_modes = 4;
  setup.krylov_dimension = 40;
  return setup;
}

hpfem::mesh::Mesh<2> strip_mesh() {
  hpfem::mesh::Mesh<2> mesh =
      hpfem::mesh::rectangle(8, 24, Point<2>(0.0, -1.5), Point<2>(1.0, 1.5));
  tag_block(mesh, 0.25, 0.75, -0.25, 0.25);
  return mesh;
}

}  // namespace

TEST_CASE("conical resonance derivatives with the left eigenvector of the Bloch pencil",
          "[physics][conical][resonance][sensitivity][periodic]") {
  const hpfem::mesh::Mesh<2> mesh = strip_mesh();
  const hpfem::fespace::NedelecDofMap<2> nd(mesh, 3);
  const hpfem::fespace::DofMap<2> h1(mesh, 3);
  const ConicalResonance problem(nd, h1, strip_setup(kStripEps, kKx, kBeta));
  const auto modes = problem.solve().modes;
  REQUIRE(!modes.empty());
  const auto& mode = modes.front();
  REQUIRE(mode.omega.imag() < 0.0);
  const Complex lambda = lambda_of(mode.omega);
  const auto adjoint = hpfem::physics::conical_resonance_adjoint(problem, mode);
  INFO("left vector residual " << adjoint.residual);
  REQUIRE(adjoint.residual < 1e-9);
  // the Bloch phases make the reduced pencil non-symmetric: the left vector is not the mode
  Vector x(mode.transverse.size() + mode.longitudinal.size());
  x << mode.transverse, mode.longitudinal;
  const Complex overlap = (adjoint.field.adjoint() * x).value();
  REQUIRE(std::abs(overlap) < 0.999 * adjoint.field.norm() * x.norm());

  const auto solve = [&](Complex eps, Real kx, Real beta) {
    return closest(ConicalResonance(nd, h1, strip_setup(eps, kx, beta)).solve().modes, lambda);
  };
  // permittivity
  const auto d_eps =
      hpfem::physics::conical_resonance_material_derivative(problem, mode, adjoint, kBlock);
  const Complex h{1e-4, 0.0};
  const Complex fd_eps =
      (solve(kStripEps + h, kKx, kBeta) - solve(kStripEps - h, kKx, kBeta)) / (2.0 * h);
  INFO("eps: " << d_eps.dlambda << " against " << fd_eps);
  REQUIRE(std::abs(d_eps.dlambda - fd_eps) < 1e-5 * std::abs(fd_eps));
  // beta
  const auto d_beta =
      hpfem::physics::conical_resonance_beta_derivative(problem, mode, adjoint, 1e-6);
  const Complex fd_beta =
      (solve(kStripEps, kKx, kBeta + 1e-4) - solve(kStripEps, kKx, kBeta - 1e-4)) / 2e-4;
  INFO("beta: " << d_beta.dlambda << " against " << fd_beta);
  REQUIRE(std::abs(d_beta.dlambda - fd_beta) < 1e-5 * std::abs(fd_beta));
  // the Bloch wavenumber: neighbour problems with the phases at kx -+ step
  const Real step = 1e-6;
  const ConicalResonance minus(nd, h1, strip_setup(kStripEps, kKx - step, kBeta));
  const ConicalResonance plus(nd, h1, strip_setup(kStripEps, kKx + step, kBeta));
  const auto d_kx =
      hpfem::physics::conical_resonance_bloch_derivative(problem, mode, adjoint, minus, plus, step);
  const Complex fd_kx =
      (solve(kStripEps, kKx + 1e-4, kBeta) - solve(kStripEps, kKx - 1e-4, kBeta)) / 2e-4;
  INFO("kx: " << d_kx.dlambda << " against " << fd_kx);
  REQUIRE(std::abs(d_kx.dlambda - fd_kx) < 1e-5 * std::abs(fd_kx));
  // the growth of the block
  const hpfem::physics::NodeField velocity = hpfem::physics::region_normal_velocity(mesh, kBlock);
  const auto d_shape =
      hpfem::physics::conical_resonance_shape_derivative(problem, mode, adjoint, velocity);
  const Real t = 1e-5;
  const auto solve_moved = [&](Real s) {
    hpfem::mesh::Mesh<2> moved = mesh;
    hpfem::physics::move_nodes(moved, velocity, s);
    const hpfem::fespace::NedelecDofMap<2> nd_moved(moved, 3);
    const hpfem::fespace::DofMap<2> h1_moved(moved, 3);
    return closest(
        ConicalResonance(nd_moved, h1_moved, strip_setup(kStripEps, kKx, kBeta)).solve().modes,
        lambda);
  };
  const Complex fd_shape = (solve_moved(t) - solve_moved(-t)) / (2 * t);
  INFO("shape: " << d_shape.dlambda << " against " << fd_shape);
  REQUIRE(std::abs(d_shape.dlambda - fd_shape) < 1e-4 * std::abs(fd_shape));
}

TEST_CASE("resonance derivatives: argument checks", "[physics][resonance][sensitivity]") {
  const hpfem::mesh::Mesh<2> mesh = cavity_mesh();
  const hpfem::fespace::NedelecDofMap<2> dofs(mesh, 2);
  const Resonance<2> problem(dofs, cavity_setup(kCavityEps));
  const auto modes = problem.solve();
  REQUIRE(!modes.empty());
  const auto adjoint = hpfem::physics::resonance_adjoint(problem, modes.front());
  REQUIRE_THROWS_AS(
      hpfem::physics::resonance_material_derivative(problem, modes.front(), adjoint, 77),
      hpfem::InvalidArgument);
  hpfem::physics::ResonantMode wrong = modes.front();
  wrong.field = Vector::Zero(3);
  REQUIRE_THROWS_AS(hpfem::physics::resonance_adjoint(problem, wrong), hpfem::InvalidArgument);
  const hpfem::physics::NodeField bad = hpfem::physics::NodeField::Zero(3, 2);
  REQUIRE_THROWS_AS(
      hpfem::physics::resonance_shape_derivative(problem, modes.front(), adjoint, bad),
      hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(hpfem::physics::resonance_shape_derivative(
                        problem, modes.front(), adjoint,
                        hpfem::physics::region_normal_velocity(mesh, kBlock), 0.0),
                    hpfem::InvalidArgument);
  // a conical problem without Bloch constraints has no Bloch derivative
  const hpfem::fespace::NedelecDofMap<2> nd(mesh, 2);
  const hpfem::fespace::DofMap<2> h1(mesh, 2);
  ConicalResonanceSetup closed;
  closed.target_omega = 3.0 * c::c0;
  closed.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  closed.num_modes = 2;
  closed.krylov_dimension = 30;
  const ConicalResonance plain(nd, h1, closed);
  const auto plain_modes = plain.solve().modes;
  REQUIRE(!plain_modes.empty());
  const auto plain_adjoint = hpfem::physics::conical_resonance_adjoint(plain, plain_modes.front());
  REQUIRE(plain_adjoint.reduced.size() == 0);  // y = x without constraints
  REQUIRE_THROWS_AS(hpfem::physics::conical_resonance_bloch_derivative(
                        plain, plain_modes.front(), plain_adjoint, plain, plain, 1e-6),
                    hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(hpfem::physics::conical_resonance_beta_derivative(plain, plain_modes.front(),
                                                                      plain_adjoint, 0.0),
                    hpfem::InvalidArgument);
}

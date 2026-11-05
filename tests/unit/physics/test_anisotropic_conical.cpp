// Crystals in the conical (2.5D) solver (ADR-0015 §4, M19 S3). Convention exp(-iωt).
#include <cmath>
#include <random>

#include <catch2/catch_test_macros.hpp>

#include "hpfem/adaptivity/conical_estimator.hpp"
#include "hpfem/adaptivity/refinement.hpp"
#include "hpfem/assembly/conical_forms.hpp"
#include "hpfem/assembly/periodic.hpp"
#include "hpfem/assembly/prolongation.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/physics/conical_goal.hpp"
#include "hpfem/physics/conical_resonance.hpp"
#include "hpfem/physics/conical_scattering.hpp"
#include "hpfem/physics/eigen_sensitivity.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sensitivity.hpp"
#include "hpfem/physics/sources.hpp"
#include "hpfem/pml/pml.hpp"
#include "hpfem/solvers/linear_solver.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::SparseMatrix;
using hpfem::Vector;
using hpfem::assembly::ConicalForm;
using hpfem::assembly::ConicalTensor;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::materials::Material;
using hpfem::materials::Tensor3;
using hpfem::mesh::Mesh;
using hpfem::physics::ConicalScattering;
using hpfem::physics::ConicalScatteringSetup;
using hpfem::physics::ConicalVector;
using hpfem::pml::PmlBox;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kK0 = 3.0;

/// Absorbing crystal; `coupling` sets eps_xz, eps_yz (the entries that break the symmetry).
Tensor3 crystal(Complex coupling) {
  Tensor3 eps;
  eps << Complex{2.5, 0.1}, 0.4, coupling, 0.4, 3.0, 0.5 * coupling, coupling, 0.5 * coupling,
      Complex{3.5, 0.05};
  return eps;
}

Mesh<2> with_scatterer(Index n) {
  Mesh<2> m = hpfem::mesh::rectangle(n, n);
  for (Index c = 0; c < m.num_cells(); ++c) {
    if ((hpfem::mesh::affine_map(m, c).centroid() - Point<2>(0.5, 0.5)).norm() < 0.2) {
      m.set_cell_tag(c, 2);
    }
  }
  return m;
}

/// Relative difference ||a - b^T|| / ||a|| (Frobenius).
Real transposed_difference(const SparseMatrix& a, const SparseMatrix& b) {
  const SparseMatrix bt = b.transpose();
  return SparseMatrix(a - bt).norm() / a.norm();
}

SparseMatrix block_operator(const NedelecDofMap<2>& nd, const DofMap<2>& h1, Real beta,
                            const ConicalTensor& eps) {
  const auto form = [&](Index c) {
    ConicalForm f;
    if (nd.mesh().cell_tag(c) == 2) f.permittivity = [eps](const Point<2>&) { return eps; };
    return f;
  };
  const auto system = hpfem::assembly::assemble_conical(nd, h1, beta, form);
  return SparseMatrix(system.stiffness - kK0 * kK0 * system.mass);
}

PmlBox<2> pml() {
  PmlBox<2>::Thickness t;
  t.fill(0.25);
  return PmlBox<2>(Point<2>::Constant(0.25), Point<2>::Constant(0.75), t, kK0);
}

const std::vector<hpfem::mesh::Tag> kSides = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin,
                                              box_tag::kYMax};

}  // namespace

TEST_CASE(
    "Anisotropic conical: the full-tensor path equals the diagonal one for a diagonal "
    "tensor",
    "[assembly][conical][anisotropic]") {
  const Mesh<2> m = with_scatterer(3);
  const NedelecDofMap<2> nd(m, 3);
  const DofMap<2> h1(m, 3);
  ConicalTensor eps = ConicalTensor::Zero();
  eps.diagonal() << Complex{2.5, 0.1}, 3.0, Complex{3.5, 0.05};
  ConicalTensor inv_mu = ConicalTensor::Zero();
  inv_mu.diagonal() << 0.9, 1.1, Complex{1.2, 0.1};
  // off-diagonal entries of 1e-300 force the general path without changing the numbers
  ConicalTensor eps_full = eps;
  ConicalTensor inv_mu_full = inv_mu;
  eps_full(0, 1) = eps_full(1, 0) = eps_full(0, 2) = eps_full(2, 0) = 1e-300;
  inv_mu_full(1, 2) = inv_mu_full(2, 1) = 1e-300;
  REQUIRE_FALSE(hpfem::assembly::is_diagonal(eps_full));
  const auto system = [&](const ConicalTensor& e, const ConicalTensor& mu) {
    return hpfem::assembly::assemble_conical(nd, h1, 1.7, [&](Index) {
      ConicalForm f;
      f.permittivity = [e](const Point<2>&) { return e; };
      f.inverse_permeability = [mu](const Point<2>&) { return mu; };
      f.source = [](const Point<2>& x) {
        return ConicalVector(std::sin(x(0)), Complex{0.2, 0.1} * x(1), Complex{0.0, 0.3});
      };
      return f;
    });
  };
  const auto diagonal = system(eps, inv_mu);
  const auto full = system(eps_full, inv_mu_full);
  REQUIRE(SparseMatrix(full.stiffness - diagonal.stiffness).norm() <
          1e-13 * diagonal.stiffness.norm());
  REQUIRE(SparseMatrix(full.mass - diagonal.mass).norm() < 1e-13 * diagonal.mass.norm());
  REQUIRE((full.rhs - diagonal.rhs).norm() < 1e-13 * diagonal.rhs.norm());
}

TEST_CASE("Anisotropic conical: symmetry with and without eps_xz, eps_yz",
          "[assembly][conical][anisotropic]") {
  const Mesh<2> m = with_scatterer(4);
  const NedelecDofMap<2> nd(m, 2);
  const DofMap<2> h1(m, 2);
  const Real beta = 1.3;
  // eps_xy alone: complex-symmetric (the LDL^T paths stay available)
  const SparseMatrix a = block_operator(nd, h1, beta, crystal(0.0));
  REQUIRE(transposed_difference(a, a) < 1e-14);
  // eps_xz, eps_yz: not symmetric, but A(beta)^T = S A(-beta) S, S = diag(I, -I)
  const ConicalTensor eps = crystal(Complex{0.3, 0.02});
  const SparseMatrix general = block_operator(nd, h1, beta, eps);
  REQUIRE(hpfem::solvers::detect_symmetry(general) == hpfem::solvers::Symmetry::kGeneral);
  Vector s = Vector::Ones(nd.num_dofs() + h1.num_dofs());
  s.tail(h1.num_dofs()).setConstant(-1.0);
  const SparseMatrix mirrored =
      s.asDiagonal() * block_operator(nd, h1, -beta, eps) * s.asDiagonal();
  REQUIRE(transposed_difference(general, mirrored) < 1e-14);
  // and at beta = 0 the coupling entries tie E_z to the in-plane field
  const SparseMatrix at_zero = block_operator(nd, h1, 0.0, eps);
  Real coupling = 0;
  for (Index row = 0; row < nd.num_dofs(); ++row) {
    for (SparseMatrix::InnerIterator it(at_zero, row); it; ++it) {
      if (it.col() >= nd.num_dofs()) coupling = std::max(coupling, std::abs(it.value()));
    }
  }
  REQUIRE(coupling > 1e-3);
}

TEST_CASE("Anisotropic conical: at beta = 0 an in-plane crystal reproduces the 2D solver",
          "[physics][conical][anisotropic]") {
  // eps_xy != 0 takes the full-tensor path of the conical forms; the in-plane (H_z)
  // polarisation must equal the 2D volume solver with the in-plane block (M19 S2)
  const Mesh<2> m = with_scatterer(8);
  const NedelecDofMap<2> nd(m, 3);
  const DofMap<2> h1(m, 3);
  const Material material = Material::anisotropic(crystal(0.0));
  const Point<3> k(kK0 * std::cos(0.4), kK0 * std::sin(0.4), 0.0);
  const ConicalVector e0(-std::sin(0.4), std::cos(0.4), 0.0);

  ConicalScatteringSetup conical;
  conical.omega = kK0 * hpfem::constants::c0;
  conical.materials.set(2, material);
  conical.incident = hpfem::physics::conical_plane_wave(e0, k);
  conical.incident_curl = hpfem::physics::conical_plane_wave_curl(e0, k);
  conical.pml = pml();
  conical.pec_tags = kSides;
  const auto solution = ConicalScattering(nd, h1, conical).solve();
  REQUIRE(solution.longitudinal.norm() < 1e-12 * solution.transverse.norm());

  hpfem::physics::ScatteringSetup<2> planar;
  planar.omega = conical.omega;
  planar.materials.set(2, material);
  planar.incident = hpfem::physics::plane_wave<2>(e0.head<2>(), Point<2>(k(0), k(1)));
  planar.formulation = hpfem::physics::Formulation::kScatteredField;
  planar.pml = pml();
  planar.incident_tags = kSides;
  const auto reference = hpfem::physics::Scattering<2>(nd, planar).solve().unknown;
  REQUIRE(reference.norm() > 0);
  REQUIRE((solution.transverse - reference).norm() < 1e-10 * reference.norm());
}

TEST_CASE("Anisotropic conical: isotropic limit and the scalar E_z path",
          "[physics][conical][anisotropic]") {
  const Mesh<2> m = with_scatterer(6);
  const NedelecDofMap<2> nd(m, 2);
  const DofMap<2> h1(m, 2);
  const Point<3> k(kK0 * std::cos(0.3) * std::cos(0.5), kK0 * std::sin(0.3) * std::cos(0.5),
                   kK0 * std::sin(0.5));
  const ConicalVector e0 = hpfem::physics::conical_polarisation(k, Point<3>(0.0, 1.0, 0.0),
                                                                hpfem::physics::Polarisation::kP);
  const auto solve = [&](const Material& material) {
    ConicalScatteringSetup setup;
    setup.omega = kK0 * hpfem::constants::c0;
    setup.beta = k(2);
    setup.materials.set(2, material);
    setup.incident = hpfem::physics::conical_plane_wave(e0, k);
    setup.incident_curl = hpfem::physics::conical_plane_wave_curl(e0, k);
    setup.pml = pml();
    setup.pec_tags = kSides;
    const auto s = ConicalScattering(nd, h1, setup).solve();
    Vector x(s.transverse.size() + s.longitudinal.size());
    x << s.transverse, s.longitudinal;
    return x;
  };
  const Complex eps{2.25, 0.3};
  const Vector scalar = solve(Material::isotropic(eps));
  const Vector tensor = solve(Material::anisotropic(Tensor3(eps * Tensor3::Identity())));
  REQUIRE((tensor - scalar).norm() < 1e-12 * scalar.norm());
  // conical incidence with eps_xz couples through the general tensor: a different field
  const Vector coupled = solve(Material::anisotropic(crystal(Complex{0.3, 0.0})));
  REQUIRE((coupled - scalar).norm() > 1e-3 * scalar.norm());

  // scalar E_z: an eps_xy crystal still decouples, eps_xz does not
  ConicalScatteringSetup ez;
  ez.omega = kK0 * hpfem::constants::c0;
  ez.scalar_ez = true;
  ez.incident =
      hpfem::physics::conical_plane_wave(ConicalVector(0.0, 0.0, 1.0), Point<3>(kK0, 0.0, 0.0));
  ez.pml = pml();
  ez.pec_tags = kSides;
  ez.materials.set(2, Material::anisotropic(crystal(0.0)));
  REQUIRE_NOTHROW(ConicalScattering(nd, h1, ez));
  ez.materials.set(2, Material::anisotropic(crystal(Complex{0.3, 0.0})));
  REQUIRE_THROWS_AS(ConicalScattering(nd, h1, ez), hpfem::AnisotropicUnsupported);
}

TEST_CASE("Anisotropic conical: the adjoint solves the transposed system",
          "[physics][conical][anisotropic][sensitivity]") {
  // a crystal with eps_xz, eps_yz (A not symmetric) in a Bloch-periodic strip with a current:
  // dQ/d delta for eps + delta I by the adjoint must match finite differences, which holds only
  // if the adjoint solves A^T z = q
  Mesh<2> mesh = hpfem::mesh::rectangle(4, 3, Point<2>(0.0, 0.0), Point<2>(1.0, 0.75));
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const Point<2> x = hpfem::mesh::affine_map(mesh, c).centroid();
    if (x(0) > 0.25 && x(0) < 0.75 && x(1) > 0.25 && x(1) < 0.5) mesh.set_cell_tag(c, 5);
  }
  const NedelecDofMap<2> nd(mesh, 3);
  const DofMap<2> h1(mesh, 3);
  const Tensor3 eps = crystal(Complex{0.6, 0.05});
  const auto make_setup = [&](Complex delta) {
    ConicalScatteringSetup setup;
    setup.omega = 4.0 * hpfem::constants::c0;
    setup.beta = 0.9;
    setup.materials.set(5, Material::anisotropic(Tensor3(eps + delta * Tensor3::Identity())));
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
  const auto q_pair = functional(nd, h1);
  const Vector& q_e = q_pair.first;
  const Vector& q_v = q_pair.second;
  const auto goal = [&](Complex delta) {
    const auto s = ConicalScattering(nd, h1, make_setup(delta)).solve();
    return (q_e.transpose() * s.transverse).value() + (q_v.transpose() * s.longitudinal).value();
  };
  const ConicalScattering problem(nd, h1, make_setup(0.0));
  const auto solution = problem.solve();
  const auto z = hpfem::physics::conical_adjoint_solution(problem, q_e, q_v);
  const Complex derivative = hpfem::physics::conical_material_sensitivity(problem, solution, z, 5);
  constexpr Real kDelta = 1e-6;
  const Complex fd = (goal(kDelta) - goal(-kDelta)) / (2 * kDelta);
  INFO("adjoint " << derivative << " FD " << fd);
  REQUIRE(std::abs(fd) > 0);
  REQUIRE(std::abs(derivative - fd) < 1e-6 * std::abs(fd));
}

TEST_CASE("Anisotropic conical: the estimator's residual matches the full-tensor forms",
          "[adaptivity][conical][anisotropic]") {
  // sum_K r_K(W) = l(W) - a(E_h, W) on the enriched space: the estimator works in physical
  // components, the assembly in scaled ones, so this checks the factors i of the coupling terms
  const Mesh<2> mesh = hpfem::mesh::rectangle(3, 2);
  const NedelecDofMap<2> nd(mesh, 2);
  const DofMap<2> h1(mesh, 2);
  const NedelecDofMap<2> nd_e(mesh, 3);
  const DofMap<2> h1_e(mesh, 3);
  const Real beta = 0.9;
  const Real k2 = 1.7;
  ConicalTensor inv_mu;
  inv_mu << 1.1, 0.2, Complex{0.1, 0.05}, 0.2, 0.9, -0.15, Complex{0.1, 0.05}, -0.15, 1.3;
  const ConicalTensor eps = crystal(Complex{0.4, 0.03});
  const auto forms = [&](Index) {
    ConicalForm form;
    form.permittivity = [eps](const Point<2>&) { return eps; };
    form.inverse_permeability = [inv_mu](const Point<2>&) { return inv_mu; };
    form.source = [](const Point<2>& x) {
      return ConicalVector(std::sin(x(0)) + 0.2, std::cos(x(1)) * x(0), Complex{0.3, 0.1} * x(1));
    };
    return form;
  };
  const auto random = [](Index n, unsigned seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<Real> normal;
    Vector v(n);
    for (Index i = 0; i < n; ++i) v(i) = Complex{normal(gen), normal(gen)};
    return v;
  };
  const Vector e = random(nd.num_dofs(), 1);
  const Vector v = random(h1.num_dofs(), 2);
  const Vector we = random(nd_e.num_dofs(), 3);
  const Vector wv = random(h1_e.num_dofs(), 4);
  const auto system = hpfem::assembly::assemble_conical(nd_e, h1_e, beta, forms, 4);
  const auto identity = hpfem::adaptivity::identity_step(mesh.num_cells());
  Vector e_full(nd_e.num_dofs() + h1_e.num_dofs());
  e_full << hpfem::assembly::prolongate(nd, e, nd_e, identity),
      hpfem::assembly::prolongate(h1, v, h1_e, identity);
  Vector w_full(nd_e.num_dofs() + h1_e.num_dofs());
  w_full << we, wv;
  const Complex a = (w_full.transpose() * ((system.stiffness - k2 * system.mass) * e_full))(0);
  const Complex l = (w_full.transpose() * system.rhs)(0);
  const auto r = hpfem::adaptivity::conical_weighted_residual(nd, h1, e, v, beta, k2, forms, nd_e,
                                                              h1_e, we, wv);
  Complex sum = 0;
  for (const Complex c : r) sum += c;
  REQUIRE(std::abs(sum - (l - a)) < 1e-8 * (std::abs(a) + std::abs(l)));
}

TEST_CASE("Anisotropic conical: resonance derivative with the left vector of a crystal",
          "[physics][conical][anisotropic][resonance][sensitivity]") {
  // a closed cavity (PEC all round: no constraints) with a coupling crystal block at beta != 0:
  // the pencil is not symmetric, so the left eigenvector differs from the mode; the derivative
  // for eps + delta I must match finite differences of the eigenvalue
  Mesh<2> mesh = hpfem::mesh::rectangle(6, 6);
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const Point<2> x = hpfem::mesh::affine_map(mesh, c).centroid();
    if (x(0) > 0.3 && x(0) < 0.8 && x(1) > 0.2 && x(1) < 0.6) mesh.set_cell_tag(c, 5);
  }
  const NedelecDofMap<2> nd(mesh, 3);
  const DofMap<2> h1(mesh, 3);
  const Tensor3 eps = crystal(Complex{0.8, 0.05});
  const auto setup = [&](Complex delta) {
    hpfem::physics::ConicalResonanceSetup s;
    s.target_omega = 4.0 * hpfem::constants::c0;
    s.beta = 1.1;
    s.materials.set(5, Material::anisotropic(Tensor3(eps + delta * Tensor3::Identity())));
    s.pec_tags = kSides;
    s.num_modes = 3;
    s.krylov_dimension = 30;
    return s;
  };
  const auto lambda_of = [](Complex omega) {
    const Complex k = omega / hpfem::constants::c0;
    return k * k;
  };
  const hpfem::physics::ConicalResonance problem(nd, h1, setup(0.0));
  REQUIRE_FALSE(problem.constraints());
  const auto modes = problem.solve().modes;
  REQUIRE(!modes.empty());
  const auto& mode = modes.front();
  const Complex lambda = lambda_of(mode.omega);
  const auto adjoint = hpfem::physics::conical_resonance_adjoint(problem, mode);
  INFO("left vector residual " << adjoint.residual);
  REQUIRE(adjoint.residual < 1e-9);
  const auto derivative =
      hpfem::physics::conical_resonance_material_derivative(problem, mode, adjoint, 5);
  const auto closest = [&](Complex delta) {
    Complex best{1e300, 0.0};
    for (const auto& m : hpfem::physics::ConicalResonance(nd, h1, setup(delta)).solve().modes) {
      if (std::abs(lambda_of(m.omega) - lambda) < std::abs(best - lambda))
        best = lambda_of(m.omega);
    }
    return best;
  };
  const Complex h{1e-4, 0.0};
  const Complex fd = (closest(h) - closest(-h)) / (2.0 * h);
  INFO("adjoint " << derivative.dlambda << " FD " << fd);
  REQUIRE(std::abs(derivative.dlambda - fd) < 1e-5 * std::abs(fd));
}

TEST_CASE("Anisotropic conical: a crystal stack as the layered background",
          "[physics][conical][anisotropic][stack]") {
  using hpfem::physics::AnisotropicStack;
  using hpfem::physics::Layer;
  using hpfem::physics::Polarisation;
  const Material cover = Material::dielectric(1.2);
  // isotropic stack: the same wave as layered_conical_wave (frame map and curl sign)
  {
    const std::vector<Layer> layers{{Material::isotropic(Complex{4.0, 0.3}), 0.13}};
    const Material sub = Material::dielectric(1.5);
    for (const Polarisation pol : {Polarisation::kS, Polarisation::kP}) {
      const auto a = hpfem::physics::anisotropic_conical_wave(AnisotropicStack(cover, layers, sub),
                                                              kK0, 0.5, 0.7, pol);
      const auto b = hpfem::physics::layered_conical_wave(
          hpfem::physics::LayerStack<2>(cover, layers, sub), kK0, 0.5, 0.7, pol);
      REQUIRE(std::abs(a.wave.beta - b.beta) < 1e-14);
      REQUIRE(std::abs(a.wave.kx - b.kx) < 1e-14);
      REQUIRE(std::abs(a.wave.ky - b.ky) < 1e-14);
      REQUIRE(std::abs(a.wave.reflectance - b.reflectance) < 1e-12);
      for (const Real y : {0.3, -0.05, -0.4}) {
        const Point<2> x(0.17, y);
        REQUIRE((a.wave.field(x) - b.field(x)).norm() < 1e-10);  // round-off of two routes
        REQUIRE((a.wave.field_curl(x) - b.field_curl(x)).norm() < 1e-10 * kK0);
        REQUIRE((a.wave.incident(x) - b.incident(x)).norm() < 1e-10);
      }
    }
  }
  // crystal stack: the physical curl matches the field (E_z = i v, d/dz = i beta)
  const Material plate = Material::anisotropic(
      Tensor3(crystal(Complex{0.3, 0.0}) + Complex{0.0, 0.02} * Tensor3::Identity()));
  const Material sub = Material::anisotropic(
      Tensor3(crystal(Complex{0.2, 0.0}) + Complex{0.0, 0.03} * Tensor3::Identity()));
  const AnisotropicStack stack(cover, {{plate, 0.2}}, sub);
  const auto cw = hpfem::physics::anisotropic_conical_wave(stack, kK0, 0.6, 0.9, Polarisation::kP);
  const Real beta = cw.wave.beta;
  for (const Real y : {0.25, -0.1, -0.5}) {
    const Point<2> x(0.11, y);
    const Real h = 1e-6;
    const auto physical = [&](const Point<2>& p) {
      const ConicalVector f = cw.wave.field(p);
      return ConicalVector(f(0), f(1), hpfem::kI * f(2));
    };
    const ConicalVector e = physical(x);
    const ConicalVector dx =
        (physical(Point<2>(x(0) + h, x(1))) - physical(Point<2>(x(0) - h, x(1)))) / (2 * h);
    const ConicalVector dy =
        (physical(Point<2>(x(0), x(1) + h)) - physical(Point<2>(x(0), x(1) - h))) / (2 * h);
    const ConicalVector curl(dy(2) - hpfem::kI * beta * e(1), hpfem::kI * beta * e(0) - dx(2),
                             dx(1) - dy(0));
    INFO("y = " << y);
    REQUIRE((curl - cw.wave.field_curl(x)).norm() < 1e-6 * curl.norm());
  }
  // the background materials are in the solver frame (y and z swapped)
  const Tensor3 lab = sub.eps_tensor();
  const Tensor3 solver = cw.background.substrate().eps_tensor();
  REQUIRE(solver(1, 1) == lab(2, 2));
  REQUIRE(solver(0, 2) == lab(0, 1));
  // zero contrast: cells with the stack's own materials give no scattered field
  Mesh<2> mesh = hpfem::mesh::rectangle(2, 20, Point<2>(0.0, -1.0), Point<2>(0.25, 1.0));
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const Real yc = hpfem::mesh::affine_map(mesh, c).centroid()(1);
    if (yc < 0 && yc > -0.2) mesh.set_cell_tag(c, 2);
    if (yc < -0.2) mesh.set_cell_tag(c, 3);
  }
  const NedelecDofMap<2> nd(mesh, 2);
  const DofMap<2> h1(mesh, 2);
  ConicalScatteringSetup setup;
  setup.omega = kK0 * hpfem::constants::c0;
  setup.beta = beta;
  setup.materials = hpfem::materials::MaterialMap(cover);
  setup.materials.set(2, cw.background.material(1)).set(3, cw.background.substrate());
  setup.background = cw.background;
  setup.incident = cw.wave.field;
  setup.incident_curl = cw.wave.field_curl;
  setup.pec_tags = {box_tag::kYMin, box_tag::kYMax};
  setup.periodic = {hpfem::assembly::PeriodicPair<2>{box_tag::kXMin, box_tag::kXMax,
                                                     Point<2>(0.25, 0.0),
                                                     std::exp(hpfem::kI * cw.wave.kx * 0.25)}};
  const auto solution = ConicalScattering(nd, h1, setup).solve();
  REQUIRE(solution.transverse.norm() == 0.0);
  REQUIRE(solution.longitudinal.norm() == 0.0);
}

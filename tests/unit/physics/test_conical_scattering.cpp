// Conical forms and solver: the gradient of a potential spans the kernel of the stiffness,
// the system is symmetric, the blocks decouple at beta = 0, the plane wave and polarisation
// helpers are consistent, the PML tensors follow the stretch factors, and the setup rejects
// what the solver cannot do.
#include <cmath>
#include <complex>
#include <functional>
#include <numbers>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/conical_forms.hpp"
#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/physics/conical_scattering.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::Index;
using hpfem::kI;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::Mesh;
using hpfem::physics::conical_plane_wave;
using hpfem::physics::conical_polarisation;
using hpfem::physics::ConicalScattering;
using hpfem::physics::ConicalScatteringSetup;
using hpfem::physics::ConicalVector;
using hpfem::physics::Polarisation;
namespace box_tag = hpfem::mesh::box_tag;

TEST_CASE("conical forms: gradient kernel, symmetry and decoupling at beta = 0",
          "[assembly][conical]") {
  const Mesh<2> mesh = hpfem::mesh::rectangle(3, 3);
  const NedelecDofMap<2> nd(mesh, 2);
  const DofMap<2> h1(mesh, 2);
  const auto vacuum = [](Index) { return hpfem::assembly::ConicalForm{}; };
  for (const Real beta : {0.0, 1.3}) {
    const auto system = hpfem::assembly::assemble_conical(nd, h1, beta, vacuum);
    REQUIRE(system.num_nedelec == nd.num_dofs());
    REQUIRE(system.num_h1 == h1.num_dofs());
    const hpfem::SparseMatrix s_t = system.stiffness.transpose();
    REQUIRE((system.stiffness - s_t).norm() < 1e-12 * system.stiffness.norm());
    const hpfem::SparseMatrix m_t = system.mass.transpose();
    REQUIRE((system.mass - m_t).norm() < 1e-12 * system.mass.norm());
    // S K_beta psi = 0 for every potential
    const hpfem::SparseMatrix k = hpfem::assembly::conical_gradient(h1, nd, beta);
    Vector psi = Vector::Zero(h1.num_dofs());
    for (Index j = 0; j < psi.size(); ++j)
      psi(j) =
          Complex{std::sin(1.0 + 0.7 * static_cast<Real>(j)), std::cos(0.3 * static_cast<Real>(j))};
    const Vector grad = k * psi;
    REQUIRE((system.stiffness * grad).norm() < 1e-10 * system.stiffness.norm() * grad.norm());
    if (beta == 0.0) {
      // no coupling between the in-plane and the longitudinal block
      const Index n_e = nd.num_dofs();
      Real coupling = 0;
      for (Index row = 0; row < system.stiffness.rows(); ++row) {
        for (hpfem::SparseMatrix::InnerIterator it(system.stiffness, row); it; ++it) {
          if ((row < n_e) != (it.col() < n_e)) coupling = std::max(coupling, std::abs(it.value()));
        }
      }
      REQUIRE(coupling == 0.0);
    }
  }
  REQUIRE_THROWS_AS(hpfem::assembly::assemble_conical(
                        nd, DofMap<2>(hpfem::mesh::rectangle(2, 2), 2), 0.0, vacuum),
                    hpfem::InvalidArgument);
}

TEST_CASE("conical plane wave and polarisation vectors", "[physics][conical]") {
  const Point<3> k(1.0, 2.0, 1.5);
  const Point<3> normal(1.0, 0.0, 0.0);
  const ConicalVector s = conical_polarisation(k, normal, Polarisation::kS);
  const ConicalVector p = conical_polarisation(k, normal, Polarisation::kP);
  const auto dot = [](const ConicalVector& a, const Point<3>& b) {
    return a(0) * b(0) + a(1) * b(1) + a(2) * b(2);
  };
  REQUIRE(std::abs(dot(s, k)) < 1e-14);
  REQUIRE(std::abs(dot(p, k)) < 1e-14);
  REQUIRE(std::abs(dot(s, normal)) < 1e-14);  // s is perpendicular to the plane of incidence
  REQUIRE(s.norm() == Approx(1.0));
  REQUIRE(p.norm() == Approx(1.0));
  REQUIRE(std::abs(s.dot(p)) < 1e-14);
  // normal incidence: s falls back to the invariant direction
  const ConicalVector s0 = conical_polarisation(Point<3>(1.0, 0.0, 0.0), normal, Polarisation::kS);
  REQUIRE(std::abs(s0(2)) == Approx(1.0));
  // scaled components of the wave: (E0x, E0y, -i E0z) e^{i(kx x + ky y)}
  const auto wave = conical_plane_wave(s, k);
  const Point<2> x(0.3, -0.2);
  const Complex phase = std::exp(kI * (k(0) * x(0) + k(1) * x(1)));
  const ConicalVector value = wave(x);
  REQUIRE(std::abs(value(0) - s(0) * phase) < 1e-14);
  REQUIRE(std::abs(value(1) - s(1) * phase) < 1e-14);
  REQUIRE(std::abs(value(2) + kI * s(2) * phase) < 1e-14);
  REQUIRE_THROWS_AS(conical_plane_wave(ConicalVector(1.0, 0.0, 0.0), k), hpfem::InvalidArgument);
}

TEST_CASE("conical PML tensors follow the stretch factors", "[physics][conical]") {
  const Real k0 = 3.0;
  const hpfem::pml::PmlBox<2> box =
      hpfem::pml::PmlBox<2>::uniform(Point<2>(-1.0, -1.0), Point<2>(1.0, 1.0), 0.5, k0);
  hpfem::materials::Material material{Complex{2.0, 0.1}, Complex{1.0, 0.0}};
  const auto form = hpfem::physics::conical_pml_form(box, material);
  for (const Point<2>& x : {Point<2>(0.2, -0.3), Point<2>(1.3, 0.1), Point<2>(1.2, 1.4)}) {
    const auto s = box.stretch(x);
    const ConicalVector eps = form.permittivity(x);
    const ConicalVector inv_mu = form.inverse_permeability(x);
    REQUIRE(std::abs(eps(0) - material.eps_r * s(1) / s(0)) < 1e-12);
    REQUIRE(std::abs(eps(1) - material.eps_r * s(0) / s(1)) < 1e-12);
    REQUIRE(std::abs(eps(2) - material.eps_r * s(0) * s(1)) < 1e-12);
    REQUIRE(std::abs(inv_mu(0) * eps(0) - material.eps_r / material.mu_r) < 1e-12);
    REQUIRE(std::abs(inv_mu(2) * eps(2) - material.eps_r / material.mu_r) < 1e-12);
    if (box.in_layer(x)) {
      REQUIRE(std::abs(eps(2) - material.eps_r) > 1e-3);  // stretched
    } else {
      REQUIRE(std::abs(eps(2) - material.eps_r) < 1e-12);  // identity inside
    }
  }
}

TEST_CASE("conical scattering: a homogeneous cell is trivial and the setup is checked",
          "[physics][conical]") {
  const Mesh<2> mesh = hpfem::mesh::rectangle(3, 3);
  const NedelecDofMap<2> nd(mesh, 2);
  const DofMap<2> h1(mesh, 2);
  ConicalScatteringSetup setup;
  setup.omega = 2.0 * hpfem::constants::c0;
  setup.beta = 0.7;
  setup.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  const Point<3> k(1.0, 1.0, 0.7);
  setup.incident =
      conical_plane_wave(conical_polarisation(k, Point<3>(1.0, 0.0, 0.0), Polarisation::kP), k);
  // no contrast: the scattered field vanishes; the progress callback sees every phase
  std::vector<std::string> phases;
  setup.progress = [&](const hpfem::ProgressEvent& e) {
    phases.push_back(e.phase);
    return true;
  };
  const ConicalScattering problem(nd, h1, setup);
  const auto solution = problem.solve();
  REQUIRE(solution.scattered);
  REQUIRE(solution.transverse.norm() < 1e-14);
  REQUIRE(solution.longitudinal.norm() < 1e-14);
  REQUIRE(phases == std::vector<std::string>{"assembly", "constraints", "factorisation", "solve",
                                             "post", "done"});
  REQUIRE(solution.timing.size() == 6);
  REQUIRE(solution.timing.at("total") > 0.0);
  ConicalScatteringSetup cancelling = setup;
  cancelling.progress = [](const hpfem::ProgressEvent& e) { return e.step < 3; };
  REQUIRE_THROWS_AS(ConicalScattering(nd, h1, cancelling).solve(), hpfem::Cancelled);
  const auto total = problem.total_field(solution, 0, Point<2>(0.2, 0.3));
  const auto incident =
      problem.incident_field(hpfem::mesh::cell_geometry(mesh, 0)->evaluate(Point<2>(0.2, 0.3)).x);
  REQUIRE((total - incident).norm() < 1e-14);
  // errors
  ConicalScatteringSetup bad = setup;
  bad.current = setup.incident;
  REQUIRE_THROWS_AS(ConicalScattering(nd, h1, bad), hpfem::InvalidArgument);
  bad = setup;
  bad.omega = 0.0;
  REQUIRE_THROWS_AS(ConicalScattering(nd, h1, bad), hpfem::InvalidArgument);
  bad = setup;
  bad.materials.set(hpfem::mesh::kNoTag + 1,
                    hpfem::materials::Material{Complex{1.0, 0.0}, Complex{2.0, 0.0}});
  Mesh<2> magnetic = hpfem::mesh::rectangle(3, 3);
  magnetic.set_cell_tag(0, hpfem::mesh::kNoTag + 1);
  const NedelecDofMap<2> nd_m(magnetic, 2);
  const DofMap<2> h1_m(magnetic, 2);
  REQUIRE_THROWS_AS(ConicalScattering(nd_m, h1_m, bad), hpfem::InvalidArgument);
}

TEST_CASE("conical scattering: the scalar E_z path reproduces the block solution",
          "[physics][conical][scalar]") {
  // a dielectric disc under an E_z plane wave (beta = 0): H1 block alone == full system
  const Mesh<2> mesh = hpfem::mesh::square_with_disc(2, 0.25, 1.0, 1.5, 2);
  const NedelecDofMap<2> nd(mesh, 2);
  const DofMap<2> h1(mesh, 2);
  ConicalScatteringSetup setup;
  setup.omega = 5.0 * hpfem::constants::c0;
  setup.beta = 0.0;
  setup.materials.set(2, hpfem::materials::Material::dielectric(1.6));
  setup.incident = conical_plane_wave(ConicalVector(0.0, 0.0, 1.0), Point<3>(5.0, 0.0, 0.0));
  setup.pml = hpfem::pml::PmlBox<2>::uniform(Point<2>(-1.0, -1.0), Point<2>(1.0, 1.0), 0.5, 5.0,
                                             1.0, hpfem::pml::PmlProfile{2, 1e-8});
  setup.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  const auto full = ConicalScattering(nd, h1, setup).solve();
  setup.scalar_ez = true;
  const auto scalar = ConicalScattering(nd, h1, setup).solve();
  REQUIRE(scalar.transverse.norm() == 0.0);
  REQUIRE(full.transverse.norm() < 1e-10 * full.longitudinal.norm());
  REQUIRE((scalar.longitudinal - full.longitudinal).norm() < 1e-10 * full.longitudinal.norm());
  // Bloch-periodic strip with PEC top and bottom at kx != 0: constraints on the H1 block
  const Mesh<2> strip = hpfem::mesh::rectangle(4, 3, Point<2>(0.0, 0.0), Point<2>(1.0, 0.75));
  const NedelecDofMap<2> nd_s(strip, 3);
  const DofMap<2> h1_s(strip, 3);
  ConicalScatteringSetup periodic;
  periodic.omega = 4.0 * hpfem::constants::c0;
  periodic.pec_tags = {box_tag::kYMin, box_tag::kYMax};
  periodic.periodic = {hpfem::assembly::PeriodicPair<2>{
      box_tag::kXMin, box_tag::kXMax, Point<2>(1.0, 0.0), std::exp(hpfem::kI * 1.3)}};
  periodic.current = [](const Point<2>& x) {
    return ConicalVector(0.0, 0.0, std::exp(hpfem::kI * 1.3 * x(0)) * x(1) * (0.75 - x(1)));
  };
  const auto full_p = ConicalScattering(nd_s, h1_s, periodic).solve();
  periodic.scalar_ez = true;
  const auto scalar_p = ConicalScattering(nd_s, h1_s, periodic).solve();
  REQUIRE(full_p.longitudinal.norm() > 0.0);
  REQUIRE((scalar_p.longitudinal - full_p.longitudinal).norm() <
          1e-10 * full_p.longitudinal.norm());
  REQUIRE(scalar_p.transverse.norm() == 0.0);
  // errors: beta != 0, in-plane excitation
  ConicalScatteringSetup bad = setup;
  bad.beta = 0.3;
  REQUIRE_THROWS_AS(ConicalScattering(nd, h1, bad), hpfem::InvalidArgument);
  bad = setup;
  bad.incident = conical_plane_wave(ConicalVector(0.0, 1.0, 0.0), Point<3>(5.0, 0.0, 0.0));
  REQUIRE_THROWS_AS(ConicalScattering(nd, h1, bad), hpfem::InvalidArgument);
}

namespace {

/// Weighted L2 error of the physical field against an exact (E_x, E_y, E_z) on all cells.
Real field_error(const NedelecDofMap<2>& nd, const DofMap<2>& h1, const Vector& e, const Vector& v,
                 Real beta, const std::function<ConicalVector(const Point<2>&)>& exact) {
  const auto& mesh = nd.mesh();
  Real err = 0;
  Real norm = 0;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const auto geometry = hpfem::mesh::cell_geometry(mesh, c);
    const auto rule = hpfem::assembly::simplex_quadrature<2>(2 * nd.cell_order(c) + 4);
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const auto g = geometry->evaluate(rule.points[q]);
      const ConicalVector value =
          hpfem::physics::conical_field_at(nd, h1, e, v, beta, c, rule.points[q]);
      const ConicalVector reference = exact(g.x);
      err += rule.weights[q] * std::abs(g.det) * (value - reference).squaredNorm();
      norm += rule.weights[q] * std::abs(g.det) * reference.squaredNorm();
    }
  }
  return std::sqrt(err / norm);
}

}  // namespace

TEST_CASE("conical scattering: manufactured E_z solution with a current converges in p",
          "[physics][conical]") {
  // E = sin(pi x) sin(pi y) z-hat, PEC box, curl curl E - k0^2 E = f with f_z = (2 pi^2 - k0^2) E_z
  const Real k0 = 2.0;
  const Mesh<2> mesh = hpfem::mesh::rectangle(4, 4);
  const auto exact = [](const Point<2>& x) {
    return ConicalVector(0.0, 0.0,
                         std::sin(std::numbers::pi * x(0)) * std::sin(std::numbers::pi * x(1)));
  };
  Real previous = 1.0;
  for (const int p : {1, 2, 3, 4}) {
    const NedelecDofMap<2> nd(mesh, p);
    const DofMap<2> h1(mesh, p);
    ConicalScatteringSetup setup;
    setup.omega = k0 * hpfem::constants::c0;
    setup.beta = 0.0;
    setup.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
    setup.current = [k0, exact](const Point<2>& x) {
      const Complex f_z = (2 * std::numbers::pi * std::numbers::pi - k0 * k0) * exact(x)(2);
      return ConicalVector(0.0, 0.0, -kI * f_z);  // scaled: f_v = -i f_z
    };
    const ConicalScattering problem(nd, h1, setup);
    const auto solution = problem.solve();
    REQUIRE_FALSE(solution.scattered);
    REQUIRE(solution.transverse.norm() < 1e-12 * solution.longitudinal.norm());
    const Real err = field_error(nd, h1, solution.transverse, solution.longitudinal, 0.0, exact);
    INFO("p = " << p << ", error " << err);
    CHECK(err < 0.3 * previous);
    previous = err;
  }
  CHECK(previous < 1e-4);
}

TEST_CASE("conical scattering: manufactured gradient mode at beta != 0 converges in p",
          "[physics][conical]") {
  // E = grad(psi e^{i beta z}) with psi = sin(pi x) sin(pi y): curl-free, n x E = 0 on the box,
  // so curl curl E - k0^2 E = f with f = -k0^2 E; (E_x, E_y, E_z) = (d_x psi, d_y psi, i beta psi)
  const Real k0 = 2.0;
  const Real beta = 1.3;
  const Mesh<2> mesh = hpfem::mesh::rectangle(4, 4);
  const auto exact = [beta](const Point<2>& x) {
    const Real sx = std::sin(std::numbers::pi * x(0));
    const Real sy = std::sin(std::numbers::pi * x(1));
    const Real cx = std::cos(std::numbers::pi * x(0));
    const Real cy = std::cos(std::numbers::pi * x(1));
    return ConicalVector(std::numbers::pi * cx * sy, std::numbers::pi * sx * cy,
                         kI * beta * sx * sy);
  };
  Real previous = 10.0;
  for (const int p : {1, 2, 3, 4}) {
    const NedelecDofMap<2> nd(mesh, p);
    const DofMap<2> h1(mesh, p);
    ConicalScatteringSetup setup;
    setup.omega = k0 * hpfem::constants::c0;
    setup.beta = beta;
    setup.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
    setup.current = [k0, exact](const Point<2>& x) {
      const ConicalVector e = exact(x);
      return ConicalVector(-k0 * k0 * e(0), -k0 * k0 * e(1), -kI * (-k0 * k0 * e(2)));
    };
    const ConicalScattering problem(nd, h1, setup);
    const auto solution = problem.solve();
    const Real err = field_error(nd, h1, solution.transverse, solution.longitudinal, beta, exact);
    INFO("p = " << p << ", error " << err);
    CHECK(err < 0.3 * previous);
    previous = err;
  }
  CHECK(previous < 1e-3);
}

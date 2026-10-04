// Waveguide ports in 2D: the port modes of the PEC parallel plate are the cosine modes with
// beta_n = sqrt(k0^2 - (n pi / a)^2), the TM mode of the slab solves the transcendental
// equation, a straight guide transmits every propagating mode with |S21| = 1 and the phase
// e^{i beta L} without reflection or mode conversion, S is symmetric and unitary, and the
// setup rejects what the ports cannot do.
#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/waveguide_port.hpp"

using Catch::Approx;
using hpfem::as_size;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::Mesh;
using hpfem::physics::PortModes;
using hpfem::physics::s_parameters;
using hpfem::physics::Scattering;
using hpfem::physics::ScatteringSetup;
using hpfem::physics::SParameters;
using hpfem::physics::WaveguidePort;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr hpfem::mesh::Tag kCore = 2;

/// Strip [0, L] x [y0, y1]; ports on x = 0 (kXMin) and x = L (kXMax), PEC on the other sides.
ScatteringSetup<2> strip_setup(Real k0, Index num_modes) {
  ScatteringSetup<2> setup;
  setup.omega = k0 * hpfem::constants::c0;
  setup.pec_tags = {box_tag::kYMin, box_tag::kYMax};
  setup.ports = {WaveguidePort{box_tag::kXMin, num_modes, {}},
                 WaveguidePort{box_tag::kXMax, num_modes, {}}};
  return setup;
}

/// Even TM mode of the symmetric slab: kappa tan(kappa d / 2) = (eps_core / eps_clad) gamma
/// (no pole of the tangent below n_core for these parameters, so plain bisection works).
Real slab_tm_even(Real k0, Real d, Real n_core, Real n_clad) {
  const auto f = [&](Real n) {
    const Real kappa = k0 * std::sqrt(n_core * n_core - n * n);
    const Real gamma = k0 * std::sqrt(n * n - n_clad * n_clad);
    return kappa * std::tan(kappa * d / 2) - (n_core * n_core) / (n_clad * n_clad) * gamma;
  };
  Real lo = n_clad + 1e-12;
  Real hi = n_core - 1e-12;
  for (int i = 0; i < 200; ++i) {
    const Real mid = 0.5 * (lo + hi);
    (f(lo) * f(mid) <= 0 ? hi : lo) = mid;
  }
  return 0.5 * (lo + hi);
}

Mesh<2> slab_mesh(Index nx, Index ny, Real length, Real half_width, Real d) {
  Mesh<2> m =
      hpfem::mesh::rectangle(nx, ny, Point<2>(0.0, -half_width), Point<2>(length, half_width));
  for (Index c = 0; c < m.num_cells(); ++c) {
    if (std::abs(hpfem::mesh::affine_map(m, c).centroid()(1)) < d / 2) m.set_cell_tag(c, kCore);
  }
  return m;
}

/// Index of the channel (port, mode) in an S-matrix.
Index channel(const SParameters& s, Index port, Index mode) {
  for (std::size_t i = 0; i < s.channels.size(); ++i) {
    if (s.channels[i].port == port && s.channels[i].mode == mode) return static_cast<Index>(i);
  }
  return hpfem::kInvalidIndex;
}

}  // namespace

TEST_CASE("port modes of the PEC parallel plate are the cosine modes", "[physics][port]") {
  const Real a = 1.0;
  const Real k0 = 5.0;  // n = 0, 1 propagate, n = 2 is evanescent
  const Mesh<2> mesh = hpfem::mesh::rectangle(6, 6, Point<2>(0.0, 0.0), Point<2>(1.0, a));
  const NedelecDofMap<2> dofs(mesh, 4);
  hpfem::materials::MaterialMap materials;
  const PortModes<2> port(dofs, box_tag::kXMin, materials, k0 * hpfem::constants::c0, 3);
  REQUIRE(port.num_modes() == 3);
  REQUIRE(port.length() == Approx(a));
  REQUIRE(port.normal()(0) == Approx(-1.0));
  REQUIRE(port.tangent()(1) == Approx(1.0));  // s increases along t' = (n_y, -n_x) = +y
  for (int n = 0; n < 3; ++n) {
    const Real beta_squared = k0 * k0 - (n * std::numbers::pi / a) * (n * std::numbers::pi / a);
    const auto& mode = port.modes()[as_size(n)];
    if (beta_squared > 0) {
      REQUIRE(mode.propagating);
      REQUIRE(mode.beta.real() == Approx(std::sqrt(beta_squared)).epsilon(1e-6));
      REQUIRE(mode.beta.imag() == 0.0);
      REQUIRE(mode.power > 0.0);
    } else {
      REQUIRE_FALSE(mode.propagating);
      REQUIRE(mode.beta.imag() == Approx(std::sqrt(-beta_squared)).epsilon(1e-5));
      REQUIRE(mode.power == 0.0);
    }
    REQUIRE(std::abs(mode.effective_index - mode.beta / k0) < 1e-12);
    // the profile is c cos(n pi s / a); the sign convention (E_t along +y positive at the
    // midpoint, positive slope for odd modes) gives c < 0 for n = 0 and c > 0 for n = 1, 2
    const Real scale = port.profile(n, 0.0);
    REQUIRE((n == 0 ? scale < 0 : scale > 0));
    for (const Real s : {0.17, 0.5, 0.83}) {
      REQUIRE(port.profile(n, s) ==
              Approx(scale * std::cos(n * std::numbers::pi * s / a))
                  .margin(2e-3 * std::abs(scale)));  // eigenfunction error O(h^(p+1))
    }
    // the tangential trace follows the profile with -beta / (omega eps0)
    const Complex ratio = port.trace(n, 0.3) / port.profile(n, 0.3);
    REQUIRE(std::abs(ratio + mode.beta / (k0 * hpfem::constants::c0 * hpfem::constants::eps0)) <
            1e-12 * std::abs(ratio));
  }
  REQUIRE(port.dofs().size() > 0);
  for (const Index d : port.dofs()) REQUIRE(d < dofs.num_dofs());
  // the opposite port sees the same physical field: E_y of the TEM mode positive on both
  const PortModes<2> other(dofs, box_tag::kXMax, materials, k0 * hpfem::constants::c0, 1);
  REQUIRE(other.tangent()(1) == Approx(-1.0));
  REQUIRE(other.profile(0, 0.5) > 0);  // sigma = -1: h > 0 here, h < 0 on the left port
  REQUIRE(port.profile(0, 0.5) < 0);
}

TEST_CASE("port modes of the slab reproduce the TM dispersion relation", "[physics][port]") {
  const Real d = 1.0;
  const Real n_core = 2.0;
  const Real k0 = 1.5;
  const Mesh<2> mesh = slab_mesh(4, 48, 1.0, 6.0, d);
  const NedelecDofMap<2> dofs(mesh, 4);
  hpfem::materials::MaterialMap materials;
  materials.set(kCore, hpfem::materials::Material::dielectric(n_core));
  const PortModes<2> port(dofs, box_tag::kXMax, materials, k0 * hpfem::constants::c0, 2);
  const Real reference = slab_tm_even(k0, d, n_core, 1.0);
  REQUIRE(port.modes()[0].propagating);
  REQUIRE(port.modes()[0].effective_index.real() == Approx(reference).epsilon(1e-6));
  REQUIRE(port.profile(0, 6.0) > 0);  // even mode, positive at the centre (sigma = -1)
  REQUIRE(port.profile(0, 6.0) > port.profile(0, 8.0));
  REQUIRE(port.profile(0, 8.0) > 0);
}

TEST_CASE("straight parallel plate: S-parameters are transmission phases without reflection",
          "[physics][port]") {
  const Real a = 1.0;
  const Real length = 1.0;
  const Real k0 = 5.0;
  const Mesh<2> mesh = hpfem::mesh::rectangle(6, 6, Point<2>(0.0, 0.0), Point<2>(length, a));
  const NedelecDofMap<2> dofs(mesh, 4);
  const auto s = s_parameters<2>(dofs, strip_setup(k0, 3));
  REQUIRE(s.channels.size() == 4);  // two propagating modes per port
  REQUIRE(s.s.rows() == 4);
  const std::vector<Real> betas = {k0, std::sqrt(k0 * k0 - std::numbers::pi * std::numbers::pi)};
  for (Index j = 0; j < 4; ++j) {
    const auto& source = s.channels[as_size(j)];
    for (Index i = 0; i < 4; ++i) {
      const auto& target = s.channels[as_size(i)];
      Complex expected{0.0, 0.0};
      if (target.mode == source.mode && target.port != source.port) {
        expected = std::exp(Complex{0.0, 1.0} * betas[as_size(source.mode)] * length);
      }
      INFO("S(" << i << ", " << j << ") = " << s.s(i, j) << ", expected " << expected);
      CHECK(std::abs(s.s(i, j) - expected) < 2e-5);
    }
  }
  // symmetric (reciprocity) and unitary (lossless)
  const hpfem::Matrix identity = hpfem::Matrix::Identity(4, 4);
  CHECK((s.s - s.s.transpose()).norm() < 1e-5);
  CHECK((s.s.adjoint() * s.s - identity).norm() < 1e-4);
}

TEST_CASE("straight slab guide: the guided mode passes with |S21| = 1 and the phase beta L",
          "[physics][port]") {
  const Real d = 1.0;
  const Real n_core = 2.0;
  const Real k0 = 1.5;
  const Real length = 2.0;
  const Mesh<2> mesh = slab_mesh(8, 24, length, 3.0, d);
  const NedelecDofMap<2> dofs(mesh, 3);
  ScatteringSetup<2> setup = strip_setup(k0, 2);
  setup.materials.set(kCore, hpfem::materials::Material::dielectric(n_core));
  // single excitation through the setup's amplitudes
  setup.ports[0].incident = {Complex{1.0, 0.0}};
  const Scattering<2> problem(dofs, setup);
  REQUIRE(problem.port_modes(0).modes()[0].propagating);
  // the closed cross-section also has propagating box modes of the cladding
  REQUIRE(problem.port_modes(0).modes()[1].propagating);
  const auto solution = problem.solve();
  const auto c = problem.port_coefficients(solution);
  REQUIRE(c.size() == 2);
  REQUIRE(c[0].incoming[0] == Complex{1.0, 0.0});
  const Real beta = problem.port_modes(1).modes()[0].beta.real();
  const Complex expected = std::exp(Complex{0.0, 1.0} * beta * length);
  CHECK(std::abs(c[1].outgoing[0] - expected) < 2e-3);
  CHECK(std::abs(c[0].outgoing[0]) < 2e-3);
  CHECK(std::abs(c[1].outgoing[1]) < 1e-2);  // no conversion into the box mode
  const auto s = s_parameters<2>(dofs, setup);
  REQUIRE(s.channels.size() == 4);
  const Index in = channel(s, 0, 0);
  const Index out = channel(s, 1, 0);
  CHECK(std::abs(s.s(out, in) - expected) < 2e-3);
  CHECK(std::abs(s.s(in, out) - s.s(out, in)) < 1e-6);
  CHECK(std::abs(std::norm(s.s(out, in)) + std::norm(s.s(in, in)) - 1.0) < 1e-3);
}

TEST_CASE("waveguide ports reject what they cannot do", "[physics][port]") {
  const Mesh<2> mesh = hpfem::mesh::rectangle(2, 2);
  const NedelecDofMap<2> dofs(mesh, 2);
  hpfem::materials::MaterialMap materials;
  REQUIRE_THROWS_AS(PortModes<2>(dofs, 99, materials, 1.0, 1), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(PortModes<2>(dofs, box_tag::kXMin, materials, 0.0, 1), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(PortModes<2>(dofs, box_tag::kXMin, materials, 1.0, 0), hpfem::InvalidArgument);
  // a lossy material on the port
  const Mesh<2> slab = slab_mesh(2, 8, 1.0, 1.0, 0.5);
  const NedelecDofMap<2> slab_dofs(slab, 2);
  hpfem::materials::MaterialMap lossy;
  lossy.set(kCore, hpfem::materials::Material{Complex{4.0, 0.1}, Complex{1.0, 0.0}});
  REQUIRE_THROWS_AS(PortModes<2>(slab_dofs, box_tag::kXMin, lossy, 1.0, 1), hpfem::InvalidArgument);
  // ports need the total-field formulation
  ScatteringSetup<2> setup = strip_setup(1.0, 1);
  setup.formulation = hpfem::physics::Formulation::kScatteredField;
  setup.incident = hpfem::physics::plane_wave<2>(
      hpfem::assembly::ComplexVector<2>(Complex{0.0, 0.0}, Complex{1.0, 0.0}), Point<2>(1.0, 0.0));
  REQUIRE_THROWS_AS(Scattering<2>(dofs, setup), hpfem::InvalidArgument);
  // 3D ports are not implemented yet
  const Mesh<3> box = hpfem::mesh::box(1, 1, 1);
  const NedelecDofMap<3> dofs3(box, 1);
  REQUIRE_THROWS_AS(PortModes<3>(dofs3, box_tag::kXMin, materials, 1.0, 1), hpfem::InvalidArgument);
  ScatteringSetup<2> none = strip_setup(1.0, 1);
  none.ports.clear();
  REQUIRE_THROWS_AS(s_parameters<2>(dofs, none), hpfem::InvalidArgument);
}

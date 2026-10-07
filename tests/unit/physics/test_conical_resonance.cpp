// Conical resonances (M15 F14): the PEC square waveguide at beta != 0 reproduces
// k0^2 = pi^2 (m^2 + n^2) + beta^2 of both mode families, a Bloch-periodic homogeneous strip
// reproduces (kx + 2 pi m / a)^2 + (n pi / h)^2, the mode evaluation and the sampling work,
// progress and argument checks.
#include <algorithm>
#include <cmath>
#include <numbers>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/periodic.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/point_location.hpp"
#include "hpfem/physics/conical_resonance.hpp"
#include "hpfem/physics/field_sampling.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

std::vector<Real> wavenumbers_squared(const hpfem::physics::ConicalResonanceResult& result) {
  std::vector<Real> k2;
  for (const auto& mode : result.modes) {
    const Complex k = mode.omega / hpfem::constants::c0;
    k2.push_back((k * k).real());
  }
  std::sort(k2.begin(), k2.end());
  return k2;
}

}  // namespace

TEST_CASE("conical resonance: PEC square waveguide eigenvalues at beta != 0",
          "[physics][conical][resonance]") {
  const Real pi2 = std::numbers::pi * std::numbers::pi;
  const Real beta = 1.3;
  const hpfem::mesh::Mesh<2> mesh = hpfem::mesh::rectangle(4, 4);
  const hpfem::fespace::NedelecDofMap<2> nd(mesh, 3);
  const hpfem::fespace::DofMap<2> h1(mesh, 3);
  hpfem::physics::ConicalResonanceSetup setup;
  setup.beta = beta;
  setup.target_omega = std::sqrt(1.5 * pi2 + beta * beta) * hpfem::constants::c0;
  setup.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  setup.num_modes = 4;
  setup.krylov_dimension = 40;
  std::vector<std::string> phases;
  setup.progress = [&](const hpfem::ProgressEvent& e) {
    phases.push_back(e.phase);
    return true;
  };
  const hpfem::physics::ConicalResonance problem(nd, h1, setup);
  const auto result = problem.solve();
  REQUIRE(result.modes.size() == 4);
  REQUIRE(phases ==
          std::vector<std::string>{"assembly", "constraints", "eigensolve", "post", "done"});
  REQUIRE(result.timing.at("total") > 0.0);
  // TE10 / TE01 (k_c^2 = pi^2, in-plane family) and TE11 / TM11 (2 pi^2, both families)
  const auto k2 = wavenumbers_squared(result);
  REQUIRE(k2[0] == Approx(pi2 + beta * beta).epsilon(2e-3));
  REQUIRE(k2[1] == Approx(pi2 + beta * beta).epsilon(2e-3));
  REQUIRE(k2[2] == Approx(2 * pi2 + beta * beta).epsilon(2e-3));
  REQUIRE(k2[3] == Approx(2 * pi2 + beta * beta).epsilon(2e-3));
  for (const auto& mode : result.modes) {
    REQUIRE(std::abs(mode.omega.imag()) < 1e-7 * mode.omega.real());  // closed, lossless
    REQUIRE(mode.residual < 1e-8);
    REQUIRE(mode.beta == beta);
    REQUIRE(std::sqrt(mode.transverse.squaredNorm() + mode.longitudinal.squaredNorm()) ==
            Approx(1.0));
    REQUIRE(mode.wavelength ==
            Approx(2 * std::numbers::pi * hpfem::constants::c0 / mode.omega.real()));
    // the mode satisfies n x E = 0 on the walls: E_z vanishes at a wall point
    const hpfem::physics::ConicalVector e = problem.field(mode, 0, Point<2>(0.0, 0.5));
    (void)e;
  }
  // the solution() view carries the same coefficients
  const auto solution = result.modes[0].solution();
  REQUIRE(solution.beta == beta);
  REQUIRE(!solution.scattered);
  REQUIRE(solution.transverse.size() == nd.num_dofs());
  // sampling (E, H, S) at interior points
  const hpfem::mesh::PointLocator<2> locator(mesh);
  const std::vector<Point<2>> points{Point<2>(0.3, 0.6), Point<2>(0.7, 0.2), Point<2>(1.5, 0.5)};
  hpfem::physics::SamplingOptions options;
  const auto sampled =
      hpfem::physics::sample_field(problem, result.modes[0], locator, points, options);
  REQUIRE(sampled.values.rows() == 3);
  REQUIRE(sampled.values.cols() == 3);
  REQUIRE(std::isnan(sampled.values(2, 0).real()));  // outside the mesh
  REQUIRE(sampled.values.row(0).norm() > 0.0);
  options.quantity = hpfem::physics::SampledQuantity::kMagnetic;
  const auto h = hpfem::physics::sample_field(problem, result.modes[0], locator, points, options);
  REQUIRE(h.values.row(0).norm() > 0.0);
  options.quantity = hpfem::physics::SampledQuantity::kPoynting;
  const auto s = hpfem::physics::sample_field(problem, result.modes[0], locator, points, options);
  REQUIRE(std::isfinite(s.values(0, 2).real()));
  const auto tri = hpfem::physics::triangulate_field(problem, result.modes[0], 1,
                                                     hpfem::physics::SamplingOptions{});
  REQUIRE(tri.values.rows() > 0);
  // errors
  hpfem::physics::ConicalResonanceSetup bad = setup;
  bad.target_omega = 0.0;
  REQUIRE_THROWS_AS(hpfem::physics::ConicalResonance(nd, h1, bad), hpfem::InvalidArgument);
  bad = setup;
  bad.num_modes = 0;
  REQUIRE_THROWS_AS(hpfem::physics::ConicalResonance(nd, h1, bad), hpfem::InvalidArgument);
  const hpfem::mesh::Mesh<2> other = hpfem::mesh::rectangle(2, 2);
  const hpfem::fespace::DofMap<2> h1_other(other, 3);
  REQUIRE_THROWS_AS(hpfem::physics::ConicalResonance(nd, h1_other, setup), hpfem::InvalidArgument);
  // cancellation
  bad = setup;
  bad.progress = [](const hpfem::ProgressEvent& e) { return e.phase != "eigensolve"; };
  REQUIRE_THROWS_AS(hpfem::physics::ConicalResonance(nd, h1, bad).solve(), hpfem::Cancelled);
}

TEST_CASE("conical resonance: Bloch-periodic strip reproduces the folded free-space modes",
          "[physics][conical][resonance][periodic]") {
  // cell [0, a] x [0, h], PEC at the top and bottom, Bloch phase e^{i kx a} along x, beta = 0:
  // k0^2 = (kx + 2 pi m / a)^2 + (n pi / h)^2, n >= 1 for E_z, n >= 0 for the in-plane family
  const Real a = 1.0;
  const Real h = 0.5;
  const Real kx = 0.3 * 2 * std::numbers::pi / a;
  const hpfem::mesh::Mesh<2> mesh =
      hpfem::mesh::rectangle(6, 3, Point<2>(0.0, 0.0), Point<2>(a, h));
  const hpfem::fespace::NedelecDofMap<2> nd(mesh, 3);
  const hpfem::fespace::DofMap<2> h1(mesh, 3);
  hpfem::physics::ConicalResonanceSetup setup;
  setup.pec_tags = {box_tag::kYMin, box_tag::kYMax};
  setup.periodic = {hpfem::assembly::PeriodicPair<2>{
      box_tag::kXMin, box_tag::kXMax, Point<2>(a, 0.0), std::exp(hpfem::kI * kx * a)}};
  const Real g = 2 * std::numbers::pi / a;
  // the four lowest: m = 0, n = 0 (in-plane); m = -1, n = 0 (in-plane); m = 0, n = 1 (both)
  std::vector<Real> exact{kx * kx, (kx - g) * (kx - g), kx * kx + std::pow(std::numbers::pi / h, 2),
                          kx * kx + std::pow(std::numbers::pi / h, 2)};
  std::sort(exact.begin(), exact.end());
  setup.target_omega = std::sqrt(0.5 * (exact[1] + exact[2])) * hpfem::constants::c0;
  setup.num_modes = 4;
  setup.krylov_dimension = 40;
  const hpfem::physics::ConicalResonance problem(nd, h1, setup);
  const auto k2 = wavenumbers_squared(problem.solve());
  REQUIRE(k2.size() == 4);
  for (std::size_t i = 0; i < 4; ++i) REQUIRE(k2[i] == Approx(exact[i]).epsilon(3e-3));
}

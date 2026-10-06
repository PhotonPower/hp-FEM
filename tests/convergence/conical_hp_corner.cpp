// hp-adaptivity for the conical solver (M15 F1): manufactured gradient mode with a re-entrant
// PEC corner. The L-shaped domain [-1, 1]^2 minus the quadrant x > 0, y < 0 with PEC walls
// carries E = nabla(psi e^{i beta z}), psi = (1 - x^2)(1 - y^2) rho^{2/3} sin(2 theta / 3)
// (polar coordinates about the corner at the origin, theta = 0 along the face y = 0, x > 0,
// theta = 3 pi / 2 along the face x = 0, y < 0), which vanishes on every wall, is curl-free
// and solves curl curl E - k^2 E = -k^2 E at beta = 1.3 with |E| ~ rho^{-1/3} at the corner.
// The loop SOLVE - ESTIMATE (conical residual estimator) - MARK (Doerfler 0.5) - DECIDE (error
// prediction) - REFINE (hanging nodes) must converge exponentially in N^{1/3}.
#include <cmath>
#include <numbers>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/adaptivity/marking.hpp"
#include "hpfem/adaptivity/prediction.hpp"
#include "hpfem/adaptivity/refinement.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/adaptive_mesh.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/physics/conical_scattering.hpp"

using hpfem::as_size;
using hpfem::Complex;
using hpfem::Index;
using hpfem::kI;
using hpfem::Point;
using hpfem::Real;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::AdaptiveMesh;
using hpfem::mesh::Mesh;
using hpfem::physics::ConicalScattering;
using hpfem::physics::ConicalScatteringSetup;
using hpfem::physics::ConicalVector;

namespace {

constexpr Real kWavenumber = 1.0;
constexpr Real kBeta = 1.3;
constexpr hpfem::mesh::Tag kWall = 9;

struct Potential {
  Real value, d_x, d_y;
};

Potential potential(const Point<2>& p) {
  const Real x = p(0), y = p(1);
  const Real rho = std::hypot(x, y);
  Real theta = std::atan2(y, x);
  if (theta < 0) theta += 2 * std::numbers::pi;  // the domain covers [0, 3 pi / 2]
  const Real nu = 2.0 / 3.0;
  const Real s = rho > 0 ? std::pow(rho, nu) * std::sin(nu * theta) : 0.0;
  const Real s_rho = rho > 0 ? nu * std::pow(rho, nu - 1.0) * std::sin(nu * theta) : 0.0;
  const Real s_theta_over_rho = rho > 0 ? nu * std::pow(rho, nu - 1.0) * std::cos(nu * theta) : 0.0;
  const Real cos_t = rho > 0 ? x / rho : 1.0;
  const Real sin_t = rho > 0 ? y / rho : 0.0;
  const Real s_x = s_rho * cos_t - s_theta_over_rho * sin_t;
  const Real s_y = s_rho * sin_t + s_theta_over_rho * cos_t;
  const Real a = (1 - x * x) * (1 - y * y);
  const Real a_x = -2 * x * (1 - y * y);
  const Real a_y = -2 * y * (1 - x * x);
  return {a * s, a_x * s + a * s_x, a_y * s + a * s_y};
}

/// Physical exact field (E_x, E_y, E_z) = (d_x psi, d_y psi, i beta psi).
ConicalVector exact(const Point<2>& x) {
  const Potential p = potential(x);
  return ConicalVector(Complex{p.d_x, 0.0}, Complex{p.d_y, 0.0}, kI * kBeta * p.value);
}

/// Scaled source f = -k^2 E: (f_x, f_y, f_v = -i f_z) = -k^2 (d_x psi, d_y psi, beta psi).
ConicalVector source(const Point<2>& x) {
  const Potential p = potential(x);
  return ConicalVector(-kWavenumber * kWavenumber * p.d_x, -kWavenumber * kWavenumber * p.d_y,
                       -kWavenumber * kWavenumber * kBeta * p.value);
}

Mesh<2> l_shape(Index n) {
  const Mesh<2> full =
      hpfem::mesh::rectangle(2 * n, 2 * n, Point<2>(-1.0, -1.0), Point<2>(1.0, 1.0));
  Mesh<2> out =
      hpfem::mesh::extract<2>(full, [](const Point<2>& c) { return !(c(0) > 0.0 && c(1) < 0.0); });
  for (const Index f : out.boundary_facets()) out.set_facet_tag(f, kWall);
  return out;
}

bool touches_corner(const Mesh<2>& mesh, Index c) {
  for (const Index v : mesh.cell_vertices(c)) {
    if (mesh.vertex(v).norm() < 1e-12) return true;
  }
  return false;
}

struct Step {
  Index dofs;
  Real error;
  Real estimate;
  int max_order;
};

template <class F>
Real slope(const std::vector<Step>& steps, Index from, F f) {
  Real sx = 0, sy = 0, sxx = 0, sxy = 0, n = 0;
  for (const Step& s : steps) {
    if (s.dofs < from) continue;
    const Real x = f(static_cast<Real>(s.dofs));
    const Real y = std::log(s.error);
    sx += x;
    sy += y;
    sxx += x * x;
    sxy += x * y;
    n += 1;
  }
  return (n * sxy - sx * sy) / (n * sxx - sx * sx);
}

}  // namespace

TEST_CASE("conical re-entrant corner: hp-adaptivity converges exponentially in N^(1/3)",
          "[convergence][adaptivity][hp][conical]") {
  AdaptiveMesh<2> adaptive(l_shape(2));
  std::vector<int> orders(as_size(adaptive.mesh().num_cells()), 1);
  std::vector<Step> steps;
  std::vector<Real> predicted;
  Index corner_h = 0, corner_p = 0;
  fmt::print(
      "\nconical re-entrant corner (beta = {}), hp-adaptivity (Doerfler 0.5, error prediction)\n"
      "{:>5} {:>8} {:>12} {:>12} {:>8} {:>6}\n",
      kBeta, "step", "DoF", "error", "eta", "eta/err", "max p");
  for (int step = 0; step < 40; ++step) {
    const Mesh<2> mesh = adaptive.mesh();
    const NedelecDofMap<2> nd(mesh, orders);
    const DofMap<2> h1(mesh, orders);
    ConicalScatteringSetup setup;
    setup.omega = kWavenumber * hpfem::constants::c0;
    setup.beta = kBeta;
    setup.pec_tags = {kWall};
    setup.current = source;
    const ConicalScattering problem(nd, h1, setup);
    const auto solution = problem.solve();
    const auto e = problem.error(solution, exact);
    const auto estimate = problem.estimate(solution);
    const Real error = std::hypot(e.l2, e.curl);
    const Index dofs = static_cast<Index>(problem.free_dofs().size());
    steps.push_back({dofs, error, estimate.total(), nd.max_order()});
    fmt::print("{:>5} {:>8} {:>12.3e} {:>12.3e} {:>8.2f} {:>6}\n", step, dofs, error,
               estimate.total(), estimate.total() / error, nd.max_order());
    if (dofs > 30000 || error < 1e-7) break;
    const auto marked = hpfem::adaptivity::dorfler_marking(estimate.indicators, 0.5);
    const auto decision =
        hpfem::adaptivity::hp_decide_by_prediction(estimate.indicators, predicted, marked);
    for (const Index c : decision.p_marked) corner_p += touches_corner(mesh, c) ? 1 : 0;
    for (const Index c : decision.h_marked) corner_h += touches_corner(mesh, c) ? 1 : 0;
    const auto hp =
        hpfem::adaptivity::hp_refine<2>(adaptive, orders, decision.h_marked, decision.p_marked);
    predicted = hpfem::adaptivity::predict_indicators(estimate.indicators, orders, hp);
    orders = hp.orders;
  }
  REQUIRE(steps.size() >= 10);
  const Real b = -slope(steps, steps[steps.size() - 10].dofs, [](Real n) { return std::cbrt(n); });
  const Real algebraic = slope(steps, 4000, [](Real n) { return std::log(n); });
  fmt::print("corner cells: {} h-decisions, {} p-decisions\n", corner_h, corner_p);
  fmt::print(
      "fit error ~ exp(-b N^(1/3)) over the last 10 steps: b = {:.3f}; algebraic slope for N >= "
      "4000: {:.2f}\n",
      b, algebraic);
  CHECK(corner_h >= 3 * corner_p);
  CHECK(b > 0.2);
  CHECK(algebraic < -1.2);
  CHECK(steps.back().error < 1e-3 * steps.front().error);
  for (const Step& s : steps) {
    CHECK(s.estimate / s.error > 0.1);
    CHECK(s.estimate / s.error < 50.0);
  }
}

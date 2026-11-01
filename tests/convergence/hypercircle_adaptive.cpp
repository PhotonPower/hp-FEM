// Hypercircle bound in the hp-adaptive loop (docs/theory/error-estimation.md#dual-formulation-
// and-guaranteed-bounds): the coercive problem curl curl E + E = f on the L-shaped domain with
// PEC walls and E = (sin(pi y), sin(pi x)) + grad(phi b), phi = r^{2/3} sin(2 theta / 3) the
// corner singularity of the Laplacian, b = (1 - x²)(1 - y²) the cut-off that makes the
// tangential trace vanish on the outer walls. The gradient part is the strongest Maxwell corner
// singularity (curl-free, E ~ r^{-1/3}); the smooth part gives a non-trivial dual field
// sigma = curl E. The loop SOLVE (primal and dual, hanging nodes) → ESTIMATE (hypercircle) →
// MARK (Dörfler) → DECIDE (error prediction) → REFINE must keep the bound eta >= ‖E - E_h‖ on
// every mesh — a bound without constants, with hanging nodes — and converge exponentially,
// error ~ exp(-b N^{1/3}).
#include <cmath>
#include <numbers>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/adaptivity/hypercircle.hpp"
#include "hpfem/adaptivity/marking.hpp"
#include "hpfem/adaptivity/prediction.hpp"
#include "hpfem/adaptivity/refinement.hpp"
#include "hpfem/assembly/dirichlet.hpp"
#include "hpfem/assembly/hanging_constraints.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/adaptive_mesh.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/solvers/linear_solver.hpp"

using hpfem::as_size;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::assembly::ComplexCurl;
using hpfem::assembly::ComplexVector;
using hpfem::assembly::MaxwellForm;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::AdaptiveMesh;
using hpfem::mesh::Mesh;

namespace {

constexpr Real kPi = std::numbers::pi;
constexpr Real kSquared = -1.0;
constexpr Real kTheta = 0.5;       // Dörfler bulk fraction
constexpr Index kMaxDofs = 12000;  // stop after the first step beyond this

Mesh<2> l_shape(Index n) {
  const Mesh<2> square =
      hpfem::mesh::rectangle(2 * n, 2 * n, Point<2>(-1.0, -1.0), Point<2>(1.0, 1.0));
  return hpfem::mesh::extract<2>(square, [](const Point<2>& x) { return !(x(0) > 0 && x(1) < 0); });
}

/// grad(phi b) with phi = r^{2/3} sin(2 theta / 3) (theta in [0, 2 pi), the walls of the
/// re-entrant corner at theta = 0 and 3 pi / 2) and b = (1 - x²)(1 - y²).
Point<2> singular_gradient(const Point<2>& x) {
  const Real r = x.norm();
  Real theta = std::atan2(x(1), x(0));
  if (theta < 0) theta += 2 * kPi;
  const Real phi = std::pow(r, 2.0 / 3.0) * std::sin(2 * theta / 3);
  const Real dr = (2.0 / 3.0) * std::pow(r, -1.0 / 3.0) * std::sin(2 * theta / 3);
  const Real dt = (2.0 / 3.0) * std::pow(r, -1.0 / 3.0) * std::cos(2 * theta / 3);
  const Point<2> grad_phi(dr * std::cos(theta) - dt * std::sin(theta),
                          dr * std::sin(theta) + dt * std::cos(theta));
  const Real b = (1 - x(0) * x(0)) * (1 - x(1) * x(1));
  const Point<2> grad_b(-2 * x(0) * (1 - x(1) * x(1)), -2 * x(1) * (1 - x(0) * x(0)));
  return b * grad_phi + phi * grad_b;
}

ComplexVector<2> smooth(const Point<2>& x) {
  return ComplexVector<2>(std::sin(kPi * x(1)), std::sin(kPi * x(0)));
}

ComplexVector<2> exact(const Point<2>& x) {
  return smooth(x) + singular_gradient(x).cast<hpfem::Complex>();
}

ComplexCurl<2> exact_curl(const Point<2>& x) {
  return ComplexCurl<2>(kPi * (std::cos(kPi * x(0)) - std::cos(kPi * x(1))));
}

MaxwellForm<2> form() {
  MaxwellForm<2> out;
  // f = curl curl E - k² E = (pi² + 1) E_smooth + grad(phi b) (the gradient is curl-free)
  out.source = [](const Point<2>& x) {
    return ComplexVector<2>((kPi * kPi + 1) * smooth(x) +
                            singular_gradient(x).cast<hpfem::Complex>());
  };
  return out;
}

/// Primal Galerkin solution with PEC on the whole boundary and the hanging nodes constrained.
Vector primal_solution(const NedelecDofMap<2>& dofs, const MaxwellForm<2>& primal) {
  auto system =
      hpfem::assembly::assemble_maxwell_operator<2>(dofs, [&](Index) { return primal; }, kSquared);
  const hpfem::fespace::Constraints c = hpfem::assembly::hanging_constraints(dofs);
  auto [matrix, load] = c.reduce(system.matrix, system.rhs);
  const auto& boundary = dofs.mesh().boundary_facets();
  const std::vector<Index> facets(boundary.begin(), boundary.end());
  hpfem::assembly::DirichletData data;
  for (const Index dof : hpfem::assembly::homogeneous_dirichlet(dofs, facets).dofs) {
    if (!c.is_constrained(dof)) data.dofs.push_back(c.reduced_index(dof));
  }
  data.values = Vector::Zero(data.size());
  hpfem::assembly::apply_dirichlet(matrix, load, data);
  return c.expand(hpfem::solvers::solve_direct(matrix, load));
}

struct Step {
  Index dofs;
  Real error;  ///< energy error (‖curl e‖² + ‖e‖²)^(1/2)
  Real eta;    ///< hypercircle bound
};

/// Least-squares slope of log(error) against f(N) over the steps with N >= from.
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

TEST_CASE("L-shape: hp-adaptivity driven by the hypercircle bound, guaranteed in every step",
          "[convergence][adaptivity][hypercircle]") {
  AdaptiveMesh<2> adaptive(l_shape(2));
  std::vector<int> orders(as_size(adaptive.mesh().num_cells()), 1);
  std::vector<Step> steps;
  std::vector<Real> predicted;  // empty in the first step: h-refinement
  const MaxwellForm<2> primal = form();
  const auto factory = [&](Index) { return primal; };
  const hpfem::adaptivity::PredictionOptions prediction;
  fmt::print(
      "\nL-shape, coercive, hp-adaptivity on the hypercircle indicators (Doerfler {}, dual "
      "order p + 1)\n{:>5} {:>8} {:>8} {:>12} {:>12} {:>8} {:>4} {:>4} {:>6}\n",
      kTheta, "step", "DoF", "dual DoF", "error", "eta", "eta/err", "#h", "#p", "max p");
  for (int step = 0; step < 40; ++step) {
    const Mesh<2> mesh = adaptive.mesh();
    const NedelecDofMap<2> dofs(mesh, orders);
    std::vector<int> dual_orders(orders);
    for (int& p : dual_orders) ++p;
    const DofMap<2> dual(mesh, dual_orders);
    const Vector e_h = primal_solution(dofs, primal);
    const Vector sigma_h = hpfem::adaptivity::dual_solution<2>(dual, factory, kSquared);
    const auto estimate =
        hpfem::adaptivity::hypercircle_estimate<2>(dofs, e_h, dual, sigma_h, factory, kSquared);
    const auto errors = hpfem::assembly::hcurl_error<2>(dofs, e_h, exact, exact_curl);
    const Real error = std::sqrt(errors.curl * errors.curl - kSquared * errors.l2 * errors.l2);
    const auto marked = hpfem::adaptivity::dorfler_marking(estimate.indicators, kTheta);
    REQUIRE(!marked.empty());
    const auto decision =
        hpfem::adaptivity::hp_decide_by_prediction(estimate.indicators, predicted, marked);
    steps.push_back({dofs.num_dofs(), error, estimate.total()});
    fmt::print("{:>5} {:>8} {:>8} {:>12.3e} {:>12.3e} {:>8.3f} {:>4} {:>4} {:>6}\n", step,
               dofs.num_dofs(), dual.num_dofs(), error, estimate.total(), estimate.total() / error,
               decision.h_marked.size(), decision.p_marked.size(), dofs.max_order());
    if (dofs.num_dofs() > kMaxDofs) break;
    const auto hp =
        hpfem::adaptivity::hp_refine<2>(adaptive, orders, decision.h_marked, decision.p_marked);
    predicted = hpfem::adaptivity::predict_indicators(estimate.indicators, orders, hp, prediction);
    orders = hp.orders;
  }
  REQUIRE(steps.size() >= 10);
  const Real b = -slope(steps, steps[steps.size() - 8].dofs, [](Real n) { return std::cbrt(n); });
  fmt::print("fit error ~ exp(-b N^(1/3)) over the last 8 steps: b = {:.3f}\n", b);
  for (const Step& s : steps) {
    // guaranteed upper bound on every mesh; the round-off margin covers the quadrature of the
    // r^{-1/3} singularity, which enters the error and the equilibrium term alike
    CHECK(s.eta >= s.error * (1 - 1e-6));
    CHECK(s.eta < 2.0 * s.error);  // sharp with the dual order p + 1
  }
  CHECK(b > 0.2);
}

// Convergence test #7 (CLAUDE.md §8): hp-adaptivity at the re-entrant corner of the L-shaped
// domain. The loop SOLVE → ESTIMATE → MARK (Dörfler) → DECIDE (error prediction) → REFINE
// (hanging-node h-refinement where the prediction fails, p-refinement with spreading
// elsewhere) must show exponential convergence, error ~ exp(-b N^{1/3}), where uniform
// refinement gives N^{-1/3} and adaptive h-refinement N^{-p/2}. Same singular solution as
// adaptive_h_refinement.cpp.
#include <algorithm>
#include <cmath>
#include <map>
#include <numbers>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/adaptivity/marking.hpp"
#include "hpfem/adaptivity/prediction.hpp"
#include "hpfem/adaptivity/refinement.hpp"
#include "hpfem/adaptivity/residual_estimator.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/adaptive_mesh.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"

using hpfem::as_size;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::assembly::ComplexCurl;
using hpfem::assembly::ComplexVector;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::AdaptiveMesh;
using hpfem::mesh::Mesh;
using hpfem::mesh::Tag;
using hpfem::physics::Formulation;
using hpfem::physics::IncidentField;
using hpfem::physics::Scattering;
using hpfem::physics::ScatteringSetup;

namespace {

constexpr Real kWavenumber = 1.0;
constexpr Tag kBoundary = 9;
constexpr Real kTheta = 0.5;       // Dörfler bulk fraction
constexpr Index kMaxDofs = 20000;  // stop after the first step beyond this

Mesh<2> l_shape(Index n) {
  const Mesh<2> square =
      hpfem::mesh::rectangle(2 * n, 2 * n, Point<2>(-1.0, -1.0), Point<2>(1.0, 1.0));
  Mesh<2> l =
      hpfem::mesh::extract<2>(square, [](const Point<2>& x) { return !(x(0) > 0 && x(1) < 0); });
  for (const Index f : l.boundary_facets()) l.set_facet_tag(f, kBoundary);
  return l;
}

Real angle(const Point<2>& x) {
  Real theta = std::atan2(x(1), x(0));
  if (theta < 0) theta += 2 * std::numbers::pi;
  return theta;
}

IncidentField<2> singular_field() {
  IncidentField<2> f;
  f.value = [](const Point<2>& x) {
    const Real r = x.norm();
    const Real theta = angle(x);
    const Real dr = (2.0 / 3.0) * std::pow(r, -1.0 / 3.0) * std::sin(2 * theta / 3);
    const Real dt = (2.0 / 3.0) * std::pow(r, -1.0 / 3.0) * std::cos(2 * theta / 3);
    return ComplexVector<2>(Complex{dr * std::cos(theta) - dt * std::sin(theta), 0.0},
                            Complex{dr * std::sin(theta) + dt * std::cos(theta), 0.0});
  };
  f.curl = [](const Point<2>&) { return ComplexCurl<2>(Complex{0.0, 0.0}); };
  return f;
}

ScatteringSetup<2> setup() {
  ScatteringSetup<2> s;
  s.omega = kWavenumber * hpfem::constants::c0;
  s.incident = singular_field();
  s.formulation = Formulation::kTotalField;
  s.incident_tags = {kBoundary};
  const auto exact = s.incident;
  s.current = [exact](const Point<2>& x) {
    return ComplexVector<2>(-kWavenumber * kWavenumber * exact.value(x));
  };
  return s;
}

struct Step {
  Index dofs;
  Real error;
  Real estimate;
  Index h_marked;
  Index p_marked;
  int max_order;
};

bool touches_corner(const Mesh<2>& mesh, Index c) {
  for (const Index v : mesh.cell_vertices(c)) {
    if (mesh.vertex(v).norm() < 1e-12) return true;
  }
  return false;
}

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

TEST_CASE("L-shape: hp-adaptivity converges exponentially in N^(1/3)",
          "[convergence][adaptivity][hp]") {
  AdaptiveMesh<2> adaptive(l_shape(2));
  std::vector<int> orders(as_size(adaptive.mesh().num_cells()), 1);
  std::vector<Step> steps;
  std::vector<Real> predicted;  // empty in the first step: h-refinement
  Index corner_h = 0;
  Index corner_p = 0;
  const hpfem::adaptivity::PredictionOptions prediction;
  fmt::print(
      "\nL-shape, hp-adaptivity (Doerfler {}, decision by error prediction)\n{:>5} {:>8} {:>12} "
      "{:>12} {:>8} {:>4} {:>4} {:>6}\n",
      kTheta, "step", "DoF", "error", "eta", "eta/err", "#h", "#p", "max p");
  for (int step = 0; step < 40; ++step) {
    const Mesh<2> mesh = adaptive.mesh();
    const NedelecDofMap<2> dofs(mesh, orders);
    const Scattering<2> problem(dofs, setup());
    const auto solution = problem.solve();
    const auto e = problem.error(solution, problem.setup().incident);
    const auto estimate = problem.estimate(solution);
    const Real error = std::hypot(e.l2, e.curl);
    const auto marked = hpfem::adaptivity::dorfler_marking(estimate.indicators, kTheta);
    REQUIRE(!marked.empty());
    const auto decision =
        hpfem::adaptivity::hp_decide_by_prediction(estimate.indicators, predicted, marked);
    steps.push_back({dofs.num_dofs(), error, estimate.total(),
                     static_cast<Index>(decision.h_marked.size()),
                     static_cast<Index>(decision.p_marked.size()), dofs.max_order()});
    fmt::print("{:>5} {:>8} {:>12.3e} {:>12.3e} {:>8.2f} {:>4} {:>4} {:>6}\n", step,
               dofs.num_dofs(), error, estimate.total(), estimate.total() / error,
               decision.h_marked.size(), decision.p_marked.size(), dofs.max_order());
    // the corner cells are (almost) always h-refined: a corner cell raised by p-spreading
    // may once look smooth to the prediction
    for (const Index c : decision.p_marked) corner_p += touches_corner(mesh, c) ? 1 : 0;
    for (const Index c : decision.h_marked) corner_h += touches_corner(mesh, c) ? 1 : 0;
    if (dofs.num_dofs() > kMaxDofs || error < 1e-7) break;
    const auto hp =
        hpfem::adaptivity::hp_refine<2>(adaptive, orders, decision.h_marked, decision.p_marked);
    predicted = hpfem::adaptivity::predict_indicators(estimate.indicators, orders, hp, prediction);
    orders = hp.orders;
  }
  {
    const Mesh<2>& mesh = adaptive.mesh();
    std::map<int, std::pair<int, int>> by_level;  // level -> (cells, max p)
    for (Index c = 0; c < mesh.num_cells(); ++c) {
      auto& entry = by_level[adaptive.level(c)];
      entry.first++;
      entry.second = std::max(entry.second, orders[as_size(c)]);
    }
    fmt::print("final mesh: {} cells, levels (cells, max p):", mesh.num_cells());
    for (const auto& [level, entry] : by_level) {
      fmt::print(" {}:({},{})", level, entry.first, entry.second);
    }
    fmt::print("\n");
  }
  REQUIRE(steps.size() >= 10);
  // exponential in N^{1/3}: negative slope of log(error) against N^{1/3} over the last steps
  const Real b = -slope(steps, steps[steps.size() - 10].dofs, [](Real n) { return std::cbrt(n); });
  // algebraic view over the asymptotic range: steeper than any fixed-p h-adaptivity reaches
  // here (p = 2: N^-1, p = 3: N^-1.5)
  const Real algebraic = slope(steps, 4000, [](Real n) { return std::log(n); });
  fmt::print(
      "fit error ~ exp(-b N^(1/3)) over the last 10 steps: b = {:.3f}; algebraic slope "
      "for N >= 4000: {:.2f}\n",
      b, algebraic);
  fmt::print("corner cells: {} h-decisions, {} p-decisions\n", corner_h, corner_p);
  CHECK(corner_h >= 3 * corner_p);
  CHECK(b > 0.2);
  CHECK(algebraic < -1.5);
  CHECK(steps.back().error < 1.5e-4);  // h-adaptive p = 2 needs ~1e-3 at 12 600 DoFs
  for (const Step& s : steps) {
    CHECK(s.estimate / s.error > 0.1);
    CHECK(s.estimate / s.error < 50.0);
  }
}

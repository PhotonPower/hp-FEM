// Goal-oriented adaptivity on the L-shaped corner: the quantity of interest is the field
// value at a point away from the corner. The DWR estimate must track the true goal error
// (effectivity bounded) and the goal-driven h-refinement loop must reduce the goal error
// faster than the energy-driven one at the same number of DoFs, since it refines only
// where the adjoint weight sees the corner.
#include "hpfem/physics/goal_oriented.hpp"

#include <cmath>
#include <numbers>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/adaptivity/marking.hpp"
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
const Point<2> kGoalPoint(-0.55, 0.45);
const ComplexVector<2> kGoalWeight(Complex{1.0, 0.0}, Complex{0.0, 0.0});

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
  Real goal_error;  ///< |Q(E) − Q(E_h)|
  Real estimate;    ///< |Σ r_K|
  Real energy_error;
};

/// Adaptive h-refinement (p fixed) driven by the DWR indicators or the energy indicators.
std::vector<Step> run(int p, bool goal_driven, Index max_dofs) {
  AdaptiveMesh<2> adaptive(l_shape(2));
  const auto functional = hpfem::physics::point_value_functional<2>(kGoalPoint, kGoalWeight);
  const Complex exact = (singular_field().value(kGoalPoint).transpose() * kGoalWeight)(0);
  std::vector<Step> steps;
  for (int step = 0; step < 30; ++step) {
    const NedelecDofMap<2> dofs(adaptive.mesh(), p);
    const Scattering<2> problem(dofs, setup());
    const auto solution = problem.solve();
    const auto goal = hpfem::physics::dwr_estimate<2>(problem, solution, functional);
    const auto energy = problem.estimate(solution);
    const auto e = problem.error(solution, problem.setup().incident);
    steps.push_back({dofs.num_dofs(), std::abs(exact - goal.value), std::abs(goal.error),
                     std::hypot(e.l2, e.curl)});
    if (dofs.num_dofs() > max_dofs) break;
    const auto marked =
        hpfem::adaptivity::dorfler_marking(goal_driven ? goal.indicators : energy.indicators, 0.5);
    adaptive.refine(marked);
  }
  return steps;
}

void print(const std::string& title, const std::vector<Step>& steps) {
  fmt::print("\n{}\n{:>5} {:>8} {:>12} {:>12} {:>8} {:>12}\n", title, "step", "DoF", "|Q err|",
             "|estimate|", "eff.", "energy err");
  for (std::size_t i = 0; i < steps.size(); ++i) {
    fmt::print("{:>5} {:>8} {:>12.3e} {:>12.3e} {:>8.2f} {:>12.3e}\n", i, steps[i].dofs,
               steps[i].goal_error, steps[i].estimate, steps[i].estimate / steps[i].goal_error,
               steps[i].energy_error);
  }
}

}  // namespace

TEST_CASE(
    "L-shape: DWR estimate tracks the goal error and goal-driven refinement beats "
    "energy-driven refinement",
    "[convergence][adaptivity][goal]") {
  const int p = 2;
  const auto goal_driven = run(p, true, 6000);
  const auto energy_driven = run(p, false, 6000);
  print("goal-driven h-adaptivity, p = 2 (point value at (-0.55, 0.45))", goal_driven);
  print("energy-driven h-adaptivity, p = 2, same goal", energy_driven);
  REQUIRE(goal_driven.size() >= 6);
  // effectivity of the DWR estimate (goal-driven run)
  for (std::size_t i = 1; i < goal_driven.size(); ++i) {
    const Real eff = goal_driven[i].estimate / goal_driven[i].goal_error;
    CHECK(eff > 0.2);
    CHECK(eff < 5.0);
  }
  // the goal error decreases and ends far below the start
  CHECK(goal_driven.back().goal_error < 1e-3 * goal_driven.front().goal_error);
  // at comparable DoFs the goal-driven loop has the smaller goal error
  const Step& g = goal_driven.back();
  Real energy_at_same = 0;
  for (const Step& s : energy_driven) {
    if (s.dofs <= g.dofs) energy_at_same = s.goal_error;
  }
  fmt::print("goal error at ~{} DoFs: goal-driven {:.3e}, energy-driven {:.3e}\n", g.dofs,
             g.goal_error, energy_at_same);
  CHECK(g.goal_error < energy_at_same);
}

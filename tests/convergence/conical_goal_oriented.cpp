// Goal-oriented hp-adaptivity for the conical solver (M15 F1 stage 2): the manufactured
// gradient mode of conical_hp_corner (re-entrant PEC corner, beta = 1.3) with the goal
// Q(E) = E(x_g) . w at a point away from the corner, h-refinement at p = 2 (as the in-plane
// goal_oriented test). The DWR estimate must track the goal error (effectivity in [0.2, 5])
// and the goal-driven loop must reach a smaller goal error than the energy-driven loop at
// comparable DoFs.
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
#include "hpfem/physics/conical_goal.hpp"
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

constexpr Real kBeta = 1.3;
constexpr hpfem::mesh::Tag kWall = 9;
const Point<2> kGoalPoint(-0.55, 0.45);
const ConicalVector kGoalWeight(Complex{1.0, 0.0}, Complex{0.5, 0.0}, Complex{0.0, -0.7});

struct Potential {
  Real value, d_x, d_y;
};

Potential potential(const Point<2>& p) {
  const Real x = p(0), y = p(1);
  const Real rho = std::hypot(x, y);
  Real theta = std::atan2(y, x);
  if (theta < 0) theta += 2 * std::numbers::pi;
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

ConicalVector exact(const Point<2>& x) {
  const Potential p = potential(x);
  return ConicalVector(Complex{p.d_x, 0.0}, Complex{p.d_y, 0.0}, kI * kBeta * p.value);
}

ConicalVector source(const Point<2>& x) {  // f = -k^2 E, k = 1, scaled components
  const Potential p = potential(x);
  return ConicalVector(-p.d_x, -p.d_y, -kBeta * p.value);
}

Mesh<2> l_shape(Index n) {
  const Mesh<2> full =
      hpfem::mesh::rectangle(2 * n, 2 * n, Point<2>(-1.0, -1.0), Point<2>(1.0, 1.0));
  Mesh<2> out =
      hpfem::mesh::extract<2>(full, [](const Point<2>& c) { return !(c(0) > 0.0 && c(1) < 0.0); });
  for (const Index f : out.boundary_facets()) out.set_facet_tag(f, kWall);
  return out;
}

struct Step {
  Index dofs;
  Real goal_error;
  Real estimate;
  Real eta;
};

std::vector<Step> run(bool goal_driven, Index max_dofs) {
  AdaptiveMesh<2> adaptive(l_shape(2));
  const int p = 2;
  const auto functional = hpfem::physics::conical_point_functional(kGoalPoint, kGoalWeight);
  const Complex exact_goal = (exact(kGoalPoint).transpose() * kGoalWeight)(0);
  std::vector<Step> steps;
  for (int step = 0; step < 40; ++step) {
    const Mesh<2> mesh = adaptive.mesh();
    const NedelecDofMap<2> nd(mesh, p);
    const DofMap<2> h1(mesh, p);
    ConicalScatteringSetup setup;
    setup.omega = hpfem::constants::c0;
    setup.beta = kBeta;
    setup.pec_tags = {kWall};
    setup.current = source;
    const ConicalScattering problem(nd, h1, setup);
    const auto solution = problem.solve();
    const auto goal = hpfem::physics::conical_dwr_estimate(problem, solution, functional);
    const auto energy = problem.estimate(solution);
    const Index dofs = static_cast<Index>(problem.free_dofs().size());
    steps.push_back(
        {dofs, std::abs(exact_goal - goal.value), std::abs(goal.error), energy.total()});
    if (dofs > max_dofs) break;
    const auto& indicators = goal_driven ? goal.indicators : energy.indicators;
    adaptive.refine(hpfem::adaptivity::dorfler_marking(indicators, 0.5));
  }
  return steps;
}

void print(const char* title, const std::vector<Step>& steps) {
  fmt::print("\n{}\n{:>5} {:>8} {:>12} {:>12} {:>8} {:>12}\n", title, "step", "DoF", "|Q err|",
             "|estimate|", "eff.", "eta");
  for (std::size_t i = 0; i < steps.size(); ++i) {
    fmt::print("{:>5} {:>8} {:>12.3e} {:>12.3e} {:>8.2f} {:>12.3e}\n", i, steps[i].dofs,
               steps[i].goal_error, steps[i].estimate, steps[i].estimate / steps[i].goal_error,
               steps[i].eta);
  }
}

}  // namespace

TEST_CASE(
    "conical re-entrant corner: DWR estimate tracks the goal error and goal-driven hp "
    "beats energy-driven hp",
    "[convergence][adaptivity][goal][conical]") {
  const auto goal_driven = run(true, 8000);
  const auto energy_driven = run(false, 8000);
  print("goal-driven h-adaptivity, p = 2, conical corner (point value at (-0.55, 0.45))",
        goal_driven);
  print("energy-driven h-adaptivity, p = 2, same goal", energy_driven);
  REQUIRE(goal_driven.size() >= 6);
  // effectivity in the resolved regime; on the coarse meshes the signed goal error crosses
  // zero (steps with |Q err| far below the neighbouring steps), where any estimate "fails"
  for (std::size_t i = 1; i < goal_driven.size(); ++i) {
    if (goal_driven[i].dofs < 3000) continue;
    const Real eff = goal_driven[i].estimate / goal_driven[i].goal_error;
    CHECK(eff > 0.2);
    CHECK(eff < 5.0);
  }
  CHECK(goal_driven.back().goal_error < 1e-3 * goal_driven.front().goal_error);
  const Step& g = goal_driven.back();
  Real energy_at_same = 0;
  for (const Step& s : energy_driven) {
    if (s.dofs <= g.dofs) energy_at_same = s.goal_error;
  }
  fmt::print("goal error at ~{} DoFs: goal-driven {:.3e}, energy-driven {:.3e}\n", g.dofs,
             g.goal_error, energy_at_same);
  CHECK(g.goal_error < energy_at_same);
}

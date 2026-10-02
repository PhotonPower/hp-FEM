// Adaptive h-refinement (SOLVE → ESTIMATE → MARK → REFINE with hanging nodes) on the
// singular Maxwell solution of the L-shaped domain: E = ∇(r^{2/3} sin(2θ/3)) solves
// curl curl E − k² E = −k² E with the exact tangential trace on the boundary. The field
// behaves like r^{-1/3} at the re-entrant corner, so uniform refinement converges like
// N^{-1/3} in the H(curl) norm for every p, while the adaptive loop must recover a rate
// close to the optimal N^{-p/2} (docs/theory/hp-adaptivity.md). The largest indicator must sit
// at the corner in every step, and the prolongated solution must satisfy the hanging-node
// constraints of the refined mesh.
#include <cmath>
#include <numbers>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/adaptivity/marking.hpp"
#include "hpfem/adaptivity/residual_estimator.hpp"
#include "hpfem/assembly/hanging_constraints.hpp"
#include "hpfem/assembly/prolongation.hpp"
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

constexpr Real kWavenumber = 1.0;  // not an eigenvalue of the L-shape (first: 1.4756)
constexpr Tag kBoundary = 9;

/// L-shaped domain [-1,1]² without the quadrant x > 0, y < 0; all boundary edges tagged.
Mesh<2> l_shape(Index n) {
  const Mesh<2> square =
      hpfem::mesh::rectangle(2 * n, 2 * n, Point<2>(-1.0, -1.0), Point<2>(1.0, 1.0));
  Mesh<2> l =
      hpfem::mesh::extract<2>(square, [](const Point<2>& x) { return !(x(0) > 0 && x(1) < 0); });
  for (const Index f : l.boundary_facets()) l.set_facet_tag(f, kBoundary);
  return l;
}

/// Polar angle in [0, 3π/2) measured from the positive x-axis (the removed quadrant is
/// (-π/2, 0)).
Real angle(const Point<2>& x) {
  Real theta = std::atan2(x(1), x(0));
  if (theta < 0) theta += 2 * std::numbers::pi;
  return theta;
}

/// E = ∇φ, φ = r^{2/3} sin(2θ/3): curl-free, singular at the corner, n × E = 0 on the two
/// corner edges.
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

struct Step {
  Index dofs;
  Real error;     ///< H(curl) error
  Real estimate;  ///< η
};

void print(const std::string& title, const std::vector<Step>& steps) {
  fmt::print("\n{}\n{:>6} {:>8} {:>12} {:>7} {:>12} {:>8}\n", title, "step", "DoF", "error", "rate",
             "eta", "eta/err");
  for (std::size_t i = 0; i < steps.size(); ++i) {
    std::string rate = "-";
    if (i > 0) {
      rate = fmt::format("{:.2f}", std::log(steps[i].error / steps[i - 1].error) /
                                       std::log(static_cast<Real>(steps[i].dofs) /
                                                static_cast<Real>(steps[i - 1].dofs)));
    }
    fmt::print("{:>6} {:>8} {:>12.3e} {:>7} {:>12.3e} {:>8.2f}\n", i, steps[i].dofs, steps[i].error,
               rate, steps[i].estimate, steps[i].estimate / steps[i].error);
  }
}

/// Least-squares slope of log(error) against log(DoF) over the last `count` steps.
Real slope(const std::vector<Step>& steps, std::size_t count) {
  const std::size_t begin = steps.size() - count;
  Real sx = 0;
  Real sy = 0;
  Real sxx = 0;
  Real sxy = 0;
  for (std::size_t i = begin; i < steps.size(); ++i) {
    const Real x = std::log(static_cast<Real>(steps[i].dofs));
    const Real y = std::log(steps[i].error);
    sx += x;
    sy += y;
    sxx += x * x;
    sxy += x * y;
  }
  const Real n = static_cast<Real>(count);
  return (n * sxy - sx * sy) / (n * sxx - sx * sx);
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

/// One solve: DoFs, H(curl) error, estimate, indicators.
Step solve(const Mesh<2>& mesh, int p, Vector* coefficients, std::vector<Real>* indicators) {
  const NedelecDofMap<2> dofs(mesh, p);
  const Scattering<2> problem(dofs, setup());
  const auto solution = problem.solve();
  const auto e = problem.error(solution, problem.setup().incident);
  const auto estimate = problem.estimate(solution);
  if (coefficients) *coefficients = solution.unknown;
  if (indicators) *indicators = estimate.indicators;
  return {dofs.num_dofs(), std::hypot(e.l2, e.curl), estimate.total()};
}

bool touches_corner(const Mesh<2>& mesh, Index c) {
  for (const Index v : mesh.cell_vertices(c)) {
    if (mesh.vertex(v).norm() < 1e-12) return true;
  }
  return false;
}

std::vector<Step> uniform(int p, int levels) {
  AdaptiveMesh<2> adaptive(l_shape(2));
  std::vector<Step> steps;
  for (int level = 0; level <= levels; ++level) {
    steps.push_back(solve(adaptive.mesh(), p, nullptr, nullptr));
    if (level < levels) adaptive.refine_all();
  }
  return steps;
}

std::vector<Step> adapt(int p, Index max_dofs, Real theta) {
  AdaptiveMesh<2> adaptive(l_shape(2));
  std::vector<Step> steps;
  for (int step = 0; step < 40; ++step) {
    const Mesh<2> mesh = adaptive.mesh();
    Vector coefficients;
    std::vector<Real> indicators;
    steps.push_back(solve(mesh, p, &coefficients, &indicators));
    // the largest indicator sits at the corner
    const auto worst = std::max_element(indicators.begin(), indicators.end()) - indicators.begin();
    REQUIRE(touches_corner(mesh, static_cast<Index>(worst)));
    if (steps.back().dofs > max_dofs) break;
    const auto marked = hpfem::adaptivity::dorfler_marking(indicators, theta);
    REQUIRE(!marked.empty());
    const auto refinement = adaptive.refine(marked);
    // the transferred solution satisfies the new hanging-node constraints
    const NedelecDofMap<2> old_dofs(mesh, p);
    const NedelecDofMap<2> new_dofs(adaptive.mesh(), p);
    const Vector transferred =
        hpfem::assembly::prolongate(old_dofs, coefficients, new_dofs, refinement);
    const auto constraints = hpfem::assembly::hanging_constraints(new_dofs);
    Real worst_constraint = 0;
    for (Index s = 0; s < new_dofs.num_dofs(); ++s) {
      if (!constraints.is_constrained(s)) continue;
      Complex sum = 0;
      for (const auto& t : constraints.terms(s)) sum += t.coefficient * transferred(t.master);
      worst_constraint = std::max(worst_constraint, std::abs(sum - transferred(s)));
    }
    REQUIRE(worst_constraint < 1e-9 * coefficients.norm());
  }
  return steps;
}

}  // namespace

TEST_CASE("L-shape: uniform refinement is limited by the corner singularity",
          "[convergence][adaptivity]") {
  const auto steps = uniform(2, 3);
  print("L-shape, uniform refinement, p = 2", steps);
  const Real s = slope(steps, 3);
  CHECK(s < -0.25);
  CHECK(s > -0.45);  // N^{-1/3}: no better than the singularity allows, regardless of p
}

TEST_CASE("L-shape: adaptive h-refinement recovers the optimal rate", "[convergence][adaptivity]") {
  for (int p = 1; p <= 2; ++p) {
    const auto steps = adapt(p, 12000, 0.5);
    print(fmt::format("L-shape, adaptive refinement (Dörfler 0.5), p = {}", p), steps);
    REQUIRE(steps.size() >= 8);
    const Real s = slope(steps, 6);
    fmt::print("slope over the last 6 steps: {:.3f} (optimal {:.3f}, uniform -0.333)\n", s,
               -0.5 * p);
    CHECK(s < -0.5 * p + 0.12);
    for (const Step& st : steps) {
      CHECK(st.estimate / st.error > 0.1);
      CHECK(st.estimate / st.error < 50.0);
    }
    CHECK(steps.back().error < steps.front().error / 10);
  }
}

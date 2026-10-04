// hp-adaptivity on the meridian plane (docs/theory/axisymmetric.md, ADR-0010): the
// manufactured gradient mode with the re-entrant PEC edge of a body of revolution
// (tests/unit/physics/axisymmetric_corner.hpp, |E| ~ rho^{-1/3} at the edge) is solved with
// the loop SOLVE - ESTIMATE (r-weighted residual estimator) - MARK (Doerfler 0.5) - DECIDE
// (error prediction) - REFINE (hanging nodes, p-spreading). The error in the weighted
// H(curl) norm must decay exponentially in N^{1/3}; uniform refinement at fixed p would
// converge algebraically like N^{-1/3}.
#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "../unit/physics/axisymmetric_corner.hpp"
#include "hpfem/adaptivity/marking.hpp"
#include "hpfem/adaptivity/prediction.hpp"
#include "hpfem/adaptivity/refinement.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/adaptive_mesh.hpp"

using hpfem::as_size;
using hpfem::Index;
using hpfem::Real;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::AdaptiveMesh;
using hpfem::mesh::Mesh;
using hpfem::physics::AxisymmetricScattering;

namespace {

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

TEST_CASE("axisymmetric re-entrant edge: hp-adaptivity converges exponentially in N^(1/3)",
          "[convergence][adaptivity][hp][axisymmetric]") {
  using namespace hpfem::physics::test;
  const auto exact = corner_field();
  AdaptiveMesh<2> adaptive(corner_mesh(2));
  std::vector<int> orders(as_size(adaptive.mesh().num_cells()), 1);
  std::vector<Step> steps;
  std::vector<Real> predicted;
  Index corner_h = 0;
  Index corner_p = 0;
  fmt::print(
      "\naxisymmetric re-entrant edge (m = 1), hp-adaptivity (Doerfler 0.5, error prediction)\n"
      "{:>5} {:>8} {:>12} {:>12} {:>8} {:>6}\n",
      "step", "DoF", "error", "eta", "eta/err", "max p");
  for (int step = 0; step < 40; ++step) {
    const Mesh<2> mesh = adaptive.mesh();
    const NedelecDofMap<2> nd(mesh, orders);
    const DofMap<2> h1(mesh, orders);
    const AxisymmetricScattering problem(nd, h1, corner_setup());
    const auto field = problem.solve();
    const auto e = problem.error(field, exact);
    const auto estimate = problem.estimate(field);
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
      "4000: {:.2f} (uniform h-refinement: -0.33)\n",
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

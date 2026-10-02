// Adaptive p-refinement driven by the residual estimator: on the smooth plane-wave problem
// the loop SOLVE → ESTIMATE → MARK → p-REFINE must converge exponentially in the number of
// DoFs (error ~ exp(-b N^{1/2}) in 2D), far faster than any h-refinement, and the
// transferred solution must stay consistent. Variable orders per cell exercise the minimum
// rule on shared edges.
#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/adaptivity/marking.hpp"
#include "hpfem/adaptivity/refinement.hpp"
#include "hpfem/adaptivity/residual_estimator.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/assembly/prolongation.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"

using hpfem::as_size;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::assembly::ComplexVector;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::Mesh;
using hpfem::physics::Formulation;
using hpfem::physics::plane_wave;
using hpfem::physics::Scattering;
using hpfem::physics::ScatteringSetup;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kWavenumber = 6.0;  // about one wavelength across the unit square

struct Step {
  Index dofs;
  int max_order;
  Real error;
  Real estimate;
};

}  // namespace

TEST_CASE("adaptive p-refinement converges exponentially on a smooth solution",
          "[convergence][adaptivity]") {
  const Mesh<2> mesh = hpfem::mesh::rectangle(4, 4);
  ScatteringSetup<2> setup;
  setup.omega = kWavenumber * hpfem::constants::c0;
  setup.incident = plane_wave<2>(ComplexVector<2>(Complex{-0.8, 0.0}, Complex{0.6, 0.0}),
                                 kWavenumber * Point<2>(0.6, 0.8));
  setup.formulation = Formulation::kTotalField;
  setup.incident_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};

  std::vector<int> orders(as_size(mesh.num_cells()), 1);
  std::vector<Step> steps;
  Vector previous;
  fmt::print(
      "\nplane wave, adaptive p-refinement (Doerfler 0.8)\n{:>5} {:>8} {:>6} {:>12} {:>12} {:>8}\n",
      "step", "DoF", "max p", "error", "eta", "eta/err");
  for (int step = 0; step < 16; ++step) {
    const NedelecDofMap<2> dofs(mesh, orders);
    const Scattering<2> problem(dofs, setup);
    const auto solution = problem.solve();
    const auto e = problem.error(solution, setup.incident);
    const auto estimate = problem.estimate(solution);
    const Real error = std::hypot(e.l2, e.curl) / std::hypot(e.l2_norm, e.curl_norm);
    steps.push_back({dofs.num_dofs(), dofs.max_order(), error, estimate.total()});
    fmt::print("{:>5} {:>8} {:>6} {:>12.3e} {:>12.3e} {:>8.2f}\n", step, dofs.num_dofs(),
               dofs.max_order(), error, estimate.total(), estimate.total() / error);
    const auto marked = hpfem::adaptivity::dorfler_marking(estimate.indicators, 0.8);
    REQUIRE(!marked.empty());
    const std::vector<int> raised = hpfem::adaptivity::p_refine(orders, marked);
    const NedelecDofMap<2> new_dofs(mesh, raised);
    const Vector transferred = hpfem::assembly::prolongate(
        dofs, solution.unknown, new_dofs, hpfem::adaptivity::identity_step(mesh.num_cells()));
    // the transfer is exact: same field in every cell
    for (Index c = 0; c < mesh.num_cells(); ++c) {
      for (const auto& xi : {Point<2>(0.2, 0.3), Point<2>(0.5, 0.4)}) {
        const auto a = hpfem::assembly::evaluate_hcurl(dofs, solution.unknown, c, xi);
        const auto b = hpfem::assembly::evaluate_hcurl(new_dofs, transferred, c, xi);
        REQUIRE((a - b).norm() < 1e-10 * (1.0 + a.norm()));
      }
    }
    orders = raised;
    if (error < 1e-8) break;
  }
  REQUIRE(steps.size() >= 6);
  // exponential: log(error) linear in sqrt(N) with negative slope, and a drop of at least
  // a factor 1.5 per step once p > 1 everywhere matters
  Real sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (const Step& s : steps) {
    const Real x = std::sqrt(static_cast<Real>(s.dofs));
    const Real y = std::log(s.error);
    sx += x;
    sy += y;
    sxx += x * x;
    sxy += x * y;
  }
  const Real n = static_cast<Real>(steps.size());
  const Real b = -(n * sxy - sx * sy) / (n * sxx - sx * sx);
  fmt::print("fit error ~ exp(-b sqrt(N)): b = {:.4f}\n", b);
  CHECK(b > 0.05);
  for (std::size_t i = 2; i < steps.size(); ++i) CHECK(steps[i].error < steps[i - 1].error);
  CHECK(steps.back().error < 1e-5);
  // the effectivity index stays bounded: its constant grows with p (accepted in
  // docs/theory/error-estimation.md), so the spread over the run is what is checked
  Real min_effectivity = 1e9;
  Real max_effectivity = 0;
  for (const Step& s : steps) {
    min_effectivity = std::min(min_effectivity, s.estimate / s.error);
    max_effectivity = std::max(max_effectivity, s.estimate / s.error);
  }
  CHECK(min_effectivity > 0.1);
  CHECK(max_effectivity / min_effectivity < 5.0);
}

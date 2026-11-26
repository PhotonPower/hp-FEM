// hp-adaptivity on the Mie cylinder (issue #234): curved interface (geometry order 2), PML,
// scattered-field formulation. The loop SOLVE → ESTIMATE → MARK (Dörfler) → DECIDE (error
// prediction) → REFINE of adaptive_hp_refinement.cpp must reduce the estimate in every step,
// spread it over many cells and bring the scattering width down to the floor set by the PML.
//
// Before the fixes of #234 the loop diverged: square_with_disc blended quadratically from the
// circle to the square, so the first ring of cells outside the circle was ~ n (l - 1) times
// thinner than wide, and with one diagonal for all squares the cells along the circle in two
// quadrants were slivers that their curved edge folded (det J < 0). The volume residual blew up
// on those few cells (eta ~ 1e5 from a handful of cells), the loop refined only them and the
// error grew; refined cells far from the origin then failed in
// QuadraticGeometry::to_reference, whose tolerance was below the rounding of the coordinates.
//
// PML (issue #243): the residual of the stretched problem is large in coarse layer cells
// (|k s| h = 6.2 at R0 = 1e-10, 3.8 at R0 = 1e-6, limit 0.75 at p = 1), although their error
// reaches the physical domain only damped by the layer. Unweighted, 60-90 % of eta lay in the
// layer and with R0 = 1e-10 the loop spent most unknowns there: the error of the scattering width
// stalled near 6e-3 at 38 000 unknowns, while uniform p on the same mesh gives 4.3e-3 at 22 000.
// The estimator now weights the layer cells by the attenuation exp(-k |Im x~|) of an outgoing
// wave from the start of the layer (EstimatorOptions::pml_weighting, default): the second test
// case runs the loop with R0 = 1e-10 and requires it to beat uniform p at equal cost.
#include <algorithm>
#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "hpfem/adaptivity/marking.hpp"
#include "hpfem/adaptivity/prediction.hpp"
#include "hpfem/adaptivity/refinement.hpp"
#include "hpfem/adaptivity/residual_estimator.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/adaptive_mesh.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/physics/mie.hpp"
#include "hpfem/physics/postprocess.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"
#include "hpfem/pml/pml.hpp"

using hpfem::as_size;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::assembly::ComplexVector;
using hpfem::fespace::NedelecDofMap;
using hpfem::materials::Material;
using hpfem::mesh::AdaptiveMesh;
using hpfem::mesh::Mesh;
using hpfem::physics::Formulation;
using hpfem::physics::Scattering;
using hpfem::physics::ScatteringSetup;
using hpfem::physics::Surface;
using hpfem::pml::PmlBox;
using hpfem::pml::PmlProfile;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kRadius = 0.25;
constexpr Real kIndex = 1.5;
constexpr Real kWavenumber = 6.0;
constexpr Real kHalfWidth = 1.0;
constexpr Real kPml = 1.0;
constexpr Real kR0 = 1e-6;  // see above; the second test case uses 1e-10
constexpr hpfem::mesh::Tag kDisc = 2;
constexpr Real kTheta = 0.5;
constexpr Index kMaxDofs = 30000;

ScatteringSetup<2> setup(Real r0 = kR0) {
  ScatteringSetup<2> s;
  s.omega = kWavenumber * hpfem::constants::c0;
  s.materials.set(kDisc, Material::dielectric(kIndex));
  s.incident =
      hpfem::physics::plane_wave<2>(ComplexVector<2>(0.0, 1.0), Point<2>(kWavenumber, 0.0));
  s.formulation = Formulation::kScatteredField;
  s.pml = PmlBox<2>::uniform(Point<2>(-kHalfWidth, -kHalfWidth), Point<2>(kHalfWidth, kHalfWidth),
                             kPml, kWavenumber, 1.0, PmlProfile{2, r0});
  s.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  return s;
}

struct Step {
  Index dofs;
  Real error;
  Real estimate;
  Real largest;  // max eta_K / eta
};

}  // namespace

TEST_CASE("Mie cylinder: hp-adaptivity on the curved mesh with PML converges (#234)",
          "[convergence][adaptivity][hp][mie]") {
  const Real exact = hpfem::physics::mie_cylinder_scattering_width(kWavenumber, kRadius, kIndex);
  AdaptiveMesh<2> adaptive(
      hpfem::mesh::square_with_disc(2, kRadius, kHalfWidth, kHalfWidth + kPml, kDisc));
  std::vector<int> orders(as_size(adaptive.mesh().num_cells()), 1);
  std::vector<Real> predicted;
  std::vector<Step> steps;
  fmt::print(
      "\nMie cylinder, hp-adaptivity (Doerfler {}, PML R0 = {})\n{:>5} {:>8} {:>12} {:>12} "
      "{:>8} {:>10} {:>4} {:>4} {:>6}\n",
      kTheta, kR0, "step", "DoF", "rel. err", "eta", "eta PML", "max eta_K", "#h", "#p", "max p");
  for (int step = 0; step < 20; ++step) {
    const Mesh<2> mesh = adaptive.mesh();
    const NedelecDofMap<2> dofs(mesh, orders);
    const Scattering<2> problem(dofs, setup());
    const auto solution = problem.solve();
    const auto cs = hpfem::physics::cross_sections<2>(problem, solution,
                                                      Surface<2>::around_cells(mesh, kDisc), 1.0);
    const auto estimate = problem.estimate(solution);
    Real pml = 0;
    Real largest = 0;
    for (Index c = 0; c < mesh.num_cells(); ++c) {
      const Point<2> x = hpfem::mesh::affine_map(mesh, c).centroid();
      const Real eta = estimate.indicators[as_size(c)];
      if (std::max(std::abs(x(0)), std::abs(x(1))) > kHalfWidth) pml += eta * eta;
      largest = std::max(largest, eta);
    }
    const Real eta = estimate.total();
    const auto marked = hpfem::adaptivity::dorfler_marking(estimate.indicators, kTheta);
    const auto decision =
        hpfem::adaptivity::hp_decide_by_prediction(estimate.indicators, predicted, marked);
    steps.push_back({dofs.num_dofs(), std::abs(cs.scattering - exact) / exact, eta, largest / eta});
    fmt::print("{:>5} {:>8} {:>12.3e} {:>12.3e} {:>7.1f}% {:>10.3f} {:>4} {:>4} {:>6}\n", step,
               dofs.num_dofs(), steps.back().error, eta, 100 * pml / (eta * eta),
               steps.back().largest, decision.h_marked.size(), decision.p_marked.size(),
               dofs.max_order());
    if (dofs.num_dofs() > kMaxDofs) break;
    const auto hp =
        hpfem::adaptivity::hp_refine<2>(adaptive, orders, decision.h_marked, decision.p_marked);
    predicted = hpfem::adaptivity::predict_indicators(estimate.indicators, orders, hp);
    orders = hp.orders;
  }
  REQUIRE(steps.size() >= 6);
  for (std::size_t i = 0; i < steps.size(); ++i) {
    INFO("step " << i);
    CHECK(steps[i].largest < 0.2);  // no handful of cells dominates (before: ~1)
    if (i > 0) CHECK(steps[i].estimate < steps[i - 1].estimate);
  }
  CHECK(steps.back().estimate < 0.1 * steps.front().estimate);
  // the error of the scattering width falls by two orders to the PML floor (~1e-3 at R0 = 1e-6)
  CHECK(steps.back().error < 3e-3);
  CHECK(steps.back().error < 0.02 * steps.front().error);
}

TEST_CASE("Mie cylinder: with a strong PML the weighted hp loop beats uniform p (#243)",
          "[convergence][adaptivity][hp][mie]") {
  constexpr Real kStrong = 1e-10;
  constexpr Index kBudget = 25000;
  const Real exact = hpfem::physics::mie_cylinder_scattering_width(kWavenumber, kRadius, kIndex);
  const Mesh<2> base =
      hpfem::mesh::square_with_disc(2, kRadius, kHalfWidth, kHalfWidth + kPml, kDisc);
  const auto width_error = [&](const Mesh<2>& mesh, const Scattering<2>& problem,
                               const hpfem::physics::ScatteringSolution<2>& solution) {
    const auto cs = hpfem::physics::cross_sections<2>(problem, solution,
                                                      Surface<2>::around_cells(mesh, kDisc), 1.0);
    return std::abs(cs.scattering - exact) / exact;
  };
  // uniform p = 3 on the base mesh: 21 696 unknowns
  const NedelecDofMap<2> uniform_dofs(base, 3);
  const Scattering<2> uniform(uniform_dofs, setup(kStrong));
  const Real uniform_error = width_error(base, uniform, uniform.solve());

  // the loop with and without the weighting of the layer cells
  const auto loop = [&](bool weighting) {
    AdaptiveMesh<2> adaptive(base);
    std::vector<int> orders(as_size(adaptive.mesh().num_cells()), 1);
    std::vector<Real> predicted;
    hpfem::adaptivity::EstimatorOptions options;
    options.pml_weighting = weighting;
    Real best = 1;
    for (int step = 0; step < 20; ++step) {
      const Mesh<2> mesh = adaptive.mesh();
      const NedelecDofMap<2> dofs(mesh, orders);
      if (dofs.num_dofs() > kBudget) break;
      const Scattering<2> problem(dofs, setup(kStrong));
      const auto solution = problem.solve();
      const Real error = width_error(mesh, problem, solution);
      best = std::min(best, error);
      const auto estimate = problem.estimate(solution, options);
      Real pml = 0;
      for (Index c = 0; c < mesh.num_cells(); ++c) {
        const Point<2> x = hpfem::mesh::affine_map(mesh, c).centroid();
        const Real eta = estimate.indicators[as_size(c)];
        if (std::max(std::abs(x(0)), std::abs(x(1))) > kHalfWidth) pml += eta * eta;
      }
      fmt::print("{:>10} {:>5} {:>8} {:>12.3e} {:>7.1f}%\n", weighting ? "weighted" : "plain", step,
                 dofs.num_dofs(), error, 100 * pml / (estimate.total() * estimate.total()));
      const auto marked = hpfem::adaptivity::dorfler_marking(estimate.indicators, kTheta);
      const auto decision =
          hpfem::adaptivity::hp_decide_by_prediction(estimate.indicators, predicted, marked);
      const auto hp =
          hpfem::adaptivity::hp_refine<2>(adaptive, orders, decision.h_marked, decision.p_marked);
      predicted = hpfem::adaptivity::predict_indicators(estimate.indicators, orders, hp);
      orders = hp.orders;
    }
    return best;
  };
  fmt::print(
      "\nMie cylinder, PML R0 = {}: uniform p = 3 ({} DoF): rel. err {:.3e}\n{:>10} {:>5} "
      "{:>8} {:>12} {:>8}\n",
      kStrong, uniform_dofs.num_dofs(), uniform_error, "estimator", "step", "DoF", "rel. err",
      "eta PML");
  const Real weighted = loop(true);
  const Real plain = loop(false);
  INFO("uniform p = 3: " << uniform_error << ", weighted loop: " << weighted
                         << ", plain loop: " << plain);
  // within the budget of uniform p = 3 the weighted loop is several times more accurate
  // (measured 3.2e-4 against 4.3e-3); the plain loop stalls in the layer (6e-3)
  CHECK(weighted < 0.25 * uniform_error);
  CHECK(weighted < 0.25 * plain);
}

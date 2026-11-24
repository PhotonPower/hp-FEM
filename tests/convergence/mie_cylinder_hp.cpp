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
// PML: R0 = 1e-6 stretches less than the R0 = 1e-10 of mie_cylinder.cpp (|k s| h = 3.8
// instead of 6.2 on the coarse layer cells at p = 1, limit 0.75). The residual estimator measures
// the error in the norm of the stretched problem and puts most of eta into the layer either
// way; with R0 = 1e-10 the loop spends most unknowns there and the error of the scattering
// width stalls near 6e-3 at 38 000 unknowns, while uniform p on the same mesh gives 4e-3 at
// 22 000 and < 1e-4 at 37 000 (follow-up: weighting of the PML cells for quantities of the
// physical domain).
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
constexpr Real kR0 = 1e-6;  // see above
constexpr hpfem::mesh::Tag kDisc = 2;
constexpr Real kTheta = 0.5;
constexpr Index kMaxDofs = 30000;

ScatteringSetup<2> setup() {
  ScatteringSetup<2> s;
  s.omega = kWavenumber * hpfem::constants::c0;
  s.materials.set(kDisc, Material::dielectric(kIndex));
  s.incident =
      hpfem::physics::plane_wave<2>(ComplexVector<2>(0.0, 1.0), Point<2>(kWavenumber, 0.0));
  s.formulation = Formulation::kScatteredField;
  s.pml = PmlBox<2>::uniform(Point<2>(-kHalfWidth, -kHalfWidth), Point<2>(kHalfWidth, kHalfWidth),
                             kPml, kWavenumber, 1.0, PmlProfile{2, kR0});
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

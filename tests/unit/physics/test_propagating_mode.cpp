// Propagating modes of a symmetric slab waveguide (TE polarisation enforced by PEC walls in
// y): the fundamental effective index against the transcendental equation, setup checks.
#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/physics/propagating_mode.hpp"

using Catch::Approx;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::materials::Material;
using hpfem::mesh::affine_map;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::physics::PropagatingMode;
using hpfem::physics::WaveguideSetup;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

/// Effective index of the even TE mode: tan(kappa d / 2) = gamma / kappa by bisection.
Real slab_te_even(Real k0, Real d, Real n_core, Real n_clad) {
  const auto f = [&](Real n) {
    const Real kappa = k0 * std::sqrt(n_core * n_core - n * n);
    const Real gamma = k0 * std::sqrt(n * n - n_clad * n_clad);
    return kappa * std::tan(kappa * d / 2) - gamma;
  };
  Real lo = n_clad + 1e-9;
  Real hi = n_core - 1e-9;
  for (int i = 0; i < 200; ++i) {
    const Real mid = 0.5 * (lo + hi);
    (f(lo) * f(mid) <= 0 ? hi : lo) = mid;
  }
  return 0.5 * (lo + hi);
}

/// Strip [-L, L] x [0, h] with the core |x| < d / 2 tagged 2, cells aligned with the core edges.
Mesh<2> slab_mesh(Index cells_per_unit, Real half_length, Real d) {
  const Index nx =
      static_cast<Index>(std::lround(2 * half_length * static_cast<Real>(cells_per_unit)));
  Mesh<2> m = rectangle(nx, 1, Point<2>(-half_length, 0.0), Point<2>(half_length, 0.5));
  for (Index c = 0; c < m.num_cells(); ++c) {
    if (std::abs(affine_map(m, c).centroid()(0)) < d / 2) m.set_cell_tag(c, 2);
  }
  return m;
}

}  // namespace

TEST_CASE("PropagatingMode: fundamental TE mode of a slab waveguide", "[physics][waveguide]") {
  const Real d = 1.0;
  const Real k0 = 2.0;  // V = 1.12 < pi/2: a single TE mode
  const Real n_core = 1.5;
  const Real exact = slab_te_even(k0, d, n_core, 1.0);
  REQUIRE(exact > 1.0);
  REQUIRE(exact < n_core);
  const Mesh<2> m = slab_mesh(4, 6.0, d);
  const int p = 3;
  const NedelecDofMap<2> nd(m, p);
  const DofMap<2> h1(m, p);
  WaveguideSetup setup;
  setup.omega = k0 * hpfem::constants::c0;
  setup.materials.set(2, Material::dielectric(n_core));
  setup.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  setup.num_modes = 3;
  const PropagatingMode<2> problem(nd, h1, setup);
  REQUIRE(problem.max_index() == Approx(n_core));
  const auto modes = problem.solve();
  REQUIRE_FALSE(modes.empty());
  REQUIRE(modes[0].effective_index == Approx(exact).epsilon(1e-6));
  REQUIRE(modes[0].beta == Approx(exact * k0).epsilon(1e-6));
  REQUIRE(modes[0].transverse.size() == nd.num_dofs());
  REQUIRE(modes[0].longitudinal.size() == h1.num_dofs());
  // the TE mode has no longitudinal field and no transverse x-component on the PEC walls
  // (numerically zero: the Arnoldi tolerance leaves a small residual component)
  REQUIRE(modes[0].longitudinal.norm() < 1e-4 * modes[0].transverse.norm());
  for (std::size_t i = 1; i < modes.size(); ++i) REQUIRE(modes[i].beta < modes[i - 1].beta);
  // only one guided mode: any further "mode" is a box-confined radiation mode below n = 1
  for (std::size_t i = 1; i < modes.size(); ++i) REQUIRE(modes[i].effective_index < 1.0);

  // setup validation
  WaveguideSetup bad = setup;
  bad.omega = 0.0;
  REQUIRE_THROWS_AS(PropagatingMode<2>(nd, h1, bad), hpfem::InvalidArgument);
  bad = setup;
  bad.materials.set(2, Material{hpfem::Complex{2.25, 0.1}, hpfem::Complex{1.0, 0.0}});
  REQUIRE_THROWS_AS(PropagatingMode<2>(nd, h1, bad), hpfem::InvalidArgument);
  const DofMap<2> other_order(m, 2);
  REQUIRE_THROWS_AS(PropagatingMode<2>(nd, other_order, setup), hpfem::InvalidArgument);
  const Mesh<2> other_mesh = slab_mesh(2, 6.0, d);
  const DofMap<2> other(other_mesh, p);
  REQUIRE_THROWS_AS(PropagatingMode<2>(nd, other, setup), hpfem::InvalidArgument);
}

#include "hpfem/assembly/hanging_constraints.hpp"
#include "hpfem/mesh/adaptive_mesh.hpp"
using hpfem::Vector;

TEST_CASE("PropagatingMode on a hanging-node mesh reproduces the slab mode",
          "[physics][waveguide][hanging]") {
  // refine the cells of the core twice on one side: the mesh is one-irregular, the
  // constrained spaces stay conforming and the effective index matches the analytic one
  const Real d = 1.0;
  const Real k0 = 2.0;
  const Real n_core = 1.5;
  const Real exact = slab_te_even(k0, d, n_core, 1.0);
  hpfem::mesh::AdaptiveMesh<2> adaptive(slab_mesh(4, 6.0, d));
  std::vector<Index> marked;
  for (Index c = 0; c < adaptive.mesh().num_cells(); ++c) {
    const Point<2> x = affine_map(adaptive.mesh(), c).centroid();
    if (x(0) > 0 && x(0) < d / 2) marked.push_back(c);
  }
  REQUIRE_FALSE(marked.empty());
  adaptive.refine(marked);
  const Mesh<2>& m = adaptive.mesh();
  REQUIRE_FALSE(m.is_conforming());
  const int p = 3;
  const NedelecDofMap<2> nd(m, p);
  const DofMap<2> h1(m, p);
  WaveguideSetup setup;
  setup.omega = k0 * hpfem::constants::c0;
  setup.materials.set(2, Material::dielectric(n_core));
  setup.pec_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  setup.num_modes = 3;
  const auto modes = PropagatingMode<2>(nd, h1, setup).solve();
  REQUIRE_FALSE(modes.empty());
  REQUIRE(modes[0].effective_index == Approx(exact).epsilon(1e-6));
  REQUIRE(modes[0].longitudinal.norm() < 1e-4 * modes[0].transverse.norm());
  // the field is continuous across the hanging edges: the constrained DoFs obey the
  // interpolation constraints of the conforming space
  const auto constraints = hpfem::assembly::hanging_constraints(nd);
  REQUIRE(constraints.num_constrained() > 0);
  const Vector& e = modes[0].transverse;
  for (Index dof = 0; dof < nd.num_dofs(); ++dof) {
    if (!constraints.is_constrained(dof)) continue;
    hpfem::Complex combination = 0;
    for (const auto& term : constraints.terms(dof))
      combination += term.coefficient * e(term.master);
    REQUIRE(std::abs(combination - e(dof)) < 1e-8 * e.norm());
  }
}

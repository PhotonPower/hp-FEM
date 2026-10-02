// Bloch-periodic constraints: slave DoFs are the phase times the matching master DoFs with
// the orientation sign, also on randomly renumbered meshes and across two periodic
// directions (box corners), and the constrained plane-wave problem is solved correctly.
#include <cmath>
#include <numbers>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "../mesh/test_meshes.hpp"
#include "hpfem/assembly/periodic.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"

using Catch::Approx;
using hpfem::as_size;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::assembly::bloch_constraints;
using hpfem::assembly::bloch_phase;
using hpfem::assembly::ComplexVector;
using hpfem::assembly::PeriodicPair;
using hpfem::fespace::Constraints;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::box;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::mesh::Tag;
using hpfem::physics::Formulation;
using hpfem::physics::plane_wave;
using hpfem::physics::Scattering;
using hpfem::physics::ScatteringSetup;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

/// Tags the sides of the unit box of a mesh without tags (e.g. after `relabel`).
template <int Dim>
void tag_unit_box_sides(Mesh<Dim>& m) {
  for (const Index f : m.boundary_facets()) {
    Point<Dim> c = Point<Dim>::Zero();
    for (const Index v : m.facet_vertices(f)) c += m.vertex(v);
    c /= static_cast<Real>(Dim);
    for (int d = 0; d < Dim; ++d) {
      if (std::abs(c(d)) < 1e-12) m.set_facet_tag(f, static_cast<Tag>(2 * d + 1));
      if (std::abs(c(d) - 1.0) < 1e-12) m.set_facet_tag(f, static_cast<Tag>(2 * d + 2));
    }
  }
}

/// Every slave DoF has exactly one master with |coefficient| = |phase| (edge DoFs map one to
/// one up to the orientation sign); returns the number of constrained DoFs.
Index check_one_to_one(const Constraints& c, Complex phase) {
  Index count = 0;
  for (Index i = 0; i < c.num_dofs(); ++i) {
    if (!c.is_constrained(i)) continue;
    ++count;
    const auto terms = c.terms(i);
    REQUIRE(terms.size() == 1);
    REQUIRE(std::abs(std::abs(terms[0].coefficient) - std::abs(phase)) < 1e-10);
    // the coefficient is the phase times +-1
    const Complex ratio = terms[0].coefficient / phase;
    REQUIRE(std::abs(ratio.imag()) < 1e-10);
    REQUIRE(std::abs(std::abs(ratio.real()) - 1.0) < 1e-10);
  }
  return count;
}

}  // namespace

TEST_CASE("bloch_constraints: 2D edges map one to one with the phase and orientation sign",
          "[assembly][periodic]") {
  const Complex phase = bloch_phase<2>(Point<2>(0.0, 0.7), Point<2>(0.0, 1.0));
  REQUIRE(std::abs(phase - std::exp(Complex{0.0, 0.7})) < 1e-15);
  const PeriodicPair<2> pair{box_tag::kYMin, box_tag::kYMax, Point<2>(0.0, 1.0), phase};
  const std::vector<PeriodicPair<2>> pairs{pair};
  for (int p = 1; p <= 3; ++p) {
    const Mesh<2> m = rectangle(3, 2);
    const NedelecDofMap<2> dofs(m, p);
    const Constraints c = bloch_constraints<2>(dofs, pairs);
    REQUIRE(check_one_to_one(c, phase) == 3 * p);  // three slave edges with p DoFs each
    // the constraints are exactly those of the y = 1 facet DoFs
    for (const Index f : m.facets_with_tag(box_tag::kYMax)) {
      for (const Index d : dofs.facet_dofs(f)) REQUIRE(c.is_constrained(d));
    }
    for (const Index f : m.facets_with_tag(box_tag::kYMin)) {
      for (const Index d : dofs.facet_dofs(f)) REQUIRE_FALSE(c.is_constrained(d));
    }
  }
  // a randomly renumbered mesh (edges may flip orientation between the two sides)
  auto [relabelled, perm] = hpfem::mesh::testing::relabel(rectangle(3, 2), 7);
  tag_unit_box_sides(relabelled);
  const NedelecDofMap<2> dofs(relabelled, 3);
  const Constraints c = bloch_constraints<2>(dofs, pairs);
  REQUIRE(check_one_to_one(c, phase) == 9);
  // mismatched sides are rejected
  const std::vector<PeriodicPair<2>> bad{
      PeriodicPair<2>{box_tag::kYMin, box_tag::kYMax, Point<2>(0.3, 1.0), phase}};
  REQUIRE_THROWS_AS(bloch_constraints<2>(dofs, bad), hpfem::InvalidArgument);
}

TEST_CASE("bloch_constraints: two directions in 3D, corner edges chain the phases",
          "[assembly][periodic]") {
  const Point<3> k(0.0, 0.4, -1.1);
  const PeriodicPair<3> py{box_tag::kYMin, box_tag::kYMax, Point<3>(0.0, 1.0, 0.0),
                           bloch_phase<3>(k, Point<3>(0.0, 1.0, 0.0))};
  const PeriodicPair<3> pz{box_tag::kZMin, box_tag::kZMax, Point<3>(0.0, 0.0, 1.0),
                           bloch_phase<3>(k, Point<3>(0.0, 0.0, 1.0))};
  const std::vector<PeriodicPair<3>> pairs{py, pz};
  auto [m, perm] = hpfem::mesh::testing::relabel(box(2, 2, 2), 11);
  tag_unit_box_sides(m);
  const NedelecDofMap<3> dofs(m, 2);
  const Constraints c = bloch_constraints<3>(dofs, pairs);
  Index corner = 0;
  for (Index i = 0; i < dofs.num_dofs(); ++i) {
    if (!c.is_constrained(i)) continue;
    const auto terms = c.terms(i);
    for (const auto& t : terms) REQUIRE_FALSE(c.is_constrained(t.master));
    const Real magnitude = std::abs(terms.size() == 1 ? terms[0].coefficient : Complex{1.0, 0.0});
    REQUIRE(magnitude == Approx(1.0).margin(1e-9));
  }
  // edges on the corner line y = 1, z = 1 resolve to y = 0, z = 0 with the product phase
  for (Index e = 0; e < m.num_edges(); ++e) {
    const auto& ev = m.edge_vertices(e);
    const Point<3> a = m.vertex(ev[0]);
    const Point<3> b = m.vertex(ev[1]);
    if (std::abs(a(1) - 1.0) > 1e-12 || std::abs(a(2) - 1.0) > 1e-12) continue;
    if (std::abs(b(1) - 1.0) > 1e-12 || std::abs(b(2) - 1.0) > 1e-12) continue;
    for (const Index d : dofs.edge_dofs(e)) {
      const auto terms = c.terms(d);
      REQUIRE(terms.size() == 1);
      const Complex expected = py.phase * pz.phase;
      const Complex ratio = terms[0].coefficient / expected;
      REQUIRE(std::abs(std::abs(ratio.real()) - 1.0) < 1e-9);
      REQUIRE(std::abs(ratio.imag()) < 1e-9);
      ++corner;
    }
  }
  REQUIRE(corner == 2 * 2);  // two corner edges with two DoFs each
}

TEST_CASE("Scattering with Bloch-periodic sides reproduces a plane wave on a renumbered mesh",
          "[assembly][periodic][physics]") {
  const Real k0 = 4.0;
  const Real angle = 0.6;
  const Point<2> k = k0 * Point<2>(std::cos(angle), std::sin(angle));
  const auto wave = plane_wave<2>(
      ComplexVector<2>(Complex{-std::sin(angle), 0.0}, Complex{std::cos(angle), 0.0}), k);
  auto [m, perm] = hpfem::mesh::testing::relabel(rectangle(4, 4), 3);
  tag_unit_box_sides(m);
  const NedelecDofMap<2> dofs(m, 3);
  ScatteringSetup<2> setup;
  setup.omega = k0 * hpfem::constants::c0;
  setup.incident = wave;
  setup.formulation = Formulation::kTotalField;
  setup.incident_tags = {box_tag::kXMin, box_tag::kXMax};
  setup.periodic = {PeriodicPair<2>{box_tag::kYMin, box_tag::kYMax, Point<2>(0.0, 1.0),
                                    bloch_phase<2>(k, Point<2>(0.0, 1.0))}};
  const Scattering<2> problem(dofs, setup);
  const auto solution = problem.solve();
  const auto err = problem.error(solution, wave);
  // plumbing check (p = 3 on a 4 x 4 mesh); the rates are in tests/convergence
  REQUIRE(std::hypot(err.l2, err.curl) / std::hypot(err.l2_norm, err.curl_norm) < 1e-2);
  // the solution satisfies the constraints
  const Constraints c = problem.constraints();
  for (Index i = 0; i < dofs.num_dofs(); ++i) {
    if (!c.is_constrained(i)) continue;
    Complex expected = 0;
    for (const auto& t : c.terms(i)) expected += t.coefficient * solution.unknown(t.master);
    REQUIRE(std::abs(solution.unknown(i) - expected) < 1e-12);
  }
}

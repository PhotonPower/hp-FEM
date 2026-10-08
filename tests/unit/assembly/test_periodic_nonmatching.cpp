// Non-matching Bloch coupling (M15 F16 stage 2): different polynomial orders on the two
// sides, one side refined without the other, unrelated facet layouts, in 2D for the Nédélec
// and H1 spaces and in 3D for matched faces with different orders. The coupled space is
// conforming: a Bloch function with a trace in the common space obeys the constraints, the
// solution of a constrained problem has a continuous tangential trace (up to the phase)
// across the faces, and the surplus modes of the coarse side are zero.
#include <array>
#include <cmath>
#include <numbers>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "hpfem/adaptivity/residual_estimator.hpp"
#include "hpfem/assembly/interpolation.hpp"
#include "hpfem/assembly/periodic.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/adaptive_mesh.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/point_location.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"

using hpfem::as_size;
using hpfem::Complex;
using hpfem::Index;
using hpfem::kI;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::assembly::bloch_constraints;
using hpfem::assembly::bloch_phase;
using hpfem::assembly::ComplexVector;
using hpfem::assembly::PeriodicLocator;
using hpfem::assembly::PeriodicPair;
using hpfem::fespace::Constraints;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

/// Per-cell orders: `left` for cells with centroid x < 0.5, `right` otherwise.
template <int Dim>
std::vector<int> split_orders(const Mesh<Dim>& mesh, int left, int right) {
  std::vector<int> orders(as_size(mesh.num_cells()));
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    orders[as_size(c)] = hpfem::mesh::affine_map(mesh, c).centroid()(0) < 0.5 ? left : right;
  }
  return orders;
}

/// The interpolant of a Bloch function obeys every constraint (to 1e-10 relative).
template <int Dim>
void check_interpolant(const DofMap<Dim>& h1, const Constraints& c,
                       const std::function<Complex(const Point<Dim>&)>& u_exact) {
  const Vector u =
      hpfem::assembly::interpolate<Dim>(h1, hpfem::assembly::physical_sampler<Dim>(u_exact));
  REQUIRE(c.num_constrained() > 0);
  for (Index dof = 0; dof < h1.num_dofs(); ++dof) {
    if (!c.is_constrained(dof)) continue;
    Complex combination = 0;
    for (const auto& term : c.terms(dof)) combination += term.coefficient * u(term.master);
    REQUIRE(std::abs(combination - u(dof)) < 1e-10 * u.norm());
  }
}

/// Solves the in-plane plane-wave problem (total field, incident traces on y_min / y_max,
/// Bloch in x with the x-faces as the periodic pair), checks the tangential continuity
/// E_t(x = 1, y) = phase E_t(x = 0, y) at sampled points and the constraint residual, and
/// returns the relative H(curl) error against the plane wave.
Real check_solve_conformity(const Mesh<2>& mesh, const std::vector<int>& orders) {
  const Real k0 = 3.0;
  const Real angle = 0.5;
  const Point<2> k = k0 * Point<2>(std::cos(angle), std::sin(angle));
  const auto wave = hpfem::physics::plane_wave<2>(
      ComplexVector<2>(Complex{-std::sin(angle), 0.0}, Complex{std::cos(angle), 0.0}), k);
  const NedelecDofMap<2> dofs(mesh, orders);
  hpfem::physics::ScatteringSetup<2> setup;
  setup.omega = k0 * hpfem::constants::c0;
  setup.incident = wave;
  setup.formulation = hpfem::physics::Formulation::kTotalField;
  setup.incident_tags = {box_tag::kYMin, box_tag::kYMax};
  const Complex phase = bloch_phase<2>(k, Point<2>(1.0, 0.0));
  setup.periodic = {PeriodicPair<2>{box_tag::kXMin, box_tag::kXMax, Point<2>(1.0, 0.0), phase}};
  const hpfem::physics::Scattering<2> problem(dofs, setup);
  const auto solution = problem.solve();
  const auto err = problem.error(solution, wave);
  const Real relative = std::hypot(err.l2, err.curl) / std::hypot(err.l2_norm, err.curl_norm);
  REQUIRE(relative < 0.2);  // plumbing; the accuracy is compared with the matching space below
  const Constraints c = problem.constraints();
  for (Index i = 0; i < dofs.num_dofs(); ++i) {
    if (!c.is_constrained(i)) continue;
    Complex expected = 0;
    for (const auto& t : c.terms(i)) expected += t.coefficient * solution.unknown(t.master);
    REQUIRE(std::abs(solution.unknown(i) - expected) < 1e-12);
  }
  // tangential trace continuity across the faces, sampled strictly inside the cells
  const hpfem::mesh::PointLocator<2> locator(mesh);
  Real scale = 0;
  Real worst = 0;
  for (int j = 0; j < 23; ++j) {
    const Real y = (j + 0.37) / 23.0;
    const auto left = locator.locate(Point<2>(1e-7, y));
    const auto right = locator.locate(Point<2>(1.0 - 1e-7, y));
    REQUIRE(left);
    REQUIRE(right);
    const ComplexVector<2> el =
        hpfem::assembly::evaluate_hcurl<2>(dofs, solution.unknown, left->cell, left->xi);
    const ComplexVector<2> er =
        hpfem::assembly::evaluate_hcurl<2>(dofs, solution.unknown, right->cell, right->xi);
    worst = std::max(worst, std::abs(er(1) - phase * el(1)));  // tangential component E_y
    scale = std::max(scale, std::abs(el(1)));
  }
  REQUIRE(scale > 0);
  // the basis is evaluated 1e-7 inside the cells: continuity up to that offset
  REQUIRE(worst < 1e-5 * scale);
  return relative;
}

Mesh<2> unstructured_square() {
  // left face vertices at y = 0, 0.4, 1 and right face at y = 0, 0.6, 1: no facet of one
  // side coincides with or nests in a facet of the other
  std::vector<Point<2>> vertices{Point<2>(0.0, 0.0), Point<2>(0.0, 0.4), Point<2>(0.0, 1.0),
                                 Point<2>(1.0, 0.0), Point<2>(1.0, 0.6), Point<2>(1.0, 1.0),
                                 Point<2>(0.5, 0.5)};
  std::vector<std::array<Index, 3>> cells{{0, 3, 6}, {3, 4, 6}, {4, 5, 6},
                                          {5, 2, 6}, {2, 1, 6}, {1, 0, 6}};
  Mesh<2> mesh(vertices, cells);
  for (const Index f : mesh.boundary_facets()) {
    Point<2> c = Point<2>::Zero();
    for (const Index v : mesh.facet_vertices(f)) c += mesh.vertex(v);
    c /= 2.0;
    if (std::abs(c(0)) < 1e-12) mesh.set_facet_tag(f, box_tag::kXMin);
    if (std::abs(c(0) - 1.0) < 1e-12) mesh.set_facet_tag(f, box_tag::kXMax);
    if (std::abs(c(1)) < 1e-12) mesh.set_facet_tag(f, box_tag::kYMin);
    if (std::abs(c(1) - 1.0) < 1e-12) mesh.set_facet_tag(f, box_tag::kYMax);
  }
  return mesh;
}

}  // namespace

TEST_CASE("non-matching Bloch: different orders on the two sides (2D, H1 and Nedelec)",
          "[assembly][periodic][nonmatching]") {
  const Mesh<2> mesh = rectangle(2, 3);
  const Point<2> k(0.9, 0.0);
  const Complex phase = bloch_phase<2>(k, Point<2>(1.0, 0.0));
  const std::vector<PeriodicPair<2>> pairs{
      PeriodicPair<2>{box_tag::kXMin, box_tag::kXMax, Point<2>(1.0, 0.0), phase}};
  for (const auto& [left, right] : std::vector<std::pair<int, int>>{{2, 4}, {4, 2}, {3, 3}}) {
    const std::vector<int> orders = split_orders(mesh, left, right);
    const DofMap<2> h1(mesh, orders);
    const Constraints c = bloch_constraints<2>(h1, pairs);
    // every slave DoF is constrained; master edge DoFs of degree above the common order are
    // constrained to zero
    const int common = std::min(left, right);
    for (const Index f : mesh.facets_with_tag(box_tag::kXMax)) {
      for (const Index d : h1.facet_dofs(f)) REQUIRE(c.is_constrained(d));
    }
    for (const Index f : mesh.facets_with_tag(box_tag::kXMin)) {
      const auto ed = h1.edge_dofs(f);
      for (Index i = 0; i < static_cast<Index>(ed.size()); ++i) {
        const Index d = ed[as_size(i)];
        if (i < hpfem::fespace::H1Counts::edge(common)) {
          REQUIRE_FALSE(c.is_constrained(d));
        } else {
          REQUIRE(c.is_constrained(d));
          for (const auto& t : c.terms(d)) REQUIRE(std::abs(t.coefficient) == 0.0);
        }
      }
    }
    // a Bloch function with a quadratic trace lies in both trace spaces
    check_interpolant<2>(h1, c, [k](const Point<2>& x) {
      return (1.0 + 0.5 * x(1) * x(1) - 0.2 * x(1)) * std::exp(kI * k.dot(x));
    });
    // the coupled space contains the uniform space of the common order: no worse than it
    const Real mixed = check_solve_conformity(mesh, orders);
    const Real uniform =
        check_solve_conformity(mesh, std::vector<int>(as_size(mesh.num_cells()), common));
    REQUIRE(mixed <= uniform * (1 + 1e-9));
  }
}

TEST_CASE("non-matching Bloch: one side refined without the other (2D)",
          "[assembly][periodic][nonmatching]") {
  const Point<2> k(0.9, 0.0);
  const Complex phase = bloch_phase<2>(k, Point<2>(1.0, 0.0));
  const std::vector<PeriodicPair<2>> pairs{
      PeriodicPair<2>{box_tag::kXMin, box_tag::kXMax, Point<2>(1.0, 0.0), phase}};
  for (const Real refined_side : {1.0, 0.0}) {  // slave side finer, then master side finer
    hpfem::mesh::AdaptiveMesh<2> adaptive(rectangle(2, 2));
    std::vector<Index> marked;
    for (Index c = 0; c < adaptive.mesh().num_cells(); ++c) {
      const Point<2> centroid = hpfem::mesh::affine_map(adaptive.mesh(), c).centroid();
      if (std::abs(centroid(0) - refined_side) < 0.5) marked.push_back(c);
    }
    adaptive.refine(marked);
    const Mesh<2>& mesh = adaptive.mesh();
    REQUIRE(mesh.facets_with_tag(box_tag::kXMin).size() !=
            mesh.facets_with_tag(box_tag::kXMax).size());
    for (const int p : {2, 3}) {
      const DofMap<2> h1(mesh, p);
      const Constraints c = bloch_constraints<2>(h1, pairs);
      check_interpolant<2>(h1, c, [k](const Point<2>& x) {
        return (1.0 + 0.5 * x(1) * x(1) - 0.2 * x(1)) * std::exp(kI * k.dot(x));
      });
      // the half-refined mesh is at least as accurate as the unrefined one
      const Real refined =
          check_solve_conformity(mesh, std::vector<int>(as_size(mesh.num_cells()), p));
      const Mesh<2> root = rectangle(2, 2);
      const Real unrefined =
          check_solve_conformity(root, std::vector<int>(as_size(root.num_cells()), p));
      REQUIRE(refined <= unrefined * (1 + 1e-9));
    }
    // the locator finds the master cell of a point on a slave facet
    const PeriodicLocator<2> locator(mesh, pairs);
    REQUIRE_FALSE(locator.empty());
    for (const Index f : mesh.facets_with_tag(box_tag::kXMax)) {
      REQUIRE(locator.is_slave(f));
      REQUIRE_FALSE(locator.is_master(f));
      Point<2> mid = Point<2>::Zero();
      for (const Index v : mesh.facet_vertices(f)) mid += mesh.vertex(v);
      mid /= 2.0;
      const auto partner = locator.partner(f, mid);
      REQUIRE(partner);
      REQUIRE(std::abs(partner->phase - phase) < 1e-14);
      const Point<2> back =
          hpfem::mesh::cell_geometry(mesh, partner->cell)->evaluate(partner->xi).x;
      REQUIRE((back - (mid - Point<2>(1.0, 0.0))).norm() < 1e-10);
    }
    REQUIRE_FALSE(locator.partner(mesh.facets_with_tag(box_tag::kXMin)[0], Point<2>(0.0, 0.5)));
  }
}

TEST_CASE("non-matching Bloch: unrelated facet layouts fall back to interpolation (2D)",
          "[assembly][periodic][nonmatching]") {
  const Mesh<2> mesh = unstructured_square();
  const Point<2> k(0.9, 0.0);
  const Complex phase = bloch_phase<2>(k, Point<2>(1.0, 0.0));
  const std::vector<PeriodicPair<2>> pairs{
      PeriodicPair<2>{box_tag::kXMin, box_tag::kXMax, Point<2>(1.0, 0.0), phase}};
  const DofMap<2> h1(mesh, 2);
  const Constraints c = bloch_constraints<2>(h1, pairs);
  REQUIRE(c.num_constrained() == static_cast<Index>(h1.dofs_on_tag(box_tag::kXMax).size()));
  // a linear trace is reproduced exactly by the piecewise interpolation
  check_interpolant<2>(
      h1, c, [k](const Point<2>& x) { return (1.0 + 0.2 * x(1)) * std::exp(kI * k.dot(x)); });
  const NedelecDofMap<2> nd(mesh, 2);
  REQUIRE_NOTHROW(bloch_constraints<2>(nd, pairs));
}

TEST_CASE("non-matching Bloch: different orders on matched faces (3D, H1)",
          "[assembly][periodic][nonmatching]") {
  const hpfem::mesh::Mesh<3> mesh = hpfem::mesh::box(2, 2, 2);
  const Point<3> k(0.7, 0.0, 0.0);
  const Complex phase = bloch_phase<3>(k, Point<3>(1.0, 0.0, 0.0));
  const std::vector<PeriodicPair<3>> pairs{
      PeriodicPair<3>{box_tag::kXMin, box_tag::kXMax, Point<3>(1.0, 0.0, 0.0), phase}};
  const std::vector<int> orders = split_orders<3>(mesh, 2, 3);
  const DofMap<3> h1(mesh, orders);
  const Constraints c = bloch_constraints<3>(h1, pairs);
  for (const Index f : mesh.facets_with_tag(box_tag::kXMax)) {
    for (const Index d : h1.facet_dofs(f)) REQUIRE(c.is_constrained(d));
  }
  check_interpolant<3>(h1, c, [k](const Point<3>& x) {
    return (1.0 + 0.5 * x(1) * x(1) + 0.3 * x(1) * x(2) - 0.2 * x(2)) * std::exp(kI * k.dot(x));
  });
  const NedelecDofMap<3> nd(mesh, orders);
  REQUIRE_NOTHROW(bloch_constraints<3>(nd, pairs));
}

TEST_CASE("residual estimator: Bloch facets contribute the phase-shifted jump to both sides",
          "[assembly][periodic][nonmatching][adaptivity]") {
  const Real k0 = 3.0;
  const Real angle = 0.5;
  const Point<2> k = k0 * Point<2>(std::cos(angle), std::sin(angle));
  const auto wave = hpfem::physics::plane_wave<2>(
      ComplexVector<2>(Complex{-std::sin(angle), 0.0}, Complex{std::cos(angle), 0.0}), k);
  // the slave side refined once: nested facets, partner cells vary along a master facet
  hpfem::mesh::AdaptiveMesh<2> adaptive(rectangle(3, 3));
  std::vector<Index> marked;
  for (Index c = 0; c < adaptive.mesh().num_cells(); ++c) {
    if (hpfem::mesh::affine_map(adaptive.mesh(), c).centroid()(0) > 2.0 / 3.0) marked.push_back(c);
  }
  adaptive.refine(marked);
  const Mesh<2>& mesh = adaptive.mesh();
  const NedelecDofMap<2> dofs(mesh, 2);
  hpfem::physics::ScatteringSetup<2> setup;
  setup.omega = k0 * hpfem::constants::c0;
  setup.incident = wave;
  setup.formulation = hpfem::physics::Formulation::kTotalField;
  setup.incident_tags = {box_tag::kYMin, box_tag::kYMax};
  const Complex phase = bloch_phase<2>(k, Point<2>(1.0, 0.0));
  setup.periodic = {PeriodicPair<2>{box_tag::kXMin, box_tag::kXMax, Point<2>(1.0, 0.0), phase}};
  const hpfem::physics::Scattering<2> problem(dofs, setup);
  const auto solution = problem.solve();
  const auto with = problem.estimate(solution);  // the problem passes its periodic pairs
  const auto without = hpfem::adaptivity::residual_estimate<2>(
      dofs, solution.unknown, k0 * k0, [&](Index c) { return problem.form_of_cell(c); });
  REQUIRE(with.parts.size() == without.parts.size());
  std::vector<bool> on_face(as_size(mesh.num_cells()), false);
  for (const hpfem::mesh::Tag tag : {box_tag::kXMin, box_tag::kXMax}) {
    for (const Index f : mesh.facets_with_tag(tag)) on_face[as_size(mesh.facet_cells(f)[0])] = true;
  }
  Real face_extra = 0;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const auto& a = with.parts[as_size(c)];
    const auto& b = without.parts[as_size(c)];
    // element terms identical, facet terms only grow, and only on the face cells
    REQUIRE(a.element == b.element);
    REQUIRE(a.tangential_jump >= b.tangential_jump * (1 - 1e-12));
    REQUIRE(a.normal_jump >= b.normal_jump * (1 - 1e-12));
    if (on_face[as_size(c)]) {
      REQUIRE(a.tangential_jump > b.tangential_jump);
      face_extra += a.tangential_jump - b.tangential_jump;
    } else {
      REQUIRE(a.tangential_jump == b.tangential_jump);
    }
  }
  // the extra is a facet jump of the discrete solution: of the size of the interior jumps
  Real interior = 0;
  for (Index c = 0; c < mesh.num_cells(); ++c)
    interior += without.parts[as_size(c)].tangential_jump;
  REQUIRE(face_extra > 1e-3 * interior);
  REQUIRE(face_extra < 10 * interior);
}

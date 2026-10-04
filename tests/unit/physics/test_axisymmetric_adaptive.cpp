// The axisymmetric solvers on locally refined (one-irregular) meridian meshes: the hanging
// DoFs are constrained, so the cavity modes of the PEC cylinder stay at the Bessel values,
// the manufactured corner problem improves under local refinement and the discrete
// solution satisfies the constraints; the estimator and the error of the scattering class
// (the estimator localises the edge only once the smooth part is resolved: convergence test).
#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "axisymmetric_corner.hpp"
#include "hpfem/assembly/hanging_constraints.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/adaptive_mesh.hpp"
#include "hpfem/mesh/generators.hpp"

using Catch::Approx;
using hpfem::as_size;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::AdaptiveMesh;
using hpfem::mesh::Mesh;
using hpfem::physics::AxisymmetricCavity;
using hpfem::physics::AxisymmetricCavitySetup;
using hpfem::physics::AxisymmetricScattering;

namespace {

constexpr hpfem::mesh::Tag kAxis = 77;
constexpr hpfem::mesh::Tag kWall = 9;

/// Meridian mesh of the PEC cylinder of radius 1 and height 1.5.
Mesh<2> cylinder(Index n) {
  Mesh<2> m = hpfem::mesh::rectangle(n, n, Point<2>(0.0, 0.0), Point<2>(1.0, 1.5));
  for (const Index f : m.boundary_facets()) {
    const auto& fv = m.facet_vertices(f);
    const bool axis = std::abs(m.vertex(fv[0])(0)) < 1e-12 && std::abs(m.vertex(fv[1])(0)) < 1e-12;
    m.set_facet_tag(f, axis ? kAxis : kWall);
  }
  return m;
}

/// Cells whose centroid lies in the box.
std::vector<Index> cells_in(const Mesh<2>& mesh, Real r0, Real r1, Real z0, Real z1) {
  std::vector<Index> out;
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const Point<2> x = hpfem::mesh::affine_map(mesh, c).centroid();
    if (x(0) > r0 && x(0) < r1 && x(1) > z0 && x(1) < z1) out.push_back(c);
  }
  return out;
}

}  // namespace

TEST_CASE("axisymmetric cavity on a one-irregular mesh: TM_010 and TE_111 of the cylinder",
          "[physics][axisymmetric][adaptivity]") {
  // k^2 = (j_01 / a)^2 for TM_010 (m = 0), (j'_11 / a)^2 + (pi / h)^2 for TE_111 (m = 1)
  const Real tm010 = 2.404825557695773;
  const Real te111 = std::sqrt(1.841183781340659 * 1.841183781340659 +
                               (hpfem::constants::pi / 1.5) * (hpfem::constants::pi / 1.5));
  AdaptiveMesh<2> adaptive(cylinder(4));
  adaptive.refine(cells_in(adaptive.mesh(), 0.5, 1.0, 0.0, 0.75));
  const Mesh<2>& mesh = adaptive.mesh();
  REQUIRE_FALSE(mesh.is_conforming());
  const NedelecDofMap<2> nd(mesh, 2);
  const DofMap<2> h1(mesh, 2);
  for (const int m : {0, 1}) {
    AxisymmetricCavitySetup setup;
    setup.pec_tags = {kWall};
    setup.axis_tag = kAxis;
    setup.azimuthal_order = m;
    setup.num_modes = 4;
    const AxisymmetricCavity cavity(nd, h1, setup);
    const auto modes = cavity.solve();
    REQUIRE(modes.size() >= 1);
    const Real reference = m == 0 ? tm010 : te111;
    REQUIRE(modes.front().wavenumber == Approx(reference).epsilon(5e-3));
    REQUIRE(modes.front().meridian.size() == nd.num_dofs());
    // the expanded eigenvector satisfies the hanging-node constraints of both spaces
    const auto nd_c = hpfem::assembly::hanging_constraints(nd);
    const auto h1_c = hpfem::assembly::hanging_constraints(h1);
    REQUIRE(nd_c.num_constrained() > 0);
    const Real scale = modes.front().meridian.norm();
    for (Index d = 0; d < nd.num_dofs(); ++d) {
      if (!nd_c.is_constrained(d)) continue;
      Complex sum = 0;
      for (const auto& t : nd_c.terms(d)) sum += t.coefficient * modes.front().meridian(t.master);
      REQUIRE(std::abs(sum - modes.front().meridian(d)) < 1e-10 * scale);
    }
    for (Index d = 0; d < h1.num_dofs(); ++d) {
      if (!h1_c.is_constrained(d)) continue;
      Complex sum = 0;
      for (const auto& t : h1_c.terms(d)) sum += t.coefficient * modes.front().azimuthal(t.master);
      REQUIRE(std::abs(sum - modes.front().azimuthal(d)) < 1e-10 * scale);
    }
  }
}

TEST_CASE("axisymmetric scattering at the re-entrant edge: local refinement reduces the error",
          "[physics][axisymmetric][adaptivity]") {
  using namespace hpfem::physics::test;
  AdaptiveMesh<2> adaptive(corner_mesh(2));
  const auto exact = corner_field();
  std::vector<Real> errors;
  std::vector<Real> estimates;
  for (int step = 0; step < 3; ++step) {
    const Mesh<2>& mesh = adaptive.mesh();
    const NedelecDofMap<2> nd(mesh, 2);
    const DofMap<2> h1(mesh, 2);
    const AxisymmetricScattering problem(nd, h1, corner_setup());
    const auto field = problem.solve();
    const auto e = problem.error(field, exact);
    const auto estimate = problem.estimate(field);
    REQUIRE(estimate.indicators.size() == as_size(mesh.num_cells()));
    errors.push_back(std::hypot(e.l2, e.curl));
    estimates.push_back(estimate.total());
    // the solution satisfies the hanging-node constraints
    if (!mesh.is_conforming()) {
      const auto nd_c = hpfem::assembly::hanging_constraints(nd);
      const Real scale = field.meridian.norm();
      for (Index d = 0; d < nd.num_dofs(); ++d) {
        if (!nd_c.is_constrained(d)) continue;
        Complex sum = 0;
        for (const auto& t : nd_c.terms(d)) sum += t.coefficient * field.meridian(t.master);
        REQUIRE(std::abs(sum - field.meridian(d)) < 1e-10 * scale);
      }
    }
    std::vector<Index> corner;
    for (Index c = 0; c < mesh.num_cells(); ++c) {
      if (touches_corner(mesh, c)) corner.push_back(c);
    }
    adaptive.refine(corner);
  }
  REQUIRE_FALSE(adaptive.mesh().is_conforming());
  for (std::size_t i = 1; i < errors.size(); ++i) REQUIRE(errors[i] < 0.9 * errors[i - 1]);
  for (std::size_t i = 0; i < errors.size(); ++i) {
    REQUIRE(estimates[i] / errors[i] > 0.1);
    REQUIRE(estimates[i] / errors[i] < 50.0);
  }
}

TEST_CASE("axisymmetric error: zero for the interpolated exact field of a polynomial mode",
          "[physics][axisymmetric]") {
  // E = nabla(psi e^{i phi}) with psi = r z: (E_r, v, E_z) = (z, r z, r), curl-free
  const Mesh<2> mesh = cylinder(2);
  const NedelecDofMap<2> nd(mesh, 2);
  const DofMap<2> h1(mesh, 2);
  const Vector psi = hpfem::assembly::interpolate<2>(
      h1, hpfem::assembly::physical_sampler<2>(
              [](const Point<2>& x) { return Complex{x(0) * x(1), 0.0}; }));
  const Vector block = hpfem::assembly::axisymmetric_gradient(h1, nd, 1) * psi;
  const auto exact = [](const Point<2>& x) {
    return Eigen::Matrix<Complex, 3, 1>(Complex{x(1), 0.0}, Complex{x(0) * x(1), 0.0},
                                        Complex{x(0), 0.0});
  };
  const auto e = hpfem::physics::axisymmetric_error(nd, h1, block.head(nd.num_dofs()),
                                                    block.tail(h1.num_dofs()), 1, exact);
  REQUIRE(e.l2 < 1e-12);
  REQUIRE(e.curl < 1e-12);
  // against a different field the error is the weighted norm of the difference: E - 0
  const auto zero = [](const Point<2>&) { return Eigen::Matrix<Complex, 3, 1>::Zero().eval(); };
  const auto norm = hpfem::physics::axisymmetric_error(nd, h1, block.head(nd.num_dofs()),
                                                       block.tail(h1.num_dofs()), 1, zero);
  // int (z^2 r + r^2 z^2 / r + r^2 r) dr dz over [0,1] x [0,1.5] = 1.125/2 + 1.125/2 + 1.5/4
  REQUIRE(norm.l2 == Approx(std::sqrt(0.5625 + 0.5625 + 0.375)).epsilon(1e-10));
  REQUIRE(norm.curl < 1e-12);
  REQUIRE_THROWS_AS(hpfem::physics::axisymmetric_error(nd, h1, Vector::Zero(3),
                                                       block.tail(h1.num_dofs()), 1, exact),
                    hpfem::InvalidArgument);
}

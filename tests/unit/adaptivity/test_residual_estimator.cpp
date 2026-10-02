// The residual estimator must vanish for a discrete solution that is exact (all four terms:
// element residual with curl-curl, divergence residual, tangential and normal jumps), must
// localise a perturbation, and must respect the per-cell forms.
#include <algorithm>
#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/adaptivity/residual_estimator.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"

using Catch::Approx;
using hpfem::as_size;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::Vector;
using hpfem::adaptivity::Estimate;
using hpfem::adaptivity::EstimatorOptions;
using hpfem::adaptivity::residual_estimate;
using hpfem::assembly::ComplexCurl;
using hpfem::assembly::ComplexVector;
using hpfem::assembly::MaxwellForm;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::box;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::mesh::Tag;
using hpfem::physics::Formulation;
using hpfem::physics::IncidentField;
using hpfem::physics::Scattering;
using hpfem::physics::ScatteringSetup;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kWavenumber = 3.0;

template <int Dim>
std::vector<Tag> all_sides() {
  if constexpr (Dim == 2) {
    return {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  } else {
    return {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin,
            box_tag::kYMax, box_tag::kZMin, box_tag::kZMax};
  }
}

template <int Dim>
Mesh<Dim> unit_mesh(Index n) {
  if constexpr (Dim == 2) {
    return rectangle(n, n);
  } else {
    return box(n, n, n);
  }
}

/// A quadratic field in ND_3 (full P_2) with non-vanishing curl curl:
/// 2D: E = (y², x y), curl E = −y, curl curl E = (−1, 0);
/// 3D: E = (y², z², x²), curl E = (−2z, −2x, −2y), curl curl E = (−2, −2, −2).
template <int Dim>
IncidentField<Dim> quadratic_field() {
  IncidentField<Dim> f;
  if constexpr (Dim == 2) {
    f.value = [](const Point<2>& x) {
      return ComplexVector<2>(Complex{x(1) * x(1), 0.0}, Complex{x(0) * x(1), 0.0});
    };
    f.curl = [](const Point<2>& x) { return ComplexCurl<2>(Complex{-x(1), 0.0}); };
  } else {
    f.value = [](const Point<3>& x) {
      return ComplexVector<3>(Complex{x(1) * x(1), 0.0}, Complex{x(2) * x(2), 0.0},
                              Complex{x(0) * x(0), 0.0});
    };
    f.curl = [](const Point<3>& x) {
      return ComplexCurl<3>(Complex{-2 * x(2), 0.0}, Complex{-2 * x(0), 0.0},
                            Complex{-2 * x(1), 0.0});
    };
  }
  return f;
}

template <int Dim>
ComplexVector<Dim> curl_curl_quadratic() {
  if constexpr (Dim == 2) {
    return ComplexVector<2>(Complex{-1.0, 0.0}, Complex{0.0, 0.0});
  } else {
    return ComplexVector<3>(Complex{-2.0, 0.0}, Complex{-2.0, 0.0}, Complex{-2.0, 0.0});
  }
}

/// Total-field problem whose exact solution is the quadratic field: Dirichlet data on all
/// sides, source f = curl curl E − k² E.
template <int Dim>
ScatteringSetup<Dim> quadratic_setup() {
  ScatteringSetup<Dim> setup;
  setup.omega = kWavenumber * hpfem::constants::c0;
  setup.incident = quadratic_field<Dim>();
  setup.formulation = Formulation::kTotalField;
  setup.incident_tags = all_sides<Dim>();
  const auto exact = setup.incident;
  setup.current = [exact](const Point<Dim>& x) {
    return ComplexVector<Dim>(curl_curl_quadratic<Dim>() -
                              kWavenumber * kWavenumber * exact.value(x));
  };
  return setup;
}

template <int Dim>
void check_zero_residual(Index n) {
  const Mesh<Dim> mesh = unit_mesh<Dim>(n);
  const NedelecDofMap<Dim> dofs(mesh, 3);
  const Scattering<Dim> problem(dofs, quadratic_setup<Dim>());
  const auto solution = problem.solve();
  const auto error = problem.error(solution, problem.setup().incident);
  REQUIRE(error.l2 / error.l2_norm < 1e-9);  // the exact field lies in ND_3
  const Estimate estimate = problem.estimate(solution);
  REQUIRE(estimate.indicators.size() == as_size(mesh.num_cells()));
  // scale: ‖f‖ ~ k² ~ 9 on cells of size 1/n; everything must cancel to rounding
  CHECK(estimate.total() < 1e-7);
  for (const auto& parts : estimate.parts) {
    CHECK(parts.element < 1e-14);
    CHECK(parts.divergence < 1e-14);
    CHECK(parts.tangential_jump < 1e-14);
    CHECK(parts.normal_jump < 1e-14);
  }
}

}  // namespace

TEST_CASE("residual estimator vanishes for an exact discrete solution (2D)", "[adaptivity]") {
  check_zero_residual<2>(3);
}

TEST_CASE("residual estimator vanishes for an exact discrete solution (3D)", "[adaptivity]") {
  check_zero_residual<3>(2);
}

TEST_CASE("residual estimator localises a perturbed DoF", "[adaptivity]") {
  const Mesh<2> mesh = rectangle(6, 6);
  const NedelecDofMap<2> dofs(mesh, 3);  // the quadratic field lies in ND_3
  const Scattering<2> problem(dofs, quadratic_setup<2>());
  auto solution = problem.solve();
  // an interior edge near the centre: its cells are the only ones with element residuals
  Index edge = hpfem::kInvalidIndex;
  for (Index e = 0; e < mesh.num_edges() && edge == hpfem::kInvalidIndex; ++e) {
    const auto& v = mesh.edge_vertices(e);
    const Point<2> mid = 0.5 * (mesh.vertex(v[0]) + mesh.vertex(v[1]));
    if (!mesh.is_boundary_facet(e) && (mid - Point<2>(0.5, 0.5)).norm() < 0.1) edge = e;
  }
  REQUIRE(edge != hpfem::kInvalidIndex);
  solution.unknown(dofs.edge_dofs(edge)[0]) += Complex{0.3, 0.1};
  const Estimate estimate = problem.estimate(solution);
  const auto touched = mesh.edge_cells(edge);
  const Index worst = estimate.argmax();
  CHECK(std::find(touched.begin(), touched.end(), worst) != touched.end());
  // every cell with a non-zero indicator is a touched cell or a facet neighbour of one
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if (estimate.indicators[as_size(c)] < 1e-9) continue;
    bool near = std::find(touched.begin(), touched.end(), c) != touched.end();
    for (const Index t : touched) {
      for (const Index nb : mesh.cell_neighbors(t)) near = near || nb == c;
    }
    CHECK(near);
  }
  // the touched cells carry the element residual, the neighbours only jump terms
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const bool is_touched = std::find(touched.begin(), touched.end(), c) != touched.end();
    if (!is_touched) CHECK(estimate.parts[as_size(c)].element < 1e-14);
  }
}

TEST_CASE("residual estimator options and argument checks", "[adaptivity]") {
  const Mesh<2> mesh = rectangle(2, 2);
  const NedelecDofMap<2> dofs(mesh, 1);
  const Vector e_h = Vector::Ones(dofs.num_dofs());
  const auto form = [](Index) { return MaxwellForm<2>{}; };
  const Estimate with = residual_estimate<2>(dofs, e_h, 1.0, form);
  EstimatorOptions options;
  options.divergence_terms = false;
  const Estimate without = residual_estimate<2>(dofs, e_h, 1.0, form, options);
  REQUIRE(with.indicators.size() == 8);
  for (std::size_t c = 0; c < 8; ++c) {
    CHECK(without.parts[c].divergence == 0.0);
    CHECK(without.parts[c].normal_jump == 0.0);
    CHECK(without.parts[c].element == Approx(with.parts[c].element));
    CHECK(without.parts[c].tangential_jump == Approx(with.parts[c].tangential_jump));
  }
  CHECK(with.total() >= without.total());
  CHECK_THROWS_AS(residual_estimate<2>(dofs, Vector::Ones(3), 1.0, form), hpfem::InvalidArgument);
  options.difference_step = 0.0;
  CHECK_THROWS_AS(residual_estimate<2>(dofs, e_h, 1.0, form, options), hpfem::InvalidArgument);
}

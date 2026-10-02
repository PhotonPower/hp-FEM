// Goal-oriented estimation: point functionals, the weighted residual identity
// Σ_K r_K(w) = ℓ(w) − a(E_h, w) on hanging meshes, Fourier functionals, and the effectivity of
// the DWR estimate for a point value of a plane-wave solution.
#include <cmath>
#include <numbers>
#include <random>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/adaptivity/refinement.hpp"
#include "hpfem/adaptivity/residual_estimator.hpp"
#include "hpfem/assembly/functionals.hpp"
#include "hpfem/assembly/hanging_constraints.hpp"
#include "hpfem/assembly/interpolation.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/assembly/prolongation.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/adaptive_mesh.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/point_location.hpp"
#include "hpfem/physics/goal_oriented.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"

using Catch::Approx;
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
using hpfem::mesh::box;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
using hpfem::mesh::Tag;
using hpfem::physics::dwr_estimate;
using hpfem::physics::Formulation;
using hpfem::physics::IncidentField;
using hpfem::physics::plane_wave;
using hpfem::physics::point_value_functional;
using hpfem::physics::Scattering;
using hpfem::physics::ScatteringSetup;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

constexpr Real kWavenumber = 3.0;

Vector random_vector(Index n, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<Real> dist(-1.0, 1.0);
  Vector v(n);
  for (Index i = 0; i < n; ++i) v(i) = Complex{dist(gen), dist(gen)};
  return v;
}

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
IncidentField<Dim> wave() {
  if constexpr (Dim == 2) {
    return plane_wave<2>(ComplexVector<2>(Complex{-0.8, 0.0}, Complex{0.6, 0.0}),
                         kWavenumber * Point<2>(0.6, 0.8));
  } else {
    return plane_wave<3>(ComplexVector<3>(Complex{0.0, 0.0}, Complex{1.0, 0.0}, Complex{0.0, 0.0}),
                         kWavenumber * Point<3>(0.6, 0.0, 0.8));
  }
}

template <int Dim>
ScatteringSetup<Dim> wave_setup() {
  ScatteringSetup<Dim> setup;
  setup.omega = kWavenumber * hpfem::constants::c0;
  setup.incident = wave<Dim>();
  setup.formulation = Formulation::kTotalField;
  setup.incident_tags = all_sides<Dim>();
  return setup;
}

template <int Dim>
Mesh<Dim> unit_mesh() {
  if constexpr (Dim == 2) {
    return rectangle(2, 2);
  } else {
    return box(2, 2, 2);
  }
}

/// Unit domain with the cell at the origin refined twice (hanging nodes).
template <int Dim>
AdaptiveMesh<Dim> refined() {
  AdaptiveMesh<Dim> adaptive(unit_mesh<Dim>());
  for (int step = 0; step < 2; ++step) {
    std::vector<Index> marked;
    const Mesh<Dim>& m = adaptive.mesh();
    for (Index c = 0; c < m.num_cells(); ++c) {
      for (const Index v : m.cell_vertices(c)) {
        if (m.vertex(v).norm() < 1e-12) marked.push_back(c);
      }
    }
    adaptive.refine(marked);
  }
  return adaptive;
}

/// Σ_K r_K(w) must equal w^T (b_{p+1} - A_{p+1} P u_h) for any w of the enriched space that
/// satisfies the hanging constraints.
template <int Dim>
void check_identity(const Mesh<Dim>& mesh, int p) {
  const NedelecDofMap<Dim> dofs(mesh, p);
  const Scattering<Dim> problem(dofs, wave_setup<Dim>());
  const auto solution = problem.solve();
  const NedelecDofMap<Dim> enriched(mesh, p + 1);
  const auto constraints = hpfem::assembly::hanging_constraints(enriched);
  const Vector w = constraints.expand(random_vector(constraints.num_free(), 3));
  const auto form = [&problem](Index c) { return problem.form_of_cell(c); };
  const auto contributions = hpfem::adaptivity::weighted_residual<Dim>(
      dofs, solution.unknown, problem.wavenumber() * problem.wavenumber(), form, enriched, w);
  REQUIRE(contributions.size() == as_size(mesh.num_cells()));
  Complex sum = 0;
  for (const Complex r : contributions) sum += r;
  auto system = hpfem::assembly::assemble_maxwell(enriched, form, 4);
  const hpfem::SparseMatrix a =
      system.stiffness - problem.wavenumber() * problem.wavenumber() * system.mass;
  const Vector u = hpfem::assembly::prolongate(dofs, solution.unknown, enriched,
                                               hpfem::adaptivity::identity_step(mesh.num_cells()));
  const Vector residual = system.rhs - a * u;
  const Complex expected = (w.transpose() * residual)(0);
  CHECK(std::abs(sum - expected) < 1e-9 * (1.0 + std::abs(expected)));
  CHECK(std::abs(expected) > 1e-6);  // the identity is not trivially 0 = 0
}

}  // namespace

TEST_CASE("point functional evaluates the discrete field", "[physics][goal]") {
  const Mesh<2> mesh = rectangle(3, 3);
  const NedelecDofMap<2> dofs(mesh, 2);
  const hpfem::mesh::PointLocator<2> locator(mesh);
  const Vector u = random_vector(dofs.num_dofs(), 1);
  const std::vector<Point<2>> points{Point<2>(0.3, 0.4), Point<2>(0.8, 0.1)};
  const std::vector<ComplexVector<2>> weights{
      ComplexVector<2>(Complex{1.0, 0.5}, Complex{0.0, 2.0}),
      ComplexVector<2>(Complex{0.0, 0.0}, Complex{1.0, 0.0})};
  const std::vector<ComplexCurl<2>> curl_weights{ComplexCurl<2>(Complex{0.0, 0.0}),
                                                 ComplexCurl<2>(Complex{3.0, -1.0})};
  const Vector q =
      hpfem::assembly::point_functional<2>(dofs, locator, points, weights, curl_weights);
  Complex expected = 0;
  for (std::size_t j = 0; j < points.size(); ++j) {
    const auto located = locator.locate(points[j]);
    REQUIRE(located);
    const auto e = hpfem::assembly::evaluate_hcurl(dofs, u, located->cell, located->xi);
    const auto c = hpfem::assembly::evaluate_hcurl_curl(dofs, u, located->cell, located->xi);
    expected += (e.transpose() * weights[j])(0) + (c.transpose() * curl_weights[j])(0);
  }
  CHECK(std::abs(hpfem::assembly::evaluate_functional(q, u) - expected) < 1e-12);
  const std::vector<Point<2>> outside{Point<2>(2.0, 2.0)};
  const std::vector<ComplexVector<2>> one{weights[0]};
  CHECK_THROWS_AS(hpfem::assembly::point_functional<2>(dofs, locator, outside, one),
                  hpfem::InvalidArgument);
}

TEST_CASE("weighted residual identity on hanging meshes (2D)", "[physics][goal]") {
  const AdaptiveMesh<2> adaptive = refined<2>();
  for (int p = 1; p <= 2; ++p) check_identity<2>(adaptive.mesh(), p);
}

TEST_CASE("weighted residual identity on hanging meshes (3D)", "[physics][goal]") {
  const AdaptiveMesh<3> adaptive = refined<3>();
  check_identity<3>(adaptive.mesh(), 1);
}

TEST_CASE("Fourier coefficient functional reproduces the plane-wave coefficient",
          "[physics][goal]") {
  // E = e0 exp(i k.x) with ky = ky0: the order-0 coefficient at x0 is e0_y exp(i kx x0)
  const Mesh<2> mesh = rectangle(4, 4);
  const NedelecDofMap<2> dofs(mesh, 6);
  const Real ky = 2.0;
  const Real kx = 1.5;
  const ComplexVector<2> e0(Complex{-ky, 0.0}, Complex{kx, 0.0});
  const auto field = plane_wave<2>(e0, Point<2>(kx, ky));
  const Vector u =
      hpfem::assembly::interpolate(dofs, hpfem::assembly::physical_sampler<2>(field.value));
  const Real x0 = 0.5;
  const auto functional = hpfem::physics::fourier_coefficient_functional(
      x0, 0.0, 1.0, ky, 0, 24, ComplexVector<2>(Complex{0.0, 0.0}, Complex{1.0, 0.0}));
  const Complex coefficient = hpfem::assembly::evaluate_functional(functional(dofs), u);
  const Complex exact = e0(1) * std::exp(Complex{0.0, kx * x0});
  CHECK(std::abs(coefficient - exact) < 1e-6);
  // a non-matching order integrates to ~0
  const auto other = hpfem::physics::fourier_coefficient_functional(
      x0, 0.0, 1.0, ky, 1, 24, ComplexVector<2>(Complex{0.0, 0.0}, Complex{1.0, 0.0}));
  CHECK(std::abs(hpfem::assembly::evaluate_functional(other(dofs), u)) < 1e-6);
}

TEST_CASE("DWR estimate of a point value: effectivity and localisation", "[physics][goal]") {
  const Point<2> x0(0.37, 0.62);
  const ComplexVector<2> weight(Complex{1.0, 0.0}, Complex{0.5, 0.0});
  const auto functional = point_value_functional<2>(x0, weight);
  const auto exact_field = wave<2>();
  const Complex exact = (exact_field.value(x0).transpose() * weight)(0);
  for (int p = 1; p <= 2; ++p) {
    for (const Index n : {4, 8}) {
      const Mesh<2> mesh = rectangle(n, n);
      const NedelecDofMap<2> dofs(mesh, p);
      const Scattering<2> problem(dofs, wave_setup<2>());
      const auto solution = problem.solve();
      const auto estimate = dwr_estimate<2>(problem, solution, functional);
      REQUIRE(estimate.indicators.size() == as_size(mesh.num_cells()));
      const Complex true_error = exact - estimate.value;
      INFO("p = " << p << ", n = " << n << ": true " << std::abs(true_error) << ", estimate "
                  << std::abs(estimate.error));
      CHECK(std::abs(true_error) > 0);
      CHECK(std::abs(estimate.error) / std::abs(true_error) > 0.3);
      CHECK(std::abs(estimate.error) / std::abs(true_error) < 3.0);
      // the signed estimate points the right way (beyond the coarsest p = 1 mesh, where
      // kh = 0.75 is pre-asymptotic and the estimate is 3x too large)
      if (p >= 2 || n >= 8) {
        CHECK(std::abs(estimate.error - true_error) < 0.7 * std::abs(true_error));
      }
      CHECK(estimate.total() >= std::abs(estimate.error) * (1 - 1e-12));
    }
  }
  const std::vector<Point<2>> none;
  CHECK_THROWS_AS(
      point_value_functional<2>(Point<2>(3.0, 3.0), weight)(NedelecDofMap<2>(rectangle(2, 2), 1)),
      hpfem::InvalidArgument);
}

TEST_CASE("DWR estimate on a hanging mesh with PEC and incident facets", "[physics][goal]") {
  const AdaptiveMesh<2> adaptive = refined<2>();
  const Mesh<2>& mesh = adaptive.mesh();
  const NedelecDofMap<2> dofs(mesh, 2);
  ScatteringSetup<2> setup = wave_setup<2>();
  const Scattering<2> problem(dofs, setup);
  const auto solution = problem.solve();
  const auto functional = point_value_functional<2>(
      Point<2>(0.7, 0.3), ComplexVector<2>(Complex{0.0, 0.0}, Complex{1.0, 0.0}));
  const auto estimate = dwr_estimate<2>(problem, solution, functional);
  const Complex exact = wave<2>().value(Point<2>(0.7, 0.3))(1);
  const Complex true_error = exact - estimate.value;
  CHECK(std::abs(estimate.error) / std::abs(true_error) > 0.3);
  CHECK(std::abs(estimate.error) / std::abs(true_error) < 3.0);
}

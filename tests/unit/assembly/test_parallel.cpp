// Parallel loops give the same results as serial ones: assembly (H1, Maxwell operator with
// condensation), the residual estimator and the weighted residual, with 1 thread and with
// all threads; parallel_for visits every index exactly once.
#include <atomic>
#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "hpfem/adaptivity/residual_estimator.hpp"
#include "hpfem/assembly/condensation.hpp"
#include "hpfem/assembly/h1_forms.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/parallel.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/adaptive_mesh.hpp"
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
using hpfem::fespace::DofMap;
using hpfem::fespace::NedelecDofMap;
using hpfem::mesh::Mesh;
using hpfem::mesh::rectangle;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

/// Restores the thread count on scope exit.
struct ThreadGuard {
  int previous = hpfem::num_threads();
  ~ThreadGuard() { hpfem::set_num_threads(previous); }
};

}  // namespace

TEST_CASE("parallel_for visits every index once", "[core][parallel]") {
  ThreadGuard guard;
  hpfem::set_num_threads(0);  // all cores
  const Index n = 10007;
  std::vector<int> visits(as_size(n), 0);
  std::vector<int> threads(as_size(n), -1);
  std::atomic<Index> sum{0};
  // no Catch2 assertions inside the body: the assertion handler is not thread-safe
  hpfem::parallel_for(n, [&](Index i, int thread) {
    visits[as_size(i)]++;
    threads[as_size(i)] = thread;
    sum += i;
  });
  for (const int v : visits) CHECK(v == 1);
  for (const int t : threads) {
    CHECK(t >= 0);
    CHECK(t < hpfem::num_threads());
  }
  CHECK(sum == n * (n - 1) / 2);
  CHECK(hpfem::has_openmp() == (hpfem::num_threads() > 1 || hpfem::has_openmp()));
}

TEST_CASE("parallel assembly and estimation agree with the serial results",
          "[assembly][adaptivity][parallel]") {
  hpfem::mesh::AdaptiveMesh<2> adaptive(rectangle(6, 6));
  const std::vector<Index> marked{0, 7, 20};
  adaptive.refine(marked);
  const Mesh<2>& mesh = adaptive.mesh();
  const int p = 3;
  const NedelecDofMap<2> nd(mesh, p);
  const DofMap<2> h1(mesh, p);
  hpfem::physics::ScatteringSetup<2> setup;
  setup.omega = 3.0 * hpfem::constants::c0;
  setup.incident = hpfem::physics::plane_wave<2>(
      ComplexVector<2>(Complex{-0.8, 0.0}, Complex{0.6, 0.0}), 3.0 * Point<2>(0.6, 0.8));
  setup.formulation = hpfem::physics::Formulation::kTotalField;
  setup.incident_tags = {box_tag::kXMin, box_tag::kXMax, box_tag::kYMin, box_tag::kYMax};
  const hpfem::physics::Scattering<2> problem(nd, setup);
  const auto form = [&problem](Index c) { return problem.form_of_cell(c); };
  hpfem::assembly::ScalarForm<2> scalar;
  scalar.diffusion = [](const Point<2>& x) { return Complex{1.0 + x(0), 0.2}; };
  scalar.source = [](const Point<2>& x) { return Complex{std::sin(x(0) + 2 * x(1)), 0.0}; };
  const auto scalar_form = [&scalar](Index) { return scalar; };
  const NedelecDofMap<2> enriched(mesh, p + 1);
  const Vector weight = Vector::Constant(enriched.num_dofs(), Complex{0.3, -0.1});

  ThreadGuard guard;
  hpfem::set_num_threads(1);
  const auto solution = problem.solve();
  hpfem::assembly::StaticCondensation cond1(nd.num_dofs());
  const auto maxwell1 = hpfem::assembly::assemble_maxwell_operator<2>(nd, form, 9.0, 2, &cond1);
  const auto h1_1 = hpfem::assembly::assemble_h1<2>(h1, scalar_form);
  const auto estimate1 = problem.estimate(solution);
  const auto weighted1 =
      hpfem::adaptivity::weighted_residual<2>(nd, solution.unknown, 9.0, form, enriched, weight);

  hpfem::set_num_threads(0);
  INFO("threads: " << hpfem::num_threads());
  const auto solution_n = problem.solve();
  hpfem::assembly::StaticCondensation cond_n(nd.num_dofs());
  const auto maxwell_n = hpfem::assembly::assemble_maxwell_operator<2>(nd, form, 9.0, 2, &cond_n);
  const auto h1_n = hpfem::assembly::assemble_h1<2>(h1, scalar_form);
  const auto estimate_n = problem.estimate(solution);
  const auto weighted_n =
      hpfem::adaptivity::weighted_residual<2>(nd, solution.unknown, 9.0, form, enriched, weight);

  CHECK((solution.unknown - solution_n.unknown).norm() < 1e-11 * solution.unknown.norm());
  CHECK((maxwell1.matrix - maxwell_n.matrix).norm() < 1e-12 * maxwell1.matrix.norm());
  CHECK((maxwell1.rhs - maxwell_n.rhs).norm() < 1e-12 * (maxwell1.rhs.norm() + 1e-300));
  CHECK(cond1.num_interior() == cond_n.num_interior());
  CHECK((cond1.recover(Vector::Ones(nd.num_dofs())) - cond_n.recover(Vector::Ones(nd.num_dofs())))
            .norm() < 1e-11 * std::sqrt(static_cast<Real>(nd.num_dofs())));
  CHECK((h1_1.matrix - h1_n.matrix).norm() < 1e-12 * h1_1.matrix.norm());
  CHECK((h1_1.rhs - h1_n.rhs).norm() < 1e-12 * h1_1.rhs.norm());
  REQUIRE(estimate1.indicators.size() == estimate_n.indicators.size());
  for (std::size_t c = 0; c < estimate1.indicators.size(); ++c) {
    CHECK(std::abs(estimate1.indicators[c] - estimate_n.indicators[c]) <=
          1e-12 * estimate1.total());
    CHECK(std::abs(weighted1[c] - weighted_n[c]) <= 1e-12 * (std::abs(weighted1[c]) + 1.0));
  }
}

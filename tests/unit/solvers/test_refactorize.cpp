// LinearSolver::refactorize (M15 F8): a second matrix on the pattern of the first reuses the
// analysis and gives the solution of a fresh factorisation; a different pattern or a first
// call falls back to factorize; every available backend.
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "hpfem/core/error.hpp"
#include "hpfem/solvers/linear_solver.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::Real;
using hpfem::SparseMatrix;
using hpfem::Vector;
using hpfem::solvers::available_backends;
using hpfem::solvers::backend_name;
using hpfem::solvers::DirectSolverBackend;
using hpfem::solvers::make_direct_solver;
using hpfem::solvers::Symmetry;

namespace {

/// Random sparse complex matrix with a dominant diagonal; `pattern_seed` fixes the
/// structure, `value_seed` the entries, so two matrices can share the pattern.
SparseMatrix random_system(Index n, unsigned pattern_seed, unsigned value_seed) {
  std::mt19937 structure(pattern_seed);
  std::mt19937 values(value_seed);
  std::uniform_real_distribution<Real> dist(-1.0, 1.0);
  std::uniform_int_distribution<Index> column(0, n - 1);
  std::vector<Eigen::Triplet<Complex, Index>> triplets;
  for (Index i = 0; i < n; ++i) {
    triplets.emplace_back(i, i, Complex{10.0 + dist(values), dist(values)});
    for (int k = 0; k < 4; ++k) {
      triplets.emplace_back(i, column(structure), Complex{dist(values), dist(values)});
    }
  }
  SparseMatrix a(n, n);
  a.setFromTriplets(triplets.begin(), triplets.end());
  a.makeCompressed();
  return a;
}

Vector random_vector(Index n, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<Real> dist(-1.0, 1.0);
  Vector v(n);
  for (Index i = 0; i < n; ++i) v(i) = Complex{dist(gen), dist(gen)};
  return v;
}

}  // namespace

TEST_CASE("refactorize reuses the analysis and matches a fresh factorisation", "[solvers]") {
  const Index n = 400;
  const SparseMatrix a1 = random_system(n, 1, 10);
  const SparseMatrix a2 = random_system(n, 1, 20);  // same pattern, other entries
  const SparseMatrix a3 = random_system(n, 2, 30);  // another pattern
  REQUIRE(a1.nonZeros() == a2.nonZeros());
  const Vector x = random_vector(n, 3);
  for (const DirectSolverBackend backend : available_backends()) {
    INFO(backend_name(backend));
    for (const Symmetry symmetry : {Symmetry::kGeneral, Symmetry::kDetect}) {
      auto solver = make_direct_solver(backend, symmetry);
      // a first call without a factorisation is a plain factorize
      solver->refactorize(a1);
      CHECK((solver->solve(a1 * x) - x).norm() < 1e-10 * x.norm());
      // the same pattern: numerical phase only, same solution as a fresh solver
      solver->refactorize(a2);
      auto fresh_solver = make_direct_solver(backend, symmetry);
      fresh_solver->factorize(a2);
      const Vector fresh = fresh_solver->solve(a2 * x);
      CHECK((solver->solve(a2 * x) - x).norm() < 1e-10 * x.norm());
      CHECK((solver->solve(a2 * x) - fresh).norm() < 1e-12 * x.norm());
      // another pattern: falls back to a full factorisation
      solver->refactorize(a3);
      CHECK((solver->solve(a3 * x) - x).norm() < 1e-10 * x.norm());
      CHECK(solver->size() == n);
      // and back to the first pattern
      solver->refactorize(a1);
      CHECK((solver->solve(a1 * x) - x).norm() < 1e-10 * x.norm());
    }
  }
}

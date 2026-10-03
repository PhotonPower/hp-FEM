// Direct solver backends: every available backend reproduces a known solution of a random
// sparse complex system, reuses its factorisation for several right-hand sides and
// reports errors; the automatic choice prefers MUMPS when compiled in.
#include <algorithm>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "hpfem/core/error.hpp"
#include "hpfem/solvers/linear_solver.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::Matrix;
using hpfem::Real;
using hpfem::SparseMatrix;
using hpfem::Vector;
using hpfem::solvers::available;
using hpfem::solvers::available_backends;
using hpfem::solvers::backend_name;
using hpfem::solvers::cudss_status;
using hpfem::solvers::DirectSolverBackend;
using hpfem::solvers::make_direct_solver;
using hpfem::solvers::solve_direct;

namespace {

/// Random sparse complex matrix with a dominant diagonal (non-singular, non-symmetric).
SparseMatrix random_system(Index n, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<Real> dist(-1.0, 1.0);
  std::uniform_int_distribution<Index> column(0, n - 1);
  std::vector<Eigen::Triplet<Complex, Index>> triplets;
  for (Index i = 0; i < n; ++i) {
    triplets.emplace_back(i, i, Complex{10.0 + dist(gen), dist(gen)});
    for (int k = 0; k < 4; ++k) {
      triplets.emplace_back(i, column(gen), Complex{dist(gen), dist(gen)});
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

TEST_CASE("direct solver backends solve random sparse complex systems", "[solvers]") {
  const Index n = 400;
  const SparseMatrix a = random_system(n, 1);
  const Vector x_exact = random_vector(n, 2);
  const Vector b = a * x_exact;
  for (const DirectSolverBackend backend : available_backends()) {
    INFO(backend_name(backend));
    auto solver = make_direct_solver(backend);
    CHECK(!solver->name().empty());
    solver->factorize(a);
    CHECK(solver->size() == n);
    const Vector x = solver->solve(b);
    CHECK((x - x_exact).norm() < 1e-10 * x_exact.norm());
    // reuse of the factorisation for another right-hand side
    const Vector y_exact = random_vector(n, 3);
    const Vector y = solver->solve(a * y_exact);
    CHECK((y - y_exact).norm() < 1e-10 * y_exact.norm());
    // several right-hand sides at once equal the column-wise solves
    Matrix rhs(n, 3);
    rhs.col(0) = b;
    rhs.col(1) = a * y_exact;
    rhs.col(2) = a * random_vector(n, 4);
    const Matrix xs = solver->solve_many(rhs);
    REQUIRE(xs.rows() == n);
    REQUIRE(xs.cols() == 3);
    for (Index j = 0; j < 3; ++j) {
      const Vector column = solver->solve(Vector(rhs.col(j)));
      CHECK((xs.col(j) - column).norm() < 1e-12 * column.norm());
    }
    CHECK(solver->solve_many(Matrix(n, 0)).cols() == 0);
    CHECK_THROWS_AS(solver->solve_many(Matrix::Ones(n + 1, 2)), hpfem::InvalidArgument);
    // the one-shot interface
    CHECK((solve_direct(a, b, backend) - x_exact).norm() < 1e-10 * x_exact.norm());
    // errors: wrong size, solve before factorize
    CHECK_THROWS_AS(solver->solve(Vector::Ones(n + 1)), hpfem::InvalidArgument);
    CHECK_THROWS_AS(make_direct_solver(backend)->solve(b), hpfem::Error);
  }
}

TEST_CASE("direct solver backends: availability and automatic choice", "[solvers]") {
  CHECK(available(DirectSolverBackend::kAuto));
  CHECK(available(DirectSolverBackend::kSparseLu));
  const auto backends = available_backends();
  CHECK(std::find(backends.begin(), backends.end(), DirectSolverBackend::kSparseLu) !=
        backends.end());
#ifdef HPFEM_HAVE_MUMPS
  CHECK(available(DirectSolverBackend::kMumps));
  CHECK(make_direct_solver(DirectSolverBackend::kAuto)->name().find("MUMPS") != std::string::npos);
#else
  CHECK(!available(DirectSolverBackend::kMumps));
  CHECK_THROWS_AS(make_direct_solver(DirectSolverBackend::kMumps), hpfem::Error);
  CHECK(make_direct_solver(DirectSolverBackend::kAuto)->name().find("SparseLU") !=
        std::string::npos);
#endif
#ifdef HPFEM_HAVE_CUDA
  // compiled in: usable only if the hpfem_gpu library loads and a device exists
  INFO(cudss_status());
  CHECK(!cudss_status().empty());
  if (available(DirectSolverBackend::kCudss)) {
    CHECK(make_direct_solver(DirectSolverBackend::kCudss)->name().find("cuDSS") !=
          std::string::npos);
  } else {
    CHECK_THROWS_AS(make_direct_solver(DirectSolverBackend::kCudss), hpfem::Error);
  }
#else
  CHECK(!available(DirectSolverBackend::kCudss));
  CHECK_THROWS_AS(make_direct_solver(DirectSolverBackend::kCudss), hpfem::Error);
#endif
  // a singular matrix is reported
  SparseMatrix singular(3, 3);
  std::vector<Eigen::Triplet<Complex, Index>> t{{0, 0, Complex{1.0, 0.0}},
                                                {1, 1, Complex{1.0, 0.0}}};
  singular.setFromTriplets(t.begin(), t.end());
  for (const DirectSolverBackend backend : available_backends()) {
    INFO(backend_name(backend));
    CHECK_THROWS_AS(make_direct_solver(backend)->factorize(singular), hpfem::Error);
  }
  SparseMatrix rectangular(2, 3);
  CHECK_THROWS_AS(make_direct_solver()->factorize(rectangular), hpfem::InvalidArgument);
}

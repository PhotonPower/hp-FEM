#include <random>

#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/sparse_assembler.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/solvers/linear_solver.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::Real;
using hpfem::SparseMatrix;
using hpfem::Vector;
using hpfem::assembly::SparseAssembler;
using hpfem::solvers::make_sparse_lu;
using hpfem::solvers::solve_direct;

namespace {

/// Random sparse complex matrix with a dominant diagonal (well conditioned).
SparseMatrix random_system(Index n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<Real> u(-1.0, 1.0);
  std::uniform_int_distribution<Index> col(0, n - 1);
  SparseAssembler assembler(n, n);
  for (Index i = 0; i < n; ++i) {
    assembler.add(i, i, Complex{10.0 + u(rng), u(rng)});
    for (int k = 0; k < 4; ++k) assembler.add(i, col(rng), Complex{u(rng), u(rng)});
  }
  return assembler.finalize();
}

}  // namespace

TEST_CASE("SparseLU solves a random complex system to machine precision", "[solvers]") {
  const Index n = 200;
  const SparseMatrix a = random_system(n, 3);
  Vector x_exact(n);
  std::mt19937 rng(4);
  std::uniform_real_distribution<Real> u(-1.0, 1.0);
  for (Index i = 0; i < n; ++i) x_exact(i) = Complex{u(rng), u(rng)};
  const Vector b = a * x_exact;

  auto solver = make_sparse_lu();
  REQUIRE(solver->name().find("SparseLU") != std::string::npos);
  REQUIRE_THROWS_AS(solver->solve(b), hpfem::Error);  // not factorised yet
  solver->factorize(a);
  REQUIRE(solver->size() == n);
  const Vector x = solver->solve(b);
  REQUIRE((x - x_exact).norm() < 1e-12 * x_exact.norm());
  // second right-hand side reuses the factorisation
  const Vector x2 = solver->solve(2.0 * b);
  REQUIRE((x2 - 2.0 * x_exact).norm() < 1e-12 * x_exact.norm());
  REQUIRE_THROWS_AS(solver->solve(Vector::Zero(n + 1)), hpfem::InvalidArgument);

  const Vector y = solve_direct(a, b);
  REQUIRE((y - x_exact).norm() < 1e-12 * x_exact.norm());
}

TEST_CASE("SparseLU reports singular and non-square matrices", "[solvers]") {
  SparseAssembler singular(3, 3);
  singular.add(0, 0, Complex{1.0, 0.0});
  singular.add(1, 1, Complex{1.0, 0.0});  // row 2 empty
  REQUIRE_THROWS_AS(solve_direct(singular.finalize(), Vector::Ones(3)), hpfem::Error);
  SparseAssembler rect(2, 3);
  rect.add(0, 0, Complex{1.0, 0.0});
  REQUIRE_THROWS_AS(make_sparse_lu()->factorize(rect.finalize()), hpfem::InvalidArgument);
}

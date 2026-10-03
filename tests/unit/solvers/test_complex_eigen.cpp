#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#if defined(__GNUC__) && !defined(__clang__)
// GCC reports a potential null dereference inside Eigen's dense storage (false positive,
// as in complex_eigen_solver.cpp)
#pragma GCC diagnostic ignored "-Wnull-dereference"
#endif

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "hpfem/core/error.hpp"
#include "hpfem/solvers/eigen_solver.hpp"

using Catch::Approx;
using hpfem::Complex;
using hpfem::Index;
using hpfem::Matrix;
using hpfem::Real;
using hpfem::SparseMatrix;
using hpfem::Vector;
using hpfem::solvers::complex_eigenpairs_near;
using hpfem::solvers::EigenOptions;

namespace {

/// Complex tridiagonal pencil (A, B) with B diagonal: a lossy 1D Helmholtz-like operator.
std::pair<SparseMatrix, SparseMatrix> pencil(Index n, Real scale) {
  std::vector<Eigen::Triplet<Complex, Index>> a;
  std::vector<Eigen::Triplet<Complex, Index>> b;
  for (Index i = 0; i < n; ++i) {
    const Real x = static_cast<Real>(i) / static_cast<Real>(n);
    a.emplace_back(i, i, Complex{2.0 + 0.5 * x, 0.1 * x} * scale);
    if (i + 1 < n) {
      a.emplace_back(i, i + 1, Complex{-1.0, 0.05} * scale);
      a.emplace_back(i + 1, i, Complex{-1.0, 0.05} * scale);
    }
    b.emplace_back(i, i, Complex{1.0 + 0.3 * x, 0.02 * x});
  }
  SparseMatrix am(n, n);
  SparseMatrix bm(n, n);
  am.setFromTriplets(a.begin(), a.end());
  bm.setFromTriplets(b.begin(), b.end());
  return {am, bm};
}

/// All eigenvalues by the dense solver on B^{-1} A, sorted by distance to sigma.
std::vector<Complex> dense_near(const SparseMatrix& a, const SparseMatrix& b, Complex sigma) {
  const Matrix dense = Matrix(b).inverse() * Matrix(a);
  Eigen::ComplexEigenSolver<Matrix> solver(dense);
  std::vector<Complex> values(solver.eigenvalues().data(),
                              solver.eigenvalues().data() + solver.eigenvalues().size());
  std::sort(values.begin(), values.end(),
            [&](Complex p, Complex q) { return std::abs(p - sigma) < std::abs(q - sigma); });
  return values;
}

}  // namespace

TEST_CASE("complex_eigenpairs_near matches a dense reference and is scale invariant",
          "[solvers][eigen][complex]") {
  const Index n = 80;
  const Complex sigma{1.7, 0.05};
  for (const Real scale : {1.0, 1e13}) {
    const auto [a, b] = pencil(n, scale);
    const Complex shift = sigma * scale;
    const auto reference = dense_near(a, b, shift);
    EigenOptions options;
    options.num_eigenvalues = 4;
    options.krylov_dimension = 24;
    options.tolerance = 1e-10;
    const auto result = complex_eigenpairs_near(a, b, shift, options);
    REQUIRE(result.num_converged == 4);
    REQUIRE(result.eigenvalues.size() == 4);
    for (Index i = 0; i < 4; ++i) {
      const Complex lambda = result.eigenvalues(i);
      REQUIRE(std::abs(lambda - reference[static_cast<std::size_t>(i)]) <
              1e-8 * std::abs(reference[static_cast<std::size_t>(i)]));
      // residual of the pencil and unit vectors
      const Vector x = result.eigenvectors.col(i);
      REQUIRE(x.norm() == Approx(1.0));
      const Vector r = a * x - lambda * (b * x);
      REQUIRE(r.norm() < 1e-8 * (a * x).norm());
      REQUIRE(result.residuals(i) < 1e-10);
    }
    // ordered by distance to the shift
    for (Index i = 1; i < 4; ++i) {
      REQUIRE(std::abs(result.eigenvalues(i) - shift) >=
              std::abs(result.eigenvalues(i - 1) - shift));
    }
  }
}

TEST_CASE("complex_eigenpairs_near rejects bad input", "[solvers][eigen][complex]") {
  const auto [a, b] = pencil(10, 1.0);
  SparseMatrix wrong(10, 9);
  REQUIRE_THROWS_AS(complex_eigenpairs_near(a, wrong, Complex{1.0, 0.0}), hpfem::InvalidArgument);
  REQUIRE_THROWS_AS(complex_eigenpairs_near(a, b, Complex{1.0, 0.0}, EigenOptions{0}),
                    hpfem::InvalidArgument);
  // a shift on an eigenvalue makes the factorisation singular or the iteration meaningless
  const auto reference = dense_near(a, b, Complex{1.5, 0.0});
  EigenOptions tiny;
  tiny.num_eigenvalues = 1;
  tiny.max_iterations = 1;
  tiny.tolerance = 1e-300;  // cannot be met: non-convergence is reported
  REQUIRE_THROWS_AS(complex_eigenpairs_near(a, b, Complex{1.5, 0.0}, tiny), hpfem::Error);
  (void)reference;
}

TEST_CASE("complex_eigenpairs_near_gauged skips the kernel spanned by the gradient",
          "[solvers][eigen][complex][gauge]") {
  // A = C^H C with a four-dimensional kernel range(G), B Hermitian positive definite: the
  // pencil has four eigenvalues at zero; the gauged solver must return the smallest
  // non-zero ones (dense reference) although the shift is closest to zero
  const Index n = 24;
  const Index kernel = 4;
  Matrix random(n, n);
  for (Index i = 0; i < n; ++i) {
    for (Index j = 0; j < n; ++j) {
      random(i, j) = Complex(std::sin(1.3 * static_cast<Real>(i * n + j)),
                             std::cos(0.7 * static_cast<Real>(i + 2 * j)));
    }
  }
  const Matrix q = Eigen::HouseholderQR<Matrix>(random).householderQ();
  const Matrix g_dense = q.leftCols(kernel) * (Matrix::Identity(kernel, kernel) +
                                               0.3 * random.topLeftCorner(kernel, kernel));
  Vector weights(n - kernel);
  for (Index i = 0; i < n - kernel; ++i) weights(i) = 1.0 + 0.5 * static_cast<Real>(i);
  const Matrix a_dense = q.rightCols(n - kernel) * weights.asDiagonal() *
                         q.rightCols(n - kernel).adjoint();
  Matrix b_dense = Matrix::Identity(n, n);
  for (Index i = 0; i < n; ++i) b_dense(i, i) = 1.0 + 0.1 * static_cast<Real>(i % 3);
  const SparseMatrix a = a_dense.sparseView();
  const SparseMatrix b = b_dense.sparseView();
  const SparseMatrix g = g_dense.sparseView();
  const Complex sigma{-0.2, 0.0};
  std::vector<Complex> reference = dense_near(a, b, sigma);
  std::erase_if(reference, [](Complex v) { return std::abs(v) < 1e-8; });
  REQUIRE(reference.size() == static_cast<std::size_t>(n - kernel));
  EigenOptions options;
  options.num_eigenvalues = 5;
  options.krylov_dimension = 16;
  const auto result = hpfem::solvers::complex_eigenpairs_near_gauged(a, b, g, sigma, options);
  REQUIRE(result.num_converged == 5);
  for (Index i = 0; i < 5; ++i) {
    REQUIRE(std::abs(result.eigenvalues(i)) > 0.1);
    REQUIRE(std::abs(result.eigenvalues(i) - reference[static_cast<std::size_t>(i)]) < 1e-8);
    // the eigenvector is B-orthogonal to the kernel and satisfies the pencil
    const Vector x = result.eigenvectors.col(i);
    REQUIRE((g_dense.adjoint() * (b_dense * x)).norm() < 1e-8);
    REQUIRE((a_dense * x - result.eigenvalues(i) * (b_dense * x)).norm() < 1e-8);
  }
  // the ungauged solver finds the kernel first
  const auto plain = complex_eigenpairs_near(a, b, sigma, options);
  REQUIRE(std::abs(plain.eigenvalues(0)) < 1e-8);
  REQUIRE_THROWS_AS(hpfem::solvers::complex_eigenpairs_near_gauged(a, b, SparseMatrix(n - 1, 2),
                                                                   sigma, options),
                    hpfem::InvalidArgument);
}

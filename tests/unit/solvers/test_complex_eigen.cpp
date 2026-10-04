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
#include "hpfem/solvers/device_arnoldi.hpp"
#include "hpfem/solvers/eigen_solver.hpp"
#include "hpfem/solvers/linear_solver.hpp"

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
  const Matrix a_dense =
      q.rightCols(n - kernel) * weights.asDiagonal() * q.rightCols(n - kernel).adjoint();
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
  REQUIRE_THROWS_AS(
      hpfem::solvers::complex_eigenpairs_near_gauged(a, b, SparseMatrix(n - 1, 2), sigma, options),
      hpfem::InvalidArgument);
}

TEST_CASE("DeviceArnoldi: the Krylov basis on the GPU reproduces the host recursion",
          "[solvers][eigen][complex][gpu]") {
  using hpfem::solvers::available;
  using hpfem::solvers::DeviceArnoldi;
  using hpfem::solvers::DirectSolverBackend;
  using hpfem::solvers::make_direct_solver;
  using hpfem::solvers::Symmetry;
  const Index n = 600;
  const Index ncv = 14;
  const auto [a, b] = pencil(n, 1.0);
  const Complex sigma{0.9, 0.05};
  SparseMatrix shifted = a - sigma * b;
  shifted.makeCompressed();
  if (!available(DirectSolverBackend::kCudss)) {
    auto lu = make_direct_solver(DirectSolverBackend::kSparseLu);
    lu->factorize(shifted);
    CHECK(!DeviceArnoldi::available(*lu));
    CHECK_THROWS_AS(DeviceArnoldi(*lu, b, nullptr, nullptr, ncv), hpfem::Error);
    return;
  }
  auto solver = make_direct_solver(DirectSolverBackend::kCudss, Symmetry::kDetect);
  solver->factorize(shifted);
  REQUIRE(DeviceArnoldi::available(*solver));
  CHECK(DeviceArnoldi::basis_bytes(n, ncv) == 16 * static_cast<std::size_t>(n * (ncv + 4)));
  // host recursion with the same factorisation: classical Gram–Schmidt twice
  Vector start(n);
  for (Index i = 0; i < n; ++i) {
    start(i) = Complex(std::sin(0.37 * static_cast<Real>(i)), std::cos(1.1 * static_cast<Real>(i)));
  }
  Matrix v = Matrix::Zero(n, ncv + 1);
  v.col(0) = start / start.norm();
  Matrix h_host = Matrix::Zero(ncv + 1, ncv);
  Matrix h_device = Matrix::Zero(ncv + 1, ncv);
  DeviceArnoldi device(*solver, b, nullptr, nullptr, ncv);
  device.set_start(start);
  for (Index j = 0; j < ncv; ++j) {
    Vector w = solver->solve(Vector(b * v.col(j)));
    for (int pass = 0; pass < 2; ++pass) {
      const Vector c = v.leftCols(j + 1).adjoint() * w;
      h_host.col(j).head(j + 1) += c;
      w -= v.leftCols(j + 1) * c;
    }
    h_host(j + 1, j) = w.norm();
    v.col(j + 1) = w / w.norm();
    Vector column;
    h_device(j + 1, j) = device.iterate(j, column);
    REQUIRE(column.size() == j + 1);
    h_device.col(j).head(j + 1) = column;
  }
  REQUIRE((h_device - h_host).norm() < 1e-12 * h_host.norm());
  // the Ritz values of both Hessenberg matrices
  Eigen::ComplexEigenSolver<Matrix> host(h_host.topLeftCorner(ncv, ncv));
  Eigen::ComplexEigenSolver<Matrix> dev(h_device.topLeftCorner(ncv, ncv));
  std::vector<Complex> ritz_host(host.eigenvalues().data(), host.eigenvalues().data() + ncv);
  std::vector<Complex> ritz_device(dev.eigenvalues().data(), dev.eigenvalues().data() + ncv);
  const auto by_modulus = [](Complex p, Complex q) { return std::abs(p) > std::abs(q); };
  std::sort(ritz_host.begin(), ritz_host.end(), by_modulus);
  std::sort(ritz_device.begin(), ritz_device.end(), by_modulus);
  for (Index k = 0; k < ncv; ++k) {
    REQUIRE(std::abs(ritz_device[hpfem::as_size(k)] - ritz_host[hpfem::as_size(k)]) <
            1e-10 * std::abs(ritz_host[hpfem::as_size(k)]));
  }
  // the basis itself, Ritz vectors as V_m y and the restart vector
  const Matrix basis = device.combine(ncv + 1, Matrix::Identity(ncv + 1, ncv + 1));
  REQUIRE((basis - v).norm() < 1e-12 * v.norm());
  const Matrix y = host.eigenvectors().leftCols(3);
  REQUIRE((device.combine(ncv, y) - v.leftCols(ncv) * y).norm() < 1e-12 * y.norm());
  Vector c(ncv);
  for (Index i = 0; i < ncv; ++i) c(i) = Complex(1.0 / static_cast<Real>(i + 1), 0.1);
  device.restart(ncv, c);
  const Vector expected = (v.leftCols(ncv) * c).normalized();
  REQUIRE((device.combine(1, Matrix::Identity(1, 1)).col(0) - expected).norm() < 1e-12);
  // errors: wrong sizes
  CHECK_THROWS_AS(device.set_start(Vector::Ones(n + 1)), hpfem::InvalidArgument);
  CHECK_THROWS_AS(device.iterate(ncv, c), hpfem::InvalidArgument);
  CHECK_THROWS_AS(device.restart(ncv + 2, c), hpfem::InvalidArgument);
  CHECK_THROWS_AS(device.combine(ncv, Matrix::Identity(3, 3)), hpfem::InvalidArgument);
  CHECK_THROWS_AS(DeviceArnoldi(*solver, a.topLeftCorner(n - 1, n - 1), nullptr, nullptr, ncv),
                  hpfem::InvalidArgument);
  // a gauge needs its factorisation on the device
  auto lu = make_direct_solver(DirectSolverBackend::kSparseLu);
  CHECK(!DeviceArnoldi::available(*lu));
  const SparseMatrix g = Matrix::Ones(n, 1).sparseView();
  CHECK_THROWS_AS(DeviceArnoldi(*solver, b, &g, lu.get(), ncv), hpfem::Error);
}

TEST_CASE("complex_eigenpairs_near on the GPU gives the eigenvalues of the host path",
          "[solvers][eigen][complex][gpu]") {
  using hpfem::solvers::available;
  using hpfem::solvers::DirectSolverBackend;
  if (!available(DirectSolverBackend::kCudss)) return;
  // the shift-invert solver with the Krylov basis on the device (cuDSS) against the host
  // basis (SparseLU): same eigenvalues to 1e-10, with and without the gauge projection
  const Index n = 800;
  const auto [a, b] = pencil(n, 1.0);
  const Complex sigma{1.2, 0.1};
  EigenOptions options;
  options.num_eigenvalues = 6;
  options.krylov_dimension = 24;
  const auto host = complex_eigenpairs_near(a, b, sigma, options, DirectSolverBackend::kSparseLu);
  const auto device = complex_eigenpairs_near(a, b, sigma, options, DirectSolverBackend::kCudss);
  REQUIRE(host.num_converged == 6);
  REQUIRE(device.num_converged == 6);
  for (Index i = 0; i < 6; ++i) {
    REQUIRE(std::abs(device.eigenvalues(i) - host.eigenvalues(i)) <
            1e-10 * std::abs(host.eigenvalues(i)));
    const Vector x = device.eigenvectors.col(i);
    REQUIRE((a * x - device.eigenvalues(i) * (b * x)).norm() < 1e-8);
  }
  // gauged: G spans eight directions; every eigenvector is B-orthogonal to them
  std::vector<Eigen::Triplet<Complex, Index>> entries;
  for (Index i = 0; i < n; ++i) {
    entries.emplace_back(i, i % 8, Complex{1.0, 0.05 * static_cast<Real>(i % 5)});
  }
  SparseMatrix g(n, 8);
  g.setFromTriplets(entries.begin(), entries.end());
  const auto host_gauged = hpfem::solvers::complex_eigenpairs_near_gauged(
      a, b, g, sigma, options, DirectSolverBackend::kSparseLu);
  const auto device_gauged = hpfem::solvers::complex_eigenpairs_near_gauged(
      a, b, g, sigma, options, DirectSolverBackend::kCudss);
  REQUIRE(host_gauged.num_converged == 6);
  REQUIRE(device_gauged.num_converged == 6);
  for (Index i = 0; i < 6; ++i) {
    REQUIRE(std::abs(device_gauged.eigenvalues(i) - host_gauged.eigenvalues(i)) <
            1e-10 * std::abs(host_gauged.eigenvalues(i)));
    const Vector x = device_gauged.eigenvectors.col(i);
    REQUIRE((g.adjoint() * (b * x)).norm() < 1e-8);
  }
}

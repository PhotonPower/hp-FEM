#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#if defined(__GNUC__) && !defined(__clang__)
// GCC 13 reports a potential null dereference inside std::complex / Eigen's dense storage
// (false positive, as in condensation.cpp)
#pragma GCC diagnostic ignored "-Wnull-dereference"
#endif

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseLU>
#include <fmt/format.h>

#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/solvers/eigen_solver.hpp"
#include "hpfem/solvers/linear_solver.hpp"

namespace hpfem::solvers {

namespace {

/// Largest modulus of the entries.
[[nodiscard]] Real max_abs(const SparseMatrix& matrix) {
  Real out = 0;
  for (Index row = 0; row < matrix.rows(); ++row) {
    for (SparseMatrix::InnerIterator it(matrix, row); it; ++it) {
      out = std::max(out, std::abs(it.value()));
    }
  }
  return out;
}

using Operator = std::function<Vector(const Vector&)>;

/// Shift-invert Arnoldi on `apply` = (A' − σ'B')⁻¹B' (every vector passed through
/// `project`) with explicit restarts; returns the result in the original scale.
ComplexEigenResult arnoldi(const Operator& apply, const Operator& project, Index n,
                           Complex sigma_scaled, Real lambda_scale, Complex sigma,
                           const EigenOptions& options, const char* function,
                           const std::string& backend_name) {
  const Index nev = std::min(options.num_eigenvalues, n - 1);
  Index ncv = options.krylov_dimension > 0 ? options.krylov_dimension : 2 * nev + 10;
  ncv = std::min(std::max(ncv, nev + 2), n);
  if (nev < 1 || ncv <= nev) {
    throw InvalidArgument(fmt::format("{}: system too small ({} DoFs) for {} eigenvalues",
                                      function, n, options.num_eigenvalues));
  }
  std::mt19937 generator(42);
  std::normal_distribution<Real> normal;
  Vector start(n);
  for (Index i = 0; i < n; ++i) start(i) = Complex(normal(generator), normal(generator));
  start = project(start);
  start /= start.norm();

  Matrix v = Matrix::Zero(n, ncv + 1);
  Matrix h = Matrix::Zero(ncv + 1, ncv);
  Vector ritz_values;
  Matrix ritz_vectors;
  std::vector<Real> residuals;
  int iterations = 0;
  Index converged = 0;
  for (iterations = 1; iterations <= options.max_iterations; ++iterations) {
    v.col(0) = start / start.norm();
    h.setZero();
    Index m = ncv;
    for (Index j = 0; j < ncv; ++j) {
      Vector w = project(apply(v.col(j)));
      for (int pass = 0; pass < 2; ++pass) {
        for (Index i = 0; i <= j; ++i) {
          const Complex coefficient = v.col(i).adjoint() * w;
          h(i, j) += coefficient;
          w -= coefficient * v.col(i);
        }
      }
      const Real beta = w.norm();
      h(j + 1, j) = beta;
      if (beta < 1e-14 * std::max(h.col(j).norm(), Real{1.0})) {
        m = j + 1;  // invariant subspace found
        break;
      }
      v.col(j + 1) = w / beta;
    }
    const Matrix hm = h.topLeftCorner(m, m);
    Eigen::ComplexEigenSolver<Matrix> eigen(hm);
    if (eigen.info() != Eigen::Success) {
      throw Error(fmt::format("{}: Hessenberg eigensolver failed", function));
    }
    const Vector& theta = eigen.eigenvalues();
    const Matrix& y = eigen.eigenvectors();
    std::vector<Index> order(as_size(m));
    std::iota(order.begin(), order.end(), Index{0});
    std::sort(order.begin(), order.end(),
              [&](Index p, Index q) { return std::abs(theta(p)) > std::abs(theta(q)); });
    const Index wanted = std::min(nev, m);
    ritz_values.resize(wanted);
    ritz_vectors.resize(n, wanted);
    residuals.assign(as_size(wanted), 0.0);
    const Real tail = std::abs(h(m, m - 1));  // 0 on an invariant subspace
    converged = 0;
    for (Index i = 0; i < wanted; ++i) {
      const Index k = order[as_size(i)];
      ritz_values(i) = theta(k);
      ritz_vectors.col(i) = v.leftCols(m) * y.col(k);
      residuals[as_size(i)] = tail * std::abs(y(m - 1, k)) / std::max(std::abs(theta(k)), 1e-300);
      if (residuals[as_size(i)] < options.tolerance) ++converged;
    }
    if (converged == wanted || m < ncv) break;
    start.setZero();
    for (Index i = 0; i < wanted; ++i) {
      start += (1.0 + residuals[as_size(i)] / options.tolerance) * ritz_vectors.col(i);
    }
  }
  if (converged < 1) {
    throw Error(fmt::format(
        "{}: Arnoldi did not converge ({} of {} eigenvalues after {} restarts); increase "
        "krylov_dimension or max_iterations",
        function, converged, nev, iterations));
  }
  ComplexEigenResult result;
  const Index count = ritz_values.size();
  result.eigenvalues.resize(count);
  result.eigenvectors.resize(n, count);
  result.residuals.resize(count);
  std::vector<Index> by_distance(as_size(count));
  std::iota(by_distance.begin(), by_distance.end(), Index{0});
  std::vector<Complex> lambda(as_size(count));
  for (Index i = 0; i < count; ++i) {
    lambda[as_size(i)] = (sigma_scaled + 1.0 / ritz_values(i)) * lambda_scale;
  }
  std::sort(by_distance.begin(), by_distance.end(), [&](Index p, Index q) {
    return std::abs(lambda[as_size(p)] - sigma) < std::abs(lambda[as_size(q)] - sigma);
  });
  for (Index i = 0; i < count; ++i) {
    const Index k = by_distance[as_size(i)];
    result.eigenvalues(i) = lambda[as_size(k)];
    result.eigenvectors.col(i) = ritz_vectors.col(k) / ritz_vectors.col(k).norm();
    result.residuals(i) = residuals[as_size(k)];
  }
  result.iterations = std::min(iterations, options.max_iterations);
  result.num_converged = converged;
  log().info(
      "{}: {} of {} eigenvalues near {:.6g}{:+.6g}i converged after {} Arnoldi restarts ({})",
      function, converged, count, sigma.real(), sigma.imag(), result.iterations, backend_name);
  return result;
}

/// Scaled pencil and the factorised shift.
struct ShiftInvert {
  SparseMatrix b;
  std::unique_ptr<LinearSolver> solver;
  Real lambda_scale = 1;
  Complex sigma_scaled;
};

ShiftInvert prepare(const SparseMatrix& a_full, const SparseMatrix& b_full, Complex sigma,
                    const EigenOptions& options, DirectSolverBackend backend,
                    const char* function) {
  if (a_full.rows() != a_full.cols() || b_full.rows() != a_full.rows() ||
      b_full.cols() != a_full.cols()) {
    throw InvalidArgument(fmt::format("{}: A and B must be square and of equal size", function));
  }
  if (options.num_eigenvalues < 1) {
    throw InvalidArgument(fmt::format("{}: num_eigenvalues must be at least 1", function));
  }
  const Real scale_a = max_abs(a_full);
  const Real scale_b = max_abs(b_full);
  if (scale_a <= 0 || scale_b <= 0) {
    throw InvalidArgument(fmt::format("{}: a matrix is zero", function));
  }
  ShiftInvert op;
  op.lambda_scale = scale_a / scale_b;
  op.sigma_scaled = sigma / op.lambda_scale;
  SparseMatrix shifted = a_full / scale_a - op.sigma_scaled * (b_full / scale_b);
  shifted.makeCompressed();
  op.b = b_full / scale_b;
  op.solver = make_direct_solver(backend);
  try {
    op.solver->factorize(shifted);
  } catch (const Error& error) {
    throw Error(fmt::format(
        "{}: factorisation of A - sigma B failed ({}); the shift hits the spectrum or the "
        "pencil is singular",
        function, error.what()));
  }
  return op;
}

}  // namespace

ComplexEigenResult complex_eigenpairs_near(const SparseMatrix& a_full, const SparseMatrix& b_full,
                                           Complex sigma, const EigenOptions& options,
                                           DirectSolverBackend backend) {
  const char* function = "complex_eigenpairs_near";
  ShiftInvert op = prepare(a_full, b_full, sigma, options, backend, function);
  const Operator apply = [&op](const Vector& x) { return op.solver->solve(Vector(op.b * x)); };
  const Operator identity = [](const Vector& x) { return x; };
  return arnoldi(apply, identity, a_full.rows(), op.sigma_scaled, op.lambda_scale, sigma,
                 options, function, op.solver->name());
}

ComplexEigenResult complex_eigenpairs_near_gauged(const SparseMatrix& a_full,
                                                  const SparseMatrix& b_full,
                                                  const SparseMatrix& gradient, Complex sigma,
                                                  const EigenOptions& options,
                                                  DirectSolverBackend backend) {
  const char* function = "complex_eigenpairs_near_gauged";
  if (gradient.rows() != a_full.rows()) {
    throw InvalidArgument(
        fmt::format("{}: the gradient must have {} rows", function, a_full.rows()));
  }
  ShiftInvert op = prepare(a_full, b_full, sigma, options, backend, function);
  // gauge projector P = I - G (G^H B G)^{-1} G^H B on the scaled mass matrix
  const SparseMatrix bg = op.b * gradient;
  const SparseMatrix gt = gradient.adjoint();
  using ColMajor = Eigen::SparseMatrix<Complex, Eigen::ColMajor, Index>;
  ColMajor k = ColMajor(gt * bg);
  k.makeCompressed();
  Eigen::SparseLU<ColMajor> gauge;
  gauge.compute(k);
  if (gauge.info() != Eigen::Success) {
    throw Error(fmt::format("{}: the gauge matrix G^H B G is singular", function));
  }
  const Operator project = [&](const Vector& w) -> Vector {
    const Vector r = gt * (op.b * w);
    return w - gradient * gauge.solve(r);
  };
  const Operator apply = [&op](const Vector& x) { return op.solver->solve(Vector(op.b * x)); };
  return arnoldi(apply, project, a_full.rows(), op.sigma_scaled, op.lambda_scale, sigma, options,
                 function, op.solver->name());
}

}  // namespace hpfem::solvers

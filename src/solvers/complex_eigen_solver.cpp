#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
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

}  // namespace

ComplexEigenResult complex_eigenpairs_near(const SparseMatrix& a_full, const SparseMatrix& b_full,
                                           Complex sigma, const EigenOptions& options,
                                           DirectSolverBackend backend) {
  if (a_full.rows() != a_full.cols() || b_full.rows() != a_full.rows() ||
      b_full.cols() != a_full.cols()) {
    throw InvalidArgument("complex_eigenpairs_near: A and B must be square and of equal size");
  }
  if (options.num_eigenvalues < 1) {
    throw InvalidArgument("complex_eigenpairs_near: num_eigenvalues must be at least 1");
  }
  const Index n = a_full.rows();
  const Index nev = std::min(options.num_eigenvalues, n - 1);
  Index ncv = options.krylov_dimension > 0 ? options.krylov_dimension : 2 * nev + 10;
  ncv = std::min(std::max(ncv, nev + 2), n);
  if (nev < 1 || ncv <= nev) {
    throw InvalidArgument(
        fmt::format("complex_eigenpairs_near: system too small ({} DoFs) for {} eigenvalues", n,
                    options.num_eigenvalues));
  }
  // scale both matrices to O(1) (absolute thresholds below, SI meshes)
  const Real scale_a = max_abs(a_full);
  const Real scale_b = max_abs(b_full);
  if (scale_a <= 0 || scale_b <= 0)
    throw InvalidArgument("complex_eigenpairs_near: a matrix is zero");
  const Real lambda_scale = scale_a / scale_b;  // lambda = lambda' * lambda_scale
  const Complex sigma_scaled = sigma / lambda_scale;
  SparseMatrix shifted = a_full / scale_a - sigma_scaled * (b_full / scale_b);
  shifted.makeCompressed();
  const SparseMatrix b = b_full / scale_b;
  std::unique_ptr<LinearSolver> solver = make_direct_solver(backend);
  try {
    solver->factorize(shifted);
  } catch (const Error& error) {
    throw Error(fmt::format(
        "complex_eigenpairs_near: factorisation of A - sigma B failed ({}); the shift hits the "
        "spectrum or the pencil is singular",
        error.what()));
  }
  // y = (A - sigma B)^{-1} B x: eigenvalues nu = 1 / (lambda' - sigma'), largest |nu| wanted
  const auto apply = [&](const Vector& x) { return solver->solve(Vector(b * x)); };

  std::mt19937 generator(42);
  std::normal_distribution<Real> normal;
  Vector start(n);
  for (Index i = 0; i < n; ++i) start(i) = Complex(normal(generator), normal(generator));
  start /= start.norm();

  Matrix v = Matrix::Zero(n, ncv + 1);
  Matrix h = Matrix::Zero(ncv + 1, ncv);
  Vector ritz_values;
  Matrix ritz_vectors;
  std::vector<Real> residuals;
  int iterations = 0;
  Index converged = 0;
  for (iterations = 1; iterations <= options.max_iterations; ++iterations) {
    // Arnoldi with modified Gram–Schmidt (twice) from `start`
    v.col(0) = start / start.norm();
    h.setZero();
    Index m = ncv;
    for (Index j = 0; j < ncv; ++j) {
      Vector w = apply(v.col(j));
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
    if (eigen.info() != Eigen::Success)
      throw Error("complex_eigenpairs_near: Hessenberg eigensolver failed");
    const Vector theta = eigen.eigenvalues();
    const Matrix y = eigen.eigenvectors();
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
      // residual of the shift-inverted problem relative to the Ritz value
      residuals[as_size(i)] = tail * std::abs(y(m - 1, k)) / std::max(std::abs(theta(k)), 1e-300);
      if (residuals[as_size(i)] < options.tolerance) ++converged;
    }
    if (converged == wanted || m < ncv) break;
    // explicit restart from the wanted Ritz vectors, weighted towards the unconverged ones
    start.setZero();
    for (Index i = 0; i < wanted; ++i) {
      start += (1.0 + residuals[as_size(i)] / options.tolerance) * ritz_vectors.col(i);
    }
  }
  if (converged < 1) {
    throw Error(fmt::format(
        "complex_eigenpairs_near: Arnoldi did not converge ({} of {} eigenvalues after {} "
        "restarts); increase krylov_dimension or max_iterations",
        converged, nev, iterations));
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
      "complex_eigenpairs_near: {} of {} eigenvalues near {:.6g}{:+.6g}i converged after {} "
      "Arnoldi restarts ({})",
      converged, count, sigma.real(), sigma.imag(), result.iterations, solver->name());
  return result;
}

}  // namespace hpfem::solvers

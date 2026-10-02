#include "hpfem/solvers/eigen_solver.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

// GCC 13 reports a false -Wmaybe-uninitialized inside Eigen::SparseLU when inlined here
// (see linear_solver.cpp); GCC 13 -O2 also reports a null-dereference false positive in
// Eigen::Ref<const SparseMatrix>::nonZeros() when Spectra wraps the reduced mass matrix.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#pragma GCC diagnostic ignored "-Wnull-dereference"
#endif
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/SymGEigsShiftSolver.h>

#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <fmt/format.h>

#include "hpfem/assembly/discrete_gradient.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"

namespace hpfem::solvers {

namespace {

using RealSparse = Eigen::SparseMatrix<Real, Eigen::ColMajor, int>;

/// Real part of a complex sparse matrix; the imaginary part must be negligible.
RealSparse real_part(const SparseMatrix& matrix, const char* name,
                     const char* function = "gauged_curl_curl_eigenpairs") {
  RealSparse out(static_cast<int>(matrix.rows()), static_cast<int>(matrix.cols()));
  std::vector<Eigen::Triplet<Real, int>> triplets;
  triplets.reserve(as_size(matrix.nonZeros()));
  Real max_imag = 0;
  Real max_abs = 0;
  for (Index row = 0; row < matrix.rows(); ++row) {
    for (SparseMatrix::InnerIterator it(matrix, row); it; ++it) {
      max_imag = std::max(max_imag, std::abs(it.value().imag()));
      max_abs = std::max(max_abs, std::abs(it.value()));
      triplets.emplace_back(static_cast<int>(row), static_cast<int>(it.col()), it.value().real());
    }
  }
  if (max_imag > 1e-12 * std::max(max_abs, Real{1.0})) {
    throw InvalidArgument(fmt::format(
        "{}: {} has imaginary parts up to {}; the solver needs real matrices (lossless media "
        "without PML)",
        function, name, max_imag));
  }
  out.setFromTriplets(triplets.begin(), triplets.end());
  out.makeCompressed();
  return out;
}

/// Spectra operator y = P (S − σM)^{-1} x with the M-orthogonal projector P onto the
/// complement of the gradient space, P = I − G K^{-1} G^T M, K = G^T M G.
class ProjectedShiftInvert {
 public:
  using Scalar = Real;

  ProjectedShiftInvert(const RealSparse& s, const RealSparse& m, const RealSparse& g)
      : s_(s), m_(m), g_(g) {
    const RealSparse mg = m * g;
    const RealSparse gt = g.transpose();
    const RealSparse k = gt * mg;
    gauge_.compute(k);
    if (gauge_.info() != Eigen::Success) {
      throw Error("gauged_curl_curl_eigenpairs: the gauge matrix G^T M G is singular");
    }
  }
  [[nodiscard]] Eigen::Index rows() const { return s_.rows(); }
  [[nodiscard]] Eigen::Index cols() const { return s_.cols(); }
  void set_shift(const Scalar& sigma) {
    const RealSparse shifted = s_ - sigma * m_;
    shift_invert_.compute(shifted);
    if (shift_invert_.info() != Eigen::Success) {
      throw Error(fmt::format(
          "gauged_curl_curl_eigenpairs: factorisation of S - {} M failed; choose a shift away "
          "from the spectrum",
          sigma));
    }
  }
  void perform_op(const Scalar* x_in, Scalar* y_out) const {
    const Eigen::Map<const RealVector> x(x_in, s_.rows());
    Eigen::Map<RealVector> y(y_out, s_.rows());
    const RealVector w = shift_invert_.solve(x);
    y = project(w);
  }
  /// P w = w − G K^{-1} G^T M w.
  [[nodiscard]] RealVector project(const RealVector& w) const {
    const RealVector r = g_.transpose() * (m_ * w);
    return w - g_ * gauge_.solve(r);
  }

 private:
  const RealSparse& s_;
  const RealSparse& m_;
  const RealSparse& g_;
  Eigen::SparseLU<RealSparse> gauge_;
  Eigen::SparseLU<RealSparse> shift_invert_;
};

/// Spectra operator y = (A − σB)^{-1} B x.
class PencilShiftInvert {
 public:
  using Scalar = Real;

  PencilShiftInvert(const RealSparse& a, const RealSparse& b, Real sigma) : b_(b) {
    const RealSparse shifted = a - sigma * b;
    lu_.compute(shifted);
    if (lu_.info() != Eigen::Success) {
      throw Error(fmt::format(
          "generalized_eigenpairs_near: factorisation of A - {} B failed; the shift hits the "
          "spectrum or the pencil is singular",
          sigma));
    }
  }
  [[nodiscard]] Eigen::Index rows() const { return b_.rows(); }
  [[nodiscard]] Eigen::Index cols() const { return b_.cols(); }
  void perform_op(const Scalar* x_in, Scalar* y_out) const {
    const Eigen::Map<const RealVector> x(x_in, b_.rows());
    Eigen::Map<RealVector> y(y_out, b_.rows());
    y = lu_.solve(RealVector(b_ * x));
  }

 private:
  const RealSparse& b_;
  Eigen::SparseLU<RealSparse> lu_;
};

}  // namespace

EigenResult generalized_eigenpairs_near(const SparseMatrix& a_full, const SparseMatrix& b_full,
                                        Real sigma, const EigenOptions& options) {
  if (a_full.rows() != a_full.cols() || b_full.rows() != a_full.rows() ||
      b_full.cols() != a_full.cols()) {
    throw InvalidArgument("generalized_eigenpairs_near: A and B must be square and of equal size");
  }
  if (options.num_eigenvalues < 1) {
    throw InvalidArgument("generalized_eigenpairs_near: num_eigenvalues must be at least 1");
  }
  const RealSparse a = real_part(a_full, "A", "generalized_eigenpairs_near");
  const RealSparse b = real_part(b_full, "B", "generalized_eigenpairs_near");
  const Index n = a.rows();
  const Index nev = std::min(options.num_eigenvalues, n - 2);
  Index ncv = options.krylov_dimension > 0 ? options.krylov_dimension : 2 * nev + 10;
  ncv = std::min(std::max(ncv, nev + 2), n);
  if (nev < 1 || ncv <= nev + 1) {
    throw InvalidArgument(
        fmt::format("generalized_eigenpairs_near: system too small ({} DoFs) for {} eigenvalues", n,
                    options.num_eigenvalues));
  }
  PencilShiftInvert op(a, b, sigma);
  Spectra::GenEigsSolver<PencilShiftInvert> solver(op, static_cast<int>(nev),
                                                   static_cast<int>(ncv));
  solver.init();
  const int converged = static_cast<int>(
      solver.compute(Spectra::SortRule::LargestMagn, options.max_iterations, options.tolerance));
  if (solver.info() != Spectra::CompInfo::Successful || converged < 1) {
    throw Error(fmt::format(
        "generalized_eigenpairs_near: Arnoldi did not converge ({} of {} eigenvalues after {} "
        "iterations); increase krylov_dimension or max_iterations",
        converged, nev, solver.num_iterations()));
  }
  const Eigen::VectorXcd nu = solver.eigenvalues();
  const Eigen::MatrixXcd vectors = solver.eigenvectors();
  // lambda = sigma + 1 / nu, sorted ascending; vectors rotated to be real
  std::vector<Index> order(as_size(converged));
  std::iota(order.begin(), order.end(), Index{0});
  std::vector<Real> lambda(as_size(converged));
  for (Index i = 0; i < converged; ++i) {
    const Complex l = sigma + 1.0 / nu(i);
    if (std::abs(l.imag()) > 1e-8 * std::max(std::abs(l), Real{1.0})) {
      throw Error(fmt::format(
          "generalized_eigenpairs_near: eigenvalue {} has a non-negligible imaginary part {}",
          l.real(), l.imag()));
    }
    lambda[as_size(i)] = l.real();
  }
  std::sort(order.begin(), order.end(),
            [&](Index x, Index y) { return lambda[as_size(x)] < lambda[as_size(y)]; });
  EigenResult result;
  result.eigenvalues.resize(converged);
  result.eigenvectors = RealMatrix::Zero(n, converged);
  for (Index k = 0; k < converged; ++k) {
    const Index i = order[as_size(k)];
    result.eigenvalues(k) = lambda[as_size(i)];
    Eigen::VectorXcd v = vectors.col(i);
    Index largest = 0;
    v.cwiseAbs().maxCoeff(&largest);
    v *= std::conj(v(largest)) / std::abs(v(largest));  // real at the largest entry
    const RealVector re = v.real();
    result.eigenvectors.col(k) = re / re.norm();
  }
  result.iterations = static_cast<int>(solver.num_iterations());
  log().info(
      "generalized_eigenpairs_near: {} eigenvalues in [{:.6g}, {:.6g}] near {:.6g} after {} "
      "iterations",
      converged, result.eigenvalues.minCoeff(), result.eigenvalues.maxCoeff(), sigma,
      solver.num_iterations());
  return result;
}

EigenResult gauged_curl_curl_eigenpairs(const SparseMatrix& stiffness, const SparseMatrix& mass,
                                        const SparseMatrix& gradient,
                                        std::span<const Index> free_nedelec,
                                        std::span<const Index> free_h1,
                                        const EigenOptions& options) {
  if (options.num_eigenvalues < 1) {
    throw InvalidArgument("gauged_curl_curl_eigenpairs: num_eigenvalues must be at least 1");
  }
  const RealSparse s = real_part(assembly::extract(stiffness, free_nedelec, free_nedelec), "S");
  const RealSparse m = real_part(assembly::extract(mass, free_nedelec, free_nedelec), "M");
  const RealSparse g = real_part(assembly::extract(gradient, free_nedelec, free_h1), "G");
  const Index n = s.rows();
  const Index nev = std::min(options.num_eigenvalues, n - 1);
  Index ncv = options.krylov_dimension > 0 ? options.krylov_dimension : 2 * nev + 10;
  ncv = std::min(std::max(ncv, nev + 1), n);
  if (nev < 1 || ncv <= nev) {
    throw InvalidArgument(fmt::format(
        "gauged_curl_curl_eigenpairs: system too small ({} free DoFs) for {} eigenvalues", n,
        options.num_eigenvalues));
  }

  ProjectedShiftInvert op(s, m, g);
  Spectra::SparseSymMatProd<Real, Eigen::Lower, Eigen::ColMajor, int> bop(m);
  Spectra::SymGEigsShiftSolver<ProjectedShiftInvert, decltype(bop), Spectra::GEigsMode::ShiftInvert>
      solver(op, bop, static_cast<int>(nev), static_cast<int>(ncv), options.shift);
  // start in the gauged subspace: a projected random vector
  RealVector start = RealVector::Random(n);
  start = op.project(start);
  solver.init(start.data());
  const int converged =
      static_cast<int>(solver.compute(Spectra::SortRule::LargestMagn, options.max_iterations,
                                      options.tolerance, Spectra::SortRule::SmallestAlge));
  if (solver.info() != Spectra::CompInfo::Successful || converged < 1) {
    throw Error(fmt::format(
        "gauged_curl_curl_eigenpairs: Lanczos did not converge ({} of {} eigenvalues after {} "
        "iterations); increase krylov_dimension or max_iterations",
        converged, nev, solver.num_iterations()));
  }
  const RealVector values = solver.eigenvalues();
  const RealMatrix vectors = solver.eigenvectors();
  log().info("gauged_curl_curl_eigenpairs: {} eigenvalues in [{:.6g}, {:.6g}] after {} iterations",
             converged, values.minCoeff(), values.maxCoeff(), solver.num_iterations());

  // scatter to full size (constrained DoFs stay zero)
  EigenResult result;
  result.eigenvalues = values;
  result.eigenvectors = RealMatrix::Zero(stiffness.rows(), converged);
  for (Index i = 0; i < n; ++i) result.eigenvectors.row(free_nedelec[as_size(i)]) = vectors.row(i);
  result.iterations = static_cast<int>(solver.num_iterations());
  return result;
}

}  // namespace hpfem::solvers

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#if defined(__GNUC__) && !defined(__clang__)
// GCC 13 reports a potential null dereference inside std::complex / Eigen's dense storage
// (false positive, as in condensation.cpp)
#pragma GCC diagnostic ignored "-Wnull-dereference"
#endif

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <fmt/format.h>

#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/solvers/device_arnoldi.hpp"
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

/// Number of wanted eigenvalues and Krylov vectors from the options.
struct KrylovSizes {
  Index nev = 0;
  Index ncv = 0;
};

[[nodiscard]] KrylovSizes krylov_sizes(const EigenOptions& options, Index n, const char* function) {
  KrylovSizes sizes;
  sizes.nev = std::min(options.num_eigenvalues, n - 1);
  sizes.ncv = options.krylov_dimension > 0 ? options.krylov_dimension : 2 * sizes.nev + 10;
  sizes.ncv = std::min(std::max(sizes.ncv, sizes.nev + 2), n);
  if (sizes.nev < 1 || sizes.ncv <= sizes.nev) {
    throw InvalidArgument(fmt::format("{}: system too small ({} DoFs) for {} eigenvalues", function,
                                      n, options.num_eigenvalues));
  }
  return sizes;
}

/// The Arnoldi basis @f$ V @f$ and the operations of one shift-invert iteration on it. The
/// host version keeps @f$ V @f$ in memory and applies the operator through callbacks, the
/// device version delegates everything to `DeviceArnoldi`; both run the same algorithm
/// (classical Gram–Schmidt applied twice), so their Ritz values agree to round-off.
class KrylovBasis {
 public:
  virtual ~KrylovBasis() = default;
  /// @f$ v_0 = P s / \|P s\| @f$.
  virtual void set_start(const Vector& start) = 0;
  /// @f$ w = P K^{-1} B v_j @f$, @f$ h_{i,j} \mathrel{+}= \langle v_i, w \rangle @f$ for
  /// @f$ i \le j @f$ (two Gram–Schmidt passes), stores @f$ v_{j+1} = w / \|w\| @f$ and
  /// returns @f$ \|w\| @f$.
  virtual Real iterate(Index j, Matrix& h) = 0;
  /// @f$ v_0 = V_m c / \|V_m c\| @f$.
  virtual void restart(Index m, const Vector& coefficients) = 0;
  /// @f$ V_m C @f$.
  [[nodiscard]] virtual Matrix combine(Index m, const Matrix& coefficients) const = 0;
  [[nodiscard]] virtual const char* description() const noexcept = 0;
};

class HostBasis final : public KrylovBasis {
 public:
  HostBasis(Operator apply, Operator project, Index n, Index ncv)
      : apply_(std::move(apply)), project_(std::move(project)), v_(Matrix::Zero(n, ncv + 1)) {}

  void set_start(const Vector& start) override {
    const Vector projected = project_(start);
    v_.col(0) = projected / projected.norm();
  }
  Real iterate(Index j, Matrix& h) override {
    Vector w = project_(apply_(v_.col(j)));
    const auto basis = v_.leftCols(j + 1);
    for (int pass = 0; pass < 2; ++pass) {
      const Vector coefficients = basis.adjoint() * w;
      h.col(j).head(j + 1) += coefficients;
      w -= basis * coefficients;
    }
    const Real beta = w.norm();
    if (beta > 0) v_.col(j + 1) = w / beta;
    return beta;
  }
  void restart(Index m, const Vector& coefficients) override {
    const Vector start = v_.leftCols(m) * coefficients;
    v_.col(0) = start / start.norm();
  }
  [[nodiscard]] Matrix combine(Index m, const Matrix& coefficients) const override {
    return v_.leftCols(m) * coefficients;
  }
  [[nodiscard]] const char* description() const noexcept override {
    return "Krylov basis on the host";
  }

 private:
  Operator apply_;
  Operator project_;
  Matrix v_;
};

class DeviceBasis final : public KrylovBasis {
 public:
  explicit DeviceBasis(std::unique_ptr<DeviceArnoldi> device) : device_(std::move(device)) {}

  void set_start(const Vector& start) override { device_->set_start(start); }
  Real iterate(Index j, Matrix& h) override {
    Vector column;
    const Real beta = device_->iterate(j, column);
    h.col(j).head(j + 1) += column;
    return beta;
  }
  void restart(Index m, const Vector& coefficients) override { device_->restart(m, coefficients); }
  [[nodiscard]] Matrix combine(Index m, const Matrix& coefficients) const override {
    return device_->combine(m, coefficients);
  }
  [[nodiscard]] const char* description() const noexcept override {
    return "Krylov basis on the device";
  }

 private:
  std::unique_ptr<DeviceArnoldi> device_;
};

/// Shift-invert Arnoldi on `basis` (the operator (A' − σ'B')⁻¹B' with every vector
/// projected) with explicit restarts; returns the result in the original scale.
ComplexEigenResult arnoldi(KrylovBasis& basis, Index n, KrylovSizes sizes, Complex sigma_scaled,
                           Real lambda_scale, Complex sigma, const EigenOptions& options,
                           const char* function, const std::string& backend_name) {
  const Index nev = sizes.nev;
  const Index ncv = sizes.ncv;
  std::mt19937 generator(42);
  std::normal_distribution<Real> normal;
  Vector start(n);
  for (Index i = 0; i < n; ++i) start(i) = Complex(normal(generator), normal(generator));
  basis.set_start(start);

  Matrix h = Matrix::Zero(ncv + 1, ncv);
  Vector ritz_values;
  Matrix ritz_coefficients;  // the wanted eigenvectors y of H_m: Ritz vectors are V_m y
  std::vector<Real> residuals;
  int iterations = 0;
  Index converged = 0;
  Index m_used = 0;
  for (iterations = 1; iterations <= options.max_iterations; ++iterations) {
    h.setZero();
    Index m = ncv;
    for (Index j = 0; j < ncv; ++j) {
      const Real beta = basis.iterate(j, h);
      h(j + 1, j) = beta;
      if (beta < 1e-14 * std::max(h.col(j).norm(), Real{1.0})) {
        m = j + 1;  // invariant subspace found
        break;
      }
    }
    m_used = m;
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
    ritz_coefficients.resize(m, wanted);
    residuals.assign(as_size(wanted), 0.0);
    const Real tail = std::abs(h(m, m - 1));  // 0 on an invariant subspace
    converged = 0;
    for (Index i = 0; i < wanted; ++i) {
      const Index k = order[as_size(i)];
      ritz_values(i) = theta(k);
      ritz_coefficients.col(i) = y.col(k);
      residuals[as_size(i)] = tail * std::abs(y(m - 1, k)) / std::max(std::abs(theta(k)), 1e-300);
      if (residuals[as_size(i)] < options.tolerance) ++converged;
    }
    if (converged == wanted || m < ncv) break;
    Vector combination = Vector::Zero(m);
    for (Index i = 0; i < wanted; ++i) {
      combination += (1.0 + residuals[as_size(i)] / options.tolerance) * ritz_coefficients.col(i);
    }
    basis.restart(m, combination);
  }
  if (converged < 1) {
    throw Error(fmt::format(
        "{}: Arnoldi did not converge ({} of {} eigenvalues after {} restarts); increase "
        "krylov_dimension or max_iterations",
        function, converged, nev, iterations));
  }
  const Matrix ritz_vectors = basis.combine(m_used, ritz_coefficients);
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
      "{}: {} of {} eigenvalues near {:.6g}{:+.6g}i converged after {} Arnoldi restarts ({}, {})",
      function, converged, count, sigma.real(), sigma.imag(), result.iterations, backend_name,
      basis.description());
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
  op.b.makeCompressed();
  // complex-symmetric pencils (curl–curl without Bloch phases) take the LDL^T paths
  op.solver = make_direct_solver(backend, Symmetry::kDetect);
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

/// The basis on the device when the shift (and the gauge) are factorised by the cuDSS
/// backend and the device memory suffices, otherwise on the host.
std::unique_ptr<KrylovBasis> make_basis(ShiftInvert& op, const SparseMatrix* gradient,
                                        LinearSolver* gauge, Index ncv, const Operator& apply,
                                        const Operator& project, const char* function) {
  const Index n = op.b.rows();
  if (DeviceArnoldi::available(*op.solver) &&
      (gauge == nullptr || DeviceArnoldi::available(*gauge))) {
    try {
      return std::make_unique<DeviceBasis>(
          std::make_unique<DeviceArnoldi>(*op.solver, op.b, gradient, gauge, ncv));
    } catch (const Error& error) {
      log().warn("{}: Krylov basis stays on the host: {}", function, error.what());
    }
  }
  return std::make_unique<HostBasis>(apply, project, n, ncv);
}

}  // namespace

ComplexEigenResult complex_eigenpairs_near(const SparseMatrix& a_full, const SparseMatrix& b_full,
                                           Complex sigma, const EigenOptions& options,
                                           DirectSolverBackend backend) {
  const char* function = "complex_eigenpairs_near";
  ShiftInvert op = prepare(a_full, b_full, sigma, options, backend, function);
  const KrylovSizes sizes = krylov_sizes(options, a_full.rows(), function);
  const Operator apply = [&op](const Vector& x) { return op.solver->solve(Vector(op.b * x)); };
  const Operator identity = [](const Vector& x) { return x; };
  const std::unique_ptr<KrylovBasis> basis =
      make_basis(op, nullptr, nullptr, sizes.ncv, apply, identity, function);
  return arnoldi(*basis, a_full.rows(), sizes, op.sigma_scaled, op.lambda_scale, sigma, options,
                 function, op.solver->name());
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
  const KrylovSizes sizes = krylov_sizes(options, a_full.rows(), function);
  // gauge projector P = I - G (G^H B G)^{-1} G^H B on the scaled mass matrix; the small
  // Hermitian gauge matrix is factorised with the same direct solver as the shift. When the
  // Krylov basis goes to the device, the gauge solve must happen there as well, so the gauge
  // is factorised by cuDSS even if it is small enough for the CPU under kAuto.
  const SparseMatrix bg = op.b * gradient;
  const SparseMatrix gt = gradient.adjoint();
  SparseMatrix k = gt * bg;
  k.makeCompressed();
  std::unique_ptr<LinearSolver> gauge;
  if (DeviceArnoldi::available(*op.solver)) {
    try {
      gauge = make_direct_solver(DirectSolverBackend::kCudss, Symmetry::kDetect);
      gauge->factorize(k);
    } catch (const Error& error) {
      log().debug("{}: gauge matrix not factorised on the device ({}); basis stays on the host",
                  function, error.what());
      gauge.reset();
    }
  }
  if (!gauge) {
    gauge = make_direct_solver(backend, Symmetry::kDetect);
    try {
      gauge->factorize(k);
    } catch (const Error& error) {
      throw Error(
          fmt::format("{}: the gauge matrix G^H B G is singular ({})", function, error.what()));
    }
  }
  const Operator project = [&](const Vector& w) -> Vector {
    const Vector r = gt * (op.b * w);
    return w - gradient * gauge->solve(r);
  };
  const Operator apply = [&op](const Vector& x) { return op.solver->solve(Vector(op.b * x)); };
  const std::unique_ptr<KrylovBasis> basis =
      make_basis(op, &gradient, gauge.get(), sizes.ncv, apply, project, function);
  return arnoldi(*basis, a_full.rows(), sizes, op.sigma_scaled, op.lambda_scale, sigma, options,
                 function, op.solver->name());
}

}  // namespace hpfem::solvers

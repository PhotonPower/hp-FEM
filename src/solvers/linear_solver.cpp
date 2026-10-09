#include "hpfem/solvers/linear_solver.hpp"

// GCC 13 reports a false -Wmaybe-uninitialized inside Eigen::SparseLU::analyzePattern when
// it is inlined into this translation unit at -O2 (system headers do not silence inlined
// code); clang and -O0 builds are clean.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include <Eigen/SparseLU>
#include <fmt/format.h>

#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/solvers/device_arnoldi.hpp"
#include "hpfem/solvers/device_matrix.hpp"
#include "hpfem/solvers/device_stepper.hpp"

namespace hpfem::solvers {

Matrix LinearSolver::solve_many(const Matrix& rhs) const {
  Matrix x(rhs.rows(), rhs.cols());
  for (Index j = 0; j < rhs.cols(); ++j) x.col(j) = solve(Vector(rhs.col(j)));
  return x;
}

Vector LinearSolver::solve_transposed(const Vector& /*rhs*/) const {
  throw Error(fmt::format("{}: no solve with the transposed matrix", name()));
}

Matrix LinearSolver::solve_transposed_many(const Matrix& rhs) const {
  Matrix x(rhs.rows(), rhs.cols());
  for (Index j = 0; j < rhs.cols(); ++j) x.col(j) = solve_transposed(Vector(rhs.col(j)));
  return x;
}

namespace {

class SparseLuSolver final : public LinearSolver {
 public:
  void factorize(const SparseMatrix& matrix) override {
    if (matrix.rows() != matrix.cols()) {
      throw InvalidArgument(
          fmt::format("SparseLU: matrix is {} x {}, not square", matrix.rows(), matrix.cols()));
    }
    const ColMajor column_major(matrix);
    lu_.compute(column_major);
    if (lu_.info() != Eigen::Success) {
      ready_ = false;
      throw Error(
          fmt::format("SparseLU: factorisation of the {} x {} system failed ({}); the "
                      "matrix is singular or badly scaled",
                      matrix.rows(), matrix.cols(), lu_.lastErrorMessage()));
    }
    ready_ = true;
    size_ = matrix.rows();
    remember_pattern(column_major);
    log().info("SparseLU: factorised {} unknowns, {} nonzeros in L + U", size_,
               lu_.nnzL() + lu_.nnzU());
  }

  [[nodiscard]] Index factor_entries() const noexcept override {
    return ready_ ? static_cast<Index>(lu_.nnzL()) + static_cast<Index>(lu_.nnzU()) : Index{-1};
  }

  void refactorize(const SparseMatrix& matrix) override {
    if (!ready_ || matrix.rows() != size_ || matrix.cols() != size_) {
      factorize(matrix);
      return;
    }
    const ColMajor column_major(matrix);
    if (!same_pattern(column_major)) {
      factorize(matrix);
      return;
    }
    lu_.factorize(column_major);  // numerical phase with the analysed pattern
    if (lu_.info() != Eigen::Success) {
      ready_ = false;
      throw Error(fmt::format("SparseLU: refactorisation of the {} x {} system failed ({})", size_,
                              size_, lu_.lastErrorMessage()));
    }
    log().debug("SparseLU: refactorised {} unknowns with the previous pattern", size_);
  }

  [[nodiscard]] Vector solve(const Vector& rhs) const override {
    if (!ready_) throw Error("SparseLU: solve() called before a successful factorize()");
    if (rhs.size() != size_) {
      throw InvalidArgument(fmt::format("SparseLU: right-hand side has {} entries, system has {}",
                                        rhs.size(), size_));
    }
    Vector x = lu_.solve(rhs);
    if (lu_.info() != Eigen::Success) throw Error("SparseLU: triangular solve failed");
    return x;
  }

  [[nodiscard]] Matrix solve_many(const Matrix& rhs) const override {
    if (!ready_) throw Error("SparseLU: solve() called before a successful factorize()");
    if (rhs.rows() != size_) {
      throw InvalidArgument(
          fmt::format("SparseLU: right-hand sides have {} rows, system has {}", rhs.rows(), size_));
    }
    if (rhs.cols() == 0) {
      Matrix empty(size_, 0);  // SparseLU's solve indexes column 0 of the right-hand side
      return empty;
    }
    Matrix x = lu_.solve(rhs);
    if (lu_.info() != Eigen::Success) throw Error("SparseLU: triangular solve failed");
    return x;
  }

  [[nodiscard]] Vector solve_transposed(const Vector& rhs) const override {
    if (rhs.size() != size_) {
      throw InvalidArgument(fmt::format("SparseLU: right-hand side has {} entries, system has {}",
                                        rhs.size(), size_));
    }
    return solve_transposed_many(Matrix(rhs)).col(0);
  }

  /// Aᵀx = b with the factors of A: Uᵀ and Lᵀ solves in reverse order, the permutations
  /// swapped (Eigen's `SparseLU::transpose()` view).
  [[nodiscard]] Matrix solve_transposed_many(const Matrix& rhs) const override {
    if (!ready_) throw Error("SparseLU: solve() called before a successful factorize()");
    if (rhs.rows() != size_) {
      throw InvalidArgument(
          fmt::format("SparseLU: right-hand sides have {} rows, system has {}", rhs.rows(), size_));
    }
    if (rhs.cols() == 0) return Matrix(size_, 0);
    // the view is non-const in Eigen but only reads the factors
    auto& lu = const_cast<Eigen::SparseLU<ColMajor, Eigen::COLAMDOrdering<Index>>&>(lu_);
    Matrix x = lu.transpose().solve(rhs);
    if (lu_.info() != Eigen::Success) throw Error("SparseLU: transposed triangular solve failed");
    return x;
  }

  [[nodiscard]] Index size() const noexcept override { return size_; }
  [[nodiscard]] std::string name() const override { return "Eigen SparseLU (COLAMD)"; }

 private:
  void remember_pattern(const Eigen::SparseMatrix<Complex, Eigen::ColMajor, Index>& m) {
    outer_.assign(m.outerIndexPtr(), m.outerIndexPtr() + m.outerSize() + 1);
    inner_.assign(m.innerIndexPtr(), m.innerIndexPtr() + m.nonZeros());
  }
  [[nodiscard]] bool same_pattern(
      const Eigen::SparseMatrix<Complex, Eigen::ColMajor, Index>& m) const {
    return m.isCompressed() && static_cast<std::size_t>(m.outerSize() + 1) == outer_.size() &&
           static_cast<std::size_t>(m.nonZeros()) == inner_.size() &&
           std::equal(outer_.begin(), outer_.end(), m.outerIndexPtr()) &&
           std::equal(inner_.begin(), inner_.end(), m.innerIndexPtr());
  }
  std::vector<Index> outer_;
  std::vector<Index> inner_;

 public:
 private:
  using ColMajor = Eigen::SparseMatrix<Complex, Eigen::ColMajor, Index>;
  Eigen::SparseLU<ColMajor, Eigen::COLAMDOrdering<Index>> lu_;
  bool ready_ = false;
  Index size_ = 0;
};

}  // namespace

std::unique_ptr<LinearSolver> make_sparse_lu(Symmetry /*symmetry*/) {
  return std::make_unique<SparseLuSolver>();  // SparseLU has no symmetric variant
}

SparseMatrix upper_triangle(const SparseMatrix& matrix) {
  SparseMatrix upper(matrix.rows(), matrix.cols());
  std::vector<Eigen::Triplet<Complex, Index>> triplets;
  triplets.reserve(as_size(matrix.nonZeros() / 2 + matrix.rows()));
  for (Index row = 0; row < matrix.outerSize(); ++row) {
    for (SparseMatrix::InnerIterator it(matrix, row); it; ++it) {
      if (it.col() >= row) triplets.emplace_back(row, it.col(), it.value());
    }
  }
  upper.setFromTriplets(triplets.begin(), triplets.end());
  upper.makeCompressed();
  return upper;
}

Real asymmetry(const SparseMatrix& matrix) {
  if (matrix.rows() != matrix.cols()) {
    throw InvalidArgument(
        fmt::format("asymmetry: matrix is {} x {}, not square", matrix.rows(), matrix.cols()));
  }
  const SparseMatrix* csr = &matrix;
  SparseMatrix compressed;
  if (!matrix.isCompressed()) {
    compressed = matrix;
    compressed.makeCompressed();
    csr = &compressed;
  }
  const auto* outer = csr->outerIndexPtr();
  const auto* inner = csr->innerIndexPtr();
  const auto* values = csr->valuePtr();
  // a_ji for a row-major compressed matrix: binary search of column i in row j (sorted)
  const auto mirrored = [&](Index i, Index j) -> Complex {
    const auto* begin = inner + outer[j];
    const auto* end = inner + outer[j + 1];
    const auto* hit = std::lower_bound(begin, end, i);
    return (hit != end && *hit == i) ? values[hit - inner] : Complex{0.0, 0.0};
  };
  Real largest = 0;
  Real largest_difference = 0;
  for (Index row = 0; row < csr->rows(); ++row) {
    for (Index k = outer[row]; k < outer[row + 1]; ++k) {
      const Index col = inner[k];
      const Complex value = values[k];
      largest = std::max(largest, std::abs(value));
      if (col > row)
        largest_difference = std::max(largest_difference, std::abs(value - mirrored(row, col)));
      if (col < row && mirrored(row, col) == Complex{0.0, 0.0}) {
        largest_difference = std::max(largest_difference, std::abs(value));  // no upper twin
      }
    }
  }
  return largest > 0 ? largest_difference / largest : 0;
}

Symmetry detect_symmetry(const SparseMatrix& matrix, Real tolerance) {
  return asymmetry(matrix) <= tolerance ? Symmetry::kComplexSymmetric : Symmetry::kGeneral;
}

bool exploit_symmetry(Symmetry symmetry, const SparseMatrix& matrix, const char* backend) {
  switch (symmetry) {
    case Symmetry::kGeneral:
      return false;
    case Symmetry::kComplexSymmetric: {
#ifndef NDEBUG
      const Real skew = asymmetry(matrix);
      if (skew > 1e-10) {
        throw InvalidArgument(
            fmt::format("{}: kComplexSymmetric requested but the matrix is not symmetric "
                        "(relative asymmetry {:.2e})",
                        backend, skew));
      }
#endif
      return true;
    }
    case Symmetry::kDetect: {
      const Real skew = asymmetry(matrix);
      const bool symmetric = skew <= 1e-12;
      log().debug("{}: symmetry: {}, asymmetry = {:.2e}", backend,
                  symmetric ? "complex-symmetric (LDL^T)" : "general", skew);
      return symmetric;
    }
  }
  return false;
}

#ifndef HPFEM_HAVE_MUMPS
std::unique_ptr<LinearSolver> make_mumps(Symmetry /*symmetry*/) {
  throw Error("MUMPS backend requested but not compiled in (configure with HPFEM_ENABLE_MUMPS)");
}
#endif

#ifdef HPFEM_HAVE_CUDA
bool cudss_available() noexcept;  // src/solvers/cudss_solver.cpp
#else
std::unique_ptr<LinearSolver> make_cudss(Symmetry /*symmetry*/) {
  throw Error("cuDSS backend requested but not compiled in (configure with HPFEM_ENABLE_CUDA)");
}
std::string cudss_status() {
  return "not compiled in (configure with HPFEM_ENABLE_CUDA)";
}

struct DeviceMatrix::Impl {};
bool DeviceMatrix::available() noexcept {
  return false;
}
hpfem_gpu_matrix* DeviceMatrix::handle() const noexcept {
  return nullptr;
}
DeviceMatrix::DeviceMatrix(const SparseMatrix& /*matrix*/) {
  throw Error("DeviceMatrix needs the GPU backend (configure with HPFEM_ENABLE_CUDA)");
}
DeviceMatrix::~DeviceMatrix() = default;
DeviceMatrix::DeviceMatrix(DeviceMatrix&&) noexcept = default;
DeviceMatrix& DeviceMatrix::operator=(DeviceMatrix&&) noexcept = default;
Vector DeviceMatrix::apply(const Vector& /*x*/) const {
  throw Error("DeviceMatrix needs the GPU backend (configure with HPFEM_ENABLE_CUDA)");
}
Matrix DeviceMatrix::apply_many(const Matrix& /*x*/) const {
  throw Error("DeviceMatrix needs the GPU backend (configure with HPFEM_ENABLE_CUDA)");
}

struct DeviceArnoldi::Impl {};
bool DeviceArnoldi::available(const LinearSolver& /*shifted*/) noexcept {
  return false;
}
std::size_t DeviceArnoldi::basis_bytes(Index n, Index ncv) noexcept {
  return 16 * static_cast<std::size_t>(n) * static_cast<std::size_t>(ncv + 4);
}
DeviceArnoldi::DeviceArnoldi(LinearSolver& /*shifted*/, const SparseMatrix& /*b*/,
                             const SparseMatrix* /*gradient*/, LinearSolver* /*gauge*/,
                             Index /*ncv*/) {
  throw Error("DeviceArnoldi needs the GPU backend (configure with HPFEM_ENABLE_CUDA)");
}
DeviceArnoldi::~DeviceArnoldi() = default;
void DeviceArnoldi::set_start(const Vector& /*start*/) {
  throw Error("DeviceArnoldi needs the GPU backend (configure with HPFEM_ENABLE_CUDA)");
}
Real DeviceArnoldi::iterate(Index /*j*/, Vector& /*h_column*/) {
  throw Error("DeviceArnoldi needs the GPU backend (configure with HPFEM_ENABLE_CUDA)");
}
void DeviceArnoldi::restart(Index /*m*/, const Vector& /*coefficients*/) {
  throw Error("DeviceArnoldi needs the GPU backend (configure with HPFEM_ENABLE_CUDA)");
}
Matrix DeviceArnoldi::combine(Index /*m*/, const Matrix& /*coefficients*/) const {
  throw Error("DeviceArnoldi needs the GPU backend (configure with HPFEM_ENABLE_CUDA)");
}

struct DeviceStepper::Impl {};
bool DeviceStepper::available(const LinearSolver& /*newmark*/) noexcept {
  return false;
}
DeviceStepper::DeviceStepper(LinearSolver& /*newmark*/, const SparseMatrix* /*damping*/,
                             const SparseMatrix& /*stiffness*/, const Vector* /*load*/, Real /*dt*/,
                             Real /*beta*/, Real /*gamma*/) {
  throw Error("DeviceStepper needs the GPU backend (configure with HPFEM_ENABLE_CUDA)");
}
DeviceStepper::~DeviceStepper() = default;
void DeviceStepper::set_state(const Vector& /*u*/, const Vector& /*v*/, const Vector& /*a*/) {}
void DeviceStepper::step(Real /*load_scale*/) {}
void DeviceStepper::get_state(Vector& /*u*/, Vector& /*v*/, Vector& /*a*/) const {}
#endif

bool available(DirectSolverBackend backend) noexcept {
  if (backend == DirectSolverBackend::kAuto || backend == DirectSolverBackend::kSparseLu) {
    return true;
  }
#ifdef HPFEM_HAVE_MUMPS
  if (backend == DirectSolverBackend::kMumps) return true;
#endif
#ifdef HPFEM_HAVE_CUDA
  if (backend == DirectSolverBackend::kCudss) return cudss_available();
#endif
  return false;
}

std::vector<DirectSolverBackend> available_backends() {
  std::vector<DirectSolverBackend> out{DirectSolverBackend::kSparseLu};
  if (available(DirectSolverBackend::kMumps)) out.push_back(DirectSolverBackend::kMumps);
  if (available(DirectSolverBackend::kCudss)) out.push_back(DirectSolverBackend::kCudss);
  return out;
}

#ifndef HPFEM_GPU_MIN_UNKNOWNS_DEFAULT
#define HPFEM_GPU_MIN_UNKNOWNS_DEFAULT 10000
#endif

Index gpu_min_unknowns() {
  if (const char* env = std::getenv("HPFEM_GPU_MIN_UNKNOWNS"); env != nullptr && *env != '\0') {
    char* end = nullptr;
    const long long value = std::strtoll(env, &end, 10);
    if (end != env) return static_cast<Index>(value);
    log().warn("HPFEM_GPU_MIN_UNKNOWNS='{}' is not a number; using the default {}", env,
               HPFEM_GPU_MIN_UNKNOWNS_DEFAULT);
  }
  return HPFEM_GPU_MIN_UNKNOWNS_DEFAULT;
}

namespace {

/// `kAuto`: the backend is chosen in `factorize` from the size of the system, so the GPU
/// library is only loaded (and cuDSS only started) when it is actually going to be used.
class AutoSolver final : public LinearSolver {
 public:
  explicit AutoSolver(Symmetry symmetry) : symmetry_(symmetry) {}

  void factorize(const SparseMatrix& matrix) override {
    solver_.reset();
    const Index threshold = gpu_min_unknowns();
    const bool want_gpu = threshold >= 0 && matrix.rows() >= threshold;
    if (want_gpu && !gpu_refused_ && available(DirectSolverBackend::kCudss)) {
      // cuDSS pivots statically and refuses matrices whose pivots it would have to perturb
      // (hp systems with hanging nodes and high orders); such a system goes to the CPU, and
      // this object stays on the CPU for its later factorisations (sweeps, time steps)
      solver_ = make_cudss(symmetry_);
      try {
        solver_->factorize(matrix);
        return;
      } catch (const Error& error) {
        log().warn(
            "auto: cuDSS could not factorise the {} x {} system ({}); this solver uses {} from "
            "now on",
            matrix.rows(), matrix.cols(), error.what(),
            available(DirectSolverBackend::kMumps) ? "MUMPS" : "SparseLU");
        gpu_refused_ = true;
        solver_.reset();
      }
    }
    solver_ =
        available(DirectSolverBackend::kMumps) ? make_mumps(symmetry_) : make_sparse_lu(symmetry_);
    try {
      solver_->factorize(matrix);
    } catch (...) {
      solver_.reset();
      throw;
    }
  }
  void refactorize(const SparseMatrix& matrix) override {
    if (solver_) {
      solver_->refactorize(matrix);
    } else {
      factorize(matrix);
    }
  }
  [[nodiscard]] Vector solve(const Vector& rhs) const override {
    if (!solver_) throw Error("auto: solve() called before a successful factorize()");
    return solver_->solve(rhs);
  }
  [[nodiscard]] Matrix solve_many(const Matrix& rhs) const override {
    if (!solver_) throw Error("auto: solve_many() called before a successful factorize()");
    return solver_->solve_many(rhs);
  }
  [[nodiscard]] Vector solve_transposed(const Vector& rhs) const override {
    if (!solver_) throw Error("auto: solve_transposed() called before a successful factorize()");
    return solver_->solve_transposed(rhs);
  }
  [[nodiscard]] Matrix solve_transposed_many(const Matrix& rhs) const override {
    if (!solver_) {
      throw Error("auto: solve_transposed_many() called before a successful factorize()");
    }
    return solver_->solve_transposed_many(rhs);
  }
  [[nodiscard]] Index size() const noexcept override { return solver_ ? solver_->size() : 0; }
  [[nodiscard]] std::string name() const override {
    return solver_ ? fmt::format("auto: {}", solver_->name()) : "auto";
  }
  [[nodiscard]] std::string details() const override { return solver_ ? solver_->details() : ""; }
  [[nodiscard]] Index factor_entries() const noexcept override {
    return solver_ ? solver_->factor_entries() : Index{-1};
  }
  [[nodiscard]] const LinearSolver* backend() const noexcept override { return solver_.get(); }

 private:
  Symmetry symmetry_;
  std::unique_ptr<LinearSolver> solver_;
  bool gpu_refused_ = false;  ///< cuDSS refused a system of this object: stay on the CPU
};

}  // namespace

std::string backend_name(DirectSolverBackend backend) {
  switch (backend) {
    case DirectSolverBackend::kAuto:
      return "auto";
    case DirectSolverBackend::kSparseLu:
      return "SparseLU";
    case DirectSolverBackend::kMumps:
      return "MUMPS";
    case DirectSolverBackend::kCudss:
      return "cuDSS";
  }
  return "?";
}

std::unique_ptr<LinearSolver> make_direct_solver(DirectSolverBackend backend, Symmetry symmetry) {
  switch (backend) {
    case DirectSolverBackend::kAuto:
      return std::make_unique<AutoSolver>(symmetry);
    case DirectSolverBackend::kSparseLu:
      return make_sparse_lu(symmetry);
    case DirectSolverBackend::kMumps:
      return make_mumps(symmetry);
    case DirectSolverBackend::kCudss:
      return make_cudss(symmetry);
  }
  throw InvalidArgument("make_direct_solver: unknown backend");
}

Vector solve_direct(const SparseMatrix& matrix, const Vector& rhs, DirectSolverBackend backend,
                    Symmetry symmetry) {
  auto solver = make_direct_solver(backend, symmetry);
  solver->factorize(matrix);
  return solver->solve(rhs);
}

}  // namespace hpfem::solvers

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

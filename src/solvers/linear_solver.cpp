#include "hpfem/solvers/linear_solver.hpp"

// GCC 13 reports a false -Wmaybe-uninitialized inside Eigen::SparseLU::analyzePattern when
// it is inlined into this translation unit at -O2 (system headers do not silence inlined
// code); clang and -O0 builds are clean.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#include <Eigen/SparseLU>
#include <fmt/format.h>

#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"

namespace hpfem::solvers {

Matrix LinearSolver::solve_many(const Matrix& rhs) const {
  Matrix x(rhs.rows(), rhs.cols());
  for (Index j = 0; j < rhs.cols(); ++j) x.col(j) = solve(Vector(rhs.col(j)));
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
    log().info("SparseLU: factorised {} unknowns, {} nonzeros in L + U", size_,
               lu_.nnzL() + lu_.nnzU());
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

  [[nodiscard]] Index size() const noexcept override { return size_; }
  [[nodiscard]] std::string name() const override { return "Eigen SparseLU (COLAMD)"; }

 private:
  using ColMajor = Eigen::SparseMatrix<Complex, Eigen::ColMajor, Index>;
  Eigen::SparseLU<ColMajor, Eigen::COLAMDOrdering<Index>> lu_;
  bool ready_ = false;
  Index size_ = 0;
};

}  // namespace

std::unique_ptr<LinearSolver> make_sparse_lu() {
  return std::make_unique<SparseLuSolver>();
}

#ifndef HPFEM_HAVE_MUMPS
std::unique_ptr<LinearSolver> make_mumps() {
  throw Error("MUMPS backend requested but not compiled in (configure with HPFEM_ENABLE_MUMPS)");
}
#endif

#ifdef HPFEM_HAVE_CUDA
bool cudss_available() noexcept;  // src/solvers/cudss_solver.cpp
#else
std::unique_ptr<LinearSolver> make_cudss() {
  throw Error("cuDSS backend requested but not compiled in (configure with HPFEM_ENABLE_CUDA)");
}
std::string cudss_status() {
  return "not compiled in (configure with HPFEM_ENABLE_CUDA)";
}
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

std::unique_ptr<LinearSolver> make_direct_solver(DirectSolverBackend backend) {
  switch (backend) {
    case DirectSolverBackend::kAuto:
      return available(DirectSolverBackend::kMumps) ? make_mumps() : make_sparse_lu();
    case DirectSolverBackend::kSparseLu:
      return make_sparse_lu();
    case DirectSolverBackend::kMumps:
      return make_mumps();
    case DirectSolverBackend::kCudss:
      return make_cudss();
  }
  throw InvalidArgument("make_direct_solver: unknown backend");
}

Vector solve_direct(const SparseMatrix& matrix, const Vector& rhs, DirectSolverBackend backend) {
  auto solver = make_direct_solver(backend);
  solver->factorize(matrix);
  return solver->solve(rhs);
}

}  // namespace hpfem::solvers

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

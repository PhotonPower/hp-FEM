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

Vector solve_direct(const SparseMatrix& matrix, const Vector& rhs) {
  auto solver = make_sparse_lu();
  solver->factorize(matrix);
  return solver->solve(rhs);
}

}  // namespace hpfem::solvers

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

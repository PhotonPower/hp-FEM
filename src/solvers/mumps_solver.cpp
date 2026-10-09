// MUMPS backend (compiled only with HPFEM_ENABLE_MUMPS): zmumps_c with the sequential MPI
// stub of the library, assembled centralised input (ICNTL(5) = 0, ICNTL(18) = 0), analysis
// and factorisation in one call (job 4), solves in place (job 3).
#include <vector>
#include <zmumps_c.h>

#include <fmt/format.h>

#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/solvers/linear_solver.hpp"

// the MPI stub shipped with the sequential MUMPS build (libmpiseq); declared here so no
// mpi.h of a real MPI can be picked up by accident
extern "C" {
int MPI_Init(int* argc, char*** argv);
}

namespace hpfem::solvers {

namespace {

constexpr MUMPS_INT kUseCommWorld = -987654;

void ensure_mpi_initialised() {
  static bool initialised = false;
  if (!initialised) {
    int argc = 0;
    char** argv = nullptr;
    MPI_Init(&argc, &argv);
    initialised = true;
  }
}

class MumpsSolver final : public LinearSolver {
 public:
  explicit MumpsSolver(Symmetry symmetry) : symmetry_(symmetry) { ensure_mpi_initialised(); }

  ~MumpsSolver() override { finalize(); }

  void factorize(const SparseMatrix& matrix) override {
    if (matrix.rows() != matrix.cols()) {
      throw InvalidArgument(
          fmt::format("MUMPS: matrix is {} x {}, not square", matrix.rows(), matrix.cols()));
    }
    ready_ = false;
    size_ = matrix.rows();
    // SYM is fixed at initialisation, so the instance is (re)initialised when the symmetric
    // path changes; SYM = 2 takes one triangle only (entries given twice would be summed)
    const bool symmetric = exploit_symmetry(symmetry_, matrix, "MUMPS");
    if (!initialised_ || symmetric != symmetric_) {
      finalize();
      initialise(symmetric);
    }
    const SparseMatrix* input = &matrix;
    SparseMatrix upper;
    if (symmetric_) {
      upper = upper_triangle(matrix);
      input = &upper;
    }
    const Index nnz = input->nonZeros();
    rows_.resize(as_size(nnz));
    cols_.resize(as_size(nnz));
    values_.resize(as_size(nnz));
    std::size_t k = 0;
    for (Index row = 0; row < input->outerSize(); ++row) {
      for (SparseMatrix::InnerIterator it(*input, row); it; ++it) {
        rows_[k] = static_cast<MUMPS_INT>(it.row() + 1);
        cols_[k] = static_cast<MUMPS_INT>(it.col() + 1);
        values_[k].r = it.value().real();
        values_[k].i = it.value().imag();
        ++k;
      }
    }
    id_.n = static_cast<MUMPS_INT>(size_);
    id_.nnz = static_cast<MUMPS_INT8>(nnz);
    id_.irn = rows_.data();
    id_.jcn = cols_.data();
    id_.a = values_.data();
    id_.job = 4;  // analysis + factorisation
    zmumps_c(&id_);
    check("factorisation");
    ready_ = true;
    log().info("MUMPS: factorised {} unknowns, {} nonzeros, {} entries in the factors", size_, nnz,
               infog(29) > 0 ? infog(29) : infog(9));
  }

  /// Numerical phase only (job 2) when the entries sit on the analysed pattern.
  void refactorize(const SparseMatrix& matrix) override {
    if (!ready_ || matrix.rows() != size_ || matrix.cols() != size_ ||
        exploit_symmetry(symmetry_, matrix, "MUMPS") != symmetric_) {
      factorize(matrix);
      return;
    }
    const SparseMatrix* input = &matrix;
    SparseMatrix upper;
    if (symmetric_) {
      upper = upper_triangle(matrix);
      input = &upper;
    }
    if (static_cast<std::size_t>(input->nonZeros()) != rows_.size()) {
      factorize(matrix);
      return;
    }
    std::size_t k = 0;
    for (Index row = 0; row < input->outerSize(); ++row) {
      for (SparseMatrix::InnerIterator it(*input, row); it; ++it, ++k) {
        if (rows_[k] != static_cast<MUMPS_INT>(it.row() + 1) ||
            cols_[k] != static_cast<MUMPS_INT>(it.col() + 1)) {
          factorize(matrix);
          return;
        }
        values_[k].r = it.value().real();
        values_[k].i = it.value().imag();
      }
    }
    ready_ = false;
    id_.a = values_.data();
    id_.job = 2;  // numerical factorisation on the analysed pattern
    zmumps_c(&id_);
    check("refactorisation");
    ready_ = true;
    log().debug("MUMPS: refactorised {} unknowns with the previous analysis", size_);
  }

  [[nodiscard]] Vector solve(const Vector& rhs) const override {
    if (rhs.size() != size_) {
      throw InvalidArgument(
          fmt::format("MUMPS: right-hand side has {} entries, system has {}", rhs.size(), size_));
    }
    return solve_many(Matrix(rhs)).col(0);
  }

  /// Native multi-rhs solve (`nrhs` columns in one call, column-major in place).
  [[nodiscard]] Matrix solve_many(const Matrix& rhs) const override {
    return solve_columns(rhs, false);
  }

  [[nodiscard]] Vector solve_transposed(const Vector& rhs) const override {
    if (rhs.size() != size_) {
      throw InvalidArgument(
          fmt::format("MUMPS: right-hand side has {} entries, system has {}", rhs.size(), size_));
    }
    return solve_columns(Matrix(rhs), true).col(0);
  }

  /// Aᵀx = b on the same factors (ICNTL(9) ≠ 1; the LDLᵀ path ignores it, there Aᵀ = A).
  [[nodiscard]] Matrix solve_transposed_many(const Matrix& rhs) const override {
    return solve_columns(rhs, true);
  }

  /// All columns of `rhs` in one call (job 3), with A or with Aᵀ.
  [[nodiscard]] Matrix solve_columns(const Matrix& rhs, bool transposed) const {
    if (!ready_) throw Error("MUMPS: solve() called before a successful factorize()");
    if (rhs.rows() != size_) {
      throw InvalidArgument(
          fmt::format("MUMPS: right-hand sides have {} rows, system has {}", rhs.rows(), size_));
    }
    if (rhs.cols() == 0) return Matrix(size_, 0);
    const Index count = size_ * rhs.cols();
    std::vector<ZMUMPS_COMPLEX> x(as_size(count));
    for (Index j = 0; j < rhs.cols(); ++j) {
      for (Index i = 0; i < size_; ++i) {
        x[as_size(j * size_ + i)].r = rhs(i, j).real();
        x[as_size(j * size_ + i)].i = rhs(i, j).imag();
      }
    }
    auto& id = const_cast<ZMUMPS_STRUC_C&>(id_);  // the solve does not modify the factors
    id.rhs = x.data();
    id.nrhs = static_cast<MUMPS_INT>(rhs.cols());
    id.lrhs = static_cast<MUMPS_INT>(size_);
    id.icntl[8] = transposed ? 2 : 1;  // ICNTL(9): 1 solves A x = b, anything else A^T x = b
    id.job = 3;
    zmumps_c(&id);
    id.rhs = nullptr;
    id.icntl[8] = 1;
    check(transposed ? "transposed solve" : "solve");
    Matrix out(size_, rhs.cols());
    for (Index j = 0; j < rhs.cols(); ++j) {
      for (Index i = 0; i < size_; ++i) {
        out(i, j) = Complex{x[as_size(j * size_ + i)].r, x[as_size(j * size_ + i)].i};
      }
    }
    return out;
  }

  [[nodiscard]] Index size() const noexcept override { return size_; }
  [[nodiscard]] std::string name() const override {
    return fmt::format("MUMPS {} (sequential{})", MUMPS_VERSION, symmetric_ ? ", LDL^T" : "");
  }
  [[nodiscard]] Index factor_entries() const noexcept override {
    if (!ready_) return -1;
    return static_cast<Index>(infog(29) > 0 ? infog(29) : infog(9));
  }
  [[nodiscard]] std::string details() const override {
    if (!ready_) return {};
    // INFOG(29): entries in the factors (INFOG(9) if the 64-bit count is not set),
    // INFOG(21): memory effectively used by this process [MB]
    const auto entries = infog(29) > 0 ? infog(29) : infog(9);
    return fmt::format("factors {} entries ({} MB host)", entries, infog(21));
  }

 private:
  void initialise(bool symmetric) {
    id_ = ZMUMPS_STRUC_C{};
    id_.comm_fortran = kUseCommWorld;
    id_.par = 1;  // the host takes part in the factorisation
    // 0: general unsymmetric LU; 2: general symmetric (A = A^T, indefinite) LDL^T on one
    // triangle — complex symmetric systems are not Hermitian, so SYM = 1 never applies
    id_.sym = symmetric ? 2 : 0;
    id_.job = -1;
    zmumps_c(&id_);
    check("initialisation");
    icntl(1) = -1;   // no error messages on stdout
    icntl(2) = -1;   // no diagnostics
    icntl(3) = -1;   // no global information
    icntl(4) = 0;    // print level
    icntl(5) = 0;    // assembled matrix
    icntl(18) = 0;   // centralised input
    icntl(7) = 7;    // automatic ordering choice
    icntl(14) = 30;  // memory relaxation [%]
    initialised_ = true;
    symmetric_ = symmetric;
  }

  void finalize() {
    if (!initialised_) return;
    id_.job = -2;
    zmumps_c(&id_);
    initialised_ = false;
  }

  [[nodiscard]] MUMPS_INT& icntl(int i) { return id_.icntl[i - 1]; }
  [[nodiscard]] MUMPS_INT infog(int i) const { return id_.infog[i - 1]; }

  void check(const char* phase) const {
    if (infog(1) < 0) {
      throw Error(fmt::format("MUMPS: {} failed with INFOG(1) = {}, INFOG(2) = {}", phase, infog(1),
                              infog(2)));
    }
  }

  Symmetry symmetry_;
  bool symmetric_ = false;
  bool initialised_ = false;
  ZMUMPS_STRUC_C id_{};
  std::vector<MUMPS_INT> rows_;
  std::vector<MUMPS_INT> cols_;
  std::vector<ZMUMPS_COMPLEX> values_;
  bool ready_ = false;
  Index size_ = 0;
};

}  // namespace

std::unique_ptr<LinearSolver> make_mumps(Symmetry symmetry) {
  return std::make_unique<MumpsSolver>(symmetry);
}

}  // namespace hpfem::solvers

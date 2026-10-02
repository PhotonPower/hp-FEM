#pragma once
/// @file linear_solver.hpp
/// Interface for the direct solvers of the assembled complex systems and the available
/// backends: Eigen's SparseLU (always) and MUMPS (`HPFEM_ENABLE_MUMPS`, sequential complex
/// double build). See docs/theory/solvers.md.

#include <memory>
#include <string>
#include <vector>

#include "hpfem/core/types.hpp"

namespace hpfem::solvers {

/// Factorise once, solve for many right-hand sides.
class LinearSolver {
 public:
  LinearSolver() = default;
  LinearSolver(const LinearSolver&) = delete;
  LinearSolver& operator=(const LinearSolver&) = delete;
  LinearSolver(LinearSolver&&) = delete;
  LinearSolver& operator=(LinearSolver&&) = delete;
  virtual ~LinearSolver() = default;

  /// @throws Error if the matrix is singular or the factorisation fails.
  virtual void factorize(const SparseMatrix& matrix) = 0;
  /// @throws Error if `factorize` has not succeeded or the size does not match.
  [[nodiscard]] virtual Vector solve(const Vector& rhs) const = 0;
  [[nodiscard]] virtual Index size() const noexcept = 0;
  [[nodiscard]] virtual std::string name() const = 0;
};

/// Direct solver backends.
enum class DirectSolverBackend {
  kAuto,      ///< MUMPS if compiled in, otherwise SparseLU
  kSparseLu,  ///< Eigen SparseLU, always available
  kMumps,     ///< MUMPS (multifrontal, zmumps), needs `HPFEM_ENABLE_MUMPS`
};

/// True if the backend can be used in this build (`kAuto` always).
[[nodiscard]] bool available(DirectSolverBackend backend) noexcept;
/// All backends usable in this build.
[[nodiscard]] std::vector<DirectSolverBackend> available_backends();
[[nodiscard]] std::string backend_name(DirectSolverBackend backend);

/// Eigen's supernodal sparse LU with COLAMD ordering; general (non-symmetric) complex
/// matrices, O(fill) memory. Adequate up to a few 10^5 unknowns in 2D.
[[nodiscard]] std::unique_ptr<LinearSolver> make_sparse_lu();
/// MUMPS multifrontal LU (sequential build, unsymmetric complex double, automatic
/// ordering); the backend of choice for 3D and for many right-hand sides.
/// @throws Error if the library was not compiled in.
[[nodiscard]] std::unique_ptr<LinearSolver> make_mumps();
/// The requested backend, `kAuto` resolved as documented above.
/// @throws Error if the backend is not available.
[[nodiscard]] std::unique_ptr<LinearSolver> make_direct_solver(
    DirectSolverBackend backend = DirectSolverBackend::kAuto);

/// Factorise and solve in one go.
[[nodiscard]] Vector solve_direct(const SparseMatrix& matrix, const Vector& rhs,
                                  DirectSolverBackend backend = DirectSolverBackend::kAuto);

}  // namespace hpfem::solvers

#pragma once
/// @file linear_solver.hpp
/// Interface for linear solvers of the assembled complex systems and the Eigen SparseLU
/// backend. Further backends (MUMPS, PARDISO, iterative) plug in behind `LinearSolver`
/// (roadmap M6).

#include <memory>
#include <string>

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

/// Eigen's supernodal sparse LU with COLAMD ordering; general (non-symmetric) complex
/// matrices, O(fill) memory. Adequate up to a few 10^5 unknowns in 2D.
[[nodiscard]] std::unique_ptr<LinearSolver> make_sparse_lu();

/// Factorise and solve in one go.
[[nodiscard]] Vector solve_direct(const SparseMatrix& matrix, const Vector& rhs);

}  // namespace hpfem::solvers

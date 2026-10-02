#pragma once
/// @file eigen_solver.hpp
/// Generalized eigenproblems @f$ S x = \lambda M x @f$ of the curl–curl operator with
/// gauge (kernel filtering) through the discrete gradient. See
/// docs/theory/maxwell.md#eigenproblems.

#include <span>
#include <vector>

#include "hpfem/core/types.hpp"

namespace hpfem::solvers {

struct EigenOptions {
  Index num_eigenvalues = 6;  ///< physical eigenvalues wanted, smallest first
  /// Shift σ of the shift-invert transform; must be below the smallest wanted eigenvalue.
  /// Negative values keep S − σM positive definite on the gauged space.
  Real shift = -1.0;
  Index krylov_dimension = 0;  ///< 0: 2·num_eigenvalues + 10
  Real tolerance = 1e-10;
  int max_iterations = 2000;
};

struct EigenResult {
  RealVector eigenvalues;   ///< ascending, size = number converged
  RealMatrix eigenvectors;  ///< full-size columns (zeros on constrained DoFs), M-orthonormal
  int iterations = 0;
};

/// Smallest eigenpairs of @f$ S x = \lambda M x @f$ on the free DoFs, with the gradient
/// kernel removed: every Lanczos vector is projected M-orthogonally onto the complement of
/// range(G) (@f$ P = I - G (G^T M G)^{-1} G^T M @f$), so the zero eigenvalues of the curl
/// operator never appear and the returned values are the physical resonances. Shift-invert
/// Lanczos (Spectra) with Eigen SparseLU on @f$ S - \sigma M @f$.
///
/// S, M, G are the full matrices (Nédélec × Nédélec, Nédélec × H1); `free_nedelec` and
/// `free_h1` list the unconstrained DoFs (PEC removes the tangential DoFs of the Nédélec
/// space and the corresponding H1 boundary DoFs). Lossless media only: the imaginary parts
/// of S and M must vanish.
/// @throws InvalidArgument for complex matrices, Error if the iteration fails to converge.
[[nodiscard]] EigenResult gauged_curl_curl_eigenpairs(const SparseMatrix& stiffness,
                                                      const SparseMatrix& mass,
                                                      const SparseMatrix& gradient,
                                                      std::span<const Index> free_nedelec,
                                                      std::span<const Index> free_h1,
                                                      const EigenOptions& options = {});

}  // namespace hpfem::solvers

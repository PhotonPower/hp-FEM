#pragma once
/// @file eigen_solver.hpp
/// Generalized eigenproblems @f$ S x = \lambda M x @f$ of the curl–curl operator with
/// gauge (kernel filtering) through the discrete gradient. See
/// docs/theory/maxwell.md#eigenproblems.

#include <span>
#include <vector>

#include "hpfem/core/types.hpp"
#include "hpfem/solvers/linear_solver.hpp"

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

/// Eigenpairs of @f$ A x = \lambda B x @f$ closest to σ for real sparse A and B, where B may be
/// indefinite and A singular (waveguide mode pencils): Arnoldi (Spectra) on the real
/// nonsymmetric operator @f$ (A - \sigma B)^{-1} B @f$ with Eigen SparseLU, eigenvalues
/// @f$ \lambda = \sigma + 1/\nu @f$. Results are real (imaginary parts of the Ritz values must
/// be negligible, otherwise `Error`); eigenvectors have unit 2-norm and the size of A.
/// `options.shift` is ignored (σ is the argument).
/// @throws InvalidArgument for complex or mismatched matrices, Error on non-convergence.
[[nodiscard]] EigenResult generalized_eigenpairs_near(const SparseMatrix& a, const SparseMatrix& b,
                                                      Real sigma, const EigenOptions& options = {});

/// Result of `complex_eigenpairs_near`: eigenvalues ordered by distance to the shift.
struct ComplexEigenResult {
  Vector eigenvalues;       ///< closest to the shift first
  Matrix eigenvectors;      ///< unit 2-norm columns, full size
  RealVector residuals;     ///< relative Arnoldi residual of each Ritz pair
  int iterations = 0;       ///< Arnoldi restarts
  Index num_converged = 0;  ///< Ritz pairs below the tolerance (the leading ones)
};

/// Eigenpairs of the complex pencil @f$ A x = \lambda B x @f$ closest to a complex shift σ:
/// shift-invert Arnoldi on @f$ (A - \sigma B)^{-1} B @f$ with a direct factorisation
/// (`backend`), modified Gram–Schmidt, explicit restarts from the wanted Ritz vectors and the
/// relative residual @f$ |h_{m+1,m}\,y_m| / |	heta| @f$ as convergence test. Lossy media, PML
/// and complex frequencies are allowed: this is the solver of the resonance (quasi-normal
/// mode) problems. Both matrices are scaled to O(1) internally. `options.shift` is ignored
/// (σ is the argument). Eigenvectors are B-independent unit 2-norm vectors.
/// @throws InvalidArgument for mismatched matrices, Error if the factorisation fails or no
///         eigenvalue converges within `max_iterations` restarts.
[[nodiscard]] ComplexEigenResult complex_eigenpairs_near(
    const SparseMatrix& a, const SparseMatrix& b, Complex sigma, const EigenOptions& options = {},
    DirectSolverBackend backend = DirectSolverBackend::kAuto);

}  // namespace hpfem::solvers

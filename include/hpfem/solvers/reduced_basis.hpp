#pragma once
/// @file reduced_basis.hpp
/// Reduced-basis hook: snapshots (full solutions at a few parameter values) spanning an
/// orthonormal basis @f$ V @f$, Galerkin projections @f$ V^H A V @f$, @f$ V^H b @f$ of
/// assembled operators and the lift @f$ u \approx V y @f$. With an affine parameter
/// dependence, e.g. @f$ A(k) = S - k^2 M @f$ of a frequency sweep without dispersive
/// materials, the projections of @f$ S @f$ and @f$ M @f$ are formed once and every further
/// frequency costs a dense solve of the size of the basis. See
/// docs/theory/solvers.md#parameter-sweeps.

#include "hpfem/core/types.hpp"

namespace hpfem::solvers {

class ReducedBasis {
 public:
  /// @param num_dofs size of the full vectors; `tolerance` drops snapshots whose component
  ///        orthogonal to the basis is below `tolerance` times their norm.
  explicit ReducedBasis(Index num_dofs, Real tolerance = 1e-10);

  /// Orthonormalises the snapshot against the basis (modified Gram–Schmidt, twice) and
  /// appends it; returns false if it was (numerically) in the span already.
  /// @throws InvalidArgument for a wrong size.
  bool add_snapshot(const Vector& snapshot);

  [[nodiscard]] Index num_dofs() const noexcept { return num_dofs_; }
  [[nodiscard]] Index size() const noexcept { return basis_.cols(); }
  /// The orthonormal basis V (num_dofs × size).
  [[nodiscard]] const Matrix& basis() const noexcept { return basis_; }

  /// @f$ V^H A V @f$ (size × size).
  [[nodiscard]] Matrix project(const SparseMatrix& matrix) const;
  /// @f$ V^H b @f$.
  [[nodiscard]] Vector project(const Vector& vector) const;
  /// @f$ V y @f$.
  [[nodiscard]] Vector lift(const Vector& reduced) const;

 private:
  Index num_dofs_;
  Real tolerance_;
  Matrix basis_;
};

}  // namespace hpfem::solvers

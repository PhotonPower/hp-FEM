#pragma once
/// @file constraints.hpp
/// Linear constraints between degrees of freedom, @f$ x_s = \sum_i c_i\, x_{m_i} @f$ (slave =
/// combination of masters with complex coefficients): Bloch-periodic boundaries now, hanging
/// nodes of irregular refinement later (docs/theory/hp-adaptivity.md). Chains (a master that
/// is itself a slave) are resolved by substitution; cycles are rejected. Constraints are
/// applied to an assembled system by the prolongation @f$ x = P x_f @f$ of the free DoFs:
/// @f$ A_f = P^H A P @f$, @f$ b_f = P^H b @f$. The conjugate transpose selects the test
/// functions with the conjugate coefficients: for Bloch phases @f$ e^{ik\cdot a} @f$ with real
/// k the boundary terms of the weak form cancel only against test functions of phase
/// @f$ e^{-ik\cdot a} @f$ (for real coefficients, hanging nodes, it is the plain transpose).
/// For a complex wave vector (|phase| ≠ 1) the conjugate is not analytic in k; a separate test
/// space @f$ Q @f$ of the same structure (`set_test`, the phases @f$ 1/\bar\lambda @f$) gives
/// @f$ Q^H A P @f$, analytic in k and equal to @f$ P^H A P @f$ for real k.
/// See docs/theory/maxwell.md#bloch-periodic-constraints.

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include "hpfem/core/types.hpp"

namespace hpfem::fespace {

class Constraints {
 public:
  struct Term {
    Index master;
    Complex coefficient;
  };

  /// @throws InvalidArgument for a negative number of DoFs.
  explicit Constraints(Index num_dofs);

  /// Adds @f$ x_{slave} = \sum \text{coefficient}_i\, x_{master_i} @f$. Masters may be slaves of
  /// other constraints (resolved later).
  /// @throws InvalidArgument if an index is out of range, the slave is already constrained,
  ///         or a master equals the slave.
  void add(Index slave, std::vector<Term> terms);
  /// Adds every constraint of `other` (same number of DoFs), e.g. hanging-node constraints
  /// followed by Bloch-periodic ones. @throws InvalidArgument on a size mismatch or a DoF
  /// constrained by both.
  void append(const Constraints& other);

  [[nodiscard]] Index num_dofs() const noexcept { return num_dofs_; }
  [[nodiscard]] Index num_constrained() const noexcept { return num_constrained_; }
  [[nodiscard]] Index num_free() const noexcept { return num_dofs_ - num_constrained_; }
  [[nodiscard]] bool is_constrained(Index dof) const { return !terms_[as_size(dof)].empty(); }
  /// Resolved terms of a slave (masters are free DoFs), empty for a free DoF.
  /// @throws InvalidArgument if the constraints contain a cycle.
  [[nodiscard]] std::span<const Term> terms(Index slave) const;
  /// Index of a free DoF among the free DoFs (ascending), `kInvalidIndex` for a slave.
  [[nodiscard]] Index reduced_index(Index dof) const;

  /// Sets the test space @f$ Q @f$: the same slaves and resolved masters as these constraints,
  /// other coefficients. `reduce`, `reduce_rhs` then form @f$ Q^H A P @f$, @f$ Q^H b @f$. Used
  /// for Bloch phases off the unit circle, where `test` holds the phases @f$ 1/\bar\lambda @f$
  /// (its conjugate coefficients are the analytic continuation of the real-k test functions).
  /// A later `add` or `append` drops the test space.
  /// @throws InvalidArgument if the DoF count, the slaves or the resolved masters differ.
  void set_test(const Constraints& test);
  /// True after `set_test`.
  [[nodiscard]] bool has_test() const noexcept { return !test_.empty(); }
  /// Prolongation @f$ P @f$ (num_dofs × num_free): @f$ x = P x_f @f$.
  [[nodiscard]] SparseMatrix prolongation() const;
  /// Test prolongation @f$ Q @f$ (equal to @f$ P @f$ without `set_test`).
  [[nodiscard]] SparseMatrix test_prolongation() const;
  /// @f$ P x_f @f$: slaves filled in from their masters.
  [[nodiscard]] Vector expand(const Vector& reduced) const;
  /// @f$ P^H A P @f$ and @f$ P^H b @f$ (@f$ Q^H @f$ with a test space): the system on the free
  /// DoFs. @throws InvalidArgument if the sizes do not match `num_dofs()`.
  /// @f$ P^H b @f$ alone (a new load for an already reduced matrix).
  [[nodiscard]] Vector reduce_rhs(const Vector& rhs) const;
  [[nodiscard]] std::pair<SparseMatrix, Vector> reduce(const SparseMatrix& matrix,
                                                       const Vector& rhs) const;

 private:
  void resolve() const;

  Index num_dofs_;
  Index num_constrained_ = 0;
  std::vector<std::vector<Term>> terms_;  ///< raw (as added) or resolved terms per DoF
  mutable bool resolved_ = false;         ///< the reduced numbering is built on first use
  mutable std::vector<Index> reduced_;    ///< reduced index per DoF, built on resolve
  /// test coefficients aligned with the resolved terms, empty without a test space
  std::vector<std::vector<Complex>> test_;
  /// coefficient of resolved term k of DoF i in the test space
  [[nodiscard]] Complex test_coefficient(Index i, std::size_t k) const;
};

}  // namespace hpfem::fespace

#pragma once
/// @file kept_factorisation.hpp
/// The factorised system of a scattering solve, kept for further solves with the same operator
/// (ADR-0012 §4, M16 S1): tangent solves @f$ A s = r @f$ (the direct mode of the sensitivities)
/// and adjoint solves @f$ A^\top z = q @f$, both for full-size vectors on the DoF map of the
/// problem, without assembling or factorising again. The forward solve maps a full load to the
/// factorised system in fixed steps — static condensation of the interior DoFs (C), selection
/// of the unknowns (S: the conical solver drops its Dirichlet DoFs), hanging-node and Bloch
/// constraints (@f$ P^H @f$), zero data on the eliminated Dirichlet DoFs — and back by the
/// prolongation P and the interior recovery. The adjoint applies the transposes of the same
/// steps (@f$ C^\top @f$, @f$ P^\top @f$, @f$ \bar P @f$, the transposed interior recovery)
/// around a transposed solve (`LinearSolver::solve_transposed`), so that
/// @f[ q^\top s = z^\top r \qquad (s = \texttt{solve}(r),\ z = \texttt{solve\_adjoint}(q)) @f]
/// holds to round-off for every pair: the adjoint is the exact transpose of the discrete
/// solution operator, including ports, condensation and constraints. Dirichlet data is
/// homogeneous in both (sensitivities perturb the operator and the load, not the prescribed
/// boundary values). See docs/theory/maxwell.md#kept-factorisation.

#include <memory>
#include <optional>
#include <vector>

#include "hpfem/assembly/condensation.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/fespace/constraints.hpp"
#include "hpfem/solvers/linear_solver.hpp"

namespace hpfem::physics {

/// Factorisation and reduction steps of a solve (`ScatteringSetup::keep_factorisation`,
/// `ConicalScatteringSetup::keep_factorisation`); shared by the solution and its copies.
class KeptFactorisation {
 public:
  /// What the solve hands over.
  struct Parts {
    std::unique_ptr<solvers::LinearSolver> solver;  ///< factorised system matrix
    Index num_dofs = 0;                             ///< size of the full vectors
    /// Full indices of the unknowns before the constraints, in system order; empty: all.
    std::vector<Index> selection;
    std::optional<fespace::Constraints> constraints;  ///< on the selected unknowns
    /// Unknowns of the factorised system whose (homogeneous) data was eliminated.
    std::vector<Index> dirichlet;
    /// Interior elimination of the assembly (null without condensation).
    std::shared_ptr<const assembly::StaticCondensation> condensation;
  };

  /// @throws InvalidArgument if the solver is missing or the parts do not fit together.
  explicit KeptFactorisation(Parts parts);

  /// @f$ s = A^{-1} r @f$ for a full-size load with homogeneous Dirichlet data: the solution
  /// change of a perturbation with residual r.
  /// @throws InvalidArgument if the size does not match.
  [[nodiscard]] Vector solve(const Vector& load) const;
  /// Several loads at once (one per column, one multi-rhs solve).
  [[nodiscard]] Matrix solve_many(const Matrix& loads) const;
  /// @f$ z = A^{-\top} q @f$ for a full-size functional vector: the adjoint of
  /// @f$ Q = q^\top e @f$, zero on the Dirichlet DoFs, in the test space of the constraints.
  /// @throws InvalidArgument if the size does not match.
  [[nodiscard]] Vector solve_adjoint(const Vector& functional) const;
  /// Several functionals at once (one per column, one multi-rhs transposed solve).
  [[nodiscard]] Matrix solve_adjoint_many(const Matrix& functionals) const;

  /// Size of the full vectors.
  [[nodiscard]] Index num_dofs() const noexcept { return num_dofs_; }
  /// Unknowns of the factorised system.
  [[nodiscard]] Index size() const noexcept { return solver_->size(); }
  [[nodiscard]] const solvers::LinearSolver& solver() const noexcept { return *solver_; }

 private:
  [[nodiscard]] Matrix to_system(const Matrix& full, bool transposed) const;
  [[nodiscard]] Matrix from_system(const Matrix& reduced, const Matrix& full,
                                   bool transposed) const;

  std::unique_ptr<solvers::LinearSolver> solver_;
  Index num_dofs_;
  std::vector<Index> selection_;
  std::optional<SparseMatrix> prolongation_;  ///< P of the constraints
  std::vector<Index> dirichlet_;
  std::shared_ptr<const assembly::StaticCondensation> condensation_;
};

}  // namespace hpfem::physics

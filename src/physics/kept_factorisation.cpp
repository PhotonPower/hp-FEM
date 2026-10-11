#include "hpfem/physics/kept_factorisation.hpp"

#include <utility>

#include <fmt/format.h>

#include "hpfem/core/error.hpp"

namespace hpfem::physics {

KeptFactorisation::KeptFactorisation(Parts parts)
    : solver_(std::move(parts.solver)),
      num_dofs_(parts.num_dofs),
      selection_(std::move(parts.selection)),
      dirichlet_(std::move(parts.dirichlet)),
      condensation_(std::move(parts.condensation)) {
  if (!solver_) throw InvalidArgument("KeptFactorisation: no factorised solver");
  const Index selected = selection_.empty() ? num_dofs_ : static_cast<Index>(selection_.size());
  for (const Index dof : selection_) {
    if (dof < 0 || dof >= num_dofs_) {
      throw InvalidArgument(
          fmt::format("KeptFactorisation: selected DoF {} outside 0..{}", dof, num_dofs_ - 1));
    }
  }
  if (parts.constraints) {
    if (parts.constraints->num_dofs() != selected) {
      throw InvalidArgument(fmt::format("KeptFactorisation: constraints on {} DoFs, {} selected",
                                        parts.constraints->num_dofs(), selected));
    }
    prolongation_ = parts.constraints->prolongation();
    if (parts.constraints->has_test()) test_prolongation_ = parts.constraints->test_prolongation();
  }
  const Index system = prolongation_ ? prolongation_->cols() : selected;
  if (system != solver_->size()) {
    throw InvalidArgument(
        fmt::format("KeptFactorisation: the reduction gives {} unknowns, the solver has {}", system,
                    solver_->size()));
  }
  for (const Index dof : dirichlet_) {
    if (dof < 0 || dof >= system) {
      throw InvalidArgument(
          fmt::format("KeptFactorisation: Dirichlet unknown {} outside 0..{}", dof, system - 1));
    }
  }
  if (condensation_ && condensation_->num_dofs() != num_dofs_) {
    throw InvalidArgument("KeptFactorisation: the condensation does not match the DoF map");
  }
}

Matrix KeptFactorisation::to_system(const Matrix& full, bool transposed,
                                    const Matrix* system_loads) const {
  if (full.rows() != num_dofs_) {
    throw InvalidArgument(
        fmt::format("KeptFactorisation: {} rows for {} DoFs", full.rows(), num_dofs_));
  }
  Matrix condensed = full;
  if (condensation_) {
    for (Index j = 0; j < full.cols(); ++j) {
      const Vector column = full.col(j);
      condensed.col(j) = transposed ? condensation_->condense_load_transposed(column)
                                    : condensation_->condense_load(column);
    }
  }
  Matrix selected;
  if (selection_.empty()) {
    selected = std::move(condensed);
  } else {
    selected.resize(static_cast<Index>(selection_.size()), full.cols());
    for (std::size_t i = 0; i < selection_.size(); ++i) {
      selected.row(static_cast<Index>(i)) = condensed.row(selection_[i]);
    }
  }
  // forward: Q^H r (the test functions with conjugate coefficients, Q = P without a separate
  // test space); adjoint: P^T q
  Matrix reduced;
  if (prolongation_) {
    const SparseMatrix& q = test_prolongation_ ? *test_prolongation_ : *prolongation_;
    reduced =
        transposed ? Matrix(prolongation_->transpose() * selected) : Matrix(q.adjoint() * selected);
  } else {
    reduced = std::move(selected);
  }
  if (system_loads != nullptr) reduced += *system_loads;
  for (const Index dof : dirichlet_) reduced.row(dof).setZero();
  return reduced;
}

Matrix KeptFactorisation::from_system(const Matrix& reduced, const Matrix& full,
                                      bool transposed) const {
  // forward: P x; adjoint: conj(Q) y
  Matrix selected;
  if (prolongation_) {
    const SparseMatrix& q = test_prolongation_ ? *test_prolongation_ : *prolongation_;
    selected = transposed ? Matrix(q.conjugate() * reduced) : Matrix(*prolongation_ * reduced);
  } else {
    selected = reduced;
  }
  Matrix out;
  if (selection_.empty()) {
    out = std::move(selected);
  } else {
    out = Matrix::Zero(num_dofs_, reduced.cols());
    for (std::size_t i = 0; i < selection_.size(); ++i) {
      out.row(selection_[i]) = selected.row(static_cast<Index>(i));
    }
  }
  if (condensation_) {
    for (Index j = 0; j < out.cols(); ++j) {
      const Vector column = out.col(j);
      const Vector given = full.col(j);
      out.col(j) = transposed ? condensation_->recover_transposed(column, given)
                              : condensation_->recover(column, given);
    }
  }
  return out;
}

Vector KeptFactorisation::solve(const Vector& load) const {
  return solve_many(Matrix(load)).col(0);
}

Matrix KeptFactorisation::solve_many(const Matrix& loads) const {
  if (loads.cols() == 0) return Matrix(num_dofs_, 0);
  return from_system(solver_->solve_many(to_system(loads, false)), loads, false);
}

Matrix KeptFactorisation::solve_many(const Matrix& loads, const Matrix& system_loads) const {
  if (system_loads.rows() != solver_->size() || system_loads.cols() != loads.cols()) {
    throw InvalidArgument(
        fmt::format("KeptFactorisation: system loads {} x {} for {} unknowns and {} loads",
                    system_loads.rows(), system_loads.cols(), solver_->size(), loads.cols()));
  }
  if (loads.cols() == 0) return Matrix(num_dofs_, 0);
  return from_system(solver_->solve_many(to_system(loads, false, &system_loads)), loads, false);
}

Vector KeptFactorisation::solve_adjoint(const Vector& functional) const {
  return solve_adjoint_many(Matrix(functional)).col(0);
}

Matrix KeptFactorisation::solve_adjoint_many(const Matrix& functionals) const {
  if (functionals.cols() == 0) return Matrix(num_dofs_, 0);
  return from_system(solver_->solve_transposed_many(to_system(functionals, true)), functionals,
                     true);
}

}  // namespace hpfem::physics

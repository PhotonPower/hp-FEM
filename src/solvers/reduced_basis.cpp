#include "hpfem/solvers/reduced_basis.hpp"

#include <fmt/format.h>

#include "hpfem/core/error.hpp"

namespace hpfem::solvers {

ReducedBasis::ReducedBasis(Index num_dofs, Real tolerance)
    : num_dofs_(num_dofs), tolerance_(tolerance), basis_(num_dofs, 0) {
  if (num_dofs < 0) throw InvalidArgument("ReducedBasis: negative number of DoFs");
}

bool ReducedBasis::add_snapshot(const Vector& snapshot) {
  if (snapshot.size() != num_dofs_) {
    throw InvalidArgument(fmt::format("ReducedBasis::add_snapshot: {} entries for {} DoFs",
                                      snapshot.size(), num_dofs_));
  }
  const Real norm = snapshot.norm();
  if (!(norm > 0)) return false;
  Vector v = snapshot;
  for (int pass = 0; pass < 2; ++pass) {  // modified Gram-Schmidt with re-orthogonalisation
    for (Index j = 0; j < basis_.cols(); ++j) {
      const Complex coefficient = basis_.col(j).dot(v);  // conjugates the basis column
      v -= coefficient * basis_.col(j);
    }
  }
  const Real remaining = v.norm();
  if (remaining <= tolerance_ * norm) return false;
  basis_.conservativeResize(Eigen::NoChange, basis_.cols() + 1);
  basis_.col(basis_.cols() - 1) = v / remaining;
  return true;
}

Matrix ReducedBasis::project(const SparseMatrix& matrix) const {
  if (matrix.rows() != num_dofs_ || matrix.cols() != num_dofs_) {
    throw InvalidArgument("ReducedBasis::project: matrix size does not match");
  }
  const Matrix av = matrix * basis_;
  return basis_.adjoint() * av;
}

Vector ReducedBasis::project(const Vector& vector) const {
  if (vector.size() != num_dofs_) {
    throw InvalidArgument("ReducedBasis::project: vector size does not match");
  }
  return basis_.adjoint() * vector;
}

Vector ReducedBasis::lift(const Vector& reduced) const {
  if (reduced.size() != basis_.cols()) {
    throw InvalidArgument(fmt::format("ReducedBasis::lift: {} coefficients for a basis of {}",
                                      reduced.size(), basis_.cols()));
  }
  return basis_ * reduced;
}

}  // namespace hpfem::solvers

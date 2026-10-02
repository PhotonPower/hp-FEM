#include "hpfem/fespace/constraints.hpp"

#include <map>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "hpfem/core/error.hpp"

namespace hpfem::fespace {

namespace {
Index checked_size(Index num_dofs) {
  if (num_dofs < 0) throw InvalidArgument("Constraints: negative number of DoFs");
  return num_dofs;
}
}  // namespace

Constraints::Constraints(Index num_dofs)
    : num_dofs_(checked_size(num_dofs)), terms_(as_size(num_dofs)) {}

void Constraints::add(Index slave, std::vector<Term> terms) {
  if (slave < 0 || slave >= num_dofs_) {
    throw InvalidArgument(
        fmt::format("Constraints::add: slave {} outside 0..{}", slave, num_dofs_ - 1));
  }
  if (!terms_[as_size(slave)].empty()) {
    throw InvalidArgument(fmt::format("Constraints::add: DoF {} is already constrained", slave));
  }
  if (terms.empty()) {
    throw InvalidArgument(fmt::format("Constraints::add: DoF {} needs at least one master", slave));
  }
  for (const auto& t : terms) {
    if (t.master < 0 || t.master >= num_dofs_) {
      throw InvalidArgument(fmt::format("Constraints::add: master {} of DoF {} outside 0..{}",
                                        t.master, slave, num_dofs_ - 1));
    }
    if (t.master == slave) {
      throw InvalidArgument(
          fmt::format("Constraints::add: DoF {} cannot be its own master", slave));
    }
  }
  terms_[as_size(slave)] = std::move(terms);
  ++num_constrained_;
  resolved_ = false;
}

void Constraints::append(const Constraints& other) {
  if (other.num_dofs_ != num_dofs_) {
    throw InvalidArgument(fmt::format("Constraints::append: {} DoFs cannot take constraints of {}",
                                      num_dofs_, other.num_dofs_));
  }
  for (Index slave = 0; slave < num_dofs_; ++slave) {
    if (!other.is_constrained(slave)) continue;
    const auto terms = other.terms(slave);
    add(slave, std::vector<Term>(terms.begin(), terms.end()));
  }
}

void Constraints::resolve() const {
  if (resolved_) return;
  auto& terms = const_cast<std::vector<std::vector<Term>>&>(terms_);
  // substitute constrained masters until every master is free; a cycle never terminates,
  // so bound the number of sweeps by the number of constraints
  for (Index sweep = 0; sweep <= num_constrained_; ++sweep) {
    bool changed = false;
    for (auto& list : terms) {
      if (list.empty()) continue;
      bool nested = false;
      for (const auto& t : list) {
        if (!terms[as_size(t.master)].empty()) nested = true;
      }
      if (!nested) continue;
      changed = true;
      std::map<Index, Complex> combined;
      for (const auto& t : list) {
        const auto& inner = terms[as_size(t.master)];
        if (inner.empty()) {
          combined[t.master] += t.coefficient;
        } else {
          for (const auto& u : inner) combined[u.master] += t.coefficient * u.coefficient;
        }
      }
      list.clear();
      for (const auto& [master, coefficient] : combined) list.push_back({master, coefficient});
    }
    if (!changed) break;
    if (sweep == num_constrained_) {
      throw InvalidArgument("Constraints: cyclic constraints (a DoF depends on itself)");
    }
  }
  reduced_.assign(as_size(num_dofs_), kInvalidIndex);
  Index next = 0;
  for (Index i = 0; i < num_dofs_; ++i) {
    if (terms_[as_size(i)].empty()) reduced_[as_size(i)] = next++;
  }
  resolved_ = true;
}

std::span<const Constraints::Term> Constraints::terms(Index slave) const {
  resolve();
  return terms_[as_size(slave)];
}

Index Constraints::reduced_index(Index dof) const {
  resolve();
  return reduced_[as_size(dof)];
}

SparseMatrix Constraints::prolongation() const {
  resolve();
  std::vector<Eigen::Triplet<Complex, Index>> triplets;
  triplets.reserve(as_size(num_dofs_) + as_size(num_constrained_));
  for (Index i = 0; i < num_dofs_; ++i) {
    const auto& list = terms_[as_size(i)];
    if (list.empty()) {
      triplets.emplace_back(i, reduced_[as_size(i)], Complex{1.0, 0.0});
    } else {
      for (const auto& t : list)
        triplets.emplace_back(i, reduced_[as_size(t.master)], t.coefficient);
    }
  }
  SparseMatrix p(num_dofs_, num_free());
  p.setFromTriplets(triplets.begin(), triplets.end());
  return p;
}

Vector Constraints::expand(const Vector& reduced) const {
  resolve();
  if (reduced.size() != num_free()) {
    throw InvalidArgument(
        fmt::format("Constraints::expand: {} values for {} free DoFs", reduced.size(), num_free()));
  }
  Vector full(num_dofs_);
  for (Index i = 0; i < num_dofs_; ++i) {
    const auto& list = terms_[as_size(i)];
    if (list.empty()) {
      full(i) = reduced(reduced_[as_size(i)]);
    } else {
      Complex v = 0;
      for (const auto& t : list) v += t.coefficient * reduced(reduced_[as_size(t.master)]);
      full(i) = v;
    }
  }
  return full;
}

Vector Constraints::reduce_rhs(const Vector& rhs) const {
  if (rhs.size() != num_dofs_) {
    throw InvalidArgument("Constraints::reduce_rhs: size does not match the constraints");
  }
  return Vector(SparseMatrix(prolongation().adjoint()) * rhs);
}

std::pair<SparseMatrix, Vector> Constraints::reduce(const SparseMatrix& matrix,
                                                    const Vector& rhs) const {
  if (matrix.rows() != num_dofs_ || matrix.cols() != num_dofs_ || rhs.size() != num_dofs_) {
    throw InvalidArgument("Constraints::reduce: system size does not match the constraints");
  }
  const SparseMatrix p = prolongation();
  const SparseMatrix pt = SparseMatrix(p.adjoint());
  SparseMatrix reduced = pt * matrix * p;
  reduced.makeCompressed();
  return {std::move(reduced), Vector(pt * rhs)};
}

}  // namespace hpfem::fespace

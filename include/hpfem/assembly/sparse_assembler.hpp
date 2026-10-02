#pragma once
/// @file sparse_assembler.hpp
/// Accumulation of element contributions into a global sparse matrix (COO → CSR) and
/// scatter of element vectors. See docs/theory/scalar-fem.md#assembly.

#include <cstddef>
#include <span>
#include <vector>

#include <Eigen/SparseCore>

#include "hpfem/core/types.hpp"

namespace hpfem::assembly {

/// Collects (row, col, value) triplets from the element loop and compresses them into the
/// row-major CSR `hpfem::SparseMatrix`; duplicate entries are summed. Finalisation costs
/// O(nnz log nnz).
class SparseAssembler {
 public:
  SparseAssembler(Index rows, Index cols);

  [[nodiscard]] Index rows() const noexcept { return rows_; }
  [[nodiscard]] Index cols() const noexcept { return cols_; }
  [[nodiscard]] std::size_t num_triplets() const noexcept { return triplets_.size(); }
  void reserve(std::size_t triplets) { triplets_.reserve(triplets); }

  /// Adds `local(i, j)` to the global entry `(rows[i], cols[j])`; sizes must match.
  void add(std::span<const Index> rows, std::span<const Index> cols, const Matrix& local);
  /// Adds one entry.
  void add(Index row, Index col, Complex value);

  /// Compressed matrix with all contributions summed.
  [[nodiscard]] SparseMatrix finalize() const;

 private:
  Index rows_;
  Index cols_;
  std::vector<Eigen::Triplet<Complex, Index>> triplets_;
};

/// Adds the element vector `local` to `global` at the positions `dofs`.
void scatter(Vector& global, std::span<const Index> dofs, const Vector& local);

/// Gathers `global` at the positions `dofs` into a dense vector.
[[nodiscard]] Vector gather(const Vector& global, std::span<const Index> dofs);

}  // namespace hpfem::assembly

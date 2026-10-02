#include "hpfem/assembly/sparse_assembler.hpp"

#include <fmt/format.h>

#include "hpfem/core/error.hpp"

namespace hpfem::assembly {

SparseAssembler::SparseAssembler(Index rows, Index cols) : rows_(rows), cols_(cols) {
  if (rows < 0 || cols < 0) {
    throw InvalidArgument(fmt::format("SparseAssembler: negative size {} x {}", rows, cols));
  }
}

void SparseAssembler::add(std::span<const Index> rows, std::span<const Index> cols,
                          const Matrix& local) {
  HPFEM_ASSERT(static_cast<Index>(rows.size()) == local.rows() &&
                   static_cast<Index>(cols.size()) == local.cols(),
               "SparseAssembler::add: index lists do not match the local matrix");
  for (Index i = 0; i < local.rows(); ++i) {
    HPFEM_ASSERT(rows[as_size(i)] >= 0 && rows[as_size(i)] < rows_, "row index out of range");
    for (Index j = 0; j < local.cols(); ++j) {
      HPFEM_ASSERT(cols[as_size(j)] >= 0 && cols[as_size(j)] < cols_, "column index out of range");
      triplets_.emplace_back(rows[as_size(i)], cols[as_size(j)], local(i, j));
    }
  }
}

void SparseAssembler::add(Index row, Index col, Complex value) {
  HPFEM_ASSERT(row >= 0 && row < rows_ && col >= 0 && col < cols_, "index out of range");
  triplets_.emplace_back(row, col, value);
}

SparseMatrix SparseAssembler::finalize() const {
  SparseMatrix matrix(rows_, cols_);
  matrix.setFromTriplets(triplets_.begin(), triplets_.end());
  matrix.makeCompressed();
  return matrix;
}

void scatter(Vector& global, std::span<const Index> dofs, const Vector& local) {
  HPFEM_ASSERT(static_cast<Index>(dofs.size()) == local.size(),
               "scatter: index list does not match the local vector");
  for (Index i = 0; i < local.size(); ++i) {
    HPFEM_ASSERT(dofs[as_size(i)] >= 0 && dofs[as_size(i)] < global.size(), "index out of range");
    global(dofs[as_size(i)]) += local(i);
  }
}

Vector gather(const Vector& global, std::span<const Index> dofs) {
  Vector local(static_cast<Index>(dofs.size()));
  for (Index i = 0; i < local.size(); ++i) {
    HPFEM_ASSERT(dofs[as_size(i)] >= 0 && dofs[as_size(i)] < global.size(), "index out of range");
    local(i) = global(dofs[as_size(i)]);
  }
  return local;
}

}  // namespace hpfem::assembly

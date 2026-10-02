#pragma once
/// @file discrete_gradient.hpp
/// The discrete gradient G: coefficients of the H1 space → coefficients of the Nédélec space
/// such that ∇(Σ u_i φ_i^{H1}) = Σ (G u)_j φ_j^{ND} exactly (docs/theory/nedelec.md#gauging).
/// Because the hierarchical Nédélec basis contains the gradients of the H1 basis explicitly,
/// G has entries ±1 (Whitney functions of the edges at a vertex) and 1 (higher functions).

#include <span>
#include <vector>

#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"

namespace hpfem::assembly {

/// Sparse N_ND × N_H1 matrix with S G = 0 for every curl–curl stiffness matrix S.
/// Both maps must live on the same mesh with the same cell orders.
/// @throws InvalidArgument otherwise.
template <int Dim>
[[nodiscard]] SparseMatrix discrete_gradient(const fespace::DofMap<Dim>& h1,
                                             const fespace::NedelecDofMap<Dim>& nedelec);

/// Rows `rows` and columns `cols` of a sparse matrix as a new compressed matrix.
[[nodiscard]] SparseMatrix extract(const SparseMatrix& matrix, std::span<const Index> rows,
                                   std::span<const Index> cols);

/// Complement of a sorted list of constrained DoFs in 0..n-1, ascending.
[[nodiscard]] std::vector<Index> free_dofs(Index n, std::span<const Index> constrained);

extern template SparseMatrix discrete_gradient<2>(const fespace::DofMap<2>&,
                                                  const fespace::NedelecDofMap<2>&);
extern template SparseMatrix discrete_gradient<3>(const fespace::DofMap<3>&,
                                                  const fespace::NedelecDofMap<3>&);

}  // namespace hpfem::assembly

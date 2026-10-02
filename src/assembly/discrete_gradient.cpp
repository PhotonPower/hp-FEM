#include "hpfem/assembly/discrete_gradient.hpp"

#include <algorithm>

#include <fmt/format.h>

#include "hpfem/assembly/sparse_assembler.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/h1_basis.hpp"
#include "hpfem/fespace/nedelec_basis.hpp"

namespace hpfem::assembly {

template <int Dim>
SparseMatrix discrete_gradient(const fespace::DofMap<Dim>& h1,
                               const fespace::NedelecDofMap<Dim>& nedelec) {
  const auto& mesh = h1.mesh();
  if (&mesh != &nedelec.mesh()) {
    throw InvalidArgument("discrete_gradient: the two DoF maps live on different meshes");
  }
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    if (h1.cell_order(c) != nedelec.cell_order(c)) {
      throw InvalidArgument(
          fmt::format("discrete_gradient: cell {} has H1 order {} but Nédélec order {}", c,
                      h1.cell_order(c), nedelec.cell_order(c)));
    }
  }
  SparseAssembler g(nedelec.num_dofs(), h1.num_dofs());

  // edges: Whitney function = gradient of the second vertex minus the first (global order),
  // higher edge functions are the gradients of the H1 edge functions one by one
  for (Index e = 0; e < mesh.num_edges(); ++e) {
    const auto& ev = mesh.edge_vertices(e);
    const auto nd = nedelec.edge_dofs(e);
    const auto h = h1.edge_dofs(e);
    g.add(nd[0], h1.vertex_dof(ev[0]), Complex{-1.0, 0.0});
    g.add(nd[0], h1.vertex_dof(ev[1]), Complex{1.0, 0.0});
    for (std::size_t i = 0; i < h.size(); ++i) g.add(nd[i + 1], h[i], Complex{1.0, 0.0});
  }
  // faces (3D): the Type-1 face functions are the gradients of the H1 face functions
  if constexpr (Dim == 3) {
    for (Index f = 0; f < mesh.num_faces(); ++f) {
      const auto nd = nedelec.face_dofs(f);
      const auto h = h1.face_dofs(f);
      for (std::size_t i = 0; i < h.size(); ++i) g.add(nd[i], h[i], Complex{1.0, 0.0});
    }
  }
  // interiors: Type-1 interior functions likewise
  for (Index c = 0; c < mesh.num_cells(); ++c) {
    const auto nd = nedelec.interior_dofs(c);
    const auto h = h1.interior_dofs(c);
    for (std::size_t i = 0; i < h.size(); ++i) g.add(nd[i], h[i], Complex{1.0, 0.0});
  }
  return g.finalize();
}

SparseMatrix extract(const SparseMatrix& matrix, std::span<const Index> rows,
                     std::span<const Index> cols) {
  std::vector<Index> col_position(as_size(matrix.cols()), kInvalidIndex);
  for (std::size_t j = 0; j < cols.size(); ++j)
    col_position[as_size(cols[j])] = static_cast<Index>(j);
  SparseAssembler out(static_cast<Index>(rows.size()), static_cast<Index>(cols.size()));
  for (std::size_t i = 0; i < rows.size(); ++i) {
    for (SparseMatrix::InnerIterator it(matrix, rows[i]); it; ++it) {
      const Index j = col_position[as_size(it.col())];
      if (j != kInvalidIndex) out.add(static_cast<Index>(i), j, it.value());
    }
  }
  return out.finalize();
}

std::vector<Index> free_dofs(Index n, std::span<const Index> constrained) {
  std::vector<char> is_constrained(as_size(n), 0);
  for (const Index c : constrained) is_constrained[as_size(c)] = 1;
  std::vector<Index> free;
  free.reserve(as_size(n) - constrained.size());
  for (Index i = 0; i < n; ++i) {
    if (is_constrained[as_size(i)] == 0) free.push_back(i);
  }
  return free;
}

template SparseMatrix discrete_gradient<2>(const fespace::DofMap<2>&,
                                           const fespace::NedelecDofMap<2>&);
template SparseMatrix discrete_gradient<3>(const fespace::DofMap<3>&,
                                           const fespace::NedelecDofMap<3>&);

}  // namespace hpfem::assembly

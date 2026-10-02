#pragma once
/// @file dof_map.hpp
/// Global numbering of the degrees of freedom of a hierarchical H1 space with variable
/// polynomial order per cell (minimum rule on shared edges and faces).
/// See docs/theory/h1-basis.md#dof-map.

#include <span>
#include <vector>

#include "hpfem/core/types.hpp"
#include "hpfem/fespace/h1_basis.hpp"
#include "hpfem/mesh/mesh.hpp"

namespace hpfem::fespace {

/// Numbers the DoFs of the H1 space on a mesh: one per vertex, p_e − 1 per edge,
/// (p_f − 1)(p_f − 2)/2 per face (3D) and the interior functions per cell, in that order
/// (vertices first, then all edges, faces, cells). The order of an edge or face is the
/// minimum of the orders of its cells, so the space is conforming and every entity function
/// is shared by all adjacent cells. Cell-local DoF lists follow the function order of
/// `H1Basis`, with `cell_layout(c)` supplying orders and orientations.
template <int Dim>
class DofMap {
 public:
  /// Variable order: one entry per cell, all ≥ 1.
  DofMap(const mesh::Mesh<Dim>& mesh, std::vector<int> cell_orders);
  /// Uniform order p on every cell.
  DofMap(const mesh::Mesh<Dim>& mesh, int order);

  [[nodiscard]] const mesh::Mesh<Dim>& mesh() const noexcept { return *mesh_; }
  [[nodiscard]] Index num_dofs() const noexcept { return num_dofs_; }
  [[nodiscard]] int cell_order(Index c) const { return cell_orders_[as_size(c)]; }
  [[nodiscard]] int edge_order(Index e) const { return edge_orders_[as_size(e)]; }
  [[nodiscard]] int face_order(Index f) const
    requires(Dim == 3)
  {
    return face_orders_[as_size(f)];
  }
  [[nodiscard]] int max_order() const noexcept { return max_order_; }

  /// Global DoF of vertex v.
  [[nodiscard]] Index vertex_dof(Index v) const noexcept { return v; }
  /// Global DoFs of the edge functions of edge e (ascending degree), possibly empty.
  [[nodiscard]] std::span<const Index> edge_dofs(Index e) const;
  [[nodiscard]] std::span<const Index> face_dofs(Index f) const
    requires(Dim == 3);
  [[nodiscard]] std::span<const Index> interior_dofs(Index c) const;

  /// Basis layout (orders and orientation codes) of cell c.
  [[nodiscard]] H1Layout<Dim> cell_layout(Index c) const;
  /// Global DoFs of cell c in `H1Basis` function order.
  [[nodiscard]] std::span<const Index> cell_dofs(Index c) const;

  /// All DoFs whose functions have support on facet f (its vertices, edges and, in 3D, the
  /// face itself), ascending. These are the DoFs fixed by a Dirichlet condition on f.
  [[nodiscard]] std::vector<Index> facet_dofs(Index f) const;
  /// Union of `facet_dofs` over all facets carrying `tag`, sorted and unique.
  [[nodiscard]] std::vector<Index> dofs_on_tag(mesh::Tag tag) const;

 private:
  void build();

  const mesh::Mesh<Dim>* mesh_;
  std::vector<int> cell_orders_;
  std::vector<int> edge_orders_;
  std::vector<int> face_orders_;
  int max_order_ = 1;
  Index num_dofs_ = 0;
  std::vector<Index> all_dofs_;      ///< vertex, edge, face and interior DoFs in one array
  std::vector<Index> edge_offsets_;  ///< into all_dofs_, size E + 1
  std::vector<Index> face_offsets_;  ///< into all_dofs_, size F + 1 (3D)
  std::vector<Index> cell_offsets_;  ///< into all_dofs_, size C + 1 (interior DoFs)
  std::vector<Index> cell_dofs_;     ///< CSR of cell-local DoF lists
  std::vector<Index> cell_dof_offsets_;
};

extern template class DofMap<2>;
extern template class DofMap<3>;

}  // namespace hpfem::fespace

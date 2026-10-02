#pragma once
/// @file dof_map.hpp
/// Global numbering of the degrees of freedom of the hierarchical spaces with variable
/// polynomial order per cell (minimum rule on shared edges and faces). One generic
/// `EntityDofMap` serves the H1 space (`DofMap`) and the H(curl) space (`NedelecDofMap`);
/// they differ only in the number of DoFs per entity. See docs/theory/h1-basis.md#dof-map.

#include <span>
#include <vector>

#include "hpfem/core/types.hpp"
#include "hpfem/fespace/cell_layout.hpp"
#include "hpfem/fespace/h1_basis.hpp"
#include "hpfem/fespace/nedelec_basis.hpp"
#include "hpfem/mesh/mesh.hpp"

namespace hpfem::fespace {

/// DoFs per entity of the H1 space: one per vertex, p − 1 per edge, …
struct H1Counts {
  static constexpr Index kVertexDofs = 1;
  [[nodiscard]] static constexpr Index edge(int p) noexcept { return h1_edge_functions(p); }
  [[nodiscard]] static constexpr Index face(int p) noexcept { return h1_face_functions(p); }
  template <int Dim>
  [[nodiscard]] static constexpr Index cell(int p) noexcept {
    return h1_cell_functions<Dim>(p);
  }
};

/// DoFs per entity of the Nédélec space: none per vertex, p per edge, p(p − 1) per face, …
struct NedelecCounts {
  static constexpr Index kVertexDofs = 0;
  [[nodiscard]] static constexpr Index edge(int p) noexcept { return nedelec_edge_functions(p); }
  [[nodiscard]] static constexpr Index face(int p) noexcept { return nedelec_face_functions(p); }
  template <int Dim>
  [[nodiscard]] static constexpr Index cell(int p) noexcept {
    return nedelec_cell_functions<Dim>(p);
  }
};

/// Numbers the DoFs of a hierarchical space on a mesh: vertex DoFs (if the space has them),
/// then the edge functions edge by edge, the face functions (3D) and the interior functions
/// cell by cell. The order of an edge or face is the minimum of the orders of its cells, so
/// the space is conforming and every entity function is shared by all adjacent cells.
/// Cell-local DoF lists follow the function order of the corresponding basis, with
/// `cell_layout(c)` supplying orders and orientations.
template <int Dim, class Counts>
class EntityDofMap {
 public:
  /// Variable order: one entry per cell, all ≥ 1.
  EntityDofMap(const mesh::Mesh<Dim>& mesh, std::vector<int> cell_orders);
  /// Uniform order p on every cell.
  EntityDofMap(const mesh::Mesh<Dim>& mesh, int order);

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

  /// Global DoF of vertex v (H1 spaces only).
  [[nodiscard]] Index vertex_dof(Index v) const noexcept
    requires(Counts::kVertexDofs == 1)
  {
    return v;
  }
  /// Global DoFs of the functions of edge e (ascending degree), possibly empty.
  [[nodiscard]] std::span<const Index> edge_dofs(Index e) const;
  [[nodiscard]] std::span<const Index> face_dofs(Index f) const
    requires(Dim == 3);
  [[nodiscard]] std::span<const Index> interior_dofs(Index c) const;

  /// Basis layout (orders and orientation codes) of cell c.
  [[nodiscard]] CellLayout<Dim> cell_layout(Index c) const;
  /// Global DoFs of cell c in basis function order.
  [[nodiscard]] std::span<const Index> cell_dofs(Index c) const;

  /// All DoFs whose functions have support on facet f (its vertices, edges and, in 3D, the
  /// face itself), ascending: the DoFs fixed by an essential condition on f.
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

/// DoF map of the hierarchical H1 space (`H1Basis`).
template <int Dim>
using DofMap = EntityDofMap<Dim, H1Counts>;
/// DoF map of the hierarchical H(curl) space (`NedelecBasis`).
template <int Dim>
using NedelecDofMap = EntityDofMap<Dim, NedelecCounts>;

extern template class EntityDofMap<2, H1Counts>;
extern template class EntityDofMap<3, H1Counts>;
extern template class EntityDofMap<2, NedelecCounts>;
extern template class EntityDofMap<3, NedelecCounts>;

}  // namespace hpfem::fespace

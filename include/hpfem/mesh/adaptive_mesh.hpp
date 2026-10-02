#pragma once
/// @file adaptive_mesh.hpp
/// Local (adaptive) h-refinement: a tree of red-refined cells over a root mesh whose leaves
/// form the current mesh. The leaf mesh is one-irregular (2:1 balanced by vertices), its
/// hanging edges and faces are registered with `Mesh::set_hanging`, and the DoF maps
/// constrain the hanging DoFs (`assembly::hanging_constraints`). See
/// docs/theory/mesh.md#local-refinement and ADR-0006.

#include <array>
#include <map>
#include <span>
#include <utility>
#include <vector>

#include "hpfem/core/types.hpp"
#include "hpfem/mesh/detail/red_refinement.hpp"
#include "hpfem/mesh/mesh.hpp"

namespace hpfem::mesh {

/// How the leaf cells of a refinement step relate to those before it: cell i of the new
/// mesh is cell `parent[i]` of the old mesh itself (`path[i]` empty) or a descendant reached
/// through the child numbers `path[i]` (red patterns, one entry per level; the closure may
/// split a cell more than once in one call). `old_reference` maps reference coordinates of
/// cell i to the old cell.
struct RefinementStep {
  std::vector<Index> parent;
  std::vector<std::vector<LocalIndex>> path;
  Index num_old_cells = 0;
  [[nodiscard]] Index num_cells() const noexcept { return static_cast<Index>(parent.size()); }
  /// True if cell i was created in this step.
  [[nodiscard]] bool refined(Index i) const { return !path[as_size(i)].empty(); }
  /// Number of levels between cell i and its old cell.
  [[nodiscard]] int levels(Index i) const { return static_cast<int>(path[as_size(i)].size()); }
  /// Reference coordinates in the old cell of reference point ξ of cell i.
  template <int Dim>
  [[nodiscard]] Point<Dim> old_reference(Index i, const Point<Dim>& xi) const {
    Point<Dim> x = xi;
    const auto& chain = path[as_size(i)];
    for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
      x = detail::parent_reference<Dim>(*it, x);
    }
    return x;
  }
};

/// Refinement hierarchy over a root mesh. `mesh()` is the current leaf mesh; `refine()`
/// red-refines the given leaf cells plus whatever the one-irregular rule requires: before a
/// cell is split, every coarser leaf cell sharing a vertex with it and every leaf cell owning
/// the parent of one of its facets is split first (2:1 balance by vertices and by facets;
/// the latter matters in 3D, where the inner child face of a face shares no vertex with the
/// coarse neighbour). The closure may split a cell created in the same call again; the
/// returned step records the chain of child numbers per new cell. Children inherit the cell tag;
/// facet tags of the root transfer to the leaf facets lying on tagged root facets; curved roots
/// place new vertices and edge nodes with the root cell maps.
template <int Dim>
class AdaptiveMesh {
 public:
  using CellVertices = typename Mesh<Dim>::CellVertices;
  static constexpr Index kChildren = detail::RedRefinement<Dim>::kChildren;

  explicit AdaptiveMesh(Mesh<Dim> root);

  [[nodiscard]] const Mesh<Dim>& root() const noexcept { return root_; }
  /// Current leaf mesh (hanging entities registered).
  [[nodiscard]] const Mesh<Dim>& mesh() const noexcept { return leaf_mesh_; }
  /// Refinement level of leaf cell c (0 for root cells).
  [[nodiscard]] int level(Index c) const { return cells_[as_size(leaves_[as_size(c)])].level; }
  [[nodiscard]] int max_level() const noexcept { return max_level_; }
  /// Root cell containing leaf cell c, and the reference coordinates in that root cell of
  /// the reference point ξ of c.
  [[nodiscard]] Index root_cell(Index c) const { return cells_[as_size(leaves_[as_size(c)])].root; }
  [[nodiscard]] Point<Dim> root_reference(Index c, const Point<Dim>& xi) const;

  /// Refines the leaf cells `marked` (ids of `mesh()`), closes the one-irregular rule and
  /// rebuilds the leaf mesh. @throws InvalidArgument for an id out of range.
  RefinementStep refine(std::span<const Index> marked);
  /// Refines every leaf cell (the result equals `refine_uniform` of the leaf mesh).
  RefinementStep refine_all();

 private:
  struct TreeCell {
    CellVertices vertices;
    Index parent = kInvalidIndex;
    Index first_child = kInvalidIndex;  ///< children are consecutive
    int level = 0;
    Tag tag = kNoTag;
    Index root = kInvalidIndex;
    std::array<Point<Dim>, static_cast<std::size_t>(Dim + 1)> root_xi;  ///< vertices in root
    [[nodiscard]] bool is_leaf() const noexcept { return first_child == kInvalidIndex; }
  };

  using FacetKey = std::array<Index, static_cast<std::size_t>(Dim)>;  ///< sorted vertices

  void refine_cell(Index tree_cell);
  void split(Index tree_cell);
  [[nodiscard]] Index midpoint(Index a, Index b, Index root, const Point<Dim>& xi);
  /// Sorted vertices of local facet k of a tree cell.
  [[nodiscard]] FacetKey facet_key(const TreeCell& cell, std::size_t k) const;
  /// The parent facet a facet is a child of (through the midpoint vertices), if any.
  [[nodiscard]] bool parent_facet(const FacetKey& facet, FacetKey& parent) const;
  void rebuild_leaf_mesh();

  Mesh<Dim> root_;
  Mesh<Dim> leaf_mesh_;
  std::vector<Point<Dim>> vertices_;
  std::vector<TreeCell> cells_;
  std::map<std::pair<Index, Index>, Index> midpoints_;   ///< (a, b) a < b → vertex
  std::vector<std::pair<Index, Index>> parent_edge_;     ///< vertex → parent edge (midpoints)
  std::vector<Index> leaves_;                            ///< tree ids, ascending
  std::vector<std::vector<Index>> vertex_leaves_;        ///< vertex → leaf tree ids
  std::map<FacetKey, std::vector<Index>> facet_leaves_;  ///< facet → leaf tree ids
  int max_level_ = 0;
};

extern template class AdaptiveMesh<2>;
extern template class AdaptiveMesh<3>;

}  // namespace hpfem::mesh

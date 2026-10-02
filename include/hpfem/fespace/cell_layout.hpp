#pragma once
/// @file cell_layout.hpp
/// Per-cell description shared by the hierarchical bases (`H1Basis`, `NedelecBasis`):
/// polynomial orders of the cell and of its edges / faces, and how the local entity
/// orientation relates to the global one (ADR-0003).

#include <array>
#include <cstddef>
#include <cstdint>

#include "hpfem/mesh/simplex_topology.hpp"

namespace hpfem::fespace {

/// Orders per entity (minimum rule applied by the DoF maps) and orientation codes straight
/// from `mesh::Mesh::cell_edge_flipped` / `cell_face_permutations`.
template <int Dim>
struct CellLayout {
  static constexpr std::size_t kNumEdges =
      static_cast<std::size_t>(mesh::SimplexTopology<Dim>::kNumEdges);
  static constexpr std::size_t kNumFaces = Dim == 3 ? 4 : 0;

  int cell_order = 1;                               ///< p ≥ 1 for the interior functions
  std::array<int, kNumEdges> edge_orders{};         ///< p_e ≥ 1
  std::array<int, 4> face_orders{};                 ///< p_f ≥ 1, 3D only
  std::array<bool, kNumEdges> edge_flipped{};       ///< local edge runs against global orientation
  std::array<std::uint8_t, 4> face_permutations{};  ///< codes into `mesh::kFacePermutations`

  /// Uniform order p on all entities, identity orientation.
  [[nodiscard]] static CellLayout uniform(int p) {
    CellLayout layout;
    layout.cell_order = p;
    layout.edge_orders.fill(p);
    layout.face_orders.fill(p);
    return layout;
  }
};

}  // namespace hpfem::fespace

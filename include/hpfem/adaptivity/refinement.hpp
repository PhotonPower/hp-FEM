#pragma once
/// @file refinement.hpp
/// The REFINE step of the adaptive loop: raising polynomial orders of marked cells
/// (p-refinement) and combining it with the local h-refinement of `mesh::AdaptiveMesh`.
/// Entity orders follow the minimum rule of the DoF maps, so a raised cell order enlarges
/// the space in the cell interior at once and on an edge / face only when all its cells
/// are raised; the spaces stay nested, hence `assembly::prolongate` transfers solutions
/// exactly. See docs/theory/hp-adaptivity.md#loop.

#include <span>
#include <vector>

#include "hpfem/core/types.hpp"
#include "hpfem/mesh/adaptive_mesh.hpp"

namespace hpfem::adaptivity {

/// Orders after p-refinement: marked cells are raised by `increment` (capped at
/// `max_order` if positive), the others keep their order.
/// @throws InvalidArgument for a cell out of range or a non-positive increment.
[[nodiscard]] std::vector<int> p_refine(std::span<const int> orders, std::span<const Index> marked,
                                        int increment = 1, int max_order = 0);

/// Result of an hp-refinement step: the mesh relation and the orders of the new cells.
struct HpStep {
  mesh::RefinementStep step;
  std::vector<int> orders;
};

/// Refines `h_marked` cells in the hierarchy (children inherit the parent's order, cells
/// refined for the one-irregular closure as well) and raises the order of `p_marked`
/// cells (a cell in both lists is h-refined and its children raised). Both lists refer to
/// the leaf mesh before the call; `orders` has one entry per leaf cell.
/// @throws InvalidArgument for a mismatch of `orders` and the mesh or a cell out of range.
template <int Dim>
[[nodiscard]] HpStep hp_refine(mesh::AdaptiveMesh<Dim>& mesh, std::span<const int> orders,
                               std::span<const Index> h_marked, std::span<const Index> p_marked,
                               int increment = 1, int max_order = 0);

/// The step relating a mesh to itself (pure p-refinement), for `assembly::prolongate`.
[[nodiscard]] mesh::RefinementStep identity_step(Index num_cells);

extern template HpStep hp_refine<2>(mesh::AdaptiveMesh<2>&, std::span<const int>,
                                    std::span<const Index>, std::span<const Index>, int, int);
extern template HpStep hp_refine<3>(mesh::AdaptiveMesh<3>&, std::span<const int>,
                                    std::span<const Index>, std::span<const Index>, int, int);

}  // namespace hpfem::adaptivity

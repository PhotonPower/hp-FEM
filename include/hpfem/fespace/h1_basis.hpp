#pragma once
/// @file h1_basis.hpp
/// Hierarchical H1-conforming basis of arbitrary order on the reference simplex, with the
/// polynomial order assignable per edge, face and cell. Formulas and ordering:
/// docs/theory/h1-basis.md. Entity functions are evaluated in the *global* orientation of
/// their entity (ADR-0003), so neighbouring cells see identical functions on shared
/// entities and the DoF map needs no sign or permutation fix-ups.

#include <array>
#include <cstddef>
#include <span>

#include "hpfem/core/types.hpp"
#include "hpfem/fespace/cell_layout.hpp"
#include "hpfem/mesh/simplex_topology.hpp"

namespace hpfem::fespace {

/// Per-cell orders and orientation codes (shared with `NedelecBasis`).
template <int Dim>
using H1Layout = CellLayout<Dim>;

/// Number of basis functions associated with an entity of the given order.
[[nodiscard]] constexpr Index h1_edge_functions(int p) noexcept {
  return p >= 2 ? p - 1 : 0;
}
[[nodiscard]] constexpr Index h1_face_functions(int p) noexcept {
  return p >= 3 ? static_cast<Index>(p - 1) * (p - 2) / 2 : 0;
}
template <int Dim>
[[nodiscard]] constexpr Index h1_cell_functions(int p) noexcept {
  if constexpr (Dim == 2) {
    return h1_face_functions(p);
  } else {
    return p >= 4 ? static_cast<Index>(p - 1) * (p - 2) * (p - 3) / 6 : 0;
  }
}

/// Hierarchical H1 basis of one cell. Function order: the Dim + 1 vertex functions, then
/// the functions of edge 0, 1, …, then (3D) of face 0, 1, …, then the interior functions;
/// within an entity by ascending polynomial degree (`docs/theory/h1-basis.md`).
template <int Dim>
class H1Basis {
 public:
  using Topology = mesh::SimplexTopology<Dim>;
  static constexpr std::size_t kNumVertices = static_cast<std::size_t>(Dim + 1);

  /// @throws InvalidArgument if any order is < 1 or an entity order exceeds the cell order.
  explicit H1Basis(const CellLayout<Dim>& layout);

  [[nodiscard]] const CellLayout<Dim>& layout() const noexcept { return layout_; }
  /// Total number of shape functions.
  [[nodiscard]] Index size() const noexcept { return size_; }
  /// Maximal polynomial degree occurring.
  [[nodiscard]] int max_order() const noexcept { return layout_.cell_order; }
  /// First function index of the functions of edge / face k and of the interior.
  [[nodiscard]] Index edge_offset(std::size_t k) const noexcept { return edge_offsets_[k]; }
  [[nodiscard]] Index face_offset(std::size_t k) const noexcept { return face_offsets_[k]; }
  [[nodiscard]] Index cell_offset() const noexcept { return cell_offset_; }

  /// Values and reference gradients of all functions at ξ; both spans need `size()`
  /// entries (`gradients` may be empty to skip them).
  void evaluate(const Point<Dim>& xi, std::span<Real> values,
                std::span<Point<Dim>> gradients) const;

 private:
  CellLayout<Dim> layout_;
  Index size_ = 0;
  std::array<Index, CellLayout<Dim>::kNumEdges> edge_offsets_{};
  std::array<Index, 4> face_offsets_{};
  Index cell_offset_ = 0;
};

extern template class H1Basis<2>;
extern template class H1Basis<3>;

}  // namespace hpfem::fespace

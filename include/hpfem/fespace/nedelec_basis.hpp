#pragma once
/// @file nedelec_basis.hpp
/// Hierarchical H(curl)-conforming Nédélec basis of the *first kind* on the reference
/// simplex, arbitrary order, orders assignable per edge, face and cell, gradient and
/// non-gradient functions explicitly separated. Formulas, ordering and the de Rham
/// structure: docs/theory/nedelec.md#hierarchical-basis. Entity functions are evaluated
/// in the global orientation of their entity (ADR-0003), exactly as `H1Basis`.
///
/// Values and curls are given on the reference element; the covariant Piola map of
/// docs/theory/nedelec.md#mapping transfers them to a physical cell.

#include <array>
#include <cstddef>
#include <span>

#include <Eigen/Core>

#include "hpfem/core/types.hpp"
#include "hpfem/fespace/cell_layout.hpp"
#include "hpfem/mesh/simplex_topology.hpp"

namespace hpfem::fespace {

/// Curl of a reference field: a scalar (stored as a 1-vector) in 2D, a 3-vector in 3D.
template <int Dim>
using CurlVector = Eigen::Matrix<Real, Dim == 2 ? 1 : 3, 1>;

/// Number of functions of the order-p space (Nédélec I, ND_p, lowest order p = 1) per entity.
[[nodiscard]] constexpr Index nedelec_edge_functions(int p) noexcept {
  return p >= 1 ? p : 0;
}
[[nodiscard]] constexpr Index nedelec_face_functions(int p) noexcept {
  return p >= 2 ? static_cast<Index>(p) * (p - 1) : 0;
}
template <int Dim>
[[nodiscard]] constexpr Index nedelec_cell_functions(int p) noexcept {
  if constexpr (Dim == 2) {
    return nedelec_face_functions(p);
  } else {
    return p >= 3 ? static_cast<Index>(p) * (p - 1) * (p - 2) / 2 : 0;
  }
}
/// dim ND_p of one cell with uniform order: p(p+2) (triangle), p(p+2)(p+3)/2 (tetrahedron).
template <int Dim>
[[nodiscard]] constexpr Index nedelec_dimension(int p) noexcept {
  return Dim == 2 ? static_cast<Index>(p) * (p + 2) : static_cast<Index>(p) * (p + 2) * (p + 3) / 2;
}

/// Nédélec (first kind) basis of one cell. Function order: edge 0, 1, … (Whitney function,
/// then the gradients ∇L_i^S for i = 2..p_e), then (3D) face 0, 1, …, then the interior
/// functions; within faces / interiors by type (gradient, non-gradient twin, Whitney-type)
/// and ascending degree, see docs/theory/nedelec.md.
template <int Dim>
class NedelecBasis {
 public:
  using Topology = mesh::SimplexTopology<Dim>;
  using Curl = CurlVector<Dim>;

  /// @throws InvalidArgument if any order is < 1 or an entity order exceeds the cell order.
  explicit NedelecBasis(const CellLayout<Dim>& layout);

  [[nodiscard]] const CellLayout<Dim>& layout() const noexcept { return layout_; }
  [[nodiscard]] Index size() const noexcept { return size_; }
  [[nodiscard]] int max_order() const noexcept { return layout_.cell_order; }
  [[nodiscard]] Index edge_offset(std::size_t k) const noexcept { return edge_offsets_[k]; }
  [[nodiscard]] Index face_offset(std::size_t k) const noexcept { return face_offsets_[k]; }
  [[nodiscard]] Index cell_offset() const noexcept { return cell_offset_; }

  /// Reference values and reference curls of all functions at ξ; `values` needs `size()`
  /// entries, `curls` either `size()` entries or none.
  void evaluate(const Point<Dim>& xi, std::span<Point<Dim>> values, std::span<Curl> curls) const;

 private:
  CellLayout<Dim> layout_;
  Index size_ = 0;
  std::array<Index, CellLayout<Dim>::kNumEdges> edge_offsets_{};
  std::array<Index, 4> face_offsets_{};
  Index cell_offset_ = 0;
};

extern template class NedelecBasis<2>;
extern template class NedelecBasis<3>;

}  // namespace hpfem::fespace

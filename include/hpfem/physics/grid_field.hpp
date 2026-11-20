#pragma once
/// @file grid_field.hpp
/// A scalar volume field on a regular grid, interpolated linearly per axis (bilinear in 2D,
/// trilinear in 3D): the input path of volume data ports such as the shrinkage or the degree of
/// polymerisation of a printed part (ADR-0016 §5, `hpfem.dataports.RegularGrid`). An axis of
/// length 1 means invariance along it. Lengths in metres.
#include <array>
#include <cstddef>
#include <vector>

#include "hpfem/core/types.hpp"

namespace hpfem::physics {

template <int Dim>
class GridField {
 public:
  /// `axes[k]` strictly increasing (or a single value), `values` in C order over the axes
  /// (the last axis fastest: index (i0 · n1 + i1) · n2 + i2), as a NumPy array `values[i0, i1,
  /// i2]` is laid out.
  /// @throws InvalidArgument for an empty or non-increasing axis, non-finite data, or a value
  ///         count that does not match the axes.
  GridField(std::array<std::vector<Real>, static_cast<std::size_t>(Dim)> axes,
            std::vector<Real> values);
  /// The interpolated value at x.
  /// @throws InvalidArgument if x lies outside the grid (beyond 1e-9 of an axis extent) along an
  ///         axis with more than one value.
  [[nodiscard]] Real operator()(const Point<Dim>& x) const;
  [[nodiscard]] const std::array<std::vector<Real>, static_cast<std::size_t>(Dim)>& axes()
      const noexcept {
    return axes_;
  }

 private:
  std::array<std::vector<Real>, static_cast<std::size_t>(Dim)> axes_;
  std::vector<Real> values_;
};

extern template class GridField<2>;
extern template class GridField<3>;

}  // namespace hpfem::physics

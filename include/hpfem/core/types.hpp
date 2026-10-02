#pragma once
/// @file types.hpp
/// Fundamental scalar and index types used throughout hpfem.
/// Convention (see CLAUDE.md §6): time dependence exp(-iωt), SI units.

#include <complex>
#include <cstddef>
#include <cstdint>

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace hpfem {

using Real = double;
using Complex = std::complex<double>;
using Index = std::int64_t;       ///< global indices (DoFs, entities)
using LocalIndex = std::int32_t;  ///< element-local indices

using Vector = Eigen::Matrix<Complex, Eigen::Dynamic, 1>;
using RealVector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using Matrix = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>;
using RealMatrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
using SparseMatrix = Eigen::SparseMatrix<Complex, Eigen::RowMajor, Index>;

template <int Dim>
using Point = Eigen::Matrix<Real, Dim, 1>;

inline constexpr Complex kI{0.0, 1.0};  ///< imaginary unit

/// Converts a non-negative global or local index to a container index.
[[nodiscard]] constexpr std::size_t as_size(Index i) noexcept {
  return static_cast<std::size_t>(i);
}

}  // namespace hpfem

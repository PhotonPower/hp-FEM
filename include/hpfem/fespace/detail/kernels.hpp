#pragma once
/// @file kernels.hpp
/// Shared building blocks of the hierarchical H1 and H(curl) bases: the edge kernel
/// @f$ L_i^S(\lambda_b - \lambda_a, \lambda_a + \lambda_b) @f$ and the bubble factor
/// @f$ \lambda\,P_j(2\lambda - 1) @f$, both with reference gradients assembled from the
/// barycentric gradients. Internal; see docs/theory/h1-basis.md. The kernels keep their
/// values in fixed-capacity arrays, so a basis evaluation allocates nothing (point
/// evaluation and quadrature loops call them once per point).
#include <array>
#include <cstddef>
#include <span>

#include "hpfem/core/error.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/fespace/polynomials.hpp"

namespace hpfem::fespace::detail {

/// Largest polynomial order the kernels hold (orders 0..kMaxKernelOrder).
inline constexpr int kMaxKernelOrder = 31;

/// Fixed-capacity array of the kernel values: `size()` entries, indexable, convertible to a
/// span of the used entries. @throws InvalidArgument beyond the capacity.
template <class T>
class KernelArray {
 public:
  explicit KernelArray(std::size_t n) : size_(n) {
    if (n > data_.size()) {
      throw InvalidArgument("basis kernels: polynomial order beyond the supported maximum");
    }
  }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] T& operator[](std::size_t i) noexcept { return data_[i]; }
  [[nodiscard]] const T& operator[](std::size_t i) const noexcept { return data_[i]; }
  [[nodiscard]] std::span<T> span() noexcept { return {data_.data(), size_}; }
  [[nodiscard]] std::span<const T> span() const noexcept { return {data_.data(), size_}; }

 private:
  std::array<T, static_cast<std::size_t>(kMaxKernelOrder) + 1> data_;
  std::size_t size_;
};

/// Scaled integrated Legendre data of the vertex pair (a, b): values and gradients of
/// @f$ L_i^S(\lambda_b - \lambda_a, \lambda_a + \lambda_b) @f$ for i = 0..order (entries 0
/// unused).
template <int Dim>
struct EdgeKernel {
  KernelArray<Real> value;
  KernelArray<Point<Dim>> gradient;

  EdgeKernel(int order, const std::array<Real, static_cast<std::size_t>(Dim + 1)>& lambda,
             const std::array<Point<Dim>, static_cast<std::size_t>(Dim + 1)>& dlambda,
             std::size_t a, std::size_t b)
      : value(static_cast<std::size_t>(order) + 1), gradient(value.size()) {
    KernelArray<Real> dx(value.size());
    KernelArray<Real> dt(value.size());
    scaled_integrated_legendre(order, lambda[b] - lambda[a], lambda[a] + lambda[b], value.span(),
                               dx.span(), dt.span());
    const Point<Dim> grad_x = dlambda[b] - dlambda[a];
    const Point<Dim> grad_t = dlambda[a] + dlambda[b];
    for (std::size_t i = 0; i < value.size(); ++i) {
      gradient[i] = dx[i] * grad_x + dt[i] * grad_t;
    }
  }
};

/// The factor @f$ g_j(\lambda) = \lambda\,P_j(2\lambda - 1) @f$, j = 0..n, with gradients.
template <int Dim>
struct BubbleKernel {
  KernelArray<Real> value;
  KernelArray<Point<Dim>> gradient;

  BubbleKernel(int n, Real lambda, const Point<Dim>& dlambda)
      : value(static_cast<std::size_t>(n) + 1), gradient(value.size()) {
    KernelArray<Real> p(value.size());
    KernelArray<Real> dp(value.size());
    legendre(n, 2.0 * lambda - 1.0, p.span(), dp.span());
    for (std::size_t j = 0; j < value.size(); ++j) {
      value[j] = lambda * p[j];
      gradient[j] = (p[j] + 2.0 * lambda * dp[j]) * dlambda;
    }
  }
};

/// Legendre factors @f$ P_j(2\lambda - 1) @f$, j = 0..n, with gradients (no λ prefactor).
template <int Dim>
struct LegendreKernel {
  KernelArray<Real> value;
  KernelArray<Point<Dim>> gradient;

  LegendreKernel(int n, Real lambda, const Point<Dim>& dlambda)
      : value(static_cast<std::size_t>(n) + 1), gradient(value.size()) {
    KernelArray<Real> dp(value.size());
    legendre(n, 2.0 * lambda - 1.0, value.span(), dp.span());
    for (std::size_t j = 0; j < value.size(); ++j) gradient[j] = 2.0 * dp[j] * dlambda;
  }
};

}  // namespace hpfem::fespace::detail

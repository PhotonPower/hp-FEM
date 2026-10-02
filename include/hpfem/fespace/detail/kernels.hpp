#pragma once
/// @file kernels.hpp
/// Shared building blocks of the hierarchical H1 and H(curl) bases: the edge kernel
/// @f$ L_i^S(\lambda_b - \lambda_a, \lambda_a + \lambda_b) @f$ and the bubble factor
/// @f$ \lambda\,P_j(2\lambda - 1) @f$, both with reference gradients assembled from the
/// barycentric gradients. Internal; see docs/theory/h1-basis.md.

#include <array>
#include <cstddef>
#include <vector>

#include "hpfem/core/types.hpp"
#include "hpfem/fespace/polynomials.hpp"

namespace hpfem::fespace::detail {

/// Scaled integrated Legendre data of the vertex pair (a, b): values and gradients of
/// @f$ L_i^S(\lambda_b - \lambda_a, \lambda_a + \lambda_b) @f$ for i = 0..order (entries 0
/// unused).
template <int Dim>
struct EdgeKernel {
  std::vector<Real> value;
  std::vector<Point<Dim>> gradient;

  EdgeKernel(int order, const std::array<Real, static_cast<std::size_t>(Dim + 1)>& lambda,
             const std::array<Point<Dim>, static_cast<std::size_t>(Dim + 1)>& dlambda,
             std::size_t a, std::size_t b)
      : value(static_cast<std::size_t>(order) + 1), gradient(value.size()) {
    std::vector<Real> dx(value.size());
    std::vector<Real> dt(value.size());
    scaled_integrated_legendre(order, lambda[b] - lambda[a], lambda[a] + lambda[b], value, dx, dt);
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
  std::vector<Real> value;
  std::vector<Point<Dim>> gradient;

  BubbleKernel(int n, Real lambda, const Point<Dim>& dlambda)
      : value(static_cast<std::size_t>(n) + 1), gradient(value.size()) {
    std::vector<Real> p(value.size());
    std::vector<Real> dp(value.size());
    legendre(n, 2.0 * lambda - 1.0, p, dp);
    for (std::size_t j = 0; j < value.size(); ++j) {
      value[j] = lambda * p[j];
      gradient[j] = (p[j] + 2.0 * lambda * dp[j]) * dlambda;
    }
  }
};

/// Legendre factors @f$ P_j(2\lambda - 1) @f$, j = 0..n, with gradients (no λ prefactor).
template <int Dim>
struct LegendreKernel {
  std::vector<Real> value;
  std::vector<Point<Dim>> gradient;

  LegendreKernel(int n, Real lambda, const Point<Dim>& dlambda)
      : value(static_cast<std::size_t>(n) + 1), gradient(value.size()) {
    std::vector<Real> dp(value.size());
    legendre(n, 2.0 * lambda - 1.0, value, dp);
    for (std::size_t j = 0; j < value.size(); ++j) gradient[j] = 2.0 * dp[j] * dlambda;
  }
};

}  // namespace hpfem::fespace::detail

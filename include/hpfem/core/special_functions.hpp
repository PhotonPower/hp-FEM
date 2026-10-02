#pragma once
/// @file special_functions.hpp
/// Cylindrical Bessel and Hankel functions of real argument, as needed by the analytic 2D
/// solutions (line sources, Mie series for the cylinder). Thin wrappers around the C++17
/// special functions of the standard library (available in libstdc++ and the MSVC STL,
/// not in libc++). Convention exp(−iωt): @f$ H_n^{(1)} = J_n + iY_n @f$ is outgoing.

#include <cmath>

#include "hpfem/core/types.hpp"

namespace hpfem {

/// Bessel function of the first kind @f$ J_n(x) @f$, any integer order (@f$ J_{-n} = (-1)^n J_n
/// @f$).
[[nodiscard]] inline Real bessel_j(int order, Real x) {
  const Real value = std::cyl_bessel_j(static_cast<Real>(order < 0 ? -order : order), x);
  return (order < 0 && (order % 2) != 0) ? -value : value;
}

/// Bessel function of the second kind @f$ Y_n(x) @f$, x > 0, any integer order.
[[nodiscard]] inline Real bessel_y(int order, Real x) {
  const Real value = std::cyl_neumann(static_cast<Real>(order < 0 ? -order : order), x);
  return (order < 0 && (order % 2) != 0) ? -value : value;
}

/// Hankel function of the first kind @f$ H_n^{(1)}(x) = J_n(x) + iY_n(x) @f$, x > 0.
[[nodiscard]] inline Complex hankel1(int order, Real x) {
  return {bessel_j(order, x), bessel_y(order, x)};
}

/// Derivative @f$ H_n^{(1)\prime}(x) = H_{n-1}^{(1)}(x) - \frac{n}{x} H_n^{(1)}(x) @f$.
[[nodiscard]] inline Complex hankel1_derivative(int order, Real x) {
  return hankel1(order - 1, x) - static_cast<Real>(order) / x * hankel1(order, x);
}

}  // namespace hpfem

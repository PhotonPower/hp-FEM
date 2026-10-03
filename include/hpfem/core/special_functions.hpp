#pragma once
/// @file special_functions.hpp
/// Cylindrical Bessel and Hankel functions of real argument (line sources, Mie series for the
/// cylinder) and spherical Bessel functions of real and complex argument (Mie series for the
/// sphere). Thin wrappers around the C++17
/// special functions of the standard library (available in libstdc++ and the MSVC STL,
/// not in libc++). Convention exp(−iωt): @f$ H_n^{(1)} = J_n + iY_n @f$ is outgoing.

#include <cmath>
#include <complex>
#include <vector>

#include "hpfem/core/error.hpp"
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

/// Derivative @f$ J_n'(x) = J_{n-1}(x) - \frac{n}{x} J_n(x) @f$.
[[nodiscard]] inline Real bessel_j_derivative(int order, Real x) {
  return bessel_j(order - 1, x) - static_cast<Real>(order) / x * bessel_j(order, x);
}

/// Hankel function of the first kind @f$ H_n^{(1)}(x) = J_n(x) + iY_n(x) @f$, x > 0.
[[nodiscard]] inline Complex hankel1(int order, Real x) {
  return {bessel_j(order, x), bessel_y(order, x)};
}

/// Derivative @f$ H_n^{(1)\prime}(x) = H_{n-1}^{(1)}(x) - \frac{n}{x} H_n^{(1)}(x) @f$.
[[nodiscard]] inline Complex hankel1_derivative(int order, Real x) {
  return hankel1(order - 1, x) - static_cast<Real>(order) / x * hankel1(order, x);
}

// --- spherical Bessel functions (Mie series of the sphere) --------------------------------------

/// Spherical Bessel function of the first kind @f$ j_n(x) @f$, n ≥ 0, real x ≥ 0.
[[nodiscard]] inline Real spherical_bessel_j(int order, Real x) {
  return std::sph_bessel(static_cast<unsigned>(order), x);
}

/// Spherical Bessel function of the second kind @f$ y_n(x) @f$, n ≥ 0, x > 0.
[[nodiscard]] inline Real spherical_bessel_y(int order, Real x) {
  return std::sph_neumann(static_cast<unsigned>(order), x);
}

/// Spherical Hankel function of the first kind @f$ h_n^{(1)}(x) = j_n(x) + i y_n(x) @f$
/// (outgoing for exp(−iωt)), x > 0.
[[nodiscard]] inline Complex spherical_hankel1(int order, Real x) {
  return {spherical_bessel_j(order, x), spherical_bessel_y(order, x)};
}

/// @f$ j_0(z), \dots, j_N(z) @f$ for complex z (e.g. inside an absorbing sphere) by downward
/// recurrence @f$ j_{n-1} = \frac{2n+1}{z} j_n - j_{n+1} @f$ started well above N and
/// normalised with @f$ j_0 = \sin z / z @f$ (stable for all orders, unlike the upward
/// recurrence). For |z| < 1e-300 the limit @f$ j_n(0) = \delta_{n0} @f$ is returned.
/// @throws InvalidArgument for max_order < 0.
[[nodiscard]] inline std::vector<Complex> spherical_bessel_j(int max_order, Complex z) {
  if (max_order < 0) throw InvalidArgument("spherical_bessel_j: max_order must be non-negative");
  std::vector<Complex> j(static_cast<std::size_t>(max_order) + 1, Complex{0.0, 0.0});
  if (std::abs(z) < 1e-300) {
    j[0] = Complex{1.0, 0.0};
    return j;
  }
  // start the recurrence where j_n is negligible relative to j_N; values grow while descending
  const int start = max_order + 20 + static_cast<int>(std::ceil(std::abs(z)));
  Complex next{0.0, 0.0};        // f_{n+1}
  Complex current{1e-300, 0.0};  // f_n, arbitrary scale
  for (int n = start; n > 0; --n) {
    const Complex previous = static_cast<Real>(2 * n + 1) / z * current - next;  // f_{n-1}
    next = current;
    current = previous;
    if (n - 1 <= max_order) j[static_cast<std::size_t>(n - 1)] = current;
    if (std::abs(current) > 1e200) {  // rescale against overflow
      constexpr Real kScale = 1e-200;
      current *= kScale;
      next *= kScale;
      for (auto& v : j) v *= kScale;
    }
  }
  const Complex scale = std::sin(z) / z / j[0];
  for (auto& v : j) v *= scale;
  return j;
}

}  // namespace hpfem

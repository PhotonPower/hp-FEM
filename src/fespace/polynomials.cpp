#include "hpfem/fespace/polynomials.hpp"

#include <cstddef>

#include "hpfem/core/error.hpp"

namespace hpfem::fespace {

void legendre(int n, Real x, std::span<Real> values, std::span<Real> derivatives) {
  HPFEM_ASSERT(n >= 0, "legendre: n must be non-negative");
  HPFEM_ASSERT(values.size() == static_cast<std::size_t>(n) + 1 &&
                   derivatives.size() == static_cast<std::size_t>(n) + 1,
               "legendre: output spans must have size n + 1");
  values[0] = 1.0;
  derivatives[0] = 0.0;
  if (n == 0) return;
  values[1] = x;
  derivatives[1] = 1.0;
  for (int i = 1; i < n; ++i) {
    const auto k = static_cast<std::size_t>(i);
    const Real a = (2.0 * i + 1.0) / (i + 1.0);
    const Real b = static_cast<Real>(i) / (i + 1.0);
    // (i+1) P_{i+1} = (2i+1) x P_i - i P_{i-1}
    values[k + 1] = a * x * values[k] - b * values[k - 1];
    derivatives[k + 1] = a * (values[k] + x * derivatives[k]) - b * derivatives[k - 1];
  }
}

void scaled_integrated_legendre(int n, Real x, Real t, std::span<Real> values, std::span<Real> dx,
                                std::span<Real> dt) {
  HPFEM_ASSERT(n >= 1, "scaled_integrated_legendre: n must be at least 1");
  HPFEM_ASSERT(values.size() == static_cast<std::size_t>(n) + 1 &&
                   dx.size() == static_cast<std::size_t>(n) + 1 &&
                   dt.size() == static_cast<std::size_t>(n) + 1,
               "scaled_integrated_legendre: output spans must have size n + 1");
  values[0] = dx[0] = dt[0] = 0.0;
  values[1] = x;
  dx[1] = 1.0;
  dt[1] = 0.0;
  if (n == 1) return;
  values[2] = 0.5 * (x * x - t * t);
  dx[2] = x;
  dt[2] = -t;
  for (int i = 2; i < n; ++i) {
    const auto k = static_cast<std::size_t>(i);
    const Real a = (2.0 * i - 1.0) / (i + 1.0);
    const Real b = (i - 2.0) / (i + 1.0);
    // (i+1) L_{i+1}^S = (2i-1) x L_i^S - (i-2) t^2 L_{i-1}^S
    values[k + 1] = a * x * values[k] - b * t * t * values[k - 1];
    dx[k + 1] = a * (values[k] + x * dx[k]) - b * t * t * dx[k - 1];
    dt[k + 1] = a * x * dt[k] - b * (2.0 * t * values[k - 1] + t * t * dt[k - 1]);
  }
}

}  // namespace hpfem::fespace

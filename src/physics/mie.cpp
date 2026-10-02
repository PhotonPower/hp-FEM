#include "hpfem/physics/mie.hpp"

#include <cmath>

#include <fmt/format.h>

#include "hpfem/core/error.hpp"
#include "hpfem/core/special_functions.hpp"

namespace hpfem::physics {

std::vector<Complex> mie_cylinder_coefficients(Real k, Real radius, Real refractive_index,
                                               int max_order) {
  if (!(k > 0) || !(radius > 0) || !(refractive_index > 0) || max_order < 0) {
    throw InvalidArgument(
        fmt::format("mie_cylinder_coefficients: k = {}, radius = {}, n = {} must be positive and "
                    "max_order = {} non-negative",
                    k, radius, refractive_index, max_order));
  }
  const Real x = k * radius;
  const Real y = refractive_index * x;
  std::vector<Complex> c;
  c.reserve(static_cast<std::size_t>(max_order) + 1);
  Complex i_power{1.0, 0.0};
  for (int n = 0; n <= max_order; ++n) {
    const Real jx = bessel_j(n, x);
    const Real djx = bessel_j_derivative(n, x);
    const Real jy = bessel_j(n, y);
    const Real djy = bessel_j_derivative(n, y);
    const Complex hx = hankel1(n, x);
    const Complex dhx = hankel1_derivative(n, x);
    const Real numerator = refractive_index * djx * jy - jx * djy;
    const Complex denominator = hx * djy - refractive_index * dhx * jy;
    c.push_back(i_power * numerator / denominator);
    i_power *= kI;
  }
  return c;
}

Real mie_cylinder_scattering_width(Real k, Real radius, Real refractive_index, int max_order) {
  if (max_order < 0) {
    // Wiscombe-type cut-off: a few orders beyond the size parameter in the cylinder
    const Real y = refractive_index * k * radius;
    max_order = static_cast<int>(std::ceil(y + 4.0 * std::cbrt(y) + 10.0));
  }
  const auto c = mie_cylinder_coefficients(k, radius, refractive_index, max_order);
  Real sum = std::norm(c[0]);
  for (std::size_t n = 1; n < c.size(); ++n) sum += 2.0 * std::norm(c[n]);
  return 4.0 / k * sum;
}

}  // namespace hpfem::physics

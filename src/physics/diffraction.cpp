#include "hpfem/physics/diffraction.hpp"

#include <cmath>
#include <numbers>

#include <fmt/format.h>

#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/core/error.hpp"

namespace hpfem::physics {

std::vector<assembly::ComplexVector<2>> fourier_coefficients(
    const std::function<assembly::ComplexVector<2>(const Point<2>&)>& field, Real x0, Real y0,
    Real period, Real ky0, int max_order, int num_points) {
  if (!(period > 0) || max_order < 0 || num_points < 1) {
    throw InvalidArgument(fmt::format(
        "fourier_coefficients: period = {} must be positive, max_order = {} and num_points = {} "
        "non-negative / positive",
        period, max_order, num_points));
  }
  // composite Gauss-Legendre: num_points points spread over the period in blocks of up to 8
  const int blocks = (num_points + 7) / 8;
  const int per_block = (num_points + blocks - 1) / blocks;
  const auto rule = assembly::gauss_legendre(per_block);
  std::vector<assembly::ComplexVector<2>> coefficients(static_cast<std::size_t>(2 * max_order + 1),
                                                       assembly::ComplexVector<2>::Zero());
  const Real block_length = period / blocks;
  for (int b = 0; b < blocks; ++b) {
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const Real y = y0 + (b + rule.points[q](0)) * block_length;
      const Real w = rule.weights[q] * block_length / period;
      const assembly::ComplexVector<2> e = field(Point<2>(x0, y));
      for (int m = -max_order; m <= max_order; ++m) {
        const Real ky = ky0 + 2.0 * std::numbers::pi * m / period;
        coefficients[static_cast<std::size_t>(m + max_order)] += w * std::exp(-kI * ky * y) * e;
      }
    }
  }
  return coefficients;
}

std::vector<assembly::ComplexVector<2>> fourier_coefficients(
    const fespace::NedelecDofMap<2>& dofs, const Vector& e_h, const mesh::PointLocator<2>& locator,
    Real x0, Real y0, Real period, Real ky0, int max_order, int num_points) {
  return fourier_coefficients(
      [&](const Point<2>& x) {
        const auto value = assembly::evaluate_hcurl(dofs, e_h, locator, x);
        if (!value) {
          throw InvalidArgument(
              fmt::format("fourier_coefficients: the sampling point ({}, {}) lies outside the mesh",
                          x(0), x(1)));
        }
        return *value;
      },
      x0, y0, period, ky0, max_order, num_points);
}

std::vector<DiffractionOrder> diffraction_efficiencies(
    const std::vector<assembly::ComplexVector<2>>& coefficients, Real k0, Real index_line,
    Real period, Real ky0, Real kx_incident, Real incident_amplitude) {
  if (coefficients.size() % 2 == 0) {
    throw InvalidArgument("diffraction_efficiencies: expected 2 max_order + 1 coefficients");
  }
  if (!(k0 > 0) || !(index_line > 0) || !(period > 0) || !(kx_incident > 0) ||
      !(incident_amplitude > 0)) {
    throw InvalidArgument(
        "diffraction_efficiencies: k0, index, period, kx_incident and the "
        "amplitude must be positive");
  }
  const int max_order = static_cast<int>(coefficients.size() / 2);
  const Real k = k0 * index_line;
  std::vector<DiffractionOrder> orders;
  for (int m = -max_order; m <= max_order; ++m) {
    DiffractionOrder o;
    o.order = m;
    o.ky = ky0 + 2.0 * std::numbers::pi * m / period;
    const Real kx2 = k * k - o.ky * o.ky;
    o.kx = kx2 >= 0 ? Complex{std::sqrt(kx2), 0.0} : Complex{0.0, std::sqrt(-kx2)};
    o.propagating = kx2 > 0;
    o.efficiency = o.propagating
                       ? o.kx.real() *
                             coefficients[static_cast<std::size_t>(m + max_order)].squaredNorm() /
                             (kx_incident * incident_amplitude * incident_amplitude)
                       : 0.0;
    orders.push_back(o);
  }
  return orders;
}

}  // namespace hpfem::physics

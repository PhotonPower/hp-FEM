#include "hpfem/physics/diffraction.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

#include <Eigen/Dense>
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

std::vector<DiffractionOrderField> diffraction_orders(const FieldFunction& field,
                                                      const OrderLine& line, Real k0,
                                                      Real index_line, Real k_tangential,
                                                      Real kn_incident,
                                                      const FieldFunction& incident, int max_order,
                                                      int num_points, Real incident_amplitude) {
  if (!field) throw InvalidArgument("diffraction_orders: empty field function");
  if (!(line.period > 0) || std::abs(line.tangent.norm() - 1.0) > 1e-9 ||
      std::abs(line.normal.norm() - 1.0) > 1e-9 || std::abs(line.tangent.dot(line.normal)) > 1e-9) {
    throw InvalidArgument(
        "diffraction_orders: the line needs a positive period and orthonormal tangent and "
        "normal vectors");
  }
  if (!(k0 > 0) || !(index_line > 0) || !(kn_incident > 0) || !(incident_amplitude > 0) ||
      max_order < 0) {
    throw InvalidArgument(
        "diffraction_orders: k0, index_line, kn_incident and incident_amplitude must be "
        "positive, max_order >= 0");
  }
  if (num_points <= 0) num_points = std::max(64, 16 * (2 * max_order + 1));
  const Real a = line.period;
  std::vector<assembly::ComplexVector<2>> coefficients(static_cast<std::size_t>(2 * max_order + 1),
                                                       assembly::ComplexVector<2>::Zero());
  // composite Gauss-Legendre as in fourier_coefficients: num_points points in blocks of up to
  // 8, exact for the piecewise polynomial FEM field when the blocks align with the cells
  const int blocks = (num_points + 7) / 8;
  const int per_block = (num_points + blocks - 1) / blocks;
  const auto rule = assembly::gauss_legendre(per_block);
  const Real block_length = a / blocks;
  for (int b = 0; b < blocks; ++b) {
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const Real t = (b + rule.points[q](0)) * block_length;
      const Real w = rule.weights[q] * block_length / a;
      const Point<2> x = line.origin + t * line.tangent;
      assembly::ComplexVector<2> e = field(x);
      if (incident) e -= incident(x);
      for (int m = -max_order; m <= max_order; ++m) {
        const Real kt = k_tangential + 2.0 * std::numbers::pi * m / a;
        coefficients[static_cast<std::size_t>(m + max_order)] += w * std::exp(-kI * kt * t) * e;
      }
    }
  }
  const Real k = k0 * index_line;
  std::vector<DiffractionOrderField> orders;
  orders.reserve(coefficients.size());
  for (int m = -max_order; m <= max_order; ++m) {
    DiffractionOrderField o;
    o.order = m;
    o.k_tangential = k_tangential + 2.0 * std::numbers::pi * m / a;
    const Real kn2 = k * k - o.k_tangential * o.k_tangential;
    o.kn = kn2 >= 0 ? Complex{std::sqrt(kn2), 0.0} : Complex{0.0, std::sqrt(-kn2)};
    o.propagating = kn2 > 0;
    o.amplitude = coefficients[static_cast<std::size_t>(m + max_order)];
    o.efficiency = o.propagating ? o.kn.real() * o.amplitude.squaredNorm() /
                                       (kn_incident * incident_amplitude * incident_amplitude)
                                 : 0.0;
    orders.push_back(o);
  }
  return orders;
}

std::vector<DiffractionOrder3D> diffraction_orders(const FieldFunction3D& field,
                                                   const OrderPlane& plane, Real k0,
                                                   Real index_plane, const Point<3>& k_tangential,
                                                   Real kn_incident,
                                                   const FieldFunction3D& incident, int max_order,
                                                   int num_points, Real incident_amplitude) {
  if (!field) throw InvalidArgument("diffraction_orders: empty field function");
  const Point<3>& nrm = plane.normal;
  const Real l1 = plane.a1.norm();
  const Real l2 = plane.a2.norm();
  if (std::abs(nrm.norm() - 1.0) > 1e-9 || !(l1 > 0) || !(l2 > 0) ||
      std::abs(plane.a1.dot(nrm)) > 1e-9 * l1 || std::abs(plane.a2.dot(nrm)) > 1e-9 * l2 ||
      plane.a1.cross(plane.a2).norm() < 1e-9 * l1 * l2) {
    throw InvalidArgument(
        "diffraction_orders: the plane needs a unit normal and two independent lattice vectors "
        "a1, a2 perpendicular to it");
  }
  if (!(k0 > 0) || !(index_plane > 0) || !(kn_incident > 0) || !(incident_amplitude > 0) ||
      max_order < 0) {
    throw InvalidArgument(
        "diffraction_orders: k0, index_plane, kn_incident and incident_amplitude must be "
        "positive, max_order >= 0");
  }
  if (std::abs(k_tangential.dot(nrm)) > 1e-9 * std::max(k_tangential.norm(), k0)) {
    throw InvalidArgument(fmt::format(
        "diffraction_orders: k_tangential has the normal component {} (it must lie in the plane)",
        k_tangential.dot(nrm)));
  }
  // reciprocal vectors in the plane: b_i = 2π Σ_j (G⁻¹)_ij a_j with the Gram matrix G
  Eigen::Matrix2d gram;
  gram << plane.a1.dot(plane.a1), plane.a1.dot(plane.a2), plane.a2.dot(plane.a1),
      plane.a2.dot(plane.a2);
  const Eigen::Matrix2d inv = gram.inverse();
  const Point<3> b1 = 2.0 * std::numbers::pi * (inv(0, 0) * plane.a1 + inv(0, 1) * plane.a2);
  const Point<3> b2 = 2.0 * std::numbers::pi * (inv(1, 0) * plane.a1 + inv(1, 1) * plane.a2);

  if (num_points <= 0) num_points = std::max(32, 8 * (2 * max_order + 1));
  const int blocks = (num_points + 7) / 8;
  const int per_block = (num_points + blocks - 1) / blocks;
  const auto rule = assembly::gauss_legendre(per_block);
  std::vector<Real> nodes;
  std::vector<Real> weights;
  for (int b = 0; b < blocks; ++b) {
    for (std::size_t q = 0; q < rule.size(); ++q) {
      nodes.push_back((b + rule.points[q](0)) / blocks);
      weights.push_back(rule.weights[q] / blocks);
    }
  }
  const int count = 2 * max_order + 1;
  const auto at = [&](int m, int n) {
    return static_cast<std::size_t>((m + max_order) * count + n + max_order);
  };
  // e^{-i k_mn·(s a1 + t a2)} = e^{-i k_t·(s a1 + t a2)} e^{-2πi m s} e^{-2πi n t}
  std::vector<assembly::ComplexVector<3>> coefficients(static_cast<std::size_t>(count * count),
                                                       assembly::ComplexVector<3>::Zero());
  std::vector<Complex> phase_n(static_cast<std::size_t>(count));
  std::vector<assembly::ComplexVector<3>> row(static_cast<std::size_t>(count));
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const Real s = nodes[i];
    // the inner sum over t for every n, then the m phases of s
    std::fill(row.begin(), row.end(), assembly::ComplexVector<3>::Zero());
    for (std::size_t j = 0; j < nodes.size(); ++j) {
      const Real t = nodes[j];
      const Point<3> d = s * plane.a1 + t * plane.a2;
      const Point<3> x = plane.origin + d;
      assembly::ComplexVector<3> e = field(x);
      if (incident) e -= incident(x);
      e *= weights[i] * weights[j] * std::exp(-kI * k_tangential.dot(d));
      for (int n = -max_order; n <= max_order; ++n) {
        row[static_cast<std::size_t>(n + max_order)] +=
            std::exp(-kI * (2.0 * std::numbers::pi * n * t)) * e;
      }
    }
    for (int m = -max_order; m <= max_order; ++m) {
      const Complex pm = std::exp(-kI * (2.0 * std::numbers::pi * m * s));
      for (int n = -max_order; n <= max_order; ++n) {
        coefficients[at(m, n)] += pm * row[static_cast<std::size_t>(n + max_order)];
      }
    }
  }
  const Real k = k0 * index_plane;
  std::vector<DiffractionOrder3D> orders;
  orders.reserve(coefficients.size());
  for (int m = -max_order; m <= max_order; ++m) {
    for (int n = -max_order; n <= max_order; ++n) {
      DiffractionOrder3D o;
      o.m = m;
      o.n = n;
      o.k_tangential = k_tangential + m * b1 + n * b2;
      const Real kn2 = k * k - o.k_tangential.squaredNorm();
      o.kn = kn2 >= 0 ? Complex{std::sqrt(kn2), 0.0} : Complex{0.0, std::sqrt(-kn2)};
      o.propagating = kn2 > 0;
      o.amplitude = coefficients[at(m, n)];
      o.efficiency = o.propagating ? o.kn.real() * o.amplitude.squaredNorm() /
                                         (kn_incident * incident_amplitude * incident_amplitude)
                                   : 0.0;
      orders.push_back(o);
    }
  }
  return orders;
}

}  // namespace hpfem::physics

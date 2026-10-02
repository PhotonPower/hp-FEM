#include "hpfem/physics/farfield.hpp"

#include <cmath>
#include <numbers>

#include <fmt/format.h>

#include "hpfem/assembly/quadrature.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/error.hpp"

namespace hpfem::physics {

namespace {

Eigen::Matrix<Complex, 3, 1> cross(const Point<3>& a, const Eigen::Matrix<Complex, 3, 1>& b) {
  return {a(1) * b(2) - a(2) * b(1), a(2) * b(0) - a(0) * b(2), a(0) * b(1) - a(1) * b(0)};
}

}  // namespace

template <int Dim>
FarField<Dim>::FarField(const mesh::Mesh<Dim>& mesh, const Surface<Dim>& surface,
                        const SurfaceField<Dim>& field, Real omega,
                        const materials::Material& background, int order) {
  if (!(omega > 0)) throw InvalidArgument("FarField: omega must be positive");
  const Complex n = background.refractive_index();
  if (std::abs(n.imag()) > 1e-12 * std::abs(n)) {
    throw InvalidArgument("FarField: the background medium must be lossless");
  }
  k_ = omega / constants::c0 * n.real();
  impedance_ = std::real(constants::Z0 * std::sqrt(background.mu_r / background.eps_r));
  const Complex h_factor = 1.0 / (kI * omega * constants::mu0 * background.mu_r);
  assembly::ComplexVector<Dim> e;
  assembly::ComplexCurl<Dim> curl;
  for (const auto& p : surface_quadrature<Dim>(mesh, surface, order)) {
    field(p, e, curl);
    samples_.push_back({p.x, p.normal, p.weight, e, assembly::ComplexCurl<Dim>(h_factor * curl)});
  }
  if (samples_.empty()) throw InvalidArgument("FarField: the surface has no facets");
}

template <int Dim>
assembly::ComplexVector<Dim> FarField<Dim>::pattern(const Point<Dim>& direction) const {
  const Point<Dim> r = direction.normalized();
  if constexpr (Dim == 2) {
    const Point<2> t(-r(1), r(0));  // z x r
    Complex n_t = 0;                // (n x H) . t with H = H_z z: n x z = (n_y, -n_x)
    Complex l_z = 0;                // -(n x E)_z = -(n_x E_y - n_y E_x)
    for (const auto& s : samples_) {
      const Complex phase = std::exp(-kI * k_ * r.dot(s.x)) * s.weight;
      const Point<2> n_cross_z(s.normal(1), -s.normal(0));
      n_t += phase * s.h(0) * n_cross_z.dot(t);
      l_z -= phase * (s.normal(0) * s.e(1) - s.normal(1) * s.e(0));
    }
    const Complex amplitude = -(k_ / 4.0) * std::sqrt(2.0 / (std::numbers::pi * k_)) *
                              std::exp(-kI * std::numbers::pi / 4.0) * (impedance_ * n_t + l_z);
    return assembly::ComplexVector<2>(amplitude * t.template cast<Complex>());
  } else {
    Eigen::Matrix<Complex, 3, 1> n_vec = Eigen::Matrix<Complex, 3, 1>::Zero();
    Eigen::Matrix<Complex, 3, 1> l_vec = Eigen::Matrix<Complex, 3, 1>::Zero();
    for (const auto& s : samples_) {
      const Complex phase = std::exp(-kI * k_ * r.dot(s.x)) * s.weight;
      n_vec += phase * cross(s.normal, s.h);
      l_vec -= phase * cross(s.normal, s.e);
    }
    Complex radial = 0;
    for (int d = 0; d < 3; ++d) radial += r(d) * n_vec(d);
    const Eigen::Matrix<Complex, 3, 1> n_t = n_vec - radial * r.template cast<Complex>();
    return assembly::ComplexVector<3>(kI * k_ / (4.0 * std::numbers::pi) *
                                      (impedance_ * n_t - cross(r, l_vec)));
  }
}

template <int Dim>
Real FarField<Dim>::radiated_power(int resolution) const {
  if (resolution < 4) throw InvalidArgument("FarField: resolution must be at least 4");
  Real integral = 0;
  if constexpr (Dim == 2) {
    // periodic trapezoidal rule in the polar angle (spectrally accurate)
    for (int i = 0; i < resolution; ++i) {
      const Real phi = 2.0 * std::numbers::pi * static_cast<Real>(i) / resolution;
      integral += pattern(Point<2>(std::cos(phi), std::sin(phi))).squaredNorm();
    }
    integral *= 2.0 * std::numbers::pi / resolution;
  } else {
    // Gauss-Legendre in cos(theta), trapezoidal in phi
    const auto rule = assembly::gauss_legendre(resolution / 2);
    for (std::size_t q = 0; q < rule.size(); ++q) {
      const Real c = 2.0 * rule.points[q](0) - 1.0;  // cos(theta) on [-1, 1]
      const Real s = std::sqrt(std::max(0.0, 1.0 - c * c));
      for (int i = 0; i < resolution; ++i) {
        const Real phi = 2.0 * std::numbers::pi * static_cast<Real>(i) / resolution;
        integral += 2.0 * rule.weights[q] * (2.0 * std::numbers::pi / resolution) *
                    pattern(Point<3>(s * std::cos(phi), s * std::sin(phi), c)).squaredNorm();
      }
    }
  }
  return integral / (2.0 * impedance_);
}

template <int Dim>
Real FarField<Dim>::scattering_cross_section(Real incident_amplitude, int resolution) const {
  return radiated_power(resolution) * 2.0 * impedance_ / (incident_amplitude * incident_amplitude);
}

template class FarField<2>;
template class FarField<3>;

}  // namespace hpfem::physics

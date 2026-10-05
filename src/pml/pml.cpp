#include "hpfem/pml/pml.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

#include <fmt/format.h>

#include "hpfem/core/error.hpp"

namespace hpfem::pml {

PmlProfile PmlProfile::for_angle(Real theta_max, Real target, Real r_amplitude, int order) {
  if (!(theta_max >= 0) || !(theta_max < std::numbers::pi / 2)) {
    throw InvalidArgument(fmt::format(
        "PmlProfile::for_angle: the angle {} must lie in [0, pi/2) (radians against the normal)",
        theta_max));
  }
  if (!(target > 0) || !(r_amplitude > 0) || order < 1) {
    throw InvalidArgument(
        "PmlProfile::for_angle: target and r_amplitude must be positive, order >= 1");
  }
  // one-way field at the far wall R0^(cos theta / 2), reflectance error 2 r R0^(cos theta / 2)
  const Real reflection = std::pow(target / (2.0 * r_amplitude), 2.0 / std::cos(theta_max));
  PmlProfile profile;
  profile.order = order;
  profile.reflection = std::clamp(reflection, 1e-300, std::nextafter(1.0, 0.0));
  return profile;
}

template <int Dim>
PmlBox<Dim>::PmlBox(const Point<Dim>& lower, const Point<Dim>& upper, const Thickness& thickness,
                    Real k0, Real background_index, PmlProfile profile)
    : lower_(lower), upper_(upper), thickness_(thickness), profile_(profile), k0_(k0) {
  for (int d = 0; d < Dim; ++d) {
    if (!(upper(d) > lower(d))) {
      throw InvalidArgument(fmt::format("PmlBox: degenerate box along axis {}", d));
    }
  }
  for (std::size_t k = 0; k < kNumSides; ++k) {
    if (!(thickness[k] >= 0)) {
      throw InvalidArgument(fmt::format("PmlBox: negative thickness on side {}", k));
    }
  }
  if (!(k0 > 0)) throw InvalidArgument(fmt::format("PmlBox: k0 = {} must be positive", k0));
  if (!(background_index > 0)) {
    throw InvalidArgument("PmlBox: the background refractive index must be positive");
  }
  if (profile.order < 1 || !(profile.reflection > 0) || !(profile.reflection < 1)) {
    throw InvalidArgument(fmt::format(
        "PmlBox: profile order {} must be >= 1 and the target reflection {} must lie in (0, 1)",
        profile.order, profile.reflection));
  }
  // sigma_max = -(m + 1) ln R0 / (2 k0 n d): round trip e^{-2 k0 n int sigma} = R0
  for (std::size_t k = 0; k < kNumSides; ++k) {
    sigma_max_[k] = thickness[k] > 0
                        ? -static_cast<Real>(profile.order + 1) * std::log(profile.reflection) /
                              (2.0 * k0 * background_index * thickness[k])
                        : 0.0;
  }
}

template <int Dim>
PmlBox<Dim> PmlBox<Dim>::uniform(const Point<Dim>& lower, const Point<Dim>& upper, Real thickness,
                                 Real k0, Real background_index, PmlProfile profile) {
  Thickness t{};
  t.fill(thickness);
  return PmlBox(lower, upper, t, k0, background_index, profile);
}

template <int Dim>
Real PmlBox<Dim>::recommended_thickness(Real k0, Real background_index, Real cell_size,
                                        Real wavelengths) {
  if (!(k0 > 0) || !(background_index > 0) || !(cell_size > 0) || !(wavelengths > 0)) {
    throw InvalidArgument("PmlBox::recommended_thickness: all arguments must be positive");
  }
  const Real wavelength = 2.0 * std::numbers::pi / (k0 * background_index);
  const Real cells = std::ceil(wavelengths * wavelength / cell_size - 1e-9);
  return std::max(cells, 1.0) * cell_size;
}

template <int Dim>
Real PmlBox<Dim>::resolution_limit(int p) noexcept {
  return p >= 4 ? 3.0 : 0.75 * static_cast<Real>(std::max(p, 1));
}

template <int Dim>
Real PmlBox<Dim>::recommended_thickness(Real k0, Real background_index, Real cell_size,
                                        const PmlProfile& profile, int p) {
  if (!(k0 > 0) || !(background_index > 0) || !(cell_size > 0)) {
    throw InvalidArgument("PmlBox::recommended_thickness: all arguments must be positive");
  }
  if (profile.order < 1 || !(profile.reflection > 0) || !(profile.reflection < 1)) {
    throw InvalidArgument("PmlBox::recommended_thickness: unusable profile");
  }
  const Real limit = resolution_limit(p);
  const Real kh = k0 * background_index * cell_size;
  if (!(kh < limit)) {
    throw InvalidArgument(fmt::format(
        "PmlBox::recommended_thickness: k0 n h = {:.3g} exceeds the resolution limit {:.3g} of "
        "order {} even without stretching; refine the mesh or raise the order",
        kh, limit, p));
  }
  // |k s| h <= limit with |s| = sqrt(1 + sigma_max^2) and sigma_max = -(m+1) ln R0 / (2 k0 n d)
  const Real sigma_limit = std::sqrt((limit / kh) * (limit / kh) - 1.0);
  const Real exponent = -static_cast<Real>(profile.order + 1) * std::log(profile.reflection);
  const Real thickness = exponent / (2.0 * k0 * background_index * sigma_limit);
  const Real cells = std::ceil(thickness / cell_size - 1e-9);
  return std::max(std::max(cells, 1.0) * cell_size,
                  recommended_thickness(k0, background_index, cell_size));
}

template <int Dim>
Real PmlBox<Dim>::max_resolution(Real cell_size, Real index) const {
  if (!(cell_size > 0) || !(index > 0)) {
    throw InvalidArgument("PmlBox::max_resolution: cell size and index must be positive");
  }
  Real worst = 0;
  for (std::size_t k = 0; k < kNumSides; ++k) {
    if (thickness_[k] <= 0) continue;
    const Real s = std::sqrt(1.0 + sigma_max_[k] * sigma_max_[k]);
    worst = std::max(worst, k0_ * index * s * cell_size);
  }
  return worst;
}

template <int Dim>
Point<Dim> PmlBox<Dim>::outer_lower() const {
  Point<Dim> x = lower_;
  for (int d = 0; d < Dim; ++d) x(d) -= thickness_[static_cast<std::size_t>(2 * d)];
  return x;
}

template <int Dim>
Point<Dim> PmlBox<Dim>::outer_upper() const {
  Point<Dim> x = upper_;
  for (int d = 0; d < Dim; ++d) x(d) += thickness_[static_cast<std::size_t>(2 * d + 1)];
  return x;
}

template <int Dim>
Real PmlBox<Dim>::depth(const Point<Dim>& x, int axis) const {
  if (x(axis) < lower_(axis)) return x(axis) - lower_(axis);
  if (x(axis) > upper_(axis)) return x(axis) - upper_(axis);
  return 0.0;
}

template <int Dim>
bool PmlBox<Dim>::in_layer(const Point<Dim>& x) const {
  for (int d = 0; d < Dim; ++d) {
    const Real depth_d = depth(x, d);
    const std::size_t side = static_cast<std::size_t>(2 * d) + (depth_d > 0 ? 1 : 0);
    if (depth_d != 0 && thickness_[side] > 0) return true;
  }
  return false;
}

template <int Dim>
ComplexCoordinates<Dim> PmlBox<Dim>::stretch(const Point<Dim>& x) const {
  ComplexCoordinates<Dim> s = ComplexCoordinates<Dim>::Ones();
  for (int d = 0; d < Dim; ++d) {
    const Real depth_d = depth(x, d);
    if (depth_d == 0) continue;
    const std::size_t side = static_cast<std::size_t>(2 * d) + (depth_d > 0 ? 1 : 0);
    if (thickness_[side] <= 0) continue;
    const Real t = std::min(std::abs(depth_d) / thickness_[side], 1.0);
    s(d) = Complex{1.0, sigma_max_[side] * std::pow(t, profile_.order)};
  }
  return s;
}

template <int Dim>
ComplexCoordinates<Dim> PmlBox<Dim>::stretched_coordinate(const Point<Dim>& x) const {
  ComplexCoordinates<Dim> xt = x.template cast<Complex>();
  for (int d = 0; d < Dim; ++d) {
    const Real depth_d = depth(x, d);
    if (depth_d == 0) continue;
    const std::size_t side = static_cast<std::size_t>(2 * d) + (depth_d > 0 ? 1 : 0);
    if (thickness_[side] <= 0) continue;
    // int_0^|depth| sigma_max (t / d)^m dt = sigma_max d / (m + 1) (|depth| / d)^(m + 1),
    // with the sign of the outward direction (s = 1 + i sigma along the axis)
    const Real t = std::min(std::abs(depth_d) / thickness_[side], 1.0);
    const Real excess = std::max(std::abs(depth_d) - thickness_[side], 0.0);
    const Real integral =
        sigma_max_[side] *
        (thickness_[side] / (profile_.order + 1) * std::pow(t, profile_.order + 1) + excess);
    xt(d) += Complex{0.0, depth_d > 0 ? integral : -integral};
  }
  return xt;
}

template <int Dim>
assembly::PermittivityTensor<Dim> PmlBox<Dim>::permittivity(Complex eps_r,
                                                            const Point<Dim>& x) const {
  const ComplexCoordinates<Dim> s = stretch(x);
  Complex det = 1.0;
  for (int d = 0; d < Dim; ++d) det *= s(d);
  assembly::PermittivityTensor<Dim> eps = assembly::PermittivityTensor<Dim>::Zero();
  for (int d = 0; d < Dim; ++d) eps(d, d) = eps_r * det / (s(d) * s(d));
  return eps;
}

template <int Dim>
assembly::InversePermeabilityTensor<Dim> PmlBox<Dim>::inverse_permeability(
    Complex mu_r, const Point<Dim>& x) const {
  const ComplexCoordinates<Dim> s = stretch(x);
  Complex det = 1.0;
  for (int d = 0; d < Dim; ++d) det *= s(d);
  assembly::InversePermeabilityTensor<Dim> inv = assembly::InversePermeabilityTensor<Dim>::Zero();
  if constexpr (Dim == 2) {
    // out-of-plane curl: s_z = 1, (mu~^-1)_zz = s_z^2 / (mu_r det)
    inv(0, 0) = 1.0 / (mu_r * det);
  } else {
    for (int d = 0; d < 3; ++d) inv(d, d) = s(d) * s(d) / (mu_r * det);
  }
  return inv;
}

template class PmlBox<2>;
template class PmlBox<3>;

}  // namespace hpfem::pml

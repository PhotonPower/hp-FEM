#include "hpfem/pml/pml.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/LU>
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
assembly::PermittivityTensor<Dim> PmlBox<Dim>::permittivity(
    const assembly::PermittivityTensor<Dim>& eps, const Point<Dim>& x) const {
  const ComplexCoordinates<Dim> s = stretch(x);
  Complex det = 1.0;
  for (int d = 0; d < Dim; ++d) det *= s(d);
  assembly::PermittivityTensor<Dim> out;
  for (int i = 0; i < Dim; ++i) {
    for (int j = 0; j < Dim; ++j) out(i, j) = eps(i, j) * det / (s(i) * s(j));
  }
  return out;
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

namespace {

/// Stretch factor and stretch integral at `depth` >= 0 into a layer of the polynomial profile;
/// beyond the layer the final stretch continues.
struct LayerStretch {
  Complex factor{1.0, 0.0};
  Real integral = 0;
};

LayerStretch layer_stretch(Real depth, Real thickness, Real sigma_max, int order) {
  const Real t = std::min(depth / thickness, 1.0);
  const Real excess = std::max(depth - thickness, 0.0);
  return {Complex{1.0, sigma_max * std::pow(t, order)},
          sigma_max * (thickness / (order + 1) * std::pow(t, order + 1) + excess)};
}

Real design_sigma_max(const PmlProfile& profile, Real k0, Real index, Real thickness) {
  return -static_cast<Real>(profile.order + 1) * std::log(profile.reflection) /
         (2.0 * k0 * index * thickness);
}

void check_radial(Real inner_radius, Real thickness, Real k0, Real index,
                  const PmlProfile& profile) {
  if (!(inner_radius > 0) || !(thickness > 0)) {
    throw InvalidArgument(fmt::format(
        "RadialPml: inner radius {} and thickness {} must be positive", inner_radius, thickness));
  }
  if (!(k0 > 0)) throw InvalidArgument(fmt::format("RadialPml: k0 = {} must be positive", k0));
  if (!(index > 0)) {
    throw InvalidArgument("RadialPml: the background refractive index must be positive");
  }
  if (profile.order < 1 || !(profile.reflection > 0) || !(profile.reflection < 1)) {
    throw InvalidArgument(fmt::format(
        "RadialPml: profile order {} must be >= 1 and the target reflection {} must lie in (0, 1)",
        profile.order, profile.reflection));
  }
}

}  // namespace

template <int Dim>
RadialPml<Dim>::RadialPml(const Point<Dim>& center, Real inner_radius, Real thickness, Real k0,
                          Real background_index, PmlProfile profile)
    : shape_(Dim == 2 ? RadialShape::kCircular : RadialShape::kSpherical),
      center_(center),
      inner_radius_(inner_radius),
      thickness_(thickness),
      profile_(profile),
      k0_(k0) {
  check_radial(inner_radius, thickness, k0, background_index, profile);
  sigma_max_ = design_sigma_max(profile, k0, background_index, thickness);
}

template <int Dim>
RadialPml<Dim> RadialPml<Dim>::cylindrical(const Point<3>& center, const Point<3>& axis,
                                           Real inner_radius, Real thickness, Real axial_lower,
                                           Real axial_upper, Real axial_thickness, Real k0,
                                           Real background_index, PmlProfile profile)
  requires(Dim == 3)
{
  check_radial(inner_radius, thickness, k0, background_index, profile);
  if (!(axis.norm() > 0)) throw InvalidArgument("RadialPml::cylindrical: zero axis");
  if (!(axial_upper > axial_lower) || !(axial_thickness >= 0)) {
    throw InvalidArgument(fmt::format(
        "RadialPml::cylindrical: need axial_lower < axial_upper and axial_thickness >= 0 (got "
        "{}, {}, {})",
        axial_lower, axial_upper, axial_thickness));
  }
  RadialPml pml;
  pml.shape_ = RadialShape::kCylindrical;
  pml.center_ = center;
  pml.axis_ = axis.normalized();
  pml.inner_radius_ = inner_radius;
  pml.thickness_ = thickness;
  pml.axial_lower_ = axial_lower;
  pml.axial_upper_ = axial_upper;
  pml.axial_thickness_ = axial_thickness;
  pml.profile_ = profile;
  pml.k0_ = k0;
  pml.sigma_max_ = design_sigma_max(profile, k0, background_index, thickness);
  pml.axial_sigma_max_ =
      axial_thickness > 0 ? design_sigma_max(profile, k0, background_index, axial_thickness) : 0;
  return pml;
}

template <int Dim>
std::pair<Real, Point<Dim>> RadialPml<Dim>::radial(const Point<Dim>& x) const {
  Point<Dim> v = x - center_;
  if constexpr (Dim == 3) {
    if (shape_ == RadialShape::kCylindrical) v -= axis_ * axis_.dot(v);
  }
  const Real r = v.norm();
  return {r, r > 0 ? Point<Dim>(v / r) : Point<Dim>(Point<Dim>::Zero())};
}

template <int Dim>
Real RadialPml<Dim>::axial_depth(const Point<Dim>& x) const {
  if constexpr (Dim == 3) {
    if (shape_ != RadialShape::kCylindrical || axial_thickness_ <= 0) return 0;
    const Real a = axis_.dot(x - center_);
    if (a < axial_lower_) return a - axial_lower_;
    if (a > axial_upper_) return a - axial_upper_;
  }
  return 0;
}

template <int Dim>
Real RadialPml<Dim>::max_resolution(Real cell_size, Real index) const {
  if (!(cell_size > 0) || !(index > 0)) {
    throw InvalidArgument("RadialPml::max_resolution: cell size and index must be positive");
  }
  const Real sigma = std::max(sigma_max_, axial_sigma_max_);
  return k0_ * index * std::sqrt(1.0 + sigma * sigma) * cell_size;
}

template <int Dim>
bool RadialPml<Dim>::in_layer(const Point<Dim>& x) const {
  return radial(x).first > inner_radius_ || axial_depth(x) != 0;
}

template <int Dim>
std::vector<Point<Dim>> RadialPml<Dim>::stretch_directions(const Point<Dim>& x) const {
  std::vector<Point<Dim>> directions;
  const auto [r, u] = radial(x);
  if (r > inner_radius_) directions.push_back(u);
  if constexpr (Dim == 3) {
    if (axial_depth(x) != 0) directions.push_back(axis_);
  }
  return directions;
}

template <int Dim>
StretchMatrix<Dim> RadialPml<Dim>::jacobian(const Point<Dim>& x) const {
  using Matrix = StretchMatrix<Dim>;
  Matrix j = Matrix::Identity();
  const auto [r, u] = radial(x);
  if (r > inner_radius_) {
    const LayerStretch l = layer_stretch(r - inner_radius_, thickness_, sigma_max_, profile_.order);
    const Complex ratio = Complex{r, l.integral} / r;
    // J = I + (r~/r - 1)(P - u u^T) + (s_r - 1) u u^T, P the projector onto the stretched space
    Matrix plane = Matrix::Identity();
    if constexpr (Dim == 3) {
      if (shape_ == RadialShape::kCylindrical) {
        plane -= (axis_ * axis_.transpose()).template cast<Complex>();
      }
    }
    const Matrix uu = (u * u.transpose()).template cast<Complex>();
    j += (ratio - 1.0) * (plane - uu) + (l.factor - 1.0) * uu;
  }
  if constexpr (Dim == 3) {
    const Real depth = axial_depth(x);
    if (depth != 0) {
      const LayerStretch l =
          layer_stretch(std::abs(depth), axial_thickness_, axial_sigma_max_, profile_.order);
      j += (l.factor - 1.0) * (axis_ * axis_.transpose()).template cast<Complex>();
    }
  }
  return j;
}

template <int Dim>
ComplexCoordinates<Dim> RadialPml<Dim>::stretched_coordinate(const Point<Dim>& x) const {
  ComplexCoordinates<Dim> xt = x.template cast<Complex>();
  const auto [r, u] = radial(x);
  if (r > inner_radius_) {
    const LayerStretch l = layer_stretch(r - inner_radius_, thickness_, sigma_max_, profile_.order);
    // x~ = x + i int sigma u (the radial part scaled by r~ / r)
    xt += Complex{0.0, l.integral} * u.template cast<Complex>();
  }
  if constexpr (Dim == 3) {
    const Real depth = axial_depth(x);
    if (depth != 0) {
      const LayerStretch l =
          layer_stretch(std::abs(depth), axial_thickness_, axial_sigma_max_, profile_.order);
      xt += Complex{0.0, depth > 0 ? l.integral : -l.integral} * axis_.template cast<Complex>();
    }
  }
  return xt;
}

template <int Dim>
assembly::PermittivityTensor<Dim> RadialPml<Dim>::permittivity(Complex eps_r,
                                                               const Point<Dim>& x) const {
  const StretchMatrix<Dim> j = jacobian(x);
  const StretchMatrix<Dim> inv = j.inverse();
  return eps_r * j.determinant() * inv * inv.transpose();
}

template <int Dim>
assembly::PermittivityTensor<Dim> RadialPml<Dim>::permittivity(
    const assembly::PermittivityTensor<Dim>& eps, const Point<Dim>& x) const {
  const StretchMatrix<Dim> j = jacobian(x);
  const StretchMatrix<Dim> inv = j.inverse();
  return j.determinant() * inv * eps * inv.transpose();
}

template <int Dim>
assembly::InversePermeabilityTensor<Dim> RadialPml<Dim>::inverse_permeability(
    Complex mu_r, const Point<Dim>& x) const {
  const StretchMatrix<Dim> j = jacobian(x);
  if constexpr (Dim == 2) {
    assembly::InversePermeabilityTensor<2> inv;
    inv(0, 0) = 1.0 / (mu_r * j.determinant());
    return inv;
  } else {
    return j * j.transpose() / (mu_r * j.determinant());
  }
}

template <int Dim>
Real Pml<Dim>::k0() const {
  return std::visit([](const auto& l) { return l.k0(); }, layer_);
}

template <int Dim>
bool Pml<Dim>::in_layer(const Point<Dim>& x) const {
  return std::visit([&x](const auto& l) { return l.in_layer(x); }, layer_);
}

template <int Dim>
std::vector<Point<Dim>> Pml<Dim>::stretch_directions(const Point<Dim>& x) const {
  if (const auto* r = radial()) return r->stretch_directions(x);
  const ComplexCoordinates<Dim> s = box()->stretch(x);
  std::vector<Point<Dim>> directions;
  for (int d = 0; d < Dim; ++d) {
    if (s(d) != Complex{1.0, 0.0}) directions.push_back(Point<Dim>::Unit(d));
  }
  return directions;
}

template <int Dim>
StretchMatrix<Dim> Pml<Dim>::jacobian(const Point<Dim>& x) const {
  if (const auto* r = radial()) return r->jacobian(x);
  return box()->stretch(x).asDiagonal();
}

template <int Dim>
ComplexCoordinates<Dim> Pml<Dim>::stretched_coordinate(const Point<Dim>& x) const {
  return std::visit([&x](const auto& l) { return l.stretched_coordinate(x); }, layer_);
}

template <int Dim>
Real Pml<Dim>::max_resolution(Real cell_size, Real index) const {
  return std::visit([=](const auto& l) { return l.max_resolution(cell_size, index); }, layer_);
}

template <int Dim>
assembly::PermittivityTensor<Dim> Pml<Dim>::permittivity(Complex eps_r, const Point<Dim>& x) const {
  return std::visit([&](const auto& l) { return l.permittivity(eps_r, x); }, layer_);
}

template <int Dim>
assembly::PermittivityTensor<Dim> Pml<Dim>::permittivity(
    const assembly::PermittivityTensor<Dim>& eps, const Point<Dim>& x) const {
  return std::visit([&](const auto& l) { return l.permittivity(eps, x); }, layer_);
}

template <int Dim>
assembly::InversePermeabilityTensor<Dim> Pml<Dim>::inverse_permeability(Complex mu_r,
                                                                        const Point<Dim>& x) const {
  return std::visit([&](const auto& l) { return l.inverse_permeability(mu_r, x); }, layer_);
}

template <int Dim>
bool Pml<Dim>::is_invariant(const Eigen::Matrix<Real, Dim, Dim>& q, const Point<Dim>& shift,
                            std::span<const Point<Dim>> points, Real tolerance) const {
  const StretchMatrix<Dim> qc = q.template cast<Complex>();
  for (const Point<Dim>& x : points) {
    const StretchMatrix<Dim> j = jacobian(x);
    const StretchMatrix<Dim> image = jacobian(Point<Dim>(q * x + shift));
    const Real scale = std::max(1.0, j.norm());
    if ((image - qc * j * qc.transpose()).norm() > tolerance * scale) return false;
  }
  return true;
}

template <int Dim>
std::optional<BackwardWave<Dim>> backward_wave(const assembly::PermittivityTensor<Dim>& eps,
                                               int axis, int samples) {
  if (axis < 0 || axis >= Dim) {
    throw InvalidArgument(fmt::format("backward_wave: axis {} outside [0, {})", axis, Dim));
  }
  if (samples < 8) throw InvalidArgument("backward_wave: at least 8 samples");
  using RealMatrix = Eigen::Matrix<Real, Dim, Dim>;
  const RealMatrix e = eps.real();
  BackwardWave<Dim> worst;
  worst.worst = 1.0;
  // product of the unit phase and group directions along the axis for one wave
  const auto consider = [&](const Point<Dim>& k, const Point<Dim>& v) {
    const Real norm = v.norm();
    if (!(norm > 0)) return;
    const Real product = k(axis) * v(axis) / norm;
    if (product < worst.worst) {
      worst.worst = product;
      worst.direction = k;
    }
  };
  if constexpr (Dim == 2) {
    // in-plane E, H_z: slowness curve k^T Q k = k0^2 with Q = R^T eps^-1 R, R k = (k_y, -k_x);
    // the group velocity is the normal Q k (no propagating wave where k^T Q k <= 0)
    if (std::abs(e.determinant()) == 0) return std::nullopt;
    RealMatrix r;
    r << 0.0, 1.0, -1.0, 0.0;
    const RealMatrix q = r.transpose() * e.inverse() * r;
    for (int j = 0; j < samples; ++j) {
      const Real theta = std::numbers::pi * (j + 0.5) / samples;
      const Point<2> k(std::cos(theta), std::sin(theta));
      if (!(k.dot(q * k) > 0)) continue;
      consider(k, q * k);
    }
  } else {
    // eigenwaves along k: P E = lambda eps E with P = I - k k^T, lambda = 1/n^2 > 0 propagating
    if (std::abs(e.determinant()) == 0) return std::nullopt;
    const RealMatrix inv = e.inverse();
    const Real golden = std::numbers::pi * (3.0 - std::sqrt(5.0));
    for (int j = 0; j < samples; ++j) {
      const Real z = (j + 0.5) / samples;  // hemisphere z > 0
      const Real rho = std::sqrt(std::max(0.0, 1.0 - z * z));
      const Point<3> k(rho * std::cos(golden * j), rho * std::sin(golden * j), z);
      const RealMatrix p = RealMatrix::Identity() - k * k.transpose();
      const Eigen::EigenSolver<RealMatrix> eig(inv * p);
      for (int m = 0; m < 3; ++m) {
        const Complex lambda = eig.eigenvalues()(m);
        if (!(lambda.real() > 1e-12) || std::abs(lambda.imag()) > 1e-9 * std::abs(lambda)) {
          continue;
        }
        // real eigenvector up to a complex phase: rotate its largest entry onto the real axis
        Eigen::Vector3cd v = eig.eigenvectors().col(m);
        Eigen::Index big = 0;
        v.cwiseAbs().maxCoeff(&big);
        v *= std::abs(v(big)) / v(big);
        const Point<3> field = v.real();
        consider(k, Point<3>(k * field.squaredNorm() - field * k.dot(field)));
      }
    }
  }
  if (worst.worst < -1e-9) return worst;
  return std::nullopt;
}

template <int Dim>
std::optional<BackwardWave<Dim>> backward_wave_along(const assembly::PermittivityTensor<Dim>& eps,
                                                     const Point<Dim>& direction, int samples) {
  if (!(direction.norm() > 0)) throw InvalidArgument("backward_wave_along: zero direction");
  // orthogonal Q with rows (u, ...): Q u = e_0, eps' = Q eps Q^T
  const Point<Dim> u = direction.normalized();
  Eigen::Matrix<Real, Dim, Dim> q;
  if constexpr (Dim == 2) {
    q << u(0), u(1), -u(1), u(0);
  } else {
    const Point<3> v = u.unitOrthogonal();
    q.row(0) = u.transpose();
    q.row(1) = v.transpose();
    q.row(2) = u.cross(v).transpose();
  }
  const Eigen::Matrix<Complex, Dim, Dim> qc = q.template cast<Complex>();
  auto wave =
      backward_wave<Dim>(assembly::PermittivityTensor<Dim>(qc * eps * qc.transpose()), 0, samples);
  if (wave) wave->direction = q.transpose() * wave->direction;
  return wave;
}

template std::optional<BackwardWave<2>> backward_wave_along<2>(
    const assembly::PermittivityTensor<2>&, const Point<2>&, int);
template std::optional<BackwardWave<3>> backward_wave_along<3>(
    const assembly::PermittivityTensor<3>&, const Point<3>&, int);

template std::optional<BackwardWave<2>> backward_wave<2>(const assembly::PermittivityTensor<2>&,
                                                         int, int);
template std::optional<BackwardWave<3>> backward_wave<3>(const assembly::PermittivityTensor<3>&,
                                                         int, int);

template class PmlBox<2>;
template class PmlBox<3>;
template class RadialPml<2>;
template class RadialPml<3>;
template class Pml<2>;
template class Pml<3>;

}  // namespace hpfem::pml

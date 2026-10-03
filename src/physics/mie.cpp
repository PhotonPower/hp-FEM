#include "hpfem/physics/mie.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

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

// --- sphere -------------------------------------------------------------------------------------

namespace {

/// Angular functions @f$ \pi_n(\mu) = P_n^1(\mu)/\sin\theta @f$ and
/// @f$ \tau_n(\mu) = dP_n^1/d\theta @f$, n = 1..N (index n − 1), by the recurrences
/// @f$ \pi_n = \frac{2n-1}{n-1}\mu\pi_{n-1} - \frac{n}{n-1}\pi_{n-2} @f$,
/// @f$ \tau_n = n\mu\pi_n - (n+1)\pi_{n-1} @f$ (Bohren & Huffman 4.47).
struct Angular {
  std::vector<Real> pi, tau;
};
Angular angular_functions(int max_order, Real mu) {
  Angular f;
  f.pi.resize(static_cast<std::size_t>(max_order));
  f.tau.resize(static_cast<std::size_t>(max_order));
  Real pi_before = 0;  // pi_{n-1}
  Real pi_n = 1;       // pi_1
  for (int n = 1; n <= max_order; ++n) {
    if (n > 1) {
      const Real next =
          (static_cast<Real>(2 * n - 1) * mu * pi_n - static_cast<Real>(n) * pi_before) /
          static_cast<Real>(n - 1);
      pi_before = pi_n;
      pi_n = next;
    }
    f.pi[static_cast<std::size_t>(n - 1)] = pi_n;
    f.tau[static_cast<std::size_t>(n - 1)] =
        static_cast<Real>(n) * mu * pi_n - static_cast<Real>(n + 1) * pi_before;
  }
  return f;
}

/// Spherical coordinates of x with the polar axis z; on the axis the azimuth is set to 0 (the
/// Cartesian field is continuous there, so any value gives the same vector).
struct Spherical {
  Real r, cos_theta, sin_theta, cos_phi, sin_phi;
};
Spherical spherical(const Point<3>& x) {
  Spherical s{};
  s.r = x.norm();
  s.cos_theta = s.r > 0 ? std::clamp(x(2) / s.r, -1.0, 1.0) : 1.0;
  s.sin_theta = std::sqrt(std::max(0.0, 1 - s.cos_theta * s.cos_theta));
  const Real rho = std::hypot(x(0), x(1));
  s.cos_phi = rho > 0 ? x(0) / rho : 1.0;
  s.sin_phi = rho > 0 ? x(1) / rho : 0.0;
  return s;
}

/// Cartesian vector from the spherical components (r, θ, φ).
assembly::ComplexVector<3> to_cartesian(const Spherical& s, Complex e_r, Complex e_theta,
                                        Complex e_phi) {
  const Point<3> r_hat(s.sin_theta * s.cos_phi, s.sin_theta * s.sin_phi, s.cos_theta);
  const Point<3> theta_hat(s.cos_theta * s.cos_phi, s.cos_theta * s.sin_phi, -s.sin_theta);
  const Point<3> phi_hat(-s.sin_phi, s.cos_phi, 0.0);
  return e_r * r_hat.cast<Complex>() + e_theta * theta_hat.cast<Complex>() +
         e_phi * phi_hat.cast<Complex>();
}

/// @f$ \sum_n E_n (\alpha_n N_{e1n} + \beta_n M_{o1n}) @f$ with the radial functions
/// z_n(ρ), n = 0..N, ρ = k r (complex inside the sphere) and E_n = i^n (2n+1)/(n(n+1)).
assembly::ComplexVector<3> harmonic_sum(const Spherical& s, Complex rho,
                                        const std::vector<Complex>& z,
                                        const std::vector<Complex>& alpha,
                                        const std::vector<Complex>& beta) {
  const int max_order = static_cast<int>(alpha.size());
  const Angular f = angular_functions(max_order, s.cos_theta);
  Complex e_r{0.0, 0.0};
  Complex e_theta{0.0, 0.0};
  Complex e_phi{0.0, 0.0};
  Complex i_power = kI;  // i^n
  for (int n = 1; n <= max_order; ++n) {
    const auto idx = static_cast<std::size_t>(n);
    const Complex e_n = i_power * static_cast<Real>(2 * n + 1) / static_cast<Real>(n * (n + 1));
    const Complex zn = z[idx];
    const Complex dzn = z[idx - 1] - static_cast<Real>(n) * zn / rho;  // [ρ z_n]' / ρ
    const Real pi_n = f.pi[idx - 1];
    const Real tau_n = f.tau[idx - 1];
    // N_e1n = cosφ n(n+1) sinθ π_n z_n/ρ r̂ + cosφ τ_n [ρz_n]'/ρ θ̂ − sinφ π_n [ρz_n]'/ρ φ̂
    // M_o1n = cosφ π_n z_n θ̂ − sinφ τ_n z_n φ̂
    const Complex an = e_n * alpha[idx - 1];
    const Complex bn = e_n * beta[idx - 1];
    e_r += an * s.cos_phi * static_cast<Real>(n * (n + 1)) * s.sin_theta * pi_n * zn / rho;
    e_theta += s.cos_phi * (an * tau_n * dzn + bn * pi_n * zn);
    e_phi -= s.sin_phi * (an * pi_n * dzn + bn * tau_n * zn);
    i_power *= kI;
  }
  return to_cartesian(s, e_r, e_theta, e_phi);
}

}  // namespace

MieSphere mie_sphere(Real k, Real radius, Complex eps_r, Real background_index, int max_order) {
  if (!(k > 0) || !(radius > 0) || !(background_index > 0) || std::imag(eps_r) < 0 ||
      max_order == 0) {
    throw InvalidArgument(fmt::format(
        "mie_sphere: k = {}, radius = {}, background index = {} must be positive, Im eps_r = {} "
        "non-negative and max_order = {} non-zero",
        k, radius, background_index, std::imag(eps_r), max_order));
  }
  MieSphere s;
  s.k = k;
  s.radius = radius;
  s.eps_r = eps_r;
  s.background_index = background_index;
  Complex m = std::sqrt(eps_r) / background_index;
  if (std::imag(m) < 0) m = -m;  // absorbing branch
  s.m = m;
  const Real x = k * radius;
  if (max_order < 0) max_order = static_cast<int>(std::ceil(x + 4.0 * std::cbrt(x) + 10.0));
  const auto n_max = static_cast<std::size_t>(max_order);
  // Riccati–Bessel functions of the real argument x, orders 0..N
  std::vector<Complex> psi(n_max + 1);
  std::vector<Complex> xi(n_max + 1);
  for (std::size_t n = 0; n <= n_max; ++n) {
    psi[n] = Complex{x * spherical_bessel_j(static_cast<int>(n), x), 0.0};
    xi[n] = x * spherical_hankel1(static_cast<int>(n), x);
  }
  const Complex mx = m * x;
  const std::vector<Complex> jm = spherical_bessel_j(max_order, mx);
  s.a.resize(n_max);
  s.b.resize(n_max);
  s.c.resize(n_max);
  s.d.resize(n_max);
  for (std::size_t n = 1; n <= n_max; ++n) {
    const Real nn = static_cast<Real>(n);
    const Complex psi_m = mx * jm[n];                    // psi_n(mx)
    const Complex dpsi_m = mx * jm[n - 1] - nn * jm[n];  // psi_n'(mx)
    const Complex dn = dpsi_m / psi_m;                   // D_n(mx)
    const Complex dpsi = psi[n - 1] - nn / x * psi[n];   // psi_n'(x)
    const Complex dxi = xi[n - 1] - nn / x * xi[n];      // xi_n'(x)
    const Complex ta = dn / m + nn / x;
    const Complex tb = m * dn + nn / x;
    s.a[n - 1] = (ta * psi[n] - psi[n - 1]) / (ta * xi[n] - xi[n - 1]);
    s.b[n - 1] = (tb * psi[n] - psi[n - 1]) / (tb * xi[n] - xi[n - 1]);
    // c_n = m W / (psi_n(mx) xi_n'(x) − m xi_n(x) psi_n'(mx)),
    // d_n = m W / (m psi_n(mx) xi_n'(x) − xi_n(x) psi_n'(mx)), W = psi_n xi_n' − xi_n psi_n' = i
    const Complex wronskian = psi[n] * dxi - xi[n] * dpsi;
    s.c[n - 1] = m * wronskian / (psi_m * dxi - m * xi[n] * dpsi_m);
    s.d[n - 1] = m * wronskian / (m * psi_m * dxi - xi[n] * dpsi_m);
  }
  return s;
}

Real MieSphere::scattering_efficiency() const {
  Real sum = 0;
  for (std::size_t n = 0; n < a.size(); ++n) {
    sum += static_cast<Real>(2 * (n + 1) + 1) * (std::norm(a[n]) + std::norm(b[n]));
  }
  const Real x = size_parameter();
  return 2.0 / (x * x) * sum;
}

Real MieSphere::extinction_efficiency() const {
  Real sum = 0;
  for (std::size_t n = 0; n < a.size(); ++n) {
    sum += static_cast<Real>(2 * (n + 1) + 1) * std::real(a[n] + b[n]);
  }
  const Real x = size_parameter();
  return 2.0 / (x * x) * sum;
}

Real MieSphere::absorption_efficiency() const {
  return extinction_efficiency() - scattering_efficiency();
}

Real MieSphere::scattering_cross_section() const {
  return scattering_efficiency() * std::numbers::pi * radius * radius;
}
Real MieSphere::extinction_cross_section() const {
  return extinction_efficiency() * std::numbers::pi * radius * radius;
}
Real MieSphere::absorption_cross_section() const {
  return absorption_efficiency() * std::numbers::pi * radius * radius;
}

assembly::ComplexVector<3> MieSphere::incident_field(const Point<3>& x) const {
  return assembly::ComplexVector<3>(std::exp(kI * k * x(2)), Complex{0.0, 0.0}, Complex{0.0, 0.0});
}

assembly::ComplexVector<3> MieSphere::scattered_field(const Point<3>& x) const {
  const Spherical s = spherical(x);
  const Real rho = k * s.r;
  const auto n_max = static_cast<std::size_t>(max_order());
  std::vector<Complex> h(n_max + 1);
  for (std::size_t n = 0; n <= n_max; ++n) h[n] = spherical_hankel1(static_cast<int>(n), rho);
  std::vector<Complex> alpha(n_max);
  std::vector<Complex> beta(n_max);
  for (std::size_t n = 0; n < n_max; ++n) {
    alpha[n] = kI * a[n];  // i a_n N_e1n
    beta[n] = -b[n];       // − b_n M_o1n
  }
  return harmonic_sum(s, Complex{rho, 0.0}, h, alpha, beta);
}

assembly::ComplexVector<3> MieSphere::internal_field(const Point<3>& x) const {
  const Spherical s = spherical(x);
  // the expansion is regular at the centre; evaluate the ratios z_n / ρ slightly off r = 0
  const Real r = std::max(s.r, 1e-9 * radius);
  const Complex rho = m * k * r;
  const auto n_max = static_cast<std::size_t>(max_order());
  const std::vector<Complex> j = spherical_bessel_j(max_order(), rho);
  std::vector<Complex> alpha(n_max);
  std::vector<Complex> beta(n_max);
  for (std::size_t n = 0; n < n_max; ++n) {
    alpha[n] = -kI * d[n];  // − i d_n N_e1n
    beta[n] = c[n];         // c_n M_o1n
  }
  return harmonic_sum(s, rho, j, alpha, beta);
}

assembly::ComplexVector<3> MieSphere::total_field(const Point<3>& x) const {
  if (x.norm() < radius) return internal_field(x);
  return incident_field(x) + scattered_field(x);
}

}  // namespace hpfem::physics

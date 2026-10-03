#pragma once
/// @file mie.hpp
/// Mie series: the infinite dielectric cylinder (2D) under a plane wave with in-plane electric
/// field (H_z polarisation, the field of `Scattering<2>`) and the sphere (3D) with complex
/// permittivity in a lossless background (Bohren & Huffman, Absorption and Scattering of Light
/// by Small Particles, chapter 4). Reference solutions of convergence test #4 and of the M10
/// validation benchmark C. Convention exp(−iωt): outgoing waves carry @f$ H_n^{(1)} @f$ /
/// @f$ h_n^{(1)} @f$. See docs/theory/maxwell.md#scattering-problems.

#include <vector>

#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/types.hpp"

namespace hpfem::physics {

/// Scattering coefficients @f$ c_n @f$, @f$ n = 0..\text{max\_order} @f$, of the scattered field
/// @f$ H_z^{s} = \sum_n c_n H_n^{(1)}(kr)e^{in\varphi} @f$ for the incident @f$ H_z = e^{ikx} @f$
/// on a lossless cylinder of radius R and refractive index n_c in vacuum:
/// @f$ c_n = i^n\,\frac{n_c J_n'(kR)J_n(n_c kR) - J_n(kR)J_n'(n_c kR)}
///                    {H_n^{(1)}(kR)J_n'(n_c kR) - n_c H_n^{(1)\prime}(kR)J_n(n_c kR)} @f$,
/// with @f$ c_{-n} = c_n @f$ up to the sign of @f$ (-1)^n i^{-2n} @f$ (same modulus).
/// @throws InvalidArgument for k, R, n_c ≤ 0 or max_order < 0.
[[nodiscard]] std::vector<Complex> mie_cylinder_coefficients(Real k, Real radius,
                                                             Real refractive_index, int max_order);

/// Scattering width @f$ \sigma_{sca} = \frac{4}{k}\big(|c_0|^2 + 2\sum_{n\ge1}|c_n|^2\big) @f$ [m]:
/// scattered power per unit length divided by the incident intensity. `max_order` defaults to
/// a value converged for @f$ kR \lesssim 20 @f$.
[[nodiscard]] Real mie_cylinder_scattering_width(Real k, Real radius, Real refractive_index,
                                                 int max_order = -1);

// --- sphere -------------------------------------------------------------------------------------

/// Mie series of a homogeneous sphere of radius a and relative permittivity ε_r (complex,
/// Im ε_r ≥ 0 for absorption) centred at the origin in a lossless background of refractive index
/// n_b, illuminated by the plane wave @f$ E^{inc} = \hat x\, e^{i k z} @f$ of unit amplitude
/// (1 V/m), @f$ k = n_b k_0 @f$. Non-magnetic media. Size parameter @f$ x = k a @f$, relative
/// index @f$ m = \sqrt{\varepsilon_r} / n_b @f$ (branch with Im m ≥ 0).
///
/// Coefficients (Bohren & Huffman 4.53, μ = μ₁) with the Riccati–Bessel functions
/// @f$ \psi_n(\rho) = \rho j_n(\rho) @f$, @f$ \xi_n(\rho) = \rho h_n^{(1)}(\rho) @f$ and the
/// logarithmic derivative @f$ D_n(mx) = \psi_n'(mx)/\psi_n(mx) @f$ (downward recurrence):
/// @f$ a_n = \frac{(D_n/m + n/x)\psi_n(x) - \psi_{n-1}(x)}{(D_n/m + n/x)\xi_n(x) - \xi_{n-1}(x)}
/// @f$,
/// @f$ b_n = \frac{(m D_n + n/x)\psi_n(x) - \psi_{n-1}(x)}{(m D_n + n/x)\xi_n(x) - \xi_{n-1}(x)}
/// @f$; the internal coefficients c_n, d_n follow from the same boundary conditions. Efficiencies
/// @f$ Q_{sca} = \frac{2}{x^2}\sum_n (2n+1)(|a_n|^2 + |b_n|^2) @f$,
/// @f$ Q_{ext} = \frac{2}{x^2}\sum_n (2n+1)\,\mathrm{Re}(a_n + b_n) @f$ (optical theorem),
/// @f$ Q_{abs} = Q_{ext} - Q_{sca} @f$; cross-sections @f$ \sigma = Q\,\pi a^2 @f$ [m²].
/// The fields are the vector spherical harmonic expansions of Bohren & Huffman (4.40, 4.45,
/// 4.50): @f$ E^{sca} = \sum_n E_n (i a_n N^{(3)}_{e1n} - b_n M^{(3)}_{o1n}) @f$ outside and
/// @f$ E^{int} = \sum_n E_n (c_n M^{(1)}_{o1n} - i d_n N^{(1)}_{e1n}) @f$ inside,
/// @f$ E_n = i^n (2n+1)/(n(n+1)) @f$.
struct MieSphere {
  Real k = 0;                 ///< background wavenumber @f$ n_b k_0 @f$ [1/m]
  Real radius = 0;            ///< a [m]
  Complex eps_r{1.0, 0.0};    ///< relative permittivity of the sphere
  Real background_index = 1;  ///< n_b
  Complex m{1.0, 0.0};        ///< relative refractive index
  std::vector<Complex> a, b;  ///< scattering coefficients, order n at index n − 1
  std::vector<Complex> c, d;  ///< internal-field coefficients, order n at index n − 1

  [[nodiscard]] int max_order() const noexcept { return static_cast<int>(a.size()); }
  [[nodiscard]] Real size_parameter() const noexcept { return k * radius; }

  [[nodiscard]] Real scattering_efficiency() const;
  [[nodiscard]] Real extinction_efficiency() const;
  [[nodiscard]] Real absorption_efficiency() const;
  [[nodiscard]] Real scattering_cross_section() const;  ///< [m²]
  [[nodiscard]] Real extinction_cross_section() const;  ///< [m²]
  [[nodiscard]] Real absorption_cross_section() const;  ///< [m²]

  /// Incident field @f$ \hat x\, e^{ikz} @f$ at x (any point).
  [[nodiscard]] assembly::ComplexVector<3> incident_field(const Point<3>& x) const;
  /// Scattered field at x; valid outside the sphere (|x| ≥ a).
  [[nodiscard]] assembly::ComplexVector<3> scattered_field(const Point<3>& x) const;
  /// Field inside the sphere at x (|x| ≤ a).
  [[nodiscard]] assembly::ComplexVector<3> internal_field(const Point<3>& x) const;
  /// Total field: internal field inside, incident + scattered outside.
  [[nodiscard]] assembly::ComplexVector<3> total_field(const Point<3>& x) const;
};

/// Builds the series. `max_order` < 0 selects the Wiscombe-type cut-off
/// @f$ N = \lceil x + 4 x^{1/3} + 10 \rceil @f$ (converged for the efficiencies and for the
/// fields away from the surface; the near field at the surface may need more terms).
/// @throws InvalidArgument for k, radius, background_index ≤ 0, Im ε_r < 0 or max_order = 0.
[[nodiscard]] MieSphere mie_sphere(Real k, Real radius, Complex eps_r, Real background_index = 1.0,
                                   int max_order = -1);

}  // namespace hpfem::physics

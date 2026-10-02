#pragma once
/// @file mie.hpp
/// Mie series for the infinite dielectric cylinder (2D) under a plane wave with in-plane
/// electric field (H_z polarisation, the field of `Scattering<2>`): scattering coefficients
/// and the scattering width (cross-section per unit length). Reference solution of
/// convergence test #4. See docs/theory/maxwell.md#scattering-problems.

#include <vector>

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

}  // namespace hpfem::physics

#pragma once
/// @file dipole_emission.hpp
/// Dipole emitters in periodic structures (M17, ADR-0013): the source of one cell problem of the
/// array scanning and the power it delivers. A point dipole of current moment p [A m] at
/// @f$ x_0 @f$, smeared over the normalised 3D Gaussian of M11,
/// @f$ g = e^{-|x - x_0|^2/2\sigma^2}/((2\pi)^{3/2}\sigma^3) @f$, decomposes over the Bloch
/// wavenumber @f$ k_x @f$ and the longitudinal wavenumber β into the cell sources
/// @f[ J_{k_x,\beta}(x, y) = p\, g_2(x - x_0, y - y_0)\, e^{-\sigma^2\beta^2/2}, \qquad
///     g_2 = e^{-\rho^2/2\sigma^2}/(2\pi\sigma^2), @f]
/// one per cell with the Bloch phase @f$ e^{ik_xP} @f$ (the z-smearing is the Fourier factor of
/// the β sample). The single dipole is recovered by
/// @f$ E = \frac{P}{2\pi}\int_{BZ}dk_x\,\frac1{2\pi}\int d\beta\,E_{k_x,\beta}\,e^{i\beta z} @f$,
/// its power by the same integral of the cell power
/// @f$ P_{cell} = -\tfrac12\mathrm{Re}\int_{cell} \bar J_{k_x,\beta}\cdot E_{k_x,\beta}\,dA @f$
/// [W/m per unit β]. Convention exp(-iωt). See docs/theory/maxwell.md#dipole-emitters.

#include <vector>

#include <Eigen/Dense>

#include "hpfem/core/types.hpp"
#include "hpfem/physics/conical_scattering.hpp"

namespace hpfem::physics {

/// Volume source @f$ f = i\omega\mu_0 J_{k_x,\beta} @f$ of the cell problem of a Gaussian
/// dipole (file comment) in the scaled components @f$ (f_x, f_y, -i f_z) @f$ of the conical
/// solver (`ConicalScatteringSetup::current`, total-field formulation): `position`
/// @f$ (x_0, y_0) @f$ [m], `moment` the physical current moment @f$ (p_x, p_y, p_z) @f$ [A m] in
/// the solver frame (x along the period, y normal, z along the lines), `sigma` the smearing
/// [m], `omega` [rad/s], `beta` the longitudinal wavenumber of the cell problem [1/m]. The
/// Bloch images of the Gaussian in the neighbouring cells are not included: keep it 6σ inside
/// the cell (`grating.emit` checks it).
/// @throws InvalidArgument for σ ≤ 0 or ω ≤ 0.
[[nodiscard]] ConicalField conical_gaussian_dipole(const Point<2>& position,
                                                   const ConicalVector& moment, Real sigma,
                                                   Real omega, Real beta);

/// Power delivered by the source of a total-field conical problem,
/// @f$ -\tfrac12\mathrm{Re}\int \bar J\cdot E\,dA @f$ with @f$ J = f/(i\omega\mu_0) @f$ of
/// `ConicalScatteringSetup::current` and the physical field @f$ (E_x, E_y, E_z = iv) @f$ of the
/// solution, by quadrature of degree 2p + `extra_order` on every cell [W/m; per unit β for a
/// cell problem of the array scanning]. For a lossless structure it equals the power leaving
/// through the PML boundaries plus the guided power; with losses the absorbed power as well.
/// @throws InvalidArgument without a current (scattered-field problem) or for mismatched
///         vectors.
[[nodiscard]] Real conical_source_power(const ConicalScattering& problem,
                                        const ConicalSolution& solution, int extra_order = 4);

/// Responses of a cell problem to several Gaussian dipoles on the factorisation its solve kept
/// (`ConicalScatteringSetup::keep_factorisation`, M16 S1): for each row of `moments` (k × 3,
/// physical current moments [A m]) the solution of the same cell problem (kx, β, materials,
/// PML) with the source `conical_gaussian_dipole(position, moment, sigma, ω, β)` — one load
/// assembly on the cells within 7σ of the position and one solve each, no factorisation (the
/// three orientations of an isotropic emitter share one; ADR-0013 §5).
/// @throws InvalidArgument without a kept factorisation, for σ ≤ 0 or a moment row that is not
///         a 3-vector.
[[nodiscard]] std::vector<ConicalSolution> conical_dipole_responses(
    const ConicalScattering& problem, const ConicalSolution& solution, const Point<2>& position,
    Real sigma, const Matrix& moments);

/// The power matrix of the Gaussian dipole in a cell problem: with the responses
/// @f$ E_j @f$ to the unit moments @f$ \hat e_x, \hat e_y, \hat e_z @f$,
/// @f$ A_{ij} = -\tfrac12\int g_2\,e^{-\sigma^2\beta^2/2}\,(E_j)_i\,dA @f$, so that the power
/// delivered by any moment p is @f$ P_{cell}(p) = \mathrm{Re}(p^H A\,p) @f$ [W/m per unit β].
/// @throws InvalidArgument unless `unit_responses` holds the three unit-moment responses.
[[nodiscard]] Eigen::Matrix3cd conical_dipole_power_matrix(
    const ConicalScattering& problem, const std::vector<ConicalSolution>& unit_responses,
    const Point<2>& position, Real sigma, int extra_order = 4);

}  // namespace hpfem::physics

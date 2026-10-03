#pragma once
/// @file sources.hpp
/// Analytic incident fields in a homogeneous background with real wavenumber k: plane
/// waves and point (3D) / line (2D) dipoles, given as value and curl so that they serve as
/// Dirichlet data, as scattered-field sources and as reference solutions. All solve
/// @f$ \nabla\times\nabla\times E - k^2 E = 0 @f$ away from the source with the exp(−iωt)
/// convention (outgoing waves). See docs/theory/maxwell.md#scattering-problems.

#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/constants.hpp"
#include "hpfem/core/types.hpp"

namespace hpfem::physics {

/// An analytic field: value and curl at physical points.
template <int Dim>
struct IncidentField {
  assembly::ComplexVectorField<Dim> value;
  assembly::ComplexCurlField<Dim> curl;
  [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(value); }
};

/// Vacuum wavenumber @f$ k_0 = \omega / c_0 @f$ [1/m] of the angular frequency ω [rad/s].
[[nodiscard]] inline Real vacuum_wavenumber(Real omega) noexcept {
  return omega / constants::c0;
}

/// Plane wave @f$ E_0 e^{i k\cdot x} @f$ with real wave vector k [1/m] and complex amplitude
/// @f$ E_0 \perp k @f$ (any polarisation, including circular via complex components).
/// @throws InvalidArgument if the amplitude is not transverse or k = 0.
template <int Dim>
[[nodiscard]] IncidentField<Dim> plane_wave(const assembly::ComplexVector<Dim>& amplitude,
                                            const Point<Dim>& wave_vector);

/// Field of a point dipole (3D) or line dipole (2D, in-plane moment) at x0: the outgoing
/// solution of @f$ \nabla\times\nabla\times E - k^2 E = p\,\delta(x - x_0) @f$,
/// @f$ E = (I + \nabla\nabla / k^2)\, g\, p @f$ with the scalar Green's function
/// @f$ g = e^{ikr}/(4\pi r) @f$ (3D) or @f$ g = \tfrac{i}{4} H_0^{(1)}(kr) @f$ (2D).
/// Singular at x0; place the source outside the mesh or in a cell to be excised.
template <int Dim>
[[nodiscard]] IncidentField<Dim> dipole_field(const Point<Dim>& position,
                                              const assembly::ComplexVector<Dim>& moment, Real k);

/// Volume source @f$ f = i\omega\mu_0 J @f$ of a dipole of moment density `moment` [A m / m^Dim
/// per unit ...] smeared over the normalised Gaussian @f$ g(x) = \exp(-|x - x_0|^2 / 2\sigma^2)
/// / ((2\pi)^{Dim/2}\sigma^{Dim}) @f$: a point (3D) or line (2D) dipole resolved by the mesh
/// (choose @f$ \sigma @f$ a fraction of the cell size and well below the wavelength), for the
/// total-field formulation (`ScatteringSetup::current`) where the analytic `dipole_field`
/// cannot serve as incident field, e.g. an emitter inside a layered structure. Evaluated in
/// C++, so it costs nothing in the parallel assembly. @throws InvalidArgument for σ ≤ 0.
template <int Dim>
[[nodiscard]] assembly::ComplexVectorField<Dim> gaussian_current(
    const Point<Dim>& position, const assembly::ComplexVector<Dim>& moment, Real sigma, Real omega);

extern template IncidentField<2> plane_wave<2>(const assembly::ComplexVector<2>&, const Point<2>&);
extern template IncidentField<3> plane_wave<3>(const assembly::ComplexVector<3>&, const Point<3>&);
extern template IncidentField<2> dipole_field<2>(const Point<2>&, const assembly::ComplexVector<2>&,
                                                 Real);
extern template IncidentField<3> dipole_field<3>(const Point<3>&, const assembly::ComplexVector<3>&,
                                                 Real);

extern template assembly::ComplexVectorField<2> gaussian_current<2>(
    const Point<2>&, const assembly::ComplexVector<2>&, Real, Real);
extern template assembly::ComplexVectorField<3> gaussian_current<3>(
    const Point<3>&, const assembly::ComplexVector<3>&, Real, Real);

}  // namespace hpfem::physics

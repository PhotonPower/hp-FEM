#pragma once
/// @file diffraction.hpp
/// Diffraction orders of Bloch-periodic structures (2D): on a line @f$ x = x_0 @f$ in a
/// homogeneous region the field is a sum of plane waves @f$ E = \sum_m A_m\,e^{i k_{y,m} y} @f$
/// with @f$ k_{y,m} = k_{y,0} + 2\pi m/a @f$, and the Fourier coefficients give the
/// reflected / transmitted diffraction efficiencies @f$ \eta_m = \mathrm{Re}(k_{x,m})\,|A_m|^2 /
/// (k_{x,0}^{inc}|E_0|^2) @f$ with @f$ k_{x,m} = \sqrt{k^2 n^2 - k_{y,m}^2} @f$. See
/// docs/theory/maxwell.md#post-processing-quantities.

#include <functional>
#include <vector>

#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/point_location.hpp"

namespace hpfem::physics {

/// Fourier coefficients @f$ A_m = \frac1a\int_{y_0}^{y_0+a} E(x_0, y)\,e^{-ik_{y,m}y}\,dy @f$ for
/// @f$ m = -\text{max\_order}..\text{max\_order} @f$ (index @f$ m + \text{max\_order} @f$) of a
/// field given as a function of the physical point, sampled with `num_points` Gauss–Legendre points
/// per period (choose a multiple of the cells across the period, e.g. 8 per cell).
[[nodiscard]] std::vector<assembly::ComplexVector<2>> fourier_coefficients(
    const std::function<assembly::ComplexVector<2>(const Point<2>&)>& field, Real x0, Real y0,
    Real period, Real ky0, int max_order, int num_points);

/// The same for a discrete field located with `locator`; points outside the mesh are an
/// error.
[[nodiscard]] std::vector<assembly::ComplexVector<2>> fourier_coefficients(
    const fespace::NedelecDofMap<2>& dofs, const Vector& e_h, const mesh::PointLocator<2>& locator,
    Real x0, Real y0, Real period, Real ky0, int max_order, int num_points);

/// One diffraction order.
struct DiffractionOrder {
  int order = 0;
  Real ky = 0;  ///< tangential wavenumber [1/m]
  Complex kx;   ///< normal wavenumber in the medium of the line (imaginary: evanescent)
  bool propagating = false;
  Real efficiency = 0;  ///< fraction of the incident power, 0 for evanescent orders
};

/// Efficiencies of the orders of `coefficients` (from `fourier_coefficients` with the same
/// `max_order`) in a medium of real refractive index `index_line` on the line, for an
/// incident plane wave of amplitude |E0| with normal wavenumber `kx_incident` (> 0) in the
/// incidence medium: @f$ \eta_m = \mathrm{Re}(k_{x,m})|A_m|^2 / (k_x^{inc}|E_0|^2) @f$ (equal
/// permeability on both sides).
[[nodiscard]] std::vector<DiffractionOrder> diffraction_efficiencies(
    const std::vector<assembly::ComplexVector<2>>& coefficients, Real k0, Real index_line,
    Real period, Real ky0, Real kx_incident, Real incident_amplitude);

}  // namespace hpfem::physics

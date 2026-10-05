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

/// Where and how the orders are taken: a line of one period in a homogeneous region, in any
/// orientation. The phase reference of the amplitudes is `origin`.
struct OrderLine {
  Point<2> origin;   ///< a point of the line
  Point<2> tangent;  ///< unit vector along the period
  Point<2> normal;   ///< unit normal pointing away from the structure (towards the orders)
  Real period = 0;
};

/// One diffraction order with its complex vector amplitude.
struct DiffractionOrderField {
  int order = 0;
  Real k_tangential = 0;  ///< k_t + 2π m / period [1/m]
  Complex kn;             ///< normal wavenumber in the medium of the line (imaginary: evanescent)
  bool propagating = false;
  assembly::ComplexVector<2> amplitude;  ///< @f$ A_m @f$, phase reference at `line.origin`
  Real efficiency = 0;  ///< Re(kn) |A_m|^2 / (kn_incident |E0|^2), 0 for evanescent orders
};

/// A field as a function of the point, e.g. `problem.total_field` through a locator.
using FieldFunction = std::function<assembly::ComplexVector<2>(const Point<2>&)>;

/// Orders of `field - incident` on the line (the reflected orders of a total field, with
/// the incident wave `incident` given, e.g. `Scattering::incident_wave`), or of `field`
/// itself when `incident` is empty (a scattered field above, the transmitted field below):
/// @f$ A_m = \frac{1}{a}\int_0^a (E - E^{inc})(o + t\,\hat t)\,e^{-i k_{t,m} t}\,dt @f$ with
/// @f$ k_{t,m} = k_t + 2\pi m / a @f$, by composite Gauss-Legendre with `num_points` points in
/// blocks of up to 8 (0: 16 per order, at least 64; exact for the piecewise polynomial FEM
/// field when the blocks align with the cells, e.g. 8 points per cell on the line).
/// `k_tangential` is the Bloch wavenumber of the incident wave along the tangent,
/// `index_line` the real index of the medium on the line, `kn_incident` the normal
/// wavenumber of the incident wave (> 0) in the incidence medium and `incident_amplitude`
/// its |E0|. @throws InvalidArgument for a degenerate line, non-positive parameters or a
/// point outside the mesh (the field function throws or returns no value).
[[nodiscard]] std::vector<DiffractionOrderField> diffraction_orders(
    const FieldFunction& field, const OrderLine& line, Real k0, Real index_line, Real k_tangential,
    Real kn_incident, const FieldFunction& incident = {}, int max_order = 3, int num_points = 0,
    Real incident_amplitude = 1.0);

}  // namespace hpfem::physics

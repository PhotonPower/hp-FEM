#pragma once
/// @file diffraction.hpp
/// Diffraction orders of Bloch-periodic structures. 2D: on a line @f$ x = x_0 @f$ in a
/// homogeneous region the field is a sum of plane waves @f$ E = \sum_m A_m\,e^{i k_{y,m} y} @f$
/// with @f$ k_{y,m} = k_{y,0} + 2\pi m/a @f$, and the Fourier coefficients give the
/// reflected / transmitted diffraction efficiencies @f$ \eta_m = \mathrm{Re}(k_{x,m})\,|A_m|^2 /
/// (k_{x,0}^{inc}|E_0|^2) @f$ with @f$ k_{x,m} = \sqrt{k^2 n^2 - k_{y,m}^2} @f$. 3D (crossed
/// gratings, M21): the same on a plane over one cell of the lattice, orders (m, n). See
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

/// A field of the 3D solver as a function of the point, e.g. `problem.total_field` through a
/// locator.
using FieldFunction3D = std::function<assembly::ComplexVector<3>(const Point<3>&)>;

/// Where the orders of a crossed (doubly periodic) grating are taken: a plane in a
/// homogeneous region, spanned by the two lattice vectors of the periodicity (any angle
/// between them). The phase reference of the amplitudes is `origin`.
struct OrderPlane {
  Point<3> origin = Point<3>::Zero();  ///< a point of the plane
  Point<3> a1 = Point<3>::Zero();      ///< first lattice vector [m], in the plane
  Point<3> a2 = Point<3>::Zero();      ///< second lattice vector [m], in the plane
  /// Unit normal pointing away from the structure (towards the orders).
  Point<3> normal = Point<3>::UnitZ();
};

/// One order (m, n) of a crossed grating with its complex vector amplitude.
struct DiffractionOrder3D {
  int m = 0;  ///< order along a1
  int n = 0;  ///< order along a2
  /// Tangential wavevector @f$ \mathbf k_t + m\mathbf b_1 + n\mathbf b_2 @f$ [1/m] with the
  /// reciprocal vectors @f$ \mathbf a_i\cdot\mathbf b_j = 2\pi\delta_{ij} @f$ in the plane.
  Point<3> k_tangential = Point<3>::Zero();
  Complex kn;  ///< normal wavenumber in the medium of the plane (imaginary: evanescent)
  bool propagating = false;
  assembly::ComplexVector<3> amplitude = assembly::ComplexVector<3>::Zero();  ///< @f$ A_{mn} @f$
  Real efficiency = 0;  ///< Re(kn) |A_mn|^2 / (kn_incident |E0|^2), 0 for evanescent orders
  /// Unit propagation direction @f$ (\mathbf k_t + \mathrm{Re}(k_n)\hat n)/|\cdot| @f$ of a
  /// propagating order (meaningless for an evanescent one).
  [[nodiscard]] Point<3> direction(const Point<3>& normal) const {
    const Point<3> k = k_tangential + kn.real() * normal;
    return k / k.norm();
  }
};

/// Orders (m, n), |m|, |n| ≤ `max_order`, of `field - incident` on the plane (of `field`
/// itself when `incident` is empty), in the order m = -max_order … max_order, n = -max_order
/// … max_order (n fastest): @f$ A_{mn} = \int_0^1\!\int_0^1 (E - E^{inc})(o + s\mathbf a_1 +
/// t\mathbf a_2)\,e^{-i\mathbf k_{mn}\cdot(s\mathbf a_1 + t\mathbf a_2)}\,ds\,dt @f$ with
/// @f$ \mathbf k_{mn} = \mathbf k_t + m\mathbf b_1 + n\mathbf b_2 @f$, by a tensor composite
/// Gauss–Legendre rule with `num_points` points per lattice direction in blocks of up to 8
/// (0: 8 per order, at least 32; exact for the piecewise polynomial FEM field when the blocks
/// align with the cells). @f$ k_n = \sqrt{k_0^2 n^2 - |\mathbf k_{mn}|^2} @f$ with Im ≥ 0 in the
/// medium of real index `index_plane`; @f$ \eta_{mn} = \mathrm{Re}(k_n)|A_{mn}|^2 /
/// (k_n^{inc}|E_0|^2) @f$ (equal permeability on both sides). `k_tangential` is the tangential
/// wavevector of the incident wave (in the plane), `kn_incident` its normal wavenumber (> 0) in
/// the incidence medium and `incident_amplitude` its |E0|. Cost: num_points² evaluations of
/// the field. @throws InvalidArgument for a degenerate lattice, lattice vectors or a
/// `k_tangential` not in the plane, a non-unit normal, non-positive parameters or a point
/// outside the mesh (the field function throws).
[[nodiscard]] std::vector<DiffractionOrder3D> diffraction_orders(
    const FieldFunction3D& field, const OrderPlane& plane, Real k0, Real index_plane,
    const Point<3>& k_tangential, Real kn_incident, const FieldFunction3D& incident = {},
    int max_order = 2, int num_points = 0, Real incident_amplitude = 1.0);

}  // namespace hpfem::physics

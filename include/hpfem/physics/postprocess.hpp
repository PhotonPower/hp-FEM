#pragma once
/// @file postprocess.hpp
/// Post-processing of time-harmonic fields: oriented surfaces made of mesh facets with a
/// quadrature on them, the time-averaged Poynting flux @f$ \int_S \tfrac12\,\mathrm{Re}(E\times
/// \bar H)\cdot n\,ds @f$ with @f$ H = (i\omega\mu_0\mu_r)^{-1}\nabla\times E @f$, the absorbed
/// power @f$ \tfrac{\omega\varepsilon_0}{2}\int \mathrm{Im}(\varepsilon_r)|E|^2 @f$ and the
/// scattering / absorption / extinction cross-sections of a `Scattering` solution. SI units:
/// E in V/m, powers in W (3D) or W/m (2D, per unit length), cross-sections in m^2 (3D) or m
/// (2D). See docs/theory/maxwell.md#post-processing-quantities.

#include <functional>
#include <vector>

#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/mesh.hpp"
#include "hpfem/physics/scattering.hpp"
#include "hpfem/physics/sources.hpp"

namespace hpfem::physics {

/// An oriented surface: mesh facets, each with the cell on its inside; the normal points out
/// of that cell. Interior facets give interfaces and flux surfaces, boundary facets the
/// domain boundary.
template <int Dim>
struct Surface {
  struct Facet {
    Index facet;
    Index inside_cell;
  };
  std::vector<Facet> facets;

  /// Closed surface around the cells carrying `cell_tag`: every facet between a tagged and
  /// an untagged (or differently tagged) cell, plus tagged cells' boundary facets; the
  /// inside is the tagged cell, so the normal points out of the region. On a locally
  /// refined mesh the hanging child facets are used (the inside cell may then be the coarse
  /// cell behind the parent facet).
  [[nodiscard]] static Surface around_cells(const mesh::Mesh<Dim>& mesh, mesh::Tag cell_tag);
  /// Boundary facets carrying `facet_tag`, normal out of the domain.
  [[nodiscard]] static Surface boundary(const mesh::Mesh<Dim>& mesh, mesh::Tag facet_tag);
  /// The whole domain boundary, normal out of the domain.
  [[nodiscard]] static Surface whole_boundary(const mesh::Mesh<Dim>& mesh);
};

/// A quadrature point on a surface: physical point, unit normal out of the inside cell,
/// weight including the surface measure, and the inside cell with reference coordinates.
template <int Dim>
struct SurfacePoint {
  Point<Dim> x;
  Point<Dim> normal;
  Real weight = 0;
  Index cell = kInvalidIndex;
  Point<Dim> xi;
};

/// Quadrature of the given degree on every facet of the surface (Gauss–Legendre on edges,
/// collapsed Gauss–Jacobi on faces), mapped through the cell geometry, so curved facets get
/// their curved measure and normals.
template <int Dim>
[[nodiscard]] std::vector<SurfacePoint<Dim>> surface_quadrature(const mesh::Mesh<Dim>& mesh,
                                                                const Surface<Dim>& surface,
                                                                int order);

/// Field sampler: value and curl of a field at a surface point.
template <int Dim>
using SurfaceField = std::function<void(const SurfacePoint<Dim>&, assembly::ComplexVector<Dim>&,
                                        assembly::ComplexCurl<Dim>&)>;

/// Sampler of a discrete field (coefficients on the DoF map).
template <int Dim>
[[nodiscard]] SurfaceField<Dim> discrete_field(const fespace::NedelecDofMap<Dim>& dofs,
                                               const Vector& e_h);
/// Sampler of an analytic field.
template <int Dim>
[[nodiscard]] SurfaceField<Dim> analytic_field(const IncidentField<Dim>& field);
/// Sampler of the sum a + factor * b.
template <int Dim>
[[nodiscard]] SurfaceField<Dim> combined_field(const SurfaceField<Dim>& a,
                                               const SurfaceField<Dim>& b, Complex factor);

/// Time-averaged power flux @f$ \int_S \tfrac12\,\mathrm{Re}(E\times\bar H)\cdot n\,ds @f$ of the
/// sampled field through the surface, @f$ H = (i\omega\mu_0\mu_r)^{-1}\nabla\times E @f$ with the
/// permeability of the inside cell. [W] in 3D, [W/m] in 2D.
template <int Dim>
[[nodiscard]] Real poynting_flux(const mesh::Mesh<Dim>& mesh, const Surface<Dim>& surface,
                                 const SurfaceField<Dim>& field, Real omega,
                                 const materials::MaterialMap& materials, int order);
/// Flux of a discrete field with the rule degree 2p + extra_order.
template <int Dim>
[[nodiscard]] Real poynting_flux(const fespace::NedelecDofMap<Dim>& dofs, const Vector& e_h,
                                 Real omega, const materials::MaterialMap& materials,
                                 const Surface<Dim>& surface, int extra_order = 2);

/// Absorbed power @f$ \tfrac{\omega\varepsilon_0}{2}\int_\Omega \mathrm{Im}(\varepsilon_r)|E_h|^2
/// @f$ over all cells with lossy material. [W] in 3D, [W/m] in 2D.
template <int Dim>
[[nodiscard]] Real absorbed_power(const fespace::NedelecDofMap<Dim>& dofs, const Vector& e_h,
                                  Real omega, const materials::MaterialMap& materials,
                                  int extra_order = 2);

/// Intensity @f$ |E_0|^2 / (2 Z) @f$ [W/m^2] of a plane wave of amplitude @f$ |E_0| @f$ [V/m] in a
/// medium with impedance @f$ Z = Z_0\sqrt{\mu_r/\varepsilon_r} @f$ (real part).
[[nodiscard]] Real plane_wave_intensity(Real amplitude, const materials::Material& medium);

/// Cross-sections of a scattering solution from the fluxes through a closed surface around
/// the scatterer: @f$ \sigma_{sca} = P_{sca}/I @f$ (scattered field outwards), @f$ \sigma_{abs} =
/// -P_{tot}/I @f$ (total field inwards), @f$ \sigma_{ext} = \sigma_{sca} + \sigma_{abs} @f$,
/// with the incident intensity I of the given amplitude in the background medium.
struct CrossSections {
  Real scattering = 0;
  Real absorption = 0;
  Real extinction = 0;
};
template <int Dim>
[[nodiscard]] CrossSections cross_sections(const Scattering<Dim>& problem,
                                           const ScatteringSolution<Dim>& solution,
                                           const Surface<Dim>& surface, Real incident_amplitude,
                                           int extra_order = 2);

extern template struct Surface<2>;
extern template struct Surface<3>;
extern template std::vector<SurfacePoint<2>> surface_quadrature<2>(const mesh::Mesh<2>&,
                                                                   const Surface<2>&, int);
extern template std::vector<SurfacePoint<3>> surface_quadrature<3>(const mesh::Mesh<3>&,
                                                                   const Surface<3>&, int);
extern template SurfaceField<2> discrete_field<2>(const fespace::NedelecDofMap<2>&, const Vector&);
extern template SurfaceField<3> discrete_field<3>(const fespace::NedelecDofMap<3>&, const Vector&);
extern template SurfaceField<2> analytic_field<2>(const IncidentField<2>&);
extern template SurfaceField<3> analytic_field<3>(const IncidentField<3>&);
extern template SurfaceField<2> combined_field<2>(const SurfaceField<2>&, const SurfaceField<2>&,
                                                  Complex);
extern template SurfaceField<3> combined_field<3>(const SurfaceField<3>&, const SurfaceField<3>&,
                                                  Complex);
extern template Real poynting_flux<2>(const mesh::Mesh<2>&, const Surface<2>&,
                                      const SurfaceField<2>&, Real, const materials::MaterialMap&,
                                      int);
extern template Real poynting_flux<3>(const mesh::Mesh<3>&, const Surface<3>&,
                                      const SurfaceField<3>&, Real, const materials::MaterialMap&,
                                      int);
extern template Real poynting_flux<2>(const fespace::NedelecDofMap<2>&, const Vector&, Real,
                                      const materials::MaterialMap&, const Surface<2>&, int);
extern template Real poynting_flux<3>(const fespace::NedelecDofMap<3>&, const Vector&, Real,
                                      const materials::MaterialMap&, const Surface<3>&, int);
extern template Real absorbed_power<2>(const fespace::NedelecDofMap<2>&, const Vector&, Real,
                                       const materials::MaterialMap&, int);
extern template Real absorbed_power<3>(const fespace::NedelecDofMap<3>&, const Vector&, Real,
                                       const materials::MaterialMap&, int);
extern template CrossSections cross_sections<2>(const Scattering<2>&, const ScatteringSolution<2>&,
                                                const Surface<2>&, Real, int);
extern template CrossSections cross_sections<3>(const Scattering<3>&, const ScatteringSolution<3>&,
                                                const Surface<3>&, Real, int);

}  // namespace hpfem::physics

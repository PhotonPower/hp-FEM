#pragma once
/// @file absorption.hpp
/// Absorbed power of the total field by volumetric quadrature of the Joule heating
/// @f$ Q = \tfrac{\omega\varepsilon_0}{2}\,\mathrm{Im}(\varepsilon_r)\,|E|^2 @f$ on the mesh
/// (M15 F4): per material tag, per cell and per quadrature point, for `Scattering<Dim>` and
/// `ConicalScattering`. Exact for the discrete field where a raster integration of a map is
/// off by a pixel at every material boundary, and free of the interface caveat of the E_z
/// Poynting flux. The total field includes the incident or background field, so the
/// absorption of a lossy layered background is counted. Units: [W] in 3D, [W/m] in 2D
/// (per unit length along z); divide by the incident power per period
/// (`plane_wave_intensity(|E0|, medium) cos θ · period`) for the absorptance.
/// Convention exp(-iωt), Im ε > 0 for loss.
#include <utility>
#include <vector>

#include "hpfem/core/types.hpp"
#include "hpfem/mesh/mesh.hpp"
#include "hpfem/physics/conical_scattering.hpp"
#include "hpfem/physics/scattering.hpp"

namespace hpfem::physics {

/// Absorbed power split by material tag and by cell.
struct AbsorbedPower {
  Real total = 0;
  /// (tag, power) of every tag with lossy cells, ascending tag.
  std::vector<std::pair<mesh::Tag, Real>> by_tag;
  std::vector<Real> per_cell;  ///< one entry per cell, 0 in lossless cells

  /// Power of a tag (0 if the tag has no lossy cells).
  [[nodiscard]] Real of_tag(mesh::Tag tag) const noexcept;
};

/// The Joule heating at the quadrature points of all lossy cells: physical point, quadrature
/// weight including the Jacobian ([m^Dim]), density Q ([W/m^3], [W/m^2] in 2D) and cell;
/// @f$ \sum_q w_q Q_q @f$ is the absorbed power.
template <int Dim>
struct AbsorptionDensity {
  std::vector<Point<Dim>> points;
  std::vector<Real> weights;
  std::vector<Real> density;
  std::vector<Index> cell;

  [[nodiscard]] Real total() const noexcept;
};

/// Absorbed power of the total field of a scattering solution, with rules of degree
/// 2p + `extra_order` (the field is a polynomial of degree p; the extra order resolves the
/// incident field and curved cells).
template <int Dim>
[[nodiscard]] AbsorbedPower absorbed_power_by_tag(const Scattering<Dim>& problem,
                                                  const ScatteringSolution<Dim>& solution,
                                                  int extra_order = 2);

/// The same for the conical solver (|E|^2 = |E_x|^2 + |E_y|^2 + |E_z|^2 of the physical
/// field; the degree is taken from the higher of the two spaces).
[[nodiscard]] AbsorbedPower absorbed_power_by_tag(const ConicalScattering& problem,
                                                  const ConicalSolution& solution,
                                                  int extra_order = 2);

/// The Joule heating at the quadrature points of the lossy cells.
template <int Dim>
[[nodiscard]] AbsorptionDensity<Dim> absorption_density(const Scattering<Dim>& problem,
                                                        const ScatteringSolution<Dim>& solution,
                                                        int extra_order = 2);

/// The same for the conical solver.
[[nodiscard]] AbsorptionDensity<2> absorption_density(const ConicalScattering& problem,
                                                      const ConicalSolution& solution,
                                                      int extra_order = 2);

extern template struct AbsorptionDensity<2>;
extern template struct AbsorptionDensity<3>;
extern template AbsorbedPower absorbed_power_by_tag<2>(const Scattering<2>&,
                                                       const ScatteringSolution<2>&, int);
extern template AbsorbedPower absorbed_power_by_tag<3>(const Scattering<3>&,
                                                       const ScatteringSolution<3>&, int);
extern template AbsorptionDensity<2> absorption_density<2>(const Scattering<2>&,
                                                           const ScatteringSolution<2>&, int);
extern template AbsorptionDensity<3> absorption_density<3>(const Scattering<3>&,
                                                           const ScatteringSolution<3>&, int);

}  // namespace hpfem::physics

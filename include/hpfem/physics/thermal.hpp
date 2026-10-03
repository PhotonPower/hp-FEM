#pragma once
/// @file thermal.hpp
/// Steady heat conduction driven by the absorbed optical power (first multiphysics step):
/// the absorbed power density @f$ q = \tfrac{\omega\varepsilon_0}{2}\,\mathrm{Im}(\varepsilon_r)
/// |E|^2 @f$ [W/m³] of a time-harmonic field (convention exp(−iωt), lossy media have
/// Im εr > 0) is interpolated into the H1 space on the same mesh and drives
/// @f$ -\nabla\cdot(\kappa\nabla T) = q @f$ with the thermal conductivity κ [W/(m K)] per
/// material tag, fixed temperatures on tagged facets (Dirichlet) and adiabatic walls
/// elsewhere (natural). SI units: T in K. See docs/theory/multiphysics.md.
#include <map>
#include <utility>
#include <vector>

#include "hpfem/assembly/h1_forms.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/mesh.hpp"
#include "hpfem/solvers/linear_solver.hpp"

namespace hpfem::physics {

/// Absorbed power density of the discrete field `e_h` (coefficients on `dofs`) as the
/// hierarchical interpolant on the H1 map `h1` of the same mesh: per cell
/// @f$ q = \tfrac{\omega\varepsilon_0}{2}\,\mathrm{Im}(\varepsilon_r)\,|E_h|^2 @f$ with the
/// material of the cell. Lossless cells contribute zero; the interpolant is discontinuous
/// in its derivative at material interfaces, as the density is.
/// @throws InvalidArgument if the maps live on different meshes or `e_h` does not match.
template <int Dim>
[[nodiscard]] Vector absorbed_power_density(const fespace::NedelecDofMap<Dim>& dofs,
                                            const Vector& e_h, Real omega,
                                            const materials::MaterialMap& materials,
                                            const fespace::DofMap<Dim>& h1);

/// Load vector of the absorbed power on the H1 map, @f$ b_i = \int \phi_i\, q @f$ with the
/// density evaluated cell by cell at the quadrature points (exact for the polynomial
/// @f$ |E_h|^2 @f$ on affine cells, discontinuities at material interfaces kept); the sum of
/// its vertex entries (the constant 1 of the hierarchical basis) is the absorbed power. The
/// right-hand side of `Thermal::solve_load`.
/// @throws InvalidArgument if the maps live on different meshes or `e_h` does not match.
template <int Dim>
[[nodiscard]] Vector absorbed_power_load(const fespace::NedelecDofMap<Dim>& dofs, const Vector& e_h,
                                         Real omega, const materials::MaterialMap& materials,
                                         const fespace::DofMap<Dim>& h1, int extra_order = 2);

/// Description of the heat-conduction problem.
struct ThermalSetup {
  Real background_conductivity = 1.0;      ///< κ [W/(m K)] of untagged / unlisted cells
  std::map<mesh::Tag, Real> conductivity;  ///< κ by cell tag
  std::vector<std::pair<mesh::Tag, Real>> fixed_temperature;  ///< facet tag → T [K]
  solvers::DirectSolverBackend solver = solvers::DirectSolverBackend::kAuto;
  int extra_quadrature_order = 2;
};

/// Assembles and solves the steady heat equation on an H1 space (hanging-node meshes
/// supported through the H1 constraints).
template <int Dim>
class Thermal {
 public:
  /// @throws InvalidArgument for a non-positive conductivity.
  Thermal(const fespace::DofMap<Dim>& dofs, ThermalSetup setup);
  [[nodiscard]] const fespace::DofMap<Dim>& dofs() const noexcept { return *dofs_; }
  [[nodiscard]] const ThermalSetup& setup() const noexcept { return setup_; }
  /// κ of a cell.
  [[nodiscard]] Real conductivity(Index cell) const;
  /// Stiffness @f$ K_{ij} = \int\kappa\nabla\phi_i\cdot\nabla\phi_j @f$ and mass
  /// @f$ M_{ij} = \int\phi_i\phi_j @f$ of the space.
  [[nodiscard]] SparseMatrix stiffness() const;
  [[nodiscard]] SparseMatrix mass() const;
  /// Temperature coefficients for the source coefficients `q` on the same map: solves
  /// @f$ K T = M q @f$ with the fixed temperatures eliminated. @throws InvalidArgument if
  /// `q` does not match the map.
  [[nodiscard]] Vector solve(const Vector& q) const;
  /// Temperature for an assembled load @f$ b_i = \int\phi_i q @f$ (`absorbed_power_load`).
  [[nodiscard]] Vector solve_load(const Vector& load) const;
  /// Total heat input @f$ \int q @f$ [W] (3D) or [W/m] (2D) of source coefficients `q`
  /// (the mass matrix against the coefficients of the constant 1).
  [[nodiscard]] Real total_power(const Vector& q) const;

 private:
  const fespace::DofMap<Dim>* dofs_;
  ThermalSetup setup_;
};

extern template Vector absorbed_power_density<2>(const fespace::NedelecDofMap<2>&, const Vector&,
                                                 Real, const materials::MaterialMap&,
                                                 const fespace::DofMap<2>&);
extern template Vector absorbed_power_density<3>(const fespace::NedelecDofMap<3>&, const Vector&,
                                                 Real, const materials::MaterialMap&,
                                                 const fespace::DofMap<3>&);
extern template Vector absorbed_power_load<2>(const fespace::NedelecDofMap<2>&, const Vector&, Real,
                                              const materials::MaterialMap&,
                                              const fespace::DofMap<2>&, int);
extern template Vector absorbed_power_load<3>(const fespace::NedelecDofMap<3>&, const Vector&, Real,
                                              const materials::MaterialMap&,
                                              const fespace::DofMap<3>&, int);
extern template class Thermal<2>;
extern template class Thermal<3>;

}  // namespace hpfem::physics

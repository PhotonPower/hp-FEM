#pragma once
/// @file axisymmetric.hpp
/// Eigenproblems of a body of revolution (2.5D): for the azimuthal order m the Maxwell
/// eigenproblem @f$ \nabla\times(\mu_r^{-1}\nabla\times E) = k_0^2\varepsilon_r E @f$ with
/// @f$ E = E_m(r, z) e^{im\varphi} @f$ is posed on the meridian mesh (x = r, y = z) with the
/// forms of `assembly::assemble_axisymmetric`, PEC walls (tangential trace of @f$ (E_r, E_z)
/// @f$ and @f$ E_\varphi @f$ zero) and the axis conditions that keep the field regular at
/// r = 0: @f$ v = -i r E_\varphi = 0 @f$ for every m and @f$ E_z = 0 @f$ for m ≠ 0 (both as
/// Dirichlet data on the axis facets). The gradient kernel is removed with the order-m
/// gradient @f$ (\nabla\psi, m\psi) @f$ as in the Cartesian cavity solver, so no spurious modes
/// appear. `AxisymmetricCavity` solves the closed (lossless) problem, `AxisymmetricResonance`
/// the open one with the cylindrical PML (quasi-normal modes with complex ω, Q factor). See
/// docs/theory/axisymmetric.md and ADR-0010.
#include <optional>
#include <vector>

#include "hpfem/assembly/axisymmetric_forms.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/mesh.hpp"
#include "hpfem/pml/pml.hpp"
#include "hpfem/solvers/linear_solver.hpp"

namespace hpfem::physics {

/// Free DoFs of the block vector (e, v) and of the gauge potential ψ after the PEC and axis
/// conditions of order m.
struct AxisymmetricDofSets {
  std::vector<Index> free;            ///< block indices (Nédélec first, then H1 of v)
  std::vector<Index> free_potential;  ///< H1 indices of ψ
};

/// Index sets of order m: PEC fixes the Nédélec trace and v; the axis fixes v for every m
/// and the tangential Nédélec DoFs (E_z) for m ≠ 0; ψ vanishes on PEC and, for m ≠ 0, on
/// the axis.
/// @throws InvalidArgument if the maps differ in mesh or order or the axis tag has no facets.
[[nodiscard]] AxisymmetricDofSets axisymmetric_dof_sets(const fespace::NedelecDofMap<2>& meridian,
                                                        const fespace::DofMap<2>& azimuthal,
                                                        const std::vector<mesh::Tag>& pec_tags,
                                                        mesh::Tag axis_tag, int azimuthal_order);

/// Cylindrical PML as material (Teixeira–Chew): with the stretch factors @f$ s_r, s_z @f$ of
/// the box and @f$ s_\varphi = \tilde r / r @f$ the tensor
/// @f$ \Lambda = \mathrm{diag}(s_\varphi s_z / s_r,\ s_r s_z / s_\varphi,\ s_r s_\varphi / s_z) @f$
/// gives @f$ \tilde\varepsilon = \varepsilon_r\Lambda @f$ and
/// @f$ \tilde\mu^{-1} = \mu_r^{-1}\Lambda^{-1} @f$ (identity inside the box). The box must
/// have no layer on the axis side (x-min thickness 0).
[[nodiscard]] assembly::AxisymmetricForm axisymmetric_pml_form(
    const pml::PmlBox<2>& box, const materials::Material& material,
    std::optional<int> quadrature_order = std::nullopt);

/// Description of an axisymmetric cavity problem (closed, lossless).
struct AxisymmetricCavitySetup {
  materials::MaterialMap materials;   ///< lossless, by cell tag
  std::vector<mesh::Tag> pec_tags;    ///< facets with n × E = 0
  mesh::Tag axis_tag = mesh::kNoTag;  ///< facets on the axis r = 0 (required)
  int azimuthal_order = 0;            ///< m
  Index num_modes = 6;
  Index krylov_dimension = 0;
  Real tolerance = 1e-10;
  int max_iterations = 2000;
  int extra_quadrature_order = 2;
};

/// One eigenmode of order m.
struct AxisymmetricMode {
  Real wavenumber = 0;  ///< k0 [1/m]
  Vector meridian;      ///< (E_r, E_z) coefficients on the Nédélec map (full size)
  Vector azimuthal;     ///< v = -i r E_phi coefficients on the H1 map (full size)
};

/// Assembles the order-m pencil and solves the gauged eigenproblem.
class AxisymmetricCavity {
 public:
  /// @throws InvalidArgument if the maps differ in mesh or order, the axis tag has no facets,
  ///         or a material is lossy.
  AxisymmetricCavity(const fespace::NedelecDofMap<2>& meridian, const fespace::DofMap<2>& azimuthal,
                     AxisymmetricCavitySetup setup);
  [[nodiscard]] const AxisymmetricCavitySetup& setup() const noexcept { return setup_; }
  /// Lowest `num_modes` modes of order m, ascending in k0.
  /// @throws Error if the eigensolver does not converge.
  [[nodiscard]] std::vector<AxisymmetricMode> solve() const;
  /// Free (unconstrained) block DoFs after PEC and axis conditions.
  [[nodiscard]] const std::vector<Index>& free_dofs() const noexcept { return sets_.free; }
  /// Free H1 DoFs of the gauge potential ψ.
  [[nodiscard]] const std::vector<Index>& free_potential() const noexcept {
    return sets_.free_potential;
  }
  [[nodiscard]] const assembly::AxisymmetricSystem& system() const noexcept { return system_; }

 private:
  const fespace::NedelecDofMap<2>* meridian_;
  const fespace::DofMap<2>* azimuthal_;
  AxisymmetricCavitySetup setup_;
  assembly::AxisymmetricSystem system_;
  SparseMatrix gradient_;
  AxisymmetricDofSets sets_;
};

/// Description of an open axisymmetric resonator: materials, PEC walls, the axis, the
/// cylindrical PML and the search centre.
struct AxisymmetricResonanceSetup {
  Real target_omega = 0;              ///< search centre of ω [rad/s] (and PML design frequency)
  materials::MaterialMap materials;   ///< by cell tag (may be lossy)
  std::vector<mesh::Tag> pec_tags;    ///< facets with n × E = 0
  mesh::Tag axis_tag = mesh::kNoTag;  ///< facets on the axis r = 0 (required)
  int azimuthal_order = 1;            ///< m
  std::optional<pml::PmlBox<2>> pml;  ///< absorbing layers (r-max, z-min, z-max)
  Index num_modes = 6;
  Index krylov_dimension = 0;
  Real tolerance = 1e-10;
  int max_iterations = 200;
  solvers::DirectSolverBackend solver = solvers::DirectSolverBackend::kAuto;
  int extra_quadrature_order = 2;
  int pml_extra_quadrature_order = 6;
};

/// A quasi-normal mode of order m.
struct AxisymmetricResonantMode {
  Complex omega;        ///< complex angular frequency [rad/s], Im ω < 0 for decay
  Real wavelength = 0;  ///< 2π c0 / Re ω [m]
  Real quality = 0;     ///< Q = Re ω / (−2 Im ω)
  Real residual = 0;    ///< Arnoldi residual
  Vector meridian;      ///< (E_r, E_z) coefficients (full size)
  Vector azimuthal;     ///< v = -i r E_phi coefficients (full size)
};

/// Assembles the complex-symmetric order-m pencil with the PML and finds the modes
/// closest to the target with the gauged complex shift-invert solver.
class AxisymmetricResonance {
 public:
  /// @throws InvalidArgument for ω ≤ 0, a missing axis tag, mismatched maps, or a PML box
  ///         with a layer on the axis side.
  AxisymmetricResonance(const fespace::NedelecDofMap<2>& meridian,
                        const fespace::DofMap<2>& azimuthal, AxisymmetricResonanceSetup setup);
  [[nodiscard]] const AxisymmetricResonanceSetup& setup() const noexcept { return setup_; }
  /// Per-cell form: material tensors, stretched inside the PML.
  [[nodiscard]] assembly::AxisymmetricForm form_of_cell(Index cell) const;
  /// Modes ordered by the distance of ω to the target.
  /// @throws Error if the eigensolver does not converge.
  [[nodiscard]] std::vector<AxisymmetricResonantMode> solve() const;
  [[nodiscard]] const std::vector<Index>& free_dofs() const noexcept { return sets_.free; }

 private:
  const fespace::NedelecDofMap<2>* meridian_;
  const fespace::DofMap<2>* azimuthal_;
  AxisymmetricResonanceSetup setup_;
  AxisymmetricDofSets sets_;
};

}  // namespace hpfem::physics

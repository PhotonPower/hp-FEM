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
#include <functional>
#include <optional>
#include <vector>

#include "hpfem/adaptivity/residual_estimator.hpp"
#include "hpfem/assembly/axisymmetric_forms.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/fespace/constraints.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/mesh.hpp"
#include "hpfem/physics/absorption.hpp"
#include "hpfem/physics/layer_stack.hpp"
#include "hpfem/physics/postprocess.hpp"
#include "hpfem/pml/pml.hpp"
#include "hpfem/solvers/linear_solver.hpp"

namespace hpfem::physics {

/// Free DoFs of the block vector (e, v) and of the gauge potential ψ after the PEC and axis
/// conditions of order m, and the hanging-node constraints of a locally refined meridian mesh.
struct AxisymmetricDofSets {
  std::vector<Index> free;            ///< block indices (Nédélec first, then H1 of v)
  std::vector<Index> free_potential;  ///< H1 indices of ψ
  /// Hanging-node constraints (`assembly::hanging_constraints`) of the block vector,
  /// restricted to `free` and numbered within it; absent on a conforming mesh.
  std::optional<fespace::Constraints> constraints;
  /// The same for ψ, restricted to `free_potential`.
  std::optional<fespace::Constraints> potential_constraints;
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
  [[nodiscard]] const fespace::NedelecDofMap<2>& meridian() const noexcept { return *meridian_; }
  [[nodiscard]] const fespace::DofMap<2>& azimuthal() const noexcept { return *azimuthal_; }
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

/// m-th Fourier component of a field in the scaled components @f$ (E_r,\ v = -i r E_\varphi,\ E_z)
/// @f$ as a function of the meridian point.
using AxisymmetricField = std::function<Eigen::Matrix<Complex, 3, 1>(const Point<2>&)>;

/// The x-polarised plane wave @f$ E_0\,\hat x\,e^{ikz} @f$ travelling along the axis has the
/// azimuthal orders m = ±1 only: @f$ (E_r, E_\varphi, E_z) = \tfrac{E_0}{2}(1,\ \pm i,\ 0)e^{ikz}
/// @f$, i.e. @f$ (E_r, v, E_z) = \tfrac{E_0}{2}(1,\ \pm r,\ 0)e^{ikz} @f$. The scattered power of
/// the full wave is the sum over both orders (equal by symmetry).
/// @throws InvalidArgument for m ∉ {−1, 1}.
[[nodiscard]] AxisymmetricField axial_plane_wave(Complex amplitude, Real k, int m);

/// Orientation of a dipole on the axis.
enum class AxisDipole { kAxial, kTransverse };

/// Volume source @f$ f = i\omega\mu_0 J @f$ of a point dipole on the axis at z = `position`
/// with current moment `moment` [A m], smeared over the normalised 3D Gaussian
/// @f$ g = \exp(-(r^2 + (z - z_0)^2) / 2\sigma^2) / ((2\pi)^{3/2}\sigma^3) @f$, in the scaled
/// components of order m: the axial dipole (moment along z) has m = 0 only,
/// @f$ (f_r, f_v, f_z) = i\omega\mu_0 p\,g\,(0, 0, 1) @f$; the transverse dipole (moment
/// along x) has m = ±1, @f$ (f_r, f_v, f_z) = i\omega\mu_0 \tfrac{p}{2} g\,(1, \pm r, 0) @f$.
/// The smearing lowers the radiated power of the free dipole by @f$ e^{-k^2\sigma^2} @f$
/// (Gaussian form factor), exactly for the total power in a lossless medium. Total-field
/// formulation (`AxisymmetricScatteringSetup::current`).
/// @throws InvalidArgument for σ ≤ 0 or an order the orientation does not radiate.
[[nodiscard]] AxisymmetricField axisymmetric_gaussian_dipole(Real position, Complex moment,
                                                             AxisDipole orientation, Real sigma,
                                                             Real omega, int m);

/// Load @f$ b_J @f$ of an order-m current density @f$ (J_r,\ -i r J_\varphi,\ J_z) @f$ in the
/// scaled components on the block space (e, v), without the factor @f$ i\omega\mu_0 @f$ and
/// without the azimuthal factor 2π (the convention of the block forms): the source of a
/// `RieszProjection` built on `axisymmetric_pencil`. `axisymmetric_gaussian_dipole` divided
/// by @f$ i\omega\mu_0 @f$ gives such a density.
[[nodiscard]] Vector axisymmetric_current_load(const fespace::NedelecDofMap<2>& meridian,
                                               const fespace::DofMap<2>& azimuthal, int m,
                                               const AxisymmetricField& current,
                                               int extra_order = 4);

/// Radiated power of a point dipole of current moment p in vacuum,
/// @f$ P_0 = Z_0 k_0^2 |p|^2 / (12\pi) @f$ [W].
[[nodiscard]] Real dipole_vacuum_power(Complex moment, Real omega);

/// Error of an order-m field against an exact one in the norms of the body of revolution
/// (factor 2π dropped): @f$ \|e\|^2 = \int(|e_r|^2 + |e_\varphi|^2 + |e_z|^2)\,r\,dr\,dz @f$ with
/// @f$ e_\varphi = i e_v / r @f$, and the same for the cylindrical curl.
struct AxisymmetricError {
  Real l2 = 0;    ///< weighted L2 norm of the field error
  Real curl = 0;  ///< weighted L2 norm of the curl error
};

/// Error of the coefficients `meridian` (E_r, E_z) and `azimuthal` (v) of order m against the
/// exact field `exact` in the scaled components (E_r, v = −i r E_φ, E_z) and its cylindrical
/// curl `exact_curl` = ((∇×E)_r, (∇×E)_φ, (∇×E)_z); an empty `exact_curl` means zero. The
/// discrete curl of the mode is @f$ (\tfrac{i}{r}(mE_z - \partial_z v),\ \partial_zE_r -
/// \partial_rE_z,\ \tfrac{i}{r}(\partial_rv - mE_r)) @f$. Quadrature of degree 2p +
/// `extra_order` (+2 on cells touching the axis and on curved cells).
/// @throws InvalidArgument if the maps differ in mesh or a vector does not match its map.
[[nodiscard]] AxisymmetricError axisymmetric_error(
    const fespace::NedelecDofMap<2>& meridian, const fespace::DofMap<2>& azimuthal,
    const Vector& meridian_coefficients, const Vector& azimuthal_coefficients, int azimuthal_order,
    const AxisymmetricField& exact, const AxisymmetricField& exact_curl = {}, int extra_order = 4);

/// Description of an axisymmetric scattering problem (scattered-field formulation): the
/// incident field is a solution in the background medium, the source
/// @f$ k_0^2(\varepsilon_r - \varepsilon_{bg})E^{inc} @f$ lives in the cells whose material
/// differs from the background, the PML absorbs the scattered field. The background is one
/// material (`materials.background()`) or a planar layer stack normal to the axis
/// (`background`, ADR-0014).
struct AxisymmetricScatteringSetup {
  Real omega = 0;                     ///< angular frequency [rad/s]
  materials::MaterialMap materials;   ///< by cell tag; the background for unlisted tags
  std::vector<mesh::Tag> pec_tags;    ///< facets with n × E = 0 (scattered field = −incident)
  mesh::Tag axis_tag = mesh::kNoTag;  ///< facets on the axis r = 0 (required)
  int azimuthal_order = 1;            ///< m of the incident component
  std::optional<pml::PmlBox<2>> pml;  ///< absorbing layers (r-max, z-min, z-max)
  /// Layered background (ADR-0014): planar layers normal to the axis, the z of the stack is the
  /// y of the meridian mesh. The incident field is then the stack's plane wave of order m
  /// (`layered_axisymmetric_wave`), the source @f$ k_0^2(\varepsilon_c -
  /// \varepsilon_{stack}(z_c))E^{inc} @f$ lives only where the material of a cell deviates
  /// from the stack at its centroid (particles and holes alike), and no cell of the PML may
  /// deviate (the deviation must be bounded). Interfaces must lie on mesh lines; the materials
  /// of the layer cells are given by tag in `materials` as usual.
  std::optional<LayerStack<3>> background;
  /// m-th component of the incident field (scattered-field formulation) ...
  AxisymmetricField incident;
  /// ... or the volume source f = iωμ0 J of order m (total-field formulation,
  /// e.g. `axisymmetric_gaussian_dipole`); exactly one of the two.
  AxisymmetricField current;
  solvers::DirectSolverBackend solver = solvers::DirectSolverBackend::kAuto;
  int extra_quadrature_order = 4;      ///< added to 2p for the non-polynomial incident field
  int pml_extra_quadrature_order = 6;  ///< added to 2p in PML cells
};

/// Field of one azimuthal order: the scattered field with an incident field, the total
/// field with a current.
struct AxisymmetricScatteredField {
  int azimuthal_order = 0;
  Vector meridian;   ///< (E_r, E_z) coefficients (full size)
  Vector azimuthal;  ///< v = -i r E_phi coefficients (full size)
};

/// Assembles the order-m operator @f$ S - k_0^2 M @f$ with the PML and the volume source and
/// solves for the scattered field. PEC walls impose the scattered field −E^inc only through
/// the homogeneous condition (the incident field must vanish there, as for closed walls of a
/// resonator); open problems have no PEC.
class AxisymmetricScattering {
 public:
  /// @throws InvalidArgument for ω ≤ 0, neither or both of incident field and current, a
  ///         missing axis tag, mismatched maps, a PML with a layer on the axis side, a cell
  ///         straddling an interface of the layered background, or (with an incident field) a
  ///         PML cell whose material deviates from the layered background.
  AxisymmetricScattering(const fespace::NedelecDofMap<2>& meridian,
                         const fespace::DofMap<2>& azimuthal, AxisymmetricScatteringSetup setup);
  [[nodiscard]] const AxisymmetricScatteringSetup& setup() const noexcept { return setup_; }
  [[nodiscard]] Real wavenumber() const noexcept { return k0_; }
  /// Background material of a cell: the layer of `setup.background` at the cell centroid,
  /// otherwise `materials.background()`. The scattered-field source is proportional to
  /// `material(c) − background_material(c)`.
  [[nodiscard]] const materials::Material& background_material(Index cell) const;
  /// Per-cell form: material tensors (stretched inside the PML) and the contrast source.
  [[nodiscard]] assembly::AxisymmetricForm form_of_cell(Index cell) const;
  /// @throws Error if the factorisation fails.
  [[nodiscard]] AxisymmetricScatteredField solve() const;
  /// Element indicators of a solution (`adaptivity::axisymmetric_residual_estimate` with the
  /// per-cell forms of this problem and k0²): the residual of the equation actually solved,
  /// PML and contrast source included.
  [[nodiscard]] adaptivity::Estimate estimate(
      const AxisymmetricScatteredField& field,
      const adaptivity::EstimatorOptions& options = {}) const;
  /// Absorbed power of the total field of a solution (ADR-0014 §4): the scattered field plus
  /// the incident field `setup.incident` in the scattered-field formulation, the field itself
  /// with a current; per cell and per tag, cells inside the PML left out (their loss is the
  /// absorber's). Units [W] for this order; the orders add up.
  [[nodiscard]] AbsorbedPower absorbed_power(const AxisymmetricScatteredField& field,
                                             int extra_order = 4) const;
  /// Absorbed power of the incident field alone in the bare background: every cell with
  /// `background_material` (the stack's layer, also in the cells of the body), PML left out;
  /// zero without an incident field or with a lossless uniform background. Summed over a
  /// bounded region, `absorbed_power` minus this is the absorption change caused by the body
  /// (negative where a hole removes absorbing material).
  [[nodiscard]] AbsorbedPower incident_absorbed_power(int extra_order = 4) const;
  /// Cells outside the PML whose material deviates from `background_material` (particles and
  /// holes): the body whose total-field absorption is the absorption cross-section.
  [[nodiscard]] std::vector<Index> scatterer_cells() const;
  /// Error of a solution against an exact field (`axisymmetric_error`).
  [[nodiscard]] AxisymmetricError error(const AxisymmetricScatteredField& field,
                                        const AxisymmetricField& exact,
                                        const AxisymmetricField& exact_curl = {}) const;
  [[nodiscard]] const std::vector<Index>& free_dofs() const noexcept { return sets_.free; }

 private:
  const fespace::NedelecDofMap<2>* meridian_;
  const fespace::DofMap<2>* azimuthal_;
  /// The checks of a layered background (interfaces on mesh lines, no deviation in the PML).
  void check_background() const;

  AxisymmetricScatteringSetup setup_;
  AxisymmetricDofSets sets_;
  Real k0_ = 0;
};

/// Time-averaged power [W] of the order-m field @f$ (e, v) @f$ through the surface of
/// revolution generated by the meridian surface: @f$ \int 2\pi r\,\tfrac12\mathrm{Re}(E\times
/// H^*)\cdot n\,ds @f$ with @f$ H = \nabla\times E / (i\omega\mu) @f$ from the cylindrical curl
/// of the mode and μ of the inside cell. Orders do not mix, so the power of a field with
/// several orders is the sum over m.
/// `added_value` / `added_curl` (both or neither): an analytic order-m field in the scaled
/// components and its cylindrical curl added at every quadrature point, e.g. the stack field
/// of `layered_axisymmetric_wave` for the flux of the total field (ADR-0009 §5).
/// @throws InvalidArgument if the coefficient vectors do not match the maps or only one of
///         `added_value` and `added_curl` is given.
[[nodiscard]] Real axisymmetric_poynting_flux(
    const fespace::NedelecDofMap<2>& meridian, const fespace::DofMap<2>& azimuthal,
    const Vector& meridian_coefficients, const Vector& azimuthal_coefficients, int azimuthal_order,
    Real omega, const materials::MaterialMap& materials, const Surface<2>& surface, int order = 8,
    const AxisymmetricField& added_value = {}, const AxisymmetricField& added_curl = {});

/// Power through a surface split by the layers of a stack (ADR-0014 §4): the part through the
/// surface above the top interface (`up`, into the cover), below the bottom interface (`down`,
/// into the substrate) and between them (`lateral`, radially along the layers: guided and
/// absorbed in the layers outside the surface). Every quadrature point is assigned by its
/// height with `LayerStack::region`; the surface should cross the interfaces, not run along
/// them. In a lossy substrate `down` depends on the depth of the surface.
struct AxisymmetricFluxChannels {
  Real up = 0;       ///< [W]
  Real down = 0;     ///< [W]
  Real lateral = 0;  ///< [W]
  [[nodiscard]] Real total() const noexcept { return up + down + lateral; }
};

/// `axisymmetric_poynting_flux` split into the channels of `stack` (arguments as there).
/// @throws InvalidArgument as `axisymmetric_poynting_flux`.
[[nodiscard]] AxisymmetricFluxChannels axisymmetric_flux_channels(
    const fespace::NedelecDofMap<2>& meridian, const fespace::DofMap<2>& azimuthal,
    const Vector& meridian_coefficients, const Vector& azimuthal_coefficients, int azimuthal_order,
    Real omega, const materials::MaterialMap& materials, const Surface<2>& surface,
    const LayerStack<3>& stack, int order = 8, const AxisymmetricField& added_value = {},
    const AxisymmetricField& added_curl = {});

/// Power of an order-m field through the disc r ≤ `radius` at height z (aperture
/// transmission, ADR-0014 §4): `total` of the field plus the added analytic field, `background`
/// of the added field alone through the same facets, both along `direction` (−1: downwards,
/// the transmitted power of a wave from the top is positive; +1: upwards). `change()` is the
/// transmission caused by the body; summed over the orders, `background` is the stack's
/// transmittance times the incident power on the disc. [W].
struct AxisymmetricDiscFlux {
  Real total = 0;
  Real background = 0;
  [[nodiscard]] Real change() const noexcept { return total - background; }
};

/// The disc must lie on a mesh line y = z and end at a mesh vertex r = `radius`.
/// @throws InvalidArgument as `axisymmetric_poynting_flux`, for a radius ≤ 0, a direction other
///         than ±1, or a disc that is not covered by facets of the mesh line.
[[nodiscard]] AxisymmetricDiscFlux axisymmetric_disc_flux(
    const fespace::NedelecDofMap<2>& meridian, const fespace::DofMap<2>& azimuthal,
    const Vector& meridian_coefficients, const Vector& azimuthal_coefficients, int azimuthal_order,
    Real omega, const materials::MaterialMap& materials, Real z, Real radius, int direction = -1,
    const AxisymmetricField& added_value = {}, const AxisymmetricField& added_curl = {},
    int order = 8);

/// Absorbed power of an order-m field (ADR-0014 §4): the Joule heating
/// @f$ \tfrac{\omega\varepsilon_0}{2}\,\mathrm{Im}\,\varepsilon_r\,|E_m|^2 @f$ integrated with
/// the weight 2πr over every lossy cell, @f$ |E_m|^2 = |E_r|^2 + |E_\varphi|^2 + |E_z|^2 @f$
/// with @f$ E_\varphi = i v / r @f$; the orders are orthogonal in φ, so the power of a field with
/// several orders is the sum over m (per cell as well). `added` (scaled components) is added at
/// every quadrature point: the incident or stack field for the total field. Cells whose
/// centroid lies in `pml` are left out. Rules of degree 2p + `extra_order`. [W].
/// @throws InvalidArgument if the coefficient vectors do not match the maps.
[[nodiscard]] AbsorbedPower axisymmetric_absorbed_power(
    const fespace::NedelecDofMap<2>& meridian, const fespace::DofMap<2>& azimuthal,
    const Vector& meridian_coefficients, const Vector& azimuthal_coefficients, int azimuthal_order,
    Real omega, const materials::MaterialMap& materials, const AxisymmetricField& added = {},
    const std::optional<pml::PmlBox<2>>& pml = std::nullopt, int extra_order = 4);

/// Far-field pattern of an order-m field, @f$ E \approx F(\theta)\,e^{im\varphi}\,e^{ikR}/R @f$
/// in the background medium, sampled at the polar angles `theta` (from the +z axis).
struct AxisymmetricFarField {
  int azimuthal_order = 0;
  Real wavenumber = 0;           ///< k of the background [1/m]
  Real impedance = 0;            ///< Z of the background [Ohm]
  std::vector<Real> theta;       ///< polar angles [rad]
  std::vector<Complex> f_theta;  ///< @f$ F_\theta(\theta) @f$ at φ = 0 [V/m · m]
  std::vector<Complex> f_phi;    ///< @f$ F_\varphi(\theta) @f$ at φ = 0 [V/m · m]
  /// Radiated power @f$ \int |F|^2 d\Omega / (2Z) @f$ [W] of this order by the trapezoidal
  /// rule over the sampled angles (which should cover 0 … π).
  [[nodiscard]] Real radiated_power() const;
  /// The same over the sampled angles in [`theta_min`, `theta_max`] only (sample the limits):
  /// the power collected by a cone, e.g. an objective of numerical aperture NA in a medium of
  /// index n above the body: [0, asin(NA / n)], below it: [π − asin(NA / n), π].
  [[nodiscard]] Real power_between(Real theta_min, Real theta_max) const;
};

/// Near-to-far-field transform of the order-m field on a closed surface of revolution in the
/// homogeneous background (same formula as `FarField<3>`: @f$ F = \frac{ik}{4\pi}[Z N_t -
/// \hat r\times L] @f$ with @f$ N = \int J e^{-ik\hat r\cdot x'} @f$, @f$ L = \int M e^{-ik\hat
/// r\cdot x'} @f$, @f$ J = n\times H @f$, @f$ M = -n\times E @f$); the azimuthal integration is
/// done analytically with the Bessel functions @f$ J_m, J_{m\pm1}(k\rho\sin\theta) @f$, so
/// only the meridian curve is integrated numerically. `background` gives k and Z; the surface
/// must enclose every source and scatterer and lie in that medium.
/// @throws InvalidArgument if the coefficient vectors do not match the maps.
[[nodiscard]] AxisymmetricFarField axisymmetric_far_field(
    const fespace::NedelecDofMap<2>& meridian, const fespace::DofMap<2>& azimuthal,
    const Vector& meridian_coefficients, const Vector& azimuthal_coefficients, int azimuthal_order,
    Real omega, const materials::MaterialMap& materials, const Surface<2>& surface,
    const std::vector<Real>& theta, int order = 8);

/// Far field of an order-m field on a layered background in both half-spaces, see
/// `axisymmetric_layered_far_field`: `up` in the cover (polar angles θ < π/2 from +z, wavenumber
/// and impedance of the cover), `down` in the substrate (θ > π/2).
struct AxisymmetricLayeredFarField {
  AxisymmetricFarField up;
  AxisymmetricFarField down;
};

/// Far field by reciprocity (ADR-0014 §4, M18 S3): with the equivalent currents of the field on a
/// closed surface S of revolution around every source and scatterer, the amplitude
/// @f$ E \approx F(\theta)\,e^{im\varphi}\,e^{iknR}/R @f$ in the direction r̂ of either half-space
/// is the overlap with the layered plane wave arriving from r̂ (incident amplitude 1, the stack's
/// reflections included):
/// @f$ F\cdot\hat e = \frac{i\omega\mu_0}{4\pi}\oint_S (E\times H_{pw} - E_{pw}\times H)\cdot n\,dS
/// @f$, with @f$ \hat e = \hat\theta @f$ for the p and @f$ -\hat\varphi @f$ for the s wave. The φ
/// integral leaves the order −m of the wave (`layered_axisymmetric_wave` at θ from the normal,
/// from the top for the cover, from the bottom for the substrate) with the factor
/// @f$ 2\pi(-1)^m @f$ at φ = 0, the phase is referred to the origin. A homogeneous stack gives
/// `axisymmetric_far_field`. Sources may lie anywhere inside S (scattered field of a body on the
/// stack, or the total field of a dipole if S lies in one layer); S may cross the layers, lossy
/// ones included (reciprocity holds in any reciprocal medium; outside S only the stack).
/// The power into the half-spaces is `up.radiated_power()` and `down.radiated_power()`
/// (sampled up to grazing); in a lossless stack the scattered power minus both is guided along
/// the layers. `theta_up` ⊂ [0, π/2), `theta_down` ⊂ (π/2, π].
/// @throws InvalidArgument if the coefficient vectors do not match the maps, an angle lies in the
///         wrong half-space or at grazing, or `theta_down` is given for a lossy substrate.
[[nodiscard]] AxisymmetricLayeredFarField axisymmetric_layered_far_field(
    const fespace::NedelecDofMap<2>& meridian, const fespace::DofMap<2>& azimuthal,
    const Vector& meridian_coefficients, const Vector& azimuthal_coefficients, int azimuthal_order,
    Real omega, const materials::MaterialMap& materials, const Surface<2>& surface,
    const LayerStack<3>& stack, const std::vector<Real>& theta_up,
    const std::vector<Real>& theta_down, int order = 8);

/// Polarisation of a plane wave relative to its plane of incidence (the x–z plane).
enum class PlanePolarisation { kS, kP };

/// Order m of the plane wave @f$ E_0\,\hat p\,e^{ik\cdot x} @f$ with the wave vector
/// @f$ k(\sin\theta_i, 0, \cos\theta_i) @f$ (angle θ_i from the axis, in the x–z plane) and the
/// polarisation @f$ \hat p = \hat y @f$ (s) or @f$ (\cos\theta_i, 0, -\sin\theta_i) @f$ (p). The
/// Jacobi–Anger expansion @f$ e^{ik_\perp\rho\cos\varphi} = \sum_n i^n
/// J_n(k_\perp\rho)e^{in\varphi}
/// @f$ gives, with @f$ a_n = i^n J_n(k_\perp\rho) @f$ and @f$ k_\perp = k\sin\theta_i @f$,
/// @f$ E_{\rho,m} = \tfrac{p_x}{2}(a_{m-1} + a_{m+1}) + \tfrac{p_y}{2i}(a_{m-1} - a_{m+1}) @f$,
/// @f$ E_{\varphi,m} = -\tfrac{p_x}{2i}(a_{m-1} - a_{m+1}) + \tfrac{p_y}{2}(a_{m-1} + a_{m+1}) @f$,
/// @f$ E_{z,m} = p_z a_m @f$, all times @f$ E_0 e^{ikz\cos\theta_i} @f$, in the scaled components
/// @f$ (E_r, v = -i r E_\varphi, E_z) @f$. At θ_i = 0 only m = ±1 survive (`axial_plane_wave`).
/// The orders decouple on a body of revolution; the full response is the sum over m, which
/// converges once |m| exceeds @f$ k_\perp @f$ times the radius of the scatterer.
[[nodiscard]] AxisymmetricField oblique_plane_wave(Complex amplitude, Real k, Real theta_i,
                                                   PlanePolarisation polarisation, int m);

/// Side of a layer stack from which a plane wave comes.
enum class StackSide {
  kTop,     ///< from the incidence medium, travelling towards −z
  kBottom,  ///< from the substrate (lossless), travelling towards +z
};

/// Order m of the plane wave on a layer stack, see `layered_axisymmetric_wave`.
struct AxisymmetricLayeredWave {
  AxisymmetricField value;  ///< @f$ (E_r,\ v = -i r E_\varphi,\ E_z) @f$ of order m
  /// Cylindrical curl @f$ ((\nabla\times E)_r, (\nabla\times E)_\varphi, (\nabla\times E)_z) @f$
  /// of order m (the convention of `axisymmetric_error`), @f$ = i\omega\mu_0 H @f$.
  AxisymmetricField curl;
  Real reflectance = 0;    ///< R of the bare stack for this side
  Real transmittance = 0;  ///< T of the bare stack for this side
  Real absorptance = 0;    ///< 1 − R − T
};

/// Order m of the plane wave of `LayerStack<3>::plane_wave` (ADR-0014 §2–3): the wave of
/// amplitude |E| = `amplitude` [V/m] at the angle θ ∈ [0, π/2) [rad] from the normal, in-plane
/// wave vector along +x, coming from the incidence medium (`kTop`, travelling towards −z) or
/// from the substrate (`kBottom`, travelling towards +z; the substrate must be lossless), with
/// the background field of the stack in every layer. In layer j the field is an up and a down
/// partial wave with the common real in-plane wavenumber @f$ k_\rho = k_0 n_{inc}\sin\theta @f$
/// and the vertical wavenumbers @f$ \pm k_{z,j} @f$ (complex in lossy layers and beyond the
/// critical angle); each is expanded by Jacobi–Anger as in `oblique_plane_wave` with its complex
/// polarisation vector, @f$ a_n = i^n J_n(k_\rho r) @f$. `kBottom` is the wave from above on the
/// reversed stack (incidence medium the substrate, layers reversed) mirrored z → −z:
/// @f$ (E_r, v, E_z)(r, z) \mapsto (E_r, v, -E_z)(r, -z) @f$, curl
/// @f$ (C_r, C_\varphi, C_z) \mapsto (-C_r, -C_\varphi, C_z) @f$. At θ = 0 only m = ±1 are
/// nonzero. Polarisation as `oblique_plane_wave` on both sides: s has E along y, p has H along +y
/// (on the bottom side the sign of the mirrored p wave is flipped accordingly), so a homogeneous
/// stack gives `oblique_plane_wave` at θ_i = π − θ (top) and θ_i = θ (bottom). Cost per point: a
/// few Bessel functions of one argument.
/// @throws InvalidArgument for k0 ≤ 0, θ outside [0, π/2), grazing propagation in a layer
///         (`LayerStack::plane_wave`) or `kBottom` on a lossy substrate.
[[nodiscard]] AxisymmetricLayeredWave layered_axisymmetric_wave(const LayerStack<3>& stack, Real k0,
                                                                Real theta, Polarisation pol, int m,
                                                                StackSide side = StackSide::kTop,
                                                                Complex amplitude = 1.0);

/// Fields of several orders and their powers through a common surface.
struct AxisymmetricOrders {
  std::vector<int> orders;                         ///< m of every solved order
  std::vector<AxisymmetricScatteredField> fields;  ///< one per order
  std::vector<Real> power;                         ///< power [W] of every order
  [[nodiscard]] Real total_power() const;          ///< Σ_m power (orders are orthogonal)
};

/// Solves the scattering problem of `setup` for the orders m = 0, ±1, ±2, … with the incident
/// component `incident_of_order(m)`, stopping when |m| ≥ 2 and the power of both ±m through
/// `surface` falls below `tolerance` times the total so far, or at `max_order`. The setup's
/// `azimuthal_order` and `incident` are overwritten per order.
/// @throws InvalidArgument for max_order < 0 or a negative tolerance.
[[nodiscard]] AxisymmetricOrders scatter_orders(
    const fespace::NedelecDofMap<2>& meridian, const fespace::DofMap<2>& azimuthal,
    AxisymmetricScatteringSetup setup,
    const std::function<AxisymmetricField(int)>& incident_of_order, int max_order,
    const Surface<2>& surface, Real tolerance = 1e-6);

/// Far-field pattern of the sum of orders at the azimuth φ: @f$ F(\theta, \varphi) = \sum_m
/// F_m(\theta)\,e^{im\varphi} @f$ in the spherical components (the patterns must share the
/// polar angles). The radiated power of the result is that of the full field.
/// @throws InvalidArgument if the patterns do not share their angles.
[[nodiscard]] AxisymmetricFarField superpose_far_field(
    const std::vector<AxisymmetricFarField>& patterns, const std::vector<int>& orders, Real phi);

}  // namespace hpfem::physics

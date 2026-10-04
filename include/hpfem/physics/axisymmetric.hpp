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

#include "hpfem/assembly/axisymmetric_forms.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/mesh.hpp"
#include "hpfem/physics/postprocess.hpp"
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

/// Radiated power of a point dipole of current moment p in vacuum,
/// @f$ P_0 = Z_0 k_0^2 |p|^2 / (12\pi) @f$ [W].
[[nodiscard]] Real dipole_vacuum_power(Complex moment, Real omega);

/// Description of an axisymmetric scattering problem (scattered-field formulation): the
/// incident field is a solution in the background medium, the source
/// @f$ k_0^2(\varepsilon_r - \varepsilon_{bg})E^{inc} @f$ lives in the cells whose material
/// differs from the background, the PML absorbs the scattered field.
struct AxisymmetricScatteringSetup {
  Real omega = 0;                     ///< angular frequency [rad/s]
  materials::MaterialMap materials;   ///< by cell tag; the background for unlisted tags
  std::vector<mesh::Tag> pec_tags;    ///< facets with n × E = 0 (scattered field = −incident)
  mesh::Tag axis_tag = mesh::kNoTag;  ///< facets on the axis r = 0 (required)
  int azimuthal_order = 1;            ///< m of the incident component
  std::optional<pml::PmlBox<2>> pml;  ///< absorbing layers (r-max, z-min, z-max)
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
  ///         missing axis tag, mismatched maps, or a PML with a layer on the axis side.
  AxisymmetricScattering(const fespace::NedelecDofMap<2>& meridian,
                         const fespace::DofMap<2>& azimuthal, AxisymmetricScatteringSetup setup);
  [[nodiscard]] const AxisymmetricScatteringSetup& setup() const noexcept { return setup_; }
  [[nodiscard]] Real wavenumber() const noexcept { return k0_; }
  /// Per-cell form: material tensors (stretched inside the PML) and the contrast source.
  [[nodiscard]] assembly::AxisymmetricForm form_of_cell(Index cell) const;
  /// @throws Error if the factorisation fails.
  [[nodiscard]] AxisymmetricScatteredField solve() const;
  [[nodiscard]] const std::vector<Index>& free_dofs() const noexcept { return sets_.free; }

 private:
  const fespace::NedelecDofMap<2>* meridian_;
  const fespace::DofMap<2>* azimuthal_;
  AxisymmetricScatteringSetup setup_;
  AxisymmetricDofSets sets_;
  Real k0_ = 0;
};

/// Time-averaged power [W] of the order-m field @f$ (e, v) @f$ through the surface of
/// revolution generated by the meridian surface: @f$ \int 2\pi r\,\tfrac12\mathrm{Re}(E\times
/// H^*)\cdot n\,ds @f$ with @f$ H = \nabla\times E / (i\omega\mu) @f$ from the cylindrical curl
/// of the mode and μ of the inside cell. Orders do not mix, so the power of a field with
/// several orders is the sum over m.
/// @throws InvalidArgument if the coefficient vectors do not match the maps.
[[nodiscard]] Real axisymmetric_poynting_flux(
    const fespace::NedelecDofMap<2>& meridian, const fespace::DofMap<2>& azimuthal,
    const Vector& meridian_coefficients, const Vector& azimuthal_coefficients, int azimuthal_order,
    Real omega, const materials::MaterialMap& materials, const Surface<2>& surface, int order = 8);

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

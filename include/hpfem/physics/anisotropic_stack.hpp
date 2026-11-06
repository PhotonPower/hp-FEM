#pragma once
/// @file anisotropic_stack.hpp
/// Planar stacks of crystal layers (ADR-0015 §5, M19 S4): plane-wave reflection and transmission
/// of layers and a substrate with symmetric permittivity tensors in the laboratory frame (x, y in
/// the layer plane, z the layer normal; the wave comes from an isotropic, lossless incidence
/// medium above the top interface and travels towards −z). Convention exp(−iωt): a lossy medium
/// has a positive semi-definite Im ε.
///
/// Per region the in-plane wavevector fixes the Berreman system @f$ \psi' = ik_0K\psi @f$ for the
/// tangential fields @f$ \psi = (E_x, E_y, \tilde H_x, \tilde H_y) @f$, @f$ \tilde H = Z_0H @f$,
/// with @f$ E_z, H_z @f$ eliminated (this needs @f$ \varepsilon_{zz} \ne 0 @f$). K is never
/// diagonalised: two ordered complex Schur forms per region give orthonormal bases of the upward
/// and the downward invariant subspaces and the triangular 2×2 blocks of K on them, which stay
/// well defined for degenerate modes (isotropic layers, an optic axis along z) and at defective
/// points (Voigt waves of absorbing biaxial media). Layers are crossed with the 2×2 exponentials
/// of these blocks, and the reflection matrices are built by a stable S-matrix recursion from the
/// substrate upwards that only ever multiplies decaying exponentials. Theory:
/// docs/theory/anisotropic-stack.md.

#include <array>
#include <vector>

#include <Eigen/Core>

#include "hpfem/core/types.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/physics/layer_stack.hpp"

namespace hpfem::physics {

/// Tangential field @f$ (E_x, E_y, Z_0H_x, Z_0H_y) @f$ of a plane wave or a mode.
using TangentialField = Eigen::Matrix<Complex, 4, 1>;

/// Kind of a downward substrate mode: s / p of an isotropic substrate, ordinary (D ⊥ optic axis)
/// / extraordinary of a uniaxial one, or one of the two eigenmodes of a biaxial one.
enum class SubstrateMode { kS, kP, kOrdinary, kExtraordinary, kBiaxial };

/// Plane-wave response of an `AnisotropicStack` for the two incident polarisations s, p of unit
/// |E| (s: E along @f$ \hat s = \hat z\times\hat k_\parallel @f$, p: E along
/// @f$ \hat s\times\hat k @f$, so that H is along +ŝ as in `LayerStack`).
struct StackResponse {
  /// Jones reflection matrix: column b = incident s, p; row a = the reflected s, p amplitude
  /// (E, at the top interface): r(0, 0) = r_ss, r(1, 0) = r_ps, r(0, 1) = r_sp, r(1, 1) = r_pp.
  Eigen::Matrix2cd reflection;
  /// Transmission into the two downward modes of the substrate (amplitudes at the bottom
  /// interface; column = incident s, p; row = mode, see `substrate_modes`).
  Eigen::Matrix2cd transmission;
  /// Tangential fields of the downward substrate modes, each with unit |E|: s and p of an
  /// isotropic substrate; ordinary then extraordinary of a uniaxial one; the eigenmodes of the
  /// downward block of a biaxial one. Where the two modes coincide (k along an optic axis) the
  /// basis of the subspace that diagonalises the cross power flux (power-orthogonal modes).
  std::array<TangentialField, 2> substrate_modes;
  std::array<SubstrateMode, 2> substrate_mode_kinds{};
  /// Power transmitted into each substrate mode, |t_m|² times the mode's flux over the incident
  /// flux (rows mode, columns incident s, p); sums to `transmittance` when the modes are
  /// power-orthogonal (always for a lossless substrate with distinct modes).
  Eigen::Matrix2d channel_transmittance;
  /// Vertical wavenumbers k_z of the downward substrate modes [1/m] (Im ≤ 0).
  std::array<Complex, 2> substrate_kz;
  /// Reflectance, transmittance (power flux into the substrate) and absorptance
  /// A = 1 − R − T of the finite layers, for incident s (entry 0) and p (entry 1).
  Eigen::Vector2d reflectance;
  Eigen::Vector2d transmittance;
  Eigen::Vector2d absorptance;
};

/// The two upward or the two downward plane-wave modes of a homogeneous half-space at the
/// normalised in-plane wavevector (a, b) = (k_x, k_y)/k0, labelled as the substrate modes of
/// `StackResponse` (s / p, ordinary / extraordinary, biaxial, power-orthogonal where they
/// coincide). Each mode has unit |E|; the isotropic s vector is @f$ \hat z\times\hat k_\parallel
/// @f$ (ŷ at a = b = 0) and p = s × k̂ / n.
struct HalfSpaceModes {
  std::array<TangentialField, 2> modes;
  /// The full E of each mode (unit norm, laboratory frame).
  std::array<Eigen::Matrix<Complex, 3, 1>, 2> fields;
  std::array<SubstrateMode, 2> kinds{};
  /// k_z / k0 of the modes (Im ≥ 0 upward, ≤ 0 downward).
  std::array<Complex, 2> kz;
  /// Power flux 2 Z0 S_z of each mode along its direction (≥ 0; 0 for an evanescent mode of a
  /// lossless medium), in units of |E|² = 1.
  std::array<Real, 2> flux{};
};

/// Modes of a half-space of permittivity `eps` (laboratory frame, z the normal) travelling or
/// decaying towards +z (`upward`) or −z. @throws InvalidArgument for ε_zz ≈ 0, a gain medium or
/// grazing propagation (k_z = 0, a Rayleigh anomaly).
[[nodiscard]] HalfSpaceModes half_space_modes(const materials::Tensor3& eps, Real a, Real b,
                                              bool upward);

/// Plane wave on an `AnisotropicStack`: the piecewise analytic total field (value, curl and the
/// value of a given region's branch, usable as `ScatteringSetup<3>::incident` with
/// `AnisotropicStack::layer_stack()` as background), the incident wave alone and the response.
struct AnisotropicPlaneWave {
  IncidentField<3> field;
  IncidentField<3> incident_wave;
  StackResponse response;
};

/// A planar stack of possibly anisotropic layers (laboratory frame, z the normal): isotropic,
/// lossless incidence medium above `top`, `layers` from top to bottom, semi-infinite substrate.
class AnisotropicStack {
 public:
  /// The incidence medium may be a lossless crystal, whose waves are then those of `mode_wave`
  /// (`response`, `plane_wave` and `reflection_at` need an isotropic one).
  /// @throws InvalidArgument for a thickness ≤ 0, a magnetic material, a lossy incidence
  ///         medium, a gain medium (Im ε not positive semi-definite) or an
  ///         epsilon-near-zero @f$ |\varepsilon_{zz}| < 10^{-8}\max|\varepsilon_{ij}| @f$.
  AnisotropicStack(materials::Material incidence_medium, std::vector<Layer> layers,
                   materials::Material substrate, Real top = 0.0);

  [[nodiscard]] int num_layers() const noexcept { return static_cast<int>(layers_.size()); }
  [[nodiscard]] const materials::Material& material(int region) const;
  [[nodiscard]] const std::vector<Layer>& layers() const noexcept { return layers_; }
  [[nodiscard]] Real top() const noexcept { return top_; }
  /// The geometry and materials as a `LayerStack<3>`, the layered background of
  /// `Scattering<3>` (its scalar `plane_wave` refuses crystals; use `plane_wave` here).
  [[nodiscard]] LayerStack<3> layer_stack() const;

  /// Response at the vacuum wavenumber k0 [1/m] for incidence at `angle` [rad] from the normal
  /// in the plane at the azimuth `azimuth` [rad] (in-plane wavevector
  /// @f$ k_0n_0\sin\theta(\cos\phi, \sin\phi) @f$).
  /// @throws InvalidArgument for k0 ≤ 0, |angle| ≥ π/2, or a region without two upward and two
  ///         downward modes (grazing propagation, k_z = 0).
  [[nodiscard]] StackResponse response(Real k0, Real angle, Real azimuth = 0.0) const;

  /// Plane wave of polarisation `pol` (s or p) and amplitude |E| = `amplitude` [V/m] incident
  /// as in `response`: the analytic field in every region (E and curl E = i k0 Z0 H),
  /// @f$ \psi(z) = W_u e^{ik_0T_u(z - z_b)}\alpha + W_d e^{ik_0T_d(z - z_t)}\delta @f$ with the
  /// upward amplitudes at the bottom and the downward ones at the top of each region, so all
  /// exponentials decay. @throws as `response`.
  [[nodiscard]] AnisotropicPlaneWave plane_wave(Real k0, Real angle, Polarisation pol,
                                                Real amplitude = 1.0, Real azimuth = 0.0) const;

  /// Jones reflection matrix (as `StackResponse::reflection`) at the normalised in-plane
  /// wavevector (a, b) = (k_x, k_y)/k0 of the incidence medium, also beyond its light cone,
  /// where the incident and the reflected wave are evanescent (k_z / k0 = ±sqrt(n0² − a² − b²),
  /// Im ≥ 0 towards the decaying side): the plane-wave spectrum of a dipole above the stack
  /// (`dipole_power_above`). ŝ = ẑ × k̂_∥ (ŷ at a = b = 0), p̂ = ŝ × k̂ / n0 with the complex k̂.
  /// Guided modes of a lossless stack are poles on the real (a, b) plane, where the matrix is
  /// singular or large. @throws InvalidArgument for k0 ≤ 0 or grazing propagation in a region
  /// (k_z = 0, e.g. |(a, b)| = n0).
  /// (a, b) may be complex: the analytic continuation on a deformed integration path below the
  /// real axis (guided-mode poles of lossless stacks, M19 S7e), with ŝ normalised by the
  /// non-conjugating norm and the up / down modes of every region chosen by Im k_z (causal).
  [[nodiscard]] Eigen::Matrix2cd reflection_at(Real k0, Complex a, Complex b) const;

  /// The analytic field of the downward mode `mode` (0 or 1, the order of `half_space_modes`:
  /// s / p, ordinary / extraordinary) of the incidence medium, with |E| = `amplitude` [V/m], at
  /// the normalised in-plane wavevector (a, b): as `plane_wave`, also for a lossless crystal
  /// incidence medium (M19 S7e: the plane wave incident from a crystal substrate, mirrored, in
  /// the reciprocity of emitters). `response` is left empty. The incident wave has the phase 1
  /// at the origin of the top interface.
  /// @throws InvalidArgument for an evanescent mode or a mode index other than 0, 1.
  [[nodiscard]] AnisotropicPlaneWave mode_wave(Real k0, Real a, Real b, int mode,
                                               Real amplitude = 1.0) const;

  /// Berreman matrix K of a medium for the normalised in-plane wavevector (a, b) = (k_x, k_y)/k0:
  /// @f$ \psi' = ik_0K\psi @f$ (exposed for tests and the theory page).
  [[nodiscard]] static Eigen::Matrix4cd berreman_matrix(const materials::Tensor3& eps, Real a,
                                                        Real b);

 private:
  materials::Material incidence_;
  std::vector<Layer> layers_;
  materials::Material substrate_;
  Real top_ = 0;
};

}  // namespace hpfem::physics

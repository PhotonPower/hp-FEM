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
  /// Tangential fields of the downward substrate modes, each with unit |E| (s and p of an
  /// isotropic substrate; the eigenmodes of the downward block of a crystal, or an orthonormal
  /// basis of the subspace where the two modes coincide).
  std::array<TangentialField, 2> substrate_modes;
  /// Vertical wavenumbers k_z of the downward substrate modes [1/m] (Im ≤ 0).
  std::array<Complex, 2> substrate_kz;
  /// Reflectance, transmittance (power flux into the substrate) and absorptance
  /// A = 1 − R − T of the finite layers, for incident s (entry 0) and p (entry 1).
  Eigen::Vector2d reflectance;
  Eigen::Vector2d transmittance;
  Eigen::Vector2d absorptance;
};

/// A planar stack of possibly anisotropic layers (laboratory frame, z the normal): isotropic,
/// lossless incidence medium above `top`, `layers` from top to bottom, semi-infinite substrate.
class AnisotropicStack {
 public:
  /// @throws InvalidArgument for a thickness ≤ 0, a magnetic material, an anisotropic or lossy
  ///         incidence medium, a gain medium (Im ε not positive semi-definite) or an
  ///         epsilon-near-zero @f$ |\varepsilon_{zz}| < 10^{-8}\max|\varepsilon_{ij}| @f$.
  AnisotropicStack(materials::Material incidence_medium, std::vector<Layer> layers,
                   materials::Material substrate, Real top = 0.0);

  [[nodiscard]] int num_layers() const noexcept { return static_cast<int>(layers_.size()); }
  [[nodiscard]] const materials::Material& material(int region) const;

  /// Response at the vacuum wavenumber k0 [1/m] for incidence at `angle` [rad] from the normal
  /// in the plane at the azimuth `azimuth` [rad] (in-plane wavevector
  /// @f$ k_0n_0\sin\theta(\cos\phi, \sin\phi) @f$).
  /// @throws InvalidArgument for k0 ≤ 0, |angle| ≥ π/2, or a region without two upward and two
  ///         downward modes (grazing propagation, k_z = 0).
  [[nodiscard]] StackResponse response(Real k0, Real angle, Real azimuth = 0.0) const;

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

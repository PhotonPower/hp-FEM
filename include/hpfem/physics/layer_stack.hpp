#pragma once
/// @file layer_stack.hpp
/// Planar layer stacks and their plane-wave solutions: the analytic background field of the
/// scattered-field formulation for structures embedded in layered media (ADR-0009). Layers are
/// perpendicular to the last coordinate (y in 2D, z in 3D); the incidence medium lies above
/// the top interface, the substrate below the bottom one. Convention exp(−iωt) (ADR-0002):
/// a lossy layer has Im ε > 0 and its vertical wavenumber Im k_z ≥ 0; the plane wave comes
/// from above and travels towards −z (−y in 2D). See docs/theory/maxwell.md#layered-background.

#include <vector>

#include "hpfem/core/types.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/physics/sources.hpp"

namespace hpfem::physics {

/// Polarisation of a plane wave on a layer stack: s has E perpendicular to the plane of
/// incidence, p has E in that plane (H perpendicular). In 2D the field of `Scattering<2>` is
/// the in-plane E, i.e. only p exists.
enum class Polarisation { kS, kP };

/// A homogeneous, non-magnetic layer of finite thickness [m].
struct Layer {
  materials::Material material;
  Real thickness = 0;
};

/// Plane wave on a layer stack, see `LayerStack::plane_wave`. `field` is the piecewise
/// analytic E (value and curl, usable as `ScatteringSetup::incident`); the amplitudes refer
/// to the scalar wave function u (E_s for s, H for p), `down[j]` at the top interface of layer
/// j and `up[j]` at its bottom interface (j = 0 incidence medium, j = N + 1 substrate), so that
/// all exponentials inside a layer decay. `reflection` / `transmission` are u-amplitude
/// ratios at the top / bottom interface, the power quantities are the reflectance R, the
/// transmittance T (power flux into the substrate, also when it is lossy) and the absorption
/// A = 1 − R − T in the finite layers of the bare stack.
template <int Dim>
struct LayeredPlaneWave {
  IncidentField<Dim> field;
  /// The incident plane wave alone (the downward wave of the incidence medium continued
  /// analytically everywhere): what a reflected field is measured against
  /// (`physics::diffraction_orders`, `power_balance`).
  IncidentField<Dim> incident_wave;
  Complex reflection{0.0, 0.0};
  Complex transmission{0.0, 0.0};
  Real reflectance = 0;
  Real transmittance = 0;
  Real absorptance = 0;
  std::vector<Complex> kz;  ///< vertical wavenumber per layer, Im ≥ 0
  std::vector<Complex> down, up;
};

/// A stack of planar layers: incidence medium (semi-infinite, lossless) above the coordinate
/// `top`, then `layers` from top to bottom, then the semi-infinite substrate. Interfaces must
/// coincide with mesh facets when the stack is used as background.
template <int Dim>
class LayerStack {
 public:
  static_assert(Dim == 2 || Dim == 3);
  /// @throws InvalidArgument for a thickness ≤ 0, a magnetic material, a lossy incidence
  /// medium or Im ε < 0 anywhere.
  LayerStack(materials::Material incidence_medium, std::vector<Layer> layers,
             materials::Material substrate, Real top = 0.0);

  [[nodiscard]] int num_layers() const noexcept { return static_cast<int>(layers_.size()); }
  /// Coordinate of interface i, i = 0 (top) … num_layers() (bottom); descending.
  [[nodiscard]] Real interface(int i) const;
  [[nodiscard]] Real top() const noexcept { return interface(0); }
  [[nodiscard]] Real bottom() const noexcept { return interface(num_layers()); }
  /// Material of the region containing the vertical coordinate z (an interface belongs to
  /// the region above it): index 0 incidence medium, 1..N layers, N + 1 substrate.
  [[nodiscard]] int region(Real z) const noexcept;
  [[nodiscard]] const materials::Material& material(int region) const;
  [[nodiscard]] const materials::Material& material_at(const Point<Dim>& x) const {
    return material(region(x(Dim - 1)));
  }
  [[nodiscard]] const materials::Material& incidence_medium() const { return material(0); }
  [[nodiscard]] const materials::Material& substrate() const { return material(num_layers() + 1); }

  /// Plane wave of unit amplitude |E| = `amplitude` [V/m] incident from above at the angle
  /// `angle` [rad] from the normal with in-plane wavevector along +x (3D: rotated by `azimuth`
  /// [rad] about the normal), vacuum wavenumber k0 [1/m]. Stable Airy / S-matrix recursion of
  /// the reflection coefficients from the substrate upwards and layer-local amplitude
  /// references, so thick absorbing layers neither overflow nor cancel. In 2D `pol` must be
  /// kP. @throws InvalidArgument for k0 ≤ 0, |angle| ≥ π/2, grazing incidence or kS in 2D.
  [[nodiscard]] LayeredPlaneWave<Dim> plane_wave(Real k0, Real angle,
                                                 Polarisation pol = Polarisation::kP,
                                                 Real amplitude = 1.0, Real azimuth = 0.0) const;

 private:
  materials::Material incidence_;
  std::vector<Layer> layers_;
  materials::Material substrate_;
  std::vector<Real> interfaces_;  ///< descending
};

extern template class LayerStack<2>;
extern template class LayerStack<3>;

}  // namespace hpfem::physics

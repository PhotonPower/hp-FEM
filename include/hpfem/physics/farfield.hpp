#pragma once
/// @file farfield.hpp
/// Far field of a field known on a closed surface (Stratton–Chu / Huygens): the equivalent
/// surface currents @f$ J = n\times H @f$ and @f$ M = -n\times E @f$ on a surface enclosing all
/// sources and scatterers radiate into the homogeneous background, and the far-field pattern
/// @f$ F(\hat r) @f$ with @f$ E \approx F(\hat r)\,e^{ikr}/r @f$ (3D) or @f$ F(\hat\varphi)\,
/// e^{ik\rho}/\sqrt\rho @f$ (2D) follows from the Fourier-type integrals
/// @f$ N = \int_S J\,e^{-ik\hat r\cdot x'}ds' @f$, @f$ L = \int_S M\,e^{-ik\hat r\cdot x'}ds' @f$:
/// @f$ F = \frac{ik}{4\pi}\,[Z N_t - \hat r\times L] @f$ in 3D (transverse part of N) and
/// @f$ F = -\frac{k}{4}\sqrt{2/(\pi k)}\,e^{-i\pi/4}\,[Z\,N\cdot\hat t + L_z]\,\hat t @f$ in 2D
/// with
/// @f$ \hat t = \hat z\times\hat r @f$. Convention exp(−iωt). See
/// docs/theory/maxwell.md#post-processing-quantities.

#include <vector>

#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/mesh.hpp"
#include "hpfem/physics/postprocess.hpp"

namespace hpfem::physics {

/// Far-field pattern of a field sampled on a closed surface (normal outwards). The surface
/// must lie in the homogeneous background medium and enclose every source and scatterer;
/// for the scattered field of a `Scattering` solution use `discrete_field` on a surface
/// inside the PML-free region.
template <int Dim>
class FarField {
 public:
  /// Samples E and H = curl E / (iωμ0 μr) on the surface with a rule of the given degree.
  FarField(const mesh::Mesh<Dim>& mesh, const Surface<Dim>& surface, const SurfaceField<Dim>& field,
           Real omega, const materials::Material& background, int order);

  /// @f$ F(\hat r) @f$ for a unit direction (normalised internally). [V/m · m] in 3D,
  /// [V/m · m^(1/2)] in 2D.
  [[nodiscard]] assembly::ComplexVector<Dim> pattern(const Point<Dim>& direction) const;

  /// Radiated power @f$ \int |F|^2 d\Omega / (2Z) @f$ [W] (3D) or @f$ \int |F|^2 d\varphi / (2Z)
  /// @f$ [W/m] (2D) from the pattern integrated over all directions (`resolution` angles per full
  /// turn).
  [[nodiscard]] Real radiated_power(int resolution = 180) const;
  /// Scattering cross-section @f$ \sigma = P / I @f$ for an incident plane wave of the given
  /// amplitude in the background medium.
  [[nodiscard]] Real scattering_cross_section(Real incident_amplitude, int resolution = 180) const;

  [[nodiscard]] Real wavenumber() const noexcept { return k_; }
  [[nodiscard]] Real impedance() const noexcept { return impedance_; }

 private:
  struct Sample {
    Point<Dim> x;
    Point<Dim> normal;
    Real weight;
    assembly::ComplexVector<Dim> e;
    assembly::ComplexCurl<Dim> h;  ///< H (vector in 3D, H_z in 2D)
  };
  std::vector<Sample> samples_;
  Real k_ = 0;
  Real impedance_ = 0;
};

extern template class FarField<2>;
extern template class FarField<3>;

}  // namespace hpfem::physics

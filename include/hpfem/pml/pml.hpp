#pragma once
/// @file pml.hpp
/// Perfectly matched layers as complex coordinate stretching around an axis-aligned
/// interior box: outside the box every coordinate ξ is continued as
/// @f$ \tilde\xi = \xi + i\int\hat\sigma(\xi)\,d\xi @f$ with the dimensionless profile
/// @f$ \hat\sigma = \sigma/\omega @f$, so that outgoing waves decay (convention exp(−iωt);
/// the imaginary part is **plus**). The stretching enters the Maxwell forms as the
/// anisotropic material tensors @f$ \tilde\varepsilon = \det\Lambda\,\Lambda^{-1}\varepsilon
/// \Lambda^{-T} @f$, @f$ \tilde\mu^{-1} = \Lambda\,\mu^{-1}\Lambda^{T}/\det\Lambda @f$ with
/// @f$ \Lambda = \mathrm{diag}(s_\xi) @f$, @f$ s_\xi = 1 + i\hat\sigma_\xi @f$; nothing else in
/// the assembly changes. Theory and profile: docs/theory/pml.md.

#include <array>
#include <cstddef>

#include <Eigen/Core>

#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/types.hpp"

namespace hpfem::pml {

/// Polynomial absorption profile @f$ \hat\sigma(d) = \hat\sigma_{\max}(d/\text{thickness})^m @f$
/// with @f$ \hat\sigma_{\max} = -\frac{(m+1)\ln R_0}{2\,k_0\,n\,\text{thickness}} @f$, so that a
/// wave at normal incidence returns from the far end with amplitude ratio @f$ R_0 @f$. At the
/// angle @f$ \theta @f$ against the layer normal the attenuation scales with @f$ \cos\theta @f$:
/// the round trip leaves @f$ R_0^{\cos\theta} @f$ and the far wall receives the one-way field
/// @f$ R_0^{\cos\theta/2} @f$, so a reflectance computed with the layer is wrong by up to
/// @f$ 2|r|R_0^{\cos\theta/2} @f$ (@f$ r @f$ the reflection amplitude of the structure). Choose
/// @f$ R_0 @f$ for the largest angle that occurs (`for_angle`); the thickness only sets how
/// steep the profile is (docs/theory/pml.md, "Oblique incidence").
struct PmlProfile {
  int order = 3;           ///< m
  Real reflection = 1e-8;  ///< R_0, round-trip field reflection at normal incidence

  /// @f$ R_0 @f$ such that the reflectance error of the layer stays below `target` up to the
  /// incidence angle `theta_max` [rad] against the layer normal for a structure of reflection
  /// amplitude `r_amplitude`: @f$ R_0 = (\text{target} / (2 r))^{2/\cos\theta_{\max}} @f$,
  /// clamped to @f$ [10^{-300}, 1) @f$. Steep profiles need thick layers: check
  /// `PmlBox::max_resolution` or use `PmlBox::recommended_thickness` with the profile.
  /// @throws InvalidArgument for θ outside [0, π/2), target ≤ 0, r ≤ 0 or order < 1.
  [[nodiscard]] static PmlProfile for_angle(Real theta_max, Real target, Real r_amplitude = 1.0,
                                            int order = 2);
};

/// Per-axis complex stretch factors @f$ s_\xi @f$ or stretched coordinates.
template <int Dim>
using ComplexCoordinates = Eigen::Matrix<Complex, Dim, 1>;

/// PML layers of given thickness on the sides of the interior box [lower, upper]. A side
/// with thickness 0 carries no layer (e.g. a periodic or PEC direction). The layers must be
/// covered by mesh cells; cell facets should coincide with the box faces.
template <int Dim>
class PmlBox {
 public:
  static constexpr std::size_t kNumSides = 2 * static_cast<std::size_t>(Dim);
  using Thickness = std::array<Real, kNumSides>;  ///< x-min, x-max, y-min, y-max[, z-min, z-max]

  /// @param k0 vacuum wavenumber [1/m]; @param background_index real part of the refractive
  ///        index of the medium the layer continues (sets σ_max); @throws InvalidArgument for
  ///        a degenerate box, negative thickness, k0 ≤ 0 or an unusable profile.
  PmlBox(const Point<Dim>& lower, const Point<Dim>& upper, const Thickness& thickness, Real k0,
         Real background_index = 1.0, PmlProfile profile = {});
  /// Layer thickness adapted to the wavelength and the mesh: `wavelengths` local wavelengths
  /// @f$ \lambda_0 / n @f$, rounded up to a whole number of cells of size `cell_size`, so the
  /// layer boundary falls on cell facets of a structured mesh. @throws InvalidArgument for
  /// non-positive arguments.
  [[nodiscard]] static Real recommended_thickness(Real k0, Real background_index, Real cell_size,
                                                  Real wavelengths = 0.5);
  /// The smallest thickness (whole cells of size `cell_size`) at which the profile stays
  /// resolved for polynomial order `p`: @f$ |k s| h \le @f$ `resolution_limit(p)` at the far
  /// end of the layer, where @f$ |s| = \sqrt{1 + \hat\sigma_{\max}^2} @f$ and
  /// @f$ \hat\sigma_{\max} \propto 1 / \text{thickness} @f$. At least the thickness of the
  /// half-wavelength rule. @throws InvalidArgument if even an unstretched layer is not
  /// resolved (@f$ k_0 n h @f$ above the limit) or for non-positive arguments.
  [[nodiscard]] static Real recommended_thickness(Real k0, Real background_index, Real cell_size,
                                                  const PmlProfile& profile, int p);
  /// Largest @f$ |k s| h @f$ the stretched field may have on cells of order `p`: 3 for
  /// @f$ p \ge 4 @f$ (docs/theory/pml.md), @f$ 0.75\,p @f$ below.
  [[nodiscard]] static Real resolution_limit(int p) noexcept;
  /// The same thickness on all sides.
  [[nodiscard]] static PmlBox uniform(const Point<Dim>& lower, const Point<Dim>& upper,
                                      Real thickness, Real k0, Real background_index = 1.0,
                                      PmlProfile profile = {});

  [[nodiscard]] const Point<Dim>& lower() const noexcept { return lower_; }
  [[nodiscard]] const Point<Dim>& upper() const noexcept { return upper_; }
  [[nodiscard]] const Thickness& thickness() const noexcept { return thickness_; }
  [[nodiscard]] const PmlProfile& profile() const noexcept { return profile_; }
  /// @f$ \hat\sigma_{\max} @f$ of side k (0 for a side without layer).
  [[nodiscard]] Real sigma_max(std::size_t side) const noexcept { return sigma_max_[side]; }
  /// Vacuum wavenumber the layers were designed for.
  [[nodiscard]] Real k0() const noexcept { return k0_; }
  /// Largest @f$ |k s| h @f$ over the layers for cells of size `cell_size` in a medium of real
  /// index `index` (@f$ k = k_0 n @f$, @f$ s @f$ at the far end of each layer): the
  /// resolution the stretched field @f$ e^{iks\xi} @f$ asks of the mesh, to be kept below
  /// `resolution_limit(p)`. A layer meshed with a medium of higher index than the
  /// `background_index` it was designed for (a substrate) is the usual offender.
  [[nodiscard]] Real max_resolution(Real cell_size, Real index) const;
  /// Outer boundary of the layers: the box enlarged by the thicknesses.
  [[nodiscard]] Point<Dim> outer_lower() const;
  [[nodiscard]] Point<Dim> outer_upper() const;

  /// True if x lies in at least one layer (some @f$ s_\xi \ne 1 @f$).
  [[nodiscard]] bool in_layer(const Point<Dim>& x) const;
  /// Stretch factors @f$ s_\xi(x) = 1 + i\hat\sigma_\xi @f$ per axis (1 inside the box).
  [[nodiscard]] ComplexCoordinates<Dim> stretch(const Point<Dim>& x) const;
  /// Stretched coordinates @f$ \tilde x_\xi = x_\xi + i\int_{\text{box}}^{x_\xi}\hat\sigma @f$
  /// (closed form for the polynomial profile); an outgoing analytic field evaluated at
  /// @f$ \tilde x @f$ is the exact solution inside the layer.
  [[nodiscard]] ComplexCoordinates<Dim> stretched_coordinate(const Point<Dim>& x) const;

  /// Effective permittivity tensor @f$ \varepsilon_r\det\Lambda\,\Lambda^{-2} @f$ of an
  /// isotropic material (diagonal).
  [[nodiscard]] assembly::PermittivityTensor<Dim> permittivity(Complex eps_r,
                                                               const Point<Dim>& x) const;
  /// Effective inverse permeability acting on curls: @f$ \mu_r^{-1}\Lambda^2/\det\Lambda @f$
  /// (3D, diagonal) or the scalar @f$ 1/(\mu_r s_x s_y) @f$ (2D, out-of-plane curl).
  [[nodiscard]] assembly::InversePermeabilityTensor<Dim> inverse_permeability(
      Complex mu_r, const Point<Dim>& x) const;

 private:
  /// Signed distance into the layers along axis d: negative beyond the lower side, positive
  /// beyond the upper side, zero inside the box.
  [[nodiscard]] Real depth(const Point<Dim>& x, int axis) const;

  Point<Dim> lower_;
  Point<Dim> upper_;
  Thickness thickness_{};
  PmlProfile profile_;
  Real k0_ = 0;
  std::array<Real, kNumSides> sigma_max_{};
};

extern template class PmlBox<2>;
extern template class PmlBox<3>;

}  // namespace hpfem::pml

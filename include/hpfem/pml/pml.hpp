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
/// wave at normal incidence returns from the far end with amplitude ratio @f$ R_0 @f$.
struct PmlProfile {
  int order = 3;           ///< m
  Real reflection = 1e-8;  ///< R_0, theoretical round-trip reflection at normal incidence
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
  std::array<Real, kNumSides> sigma_max_{};
};

extern template class PmlBox<2>;
extern template class PmlBox<3>;

}  // namespace hpfem::pml

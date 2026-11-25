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
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <variant>
#include <vector>

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
  /// Effective permittivity tensor of an anisotropic material,
  /// @f$ \tilde\varepsilon_{ij} = \varepsilon_{ij}\det\Lambda/(s_i s_j) @f$ (ADR-0015 §2); in 2D
  /// `eps` is the in-plane block and @f$ s_z = 1 @f$. Equals the scalar overload for
  /// @f$ \varepsilon = \varepsilon_r I @f$.
  [[nodiscard]] assembly::PermittivityTensor<Dim> permittivity(
      const assembly::PermittivityTensor<Dim>& eps, const Point<Dim>& x) const;
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

/// Jacobian @f$ J = \partial\tilde x / \partial x @f$ of a complex coordinate stretch.
template <int Dim>
using StretchMatrix = Eigen::Matrix<Complex, Dim, Dim>;

/// Shape of a `RadialPml`.
enum class RadialShape : std::uint8_t {
  kCircular,     ///< 2D: stretching of the distance from the centre
  kSpherical,    ///< 3D: stretching of the distance from the centre
  kCylindrical,  ///< 3D: stretching of the distance from an axis, optional layers along it
};

/// PML of radial shape (issue #244 S4, docs/theory/pml.md "Radial layers"): beyond the
/// distance @f$ r_0 @f$ from the centre (circle, sphere) or from the axis (cylinder) the distance
/// is continued as @f$ \tilde r = r + i\int_{r_0}^{r}\hat\sigma @f$ with the profile of
/// `PmlProfile`, and the point as @f$ \tilde x = c + (\tilde r/r)(x - c) @f$ (cylinder: the part
/// perpendicular to the axis). The Jacobian
/// @f$ J = s_r\,\hat r\hat r^T + (\tilde r/r)(P - \hat r\hat r^T) + (I - P) @f$ (@f$ P @f$ the
/// projector onto the stretched plane or space, @f$ s_r = 1 + i\hat\sigma(r) @f$) is a full
/// symmetric tensor in Cartesian components; the materials become
/// @f$ \tilde\varepsilon = \det J\,J^{-1}\varepsilon J^{-T} @f$ and
/// @f$ \tilde\mu^{-1} = J\mu^{-1}J^T/\det J @f$ exactly as for `PmlBox`. Unlike the box, the
/// layer is invariant under every rotation about the centre (axis), so a sector of a structure
/// with cyclic symmetry C_n of any order can carry it. The cylinder takes, in addition, slab
/// layers on the two ends of an interval along the axis (as a `PmlBox` would). Points beyond the
/// layer keep the final stretch @f$ \hat\sigma_{\max} @f$ (a square mesh around a circular layer
/// is allowed). Convention exp(−iωt): the imaginary part of the stretch is positive.
template <int Dim>
class RadialPml {
 public:
  /// Circular (2D) or spherical (3D) layer @f$ r_0 \le |x - c| \le r_0 + d @f$.
  /// @param k0 vacuum wavenumber [1/m]; @param background_index real refractive index of the
  ///        medium the layer continues (sets @f$ \hat\sigma_{\max} @f$ as for `PmlBox`).
  /// @throws InvalidArgument for r0 ≤ 0, thickness ≤ 0, k0 ≤ 0 or an unusable profile.
  RadialPml(const Point<Dim>& center, Real inner_radius, Real thickness, Real k0,
            Real background_index = 1.0, PmlProfile profile = {});
  /// Cylindrical layer (3D): distance @f$ \rho @f$ from the axis through `center` along `axis`
  /// stretched beyond `inner_radius`, and the axial coordinate @f$ a = \hat a\cdot(x - c) @f$
  /// stretched below `axial_lower` and above `axial_upper` over `axial_thickness` (0: no axial
  /// layers, e.g. a periodic or PEC direction). @throws InvalidArgument for a zero axis,
  /// axial_lower ≥ axial_upper, negative axial thickness or the conditions of the constructor.
  [[nodiscard]] static RadialPml cylindrical(const Point<3>& center, const Point<3>& axis,
                                             Real inner_radius, Real thickness, Real axial_lower,
                                             Real axial_upper, Real axial_thickness, Real k0,
                                             Real background_index = 1.0, PmlProfile profile = {})
    requires(Dim == 3);

  [[nodiscard]] RadialShape shape() const noexcept { return shape_; }
  [[nodiscard]] const Point<Dim>& center() const noexcept { return center_; }
  /// Unit axis of the cylinder (the z axis for the other shapes).
  [[nodiscard]] const Point<3>& axis() const noexcept { return axis_; }
  [[nodiscard]] Real inner_radius() const noexcept { return inner_radius_; }
  [[nodiscard]] Real thickness() const noexcept { return thickness_; }
  [[nodiscard]] Real axial_lower() const noexcept { return axial_lower_; }
  [[nodiscard]] Real axial_upper() const noexcept { return axial_upper_; }
  [[nodiscard]] Real axial_thickness() const noexcept { return axial_thickness_; }
  [[nodiscard]] const PmlProfile& profile() const noexcept { return profile_; }
  [[nodiscard]] Real k0() const noexcept { return k0_; }
  /// @f$ \hat\sigma_{\max} @f$ of the radial layer.
  [[nodiscard]] Real sigma_max() const noexcept { return sigma_max_; }
  /// @f$ \hat\sigma_{\max} @f$ of the axial layers (0 without them).
  [[nodiscard]] Real axial_sigma_max() const noexcept { return axial_sigma_max_; }
  /// Largest @f$ |k s| h @f$ in the layer (see `PmlBox::max_resolution`); the factor
  /// @f$ |\tilde r/r| @f$ of the tangential directions never exceeds the radial @f$ |s_r| @f$.
  [[nodiscard]] Real max_resolution(Real cell_size, Real index) const;

  /// True if x lies in the layer (radially beyond r0 or axially beyond the interval).
  [[nodiscard]] bool in_layer(const Point<Dim>& x) const;
  /// Directions of stretching at x: the radial unit vector and, in an axial layer, the axis.
  [[nodiscard]] std::vector<Point<Dim>> stretch_directions(const Point<Dim>& x) const;
  /// Jacobian @f$ J(x) @f$ of the stretch (identity outside the layer).
  [[nodiscard]] StretchMatrix<Dim> jacobian(const Point<Dim>& x) const;
  /// Stretched point @f$ \tilde x @f$ (closed form for the polynomial profile); an outgoing
  /// analytic field E gives the exact solution @f$ J^T E(\tilde x) @f$ inside the layer.
  [[nodiscard]] ComplexCoordinates<Dim> stretched_coordinate(const Point<Dim>& x) const;

  /// Effective permittivity @f$ \varepsilon_r\det J\,J^{-1}J^{-T} @f$ of an isotropic material.
  [[nodiscard]] assembly::PermittivityTensor<Dim> permittivity(Complex eps_r,
                                                               const Point<Dim>& x) const;
  /// Effective permittivity @f$ \det J\,J^{-1}\varepsilon J^{-T} @f$ of an anisotropic material
  /// (2D: the in-plane block; the out-of-plane stretch is 1).
  [[nodiscard]] assembly::PermittivityTensor<Dim> permittivity(
      const assembly::PermittivityTensor<Dim>& eps, const Point<Dim>& x) const;
  /// Effective inverse permeability on curls: @f$ J J^T/(\mu_r\det J) @f$ (3D) or the scalar
  /// @f$ 1/(\mu_r\det J) @f$ (2D, out-of-plane curl).
  [[nodiscard]] assembly::InversePermeabilityTensor<Dim> inverse_permeability(
      Complex mu_r, const Point<Dim>& x) const;

 private:
  RadialPml() = default;
  /// Distance from the centre (axis) and the unit radial vector (zero at the centre).
  [[nodiscard]] std::pair<Real, Point<Dim>> radial(const Point<Dim>& x) const;
  /// Signed depth into the axial layers (0 inside the interval or without axial layers).
  [[nodiscard]] Real axial_depth(const Point<Dim>& x) const;

  RadialShape shape_ = RadialShape::kCircular;
  Point<Dim> center_ = Point<Dim>::Zero();
  Point<3> axis_ = Point<3>::UnitZ();
  Real inner_radius_ = 0;
  Real thickness_ = 0;
  Real axial_lower_ = 0;
  Real axial_upper_ = 0;
  Real axial_thickness_ = 0;
  PmlProfile profile_;
  Real k0_ = 0;
  Real sigma_max_ = 0;
  Real axial_sigma_max_ = 0;
};

/// The absorbing layers of a problem: a `PmlBox` or a `RadialPml`, with the operations the
/// solvers need (layer membership, Jacobian, effective tensors, resolution). Converts
/// implicitly from both, so `setup.pml = PmlBox<2>::uniform(...)` keeps working.
template <int Dim>
class Pml {
 public:
  Pml(PmlBox<Dim> box) : layer_(std::move(box)) {}           // NOLINT(google-explicit-constructor)
  Pml(RadialPml<Dim> radial) : layer_(std::move(radial)) {}  // NOLINT(google-explicit-constructor)

  /// The box, or null for a radial layer.
  [[nodiscard]] const PmlBox<Dim>* box() const noexcept {
    return std::get_if<PmlBox<Dim>>(&layer_);
  }
  /// The radial layer, or null for a box.
  [[nodiscard]] const RadialPml<Dim>* radial() const noexcept {
    return std::get_if<RadialPml<Dim>>(&layer_);
  }
  [[nodiscard]] Real k0() const;
  [[nodiscard]] bool in_layer(const Point<Dim>& x) const;
  /// Unit directions of stretching at x (box: the axes with a layer at x).
  [[nodiscard]] std::vector<Point<Dim>> stretch_directions(const Point<Dim>& x) const;
  /// Jacobian @f$ J = \partial\tilde x/\partial x @f$ (box: diag of the stretch factors).
  [[nodiscard]] StretchMatrix<Dim> jacobian(const Point<Dim>& x) const;
  [[nodiscard]] ComplexCoordinates<Dim> stretched_coordinate(const Point<Dim>& x) const;
  [[nodiscard]] Real max_resolution(Real cell_size, Real index) const;
  [[nodiscard]] assembly::PermittivityTensor<Dim> permittivity(Complex eps_r,
                                                               const Point<Dim>& x) const;
  [[nodiscard]] assembly::PermittivityTensor<Dim> permittivity(
      const assembly::PermittivityTensor<Dim>& eps, const Point<Dim>& x) const;
  [[nodiscard]] assembly::InversePermeabilityTensor<Dim> inverse_permeability(
      Complex mu_r, const Point<Dim>& x) const;
  /// True if the layers are invariant under the orthogonal map @f$ x \mapsto Qx + a @f$ at the
  /// given points: @f$ J(Qx + a) = Q J(x) Q^T @f$ up to `tolerance` relative to
  /// @f$ \max(1, |J(x)|) @f$. Symmetry reductions check this on the cells of the reduced mesh.
  [[nodiscard]] bool is_invariant(const Eigen::Matrix<Real, Dim, Dim>& q, const Point<Dim>& shift,
                                  std::span<const Point<Dim>> points, Real tolerance = 1e-8) const;

 private:
  std::variant<PmlBox<Dim>, RadialPml<Dim>> layer_;
};

/// A wave that violates the geometric stability condition of the coordinate-stretching PML.
template <int Dim>
struct BackwardWave {
  /// Unit wave vector (phase direction) of the worst sampled wave.
  Point<Dim> direction = Point<Dim>::Zero();
  /// @f$ \min \hat k_d\,\hat v_d @f$ over the sampled waves (unit phase and group directions):
  /// negative where the phase travels into the layer and the energy out of it (or back).
  Real worst = 0;
};

/// Geometric stability condition of a PML that stretches along `axis` in a homogeneous medium
/// of relative permittivity `eps` (Bécache, Fauqueux & Joly 2003; ADR-0015 §2): every
/// propagating wave must have the components of its phase and its group velocity along the
/// stretching direction of the same sign, @f$ k_d\,v_{g,d} \ge 0 @f$, otherwise the stretch
/// amplifies it. Checks the real part of `eps` (in 2D the in-plane block, waves with in-plane k
/// and E, slowness curve @f$ k^T R^T\varepsilon^{-1}R\,k = k_0^2 @f$; in 3D both sheets of the
/// Fresnel surface, group velocity along the Poynting vector
/// @f$ \hat k|E|^2 - E(\hat k\cdot E) @f$) on `samples` wave directions (a half circle in 2D, a
/// Fibonacci hemisphere in 3D; k and −k give the same product). Isotropic media and uniaxial
/// crystals with the optic axis along `axis` or perpendicular to it never fail; a tilted axis
/// does, already for an ordinary uniaxial crystal, and so does an aligned **biaxial** crystal
/// along its intermediate principal axis (near the optic axes the group velocity sweeps the cone
/// of internal conical refraction). Directions with no propagating wave (metals, evanescent
/// sheets) are skipped. A tilt below the sampling resolution (about @f$ \pi/\text{samples} @f$ in
/// 2D) may go unnoticed; its amplification is correspondingly small.
/// @returns the worst wave if the condition fails (`worst` < −1e-9), nothing otherwise.
/// @throws InvalidArgument for an axis outside [0, Dim) or fewer than 8 samples.
template <int Dim>
[[nodiscard]] std::optional<BackwardWave<Dim>> backward_wave(
    const assembly::PermittivityTensor<Dim>& eps, int axis, int samples = 2048);

/// `backward_wave` for the stretching direction `direction` (any non-zero vector; the
/// tensor is rotated so that it becomes the first axis). The returned phase direction is in
/// the original frame. Used for radial layers, whose stretching direction turns with the point.
/// @throws InvalidArgument for a zero direction or fewer than 8 samples.
template <int Dim>
[[nodiscard]] std::optional<BackwardWave<Dim>> backward_wave_along(
    const assembly::PermittivityTensor<Dim>& eps, const Point<Dim>& direction, int samples = 2048);

extern template class PmlBox<2>;
extern template class PmlBox<3>;
extern template class RadialPml<2>;
extern template class RadialPml<3>;
extern template class Pml<2>;
extern template class Pml<3>;

}  // namespace hpfem::pml

#pragma once
/// @file conical_postprocess.hpp
/// Post-processing of the conical solver (M15 F10 / F11): diffraction orders on any line with
/// complex vector amplitudes, the flux-based power balance, cross-sections of isolated
/// scatterers and the far-field pattern of the 2.5D field, plus the frame conversion to the
/// grating literature. All fields are physical @f$ (E_x, E_y, E_z) @f$ of @f$ E(x,y)e^{i\beta z}
/// @f$; the solver frame has the period along x, the normal along y and the invariant direction z,
/// the literature frame (BEM / RCWA) the period along x', the invariant direction y' and the
/// normal z': @f$ (x', y', z') = (x, -z, y) @f$ (right-handed). Convention exp(-iωt). See
/// docs/theory/maxwell.md#conical-incidence-and-the-e_z-polarisation.

#include <functional>
#include <optional>
#include <vector>

#include "hpfem/core/types.hpp"
#include "hpfem/physics/conical_scattering.hpp"
#include "hpfem/physics/diffraction.hpp"
#include "hpfem/physics/postprocess.hpp"

namespace hpfem::physics {

/// A physical 3-vector field as a function of the in-plane point.
using ConicalFieldFunction = std::function<ConicalVector(const Point<2>&)>;

/// Solver frame → literature frame, @f$ (v_x, v_y, v_z) \mapsto (v_x, -v_z, v_y) @f$.
[[nodiscard]] ConicalVector to_literature_frame(const ConicalVector& v);
/// The inverse.
[[nodiscard]] ConicalVector from_literature_frame(const ConicalVector& v);

/// Curl of a physical 2.5D field by central differences of the in-plane derivatives
/// (`step`, in metres) with @f$ \partial_z = i\beta @f$:
/// @f$ (\partial_yE_z - i\beta E_y,\ i\beta E_x - \partial_xE_z,\ \partial_xE_y - \partial_yE_x)
/// @f$.
[[nodiscard]] ConicalVector conical_curl_of(const ConicalFieldFunction& field, const Point<2>& x,
                                            Real beta, Real step);

/// Diffraction orders of `field - incident` (or of `field` when `incident` is empty) on the
/// line, as `diffraction_orders` for the in-plane solver: @f$ A_m = \frac1a\int_0^a (E -
/// E^{inc})(o + t\hat t)e^{-ik_{t,m}t}dt @f$ with the composite Gauss–Legendre rule of
/// `conical_fourier_coefficients` (`num_points` 0: 16 per order, at least 64), the normal
/// wavenumber @f$ k_{n,m} = \sqrt{k_0^2n^2 - k_{t,m}^2 - \beta^2} @f$ in the medium of the line
/// and the efficiency @f$ \mathrm{Re}(k_{n,m})|A_m|^2/(k_n^{inc}|E_0|^2) @f$. The amplitudes are
/// in the solver frame (`to_literature_frame` converts them).
/// @throws InvalidArgument for a degenerate line or non-positive parameters.
[[nodiscard]] std::vector<ConicalDiffractionOrder> conical_diffraction_orders(
    const ConicalFieldFunction& field, const OrderLine& line, Real k0, Real index_line, Real kt0,
    Real beta, Real kn_incident, const ConicalFieldFunction& incident = {}, int max_order = 3,
    int num_points = 0, Real incident_amplitude = 1.0);

/// Flux-based energy balance of a periodic conical problem (scattered-field formulation):
/// the incident power per period @f$ \tfrac12|E_0|^2\,k_n^{inc}/(k_0Z_0)\,a @f$, the power of
/// the reflected field (total field minus the downward incident wave `incident_wave`, whose
/// H is taken by central differences) through `reflection` (a `Surface::plane` in the
/// incidence medium, normal away from the structure), the transmitted power of the total
/// field through `transmission` (or 0) and the volumetric absorbed power of the total field
/// (`absorbed_power_by_tag`); `relative_residual()` is the reference-free quality indicator.
/// [W/m]. @throws InvalidArgument without an incident field.
[[nodiscard]] PowerBalance conical_power_balance(
    const ConicalScattering& problem, const ConicalSolution& solution, const Surface<2>& reflection,
    Real period, Real kn_incident, const ConicalFieldFunction& incident_wave,
    Real incident_amplitude = 1.0, const Surface<2>* transmission = nullptr, int extra_order = 2);

/// Cross-sections per unit length [m] of an isolated scatterer (scattered-field formulation
/// in a homogeneous background): @f$ \sigma_{sca} @f$ from the flux of the unknown (scattered)
/// field outwards through the closed `surface` (normal outwards, e.g. `Surface::around_cells`
/// of a tagged region around the scatterer in the background), @f$ \sigma_{abs} @f$ from the
/// volumetric absorbed power of the total field, @f$ \sigma_{ext} = \sigma_{sca} + \sigma_{abs}
/// @f$, all divided by the incident intensity @f$ \tfrac12 n|E_0|^2/Z_0 @f$ of the lossless
/// background. @throws InvalidArgument without an incident field or with a lossy background.
[[nodiscard]] CrossSections conical_cross_sections(const ConicalScattering& problem,
                                                   const ConicalSolution& solution,
                                                   const Surface<2>& surface,
                                                   Real incident_amplitude = 1.0,
                                                   int extra_order = 2);

/// Far-field pattern of the scattered 2.5D field sampled on a closed surface in the lossless
/// homogeneous background (normal outwards). With @f$ k = k_0n @f$, the transverse wavenumber
/// @f$ k_t = \sqrt{k^2-\beta^2} @f$, the direction @f$ \hat k = (k_t\cos\varphi, k_t\sin\varphi,
/// \beta)/k @f$ and the equivalent currents @f$ N = \oint (n\times H)e^{-ik_t\hat\varphi\cdot x'}ds
/// @f$,
/// @f$ L = -\oint (n\times E)e^{-ik_t\hat\varphi\cdot x'}ds @f$ (in-plane normal n), the field
/// is @f$ E \to F(\varphi)\,e^{ik_t\rho}/\sqrt\rho\,e^{i\beta z} @f$ with
/// @f$ F = \tfrac{k}{4}\sqrt{\tfrac{2}{\pi k_t}}e^{-i\pi/4}\big(\hat k\times L - Z N_\perp\big)
/// @f$,
/// @f$ N_\perp = N - (N\cdot\hat k)\hat k @f$ (the 3D Stratton–Chu far field integrated along
/// z). The radiated power per unit length is @f$ \tfrac{k_t}{k}\oint |F|^2/(2Z)\,d\varphi @f$.
/// At β = 0 with an in-plane field it agrees with `FarField<2>`.
/// @throws InvalidArgument for a lossy background, @f$ |\beta| \ge k @f$ or an empty surface.
class ConicalFarField {
 public:
  ConicalFarField(const ConicalScattering& problem, const ConicalSolution& solution,
                  const Surface<2>& surface, int extra_order = 2);
  /// @f$ F(\varphi) @f$ for the angle φ of the in-plane direction [V/m · m^(1/2)].
  [[nodiscard]] ConicalVector pattern(Real phi) const;
  /// Radiated power per unit length [W/m] from `resolution` angles per full turn.
  [[nodiscard]] Real radiated_power(int resolution = 360) const;
  /// @f$ \sigma_{sca} = P/I @f$ [m] for the incident amplitude in the background.
  [[nodiscard]] Real scattering_cross_section(Real incident_amplitude, int resolution = 360) const;
  [[nodiscard]] Real wavenumber() const noexcept { return k_; }
  [[nodiscard]] Real transverse_wavenumber() const noexcept { return kt_; }
  [[nodiscard]] Real impedance() const noexcept { return impedance_; }

 private:
  struct Sample {
    Point<2> x;
    Point<2> normal;
    Real weight;
    ConicalVector e;
    ConicalVector h;
  };
  std::vector<Sample> samples_;
  Real k_ = 0;
  Real kt_ = 0;
  Real beta_ = 0;
  Real impedance_ = 0;
};

}  // namespace hpfem::physics

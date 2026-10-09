#pragma once
/// @file eigen_sensitivity.hpp
/// Derivatives of resonances (quasi-normal modes) with respect to parameters (M16 S4,
/// ADR-0012 §4): the eigenvalue @f$ \lambda = k_0^2 = (\omega/c_0)^2 @f$ of the pencil
/// @f$ A x = \lambda B x @f$ (stiffness and mass, PML-stretched, lossy, reduced by the PEC,
/// hanging-node and Bloch constraints) changes for a simple eigenvalue by
/// @f[ \frac{d\lambda}{dp} = \frac{y^\top(\partial_p A - \lambda\,\partial_p B)\,x}
///                                 {y^\top B\,x} @f]
/// with the right eigenvector x and the left eigenvector y, @f$ A^\top y = \lambda B^\top y
/// @f$ (unconjugated: the pencils are non-Hermitian with PML and losses). The element
/// matrices are complex symmetric, so without Bloch phases the left vector is the mode itself
/// (`Resonance`, also with the real hanging-node constraints); the Bloch-reduced pencil
/// @f$ P^H A P @f$ of `ConicalResonance` has the transpose @f$ P^\top A \bar P @f$, the pencil
/// at @f$ -k_x @f$, and its left vector comes from a transposed inverse iteration at λ. In
/// full DoF coefficients, with @f$ \tilde y = \bar P y @f$ and @f$ x = P x_r @f$,
/// @f$ d\lambda/dp = \tilde y^\top(\partial_p S - \lambda\,\partial_p M)\,x / (\tilde y^\top M
/// x) @f$. Then @f$ d\omega/dp = c_0\,(d\lambda/dp)/(2k_0) @f$ (complex: the real part moves
/// the resonance, the imaginary part its width), @f$ Q = \mathrm{Re}\,\omega/(-2\,\mathrm{Im}\,
/// \omega) @f$ and its derivative follow. Parameters: the permittivity of a tag (holomorphic,
/// @f$ \partial_\varepsilon M = M_{\text{tag}} @f$), a mesh velocity (central directional
/// differences of the element matrices, ADR-0011), the longitudinal wavenumber β and the Bloch
/// wavenumber (the derivative of P, as in `parameter_sensitivity.hpp`). The PML stretch is the
/// one of the setup (designed at the target frequency, fixed), so the pencil is linear in λ.
/// Time convention @f$ e^{-i\omega t} @f$: a decaying mode has Im ω < 0. See
/// docs/theory/maxwell.md "Resonance derivatives".

#include "hpfem/core/types.hpp"
#include "hpfem/physics/conical_resonance.hpp"
#include "hpfem/physics/resonance.hpp"
#include "hpfem/physics/shape_sensitivity.hpp"

namespace hpfem::physics {

/// Derivative of one resonance with respect to one real or complex parameter p.
struct ResonanceDerivative {
  Complex dlambda;       ///< @f$ d(k_0^2)/dp @f$ [1/m² per unit of p]
  Complex domega;        ///< @f$ d\omega/dp @f$ [rad/s per unit of p]
  Real dquality = 0;     ///< @f$ dQ/dp @f$ (for a real parameter)
  Real dwavelength = 0;  ///< @f$ d(2\pi c_0/\mathrm{Re}\,\omega)/dp @f$ [m per unit of p]
};

/// The derivative of ω, Q and the wavelength from the derivative @f$ \partial_p\hat\lambda @f$
/// of the eigenvalue of the pencil at the mode's ω. With dispersive materials the resonance
/// solves @f$ \hat\lambda(\omega, p) = (\omega/c_0)^2 @f$, where the pencil's eigenvalue
/// @f$ \hat\lambda @f$ depends on ω through @f$ \varepsilon(\omega) @f$:
/// @f[ \frac{d\omega}{dp} = \frac{\partial_p\hat\lambda}{2\omega/c_0^2 -
///     \partial_\omega\hat\lambda},\qquad \partial_\omega\hat\lambda = \sum_t
///     \frac{\partial\hat\lambda}{\partial\varepsilon_t}\,\varepsilon_t'(\omega) @f]
/// (`dlambda_domega`, from the material derivatives of the dispersive tags); 0 (the default)
/// for a linear pencil, @f$ d\omega/dp = c_0\,\partial_p\hat\lambda/(2k_0) @f$. `dlambda` of the
/// result is the total change of @f$ (\omega/c_0)^2 @f$.
[[nodiscard]] ResonanceDerivative resonance_derivative_from(Complex omega, Complex dlambda,
                                                            Complex dlambda_domega = {});

/// Left eigenvector of a mode in full DoF coefficients, @f$ \tilde y = \bar P y @f$ (zero on
/// PEC DoFs), and the normalisation @f$ \tilde y^\top M x @f$; for the conical block space
/// stacked (in-plane | scaled longitudinal).
struct ModeAdjoint {
  Vector field;           ///< @f$ \tilde y @f$ (full size)
  Complex normalisation;  ///< @f$ \tilde y^\top M x @f$ (non-zero for a simple eigenvalue)
  Vector reduced;         ///< y on the reduced unknowns (empty without constraints)
  Vector mode_reduced;    ///< @f$ x_r @f$ on the reduced unknowns (empty without constraints)
  Real residual = 0;      ///< @f$ \|(A - \lambda B)^\top y\| / (\|A\|_1 \|y\|) @f$ (0 if y = x)
};

/// The adjoint of a mode of `Resonance` (the mode itself; one mass assembly).
/// @throws InvalidArgument if the field does not match the DoF map.
template <int Dim>
[[nodiscard]] ModeAdjoint resonance_adjoint(const Resonance<Dim>& problem,
                                            const ResonantMode& mode);

/// @f$ d/d\varepsilon_r @f$ of the cells with `tag` (complex, holomorphic: the derivative with
/// respect to Im ε is i times it; cells in the PML keep their stretch).
/// @throws InvalidArgument if no cell has the tag or the sizes do not match.
template <int Dim>
[[nodiscard]] ResonanceDerivative resonance_material_derivative(const Resonance<Dim>& problem,
                                                                const ResonantMode& mode,
                                                                const ModeAdjoint& adjoint,
                                                                mesh::Tag tag);

/// Derivative along a mesh velocity (array `(num_geometry_nodes, Dim)`, ADR-0011), by one
/// central directional difference of the element matrices per moving cell (largest node
/// displacement `relative_step` cell diameters).
/// @throws InvalidArgument if the sizes do not match or the step is not positive.
template <int Dim>
[[nodiscard]] ResonanceDerivative resonance_shape_derivative(const Resonance<Dim>& problem,
                                                             const ResonantMode& mode,
                                                             const ModeAdjoint& adjoint,
                                                             const NodeField& velocity,
                                                             Real relative_step = 1e-6);

/// The adjoint of a conical resonance: the mode itself without Bloch phases, else the left
/// eigenvector of the reduced pencil by two steps of transposed inverse iteration at
/// @f$ \lambda(1 + 10^{-10}) @f$ (one factorisation, `ConicalResonanceSetup::solver`).
/// @throws InvalidArgument if the mode does not match the maps.
[[nodiscard]] ModeAdjoint conical_resonance_adjoint(const ConicalResonance& problem,
                                                    const ConicalResonantMode& mode);

/// @f$ d/d\varepsilon_r @f$ of the cells with `tag` for a conical resonance.
/// @throws InvalidArgument as `resonance_material_derivative`.
[[nodiscard]] ResonanceDerivative conical_resonance_material_derivative(
    const ConicalResonance& problem, const ConicalResonantMode& mode, const ModeAdjoint& adjoint,
    mesh::Tag tag);

/// Derivative of a conical resonance along a mesh velocity.
/// @throws InvalidArgument as `resonance_shape_derivative`.
[[nodiscard]] ResonanceDerivative conical_resonance_shape_derivative(
    const ConicalResonance& problem, const ConicalResonantMode& mode, const ModeAdjoint& adjoint,
    const NodeField& velocity, Real relative_step = 1e-6);

/// @f$ d/d\beta @f$ (longitudinal wavenumber, [1/m]) of a conical resonance: central
/// difference of the assembled pencil at β ± `step` (the Bloch constraints do not depend on β).
/// `step` is absolute [1/m]: take about @f$ 10^{-6}\max(|\beta|, k_0) @f$ (a step far below
/// that leaves only round-off of the assembled matrices; `hpfem.grating.resonance_sensitivity`
/// scales it so).
/// @throws InvalidArgument for a non-positive step.
[[nodiscard]] ResonanceDerivative conical_resonance_beta_derivative(const ConicalResonance& problem,
                                                                    const ConicalResonantMode& mode,
                                                                    const ModeAdjoint& adjoint,
                                                                    Real step);

/// Derivative with respect to a parameter of the Bloch constraints (the Bloch wavenumber
/// @f$ k_x @f$ of the periodic pairs): `minus` and `plus` are the problem at the parameter
/// ∓ `step` on the same maps (only the phases differ). With
/// @f$ x' = \partial P\,x_r @f$ and @f$ \tilde y' = \overline{\partial P}\,y @f$,
/// @f$ d\lambda = [\tilde y^\top(S - \lambda M)\,x' + \tilde y'^\top(S - \lambda M)\,x] /
/// (\tilde y^\top M x) @f$ — the complex dispersion @f$ d\omega/dk_x @f$ of a leaky mode.
/// @throws InvalidArgument without Bloch constraints, for a non-positive step or problems
///         with another constraint pattern.
[[nodiscard]] ResonanceDerivative conical_resonance_bloch_derivative(
    const ConicalResonance& problem, const ConicalResonantMode& mode, const ModeAdjoint& adjoint,
    const ConicalResonance& minus, const ConicalResonance& plus, Real step);

extern template ModeAdjoint resonance_adjoint<2>(const Resonance<2>&, const ResonantMode&);
extern template ModeAdjoint resonance_adjoint<3>(const Resonance<3>&, const ResonantMode&);
extern template ResonanceDerivative resonance_material_derivative<2>(const Resonance<2>&,
                                                                     const ResonantMode&,
                                                                     const ModeAdjoint&, mesh::Tag);
extern template ResonanceDerivative resonance_material_derivative<3>(const Resonance<3>&,
                                                                     const ResonantMode&,
                                                                     const ModeAdjoint&, mesh::Tag);
extern template ResonanceDerivative resonance_shape_derivative<2>(const Resonance<2>&,
                                                                  const ResonantMode&,
                                                                  const ModeAdjoint&,
                                                                  const NodeField&, Real);
extern template ResonanceDerivative resonance_shape_derivative<3>(const Resonance<3>&,
                                                                  const ResonantMode&,
                                                                  const ModeAdjoint&,
                                                                  const NodeField&, Real);

}  // namespace hpfem::physics

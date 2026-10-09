#pragma once
/// @file parameter_sensitivity.hpp
/// Derivatives of the conical solution with respect to parameters of the setup as a whole —
/// the frequency and the angles of incidence (M16 S1, ADR-0012 §4). Such a parameter
/// @f$ \theta @f$ changes the operator @f$ A(\theta) @f$ everywhere (@f$ k_0^2 @f$, dispersive
/// permittivities, @f$ \beta = k_z @f$), the load @f$ b(\theta) @f$ (the incident field of the
/// scattered-field formulation) and the Bloch phases @f$ e^{ik_x a} @f$ of the constraints
/// @f$ P(\theta) @f$. With the reduced coefficients @f$ u @f$ (the masters) the discrete
/// equations are @f$ G(u,\theta) = P^H(b - A P u) = 0 @f$, hence the tangent
/// @f[ P^H A P\,\frac{du}{d\theta} = P^H\rho' + (\partial_\theta P)^H\rho_0,\qquad
///     \frac{de}{d\theta} = P\,\frac{du}{d\theta} + (\partial_\theta P)\,u @f]
/// with the full residual of the solution @f$ \rho_0 = b - Ae @f$ (only @f$ P^H\rho_0 @f$
/// vanishes) and the residual derivative along the transported coefficients
/// @f$ \rho' = \frac{d}{d\theta}\big[b - A\,P u\big]_{u\ \text{fixed}} @f$. The caller builds
/// the problems at @f$ \theta \pm h @f$ on the same DoF maps; @f$ \rho' @f$,
/// @f$ \partial_\theta P @f$ and @f$ \partial_\theta P\,u @f$ are central differences between
/// them (exact up to @f$ O(h^2) @f$, no solve involved), the tangent is one solve on the
/// factorisation the solution kept. Time convention @f$ e^{-i\omega t} @f$. See
/// docs/theory/maxwell.md "Frequency and angle derivatives".

#include "hpfem/core/types.hpp"
#include "hpfem/physics/conical_scattering.hpp"

namespace hpfem::physics {

/// Full residual @f$ \rho = b - (K - k_0^2 M)\,e @f$ of the conical problem at the block
/// coefficients (in-plane | scaled longitudinal), stacked in that order, on all DoFs (the PEC
/// rows included). At the solution @f$ P^H\rho @f$ vanishes on `system_dofs()`. One assembly.
/// @throws InvalidArgument if the vectors do not match the maps.
[[nodiscard]] Vector conical_residual(const ConicalScattering& problem, const Vector& transverse,
                                      const Vector& longitudinal);

/// The solution with its reduced coefficients (the constraint masters on `system_dofs()`) kept
/// and the slaves recomputed from the constraints of `problem`: the coefficients
/// @f$ P(\theta')\,u @f$ of another parameter value on the same maps. Without constraints a
/// copy. `beta` is the one of `problem`; no factorisation.
/// @throws InvalidArgument if the vectors do not match the maps.
[[nodiscard]] ConicalSolution conical_transported_solution(const ConicalScattering& problem,
                                                           const ConicalSolution& solution);

/// Derivative of the solution coefficients with respect to a parameter of the setup.
struct ConicalTangent {
  Vector transverse;    ///< @f$ de/d\theta @f$, in-plane block (full size)
  Vector longitudinal;  ///< @f$ dv/d\theta @f$, scaled longitudinal block (full size)
};

/// @f$ de/d\theta @f$ of `solution` (solved on `problem` with
/// `ConicalScatteringSetup::keep_factorisation`) from the problems at @f$ \theta - h @f$
/// (`minus`) and @f$ \theta + h @f$ (`plus`, `step` = h): three assemblies and one tangent
/// solve on the kept factorisation (see the file comment). The three problems must share the
/// DoF maps, the scalar-path choice and the constraint pattern; anything in their setups may
/// differ smoothly in @f$ \theta @f$ (frequency, @f$ \beta @f$, materials, PML, incident field,
/// currents, Bloch phases). An observable @f$ Q(e,\theta) @f$ then has
/// @f$ dQ/d\theta = \partial_e Q\cdot de/d\theta + \partial_\theta Q|_e @f$.
/// @throws InvalidArgument without a kept factorisation, for h ≤ 0 or if the problems or the
/// solution do not fit together.
[[nodiscard]] ConicalTangent conical_parameter_tangent(const ConicalScattering& problem,
                                                       const ConicalSolution& solution,
                                                       const ConicalScattering& minus,
                                                       const ConicalScattering& plus, Real step);

}  // namespace hpfem::physics

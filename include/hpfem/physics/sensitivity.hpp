#pragma once
/// @file sensitivity.hpp
/// Sensitivities of linear observables by the adjoint solve (M12). For the discrete problem
/// @f$ A(\varepsilon)\,e = b(\varepsilon) @f$ and a linear goal @f$ Q(e) = q^\top e @f$ (no
/// conjugation, as the forms) the derivative with respect to a parameter θ is
/// @f$ \partial_\theta Q = z^\top(\partial_\theta b - \partial_\theta A\, e) @f$ with the
/// adjoint @f$ z @f$ of @f$ A^\top z = q @f$. With hanging-node and Bloch constraints
/// (prolongation P, reduced system @f$ P^HAP @f$) the adjoint lives in the test space,
/// @f$ (P^\top A\bar P)\,z_r = P^\top q,\ z = \bar P z_r @f$, exactly as the goal-oriented
/// estimator (`goal_oriented.hpp`), but on the primal space instead of the enriched one.
///
/// Material derivatives: θ is the relative permittivity @f$ \varepsilon_r @f$ of all cells of
/// a tag (holomorphic in @f$ \varepsilon_r @f$, so the real and imaginary parts follow from
/// one complex derivative). Then @f$ \partial_\varepsilon A = -k_0^2 M_{\text{tag}} @f$ and,
/// in the scattered-field formulation, @f$ \partial_\varepsilon b = k_0^2\,\ell_{\text{tag}}
/// (E^{inc}) @f$ with @f$ \ell_{\text{tag}}(v) = \int_{\text{tag}} E^{inc}\cdot v @f$, so that
/// @f[ \frac{dQ}{d\varepsilon_{\text{tag}}} = k_0^2\int_{\text{tag}} E_{\text{tot}}\cdot z\,dx
///   = k_0^2\, z^\top\big(M_{\text{tag}}\,e + \ell_{\text{tag}}(E^{inc})\big) , @f]
/// the derivative of the *discrete* goal (verified against finite differences to 1e-6 in
/// `test_sensitivity.cpp`). Cells of the tag inside a PML are refused (the stretch couples
/// the layer to ε). Quadratic observables (efficiencies, fluxes) follow by the chain rule
/// from the complex amplitude, e.g. @f$ \partial R_m = 2R_m\,\mathrm{Re}(\partial Q)/|A_m| @f$
/// with @f$ Q = A_m\cdot\bar A_m/|A_m| @f$ (`hpfem.grating.sensitivity`).
/// See docs/theory/maxwell.md#sensitivities-by-the-adjoint-solve.
#include <vector>

#include "hpfem/core/types.hpp"
#include "hpfem/mesh/mesh.hpp"
#include "hpfem/physics/conical_scattering.hpp"
#include "hpfem/physics/scattering.hpp"

namespace hpfem::physics {

/// Adjoint solution on the primal space of the problem for the functional vector q
/// (@f$ Q(e) = q^\top e @f$ on `problem.dofs()`): homogeneous Dirichlet data on the PEC and
/// incident facets, hanging-node and Bloch constraints in the test space. Full-size vector.
/// @throws InvalidArgument if q does not match the DoF map.
template <int Dim>
[[nodiscard]] Vector adjoint_solution(const Scattering<Dim>& problem, const Vector& q);

/// @f$ dQ/d\varepsilon_{\text{tag}} @f$ of the goal whose adjoint is `adjoint`, for the
/// solution of the problem (scattered or total field as the formulation requires).
/// @throws InvalidArgument if no cell carries the tag, a tagged cell lies in the PML, or the
///         vectors do not match the map.
template <int Dim>
[[nodiscard]] Complex material_sensitivity(const Scattering<Dim>& problem,
                                           const ScatteringSolution<Dim>& solution,
                                           const Vector& adjoint, mesh::Tag tag);

/// Adjoint of the conical solver as block coefficients (in-plane and scaled longitudinal).
struct ConicalAdjoint {
  Vector transverse;
  Vector longitudinal;
};

/// Adjoint solution of the conical problem for the functional pair @f$ (q_e, q_v) @f$ of a
/// `ConicalFunctional` evaluated on the problem's maps.
/// @throws InvalidArgument if the vectors do not match the maps.
[[nodiscard]] ConicalAdjoint conical_adjoint_solution(const ConicalScattering& problem,
                                                      const Vector& q_e, const Vector& q_v);

/// @f$ dQ/d\varepsilon_{\text{tag}} @f$ for the conical solver: @f$ k_0^2 \big[(z_e, z_v)^\top
/// M_{\text{tag}}(e, v) + (z_e, z_v)^\top\ell_{\text{tag}}\big] @f$ with the block mass of the
/// conical forms (the pairing @f$ \varepsilon_zvw @f$ of the scaled longitudinal part).
/// @throws InvalidArgument as `material_sensitivity`.
[[nodiscard]] Complex conical_material_sensitivity(const ConicalScattering& problem,
                                                   const ConicalSolution& solution,
                                                   const ConicalAdjoint& adjoint, mesh::Tag tag);

}  // namespace hpfem::physics

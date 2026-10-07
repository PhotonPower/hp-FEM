#pragma once
/// @file conical_goal.hpp
/// Goal-oriented (dual-weighted residual) error estimation for the conical solver
/// (`physics/conical_scattering.hpp`), the counterpart of `physics/goal_oriented.hpp` for the
/// coupled system @f$ (E_x, E_y, v = -iE_z) @f$: a linear functional @f$ Q(E) = q_e^\top e +
/// q_v^\top v @f$ on the two DoF maps, the adjoint @f$ a(V, z) = Q(V) @f$ solved on the maps with
/// every order raised by one (same constraints and PEC facets as the primal problem, Bloch phases
/// conjugated for the test space), the weight @f$ z - I_pz @f$ and the cell contributions of
/// `adaptivity::conical_weighted_residual`, whose sum estimates @f$ Q(E) - Q(E_h) @f$. The
/// functionals of this file return the physical field, @f$ E_z = iv @f$ (the @f$ q_v @f$ weights
/// carry the factor i). Convention exp(-iωt). See docs/theory/error-estimation.md#goal-oriented.

#include <functional>
#include <utility>

#include "hpfem/adaptivity/residual_estimator.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/physics/conical_scattering.hpp"
#include "hpfem/physics/goal_oriented.hpp"

namespace hpfem::physics {

/// A linear functional of the conical field given on any pair of maps of the mesh:
/// returns (q_e, q_v) with Q(E_h) = q_eᵀ e + q_vᵀ v.
using ConicalFunctional = std::function<std::pair<Vector, Vector>(const fespace::NedelecDofMap<2>&,
                                                                  const fespace::DofMap<2>&)>;

/// DWR estimate of the error of the goal for a solution of the conical problem.
/// @throws InvalidArgument if the solution or the functional does not match the maps.
[[nodiscard]] GoalEstimate conical_dwr_estimate(const ConicalScattering& problem,
                                                const ConicalSolution& solution,
                                                const ConicalFunctional& functional,
                                                const adaptivity::EstimatorOptions& options = {});

/// Point value of the physical field, @f$ Q(E) = E(x)\cdot w = E_xw_x + E_yw_y + E_zw_z @f$
/// (w not conjugated).
[[nodiscard]] ConicalFunctional conical_point_functional(const Point<2>& x,
                                                         const ConicalVector& weight);

/// Vector amplitude of a diffraction order along a polarisation vector e (not conjugated):
/// @f$ Q(E) = \frac1P\int_0^P E(x_0 + s\,t)\cdot e\; e^{-ik_{t,m}s}\,ds @f$ on the line from
/// `origin` along the unit `tangent` over one `period`, @f$ k_{t,m} = k_{t,0} + 2\pi m/P @f$
/// (the same Gauss–Legendre rule as `conical_fourier_coefficients`). For the
/// efficiency of the order the natural choice is @f$ e = \overline{A_m}/|A_m| @f$ of the current
/// amplitude: then @f$ \Delta R_m \approx 2 R_m\,\mathrm{Re}(\Delta Q)/|A_m| @f$.
[[nodiscard]] ConicalFunctional conical_order_functional(const Point<2>& origin,
                                                         const Point<2>& tangent, Real period,
                                                         Real kt0, int order, int num_points,
                                                         const ConicalVector& e);

}  // namespace hpfem::physics

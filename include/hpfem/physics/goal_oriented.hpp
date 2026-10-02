#pragma once
/// @file goal_oriented.hpp
/// Goal-oriented (dual-weighted residual, DWR) error estimation for a linear quantity of
/// interest @f$ Q(E) @f$ of a scattering solution (Becker & Rannacher 2001): with the adjoint
/// solution @f$ z @f$ of @f$ a(v, z) = Q(v) @f$ for all @f$ v @f$ and the Galerkin orthogonality,
/// @f$ Q(E) - Q(E_h) = \ell(z - z_h) - a(E_h, z - z_h) = \sum_K r_K(z - z_h) @f$, where
/// @f$ r_K @f$ is the cell-wise weighted residual (element residual and facet jumps of
/// `adaptivity::weighted_residual`). The adjoint is solved on the mesh with every order
/// raised by one, the weight is @f$ z_{p+1} - I_p z_{p+1} @f$ (hierarchical interpolation).
/// The sum of the contributions estimates the signed error of the goal, their moduli are
/// the indicators for marking. Constraints (hanging nodes, Bloch) and the Dirichlet facets
/// of the problem are respected: the adjoint has homogeneous Dirichlet data and lives in
/// the test space of the primal problem. See docs/theory/error-estimation.md#goal-oriented.

#include <functional>
#include <vector>

#include "hpfem/adaptivity/residual_estimator.hpp"
#include "hpfem/assembly/functionals.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/physics/scattering.hpp"

namespace hpfem::physics {

/// A linear functional given on any DoF map of the mesh: returns q with Q(E_h) = qᵀ e_h.
template <int Dim>
using Functional = std::function<Vector(const fespace::NedelecDofMap<Dim>&)>;

/// Result of the DWR estimate.
struct GoalEstimate {
  Complex error;                       ///< Σ_K r_K(w): estimated Q(E) − Q(E_h)
  std::vector<Complex> contributions;  ///< r_K(w) per cell
  std::vector<Real> indicators;        ///< |r_K(w)| per cell
  Complex value;                       ///< Q(E_h)
  /// Σ_K |r_K|, an upper bound of the estimated goal error.
  [[nodiscard]] Real total() const;
};

/// Estimates the error of the goal for the solution of the problem.
/// @throws InvalidArgument if the solution does not match the problem.
template <int Dim>
[[nodiscard]] GoalEstimate dwr_estimate(const Scattering<Dim>& problem,
                                        const ScatteringSolution<Dim>& solution,
                                        const Functional<Dim>& functional,
                                        const adaptivity::EstimatorOptions& options = {});

/// Point value @f$ Q(E) = E(x)\cdot w @f$ (w not conjugated).
template <int Dim>
[[nodiscard]] Functional<Dim> point_value_functional(const Point<Dim>& x,
                                                     const assembly::ComplexVector<Dim>& weight);

/// Fourier coefficient of a diffraction order (2D, `physics::fourier_coefficients`):
/// @f$ Q(E) = \frac1a\int_{y_0}^{y_0+a} E(x_0, y)\cdot e\; e^{-i k_{y,m} y}\,dy @f$ with
/// @f$ k_{y,m} = k_{y,0} + 2\pi m / a @f$, polarisation vector e, Gauss–Legendre with
/// `num_points` points.
[[nodiscard]] Functional<2> fourier_coefficient_functional(Real x0, Real y0, Real period, Real ky0,
                                                           int order, int num_points,
                                                           const assembly::ComplexVector<2>& e);

extern template GoalEstimate dwr_estimate<2>(const Scattering<2>&, const ScatteringSolution<2>&,
                                             const Functional<2>&,
                                             const adaptivity::EstimatorOptions&);
extern template GoalEstimate dwr_estimate<3>(const Scattering<3>&, const ScatteringSolution<3>&,
                                             const Functional<3>&,
                                             const adaptivity::EstimatorOptions&);
extern template Functional<2> point_value_functional<2>(const Point<2>&,
                                                        const assembly::ComplexVector<2>&);
extern template Functional<3> point_value_functional<3>(const Point<3>&,
                                                        const assembly::ComplexVector<3>&);

}  // namespace hpfem::physics

#pragma once
/// @file time_domain.hpp
/// Transient Maxwell solver: the second-order wave equation for the electric field
/// @f$ \varepsilon\,\partial_t^2 E + \sigma\,\partial_t E + \nabla\times(\mu^{-1}\nabla\times E)
///     = -\partial_t J @f$
/// (SI units, real materials) discretised with the Nédélec space in space and the implicit
/// Newmark-β scheme in time. With β = 1/4, γ = 1/2 (default) the scheme is the trapezoidal
/// rule: unconditionally stable, second order, and it conserves the discrete energy
/// @f$ \tfrac12(\dot u^T M \dot u + u^T S u) @f$ of the undamped problem exactly, so the
/// time step is chosen for accuracy, not stability. Boundary conditions: PEC (tangential
/// trace eliminated) and the first-order Silver–Müller absorbing condition
/// @f$ n\times\mu^{-1}\nabla\times E = -Z^{-1}\,\partial_t E_t @f$, a boundary damping term
/// with the wave impedance Z of the adjacent cell. Sources are a spatial current density
/// J(x) times a time signal g(t), entering as @f$ -J(x)\,g'(t) @f$. See
/// docs/theory/maxwell.md#time-domain.
#include <functional>
#include <map>
#include <memory>
#include <vector>

#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/core/types.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/materials/material.hpp"
#include "hpfem/mesh/mesh.hpp"
#include "hpfem/solvers/linear_solver.hpp"

namespace hpfem::physics {

/// Time signal g(t) with its derivative (sources enter through g').
struct TimeSignal {
  std::function<Real(Real)> value;
  std::function<Real(Real)> derivative;
};

/// Gaussian pulse @f$ g(t) = \exp(-(t - t_0)^2 / (2 w^2)) @f$.
[[nodiscard]] TimeSignal gaussian_pulse(Real t0, Real width);
/// Modulated Gaussian @f$ g(t) = \sin(\omega (t - t_0))\,\exp(-(t - t_0)^2 / (2 w^2)) @f$.
[[nodiscard]] TimeSignal modulated_gaussian(Real omega, Real t0, Real width);

/// Description of a transient problem.
template <int Dim>
struct TimeDomainSetup {
  materials::MaterialMap materials;        ///< real εr, μr by cell tag
  std::map<mesh::Tag, Real> conductivity;  ///< σ [S/m] by cell tag (unlisted: 0)
  std::vector<mesh::Tag> pec_tags;         ///< facets with n × E = 0
  std::vector<mesh::Tag> absorbing_tags;   ///< boundary facets with the Silver–Müller condition
  assembly::ComplexVectorField<Dim> current;  ///< J(x) [A/m²] (real part), optional
  TimeSignal signal;                          ///< g(t) of the current, optional
  Real dt = 0;                                ///< time step [s]
  Real beta = 0.25;                           ///< Newmark β
  Real gamma = 0.5;                           ///< Newmark γ (> 1/2: numerical damping)
  solvers::DirectSolverBackend solver = solvers::DirectSolverBackend::kAuto;
  int extra_quadrature_order = 2;
};

/// Field, velocity and acceleration coefficients (full DoF vectors, PEC entries zero).
template <int Dim>
struct TimeState {
  Real time = 0;
  Vector u;  ///< E
  Vector v;  ///< ∂t E
  Vector a;  ///< ∂t² E
  int step = 0;
};

/// Assembles the matrices once, factorises the Newmark operator and advances a state.
template <int Dim>
class TimeDomain {
 public:
  /// @throws InvalidArgument for dt ≤ 0, complex materials, an absorbing tag on an interior
  ///         facet, or a current without signal (and vice versa).
  TimeDomain(const fespace::NedelecDofMap<Dim>& dofs, TimeDomainSetup<Dim> setup);
  [[nodiscard]] const fespace::NedelecDofMap<Dim>& dofs() const noexcept { return *dofs_; }
  [[nodiscard]] const TimeDomainSetup<Dim>& setup() const noexcept { return setup_; }
  [[nodiscard]] Index num_free_dofs() const noexcept { return static_cast<Index>(free_.size()); }

  /// State at t0 from the initial field and velocity (PEC entries are zeroed); the
  /// acceleration is the consistent one, @f$ M a = f(t_0) - C v - S u @f$.
  /// @throws InvalidArgument if a vector does not match the DoF map.
  [[nodiscard]] TimeState<Dim> initialize(const Vector& u0, const Vector& v0, Real t0 = 0) const;
  /// Zero initial state.
  [[nodiscard]] TimeState<Dim> initialize(Real t0 = 0) const;
  /// One Newmark step of length dt.
  void step(TimeState<Dim>& state) const;
  /// `steps` steps; `observer(state)` is called after each one (if given).
  void run(TimeState<Dim>& state, int steps,
           const std::function<void(const TimeState<Dim>&)>& observer = {}) const;
  /// Discrete energy @f$ \tfrac12(v^T M v + u^T S u) @f$ [J].
  [[nodiscard]] Real energy(const TimeState<Dim>& state) const;
  /// Load @f$ -g'(t)\,(J, \phi_i) @f$ on the free DoFs at time t (zero without source).
  [[nodiscard]] Vector load(Real t) const;

  /// Matrices on the free DoFs: stiffness (μ⁻¹), mass (ε), damping (σ and absorbing).
  [[nodiscard]] const SparseMatrix& stiffness() const noexcept { return s_; }
  [[nodiscard]] const SparseMatrix& mass() const noexcept { return m_; }
  [[nodiscard]] const SparseMatrix& damping() const noexcept { return c_; }

 private:
  [[nodiscard]] Vector restrict(const Vector& full) const;
  [[nodiscard]] Vector expand(const Vector& reduced) const;

  const fespace::NedelecDofMap<Dim>* dofs_;
  TimeDomainSetup<Dim> setup_;
  std::vector<Index> free_;
  SparseMatrix s_, m_, c_;
  Vector current_load_;  ///< (J, φ_i) on the free DoFs
  std::unique_ptr<solvers::LinearSolver> newmark_;
  std::unique_ptr<solvers::LinearSolver> mass_solver_;
};

extern template struct TimeDomainSetup<2>;
extern template struct TimeDomainSetup<3>;
extern template struct TimeState<2>;
extern template struct TimeState<3>;
extern template class TimeDomain<2>;
extern template class TimeDomain<3>;

}  // namespace hpfem::physics

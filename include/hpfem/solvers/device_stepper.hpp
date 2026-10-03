#pragma once
/// @file device_stepper.hpp
/// Newmark time stepping on the GPU (`HPFEM_ENABLE_CUDA`, hpfem_gpu library of API
/// version 3): the state @f$ u, v, a @f$ stays on the device, every step is two sparse
/// products, the solve with the factorised Newmark operator and the vector updates of
/// @f$ u_{\text{pred}} = u + \Delta t\, v + \Delta t^2 (\tfrac12 - \beta) a @f$,
/// @f$ v_{\text{pred}} = v + \Delta t (1 - \gamma) a @f$,
/// @f$ a_{\text{new}} = K^{-1} (s\, b_J - C v_{\text{pred}} - S u_{\text{pred}}) @f$,
/// @f$ u = u_{\text{pred}} + \beta \Delta t^2 a_{\text{new}} @f$,
/// @f$ v = v_{\text{pred}} + \gamma \Delta t\, a_{\text{new}} @f$ — exactly the recursion of
/// `physics::TimeDomain::step`, so results agree with the host loop to round-off. The
/// state is downloaded only on request (observers, energy, the end of a run). See
/// docs/theory/solvers.md.

#include <memory>

#include "hpfem/core/types.hpp"
#include "hpfem/solvers/linear_solver.hpp"

namespace hpfem::solvers {

class DeviceStepper {
 public:
  /// True if `newmark` is (or, for `kAuto`, chose) the cuDSS backend of a library with API
  /// version 3 or later, so that the step can run on the device.
  [[nodiscard]] static bool available(const LinearSolver& newmark) noexcept;

  /// @param newmark   the factorised Newmark operator @f$ K @f$ (see above)
  /// @param damping   @f$ C @f$ or null (no conductivity, no absorbing walls)
  /// @param stiffness @f$ S @f$
  /// @param load      @f$ b_J @f$, the current load on the free DoFs, or null (no source);
  ///                  the time dependence enters as the scalar of `step`
  /// @throws Error if `available(newmark)` is false or a device allocation fails,
  ///         InvalidArgument for mismatched sizes.
  DeviceStepper(LinearSolver& newmark, const SparseMatrix* damping, const SparseMatrix& stiffness,
                const Vector* load, Real dt, Real beta, Real gamma);
  ~DeviceStepper();
  DeviceStepper(const DeviceStepper&) = delete;
  DeviceStepper& operator=(const DeviceStepper&) = delete;

  /// Uploads the state (reduced vectors of the free DoFs).
  void set_state(const Vector& u, const Vector& v, const Vector& a);
  /// One Newmark step with the load @f$ s\, b_J @f$ (`load_scale` = @f$ -g'(t_{\text{new}}) @f$).
  void step(Real load_scale);
  /// Downloads the state (synchronises the device).
  void get_state(Vector& u, Vector& v, Vector& a) const;
  [[nodiscard]] Index size() const noexcept { return size_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Index size_ = 0;
};

}  // namespace hpfem::solvers

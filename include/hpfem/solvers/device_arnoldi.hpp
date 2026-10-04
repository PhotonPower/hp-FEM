#pragma once
/// @file device_arnoldi.hpp
/// Shift-invert Arnoldi with the Krylov basis on the GPU (`HPFEM_ENABLE_CUDA`, hpfem_gpu
/// library of API version 4): the basis @f$ V = [v_0, \dots, v_k] @f$ stays on the device.
/// One iteration computes @f$ w = P (A - \sigma B)^{-1} B v_j @f$ there (cuDSS solve, the
/// library's sparse products, the gauge projection
/// @f$ P = I - G (G^H B G)^{-1} G^H B @f$ when a gradient is given), orthogonalises @f$ w @f$
/// against @f$ v_0, \dots, v_j @f$ by classical Gram–Schmidt applied twice (a reduction and
/// an update kernel per pass), normalises and stores @f$ v_{j+1} @f$; only the @f$ j + 1 @f$
/// Hessenberg entries and @f$ h_{j+1,j} @f$ return to the host. Restart vectors and Ritz
/// vectors are linear combinations @f$ V_m c @f$ formed on the device. The control flow
/// (Hessenberg eigenproblem, convergence test, restarts) stays in
/// `solvers::complex_eigenpairs_near`, which uses this class whenever the shifted matrix is
/// factorised by the cuDSS backend and the device memory suffices; `HPFEM_GPU_ARNOLDI=0`
/// keeps the basis on the host. See docs/theory/solvers.md.

#include <cstddef>
#include <memory>

#include "hpfem/core/types.hpp"
#include "hpfem/solvers/linear_solver.hpp"

namespace hpfem::solvers {

class DeviceArnoldi {
 public:
  /// True if `shifted` is (or, for `kAuto`, chose) the cuDSS backend of a library with API
  /// version 4 or later and `HPFEM_GPU_ARNOLDI` is not `0`.
  [[nodiscard]] static bool available(const LinearSolver& shifted) noexcept;

  /// Device memory for the basis and the work vectors of `n` unknowns and `ncv` Krylov
  /// vectors (without the matrices and the factors): @f$ 16\,(n (k + 4)) @f$ bytes.
  [[nodiscard]] static std::size_t basis_bytes(Index n, Index ncv) noexcept;

  /// @param shifted  the factorised @f$ A - \sigma B @f$ (cuDSS backend)
  /// @param b        @f$ B @f$ (uploaded to the device)
  /// @param gradient @f$ G @f$ (n × m) or null for no gauge projection
  /// @param gauge    the factorised @f$ G^H B G @f$ (cuDSS backend), required with a gradient
  /// @param ncv      number of Krylov vectors; the basis holds `ncv + 1` columns
  /// @throws Error if `available(shifted)` is false, the gauge is not on the device, the
  ///         estimated memory exceeds the free device memory (the estimate is logged) or an
  ///         allocation fails; InvalidArgument for mismatched sizes. Callers fall back to a
  ///         host basis on Error.
  DeviceArnoldi(LinearSolver& shifted, const SparseMatrix& b, const SparseMatrix* gradient,
                LinearSolver* gauge, Index ncv);
  ~DeviceArnoldi();
  DeviceArnoldi(const DeviceArnoldi&) = delete;
  DeviceArnoldi& operator=(const DeviceArnoldi&) = delete;

  /// @f$ v_0 = P s / \|P s\| @f$.
  void set_start(const Vector& start);
  /// One Arnoldi step for column `j` (0-based, `j < ncv`): returns @f$ h_{j+1,j} = \|w\| @f$
  /// and the `j + 1` entries @f$ h_{0..j,j} @f$ (both Gram–Schmidt passes summed) in
  /// `h_column`; stores @f$ v_{j+1} @f$ when the norm is positive.
  Real iterate(Index j, Vector& h_column);
  /// @f$ v_0 = V_m c / \|V_m c\| @f$ for the explicit restart (`c` has `m` entries).
  void restart(Index m, const Vector& coefficients);
  /// @f$ V_m C @f$ (`coefficients` is m × count): the Ritz vectors.
  [[nodiscard]] Matrix combine(Index m, const Matrix& coefficients) const;

  [[nodiscard]] Index size() const noexcept { return size_; }
  [[nodiscard]] Index krylov_dimension() const noexcept { return ncv_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Index size_ = 0;
  Index ncv_ = 0;
};

}  // namespace hpfem::solvers

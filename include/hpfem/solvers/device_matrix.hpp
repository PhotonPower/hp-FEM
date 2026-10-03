#pragma once
/// @file device_matrix.hpp
/// A sparse matrix kept on the GPU for repeated products (`HPFEM_ENABLE_CUDA`, hpfem_gpu
/// library of API version 3 or later): uploaded once, `apply` computes @f$ y = A x @f$ on the
/// device for one or several vectors, which the transient solver uses for the two products
/// of every Newmark step. Without the library or a device, `available()` is false and the
/// constructor throws. See docs/theory/solvers.md.

#include <memory>

#include "hpfem/core/types.hpp"

namespace hpfem::solvers {

class DeviceMatrix {
 public:
  /// Uploads the (compressed) CSR matrix; the host copy may be released afterwards.
  /// @throws Error if the GPU library is not usable or lacks API version 3, or on a device
  ///         error (out of memory).
  explicit DeviceMatrix(const SparseMatrix& matrix);
  ~DeviceMatrix();
  DeviceMatrix(const DeviceMatrix&) = delete;
  DeviceMatrix& operator=(const DeviceMatrix&) = delete;
  DeviceMatrix(DeviceMatrix&&) noexcept;
  DeviceMatrix& operator=(DeviceMatrix&&) noexcept;

  /// @f$ y = A x @f$ (one vector: uploaded, multiplied, downloaded).
  [[nodiscard]] Vector apply(const Vector& x) const;
  /// One product per column (one transfer for all).
  [[nodiscard]] Matrix apply_many(const Matrix& x) const;
  [[nodiscard]] Index rows() const noexcept { return rows_; }
  [[nodiscard]] Index cols() const noexcept { return cols_; }

  /// True if the GPU library is loaded with API version 3 or later and a device is present.
  [[nodiscard]] static bool available() noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Index rows_ = 0;
  Index cols_ = 0;
};

}  // namespace hpfem::solvers

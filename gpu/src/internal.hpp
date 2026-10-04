// Internal structures of the hpfem_gpu library shared by its translation units (solver,
// device matrices, time stepper). Nothing here crosses the C ABI.
#pragma once

#include <cstdint>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cudss.h>
#include <string>
#include <vector>

#include "hpfem_gpu.h"

struct DeviceBuffer {
  void* ptr = nullptr;
  size_t bytes = 0;

  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() { release(); }
  void release() {
    if (ptr != nullptr) cudaFree(ptr);
    ptr = nullptr;
    bytes = 0;
  }
  /// Allocates at least `needed` bytes (keeps a larger existing allocation).
  cudaError_t reserve(size_t needed) {
    if (needed <= bytes && ptr != nullptr) return cudaSuccess;
    release();
    const cudaError_t status = cudaMalloc(&ptr, needed > 0 ? needed : 1);
    if (status == cudaSuccess) bytes = needed;
    return status;
  }
  template <typename T>
  [[nodiscard]] T* as() const {
    return static_cast<T*>(ptr);
  }
};

const char* hpfem_gpu_cudss_status_name(cudssStatus_t status);

struct hpfem_gpu_solver {
  cudssHandle_t handle = nullptr;
  cudssConfig_t config = nullptr;
  cudssData_t data = nullptr;
  cudssMatrix_t matrix = nullptr;
  cudaStream_t stream = nullptr;

  int64_t n = 0;
  int64_t nnz = 0;
  bool factorized = false;
  double scale = 1.0;   // the factors are those of D (scale * A) D (max |a_ij| = 1)
  bool hybrid = false;  // factors (partly) in host memory
  size_t device_estimate = 0;
  size_t host_estimate = 0;
  size_t device_free = 0;
  bool equilibrated = false;   // D applied (diagonal equilibration)
  DeviceBuffer equilibration;  // D on the device (n doubles) when equilibrated

  DeviceBuffer row_ptr;
  DeviceBuffer col;
  DeviceBuffer values;
  DeviceBuffer input;     // host right-hand sides land here
  DeviceBuffer rhs;       // D * input, what cuDSS reads
  DeviceBuffer solution;  // what cuDSS writes
  DeviceBuffer output;    // scale * D * solution, what the host receives

  std::string last_error;

  hpfem_gpu_status fail(hpfem_gpu_status status, const std::string& message) {
    last_error = message;
    return status;
  }
  hpfem_gpu_status fail_cuda(const char* what, cudaError_t status) {
    cudaGetLastError();  // reset the sticky error
    return fail(
        status == cudaErrorMemoryAllocation ? HPFEM_GPU_ERR_OUT_OF_MEMORY : HPFEM_GPU_ERR_CUDA,
        std::string(what) + ": " + cudaGetErrorString(status));
  }
  hpfem_gpu_status fail_cudss(const char* what, cudssStatus_t status) {
    return fail(
        status == CUDSS_STATUS_ALLOC_FAILED ? HPFEM_GPU_ERR_OUT_OF_MEMORY : HPFEM_GPU_ERR_CUDSS,
        std::string(what) + ": cuDSS " + hpfem_gpu_cudss_status_name(status) + " (" +
            std::to_string(static_cast<int>(status)) + ")");
  }

  void destroy_matrix() {
    if (matrix != nullptr) cudssMatrixDestroy(matrix);
    matrix = nullptr;
  }
  void destroy_data() {
    if (data != nullptr && handle != nullptr) cudssDataDestroy(handle, data);
    data = nullptr;
  }
};

#define HPFEM_GPU_CUDA(solver, what, call)                                                     \
  do {                                                                                         \
    const cudaError_t hpfem_gpu_status_ = (call);                                              \
    if (hpfem_gpu_status_ != cudaSuccess) return (solver)->fail_cuda(what, hpfem_gpu_status_); \
  } while (0)
#define HPFEM_GPU_CUDSS(solver, what, call)                 \
  do {                                                      \
    const cudssStatus_t hpfem_gpu_status_ = (call);         \
    if (hpfem_gpu_status_ != CUDSS_STATUS_SUCCESS)          \
      return (solver)->fail_cudss(what, hpfem_gpu_status_); \
  } while (0)

/// Solves for `nrhs` device-resident right-hand sides (column major, n x nrhs) into a
/// device-resident solution, scaling and equilibration included; runs on the solver's
/// stream and synchronises it. `d_b` and `d_x` may alias.
hpfem_gpu_status hpfem_gpu_solve_device(hpfem_gpu_solver* solver, int64_t nrhs,
                                        const cuDoubleComplex* d_b, cuDoubleComplex* d_x);

struct hpfem_gpu_matrix {
  int64_t n = 0;     // rows (kept for the square case)
  int64_t rows = 0;  // rows and columns; square unless created with hpfem_gpu_matrix_create_rect
  int64_t cols = 0;
  int64_t nnz = 0;
  DeviceBuffer row_ptr;
  DeviceBuffer col;
  DeviceBuffer values;
  DeviceBuffer x;
  DeviceBuffer y;
  cudaStream_t stream = nullptr;
  std::string last_error;

  hpfem_gpu_status fail(hpfem_gpu_status status, const std::string& message) {
    last_error = message;
    return status;
  }
  hpfem_gpu_status fail_cuda(const char* what, cudaError_t status) {
    cudaGetLastError();
    return fail(
        status == cudaErrorMemoryAllocation ? HPFEM_GPU_ERR_OUT_OF_MEMORY : HPFEM_GPU_ERR_CUDA,
        std::string(what) + ": " + cudaGetErrorString(status));
  }
};

/// y = A x for one device-resident vector on the given stream (no synchronisation).
void hpfem_gpu_matrix_apply_device(const hpfem_gpu_matrix* matrix, const cuDoubleComplex* d_x,
                                   cuDoubleComplex* d_y, cudaStream_t stream);

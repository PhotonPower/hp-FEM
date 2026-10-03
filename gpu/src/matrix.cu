// Device-resident sparse matrices and their products (API version 3): a CSR matrix is
// uploaded once and multiplied many times, y = A x for complex double vectors on the host
// (uploaded and downloaded per call) — the two products of a Newmark time step. Own
// warp-per-row kernel instead of cuSPARSE so that the library keeps depending on the cuDSS
// DLL only; for rows of a few dozen entries it is bandwidth bound like cuSPARSE.
#include <cstdint>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <new>
#include <string>
#include <vector>

#include "hpfem_gpu.h"
#include "internal.hpp"

/// y = A x, one warp per row; the lanes stride over the row and reduce with shuffles.
__global__ void hpfem_gpu_spmv_warp_per_row(int64_t n, const int64_t* __restrict__ row_ptr,
                                            const int64_t* __restrict__ col,
                                            const cuDoubleComplex* __restrict__ values,
                                            const cuDoubleComplex* __restrict__ x,
                                            cuDoubleComplex* __restrict__ y) {
  const int64_t warp = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x % 32;
  if (warp >= n) return;
  const int64_t begin = row_ptr[warp];
  const int64_t end = row_ptr[warp + 1];
  double re = 0.0;
  double im = 0.0;
  for (int64_t k = begin + lane; k < end; k += 32) {
    const cuDoubleComplex a = values[k];
    const cuDoubleComplex v = x[col[k]];
    re += a.x * v.x - a.y * v.y;
    im += a.x * v.y + a.y * v.x;
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    re += __shfl_down_sync(0xffffffffu, re, offset);
    im += __shfl_down_sync(0xffffffffu, im, offset);
  }
  if (lane == 0) y[warp] = make_cuDoubleComplex(re, im);
}

#define HPFEM_GPU_MATRIX_CUDA(matrix, what, call)                                              \
  do {                                                                                         \
    const cudaError_t hpfem_gpu_status_ = (call);                                              \
    if (hpfem_gpu_status_ != cudaSuccess) return (matrix)->fail_cuda(what, hpfem_gpu_status_); \
  } while (0)

void hpfem_gpu_matrix_apply_device(const hpfem_gpu_matrix* matrix, const cuDoubleComplex* d_x,
                                   cuDoubleComplex* d_y, cudaStream_t stream) {
  const int threads = 256;
  const int64_t warps_per_block = threads / 32;
  const int64_t blocks = (matrix->n + warps_per_block - 1) / warps_per_block;
  hpfem_gpu_spmv_warp_per_row<<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
      matrix->n, matrix->row_ptr.as<const int64_t>(), matrix->col.as<const int64_t>(),
      matrix->values.as<const cuDoubleComplex>(), d_x, d_y);
}

extern "C" {

hpfem_gpu_status hpfem_gpu_matrix_create(hpfem_gpu_matrix** out, int64_t n, int64_t nnz,
                                         const int64_t* row_ptr, const int64_t* col,
                                         const double* values) {
  if (out == nullptr) return HPFEM_GPU_ERR_INVALID_ARG;
  *out = nullptr;
  if (n <= 0 || nnz < 0 || row_ptr == nullptr ||
      (nnz > 0 && (col == nullptr || values == nullptr)) || row_ptr[0] != 0 || row_ptr[n] != nnz) {
    return HPFEM_GPU_ERR_INVALID_ARG;
  }
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    cudaGetLastError();
    return HPFEM_GPU_ERR_NO_DEVICE;
  }
  hpfem_gpu_matrix* matrix = new (std::nothrow) hpfem_gpu_matrix();
  if (matrix == nullptr) return HPFEM_GPU_ERR_OUT_OF_MEMORY;
  matrix->n = n;
  matrix->nnz = nnz;
  const size_t n_size = static_cast<size_t>(n);
  const size_t nnz_size = static_cast<size_t>(nnz);
  const auto upload = [&]() -> hpfem_gpu_status {
    HPFEM_GPU_MATRIX_CUDA(matrix, "matrix: create stream", cudaStreamCreate(&matrix->stream));
    HPFEM_GPU_MATRIX_CUDA(matrix, "matrix: allocate row pointer",
                          matrix->row_ptr.reserve((n_size + 1) * sizeof(int64_t)));
    HPFEM_GPU_MATRIX_CUDA(matrix, "matrix: allocate column indices",
                          matrix->col.reserve(nnz_size * sizeof(int64_t)));
    HPFEM_GPU_MATRIX_CUDA(matrix, "matrix: allocate values",
                          matrix->values.reserve(nnz_size * 2 * sizeof(double)));
    HPFEM_GPU_MATRIX_CUDA(matrix, "matrix: allocate vectors",
                          matrix->x.reserve(n_size * 2 * sizeof(double)));
    HPFEM_GPU_MATRIX_CUDA(matrix, "matrix: allocate vectors",
                          matrix->y.reserve(n_size * 2 * sizeof(double)));
    HPFEM_GPU_MATRIX_CUDA(matrix, "matrix: upload row pointer",
                          cudaMemcpy(matrix->row_ptr.ptr, row_ptr, (n_size + 1) * sizeof(int64_t),
                                     cudaMemcpyHostToDevice));
    if (nnz > 0) {
      HPFEM_GPU_MATRIX_CUDA(
          matrix, "matrix: upload column indices",
          cudaMemcpy(matrix->col.ptr, col, nnz_size * sizeof(int64_t), cudaMemcpyHostToDevice));
      HPFEM_GPU_MATRIX_CUDA(matrix, "matrix: upload values",
                            cudaMemcpy(matrix->values.ptr, values, nnz_size * 2 * sizeof(double),
                                       cudaMemcpyHostToDevice));
    }
    return HPFEM_GPU_OK;
  };
  const hpfem_gpu_status status = upload();
  if (status != HPFEM_GPU_OK) {
    hpfem_gpu_matrix_destroy(matrix);
    return status;
  }
  *out = matrix;
  return HPFEM_GPU_OK;
}

void hpfem_gpu_matrix_destroy(hpfem_gpu_matrix* matrix) {
  if (matrix == nullptr) return;
  if (matrix->stream != nullptr) cudaStreamDestroy(matrix->stream);
  delete matrix;
}

hpfem_gpu_status hpfem_gpu_matrix_apply(hpfem_gpu_matrix* matrix, int64_t nrhs, const double* x,
                                        double* y) {
  if (matrix == nullptr) return HPFEM_GPU_ERR_INVALID_ARG;
  if (nrhs <= 0 || x == nullptr || y == nullptr) {
    return matrix->fail(HPFEM_GPU_ERR_INVALID_ARG, "apply: null array or nrhs <= 0");
  }
  const size_t n_size = static_cast<size_t>(matrix->n);
  const size_t count = n_size * static_cast<size_t>(nrhs);
  HPFEM_GPU_MATRIX_CUDA(matrix, "apply: allocate input",
                        matrix->x.reserve(count * 2 * sizeof(double)));
  HPFEM_GPU_MATRIX_CUDA(matrix, "apply: allocate output",
                        matrix->y.reserve(count * 2 * sizeof(double)));
  HPFEM_GPU_MATRIX_CUDA(matrix, "apply: upload input",
                        cudaMemcpyAsync(matrix->x.ptr, x, count * 2 * sizeof(double),
                                        cudaMemcpyHostToDevice, matrix->stream));
  for (int64_t j = 0; j < nrhs; ++j) {
    hpfem_gpu_matrix_apply_device(matrix, matrix->x.as<cuDoubleComplex>() + j * matrix->n,
                                  matrix->y.as<cuDoubleComplex>() + j * matrix->n, matrix->stream);
  }
  HPFEM_GPU_MATRIX_CUDA(matrix, "apply: launch", cudaGetLastError());
  HPFEM_GPU_MATRIX_CUDA(matrix, "apply: download output",
                        cudaMemcpyAsync(y, matrix->y.ptr, count * 2 * sizeof(double),
                                        cudaMemcpyDeviceToHost, matrix->stream));
  HPFEM_GPU_MATRIX_CUDA(matrix, "apply: synchronise", cudaStreamSynchronize(matrix->stream));
  return HPFEM_GPU_OK;
}

const char* hpfem_gpu_matrix_last_error(const hpfem_gpu_matrix* matrix) {
  return matrix == nullptr ? "" : matrix->last_error.c_str();
}

}  // extern "C"

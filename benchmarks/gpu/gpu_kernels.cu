// CUDA side of the GPU micro-benchmark (see bench.hpp): cuSPARSE generic SpMV and cuDSS LU.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cudss.h>
#include <cusparse.h>
#include <string>
#include <vector>

#include "bench.hpp"

namespace {

#define CUDA_CHECK(call)                                                                     \
  do {                                                                                       \
    const cudaError_t status = (call);                                                       \
    if (status != cudaSuccess) {                                                             \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(status), __FILE__, \
                   __LINE__);                                                                \
      std::exit(1);                                                                          \
    }                                                                                        \
  } while (0)
#define CUSPARSE_CHECK(call)                                                                 \
  do {                                                                                       \
    const cusparseStatus_t status = (call);                                                  \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                                 \
      std::fprintf(stderr, "cuSPARSE error %d at %s:%d\n", int(status), __FILE__, __LINE__); \
      std::exit(1);                                                                          \
    }                                                                                        \
  } while (0)
#define CUDSS_CHECK(call)                                                                 \
  do {                                                                                    \
    const cudssStatus_t status = (call);                                                  \
    if (status != CUDSS_STATUS_SUCCESS) {                                                 \
      std::fprintf(stderr, "cuDSS error %d at %s:%d\n", int(status), __FILE__, __LINE__); \
      std::exit(1);                                                                       \
    }                                                                                     \
  } while (0)

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

double now() {
  return static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
                                 std::chrono::steady_clock::now().time_since_epoch())
                                 .count()) *
         1e-6;
}

struct DeviceCsr {
  int* row = nullptr;
  int* col = nullptr;
  cuDoubleComplex* val = nullptr;
  explicit DeviceCsr(const Csr& a) {
    const int nnz = static_cast<int>(a.val.size());
    CUDA_CHECK(cudaMalloc(&row, sizeof(int) * (a.n + 1)));
    CUDA_CHECK(cudaMalloc(&col, sizeof(int) * nnz));
    CUDA_CHECK(cudaMalloc(&val, sizeof(cuDoubleComplex) * nnz));
    CUDA_CHECK(cudaMemcpy(row, a.row_ptr.data(), sizeof(int) * (a.n + 1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(col, a.col.data(), sizeof(int) * nnz, cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(val, a.val.data(), sizeof(cuDoubleComplex) * nnz, cudaMemcpyHostToDevice));
  }
  ~DeviceCsr() {
    cudaFree(row);
    cudaFree(col);
    cudaFree(val);
  }
};

cuDoubleComplex* device_vector(const std::vector<std::complex<double>>& x, int copies = 1) {
  cuDoubleComplex* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, sizeof(cuDoubleComplex) * x.size() * copies));
  for (int c = 0; c < copies; ++c) {
    CUDA_CHECK(cudaMemcpy(d + static_cast<std::ptrdiff_t>(c) * x.size(), x.data(),
                          sizeof(cuDoubleComplex) * x.size(), cudaMemcpyHostToDevice));
  }
  return d;
}

}  // namespace

std::string gpu_name() {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  return prop.name;
}

double gpu_spmv(const Csr& a, const std::vector<std::complex<double>>& x,
                std::vector<std::complex<double>>& y, int reps) {
  const int nnz = static_cast<int>(a.val.size());
  const DeviceCsr d(a);
  cuDoubleComplex* d_x = device_vector(x);
  cuDoubleComplex* d_y = nullptr;
  CUDA_CHECK(cudaMalloc(&d_y, sizeof(cuDoubleComplex) * a.n));
  cusparseHandle_t handle;
  CUSPARSE_CHECK(cusparseCreate(&handle));
  cusparseSpMatDescr_t mat;
  cusparseDnVecDescr_t vx, vy;
  CUSPARSE_CHECK(cusparseCreateCsr(&mat, a.n, a.n, nnz, d.row, d.col, d.val, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F));
  CUSPARSE_CHECK(cusparseCreateDnVec(&vx, a.n, d_x, CUDA_C_64F));
  CUSPARSE_CHECK(cusparseCreateDnVec(&vy, a.n, d_y, CUDA_C_64F));
  const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
  const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
  std::size_t buffer_size = 0;
  CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mat, vx,
                                         &zero, vy, CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                         &buffer_size));
  void* buffer = nullptr;
  CUDA_CHECK(cudaMalloc(&buffer, buffer_size));
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<double> times;
  for (int r = 0; r < reps; ++r) {
    CUDA_CHECK(cudaEventRecord(start));
    CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mat, vx, &zero, vy,
                                CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    times.push_back(ms * 1e-3);
  }
  CUDA_CHECK(cudaMemcpy(y.data(), d_y, sizeof(cuDoubleComplex) * a.n, cudaMemcpyDeviceToHost));
  cusparseDestroySpMat(mat);
  cusparseDestroyDnVec(vx);
  cusparseDestroyDnVec(vy);
  cusparseDestroy(handle);
  cudaFree(buffer);
  cudaFree(d_x);
  cudaFree(d_y);
  return median(times);
}

SolveTimes gpu_solve(const Csr& a, const std::vector<std::complex<double>>& b,
                     std::vector<std::complex<double>>& x, int reps) {
  const int nnz = static_cast<int>(a.val.size());
  const DeviceCsr d(a);
  cuDoubleComplex* d_b = device_vector(b);
  cuDoubleComplex* d_b8 = device_vector(b, 8);
  cuDoubleComplex *d_x = nullptr, *d_x8 = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, sizeof(cuDoubleComplex) * a.n));
  CUDA_CHECK(cudaMalloc(&d_x8, sizeof(cuDoubleComplex) * a.n * 8));
  cudssHandle_t handle;
  cudssConfig_t config;
  cudssData_t data;
  CUDSS_CHECK(cudssCreate(&handle));
  CUDSS_CHECK(cudssConfigCreate(&config));
  CUDSS_CHECK(cudssDataCreate(handle, &data));
  cudssMatrix_t mat, x1, b1, x8, b8;
  CUDSS_CHECK(cudssMatrixCreateCsr(&mat, a.n, a.n, nnz, d.row, nullptr, d.col, d.val, CUDSS_R_32I,
                                   CUDSS_R_32I, CUDSS_C_64F, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL,
                                   CUDSS_BASE_ZERO));
  CUDSS_CHECK(cudssMatrixCreateDn(&x1, a.n, 1, a.n, d_x, CUDSS_C_64F, CUDSS_LAYOUT_COL_MAJOR));
  CUDSS_CHECK(cudssMatrixCreateDn(&b1, a.n, 1, a.n, d_b, CUDSS_C_64F, CUDSS_LAYOUT_COL_MAJOR));
  CUDSS_CHECK(cudssMatrixCreateDn(&x8, a.n, 8, a.n, d_x8, CUDSS_C_64F, CUDSS_LAYOUT_COL_MAJOR));
  CUDSS_CHECK(cudssMatrixCreateDn(&b8, a.n, 8, a.n, d_b8, CUDSS_C_64F, CUDSS_LAYOUT_COL_MAJOR));
  SolveTimes out;
  double t0 = now();
  CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, data, mat, x1, b1));
  CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config, data, mat, x1, b1));
  CUDA_CHECK(cudaDeviceSynchronize());
  out.factorize = now() - t0;
  std::vector<double> times;
  for (int r = 0; r < reps; ++r) {
    t0 = now();
    CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data, mat, x1, b1));
    CUDA_CHECK(cudaDeviceSynchronize());
    times.push_back(now() - t0);
  }
  out.solve = median(times);
  times.clear();
  for (int r = 0; r < std::max(1, reps / 4); ++r) {
    t0 = now();
    CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data, mat, x8, b8));
    CUDA_CHECK(cudaDeviceSynchronize());
    times.push_back(now() - t0);
  }
  out.solve_8 = median(times);
  x.resize(static_cast<std::size_t>(a.n));
  CUDA_CHECK(cudaMemcpy(x.data(), d_x, sizeof(cuDoubleComplex) * a.n, cudaMemcpyDeviceToHost));
  cudssMatrixDestroy(mat);
  cudssMatrixDestroy(x1);
  cudssMatrixDestroy(b1);
  cudssMatrixDestroy(x8);
  cudssMatrixDestroy(b8);
  cudssDataDestroy(handle, data);
  cudssConfigDestroy(config);
  cudssDestroy(handle);
  cudaFree(d_b);
  cudaFree(d_b8);
  cudaFree(d_x);
  cudaFree(d_x8);
  return out;
}

// Implementation of hpfem_gpu.h on cuDSS: one cuDSS handle / config / data object per
// solver, the CSR matrix and the dense right-hand sides live on the device, indices are
// passed as 64-bit (CUDSS_R_64I) exactly as the library stores them.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <cudss.h>
#include <new>
#include <string>
#include <vector>

#include "hpfem_gpu.h"

namespace {

struct DeviceBuffer {
  void* ptr = nullptr;
  size_t bytes = 0;

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
    const cudaError_t status = cudaMalloc(&ptr, needed);
    if (status == cudaSuccess) bytes = needed;
    return status;
  }
};

const char* cudss_status_name(cudssStatus_t status) {
  switch (status) {
    case CUDSS_STATUS_SUCCESS:
      return "success";
    case CUDSS_STATUS_NOT_INITIALIZED:
      return "not initialised";
    case CUDSS_STATUS_ALLOC_FAILED:
      return "allocation failed";
    case CUDSS_STATUS_INVALID_VALUE:
      return "invalid value";
    case CUDSS_STATUS_NOT_SUPPORTED:
      return "not supported";
    case CUDSS_STATUS_EXECUTION_FAILED:
      return "execution failed";
    case CUDSS_STATUS_INTERNAL_ERROR:
      return "internal error";
    default:
      return "unknown status";
  }
}

}  // namespace

struct hpfem_gpu_solver {
  cudssHandle_t handle = nullptr;
  cudssConfig_t config = nullptr;
  cudssData_t data = nullptr;
  cudssMatrix_t matrix = nullptr;
  cudaStream_t stream = nullptr;

  int64_t n = 0;
  int64_t nnz = 0;
  bool factorized = false;
  double scale = 1.0;  // the factors are those of scale * A (max |a_ij| = 1)

  DeviceBuffer row_ptr;
  DeviceBuffer col;
  DeviceBuffer values;
  DeviceBuffer rhs;
  DeviceBuffer solution;

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
        std::string(what) + ": cuDSS " + cudss_status_name(status) + " (" +
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

extern "C" {

int hpfem_gpu_api_version(void) {
  return HPFEM_GPU_API_VERSION;
}

const char* hpfem_gpu_version(void) {
  static char buffer[128] = "";
  if (buffer[0] == '\0') {
    int cudss_version = 0;
    cudssGetProperty(MAJOR_VERSION, &cudss_version);
    int major = cudss_version, minor = 0, patch = 0;
    cudssGetProperty(MINOR_VERSION, &minor);
    cudssGetProperty(PATCH_LEVEL, &patch);
    int runtime = 0;
    cudaRuntimeGetVersion(&runtime);
    std::snprintf(buffer, sizeof(buffer), "cuDSS %d.%d.%d, CUDA runtime %d.%d", major, minor, patch,
                  runtime / 1000, (runtime % 1000) / 10);
  }
  return buffer;
}

hpfem_gpu_status hpfem_gpu_device_info(char* name, size_t name_len, size_t* free_bytes,
                                       size_t* total_bytes) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    cudaGetLastError();
    return HPFEM_GPU_ERR_NO_DEVICE;
  }
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) return HPFEM_GPU_ERR_NO_DEVICE;
  if (name != nullptr && name_len > 0) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return HPFEM_GPU_ERR_CUDA;
    std::snprintf(name, name_len, "%s", prop.name);
  }
  if (free_bytes != nullptr || total_bytes != nullptr) {
    size_t free_mem = 0, total_mem = 0;
    if (cudaMemGetInfo(&free_mem, &total_mem) != cudaSuccess) return HPFEM_GPU_ERR_CUDA;
    if (free_bytes != nullptr) *free_bytes = free_mem;
    if (total_bytes != nullptr) *total_bytes = total_mem;
  }
  return HPFEM_GPU_OK;
}

hpfem_gpu_status hpfem_gpu_create(hpfem_gpu_solver** out) {
  if (out == nullptr) return HPFEM_GPU_ERR_INVALID_ARG;
  *out = nullptr;
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    cudaGetLastError();
    return HPFEM_GPU_ERR_NO_DEVICE;
  }
  hpfem_gpu_solver* solver = new (std::nothrow) hpfem_gpu_solver();
  if (solver == nullptr) return HPFEM_GPU_ERR_OUT_OF_MEMORY;
  const cudaError_t stream_status = cudaStreamCreate(&solver->stream);
  if (stream_status != cudaSuccess) {
    delete solver;
    cudaGetLastError();
    return HPFEM_GPU_ERR_CUDA;
  }
  cudssStatus_t status = cudssCreate(&solver->handle);
  if (status == CUDSS_STATUS_SUCCESS) status = cudssSetStream(solver->handle, solver->stream);
  if (status == CUDSS_STATUS_SUCCESS) status = cudssConfigCreate(&solver->config);
  if (status != CUDSS_STATUS_SUCCESS) {
    hpfem_gpu_destroy(solver);
    return HPFEM_GPU_ERR_CUDSS;
  }
  *out = solver;
  return HPFEM_GPU_OK;
}

void hpfem_gpu_destroy(hpfem_gpu_solver* solver) {
  if (solver == nullptr) return;
  solver->destroy_matrix();
  solver->destroy_data();
  if (solver->config != nullptr) cudssConfigDestroy(solver->config);
  if (solver->handle != nullptr) cudssDestroy(solver->handle);
  if (solver->stream != nullptr) cudaStreamDestroy(solver->stream);
  delete solver;
}

hpfem_gpu_status hpfem_gpu_factorize(hpfem_gpu_solver* solver, int64_t n, int64_t nnz,
                                     const int64_t* row_ptr, const int64_t* col,
                                     const double* values, hpfem_gpu_matrix_type type) {
  if (solver == nullptr) return HPFEM_GPU_ERR_INVALID_ARG;
  solver->factorized = false;
  solver->destroy_matrix();
  solver->destroy_data();
  if (n <= 0 || nnz < 0 || row_ptr == nullptr ||
      (nnz > 0 && (col == nullptr || values == nullptr))) {
    return solver->fail(HPFEM_GPU_ERR_INVALID_ARG, "factorize: null array or non-positive size");
  }
  if (row_ptr[0] != 0 || row_ptr[n] != nnz) {
    return solver->fail(
        HPFEM_GPU_ERR_INVALID_ARG,
        "factorize: row pointer does not span [0, nnz) (nnz = " + std::to_string(nnz) +
            ", row_ptr[n] = " + std::to_string(row_ptr[n]) + ")");
  }
  if (type != HPFEM_GPU_MATRIX_GENERAL && type != HPFEM_GPU_MATRIX_SYMMETRIC) {
    return solver->fail(HPFEM_GPU_ERR_INVALID_ARG, "factorize: unknown matrix type");
  }
  solver->n = n;
  solver->nnz = nnz;

  const size_t n_size = static_cast<size_t>(n);
  const size_t nnz_size = static_cast<size_t>(nnz);
  // cuDSS judges "tiny" pivots by an absolute threshold, so SI-scaled systems (entries
  // around 1e-15) would be perturbed wholesale; factorise scale * A with max |a_ij| = 1
  // instead and undo the scale on every solution (x = scale * (scale A)^{-1} b)
  double max_abs = 0.0;
  for (size_t k = 0; k < nnz_size; ++k) {
    max_abs = std::max(max_abs, std::hypot(values[2 * k], values[2 * k + 1]));
  }
  if (!(max_abs > 0.0) || !std::isfinite(max_abs)) {
    return solver->fail(HPFEM_GPU_ERR_SINGULAR,
                        "factorize: the matrix is zero or contains non-finite entries");
  }
  solver->scale = 1.0 / max_abs;
  std::vector<double> scaled(2 * nnz_size);
  for (size_t k = 0; k < 2 * nnz_size; ++k) scaled[k] = values[k] * solver->scale;
  HPFEM_GPU_CUDA(solver, "factorize: allocate row pointer",
                 solver->row_ptr.reserve((n_size + 1) * sizeof(int64_t)));
  HPFEM_GPU_CUDA(solver, "factorize: allocate column indices",
                 solver->col.reserve((nnz_size > 0 ? nnz_size : 1) * sizeof(int64_t)));
  HPFEM_GPU_CUDA(solver, "factorize: allocate values",
                 solver->values.reserve((nnz_size > 0 ? nnz_size : 1) * 2 * sizeof(double)));
  HPFEM_GPU_CUDA(solver, "factorize: upload row pointer",
                 cudaMemcpyAsync(solver->row_ptr.ptr, row_ptr, (n_size + 1) * sizeof(int64_t),
                                 cudaMemcpyHostToDevice, solver->stream));
  if (nnz > 0) {
    HPFEM_GPU_CUDA(solver, "factorize: upload column indices",
                   cudaMemcpyAsync(solver->col.ptr, col, nnz_size * sizeof(int64_t),
                                   cudaMemcpyHostToDevice, solver->stream));
    HPFEM_GPU_CUDA(solver, "factorize: upload values",
                   cudaMemcpyAsync(solver->values.ptr, scaled.data(), nnz_size * 2 * sizeof(double),
                                   cudaMemcpyHostToDevice, solver->stream));
  }
  // the dense operands of analysis / factorisation are not read; cuDSS only wants objects
  HPFEM_GPU_CUDA(solver, "factorize: allocate work vector",
                 solver->rhs.reserve(n_size * 2 * sizeof(double)));
  HPFEM_GPU_CUDA(solver, "factorize: allocate work vector",
                 solver->solution.reserve(n_size * 2 * sizeof(double)));

  const cudssMatrixType_t mtype =
      type == HPFEM_GPU_MATRIX_SYMMETRIC ? CUDSS_MTYPE_SYMMETRIC : CUDSS_MTYPE_GENERAL;
  const cudssMatrixViewType_t mview =
      type == HPFEM_GPU_MATRIX_SYMMETRIC ? CUDSS_MVIEW_UPPER : CUDSS_MVIEW_FULL;
  HPFEM_GPU_CUDSS(solver, "factorize: create data", cudssDataCreate(solver->handle, &solver->data));
  HPFEM_GPU_CUDSS(solver, "factorize: create CSR matrix",
                  cudssMatrixCreateCsr(&solver->matrix, n, n, nnz, solver->row_ptr.ptr, nullptr,
                                       solver->col.ptr, solver->values.ptr, CUDSS_R_64I,
                                       CUDSS_R_64I, CUDSS_C_64F, mtype, mview, CUDSS_BASE_ZERO));
  cudssMatrix_t x = nullptr;
  cudssMatrix_t b = nullptr;
  HPFEM_GPU_CUDSS(
      solver, "factorize: create dense operands",
      cudssMatrixCreateDn(&x, n, 1, n, solver->solution.ptr, CUDSS_C_64F, CUDSS_LAYOUT_COL_MAJOR));
  cudssStatus_t status =
      cudssMatrixCreateDn(&b, n, 1, n, solver->rhs.ptr, CUDSS_C_64F, CUDSS_LAYOUT_COL_MAJOR);
  if (status == CUDSS_STATUS_SUCCESS) {
    status = cudssExecute(solver->handle, CUDSS_PHASE_ANALYSIS, solver->config, solver->data,
                          solver->matrix, x, b);
  }
  if (status == CUDSS_STATUS_SUCCESS) {
    status = cudssExecute(solver->handle, CUDSS_PHASE_FACTORIZATION, solver->config, solver->data,
                          solver->matrix, x, b);
  }
  cudaError_t sync = cudaStreamSynchronize(solver->stream);
  if (x != nullptr) cudssMatrixDestroy(x);
  if (b != nullptr) cudssMatrixDestroy(b);
  if (status != CUDSS_STATUS_SUCCESS) return solver->fail_cudss("factorize", status);
  if (sync != cudaSuccess) return solver->fail_cuda("factorize: synchronise", sync);

  int info = 0;
  size_t written = 0;
  status =
      cudssDataGet(solver->handle, solver->data, CUDSS_DATA_INFO, &info, sizeof(info), &written);
  if (status != CUDSS_STATUS_SUCCESS) return solver->fail_cudss("factorize: query info", status);
  if (info != 0) {
    return solver->fail(info > 0 ? HPFEM_GPU_ERR_SINGULAR : HPFEM_GPU_ERR_CUDSS,
                        info > 0 ? "factorize: zero pivot at row " + std::to_string(info - 1)
                                 : "factorize: cuDSS info = " + std::to_string(info));
  }
  // cuDSS replaces zero / tiny pivots by a perturbation instead of failing; such a
  // factorisation does not solve the given system, so it is reported like a singular matrix
  // the count comes in the index type of the matrix (int64 here; int for 32-bit indices)
  int64_t perturbed = 0;
  written = 0;
  status = cudssDataGet(solver->handle, solver->data, CUDSS_DATA_NPIVOTS, &perturbed,
                        sizeof(perturbed), &written);
  if (status != CUDSS_STATUS_SUCCESS) {
    int narrow = 0;
    status = cudssDataGet(solver->handle, solver->data, CUDSS_DATA_NPIVOTS, &narrow, sizeof(narrow),
                          &written);
    perturbed = narrow;
  }
  if (status != CUDSS_STATUS_SUCCESS) {
    return solver->fail_cudss("factorize: query perturbed pivots", status);
  }
  if (perturbed > 0) {
    return solver->fail(HPFEM_GPU_ERR_SINGULAR,
                        "factorize: " + std::to_string(perturbed) +
                            " zero or tiny pivot(s) were perturbed; the matrix is singular or "
                            "badly scaled");
  }
  solver->factorized = true;
  solver->last_error.clear();
  return HPFEM_GPU_OK;
}

hpfem_gpu_status hpfem_gpu_solve(hpfem_gpu_solver* solver, int64_t nrhs, const double* b,
                                 double* x) {
  if (solver == nullptr) return HPFEM_GPU_ERR_INVALID_ARG;
  if (!solver->factorized) {
    return solver->fail(HPFEM_GPU_ERR_NOT_FACTORIZED,
                        "solve: no factorisation (call factorize first)");
  }
  if (nrhs <= 0 || b == nullptr || x == nullptr) {
    return solver->fail(HPFEM_GPU_ERR_INVALID_ARG, "solve: null array or nrhs <= 0");
  }
  const size_t count = static_cast<size_t>(solver->n) * static_cast<size_t>(nrhs);
  HPFEM_GPU_CUDA(solver, "solve: allocate right-hand side",
                 solver->rhs.reserve(count * 2 * sizeof(double)));
  HPFEM_GPU_CUDA(solver, "solve: allocate solution",
                 solver->solution.reserve(count * 2 * sizeof(double)));
  HPFEM_GPU_CUDA(solver, "solve: upload right-hand side",
                 cudaMemcpyAsync(solver->rhs.ptr, b, count * 2 * sizeof(double),
                                 cudaMemcpyHostToDevice, solver->stream));
  cudssMatrix_t dx = nullptr;
  cudssMatrix_t db = nullptr;
  cudssStatus_t status = cudssMatrixCreateDn(&dx, solver->n, nrhs, solver->n, solver->solution.ptr,
                                             CUDSS_C_64F, CUDSS_LAYOUT_COL_MAJOR);
  if (status == CUDSS_STATUS_SUCCESS) {
    status = cudssMatrixCreateDn(&db, solver->n, nrhs, solver->n, solver->rhs.ptr, CUDSS_C_64F,
                                 CUDSS_LAYOUT_COL_MAJOR);
  }
  if (status == CUDSS_STATUS_SUCCESS) {
    status = cudssExecute(solver->handle, CUDSS_PHASE_SOLVE, solver->config, solver->data,
                          solver->matrix, dx, db);
  }
  cudaError_t copy = cudaSuccess;
  if (status == CUDSS_STATUS_SUCCESS) {
    copy = cudaMemcpyAsync(x, solver->solution.ptr, count * 2 * sizeof(double),
                           cudaMemcpyDeviceToHost, solver->stream);
  }
  const cudaError_t sync = cudaStreamSynchronize(solver->stream);
  if (dx != nullptr) cudssMatrixDestroy(dx);
  if (db != nullptr) cudssMatrixDestroy(db);
  if (status != CUDSS_STATUS_SUCCESS) return solver->fail_cudss("solve", status);
  if (copy != cudaSuccess) return solver->fail_cuda("solve: download solution", copy);
  if (sync != cudaSuccess) return solver->fail_cuda("solve: synchronise", sync);
  if (solver->scale != 1.0) {
    for (size_t k = 0; k < 2 * count; ++k) x[k] *= solver->scale;
  }
  return HPFEM_GPU_OK;
}

hpfem_gpu_status hpfem_gpu_factor_info(const hpfem_gpu_solver* solver, int64_t* nnz_factors,
                                       size_t* device_bytes) {
  if (solver == nullptr) return HPFEM_GPU_ERR_INVALID_ARG;
  if (nnz_factors != nullptr) {
    *nnz_factors = 0;
    if (solver->factorized) {
      int64_t lu_nnz = 0;
      size_t written = 0;
      if (cudssDataGet(solver->handle, solver->data, CUDSS_DATA_LU_NNZ, &lu_nnz, sizeof(lu_nnz),
                       &written) == CUDSS_STATUS_SUCCESS) {
        *nnz_factors = lu_nnz;
      }
    }
  }
  if (device_bytes != nullptr) {
    *device_bytes = solver->row_ptr.bytes + solver->col.bytes + solver->values.bytes +
                    solver->rhs.bytes + solver->solution.bytes;
    if (solver->factorized) {
      // the factors are owned by cuDSS; its memory estimate (if available) is added
      int64_t estimates[16] = {};
      size_t written = 0;
      if (cudssDataGet(solver->handle, solver->data, CUDSS_DATA_MEMORY_ESTIMATES, estimates,
                       sizeof(estimates), &written) == CUDSS_STATUS_SUCCESS) {
        *device_bytes += static_cast<size_t>(estimates[0] > 0 ? estimates[0] : 0);
      }
    }
  }
  return HPFEM_GPU_OK;
}

const char* hpfem_gpu_last_error(const hpfem_gpu_solver* solver) {
  return solver == nullptr ? "" : solver->last_error.c_str();
}

}  // extern "C"

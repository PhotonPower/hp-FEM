// Implementation of hpfem_gpu.h on cuDSS: one cuDSS handle / config / data object per
// solver, the CSR matrix and the dense right-hand sides live on the device, indices are
// passed as 64-bit (CUDSS_R_64I) exactly as the library stores them.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cudss.h>
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/sysinfo.h>
#endif
#include <new>
#include <string>
#include <vector>

#include "hpfem_gpu.h"
#include "internal.hpp"

namespace {

/// Free physical host memory in bytes (0 if unknown).
size_t free_host_memory() {
#if defined(_WIN32)
  MEMORYSTATUSEX status{};
  status.dwLength = sizeof(status);
  if (GlobalMemoryStatusEx(&status) == 0) return 0;
  return static_cast<size_t>(status.ullAvailPhys);
#else
  struct sysinfo info {};
  if (sysinfo(&info) != 0) return 0;
  return static_cast<size_t>(info.freeram) * info.mem_unit;
#endif
}

}  // namespace

const char* hpfem_gpu_cudss_status_name(cudssStatus_t status) {
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

/// out = factor * d[i] * in (d may be null: out = factor * in), column major n x nrhs.
__global__ void hpfem_gpu_scale_rows(int64_t n, int64_t nrhs, const double* __restrict__ d,
                                     double factor, const cuDoubleComplex* __restrict__ in,
                                     cuDoubleComplex* __restrict__ out) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n * nrhs) return;
  const int64_t i = idx % n;
  const double f = d != nullptr ? d[i] * factor : factor;
  out[idx] = make_cuDoubleComplex(in[idx].x * f, in[idx].y * f);
}

namespace {}  // namespace

hpfem_gpu_status hpfem_gpu_solve_device(hpfem_gpu_solver* solver, int64_t nrhs,
                                        const cuDoubleComplex* d_b, cuDoubleComplex* d_x) {
  if (!solver->factorized) {
    return solver->fail(HPFEM_GPU_ERR_NOT_FACTORIZED,
                        "solve: no factorisation (call factorize first)");
  }
  const size_t count = static_cast<size_t>(solver->n) * static_cast<size_t>(nrhs);
  HPFEM_GPU_CUDA(solver, "solve: allocate right-hand side",
                 solver->rhs.reserve(count * sizeof(cuDoubleComplex)));
  HPFEM_GPU_CUDA(solver, "solve: allocate solution",
                 solver->solution.reserve(count * sizeof(cuDoubleComplex)));
  const unsigned blocks = static_cast<unsigned>((count + 255) / 256);
  const double* d = solver->equilibrated ? solver->equilibration.as<double>() : nullptr;
  // (D A D) y = D b: the scaled right-hand side cuDSS reads
  hpfem_gpu_scale_rows<<<blocks, 256, 0, solver->stream>>>(solver->n, nrhs, d, 1.0, d_b,
                                                           solver->rhs.as<cuDoubleComplex>());
  HPFEM_GPU_CUDA(solver, "solve: scale right-hand side", cudaGetLastError());
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
  if (dx != nullptr) cudssMatrixDestroy(dx);
  if (db != nullptr) cudssMatrixDestroy(db);
  if (status != CUDSS_STATUS_SUCCESS) return solver->fail_cudss("solve", status);
  // x = scale * D * y, the solution of the original system
  hpfem_gpu_scale_rows<<<blocks, 256, 0, solver->stream>>>(
      solver->n, nrhs, d, solver->scale, solver->solution.as<cuDoubleComplex>(), d_x);
  HPFEM_GPU_CUDA(solver, "solve: rescale solution", cudaGetLastError());
  HPFEM_GPU_CUDA(solver, "solve: synchronise", cudaStreamSynchronize(solver->stream));
  return HPFEM_GPU_OK;
}

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

namespace {

/* cuDSS judges "tiny" pivots by an absolute threshold, so SI-scaled systems (entries around
 * 1e-15) would be perturbed wholesale; factorise scale * A with max |a_ij| = 1 instead and
 * undo the scale on every solution (x = scale * (scale A)^{-1} b). Then the diagonal
 * equilibration D A D with d_i = 1 / sqrt(|a_ii|) (row 2-norm where the diagonal is tiny):
 * keeps the complex symmetry and evens out the scales of edge and high-order interior
 * functions, which cuDSS's static pivoting is sensitive to: on hp systems with hanging nodes
 * it removes every perturbed pivot (ADR-0008). HPFEM_GPU_EQUILIBRATE=0 switches it off for
 * comparisons. Uses the pattern kept in the solver (host_row_ptr, host_col); writes the
 * scaled, equilibrated values and uploads D. */
hpfem_gpu_status scale_and_equilibrate(hpfem_gpu_solver* solver, const char* phase,
                                       const double* values, std::vector<double>& scaled) {
  const int64_t n = solver->n;
  const size_t n_size = static_cast<size_t>(n);
  const size_t nnz_size = static_cast<size_t>(solver->nnz);
  const int64_t* row_ptr = solver->host_row_ptr.data();
  const int64_t* col = solver->host_col.data();
  double max_abs = 0.0;
  for (size_t k = 0; k < nnz_size; ++k) {
    max_abs = std::max(max_abs, std::hypot(values[2 * k], values[2 * k + 1]));
  }
  if (!(max_abs > 0.0) || !std::isfinite(max_abs)) {
    return solver->fail(HPFEM_GPU_ERR_SINGULAR, std::string(phase) +
                                                    ": the matrix is zero or contains "
                                                    "non-finite entries");
  }
  solver->scale = 1.0 / max_abs;
  scaled.resize(2 * nnz_size);
  for (size_t k = 0; k < 2 * nnz_size; ++k) scaled[k] = values[k] * solver->scale;
  solver->equilibrated = false;
  const char* equilibrate = std::getenv("HPFEM_GPU_EQUILIBRATE");
  if (equilibrate == nullptr || *equilibrate != '0') {
    std::vector<double> d(n_size, 1.0);
    for (int64_t i = 0; i < n; ++i) {
      double diagonal = 0.0;
      double row_norm = 0.0;
      for (int64_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
        const double magnitude = std::hypot(scaled[2 * k], scaled[2 * k + 1]);
        row_norm += magnitude * magnitude;
        if (col[k] == i) diagonal = magnitude;
      }
      row_norm = std::sqrt(row_norm);
      // the diagonal whenever it exists (a tiny diagonal is still the right scale of its
      // row after a two-sided scaling), the row norm only where it is zero
      const double pivot = diagonal > 1e-300 ? diagonal : row_norm;
      d[static_cast<size_t>(i)] = pivot > 0.0 ? 1.0 / std::sqrt(pivot) : 1.0;
    }
    for (int64_t i = 0; i < n; ++i) {
      for (int64_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
        const double factor = d[static_cast<size_t>(i)] * d[static_cast<size_t>(col[k])];
        scaled[2 * k] *= factor;
        scaled[2 * k + 1] *= factor;
      }
    }
    HPFEM_GPU_CUDA(solver, "allocate equilibration",
                   solver->equilibration.reserve(n_size * sizeof(double)));
    HPFEM_GPU_CUDA(solver, "upload equilibration",
                   cudaMemcpy(solver->equilibration.ptr, d.data(), n_size * sizeof(double),
                              cudaMemcpyHostToDevice));
    solver->equilibrated = true;
  }
  return HPFEM_GPU_OK;
}

}  // namespace

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
  solver->host_row_ptr.assign(row_ptr, row_ptr + n + 1);
  solver->host_col.assign(col, col + nnz);

  const size_t n_size = static_cast<size_t>(n);
  const size_t nnz_size = static_cast<size_t>(nnz);
  std::vector<double> scaled;
  const hpfem_gpu_status prepared = scale_and_equilibrate(solver, "factorize", values, scaled);
  if (prepared != HPFEM_GPU_OK) return prepared;
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
  // hybrid memory mode: the factors live (partly) in host memory when they would not fit
  // the device. cuDSS wants the mode set before the analysis, so: analysis in the mode
  // requested or guessed, then the peak-memory estimates decide; a wrong guess repeats the
  // (cheap) analysis in the other mode. HPFEM_GPU_HYBRID=1/0 forces the mode.
  size_t free_device = 0, total_device = 0;
  if (status == CUDSS_STATUS_SUCCESS &&
      cudaMemGetInfo(&free_device, &total_device) != cudaSuccess) {
    cudaGetLastError();
    free_device = 0;
  }
  solver->device_free = free_device;
  const char* forced = std::getenv("HPFEM_GPU_HYBRID");
  const int force = (forced == nullptr || *forced == '\0') ? -1 : (*forced == '1' ? 1 : 0);
  int64_t estimates[16] = {};
  size_t written = 0;
  auto analyse = [&](bool hybrid) -> cudssStatus_t {
    int mode = hybrid ? 1 : 0;
    cudssStatus_t st =
        cudssConfigSet(solver->config, CUDSS_CONFIG_HYBRID_MEMORY_MODE, &mode, sizeof(mode));
    if (st != CUDSS_STATUS_SUCCESS) return st;
    st = cudssExecute(solver->handle, CUDSS_PHASE_ANALYSIS, solver->config, solver->data,
                      solver->matrix, x, b);
    if (st != CUDSS_STATUS_SUCCESS) return st;
    return cudssDataGet(solver->handle, solver->data, CUDSS_DATA_MEMORY_ESTIMATES, estimates,
                        sizeof(estimates), &written);
  };
  bool hybrid = force == 1;
  if (status == CUDSS_STATUS_SUCCESS) status = analyse(hybrid);
  if (status == CUDSS_STATUS_SUCCESS && force == -1 && !hybrid) {
    const size_t peak = static_cast<size_t>(estimates[1] > 0 ? estimates[1] : 0);
    if (free_device > 0 && peak > free_device * 9 / 10) {
      // too large for the device: redo the analysis in hybrid mode
      hybrid = true;
      cudssDataDestroy(solver->handle, solver->data);
      solver->data = nullptr;
      status = cudssDataCreate(solver->handle, &solver->data);
      if (status == CUDSS_STATUS_SUCCESS) status = analyse(true);
    }
  }
  solver->hybrid = hybrid;
  if (status == CUDSS_STATUS_SUCCESS) {
    // [0]/[1] device stable/peak, [2]/[3] host stable/peak, [4]/[5] hybrid peak GPU/CPU
    solver->device_estimate = static_cast<size_t>(hybrid ? estimates[4] : estimates[1]);
    solver->host_estimate = static_cast<size_t>(hybrid ? estimates[5] : estimates[3]);
    if (hybrid) {
      const size_t free_host = free_host_memory();
      if (free_host > 0 && solver->host_estimate > free_host * 9 / 10) {
        if (x != nullptr) cudssMatrixDestroy(x);
        if (b != nullptr) cudssMatrixDestroy(b);
        return solver->fail(
            HPFEM_GPU_ERR_OUT_OF_MEMORY,
            "factorize: the factors need about " + std::to_string(solver->host_estimate >> 20) +
                " MB of host memory in hybrid mode (" +
                std::to_string(solver->device_estimate >> 20) + " MB on the device), but only " +
                std::to_string(free_host >> 20) + " MB of host memory are free");
      }
      if (free_device > 0) {
        // leave a reserve for the work vectors and other users of the device
        int64_t limit = static_cast<int64_t>(free_device - free_device / 10);
        status = cudssConfigSet(solver->config, CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT, &limit,
                                sizeof(limit));
      }
    }
  }
  if (status == CUDSS_STATUS_SUCCESS) {
    status = cudssExecute(solver->handle, CUDSS_PHASE_FACTORIZATION, solver->config, solver->data,
                          solver->matrix, x, b);
  }
  cudaError_t sync = cudaStreamSynchronize(solver->stream);
  if (x != nullptr) cudssMatrixDestroy(x);
  if (b != nullptr) cudssMatrixDestroy(b);
  if (status == CUDSS_STATUS_ALLOC_FAILED) {
    return solver->fail(HPFEM_GPU_ERR_OUT_OF_MEMORY,
                        std::string("factorize: cuDSS ran out of memory (") +
                            (hybrid ? "hybrid mode, " : "device mode, ") + "estimates: device " +
                            std::to_string(solver->device_estimate >> 20) + " MB, host " +
                            std::to_string(solver->host_estimate >> 20) + " MB; free device " +
                            std::to_string(free_device >> 20) + " MB)");
  }
  if (status != CUDSS_STATUS_SUCCESS) return solver->fail_cudss("factorize", status);
  if (sync != cudaSuccess) return solver->fail_cuda("factorize: synchronise", sync);

  int info = 0;
  written = 0;
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

hpfem_gpu_status hpfem_gpu_refactorize(hpfem_gpu_solver* solver, int64_t nnz,
                                       const double* values) {
  if (solver == nullptr) return HPFEM_GPU_ERR_INVALID_ARG;
  if (!solver->factorized || solver->matrix == nullptr || solver->data == nullptr) {
    return solver->fail(HPFEM_GPU_ERR_INVALID_ARG, "refactorize: no factorisation to reuse");
  }
  if (nnz != solver->nnz || (nnz > 0 && values == nullptr)) {
    return solver->fail(HPFEM_GPU_ERR_INVALID_ARG,
                        "refactorize: nnz differs from the analysed pattern");
  }
  solver->factorized = false;
  const int64_t n = solver->n;
  const size_t nnz_size = static_cast<size_t>(nnz);
  std::vector<double> scaled;
  const hpfem_gpu_status prepared = scale_and_equilibrate(solver, "refactorize", values, scaled);
  if (prepared != HPFEM_GPU_OK) return prepared;
  if (nnz > 0) {
    HPFEM_GPU_CUDA(solver, "refactorize: upload values",
                   cudaMemcpyAsync(solver->values.ptr, scaled.data(), nnz_size * 2 * sizeof(double),
                                   cudaMemcpyHostToDevice, solver->stream));
  }
  // the CSR descriptor keeps pointing at the value buffer; the dense operands are not read
  cudssMatrix_t x = nullptr;
  cudssMatrix_t b = nullptr;
  HPFEM_GPU_CUDSS(
      solver, "refactorize: create dense operands",
      cudssMatrixCreateDn(&x, n, 1, n, solver->solution.ptr, CUDSS_C_64F, CUDSS_LAYOUT_COL_MAJOR));
  cudssStatus_t status =
      cudssMatrixCreateDn(&b, n, 1, n, solver->rhs.ptr, CUDSS_C_64F, CUDSS_LAYOUT_COL_MAJOR);
  if (status == CUDSS_STATUS_SUCCESS) {
    status = cudssExecute(solver->handle, CUDSS_PHASE_REFACTORIZATION, solver->config, solver->data,
                          solver->matrix, x, b);
  }
  const cudaError_t sync = cudaStreamSynchronize(solver->stream);
  if (x != nullptr) cudssMatrixDestroy(x);
  if (b != nullptr) cudssMatrixDestroy(b);
  if (status == CUDSS_STATUS_ALLOC_FAILED) {
    return solver->fail(HPFEM_GPU_ERR_OUT_OF_MEMORY, "refactorize: cuDSS ran out of memory");
  }
  if (status != CUDSS_STATUS_SUCCESS) return solver->fail_cudss("refactorize", status);
  if (sync != cudaSuccess) return solver->fail_cuda("refactorize: synchronise", sync);
  int perturbed = 0;
  size_t written = 0;
  if (cudssDataGet(solver->handle, solver->data, CUDSS_DATA_INFO, &perturbed, sizeof(perturbed),
                   &written) == CUDSS_STATUS_SUCCESS &&
      perturbed != 0) {
    return solver->fail(HPFEM_GPU_ERR_SINGULAR,
                        "refactorize: " + std::to_string(perturbed) +
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
  HPFEM_GPU_CUDA(solver, "solve: allocate input",
                 solver->input.reserve(count * sizeof(cuDoubleComplex)));
  HPFEM_GPU_CUDA(solver, "solve: allocate output",
                 solver->output.reserve(count * sizeof(cuDoubleComplex)));
  // HPFEM_GPU_TIMING=1 prints the split upload / solve / download of every call to stderr
  static const bool timing = [] {
    const char* env = std::getenv("HPFEM_GPU_TIMING");
    return env != nullptr && *env == '1';
  }();
  const auto t0 = std::chrono::steady_clock::now();
  HPFEM_GPU_CUDA(solver, "solve: upload right-hand side",
                 cudaMemcpyAsync(solver->input.ptr, b, count * sizeof(cuDoubleComplex),
                                 cudaMemcpyHostToDevice, solver->stream));
  if (timing) HPFEM_GPU_CUDA(solver, "solve: timing", cudaStreamSynchronize(solver->stream));
  const auto t1 = std::chrono::steady_clock::now();
  const hpfem_gpu_status status = hpfem_gpu_solve_device(
      solver, nrhs, solver->input.as<cuDoubleComplex>(), solver->output.as<cuDoubleComplex>());
  if (status != HPFEM_GPU_OK) return status;
  const auto t2 = std::chrono::steady_clock::now();
  HPFEM_GPU_CUDA(
      solver, "solve: download solution",
      cudaMemcpy(x, solver->output.ptr, count * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
  if (timing) {
    const auto t3 = std::chrono::steady_clock::now();
    const auto ms = [](auto from, auto to) {
      return std::chrono::duration<double, std::milli>(to - from).count();
    };
    std::fprintf(
        stderr,
        "hpfem_gpu_solve: n = %lld, nrhs = %lld: upload %.1f ms, solve %.1f ms, download %.1f ms\n",
        static_cast<long long>(solver->n), static_cast<long long>(nrhs), ms(t0, t1), ms(t1, t2),
        ms(t2, t3));
  }
  return HPFEM_GPU_OK;
}

hpfem_gpu_status hpfem_gpu_factor_info2(const hpfem_gpu_solver* solver,
                                        hpfem_gpu_factor_info_t* info) {
  if (solver == nullptr || info == nullptr) return HPFEM_GPU_ERR_INVALID_ARG;
  *info = hpfem_gpu_factor_info_t{};
  hpfem_gpu_factor_info(solver, &info->nnz_factors, &info->device_bytes);
  info->hybrid = solver->hybrid ? 1 : 0;
  info->device_estimate = solver->factorized ? solver->device_estimate : 0;
  info->host_estimate = solver->factorized ? solver->host_estimate : 0;
  info->device_free = solver->device_free;
  if (solver->factorized && solver->hybrid) {
    // the host part of the factors: cuDSS's stable host estimate of the hybrid mode
    int64_t estimates[16] = {};
    size_t written = 0;
    if (cudssDataGet(solver->handle, solver->data, CUDSS_DATA_MEMORY_ESTIMATES, estimates,
                     sizeof(estimates), &written) == CUDSS_STATUS_SUCCESS) {
      info->host_bytes = static_cast<size_t>(estimates[5] > 0 ? estimates[5] : 0);
    }
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
        // device mode: the stable device estimate; hybrid mode: the device part of the
        // hybrid peak (the rest of the factors lives in host memory)
        const int64_t device_part = solver->hybrid ? estimates[4] : estimates[0];
        *device_bytes += static_cast<size_t>(device_part > 0 ? device_part : 0);
      }
    }
  }
  return HPFEM_GPU_OK;
}

const char* hpfem_gpu_last_error(const hpfem_gpu_solver* solver) {
  return solver == nullptr ? "" : solver->last_error.c_str();
}

}  // extern "C"

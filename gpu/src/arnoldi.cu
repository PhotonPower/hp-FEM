// Shift-invert Arnoldi with the Krylov basis on the device (API version 4). The host keeps
// the control flow of solvers::complex_eigenpairs_near (restarts, Hessenberg eigenproblem,
// Ritz selection); the device holds V (n x (ncv + 1)), applies the operator
//   w = P K^{-1} (B v_j),   P w = w - G K_g^{-1} G^H B w  (optional gauge projection),
// orthogonalises w against V[:, 0..j] with classical Gram-Schmidt applied twice (two GEMV
// pairs per pass), normalises, stores v_{j+1} and returns the Hessenberg column. Restart
// vectors and Ritz vectors are linear combinations V_m c computed on the device.
#include <cmath>
#include <cstdint>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <new>
#include <string>
#include <vector>

#include "hpfem_gpu.h"
#include "internal.hpp"

namespace {

constexpr int kThreads = 256;

__device__ inline cuDoubleComplex cmul(cuDoubleComplex a, cuDoubleComplex b) {
  return make_cuDoubleComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

}  // namespace

/// h[k] = conj(V[:, k]) . w for k < count: one block per column, shared-memory reduction.
__global__ void hpfem_gpu_dots_conj(int64_t n, const cuDoubleComplex* __restrict__ v,
                                    const cuDoubleComplex* __restrict__ w,
                                    cuDoubleComplex* __restrict__ h) {
  __shared__ double re[kThreads];
  __shared__ double im[kThreads];
  const cuDoubleComplex* column = v + static_cast<int64_t>(blockIdx.x) * n;
  double sr = 0.0, si = 0.0;
  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
    const cuDoubleComplex a = column[i];
    const cuDoubleComplex b = w[i];
    sr += a.x * b.x + a.y * b.y;  // conj(a) * b
    si += a.x * b.y - a.y * b.x;
  }
  re[threadIdx.x] = sr;
  im[threadIdx.x] = si;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      re[threadIdx.x] += re[threadIdx.x + stride];
      im[threadIdx.x] += im[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) h[blockIdx.x] = make_cuDoubleComplex(re[0], im[0]);
}

/// w -= sum_k h[k] V[:, k] (count columns); one thread per row.
__global__ void hpfem_gpu_subtract_combination(int64_t n, int64_t count,
                                               const cuDoubleComplex* __restrict__ v,
                                               const cuDoubleComplex* __restrict__ h,
                                               cuDoubleComplex* __restrict__ w) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double re = w[i].x, im = w[i].y;
  for (int64_t k = 0; k < count; ++k) {
    const cuDoubleComplex p = cmul(h[k], v[k * n + i]);
    re -= p.x;
    im -= p.y;
  }
  w[i] = make_cuDoubleComplex(re, im);
}

/// out = sum_k c[k] V[:, k] (count columns) scaled by `factor`; one thread per row.
__global__ void hpfem_gpu_combination(int64_t n, int64_t count,
                                      const cuDoubleComplex* __restrict__ v,
                                      const cuDoubleComplex* __restrict__ c, double factor,
                                      cuDoubleComplex* __restrict__ out) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double re = 0.0, im = 0.0;
  for (int64_t k = 0; k < count; ++k) {
    const cuDoubleComplex p = cmul(c[k], v[k * n + i]);
    re += p.x;
    im += p.y;
  }
  out[i] = make_cuDoubleComplex(re * factor, im * factor);
}

/// partial[blockIdx] = sum |w_i|^2 over the block's rows.
__global__ void hpfem_gpu_norm_squared(int64_t n, const cuDoubleComplex* __restrict__ w,
                                       double* __restrict__ partial) {
  __shared__ double acc[kThreads];
  double s = 0.0;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    s += w[i].x * w[i].x + w[i].y * w[i].y;
  }
  acc[threadIdx.x] = s;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) acc[threadIdx.x] += acc[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) partial[blockIdx.x] = acc[0];
}

/// z = x - y (projection update) or out = in * factor; small helpers.
__global__ void hpfem_gpu_subtract(int64_t n, const cuDoubleComplex* __restrict__ x,
                                   const cuDoubleComplex* __restrict__ y,
                                   cuDoubleComplex* __restrict__ z) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  z[i] = make_cuDoubleComplex(x[i].x - y[i].x, x[i].y - y[i].y);
}

__global__ void hpfem_gpu_scale(int64_t n, double factor, const cuDoubleComplex* __restrict__ in,
                                cuDoubleComplex* __restrict__ out) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = make_cuDoubleComplex(in[i].x * factor, in[i].y * factor);
}

struct hpfem_gpu_arnoldi {
  hpfem_gpu_solver* solver = nullptr;            // K = A - sigma B, factorised
  hpfem_gpu_matrix* b = nullptr;                 // B (n x n)
  hpfem_gpu_matrix* gradient = nullptr;          // G (n x m), optional
  hpfem_gpu_matrix* gradient_adjoint = nullptr;  // G^H (m x n), optional
  hpfem_gpu_solver* gauge = nullptr;             // K_g = G^H B G, factorised, optional
  int64_t n = 0;
  int64_t ncv = 0;
  int64_t m_gauge = 0;
  DeviceBuffer basis;             // V, n x (ncv + 1) column major
  DeviceBuffer w, tmp, tmp2;      // n
  DeviceBuffer small;             // ncv + 1 complex (coefficients)
  DeviceBuffer gauge_r, gauge_z;  // m_gauge
  DeviceBuffer partial;           // norm reduction
  std::vector<cuDoubleComplex> host_small;
  std::vector<double> host_partial;
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
  [[nodiscard]] cudaStream_t stream() const { return solver->stream; }
  [[nodiscard]] cuDoubleComplex* column(int64_t j) const {
    return basis.as<cuDoubleComplex>() + j * n;
  }
  [[nodiscard]] unsigned blocks() const {
    return static_cast<unsigned>((n + kThreads - 1) / kThreads);
  }
};

#define HPFEM_GPU_ARNOLDI_CUDA(arnoldi, what, call)                                             \
  do {                                                                                          \
    const cudaError_t hpfem_gpu_status_ = (call);                                               \
    if (hpfem_gpu_status_ != cudaSuccess) return (arnoldi)->fail_cuda(what, hpfem_gpu_status_); \
  } while (0)

namespace {

/// ||w||_2 (synchronises).
hpfem_gpu_status norm(hpfem_gpu_arnoldi* a, const cuDoubleComplex* w, double* out) {
  const unsigned blocks = std::min<unsigned>(a->blocks(), 1024u);
  hpfem_gpu_norm_squared<<<blocks, kThreads, 0, a->stream()>>>(a->n, w, a->partial.as<double>());
  HPFEM_GPU_ARNOLDI_CUDA(a, "arnoldi: norm", cudaGetLastError());
  HPFEM_GPU_ARNOLDI_CUDA(
      a, "arnoldi: norm download",
      cudaMemcpyAsync(a->host_partial.data(), a->partial.ptr, blocks * sizeof(double),
                      cudaMemcpyDeviceToHost, a->stream()));
  HPFEM_GPU_ARNOLDI_CUDA(a, "arnoldi: norm synchronise", cudaStreamSynchronize(a->stream()));
  double s = 0.0;
  for (unsigned i = 0; i < blocks; ++i) s += a->host_partial[i];
  *out = std::sqrt(s);
  return HPFEM_GPU_OK;
}

/// w <- P w = w - G K_g^{-1} G^H (B w) when the gauge is set.
hpfem_gpu_status project(hpfem_gpu_arnoldi* a, cuDoubleComplex* w) {
  if (a->gauge == nullptr) return HPFEM_GPU_OK;
  auto* bw = a->tmp.as<cuDoubleComplex>();
  hpfem_gpu_matrix_apply_device(a->b, w, bw, a->stream());
  hpfem_gpu_matrix_apply_device(a->gradient_adjoint, bw, a->gauge_r.as<cuDoubleComplex>(),
                                a->stream());
  HPFEM_GPU_ARNOLDI_CUDA(a, "arnoldi: gauge products", cudaGetLastError());
  // the gauge solver works on its own stream: wait for the products before it starts (the
  // solve synchronises its stream at the end, so the following kernels see its result)
  HPFEM_GPU_ARNOLDI_CUDA(a, "arnoldi: wait for the gauge products",
                         cudaStreamSynchronize(a->stream()));
  const hpfem_gpu_status solved = hpfem_gpu_solve_device(
      a->gauge, 1, a->gauge_r.as<cuDoubleComplex>(), a->gauge_z.as<cuDoubleComplex>());
  if (solved != HPFEM_GPU_OK)
    return a->fail(solved, "arnoldi: gauge solve: " + a->gauge->last_error);
  hpfem_gpu_matrix_apply_device(a->gradient, a->gauge_z.as<cuDoubleComplex>(), bw, a->stream());
  hpfem_gpu_subtract<<<a->blocks(), kThreads, 0, a->stream()>>>(a->n, w, bw, w);
  HPFEM_GPU_ARNOLDI_CUDA(a, "arnoldi: projection", cudaGetLastError());
  return HPFEM_GPU_OK;
}

}  // namespace

extern "C" {

hpfem_gpu_status hpfem_gpu_arnoldi_create(hpfem_gpu_arnoldi** out, hpfem_gpu_solver* shifted,
                                          hpfem_gpu_matrix* b, hpfem_gpu_matrix* gradient,
                                          hpfem_gpu_matrix* gradient_adjoint,
                                          hpfem_gpu_solver* gauge, int64_t n, int64_t ncv) {
  if (out == nullptr) return HPFEM_GPU_ERR_INVALID_ARG;
  *out = nullptr;
  if (shifted == nullptr || b == nullptr || n <= 0 || ncv <= 0 || !shifted->factorized ||
      shifted->n != n || b->rows != n || b->cols != n) {
    return HPFEM_GPU_ERR_INVALID_ARG;
  }
  const bool gauged = gradient != nullptr || gradient_adjoint != nullptr || gauge != nullptr;
  if (gauged && (gradient == nullptr || gradient_adjoint == nullptr || gauge == nullptr ||
                 !gauge->factorized || gradient->rows != n || gradient_adjoint->cols != n ||
                 gradient->cols != gradient_adjoint->rows || gauge->n != gradient->cols)) {
    return HPFEM_GPU_ERR_INVALID_ARG;
  }
  hpfem_gpu_arnoldi* a = new (std::nothrow) hpfem_gpu_arnoldi();
  if (a == nullptr) return HPFEM_GPU_ERR_OUT_OF_MEMORY;
  a->solver = shifted;
  a->b = b;
  a->gradient = gradient;
  a->gradient_adjoint = gradient_adjoint;
  a->gauge = gauge;
  a->n = n;
  a->ncv = ncv;
  a->m_gauge = gauged ? gradient->cols : 0;
  const size_t vec = static_cast<size_t>(n) * sizeof(cuDoubleComplex);
  const auto allocate = [&]() -> hpfem_gpu_status {
    HPFEM_GPU_ARNOLDI_CUDA(a, "arnoldi: allocate basis",
                           a->basis.reserve(vec * static_cast<size_t>(ncv + 1)));
    for (DeviceBuffer* buffer : {&a->w, &a->tmp, &a->tmp2}) {
      HPFEM_GPU_ARNOLDI_CUDA(a, "arnoldi: allocate work vectors", buffer->reserve(vec));
    }
    HPFEM_GPU_ARNOLDI_CUDA(
        a, "arnoldi: allocate coefficients",
        a->small.reserve(static_cast<size_t>(ncv + 1) * sizeof(cuDoubleComplex)));
    HPFEM_GPU_ARNOLDI_CUDA(a, "arnoldi: allocate reduction",
                           a->partial.reserve(1024 * sizeof(double)));
    if (gauged) {
      const size_t gvec = static_cast<size_t>(a->m_gauge) * sizeof(cuDoubleComplex);
      HPFEM_GPU_ARNOLDI_CUDA(a, "arnoldi: allocate gauge vectors", a->gauge_r.reserve(gvec));
      HPFEM_GPU_ARNOLDI_CUDA(a, "arnoldi: allocate gauge vectors", a->gauge_z.reserve(gvec));
    }
    return HPFEM_GPU_OK;
  };
  const hpfem_gpu_status status = allocate();
  if (status != HPFEM_GPU_OK) {
    delete a;
    return status;
  }
  a->host_small.resize(static_cast<size_t>(ncv + 1));
  a->host_partial.resize(1024);
  *out = a;
  return HPFEM_GPU_OK;
}

void hpfem_gpu_arnoldi_destroy(hpfem_gpu_arnoldi* a) {
  delete a;
}

hpfem_gpu_status hpfem_gpu_arnoldi_set_start(hpfem_gpu_arnoldi* a, const double* start) {
  if (a == nullptr) return HPFEM_GPU_ERR_INVALID_ARG;
  if (start == nullptr) return a->fail(HPFEM_GPU_ERR_INVALID_ARG, "set_start: null array");
  const size_t vec = static_cast<size_t>(a->n) * sizeof(cuDoubleComplex);
  HPFEM_GPU_ARNOLDI_CUDA(
      a, "set_start: upload",
      cudaMemcpyAsync(a->w.ptr, start, vec, cudaMemcpyHostToDevice, a->stream()));
  const hpfem_gpu_status projected = project(a, a->w.as<cuDoubleComplex>());
  if (projected != HPFEM_GPU_OK) return projected;
  double length = 0.0;
  const hpfem_gpu_status normed = norm(a, a->w.as<cuDoubleComplex>(), &length);
  if (normed != HPFEM_GPU_OK) return normed;
  if (!(length > 0.0)) return a->fail(HPFEM_GPU_ERR_INVALID_ARG, "set_start: zero start vector");
  hpfem_gpu_scale<<<a->blocks(), kThreads, 0, a->stream()>>>(
      a->n, 1.0 / length, a->w.as<cuDoubleComplex>(), a->column(0));
  HPFEM_GPU_ARNOLDI_CUDA(a, "set_start: normalise", cudaGetLastError());
  return HPFEM_GPU_OK;
}

hpfem_gpu_status hpfem_gpu_arnoldi_iterate(hpfem_gpu_arnoldi* a, int64_t j, double* h_column,
                                           double* beta) {
  if (a == nullptr) return HPFEM_GPU_ERR_INVALID_ARG;
  if (j < 0 || j >= a->ncv || h_column == nullptr || beta == nullptr) {
    return a->fail(HPFEM_GPU_ERR_INVALID_ARG, "iterate: bad column or null array");
  }
  auto* w = a->w.as<cuDoubleComplex>();
  auto* bv = a->tmp2.as<cuDoubleComplex>();
  // w = P K^{-1} (B v_j)
  hpfem_gpu_matrix_apply_device(a->b, a->column(j), bv, a->stream());
  HPFEM_GPU_ARNOLDI_CUDA(a, "iterate: B v", cudaGetLastError());
  const hpfem_gpu_status solved = hpfem_gpu_solve_device(a->solver, 1, bv, w);
  if (solved != HPFEM_GPU_OK) return a->fail(solved, "iterate: solve: " + a->solver->last_error);
  const hpfem_gpu_status projected = project(a, w);
  if (projected != HPFEM_GPU_OK) return projected;
  // classical Gram-Schmidt twice against v_0 .. v_j
  const int64_t count = j + 1;
  auto* coefficients = a->small.as<cuDoubleComplex>();
  std::vector<cuDoubleComplex> accumulated(static_cast<size_t>(count),
                                           make_cuDoubleComplex(0.0, 0.0));
  for (int pass = 0; pass < 2; ++pass) {
    hpfem_gpu_dots_conj<<<static_cast<unsigned>(count), kThreads, 0, a->stream()>>>(
        a->n, a->basis.as<cuDoubleComplex>(), w, coefficients);
    hpfem_gpu_subtract_combination<<<a->blocks(), kThreads, 0, a->stream()>>>(
        a->n, count, a->basis.as<cuDoubleComplex>(), coefficients, w);
    HPFEM_GPU_ARNOLDI_CUDA(a, "iterate: Gram-Schmidt", cudaGetLastError());
    HPFEM_GPU_ARNOLDI_CUDA(a, "iterate: coefficients download",
                           cudaMemcpyAsync(a->host_small.data(), coefficients,
                                           static_cast<size_t>(count) * sizeof(cuDoubleComplex),
                                           cudaMemcpyDeviceToHost, a->stream()));
    HPFEM_GPU_ARNOLDI_CUDA(a, "iterate: synchronise", cudaStreamSynchronize(a->stream()));
    for (int64_t i = 0; i < count; ++i) {
      accumulated[static_cast<size_t>(i)].x += a->host_small[static_cast<size_t>(i)].x;
      accumulated[static_cast<size_t>(i)].y += a->host_small[static_cast<size_t>(i)].y;
    }
  }
  double length = 0.0;
  const hpfem_gpu_status normed = norm(a, w, &length);
  if (normed != HPFEM_GPU_OK) return normed;
  for (int64_t i = 0; i < count; ++i) {
    h_column[2 * i] = accumulated[static_cast<size_t>(i)].x;
    h_column[2 * i + 1] = accumulated[static_cast<size_t>(i)].y;
  }
  *beta = length;
  if (length > 0.0) {
    hpfem_gpu_scale<<<a->blocks(), kThreads, 0, a->stream()>>>(a->n, 1.0 / length, w,
                                                               a->column(j + 1));
    HPFEM_GPU_ARNOLDI_CUDA(a, "iterate: store v_{j+1}", cudaGetLastError());
  }
  return HPFEM_GPU_OK;
}

hpfem_gpu_status hpfem_gpu_arnoldi_combine(hpfem_gpu_arnoldi* a, int64_t m, int64_t count,
                                           const double* coefficients, double* out) {
  if (a == nullptr) return HPFEM_GPU_ERR_INVALID_ARG;
  if (m <= 0 || m > a->ncv + 1 || count <= 0 || coefficients == nullptr || out == nullptr) {
    return a->fail(HPFEM_GPU_ERR_INVALID_ARG, "combine: bad sizes or null array");
  }
  const size_t vec = static_cast<size_t>(a->n) * sizeof(cuDoubleComplex);
  for (int64_t c = 0; c < count; ++c) {
    HPFEM_GPU_ARNOLDI_CUDA(a, "combine: upload coefficients",
                           cudaMemcpyAsync(a->small.ptr, coefficients + 2 * c * m,
                                           static_cast<size_t>(m) * sizeof(cuDoubleComplex),
                                           cudaMemcpyHostToDevice, a->stream()));
    hpfem_gpu_combination<<<a->blocks(), kThreads, 0, a->stream()>>>(
        a->n, m, a->basis.as<cuDoubleComplex>(), a->small.as<cuDoubleComplex>(), 1.0,
        a->w.as<cuDoubleComplex>());
    HPFEM_GPU_ARNOLDI_CUDA(a, "combine: launch", cudaGetLastError());
    HPFEM_GPU_ARNOLDI_CUDA(
        a, "combine: download",
        cudaMemcpyAsync(out + 2 * c * a->n, a->w.ptr, vec, cudaMemcpyDeviceToHost, a->stream()));
  }
  HPFEM_GPU_ARNOLDI_CUDA(a, "combine: synchronise", cudaStreamSynchronize(a->stream()));
  return HPFEM_GPU_OK;
}

hpfem_gpu_status hpfem_gpu_arnoldi_restart(hpfem_gpu_arnoldi* a, int64_t m,
                                           const double* coefficients) {
  if (a == nullptr) return HPFEM_GPU_ERR_INVALID_ARG;
  if (m <= 0 || m > a->ncv + 1 || coefficients == nullptr) {
    return a->fail(HPFEM_GPU_ERR_INVALID_ARG, "restart: bad size or null array");
  }
  HPFEM_GPU_ARNOLDI_CUDA(
      a, "restart: upload coefficients",
      cudaMemcpyAsync(a->small.ptr, coefficients, static_cast<size_t>(m) * sizeof(cuDoubleComplex),
                      cudaMemcpyHostToDevice, a->stream()));
  hpfem_gpu_combination<<<a->blocks(), kThreads, 0, a->stream()>>>(
      a->n, m, a->basis.as<cuDoubleComplex>(), a->small.as<cuDoubleComplex>(), 1.0,
      a->w.as<cuDoubleComplex>());
  HPFEM_GPU_ARNOLDI_CUDA(a, "restart: combination", cudaGetLastError());
  double length = 0.0;
  const hpfem_gpu_status normed = norm(a, a->w.as<cuDoubleComplex>(), &length);
  if (normed != HPFEM_GPU_OK) return normed;
  if (!(length > 0.0)) return a->fail(HPFEM_GPU_ERR_INVALID_ARG, "restart: zero vector");
  hpfem_gpu_scale<<<a->blocks(), kThreads, 0, a->stream()>>>(
      a->n, 1.0 / length, a->w.as<cuDoubleComplex>(), a->column(0));
  HPFEM_GPU_ARNOLDI_CUDA(a, "restart: normalise", cudaGetLastError());
  return HPFEM_GPU_OK;
}

const char* hpfem_gpu_arnoldi_last_error(const hpfem_gpu_arnoldi* a) {
  return a == nullptr ? "" : a->last_error.c_str();
}

}  // extern "C"

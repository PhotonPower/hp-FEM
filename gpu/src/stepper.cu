// Newmark time stepper on the device (API version 3): the state vectors u, v, a of the
// transient solver stay on the GPU, every step is two sparse products, a few fused vector
// updates and one solve with the factorised Newmark operator, without host contact except
// the scalar of the load. The recursion is exactly that of physics::TimeDomain::step:
//   u_pred = u + dt v + dt^2 (1/2 - beta) a,   v_pred = v + dt (1 - gamma) a,
//   a_new = K^{-1} (scale * b_J - C v_pred - S u_pred),
//   u = u_pred + beta dt^2 a_new,   v = v_pred + gamma dt a_new,   a = a_new.
#include <cstdint>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <new>
#include <string>

#include "hpfem_gpu.h"
#include "internal.hpp"

/// z = x + alpha y + beta w (real scalars; `w` may be null for z = x + alpha y).
__global__ void hpfem_gpu_axpby(int64_t n, const cuDoubleComplex* __restrict__ x, double alpha,
                                const cuDoubleComplex* __restrict__ y, double beta,
                                const cuDoubleComplex* __restrict__ w,
                                cuDoubleComplex* __restrict__ z) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double re = x[i].x + alpha * y[i].x;
  double im = x[i].y + alpha * y[i].y;
  if (w != nullptr) {
    re += beta * w[i].x;
    im += beta * w[i].y;
  }
  z[i] = make_cuDoubleComplex(re, im);
}

/// r = scale * b - c - s (b may be null: r = -c - s), the right-hand side of the step.
__global__ void hpfem_gpu_step_rhs(int64_t n, double scale, const cuDoubleComplex* __restrict__ b,
                                   const cuDoubleComplex* __restrict__ c,
                                   const cuDoubleComplex* __restrict__ s,
                                   cuDoubleComplex* __restrict__ r) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const double bre = b != nullptr ? scale * b[i].x : 0.0;
  const double bim = b != nullptr ? scale * b[i].y : 0.0;
  r[i] = make_cuDoubleComplex(bre - c[i].x - s[i].x, bim - c[i].y - s[i].y);
}

namespace {

unsigned blocks_for(int64_t n) {
  return static_cast<unsigned>((n + 255) / 256);
}

}  // namespace

struct hpfem_gpu_stepper {
  hpfem_gpu_solver* solver = nullptr;
  hpfem_gpu_matrix* damping = nullptr;  // may be null (no conductivity, no absorbing walls)
  hpfem_gpu_matrix* stiffness = nullptr;
  int64_t n = 0;
  double dt = 0;
  double beta = 0.25;
  double gamma = 0.5;
  bool has_load = false;
  DeviceBuffer u, v, a, u_pred, v_pred, rhs, tmp_c, tmp_s, load;
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
};

#define HPFEM_GPU_STEPPER_CUDA(stepper, what, call)                                             \
  do {                                                                                          \
    const cudaError_t hpfem_gpu_status_ = (call);                                               \
    if (hpfem_gpu_status_ != cudaSuccess) return (stepper)->fail_cuda(what, hpfem_gpu_status_); \
  } while (0)

extern "C" {

hpfem_gpu_status hpfem_gpu_stepper_create(hpfem_gpu_stepper** out, hpfem_gpu_solver* newmark,
                                          hpfem_gpu_matrix* damping, hpfem_gpu_matrix* stiffness,
                                          int64_t n, const double* load, double dt, double beta,
                                          double gamma) {
  if (out == nullptr) return HPFEM_GPU_ERR_INVALID_ARG;
  *out = nullptr;
  if (newmark == nullptr || stiffness == nullptr || n <= 0 || !(dt > 0.0)) {
    return HPFEM_GPU_ERR_INVALID_ARG;
  }
  if (!newmark->factorized || newmark->n != n || stiffness->n != n ||
      (damping != nullptr && damping->n != n)) {
    return HPFEM_GPU_ERR_INVALID_ARG;
  }
  hpfem_gpu_stepper* stepper = new (std::nothrow) hpfem_gpu_stepper();
  if (stepper == nullptr) return HPFEM_GPU_ERR_OUT_OF_MEMORY;
  stepper->solver = newmark;
  stepper->damping = damping;
  stepper->stiffness = stiffness;
  stepper->n = n;
  stepper->dt = dt;
  stepper->beta = beta;
  stepper->gamma = gamma;
  stepper->has_load = load != nullptr;
  const size_t bytes = static_cast<size_t>(n) * sizeof(cuDoubleComplex);
  const auto allocate = [&]() -> hpfem_gpu_status {
    for (DeviceBuffer* buffer :
         {&stepper->u, &stepper->v, &stepper->a, &stepper->u_pred, &stepper->v_pred, &stepper->rhs,
          &stepper->tmp_c, &stepper->tmp_s}) {
      HPFEM_GPU_STEPPER_CUDA(stepper, "stepper: allocate state", buffer->reserve(bytes));
      HPFEM_GPU_STEPPER_CUDA(stepper, "stepper: clear state", cudaMemset(buffer->ptr, 0, bytes));
    }
    if (load != nullptr) {
      HPFEM_GPU_STEPPER_CUDA(stepper, "stepper: allocate load", stepper->load.reserve(bytes));
      HPFEM_GPU_STEPPER_CUDA(stepper, "stepper: upload load",
                             cudaMemcpy(stepper->load.ptr, load, bytes, cudaMemcpyHostToDevice));
    }
    return HPFEM_GPU_OK;
  };
  const hpfem_gpu_status status = allocate();
  if (status != HPFEM_GPU_OK) {
    delete stepper;
    return status;
  }
  *out = stepper;
  return HPFEM_GPU_OK;
}

void hpfem_gpu_stepper_destroy(hpfem_gpu_stepper* stepper) {
  delete stepper;
}

hpfem_gpu_status hpfem_gpu_stepper_set_state(hpfem_gpu_stepper* stepper, const double* u,
                                             const double* v, const double* a) {
  if (stepper == nullptr) return HPFEM_GPU_ERR_INVALID_ARG;
  if (u == nullptr || v == nullptr || a == nullptr) {
    return stepper->fail(HPFEM_GPU_ERR_INVALID_ARG, "set_state: null array");
  }
  const size_t bytes = static_cast<size_t>(stepper->n) * sizeof(cuDoubleComplex);
  HPFEM_GPU_STEPPER_CUDA(stepper, "set_state: upload u",
                         cudaMemcpy(stepper->u.ptr, u, bytes, cudaMemcpyHostToDevice));
  HPFEM_GPU_STEPPER_CUDA(stepper, "set_state: upload v",
                         cudaMemcpy(stepper->v.ptr, v, bytes, cudaMemcpyHostToDevice));
  HPFEM_GPU_STEPPER_CUDA(stepper, "set_state: upload a",
                         cudaMemcpy(stepper->a.ptr, a, bytes, cudaMemcpyHostToDevice));
  return HPFEM_GPU_OK;
}

hpfem_gpu_status hpfem_gpu_stepper_get_state(const hpfem_gpu_stepper* stepper, double* u, double* v,
                                             double* a) {
  if (stepper == nullptr) return HPFEM_GPU_ERR_INVALID_ARG;
  const size_t bytes = static_cast<size_t>(stepper->n) * sizeof(cuDoubleComplex);
  auto* mutable_stepper = const_cast<hpfem_gpu_stepper*>(stepper);  // error string only
  HPFEM_GPU_STEPPER_CUDA(mutable_stepper, "get_state: synchronise",
                         cudaStreamSynchronize(stepper->stream()));
  if (u != nullptr) {
    HPFEM_GPU_STEPPER_CUDA(mutable_stepper, "get_state: download u",
                           cudaMemcpy(u, stepper->u.ptr, bytes, cudaMemcpyDeviceToHost));
  }
  if (v != nullptr) {
    HPFEM_GPU_STEPPER_CUDA(mutable_stepper, "get_state: download v",
                           cudaMemcpy(v, stepper->v.ptr, bytes, cudaMemcpyDeviceToHost));
  }
  if (a != nullptr) {
    HPFEM_GPU_STEPPER_CUDA(mutable_stepper, "get_state: download a",
                           cudaMemcpy(a, stepper->a.ptr, bytes, cudaMemcpyDeviceToHost));
  }
  return HPFEM_GPU_OK;
}

hpfem_gpu_status hpfem_gpu_stepper_step(hpfem_gpu_stepper* stepper, double load_scale) {
  if (stepper == nullptr) return HPFEM_GPU_ERR_INVALID_ARG;
  const int64_t n = stepper->n;
  const double dt = stepper->dt;
  cudaStream_t stream = stepper->stream();
  const unsigned blocks = blocks_for(n);
  auto* u = stepper->u.as<cuDoubleComplex>();
  auto* v = stepper->v.as<cuDoubleComplex>();
  auto* a = stepper->a.as<cuDoubleComplex>();
  auto* u_pred = stepper->u_pred.as<cuDoubleComplex>();
  auto* v_pred = stepper->v_pred.as<cuDoubleComplex>();
  auto* rhs = stepper->rhs.as<cuDoubleComplex>();
  auto* cv = stepper->tmp_c.as<cuDoubleComplex>();
  auto* su = stepper->tmp_s.as<cuDoubleComplex>();
  // predictors
  hpfem_gpu_axpby<<<blocks, 256, 0, stream>>>(n, u, dt, v, dt * dt * (0.5 - stepper->beta), a,
                                              u_pred);
  hpfem_gpu_axpby<<<blocks, 256, 0, stream>>>(n, v, dt * (1.0 - stepper->gamma), a, 0.0, nullptr,
                                              v_pred);
  // right-hand side: scale * b_J - C v_pred - S u_pred
  if (stepper->damping != nullptr) {
    hpfem_gpu_matrix_apply_device(stepper->damping, v_pred, cv, stream);
  } else {
    HPFEM_GPU_STEPPER_CUDA(
        stepper, "step: clear damping product",
        cudaMemsetAsync(cv, 0, static_cast<size_t>(n) * sizeof(cuDoubleComplex), stream));
  }
  hpfem_gpu_matrix_apply_device(stepper->stiffness, u_pred, su, stream);
  hpfem_gpu_step_rhs<<<blocks, 256, 0, stream>>>(
      n, load_scale, stepper->has_load ? stepper->load.as<cuDoubleComplex>() : nullptr, cv, su,
      rhs);
  HPFEM_GPU_STEPPER_CUDA(stepper, "step: launch", cudaGetLastError());
  // a_new = K^{-1} rhs (the solver synchronises its stream)
  const hpfem_gpu_status solved = hpfem_gpu_solve_device(stepper->solver, 1, rhs, a);
  if (solved != HPFEM_GPU_OK) {
    return stepper->fail(solved, std::string("step: ") + stepper->solver->last_error);
  }
  // correctors
  hpfem_gpu_axpby<<<blocks, 256, 0, stream>>>(n, u_pred, stepper->beta * dt * dt, a, 0.0, nullptr,
                                              u);
  hpfem_gpu_axpby<<<blocks, 256, 0, stream>>>(n, v_pred, stepper->gamma * dt, a, 0.0, nullptr, v);
  HPFEM_GPU_STEPPER_CUDA(stepper, "step: launch correctors", cudaGetLastError());
  return HPFEM_GPU_OK;
}

const char* hpfem_gpu_stepper_last_error(const hpfem_gpu_stepper* stepper) {
  return stepper == nullptr ? "" : stepper->last_error.c_str();
}

}  // extern "C"

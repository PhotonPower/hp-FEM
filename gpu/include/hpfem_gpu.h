/* hpfem_gpu.h — C interface of the hp-FEM GPU direct-solver DLL (cuDSS).
 *
 * The library (MinGW / GCC / Clang build) loads this DLL at run time and talks to it only
 * through the functions declared here; nothing of CUDA or cuDSS leaks across the boundary.
 * See docs/adr/0008-gpu-backend.md and gpu/README.md.
 *
 * Conventions
 * - Matrices are square CSR, zero based, with 64-bit row pointers and column indices.
 * - Complex values are interleaved double pairs (re, im), layout compatible with
 *   std::complex<double> and cuDoubleComplex.
 * - Dense right-hand sides / solutions are n x nrhs, column major, leading dimension n.
 * - A solver object holds one factorisation (factorise once, solve many); factorize() may
 *   be called again on the same object with a new matrix (re-factorisation in a sweep),
 *   which discards the previous factors. An object must not be used from several threads
 *   at the same time; several objects may be used from different threads.
 * - Every function that can fail returns a status; the message of the last failure of a
 *   solver object is available through hpfem_gpu_last_error().
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Bumped whenever the ABI changes. Version 2 adds hpfem_gpu_factor_info2() and the hybrid
 * memory mode, version 3 the device-resident matrices and the time stepper, version 4
 * rectangular matrices and the Arnoldi object with the Krylov basis on the device, version 5
 * the numerical refactorisation on the analysed pattern; everything of the earlier versions
 * is unchanged, so an older library still works with a loader that knows a newer version
 * (without the newer functions). */
#define HPFEM_GPU_API_VERSION 5

#if defined(_WIN32)
#if defined(HPFEM_GPU_BUILD)
#define HPFEM_GPU_API __declspec(dllexport)
#else
#define HPFEM_GPU_API __declspec(dllimport)
#endif
#else
#define HPFEM_GPU_API __attribute__((visibility("default")))
#endif

typedef struct hpfem_gpu_solver hpfem_gpu_solver; /* opaque */

typedef enum hpfem_gpu_status {
  HPFEM_GPU_OK = 0,
  HPFEM_GPU_ERR_NO_DEVICE = 1,     /* no CUDA device or no usable driver */
  HPFEM_GPU_ERR_INVALID_ARG = 2,   /* null pointer, bad size, nnz / row pointer mismatch */
  HPFEM_GPU_ERR_CUDA = 3,          /* CUDA runtime call failed */
  HPFEM_GPU_ERR_CUDSS = 4,         /* cuDSS call failed */
  HPFEM_GPU_ERR_OUT_OF_MEMORY = 5, /* device memory exhausted */
  HPFEM_GPU_ERR_SINGULAR = 6,      /* factorisation found a zero pivot */
  HPFEM_GPU_ERR_NOT_FACTORIZED = 7 /* solve() before a successful factorize() */
} hpfem_gpu_status;

typedef enum hpfem_gpu_matrix_type {
  HPFEM_GPU_MATRIX_GENERAL = 0,  /* LU, full pattern given */
  HPFEM_GPU_MATRIX_SYMMETRIC = 1 /* complex symmetric (A = A^T, not Hermitian), upper
                                    triangle given, LDL^T */
} hpfem_gpu_matrix_type;

/* ABI version compiled into the DLL (compare with HPFEM_GPU_API_VERSION). */
HPFEM_GPU_API int hpfem_gpu_api_version(void);

/* Human-readable library versions, e.g. "cuDSS 0.8.0, CUDA runtime 13.4"; static storage. */
HPFEM_GPU_API const char* hpfem_gpu_version(void);

/* Name and memory of the device the solvers run on. `name` may be null; `free_bytes` and
 * `total_bytes` may be null. Fails with HPFEM_GPU_ERR_NO_DEVICE if there is none. */
HPFEM_GPU_API hpfem_gpu_status hpfem_gpu_device_info(char* name, size_t name_len,
                                                     size_t* free_bytes, size_t* total_bytes);

HPFEM_GPU_API hpfem_gpu_status hpfem_gpu_create(hpfem_gpu_solver** out);
HPFEM_GPU_API void hpfem_gpu_destroy(hpfem_gpu_solver* solver);

/* Analysis + numerical factorisation of the n x n CSR matrix with nnz entries. The host
 * arrays are copied to the device; they may be released after the call. A previous
 * factorisation held by the object is discarded. */
HPFEM_GPU_API hpfem_gpu_status hpfem_gpu_factorize(hpfem_gpu_solver* solver, int64_t n, int64_t nnz,
                                                   const int64_t* row_ptr, const int64_t* col,
                                                   const double* values,
                                                   hpfem_gpu_matrix_type type);

/* API version 5: numerical refactorisation with new values on the pattern of the last
 * hpfem_gpu_factorize (same n, nnz, row pointer, columns and matrix type): the analysis
 * (ordering, symbolic factorisation) is reused, the scaling and equilibration are recomputed
 * from the new values. Fails if nothing was factorised or nnz differs. */
HPFEM_GPU_API hpfem_gpu_status hpfem_gpu_refactorize(hpfem_gpu_solver* solver, int64_t nnz,
                                                     const double* values);

/* Solve A X = B for nrhs right-hand sides (n x nrhs column major, interleaved complex).
 * `x` may alias `b`. */
HPFEM_GPU_API hpfem_gpu_status hpfem_gpu_solve(hpfem_gpu_solver* solver, int64_t nrhs,
                                               const double* b, double* x);

/* Size of the factors (nonzeros in L + U) and the device memory currently held by the
 * object; either pointer may be null. Zero before a factorisation. */
HPFEM_GPU_API hpfem_gpu_status hpfem_gpu_factor_info(const hpfem_gpu_solver* solver,
                                                     int64_t* nnz_factors, size_t* device_bytes);

/* Memory and mode of the current factorisation (API version 2). The device and host
 * estimates are cuDSS's peak estimates for the chosen mode, the *_bytes fields what the
 * object holds after the factorisation (device: input arrays, work vectors and the factors
 * resident on the device; host: factors kept in host memory in hybrid mode). `hybrid` is 1
 * when the factors live (partly) in host memory: chosen automatically when the device
 * estimate exceeds about 90 % of the free device memory, forced on/off by the environment
 * variable HPFEM_GPU_HYBRID=1/0. If even the hybrid estimate exceeds the free host memory,
 * factorize() fails with HPFEM_GPU_ERR_OUT_OF_MEMORY and the numbers in the message. */
typedef struct hpfem_gpu_factor_info_t {
  int64_t nnz_factors;    /* nonzeros in L + U (or L + D + L^T) */
  size_t device_bytes;    /* bytes held on the device by this object */
  size_t host_bytes;      /* bytes of factors held in host memory (hybrid mode) */
  size_t device_estimate; /* cuDSS peak device-memory estimate of the chosen mode */
  size_t host_estimate;   /* cuDSS peak host-memory estimate of the chosen mode */
  size_t device_free;     /* free device memory seen before the factorisation */
  int hybrid;             /* 1: factors in host memory (hybrid memory mode) */
} hpfem_gpu_factor_info_t;

HPFEM_GPU_API hpfem_gpu_status hpfem_gpu_factor_info2(const hpfem_gpu_solver* solver,
                                                      hpfem_gpu_factor_info_t* info);

/* Message of the last failed call on this object (empty string if none); valid until the
 * next call on the same object. */
HPFEM_GPU_API const char* hpfem_gpu_last_error(const hpfem_gpu_solver* solver);

/* Device-resident sparse matrix (API version 3): uploaded once, multiplied many times.
 * y = A x for nrhs column-major complex vectors on the host (uploaded and downloaded per
 * call). Not thread safe per object. */
typedef struct hpfem_gpu_matrix hpfem_gpu_matrix; /* opaque */

HPFEM_GPU_API hpfem_gpu_status hpfem_gpu_matrix_create(hpfem_gpu_matrix** out, int64_t n,
                                                       int64_t nnz, const int64_t* row_ptr,
                                                       const int64_t* col, const double* values);
HPFEM_GPU_API void hpfem_gpu_matrix_destroy(hpfem_gpu_matrix* matrix);
HPFEM_GPU_API hpfem_gpu_status hpfem_gpu_matrix_apply(hpfem_gpu_matrix* matrix, int64_t nrhs,
                                                      const double* x, double* y);
HPFEM_GPU_API const char* hpfem_gpu_matrix_last_error(const hpfem_gpu_matrix* matrix);
/* API version 4: a rectangular matrix (rows x cols); apply() then takes cols entries per
 * vector and returns rows entries. */
HPFEM_GPU_API hpfem_gpu_status hpfem_gpu_matrix_create_rect(hpfem_gpu_matrix** out, int64_t rows,
                                                            int64_t cols, int64_t nnz,
                                                            const int64_t* row_ptr,
                                                            const int64_t* col,
                                                            const double* values);

/* Shift-invert Arnoldi with the Krylov basis on the device (API version 4). The host keeps
 * the control flow (restarts, Hessenberg eigenproblem, Ritz selection) of
 * solvers::complex_eigenpairs_near; the object holds V (n x (ncv + 1)) and performs per
 * column j:  w = P K^{-1} (B v_j), classical Gram-Schmidt twice against v_0 .. v_j,
 * h_{j+1,j} = ||w||, v_{j+1} = w / h_{j+1,j}.  `shifted` is the factorised K = A - sigma B
 * (n unknowns), `b` the matrix B; the gauge projection P w = w - G K_g^{-1} G^H B w is
 * applied when `gradient` (n x m), `gradient_adjoint` (m x n) and the factorised `gauge`
 * (m unknowns) are given (all three or none). set_start() projects and normalises the
 * start vector into v_0; iterate(j) returns the j + 1 Hessenberg entries h_{0..j, j}
 * (interleaved complex) and beta = h_{j+1, j}; restart(m, c) sets v_0 = V_m c / ||V_m c||
 * (c: m interleaved complex); combine(m, count, C, out) returns V_m C (C: m x count,
 * column major interleaved) in out (n x count). The objects must outlive the Arnoldi
 * object; one stream, not thread safe. */
typedef struct hpfem_gpu_arnoldi hpfem_gpu_arnoldi; /* opaque */

HPFEM_GPU_API hpfem_gpu_status
hpfem_gpu_arnoldi_create(hpfem_gpu_arnoldi** out, hpfem_gpu_solver* shifted, hpfem_gpu_matrix* b,
                         hpfem_gpu_matrix* gradient, hpfem_gpu_matrix* gradient_adjoint,
                         hpfem_gpu_solver* gauge, int64_t n, int64_t ncv);
HPFEM_GPU_API void hpfem_gpu_arnoldi_destroy(hpfem_gpu_arnoldi* arnoldi);
HPFEM_GPU_API hpfem_gpu_status hpfem_gpu_arnoldi_set_start(hpfem_gpu_arnoldi* arnoldi,
                                                           const double* start);
HPFEM_GPU_API hpfem_gpu_status hpfem_gpu_arnoldi_iterate(hpfem_gpu_arnoldi* arnoldi, int64_t j,
                                                         double* h_column, double* beta);
HPFEM_GPU_API hpfem_gpu_status hpfem_gpu_arnoldi_restart(hpfem_gpu_arnoldi* arnoldi, int64_t m,
                                                         const double* coefficients);
HPFEM_GPU_API hpfem_gpu_status hpfem_gpu_arnoldi_combine(hpfem_gpu_arnoldi* arnoldi, int64_t m,
                                                         int64_t count, const double* coefficients,
                                                         double* out);
HPFEM_GPU_API const char* hpfem_gpu_arnoldi_last_error(const hpfem_gpu_arnoldi* arnoldi);

/* Newmark time stepper on the device (API version 3): the state u, v, a stays on the GPU,
 * step() is two products, the solve with the factorised Newmark operator `newmark`
 * (K = M + gamma dt C + beta dt^2 S, factorised by the caller) and the vector updates of
 *   u_pred = u + dt v + dt^2 (1/2 - beta) a,  v_pred = v + dt (1 - gamma) a,
 *   a_new = K^{-1} (load_scale * load - C v_pred - S u_pred),
 *   u = u_pred + beta dt^2 a_new,  v = v_pred + gamma dt a_new,  a = a_new.
 * `damping` may be NULL (no C), `load` may be NULL (no source; load_scale ignored). The
 * solver and the matrices must outlive the stepper and must not be used by other threads
 * meanwhile. get_state() synchronises and downloads whichever of u, v, a is non-NULL. */
typedef struct hpfem_gpu_stepper hpfem_gpu_stepper; /* opaque */

HPFEM_GPU_API hpfem_gpu_status hpfem_gpu_stepper_create(hpfem_gpu_stepper** out,
                                                        hpfem_gpu_solver* newmark,
                                                        hpfem_gpu_matrix* damping,
                                                        hpfem_gpu_matrix* stiffness, int64_t n,
                                                        const double* load, double dt, double beta,
                                                        double gamma);
HPFEM_GPU_API void hpfem_gpu_stepper_destroy(hpfem_gpu_stepper* stepper);
HPFEM_GPU_API hpfem_gpu_status hpfem_gpu_stepper_set_state(hpfem_gpu_stepper* stepper,
                                                           const double* u, const double* v,
                                                           const double* a);
HPFEM_GPU_API hpfem_gpu_status hpfem_gpu_stepper_get_state(const hpfem_gpu_stepper* stepper,
                                                           double* u, double* v, double* a);
HPFEM_GPU_API hpfem_gpu_status hpfem_gpu_stepper_step(hpfem_gpu_stepper* stepper,
                                                      double load_scale);
HPFEM_GPU_API const char* hpfem_gpu_stepper_last_error(const hpfem_gpu_stepper* stepper);

/* Function pointer types for run-time loading (dlsym / GetProcAddress). */
typedef int (*hpfem_gpu_api_version_fn)(void);
typedef const char* (*hpfem_gpu_version_fn)(void);
typedef hpfem_gpu_status (*hpfem_gpu_device_info_fn)(char*, size_t, size_t*, size_t*);
typedef hpfem_gpu_status (*hpfem_gpu_create_fn)(hpfem_gpu_solver**);
typedef void (*hpfem_gpu_destroy_fn)(hpfem_gpu_solver*);
typedef hpfem_gpu_status (*hpfem_gpu_factorize_fn)(hpfem_gpu_solver*, int64_t, int64_t,
                                                   const int64_t*, const int64_t*, const double*,
                                                   hpfem_gpu_matrix_type);
typedef hpfem_gpu_status (*hpfem_gpu_solve_fn)(hpfem_gpu_solver*, int64_t, const double*, double*);
typedef hpfem_gpu_status (*hpfem_gpu_factor_info_fn)(const hpfem_gpu_solver*, int64_t*, size_t*);
typedef const char* (*hpfem_gpu_last_error_fn)(const hpfem_gpu_solver*);
typedef hpfem_gpu_status (*hpfem_gpu_factor_info2_fn)(const hpfem_gpu_solver*,
                                                      hpfem_gpu_factor_info_t*);
typedef hpfem_gpu_status (*hpfem_gpu_refactorize_fn)(hpfem_gpu_solver*, int64_t, const double*);
typedef hpfem_gpu_status (*hpfem_gpu_matrix_create_fn)(hpfem_gpu_matrix**, int64_t, int64_t,
                                                       const int64_t*, const int64_t*,
                                                       const double*);
typedef void (*hpfem_gpu_matrix_destroy_fn)(hpfem_gpu_matrix*);
typedef hpfem_gpu_status (*hpfem_gpu_matrix_apply_fn)(hpfem_gpu_matrix*, int64_t, const double*,
                                                      double*);
typedef const char* (*hpfem_gpu_matrix_last_error_fn)(const hpfem_gpu_matrix*);
typedef hpfem_gpu_status (*hpfem_gpu_stepper_create_fn)(hpfem_gpu_stepper**, hpfem_gpu_solver*,
                                                        hpfem_gpu_matrix*, hpfem_gpu_matrix*,
                                                        int64_t, const double*, double, double,
                                                        double);
typedef void (*hpfem_gpu_stepper_destroy_fn)(hpfem_gpu_stepper*);
typedef hpfem_gpu_status (*hpfem_gpu_stepper_set_state_fn)(hpfem_gpu_stepper*, const double*,
                                                           const double*, const double*);
typedef hpfem_gpu_status (*hpfem_gpu_stepper_get_state_fn)(const hpfem_gpu_stepper*, double*,
                                                           double*, double*);
typedef hpfem_gpu_status (*hpfem_gpu_stepper_step_fn)(hpfem_gpu_stepper*, double);
typedef const char* (*hpfem_gpu_stepper_last_error_fn)(const hpfem_gpu_stepper*);
typedef hpfem_gpu_status (*hpfem_gpu_matrix_create_rect_fn)(hpfem_gpu_matrix**, int64_t, int64_t,
                                                            int64_t, const int64_t*, const int64_t*,
                                                            const double*);
typedef hpfem_gpu_status (*hpfem_gpu_arnoldi_create_fn)(hpfem_gpu_arnoldi**, hpfem_gpu_solver*,
                                                        hpfem_gpu_matrix*, hpfem_gpu_matrix*,
                                                        hpfem_gpu_matrix*, hpfem_gpu_solver*,
                                                        int64_t, int64_t);
typedef void (*hpfem_gpu_arnoldi_destroy_fn)(hpfem_gpu_arnoldi*);
typedef hpfem_gpu_status (*hpfem_gpu_arnoldi_set_start_fn)(hpfem_gpu_arnoldi*, const double*);
typedef hpfem_gpu_status (*hpfem_gpu_arnoldi_iterate_fn)(hpfem_gpu_arnoldi*, int64_t, double*,
                                                         double*);
typedef hpfem_gpu_status (*hpfem_gpu_arnoldi_restart_fn)(hpfem_gpu_arnoldi*, int64_t,
                                                         const double*);
typedef hpfem_gpu_status (*hpfem_gpu_arnoldi_combine_fn)(hpfem_gpu_arnoldi*, int64_t, int64_t,
                                                         const double*, double*);
typedef const char* (*hpfem_gpu_arnoldi_last_error_fn)(const hpfem_gpu_arnoldi*);

#ifdef __cplusplus
}
#endif

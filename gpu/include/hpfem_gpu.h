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

/* Bumped whenever the ABI changes; the loader refuses a DLL with another version. */
#define HPFEM_GPU_API_VERSION 1

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

/* Solve A X = B for nrhs right-hand sides (n x nrhs column major, interleaved complex).
 * `x` may alias `b`. */
HPFEM_GPU_API hpfem_gpu_status hpfem_gpu_solve(hpfem_gpu_solver* solver, int64_t nrhs,
                                               const double* b, double* x);

/* Size of the factors (nonzeros in L + U) and the device memory currently held by the
 * object; either pointer may be null. Zero before a factorisation. */
HPFEM_GPU_API hpfem_gpu_status hpfem_gpu_factor_info(const hpfem_gpu_solver* solver,
                                                     int64_t* nnz_factors, size_t* device_bytes);

/* Message of the last failed call on this object (empty string if none); valid until the
 * next call on the same object. */
HPFEM_GPU_API const char* hpfem_gpu_last_error(const hpfem_gpu_solver* solver);

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

#ifdef __cplusplus
}
#endif

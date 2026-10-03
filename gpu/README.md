# hpfem_gpu — GPU direct solver (cuDSS)

The GPU backend of `solvers::LinearSolver` (`DirectSolverBackend::kCudss`) lives in a
separate shared library, `hpfem_gpu.dll` / `hpfem_gpu.so`, with the C interface in
`include/hpfem_gpu.h`. The hp-FEM library itself is built with MinGW GCC (Windows) or
GCC / Clang (Linux), which cannot link objects produced by nvcc and MSVC; the DLL is therefore
built on its own with the CUDA toolchain and loaded at run time. Nothing of CUDA or cuDSS is
needed to build the library (`HPFEM_ENABLE_CUDA` only compiles the loader), and a missing DLL
or GPU degrades gracefully: `available(kCudss)` is false and `kAuto` picks MUMPS or SparseLU.
Rationale and measurements: `docs/adr/0008-gpu-backend.md`.

What the DLL does: analysis + LU (or LDLᵀ for complex-symmetric input) factorisation of a
complex double CSR matrix on the device, then solves for one or many right-hand sides
(factorise once, solve many). Indices are 64-bit end to end (`CUDSS_R_64I`), exactly the
storage of `hpfem::SparseMatrix`, so no host-side conversion is needed. The matrix is
factorised as `s·A` with `s = 1 / max|a_ij|` (cuDSS judges tiny pivots by an absolute
threshold, which SI-scaled systems with entries around 1e-15 would trip wholesale) and every
solution is rescaled; zero or tiny pivots that cuDSS would still perturb are reported as a
singular matrix instead of silently producing a wrong solution.

## Requirements

| Component | Version used | Notes |
|-----------|--------------|-------|
| NVIDIA driver | ≥ 591 (CUDA 13.1) | the driver's CUDA version must cover the toolkit used |
| CUDA toolkit | 13.4 (≥ 12.0 should work) | `nvcc`, `cudart` (linked statically) |
| cuDSS | 0.8.0 (`lib/13`) | <https://developer.nvidia.com/cudss>; run-time DLLs `cudss64_0.dll`, `cudss_mtlayer_*.dll` |
| Host compiler | MSVC 19.44 (VS 2022 Build Tools) on Windows, GCC on Linux | MSVC runtime linked statically |
| GPU | compute capability 8.6 (RTX 3090) by default | `-DCMAKE_CUDA_ARCHITECTURES=…` for others |

## Build (Windows, PowerShell or Git Bash)

```bash
cmake -S gpu -B build/gpu -G "Visual Studio 17 2022" -A x64 \
      -Dcudss_DIR="C:/Program Files/NVIDIA cuDSS/v0.8/lib/13/cmake/cudss"
cmake --build build/gpu --config Release
ctest --test-dir build/gpu -C Release --output-on-failure     # self-test on the GPU
```

Linux:

```bash
cmake -S gpu -B build/gpu -DCMAKE_BUILD_TYPE=Release -Dcudss_DIR=<cudss>/lib/cmake/cudss
cmake --build build/gpu && ctest --test-dir build/gpu
```

The DLL ends up in `build/gpu/bin/Release/hpfem_gpu.dll` (Linux: `build/gpu/bin/hpfem_gpu.so`).
Its only run-time dependencies are the cuDSS DLLs and the system libraries; the cuDSS `bin`
directory must be on `PATH` (the self-test gets it prepended by CTest).

## Using it from the library

Configure the library with `-DHPFEM_ENABLE_CUDA=ON`. At run time the loader looks for the
DLL in this order:

1. the environment variable `HPFEM_GPU_DLL` (full path),
2. the path compiled in through the CMake cache variable `HPFEM_GPU_DLL`,
3. `hpfem_gpu.dll` / `hpfem_gpu.so` next to the executable,
4. the plain name on the system loader path.

`solvers::available(DirectSolverBackend::kCudss)` tells whether the DLL could be loaded and a
device is present; `solvers::make_cudss()` returns the solver. Python: `DirectSolverBackend.CUDSS`.

## Self-test

`hpfem_gpu_selftest [n]` runs without the library: a 1D Helmholtz system (general and
complex-symmetric input), a random sparse system, one and several right-hand sides, in-place
solves, and the error paths (inconsistent CSR, singular matrix, solve before factorisation,
reuse after a failure). All solutions must agree with the reference to 1e-10.

## Interface summary (`include/hpfem_gpu.h`, API version 1)

| Function | Purpose |
|----------|---------|
| `hpfem_gpu_api_version()` | ABI check by the loader |
| `hpfem_gpu_version()` | cuDSS / CUDA runtime versions |
| `hpfem_gpu_device_info(name, len, free, total)` | device name and memory; `ERR_NO_DEVICE` without GPU |
| `hpfem_gpu_create(&solver)` / `hpfem_gpu_destroy(solver)` | solver object with its own cuDSS handle and stream |
| `hpfem_gpu_factorize(solver, n, nnz, row_ptr, col, values, type)` | CSR (int64, interleaved complex) → device, analysis + factorisation |
| `hpfem_gpu_solve(solver, nrhs, b, x)` | n × nrhs column-major solve; `x` may alias `b` |
| `hpfem_gpu_factor_info(solver, &nnz_factors, &device_bytes)` | size of the factors, device memory held |
| `hpfem_gpu_last_error(solver)` | message of the last failure on this object |

Statuses: `OK`, `ERR_NO_DEVICE`, `ERR_INVALID_ARG`, `ERR_CUDA`, `ERR_CUDSS`,
`ERR_OUT_OF_MEMORY`, `ERR_SINGULAR`, `ERR_NOT_FACTORIZED`. A solver object is not thread
safe; use one object per thread.

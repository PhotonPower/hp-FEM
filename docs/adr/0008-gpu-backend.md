# 0008 — GPU backend: cuDSS direct solver behind `LinearSolver`, no GPU assembly
**Status:** accepted · **Date:** 2026-10-03 · amends 0007

## Context
The backlog asked for "GPU assembly". The maintainer's machine has an NVIDIA GeForce RTX
3090 (24 GB, compute capability 8.6, CUDA 13.4, cuDSS 0.8); the library is built with
MinGW GCC (CLAUDE.md §4a) and CI runs on Ubuntu without a GPU. Three facts shape the
decision:

1. **Toolchain.** On Windows nvcc only accepts MSVC as host compiler, and MSVC and MinGW
   objects cannot be linked into one C++ library (different ABI and runtimes). GPU code
   can therefore not be a `.cu` file inside `src/`.
2. **Arithmetic.** Everything in hp-FEM is `complex<double>` (principle 4) and *hp*
   convergence to 1e-10 does not tolerate single precision. The RTX 3090 is a consumer
   card with about 0.5 TFLOPS in double precision, the same order as the 24-thread CPU
   with AVX2, so compute-bound element assembly cannot gain much. GPUs win on
   bandwidth-bound work: 936 GB/s against about 60 GB/s.
3. **Measurement** (`benchmarks/gpu/`, results in `benchmarks/results/2026-10-03-VR-gpu.json`,
   complex double, 24 OpenMP threads, MUMPS 5.9 sequential, Eigen SparseLU, cuSPARSE
   generic SpMV, cuDSS 0.8 LU with `CUDSS_MTYPE_GENERAL`):

   | matrix | unknowns / nnz | SpMV CPU → GPU | factorise SparseLU / MUMPS / cuDSS | one solve SparseLU / MUMPS / cuDSS | 8 rhs SparseLU / cuDSS |
   |---|---|---|---|---|---|
   | Newmark operator, square, p = 2 | 163 k / 1.9 M | 0.30 ms → 0.13 ms (2.3×) | 5.0 s / 1.8 s / 0.78 s | 59 ms / 74 ms / 2.6 ms | 0.43 s / 0.012 s |
   | scattering S − k₀²M with PML, cube, p = 2 | 70 k / 2.9 M | 0.54 ms → 0.18 ms (2.9×) | 591 s / 6.8 s / 3.8 s | 0.78 s / 0.84 s / 5.9 ms | 6.2 s / 0.030 s |
   | Newmark operator, square, p = 3 | 1.37 M / 27.6 M | 21.7 ms → 1.6 ms (14×) | 334 s / 12.3 s / 3.6 s | 2.3 s / 0.48 s / 13 ms | 13.8 s / 0.072 s |

   Residuals are 1e-13 … 1e-15 for all backends. Against MUMPS the GPU factorisation is
   1.8–3.4× faster, a repeated solve 30–140× faster; SpMV gains 2–3× while the matrix fits the
   CPU caches and 14× beyond (1.37 M unknowns). The cuDSS factorisation times are those of
   the first factorisation in a process (cold, including the cuDSS initialisation); a second
   factorisation in the same process is faster for the two smaller matrices (0.42 s / 1.7 s,
   `factorize_gpu_s` of the re-measured lines in the JSON; 3.0 s unchanged for the largest).
   cuDSS judges tiny pivots by an *absolute* threshold: the benchmark therefore factorises
   `s·A` with `s = 1/max|a_ij|` and rescales the solutions, exactly as the `hpfem_gpu`
   library does, and records `CUDSS_DATA_NPIVOTS` (0 for all three matrices, scaled and
   unscaled; the free-DoF Newmark and mass operators of `physics::TimeDomain`, whose
   largest entries are ~1e-15, had 32770 pivots perturbed unscaled and are refused by the
   library without the scaling).

## Decision
- The GPU work goes into a **direct-solver backend on cuDSS** behind `solvers::LinearSolver`
  (`DirectSolverBackend::kCudss`, opt-in; `kAuto` keeps preferring MUMPS until a later
  decision on a size threshold). The targets are the workloads that factorise once and
  solve many times: Newmark time stepping, shift-invert Arnoldi, parameter sweeps and
  reduced bases, goal-oriented dual solves.
- The backend is a **separate DLL with a C ABI** (`gpu/`: own CMake project built with
  nvcc + MSVC, header `hpfem_gpu.h` without CUDA types, opaque handle, `int64` indices,
  interleaved double pairs for complex values, status codes and a per-handle error string,
  an API version check). The MinGW library loads it at run time (`LoadLibrary` / `dlopen`)
  behind `HPFEM_ENABLE_CUDA` and needs no CUDA at build time; without the DLL or a GPU
  `available(kCudss)` is false and `make_direct_solver(kCudss)` throws with the searched
  paths. CI can build with the option on and tests the failure path.
- `LinearSolver` gains `solve_many(const Matrix&)` for several right-hand sides (default
  column loop; native in SparseLU, MUMPS and cuDSS). This amends ADR-0007.
- **No GPU assembly.** Element assembly stays on the CPU (OpenMP); the backlog item is
  reworded accordingly. It can be revisited for a data-centre GPU with full-rate FP64.

## Consequences
- One more optional, platform-specific build (`gpu/`), documented in `gpu/README.md`; the
  DLL depends only on `cudss64_0.dll`. Linux is covered by the same C ABI (`.so`), untested
  for now.
- The transient solver, the eigensolvers and the sweeps profit without code changes once
  they pass `kCudss` (and use `solve_many` where they have several right-hand sides).
- cuDSS pivots statically with an absolute tiny-pivot threshold; the DLL factorises the
  matrix scaled to `max|a_ij| = 1`, rescales every solution, and checks
  `CUDSS_DATA_NPIVOTS` to report singular matrices instead of returning a wrong solution.
- `benchmarks/solver_integration.cpp` records the end-to-end effect
  (`benchmarks/results/2026-10-03-VR-gpu-integration.json`, summarised in
  docs/theory/solvers.md): time step 3.8× faster than MUMPS, batched 8-rhs solve 5.7×,
  resonance 1.9×; the real gauged Lanczos gains nothing from the complexified GPU
  factorisation at 92 k unknowns.
- `solvers::Symmetry::kComplexSymmetric` lets cuDSS (upper triangle, `CUDSS_MTYPE_SYMMETRIC`)
  and MUMPS (`SYM = 2`) factorise the complex-symmetric curl–curl systems as LDLᵀ; the caller
  guarantees the structure, SparseLU ignores the flag.
- Double precision stays the only precision; no mixed-precision shortcuts.
- `kAuto` prefers cuDSS from `HPFEM_GPU_MIN_UNKNOWNS` unknowns on (default 10 000, decided in
  `factorize`; the GPU library is loaded only then). The value is where the GPU
  factorisation draws level with sequential MUMPS on the RTX 3090 in 2D and 3D
  (`benchmarks/results/2026-10-03-VR-backend-threshold.json`); below it every factorisation
  takes less than 40 ms. Backends agree only to about 1e-12 relative, so no test may rely on
  bitwise equality between runs. Where cuDSS refuses a system (perturbed pivots, which the
  *hp*-adaptive systems with hanging nodes trigger even after the scaling), `kAuto` warns and
  falls back to MUMPS / SparseLU; accepting perturbed factorisations with iterative refinement
  lost accuracy at 2 612 perturbed pivots and is not done.

## Alternatives considered
- **GPU assembly (CUDA kernels for the element matrices)**: rejected for this hardware
  (FP64 rate) and because it would require the MSVC toolchain inside the library build.
- **Switching the Windows build to MSVC** so that `.cu` files could live in `src/`: rejected,
  the whole dependency stack (MUMPS, OpenBLAS, pybind11 module) is set up for MinGW and CI.
- **OpenCL or Vulkan compute** (no second host compiler): rejected, no vendor direct
  solver comparable to cuDSS and poor complex-double support.
- **cuSOLVER / cuSPARSE triangular solves on a CPU factorisation**: rejected, the
  factorisation itself is the larger part for 3D problems.

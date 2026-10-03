# Benchmarks
Not run in CI (`cmake --preset release -DHPFEM_BUILD_BENCHMARKS=ON`, targets `bench_*`).
Each benchmark prints one JSON line per run (host, library version, problem size, DoFs,
nonzeros, threads, condensation, solver backend, assembly / factorisation / solve
seconds) and appends them to the file given as its first argument. Results are kept in
`benchmarks/results/<date>-<host>.json` together with a note on the machine.

| Benchmark | What it measures |
|---|---|
| `bench_assembly_solve` | Maxwell plane wave on the unit square, (n, p) ∈ {(64, 2), (64, 4), (128, 3)}: assembly with 1 and all threads, with and without static condensation, factorisation and solve with every available direct solver |
| `gpu/spmv_solve_bench` | GPU micro-benchmark behind ADR-0008: complex double SpMV (OpenMP CSR vs cuSPARSE) and direct solves (Eigen SparseLU vs cuDSS LU, one and eight right-hand sides) on matrices exported by `gpu/export_matrices.py` (Newmark operator of the transient solver, 3D scattering operator with PML). Built on its own with nvcc + MSVC, see `gpu/CMakeLists.txt`; MUMPS timings of the same matrices via `python/` (`hpfem.make_direct_solver`) |

Results so far: `results/2026-10-02-VR.json` — Windows 11, MSYS2 GCC 16.1 (UCRT64),
RelWithDebInfo, 24 cores, MUMPS 5.9 sequential with OpenBLAS; summarised in
`docs/theory/solvers.md#measured`.

`results/2026-10-03-VR-gpu.json` — the GPU benchmark on the same machine with an NVIDIA GeForce
RTX 3090 (CUDA 13.4, cuDSS 0.8); summarised in `docs/adr/0008-gpu-backend.md`.

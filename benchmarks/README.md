# Benchmarks
Not run in CI (`cmake --preset release -DHPFEM_BUILD_BENCHMARKS=ON`, targets `bench_*`).
Each benchmark prints one JSON line per run (host, library version, problem size, DoFs,
nonzeros, threads, condensation, solver backend, assembly / factorisation / solve
seconds) and appends them to the file given as its first argument. Results are kept in
`benchmarks/results/<date>-<host>.json` together with a note on the machine.

| Benchmark | What it measures |
|---|---|
| `bench_assembly_solve` | Maxwell plane wave on the unit square, (n, p) ∈ {(64, 2), (64, 4), (128, 3)}: assembly with 1 and all threads, with and without static condensation, factorisation and solve with every available direct solver |
| `bench_solver_integration` | Where one factorisation serves many solves, with every available direct solver (SparseLU, MUMPS, cuDSS): the transient PEC cavity (`physics::TimeDomain`, n = 128, p = 2: setup with two factorisations, seconds per Newmark step), an 8-angle sweep of `physics::ScatteringOperator` (n = 96, p = 3: one-by-one `solve` against batched `solve_many`, with and without load assembly), `physics::Resonance` (complex shift-invert Arnoldi, n = 96, p = 2) and the real gauged Lanczos `gauged_curl_curl_eigenpairs` with the real SparseLU against the complexified MUMPS / cuDSS factorisation |
| `bench_backend_threshold` | Factorisation and median solve time of every available backend over the problem size (Maxwell operator of a plane wave, square p = 2 with n = 8 … 128 and cube p = 2 with n = 3 … 12), the basis of the `kAuto` threshold `HPFEM_GPU_MIN_UNKNOWNS` |
| `bench_device_arnoldi` | Complex shift-invert Arnoldi with the Krylov basis on the host against the basis on the device (`solvers::DeviceArnoldi`): `physics::Resonance` of the closed PEC square (ungauged) and `physics::BandStructure` of the empty square lattice (gauged), p = 2, 6 eigenvalues, 24 Krylov vectors, every backend; arguments `[results.json] [n]` (default n = 96, 92 k DoFs) |
| `bench_sweep_shares` | Shares of a parameter sweep with every backend: the angle sweep of `ScatteringOperator::solve_many` (unit square, p = 3, 100 incident fields) split into factorisation, loads (Scattering setup, `assemble_maxwell_load`, Dirichlet data), the batched solve and, on cuDSS, the pure host-device transfer of the same volume; and a frequency sweep with `ReducedBasis` (8 snapshots, 100 reduced solves) split into snapshots, projections, loads, reduced solves and lifts; arguments `[results.json] [n] [p] [nrhs] [nfreq]` |
| `bench_hybrid_memory` | 3D Maxwell operator of a plane wave on the unit cube at growing size (n = 12, 16, 20, p = 2 by default; arguments `[results.json] [max_n] [p]`) with every backend: factor size, memory, mode (cuDSS device or hybrid) and time, the basis of the hybrid-memory statement in ADR-0008; runs for a long time |
| `gpu/spmv_solve_bench` | GPU micro-benchmark behind ADR-0008: complex double SpMV (OpenMP CSR vs cuSPARSE) and direct solves (Eigen SparseLU vs cuDSS LU, one and eight right-hand sides) on matrices exported by `gpu/export_matrices.py` (Newmark operator of the transient solver, 3D scattering operator with PML). Built on its own with nvcc + MSVC, see `gpu/CMakeLists.txt`; MUMPS timings of the same matrices via `python/` (`hpfem.make_direct_solver`) |

Results so far: `results/2026-10-02-VR.json` — Windows 11, MSYS2 GCC 16.1 (UCRT64),
RelWithDebInfo, 24 cores, MUMPS 5.9 sequential with OpenBLAS; summarised in
`docs/theory/solvers.md#measured`.

`results/2026-10-03-VR-gpu.json` — the GPU benchmark on the same machine with an NVIDIA GeForce
RTX 3090 (CUDA 13.4, cuDSS 0.8); summarised in `docs/adr/0008-gpu-backend.md`. The lines with
`pivots_gpu` are the re-measurement with the matrix scaled to max |a_ij| = 1 (as the
`hpfem_gpu` library does) and the perturbed-pivot count; `factorize_gpu_unscaled_s` is the
cold first factorisation of the process, `factorize_gpu_s` the warm second one.

`results/2026-10-03-VR-gpu-integration.json` — `bench_solver_integration` on the same
machine (release build with MUMPS and cuDSS); summarised in `docs/theory/solvers.md`.

`results/2026-10-03-VR-backend-threshold.json` — `bench_backend_threshold` on the same
machine; the cuDSS factorisation draws level with sequential MUMPS at about 10 000 unknowns
(`HPFEM_GPU_MIN_UNKNOWNS`), see `docs/theory/solvers.md`.

`results/2026-10-03-VR-hybrid-memory.json` — `bench_hybrid_memory` on the same machine
(3D Maxwell p = 2, n = 12 … 32, MUMPS against cuDSS with the automatic hybrid memory mode; the
last line is the forced hybrid mode at n = 20); summarised in `docs/theory/solvers.md`.

`results/2026-10-03-VR-backend-symmetry.json` — the same benchmark with the general and the
complex-symmetric (LDLᵀ) factorisation of MUMPS and cuDSS (`solvers::Symmetry`); the
`cube n = 6` general MUMPS line (4.7 s against 0.23 s in the threshold run) was disturbed by
other sessions on the machine.

`results/2026-10-04-VR-device-arnoldi.json` — `bench_device_arnoldi` on the same machine
(RTX 3090, cuDSS 0.8) for n = 96 and n = 192 (92 k and 369 k DoFs): the device basis saves
20–35 % against the host basis with cuDSS solves, SparseLU is 3–16× slower; the numbers are
in docs/theory/solvers.md.

`results/2026-10-04-VR-sweep-shares.json` — `bench_sweep_shares` on the same machine at
194 k DoFs: on cuDSS the 100-angle sweep takes 3.2 s, of which the loads are 2.6 s (26 ms
per incident field, 20 ms of it the load assembly) and the batched solve 0.3 s, where the
transfers alone are 0.21 s; the frequency sweep is bounded by the 8 snapshot factorisations
(5.7 s) against 0.07 s of projections and 0.14 s of reduced solves. Device-resident sweep
vectors would therefore save at most 6 %; the lever is the per-field load assembly.

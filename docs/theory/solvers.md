# Linear solvers

## Direct solvers (`solvers/linear_solver.hpp`)

All assembled systems are complex, sparse and (for the curl–curl problem) complex
*symmetric* but not Hermitian, so a general LU factorisation is used. `LinearSolver`
factorises once and solves for any number of right-hand sides (parameter sweeps, the
adjoint of the goal-oriented estimator, several incident fields on one mesh).

| Backend (`DirectSolverBackend`) | Library | When |
|---|---|---|
| `kSparseLu` | Eigen `SparseLU`, COLAMD ordering | always available; 2D up to a few $10^5$ unknowns |
| `kMumps` | MUMPS (multifrontal, `zmumps`, sequential build) | `HPFEM_ENABLE_MUMPS`; 3D and large 2D systems, many right-hand sides |
| `kAuto` | MUMPS if compiled in, otherwise SparseLU | the default of `solve_direct` and `ScatteringSetup::solver` |

`make_direct_solver(backend)` returns the solver, `available(backend)` /
`available_backends()` tell what this build offers, `solve_direct(A, b, backend)` does
both steps. `physics::ScatteringSetup::solver` selects the backend of a scattering problem
(and of its DWR adjoint).

### MUMPS backend

`src/solvers/mumps_solver.cpp` wraps `zmumps_c`: the matrix is passed as centralised
assembled COO input (`ICNTL(5) = 0`, `ICNTL(18) = 0`, 1-based indices), analysis and
numerical factorisation run in one call (`job = 4`) with the automatic ordering choice
(`ICNTL(7) = 7`, METIS/SCOTCH/PORD as available in the library) and 30 % memory
relaxation, solves run in place (`job = 3`), all library output is switched off and errors
are reported from `INFOG(1)` / `INFOG(2)` as `hpfem::Error`. The sequential library ships
its own MPI stub, whose `MPI_Init` is called once.

Build support (`cmake/FindMUMPS.cmake`, option `HPFEM_ENABLE_MUMPS`, preset `mumps`):

- Ubuntu / Debian: `apt-get install libmumps-seq-dev libmumps-headers-dev`
  (`libzmumps_seq`, `libmumps_common_seq`, `libpord_seq`, `libmpiseq_seq`, headers in
  `/usr/include`). The CI runs the whole test suite with this build.
- MSYS2 (Windows): `pacman -S mingw-w64-ucrt-x86_64-mumps` (module `mumps-zso`,
  sequential complex double, with OpenBLAS, METIS and SCOTCH). The DLLs live in
  `C:\msys64\ucrt64\bin`; the test targets get that directory prepended to `PATH`
  (`MUMPS_BIN_DIR`), programs started by hand need it on `PATH`.

### Verification

`tests/unit/solvers/test_linear_solver.cpp` solves a random sparse complex system with
every available backend to $10^{-10}$, reuses the factorisation for a second right-hand side,
checks the one-shot interface, the error reporting (singular, non-square, wrong size,
solve before factorisation) and that `kAuto` picks MUMPS when compiled in. With the `mumps`
preset the complete convergence suite runs on MUMPS (all tolerances unchanged).

## Eigenvalue solvers

See [maxwell.md](maxwell.md#eigenproblems): Spectra's shift-invert Lanczos / Arnoldi on top
of the SparseLU factorisation of $S - \sigma M$ (gauged curl–curl eigenproblems and the
Lee–Sun–Cendes pencil of `PropagatingMode`). Switching the inner factorisation to the
direct solver backends is planned together with static condensation (M6).

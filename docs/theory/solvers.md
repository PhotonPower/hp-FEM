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
| `kCudss` | NVIDIA cuDSS (LU on the GPU, factors stay on the device) | `HPFEM_ENABLE_CUDA` + the `hpfem_gpu` library; opt-in, for many solves of one factorisation (time stepping, Arnoldi, sweeps) |
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

### cuDSS backend (GPU)

The library is built with MinGW GCC (Windows) or GCC / Clang (Linux), which cannot link
objects produced by nvcc and MSVC. The GPU solver therefore lives in the separately built
shared library `hpfem_gpu` (`gpu/`, see `gpu/README.md`) with a pure C interface
(`gpu/include/hpfem_gpu.h`: opaque solver handle, status codes, last-error message, CSR with
64-bit indices and interleaved complex values in, solutions out, factorise once / solve many
right-hand sides, re-factorisation on the same handle). `src/solvers/cudss_solver.cpp` loads
it at run time (`LoadLibrary` / `dlopen`), checks the API version and the presence of a CUDA
device, and wraps it as a `LinearSolver`. Nothing of CUDA is needed to build the library;
without the DLL or a GPU, `available(kCudss)` is false, `make_cudss()` throws an error that
names the search paths, and `kAuto` is unaffected, so CI (no GPU) builds and tests the option
as well. cuDSS is opt-in in this version (`DirectSolverBackend::kCudss`); whether `kAuto`
should prefer it above a size threshold is decided from the ADR-0008 measurements.

Numerics: cuDSS factorises the general complex matrix (`CUDSS_MTYPE_GENERAL`; the interface
also offers the complex-symmetric LDL^T path for later use). `hpfem::SparseMatrix` is already
CSR with `int64` indices, so the matrix is uploaded without conversion. cuDSS perturbs zero or
tiny pivots instead of failing; the DLL reads the perturbation count after the factorisation
and reports the matrix as singular (`hpfem::Error`) rather than returning a wrong solution.

Finding the DLL at run time, in this order: environment variable `HPFEM_GPU_DLL`, the path
compiled in from the CMake cache variable `HPFEM_GPU_DLL`, `hpfem_gpu.dll` next to the
executable, the plain name on the loader path. The cuDSS runtime DLLs must be on `PATH`
(`HPFEM_GPU_BIN_DIR` prepends them for the tests, like `MUMPS_BIN_DIR`). `cudss_status()`
tells why the backend is or is not usable. Rationale and measurements: ADR-0008. In short
(RTX 3090, complex double, 24 CPU threads): the GPU factorisation is 1.8–2.3× faster than
sequential MUMPS on a 163k-unknown Newmark operator and a 70k-unknown 3D PML scattering
system, and a repeated solve of one factorisation is 30–140× faster (milliseconds instead of
tenths of a second), which is where time stepping, Arnoldi iterations and sweeps spend their
time; residuals are $10^{-13}$ to $10^{-15}$ on both.

### Verification

`tests/unit/solvers/test_linear_solver.cpp` solves a random sparse complex system with
every available backend to $10^{-10}$, reuses the factorisation for a second right-hand side,
checks the one-shot interface, the error reporting (singular, non-square, wrong size,
solve before factorisation), that `solve_many` equals column-wise `solve` and that
`kAuto` picks MUMPS when compiled in; cuDSS joins the loop whenever its library loads and a
GPU is present. With the `mumps` preset the complete convergence suite runs on MUMPS (all
tolerances unchanged). The DLL itself has a stand-alone self-test (`gpu/selftest`, run by hand
on a GPU machine): Helmholtz and random systems, general and complex-symmetric input, several
right-hand sides, in-place solves, re-factorisation, error paths.

## Static condensation (`assembly/condensation.hpp`)

The interior (cell-bubble) functions of the hierarchical bases have support in one cell
only, so their DoFs couple to nothing but the cell's own exterior DoFs. With the local
system split into exterior (E) and interior (B) blocks, `StaticCondensation` assembles the
Schur complement

$$
\tilde K_{EE} = K_{EE} - K_{EB}K_{BB}^{-1}K_{BE}, \qquad \tilde f_E = f_E - K_{EB}K_{BB}^{-1} f_B
$$

cell by cell (a dense LU of $K_{BB}$ per cell, $K_{BB}^{-1}K_{BE}$ and $K_{BB}^{-1}f_B$ are
kept) and recovers $u_B = K_{BB}^{-1}(f_B - K_{BE}u_E)$ after the solve. The global
numbering is unchanged: the interior rows of the condensed matrix are identity rows with
zero load, so Dirichlet elimination and the hanging-node / Bloch constraints — which never
involve interior DoFs — apply exactly as before, and `recover` completes the solution.
`assemble_maxwell_operator` ($A = S - k^2 M$ with the load in one pass) and the
per-cell-form `assemble_h1` take an optional `StaticCondensation`; `physics::Scattering`
condenses by default (`ScatteringSetup::condense`). Not condensed: the eigenproblems (the
pencil $S - \lambda M$ is not linear in the unknown block) and the DWR adjoint.

Interior DoFs per cell: H1 $(p-1)(p-2)/2$ (2D), Nédélec $p(p-1)$ (2D), i.e. for $p = 4$
a third of the Nédélec DoFs of a triangle and for $p = 6$ half of them. The direct solver
then factorises a matrix whose coupled part lives on the entity DoFs only.

Verification (`tests/unit/assembly/test_condensation.cpp`): H1 (2D $p = 3$–$5$, 3D $p = 4$)
and Maxwell (2D $p = 2$–$4$, 3D $p = 3$) solutions with Dirichlet / PEC data agree with the
full systems to $10^{-10}$, the condensed matrices have fewer nonzeros and identity interior
rows, a `Scattering` solve on a hanging mesh with PEC and prescribed traces agrees with the
uncondensed solve, and singular interior blocks are reported.

## Parallel assembly (`core/parallel.hpp`)

With OpenMP (`HPFEM_ENABLE_OPENMP`, on by default when the compiler supports it) the cell
loops of `assemble_maxwell`, `assemble_maxwell_operator` and `assemble_h1` and the cell and
facet loops of the residual estimator and the weighted residual run in parallel
(`parallel_for`, dynamic scheduling). Every thread owns its triplet buffer, right-hand
side and quadrature-rule cache; the buffers are concatenated afterwards and summed by the
sparse assembler, the facet contributions are stored per facet and accumulated into the
cells serially, and `StaticCondensation` guards its recovery table with a mutex. The results
are bit-for-bit reproducible up to the summation order of duplicate entries
(`tests/unit/assembly/test_parallel.cpp` compares 1 and all threads to $10^{-12}$). The
per-cell form factories must be reentrant (the ones of `physics::Scattering` are).
`hpfem::num_threads()` / `set_num_threads()` control the thread count; on MinGW the OpenMP
runtime is linked statically like the rest of the GCC runtime.

## Parameter sweeps (`physics/sweep.hpp`, `solvers/reduced_basis.hpp`)

**Angle sweeps, one factorisation.** At a fixed frequency the operator of a scattering
problem does not depend on the incident field or the current. `physics::ScatteringOperator`
assembles the operator once (static condensation, hanging / Bloch constraints, Dirichlet
elimination of the problem's PEC and incident facets with `assembly::DirichletElimination`,
which keeps the eliminated columns $A_{:,D}$ for later loads) and factorises it with the
chosen backend; `solve(incident, current)` then only assembles the new load
(`assemble_maxwell_load`), condenses it (`StaticCondensation::condense_load`, with the stored
$K_{BB}^{-1}$ and $K_{EB}$ per cell), reduces it by the constraints, applies the new Dirichlet
values and recovers the interior unknowns (`recover(x, load)`). `solve_many` and
`plane_wave_sweep` wrap this for lists of incident fields / wave vectors — the angle sweep
of a scatterometry measurement costs one factorisation plus a triangular solve per angle.

**Frequency sweeps, reduced basis.** `solvers::ReducedBasis` collects snapshots
(full solutions at a few parameter values) into an orthonormal basis $V$ (modified
Gram–Schmidt with re-orthogonalisation, dependent snapshots dropped) and provides the
Galerkin projections $V^H A V$, $V^H b$ and the lift $u \approx V y$. For an affine operator
such as $A(k) = S - k^2 M$ (no dispersive materials, PML frozen at a reference
wavenumber) the projections of $S$ and $M$ are formed once and every further frequency is
a dense solve of the size of the basis; `DirichletElimination` with `unit_diagonal = false`
on $M$ keeps the eliminated system affine. This is the hook for the reduced-basis and
dispersion work of later milestones; frequency-dependent PML and materials still need a
full re-assembly per frequency.

Verification (`tests/unit/physics/test_sweep.cpp`): solutions of three incidence angles by
one factorisation agree with individual solves to $10^{-9}$ for the total- and
scattered-field formulations with and without condensation on a hanging mesh with PEC
and prescribed traces; the reusable elimination and the condensed loads reproduce the
one-shot pipeline; a basis of five frequency snapshots ($k = 2 \dots 4$ on a $6 \times 6$,
$p = 3$ box with the exact trace) reproduces full solutions at four other frequencies
within $10^{-2}$ and the snapshot frequencies to $10^{-8}$.

## Measured (`benchmarks/results/2026-10-02-VR.json`)

Maxwell plane wave on the unit square, MSYS2 GCC 16 release build, 24 cores, MUMPS 5.9
sequential (`bench_assembly_solve`):

| n, p | DoFs | nnz (full / condensed) | assembly 1 / 24 threads | factorise SparseLU full / cond. | factorise MUMPS full / cond. |
|---|---|---|---|---|---|
| 64, 2 | 41 216 | 469 k / 258 k | 0.27 s / 0.05 s | 0.38 s / 0.17 s | 0.14 s / 0.10 s |
| 64, 4 | 147 968 | 4.48 M / 1.06 M | 1.57 s / 0.34 s | 10.1 s / 1.10 s | 0.50 s / 0.29 s |
| 128, 3 | 344 832 | 6.89 M / 2.39 M | 2.97 s / 0.56 s | 21.6 s / 4.11 s | 1.33 s / 0.84 s |

Assembly scales 5–7× on 24 threads (the per-thread triplet merge and the sparse
compression stay serial); static condensation removes half to three quarters of the
nonzeros and cuts the SparseLU factorisation up to 9×; MUMPS factorises the 345k system
16× faster than SparseLU. Solves take 10–100 ms. The Mie example (86k DoFs, p = 3) went
from 5 s to 1.5 s with MUMPS and condensation.

## Eigenvalue solvers

See [maxwell.md](maxwell.md#eigenproblems): Spectra's shift-invert Lanczos / Arnoldi on top
of the SparseLU factorisation of $S - \sigma M$ (gauged curl–curl eigenproblems and the
Lee–Sun–Cendes pencil of `PropagatingMode`), real pencils only. Complex pencils — lossy
media, PML, the resonance problems of [maxwell.md](maxwell.md#resonances) — go through
`complex_eigenpairs_near`: an own shift-invert Arnoldi in complex arithmetic on
$(A - \sigma B)^{-1} B$ with the direct solver backends of this page for the factorisation
(MUMPS when compiled in), modified Gram–Schmidt with re-orthogonalisation, explicit restarts
from the wanted Ritz vectors and the relative residual $|h_{m+1,m}\,y_m|/|\theta|$ as
convergence test; eigenvalues come back ordered by distance to the shift with their
residuals. All three solvers scale the pencil to $O(1)$ matrices internally: on SI meshes the
mass entries are $\sim h^2 \sim 10^{-14}$ and the eigenvalues $\sim 10^{13}$, which made the
absolute thresholds of the Krylov iterations stop early with non-converged values.

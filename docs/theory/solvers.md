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

## Eigenvalue solvers

See [maxwell.md](maxwell.md#eigenproblems): Spectra's shift-invert Lanczos / Arnoldi on top
of the SparseLU factorisation of $S - \sigma M$ (gauged curl–curl eigenproblems and the
Lee–Sun–Cendes pencil of `PropagatingMode`). Switching the inner factorisation to the
direct solver backends is planned together with static condensation (M6).

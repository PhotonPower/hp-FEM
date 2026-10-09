# 0007 — Direct solver backends behind one interface
**Status:** accepted, amended by [0008](0008-gpu-backend.md) (GPU backend behind the same
interface; `LinearSolver::solve_many` for several right-hand sides at once) and by
[0012](0012-optimisation-and-uq.md) (`LinearSolver::solve_transposed`, kept factorisation for
the sensitivities) ·
**Date:** 2026-10-02

## Decision
All linear solves go through `solvers::LinearSolver` (factorise once, solve many) with a
backend chosen by `solvers::DirectSolverBackend`: Eigen's SparseLU is always built, MUMPS
(sequential complex double build, `zmumps_c`) is an optional dependency behind
`HPFEM_ENABLE_MUMPS` found by `cmake/FindMUMPS.cmake`, and `kAuto` prefers MUMPS when it is
compiled in. Problem classes expose the choice as an option (`ScatteringSetup::solver`);
nothing in the library depends on a particular backend.

## Consequences
- The system is handed over as centralised COO input; backends are free to reorder, scale
  and pivot. MUMPS gives the memory and time behaviour needed for 3D and for many
  right-hand sides (sweeps, adjoints); SparseLU keeps every build self-contained.
- The CI runs the complete suite with both backends (`mumps` preset on Ubuntu), so the
  tolerances of the convergence tests are met by both.
- A distributed MUMPS (MPI) or PARDISO backend can be added as another `make_*` without
  touching callers; the iterative solvers of later milestones use the same interface for
  preconditioner factorisations.

## Alternatives considered
PETSc/SLEPc as the single solver layer: heavier dependency and build friction on Windows;
revisit for MPI (M6 optional item). Linking MUMPS statically: avoids the DLL path issue on
Windows but needs the full Fortran/BLAS/SCOTCH chain at link time; dynamic linking with
`MUMPS_BIN_DIR` on the test PATH is simpler and matches the Ubuntu packages.

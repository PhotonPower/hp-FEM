# 0001 — Language, build system and core dependencies
**Status:** accepted · **Date:** 2026-10-02

## Context
A high-order FEM code needs predictable performance in element loops, mature sparse
linear algebra, and a scripting layer for users. Development is done largely with
Claude Code, so the ecosystem should be one where tooling, libraries and reference
implementations (deal.II, MFEM, NGSolve, FEniCS) are abundant.

## Decision
- **C++20** core library (`hpfem`), **Python** user API via **pybind11**.
- **CMake ≥ 3.25** with presets and **Ninja**; dependencies via FetchContent with
  pinned tags, system packages preferred when present.
- Dense/sparse containers: **Eigen 3.4**. Logging: spdlog/fmt. Config/results: nlohmann_json.
  Tests: Catch2 v3. Eigensolvers: Spectra (header-only, on top of Eigen) initially;
  added in M3 (v1.2.0, FetchContent, real symmetric Lanczos with our own gauge
  projector; complex / lossy eigenproblems need an Arnoldi variant later).
- Optional heavy backends behind CMake options: MUMPS, PETSc/SLEPc, MPI.
- Mesh generation is external (**Gmsh**), hpfem reads `.msh` v4.
- Code, comments and documentation are in **English**; user-facing conversation may be
  in any language.

## Consequences
- Fast element kernels, templates on `Dim` and (where it pays) on `p`.
- Python layer is thin: units, materials library, plotting, scripting.
- Windows is not a first-class target until M7; Linux/macOS CI first.

**Amendment (2026-11-17, macOS / libc++):** the library uses the C++17 mathematical special
functions (`std::cyl_bessel_j`, `std::cyl_neumann`, `std::sph_bessel`, `std::sph_neumann`) for
line sources, Mie series and analytic references. libc++ (Apple Clang) does not implement them.
`cmake/Dependencies.cmake` checks for them and, only where they are missing, fetches the
header-only Boost.Math in standalone mode (pinned `boost-1.86.0`, no other Boost library);
`core/special_functions.hpp` wraps both behind `hpfem::cyl_bessel_j` etc. with the error
behaviour of the standard (NaN / ±inf, no exceptions). Builds with libstdc++ (Linux CI, MinGW)
and the MSVC STL are unchanged. Not load-bearing: dropping macOS support removes it.

## Alternatives considered
- *Rust*: attractive safety story, but no mature sparse direct/eigen solver stack and
  far fewer FEM references. Rejected for now.
- *Julia*: excellent for prototyping, weaker story for a shippable library with
  C-ABI and long-running services. Rejected.
- *Building on deal.II or MFEM*: would give hanging nodes and Nédélec elements for free,
  but variable-p Nédélec and the hp-decision machinery are exactly what we want to own and
  control. May revisit for individual components (e.g. MFEM's quadrature tables).

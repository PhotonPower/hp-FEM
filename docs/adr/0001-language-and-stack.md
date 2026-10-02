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
  Tests: Catch2 v3. Eigensolvers: Spectra (header-only, on top of Eigen) initially.
- Optional heavy backends behind CMake options: MUMPS, PETSc/SLEPc, MPI.
- Mesh generation is external (**Gmsh**), hpfem reads `.msh` v4.
- Code, comments and documentation are in **English**; user-facing conversation may be
  in any language.

## Consequences
- Fast element kernels, templates on `Dim` and (where it pays) on `p`.
- Python layer is thin: units, materials library, plotting, scripting.
- Windows is not a first-class target until M7; Linux/macOS CI first.

## Alternatives considered
- *Rust*: attractive safety story, but no mature sparse direct/eigen solver stack and
  far fewer FEM references. Rejected for now.
- *Julia*: excellent for prototyping, weaker story for a shippable library with
  C-ABI and long-running services. Rejected.
- *Building on deal.II or MFEM*: would give hanging nodes and Nédélec elements for free,
  but variable-p Nédélec and the hp-decision machinery are exactly what we want to own and
  control. May revisit for individual components (e.g. MFEM's quadrature tables).

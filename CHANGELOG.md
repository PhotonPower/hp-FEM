# Changelog
All notable changes to this project are documented here (Keep a Changelog, SemVer).

## [Unreleased]
### Added
- `mesh::SimplexTopology<Dim>`: binding local numbering of the reference triangle and
  tetrahedron (vertices, edges, faces, face-edge table, face permutation codes).
- `mesh::Mesh<Dim>` (2D/3D): vertices and cells with derived, lexicographically numbered
  edges and faces, per-cell edge flip flags and face permutation codes following the
  lowest-vertex-first rule (ADR-0003); unit tests (local tables, orientation
  consistency, Euler characteristic, renumbering invariance) and `docs/theory/mesh.md`.
- `hpfem::log()` accessor for the spdlog logger; `hpfem::as_size()` index helper.
- `mesh::Mesh<Dim>` connectivity tables: facet → cells with local facet numbers, cell →
  neighbours, edge → cells (CSR), sorted boundary facets; non-manifold meshes are
  rejected. `hpfem::kInvalidIndex` marker. Unit tests for inverse tables, neighbour
  symmetry, boundary counts and edge rings.

## [0.1.0] — 2026-10-02
### Added
- Project scaffold: CMake presets, CI matrix, clang-format/tidy, pre-commit, devcontainer.
- `hpfem::version()`, core types/constants/error handling with tests.
- Python package skeleton (scikit-build-core + pybind11).
- Documentation site with theory pages (Maxwell, Nédélec, hp-adaptivity, error
  estimation, PML), architecture, roadmap and ADRs 0001–0005.

# Changelog
All notable changes to this project are documented here (Keep a Changelog, SemVer).

## [Unreleased]
### Fixed
- MinGW builds link the GCC runtime (libstdc++, libgcc, winpthread) statically
  (`HPFEM_STATIC_RUNTIME`, default ON), so test executables no longer crash with
  `STATUS_ENTRYPOINT_NOT_FOUND` when an older `libstdc++-6.dll` (Git for Windows) is
  first on `PATH`.

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
- `mesh::Mesh<Dim>` tags (physical groups): material tag per cell, tag per facet,
  tagging from unordered vertex tuples (`set_facet_tags`), `tag_boundary`, lookups
  `edge_id` / `face_id` / `facet_id` by binary search, physical names per dimension.
- `mesh::rectangle` / `mesh::box` structured generators with per-side boundary tags
  (`box_tag`), and `mesh::read_gmsh<Dim>` for Gmsh MSH 4.1 ASCII files (cells, facet
  tags and physical names from physical groups, sparse node tags, unit scale).
- `mesh::AffineMap<Dim>` / `affine_map`: per-cell origin, Jacobian, inverse transpose,
  signed determinant, diameter, `to_physical` / `to_reference`, `volume`, `centroid`;
  `facet_measure` and `outward_normal`; degenerate cells rejected.
- Curved-geometry hook: `Mesh::set_edge_nodes` (one node per edge, order 2),
  `CellGeometry<Dim>` interface with `AffineGeometry` and `QuadraticGeometry` (quadratic
  Lagrange map, Newton inversion), `cell_geometry(mesh, c)` dispatch.
- `mesh::refine_uniform`: red refinement (triangle → 4, tetrahedron → 8 by Bey's rule)
  with child → parent map, inherited cell/facet tags and names, and curved geometry
  carried over through the parent cell maps. `Mesh::tag_names(dim)` accessor.

## [0.1.0] — 2026-10-02
### Added
- Project scaffold: CMake presets, CI matrix, clang-format/tidy, pre-commit, devcontainer.
- `hpfem::version()`, core types/constants/error handling with tests.
- Python package skeleton (scikit-build-core + pybind11).
- Documentation site with theory pages (Maxwell, Nédélec, hp-adaptivity, error
  estimation, PML), architecture, roadmap and ADRs 0001–0005.

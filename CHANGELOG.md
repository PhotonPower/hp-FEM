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
- `io::VtkWriter<Dim>` (.vtu, ASCII or inline base64): cells incl. quadratic cells of
  second-order meshes, `cell_tag`, fluent cell/point scalars, ids, complex and vector
  data; `io::write_vtu_facets` for boundary/interface facets with tags.
- Milestone M1 (mesh infrastructure) complete.
- `fespace::ReferenceElement<Dim>`: reference simplex geometry (vertices, barycentric
  coordinates and gradients, facet normals/measures, edge and facet parametrisations)
  on the binding local numbering of `mesh::SimplexTopology`.
- `assembly::gauss_legendre`, `gauss_jacobi` (Golub–Welsch, any order) and collapsed
  `simplex_quadrature<Dim>(order)` on triangle/tetrahedron, positive weights, interior
  points, exactness verified up to order 20; `docs/theory/quadrature.md`.
- `fespace::legendre` / `scaled_integrated_legendre` recurrences; hierarchical
  `fespace::H1Basis<Dim>` of arbitrary order with per-entity orders and orientation-aware
  edge/face functions; `fespace::DofMap<Dim>` with minimum rule, cell DoF lists, facet and
  tagged-boundary DoFs; `docs/theory/h1-basis.md`.
- `assembly::SparseAssembler` (COO → CSR), `assemble_h1` / `element_h1` for
  α∇u·∇v + βuv = fv with complex coefficients, `h1_error`, `evaluate_h1`;
  `assembly::dirichlet_values` (hierarchical boundary interpolation) and
  `apply_dirichlet` (symmetric elimination); `solvers::LinearSolver` with Eigen SparseLU.
- Convergence test #1: Poisson with manufactured solution, rates p+1 (L2) / p (H1) in
  2D and 3D and exponential p-convergence; `docs/theory/scalar-fem.md`.
- Milestone M2 (scalar FEM) complete.
- `fespace::NedelecBasis<Dim>`: hierarchical Nédélec (first kind) basis of arbitrary
  order with gradient / non-gradient / Whitney-type functions, orientation-aware edge
  and face functions, reference curls; `fespace::NedelecDofMap` via the generic
  `EntityDofMap` (H1 and H(curl) share numbering, minimum rule and facet queries);
  `CellLayout` and shared polynomial kernels. Tests: span = ND_p, de Rham inclusion,
  finite-difference curls, tangential traces, tangential continuity across facets.

## [0.1.0] — 2026-10-02
### Added
- Project scaffold: CMake presets, CI matrix, clang-format/tidy, pre-commit, devcontainer.
- `hpfem::version()`, core types/constants/error handling with tests.
- Python package skeleton (scikit-build-core + pybind11).
- Documentation site with theory pages (Maxwell, Nédélec, hp-adaptivity, error
  estimation, PML), architecture, roadmap and ADRs 0001–0005.

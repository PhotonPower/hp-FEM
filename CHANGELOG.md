# Changelog
All notable changes to this project are documented here (Keep a Changelog, SemVer).

## [Unreleased]
### Added
- Python bindings of the whole pipeline (`python/bindings/bind_*.cpp`, pybind11): meshes
  (generators, Gmsh input, uniform and adaptive refinement, point location), DoF maps and
  constraints (hanging, Bloch), forms and assembly to SciPy sparse matrices, Dirichlet data,
  interpolation / evaluation / error norms, materials and PML, direct solvers (SparseLU /
  MUMPS) and eigensolvers, `Scattering`, sweeps and reduced basis, waveguide modes, surfaces,
  fluxes, cross-sections, far field, diffraction orders, Mie series, estimators, marking, hp
  refinement and decision, goal-oriented estimation, VTK export. Dimension-templated classes
  are bound as `<Name>2D` / `<Name>3D`, free functions overload on the argument type,
  Python callbacks run with the GIL inside the OpenMP loops and their exceptions propagate.
  `python/tests/` (mesh, Poisson rate, cavity, Mie, slab waveguide, grating, hp loop, DWR,
  export), `docs/python.md`.
- `hpfem::parallel_for` rethrows an exception of a loop body on the calling thread (the
  remaining indices are skipped) instead of terminating the process.

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
- `assembly::element_maxwell` / `assemble_maxwell`: curl–curl stiffness, mass and load on
  the Nédélec space with complex tensor ε and μ⁻¹ (covariant Piola map), `hcurl_error`,
  `evaluate_hcurl`; PEC via `homogeneous_dirichlet` on `NedelecDofMap` (now generic over
  the DoF map type), PMC natural; `docs/theory/maxwell.md` discrete-forms section.
- `assembly::discrete_gradient` (G with S G = 0, Gᵀ M G = K_H1), `extract`, `free_dofs`;
  `solvers::gauged_curl_curl_eigenpairs`: shift-invert Lanczos (Spectra 1.2.0, new
  dependency) with the M-orthogonal gauge projector; convergence test #2 (PEC square and
  cube eigenvalues, rate 2p, zero spurious modes).
- `mesh::PointLocator`: point location with a uniform background grid (affine and curved
  cells, tolerance, hinted search, lowest-cell-id rule on shared facets);
  `assembly::evaluate_h1` / `evaluate_hcurl` / new `evaluate_hcurl_curl` overloads that
  evaluate a discrete field at arbitrary physical points.
- `mesh::subdivide`: uniform subdivision of every cell into n^Dim sub-simplices with
  per-cell duplicated vertices (curved cells piecewise straight); `io::FieldExporter`
  writes H1 and Nédélec fields (values and curls, complex as `_re` / `_im`) as point data
  on the subdivided mesh, `io::cell_averages` / `cell_average_curls` the cell means;
  `VtkWriter` accepts complex vector arrays. M3 complete.
- `assembly::tangential_dirichlet_values`: prescribed tangential trace on the Nédélec
  space by hierarchical L2 projection (edge functions, then face functions), exact for
  traces in the discrete space; `MaxwellForm::curl_source` adds a load paired with
  curl v (the permeability term of the scattered-field formulation).
- `physics::Scattering<Dim>` (M4): total- and scattered-field formulations of the
  time-harmonic problem with isotropic materials by cell tag (`materials::Material`,
  `MaterialMap`), PEC and prescribed-incident-field facets, current sources, field
  evaluation and error norms; analytic sources `plane_wave` and `dipole_field` (3D
  Hertz dipole, 2D line dipole via Hankel functions in `core/special_functions.hpp`);
  `assemble_maxwell` with a per-cell form factory. Convergence test: plane wave and
  dipole field reproduced with rate p, scattered and total formulations agree.
- `pml::PmlBox<Dim>` (M4): perfectly matched layers as complex coordinate stretching
  around an interior box (polynomial profile, σ_max from the target reflection,
  stretched coordinates in closed form, effective ε / μ⁻¹ tensors), used by
  `physics::Scattering` through `ScatteringSetup::pml`. Convergence test #3: plane
  wave into the layer at normal and 60° incidence, rate p under h-refinement and
  error below 1e-6 under p-refinement. `hcurl_error` and `Scattering::error` over a
  cell subset, `Scattering::interior_cells`. Nédélec span/continuity tests extended to
  p = 6 in 2D; p-refinement tests of the L2 projection and the plane-wave problem.
- `fespace::Constraints` (linear DoF constraints with chain resolution and P^T A P
  reduction) and `assembly::bloch_constraints` / `PeriodicPair` (M4): Bloch-periodic
  boundaries as constraints obtained by projecting shifted master basis functions onto
  the slave trace space (orientation flips and face permutations automatic);
  `ScatteringSetup::periodic`. Convergence test: Bloch plane wave with rate p in 2D (one
  direction) and 3D (two directions).
- Curved elements (M4): second-order Gmsh files (6-node triangles, 10-node tetrahedra)
  become edge nodes; `mesh::disc` / `mesh::ball` generators with the boundary projected
  onto the circle / sphere and `mesh::curve_boundary` for any tagged boundary; the
  assemblers raise the quadrature degree on curved cells, `MaxwellForm::quadrature_order`
  overrides it per cell and PML cells use `pml_extra_quadrature_order`. Convergence test:
  Poisson on the disc and ball with full rates on curved meshes (rate capped at 2 on
  polygonal ones), Maxwell plane wave on the disc with rate p.
- `physics/postprocess.hpp` (M4): oriented `Surface`s of facets with quadrature through
  the cell geometry, `poynting_flux` of discrete / analytic / combined fields,
  `absorbed_power`, `plane_wave_intensity` and `cross_sections` (scattering, absorption,
  extinction) of a scattering solution; tests against analytic fluxes and the energy
  balance of a lossy scatterer.
- `physics/mie.hpp` (Mie series of the dielectric cylinder, H_z polarisation),
  `mesh::square_with_disc` (box with a curved circular inclusion and PML space) and
  convergence test #4: the FEM scattering width converges to the Mie value.
- `physics::FarField` (Stratton–Chu far-field pattern, radiated power and
  cross-section from a closed surface, 2D and 3D) and `physics/diffraction.hpp`
  (`fourier_coefficients`, `diffraction_efficiencies` of Bloch-periodic problems).
- `physics::PropagatingMode` (M4): waveguide modes of a 2D cross-section from the
  Nédélec / H1 (Lee–Sun–Cendes) pencil; `solvers::generalized_eigenpairs_near` (real
  nonsymmetric shift-invert Arnoldi for indefinite pencils); `assemble_h1` with a
  per-cell form factory. Convergence test #5: slab waveguide effective index with rate
  2p and exponential p-convergence.
- Convergence test #6: lamellar grating diffraction efficiencies (Bloch unit cell, PML,
  scattered-field formulation, Fourier coefficients above and below) against an RCWA
  with Li's rules written in the test.
- Examples `cavity_modes`, `mie_cylinder`, `slab_waveguide`, `lamellar_grating` as C++
  drivers (`HPFEM_BUILD_EXAMPLES`, on by default) with READMEs. M4 complete.
- `adaptivity::residual_estimate`: residual-based a-posteriori estimator for the
  curl–curl problem (element residual with curl curl, Gauss-law divergence residual,
  tangential-curl and normal-flux jumps, h/p weights, per-cell forms incl. PML),
  `physics::Scattering::estimate`; `adaptivity::dorfler_marking` / `maximum_marking`.
  Convergence test: effectivity index and rate of the estimate on a plane wave (2D/3D).
- Local h-refinement with hanging nodes (ADR-0006): `mesh::AdaptiveMesh` (red refinement
  tree, one-irregular closure by vertices and facets, leaf mesh with registered hanging
  edges / faces, tag and curved-geometry transfer, `RefinementStep`), `Mesh::set_hanging`
  and hanging queries, `mesh::extract` sub-meshes; `assembly::interpolate` (hierarchical
  interpolation on entity sets, now behind the Dirichlet functions),
  `assembly::hanging_constraints`, `assembly::prolongate`, `Constraints::append`; the
  DoF maps' minimum rule includes hanging children; `physics::Scattering` reduces by the
  constraints before imposing Dirichlet data; estimator and flux surfaces handle hanging
  facets. Convergence test: adaptive h-refinement on the L-shaped corner recovers the
  optimal rate N^(-p/2) where uniform refinement is limited to N^(-1/3).
- `adaptivity::p_refine`, `hp_refine` (h-step with inherited orders plus raised orders of
  p-marked cells) and `identity_step`; solutions transfer exactly across p- and hp-steps
  by `assembly::prolongate`. Convergence test: estimator-driven adaptive p-refinement on a
  plane wave converges like exp(-0.41 sqrt(N)).
- hp decision: `adaptivity::hp_decide_by_prediction` / `predict_indicators` (error
  prediction after Melenk–Wohlmuth, the default) and `adaptivity::hp_decide` /
  `coefficient_decay` (decay of the Dubiner coefficients, `fespace::DubinerBasis`,
  `fespace::jacobi`, order-dependent threshold calibrated on the corner singularity);
  `hp_refine` spreads p-refinement to lower-order facet neighbours. Convergence test #7:
  hp-adaptivity on the L-shaped corner, error ~ exp(-0.28 N^(1/3)), 6e-5 at 26 000 DoFs
  where h-adaptivity with p = 2 needs ~1e-3 at 12 600.
- Goal-oriented (dual-weighted residual) estimation: `physics::dwr_estimate` (adjoint on
  the p+1 space in the test space of the constrained problem, weight z − I_p z, signed
  cell contributions), `adaptivity::weighted_residual` (incl. natural boundary terms),
  `assembly::point_functional`, `physics::point_value_functional`,
  `physics::fourier_coefficient_functional`. `Constraints` without constraints no longer
  crash in `prolongation` / `reduced_index`. Convergence test: goal-driven refinement of
  a point value on the L-shape reaches 1.5e-5 at 6 800 DoFs where energy-driven
  refinement gives 1.3e-3, effectivity 0.5–0.9.
- Example `plasmonic_dimer` (two gold-like rods with a gap, scattered-field formulation
  with PML, seven hp steps from a 40 × 40 / p = 2 start, DWR estimate of the gap field,
  VTK output with orders, levels and indicators) and convergence test #7, second part:
  the manufactured plasmonic-wedge solution (complex corner exponent 0.573 − 0.015i) is
  resolved exponentially by the hp loop (b = 0.28, algebraic slope −2.0 against −0.29 for
  uniform refinement). `RefinementStep` now records multi-level chains (`path`,
  `old_reference`): the one-irregular closure may split a cell twice in one call. M5
  complete.
- MUMPS direct solver backend (`solvers::make_mumps`, `DirectSolverBackend`,
  `make_direct_solver`, `available_backends`; `solve_direct` and
  `ScatteringSetup::solver` take the backend, `kAuto` prefers MUMPS): option
  `HPFEM_ENABLE_MUMPS`, `cmake/FindMUMPS.cmake` (pkg-config `mumps-zso`, Debian
  `*_seq` libraries, plain installs), preset `mumps`, CI job on Ubuntu with
  `libmumps-seq-dev`; test executables get the MSYS2 DLL directory on `PATH`.
  ADR-0007, `docs/theory/solvers.md`.
- Static condensation of the interior DoFs (`assembly::StaticCondensation`,
  `assemble_maxwell_operator`, `assemble_h1` with a condensation argument): Schur
  complements per cell, identity rows for the interior DoFs so Dirichlet data and
  constraints apply unchanged, recovery after the solve; `Scattering::solve` condenses by
  default (`ScatteringSetup::condense`).
- OpenMP parallel assembly and estimation (`core/parallel.hpp`: `parallel_for`,
  `num_threads`, `set_num_threads`; per-thread triplet buffers, right-hand sides and
  quadrature caches in `assemble_maxwell`, `assemble_maxwell_operator`, `assemble_h1`,
  per-facet storage in `residual_estimate` / `weighted_residual`, mutex in
  `StaticCondensation`), option `HPFEM_ENABLE_OPENMP`, static libgomp on MinGW.
- Benchmark `bench_assembly_solve` (`HPFEM_BUILD_BENCHMARKS`): assembly, factorisation and
  solve times for (n, p) pairs, thread counts, condensation and solver backends as JSON
  lines; results in `benchmarks/results/`.
- Parameter sweeps: `physics::ScatteringOperator` (operator factorised once, solves for
  any incident field / current with new loads, condensed loads and Dirichlet values),
  `solve_many`, `plane_wave_sweep`; `assembly::DirichletElimination` (reusable
  elimination keeping the eliminated columns), `StaticCondensation::condense_load` /
  `recover(x, load)`, `Constraints::reduce_rhs`, `assemble_maxwell_load`;
  `solvers::ReducedBasis` (orthonormal snapshot basis, Galerkin projections, lift) for
  affine frequency sweeps. M6 complete apart from the optional MPI item.

## [0.1.0] — 2026-10-02
### Added
- Project scaffold: CMake presets, CI matrix, clang-format/tidy, pre-commit, devcontainer.
- `hpfem::version()`, core types/constants/error handling with tests.
- Python package skeleton (scikit-build-core + pybind11).
- Documentation site with theory pages (Maxwell, Nédélec, hp-adaptivity, error
  estimation, PML), architecture, roadmap and ADRs 0001–0005.

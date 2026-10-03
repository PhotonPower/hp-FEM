# Changelog
All notable changes to this project are documented here (Keep a Changelog, SemVer).

## [Unreleased]
### Added
- `solvers::Symmetry::kComplexSymmetric` on `make_direct_solver` / `solve_direct`: cuDSS
  factorises the upper triangle as LDLᵀ (`CUDSS_MTYPE_SYMMETRIC`) and MUMPS runs with
  `SYM = 2`, both with about half the factor work (RTX 3090: 164 k unknowns in 2D 1.16 →
  0.85 s on MUMPS, 0.72 → 0.52 s on cuDSS; 70 k in 3D 4.65 → 2.39 s and 1.84 → 1.24 s;
  `benchmarks/results/2026-10-03-VR-backend-symmetry.json`); SparseLU ignores the flag,
  `kAuto` forwards it. The caller guarantees `A = Aᵀ`; helpers `upper_triangle`,
  `asymmetry`. Python: `Symmetry`, `asymmetry`, `symmetry` arguments.

- Layered background for the scattered-field formulation (ADR-0009, M10 D1):
  `physics::LayerStack<Dim>` (planar layers perpendicular to the last coordinate, stable
  Airy / S-matrix recursion with layer-local amplitudes, `plane_wave` returns the exact
  field as `IncidentField` plus R, T, A), `ScatteringSetup::background` (source only where a
  cell deviates from the stack, `Scattering::background_material` /
  `incidence_material`, straddling cells rejected), Python `LayerStack2D/3D`, `Layer`,
  `Polarisation`, `ScatteringSetup.background`; theory section in docs/theory/maxwell.md.
- Mie series of the sphere (M10 C): `physics::mie_sphere` / `MieSphere` (a_n, b_n, c_n, d_n,
  Q_sca / Q_ext / Q_abs, cross-sections, fields inside and outside for complex ε), spherical
  Bessel functions incl. complex `spherical_bessel_j` by downward recurrence,
  `mesh::box_with_ball` (cube with a curved spherical inclusion), Python bindings; the 3D half
  of convergence test #4 (`mie_sphere`, quarter domain by symmetry) and a cross-check of the
  series against `miepython` (eight digits, near field to 1e-6).
- Validation benchmarks B (metallic lamellar grating, Granet & Guizal 1996) and D2
  (slit–groove in silver, Besbes et al. 2007 / Burger et al. 2013) with their long local
  studies in `benchmarks/results/`; `docs/validation.md` sections B–D. Findings: the PML of
  one wavelength / order 2 leaves a 1e-4 energy defect in efficiencies (3 µm / order 4 /
  1e-14 fixes it); the slit–groove result is limited to ~1e-5 by the surface-plasmon
  truncation, and the 0.1 % difference between the two slit–groove sources is their
  substrate permittivity.
- `DirectSolverBackend::kAuto` prefers cuDSS for systems of at least `HPFEM_GPU_MIN_UNKNOWNS`
  unknowns (CMake cache variable, default 10 000, environment variable of the same name
  overrides; 0 always, negative never) when the GPU library and a device are present; the
  choice is made in `factorize`, the GPU library is loaded only then, `name()` reports
  "auto: <backend>". The default is where the GPU factorisation draws level with sequential
  MUMPS on the RTX 3090 (`bench_backend_threshold`,
  `benchmarks/results/2026-10-03-VR-backend-threshold.json`). If cuDSS refuses a system
  (perturbed pivots, as on hp-adaptive systems with hanging nodes), `kAuto` warns and
  factorises it with MUMPS / SparseLU instead. Python: `gpu_min_unknowns()`.
- cuDSS and `solve_many` where one factorisation serves many solves (ADR-0008 follow-up):
  `ScatteringOperator::solve_many(incidents, current)` assembles the loads of several
  incident fields and applies the factorisation in one batched solve (`physics::solve_many`
  and `plane_wave_sweep` use it); the real shift-invert eigensolvers
  `gauged_curl_curl_eigenpairs` and `generalized_eigenpairs_near` take a
  `DirectSolverBackend` (`kAuto` keeps the real SparseLU, `kMumps` / `kCudss` factorise
  the complexified shifted matrix), `WaveguideSetup::solver` passes it on; Python
  `ScatteringOperator2D/3D.solve_many`, `backend` arguments of both eigensolvers. The GPU
  library factorises `s·A` with `s = 1/max|a_ij|` (cuDSS judges tiny pivots by an absolute
  threshold, which the SI-scaled Newmark operator trips) and rescales the solutions; the
  micro-benchmark does the same and records the perturbed-pivot count.
  `benchmarks/solver_integration.cpp` (transient cavity, 8-angle sweep one by one against
  batched, resonance Arnoldi, real gauged Lanczos with every backend) with results in
  `benchmarks/results/2026-10-03-VR-gpu-integration.json`: on the RTX 3090 the cuDSS
  time step is 3.8× faster than MUMPS, the batched 8-rhs solve 5.7×, the resonance solve
  1.9×.
- `physics::AxisymmetricCavity` (M11, ADR-0010): eigenmodes of bodies of revolution on the
  meridian mesh, one 2D problem per azimuthal order m. `assembly::assemble_axisymmetric`
  builds the order-m curl–curl and mass forms with (E_r, E_z) in the 2D Nédélec space and
  the scaled azimuthal unknown v = −i r E_φ in H1 (diagonal material tensors in (r, φ, z),
  weight r, higher quadrature on axis cells), `axisymmetric_gradient` the order-m gradient
  [G; m I] that spans the kernel exactly, so the gauged eigensolver finds no spurious modes;
  axis conditions per m. Convergence test `axisymmetric_cavity` (PEC cylinder against the
  Bessel zeros, rate 2p for m = 0, 1, 2), unit tests, `docs/theory/axisymmetric.md`.
- `physics::AxisymmetricResonance` (M11): quasi-normal modes of open bodies of revolution
  with the cylindrical PML as material (`axisymmetric_pml_form`, Teixeira–Chew tensors
  with the stretched radius) and the complex gauged shift-invert solver; modes with ω, λ, Q
  and residual. Convergence test `axisymmetric_sphere_resonance` (TE_1 and TM_1 Mie poles
  of a dielectric sphere, exponential in p down to 2e-6), unit tests of the PML tensors.
- `physics::AxisymmetricScattering` (M11): scattered-field formulation of one azimuthal
  order with the axial plane wave (`axial_plane_wave`, m = ±1), volume sources in the
  axisymmetric forms, and `axisymmetric_poynting_flux` for the power through surfaces of
  revolution; convergence test `axisymmetric_mie_sphere` against the Mie cross-section of a
  sphere (exponential in p). Dipole sources on the axis (`axisymmetric_gaussian_dipole`,
  axial m = 0, transverse m = ±1) with the total-field formulation
  (`AxisymmetricScatteringSetup::current`) and `dipole_vacuum_power`; convergence test
  `axisymmetric_dipole` against the Larmor power of the smeared dipole, Purcell peak at the
  TM_1 Mie pole. Python: `AxisymmetricCavity`, `AxisymmetricResonance`,
  `AxisymmetricScattering`, `axial_plane_wave`, `axisymmetric_gaussian_dipole`,
  `axisymmetric_poynting_flux`, `AxisDipole`.
- `examples/micropillar_qd` (M11): GaAs/AlAs micropillar as a body of revolution —
  fundamental m = 1 resonance (934.6 nm, Q ≈ 1000 for 10/16 pairs, r = 1 µm), Purcell
  spectrum of the in-plane quantum-dot dipole peaking on the resonance (F_P = 5.1) and the
  β factor (0.47); regression test in the quick configuration.
- cuDSS GPU direct solver (backlog "GPU backend", ADR-0008): `DirectSolverBackend::kCudss`
  behind `solvers::LinearSolver`, opt-in (`kAuto` unchanged). The solver lives in the
  separately built shared library `hpfem_gpu` (`gpu/`: nvcc + MSVC or GCC, cuDSS 0.8, pure C
  interface `gpu/include/hpfem_gpu.h` with 64-bit CSR indices, factorise once / solve many,
  re-factorisation, stand-alone self-test) and is loaded at run time behind
  `HPFEM_ENABLE_CUDA` (`HPFEM_GPU_DLL`, `HPFEM_GPU_BIN_DIR`); no CUDA is needed to build the
  library, without the DLL or a GPU the backend reports itself unavailable
  (`cudss_status()`). Zero pivots that cuDSS would perturb are reported as a singular
  matrix. `LinearSolver::solve_many(const Matrix&)` solves several right-hand sides at once
  (native in SparseLU, MUMPS and cuDSS; ADR-0007 amended). Python:
  `DirectSolverBackend.CUDSS`, `cudss_status`, `LinearSolver.solve_many`, project key
  `solver.backend: cudss`. CI builds the gcc release job with the option to cover the
  loader fallback. Docs: docs/theory/solvers.md "cuDSS backend (GPU)", `gpu/README.md`.
- Validation against the literature (M10, `docs/validation.md`): benchmark A, the rib
  waveguide of Vassallo (1997) — the quasi-TE and quasi-TM effective indices of
  `physics::PropagatingMode<2>` reproduce the MTRM reference for all nine guided cases within
  the four digits of the source (`tests/convergence/rib_waveguide.cpp`; the quasi-TE mode at
  t = 0.9 µm needs the lateral wall at 10 µm, the leaky quasi-TM mode there is documented
  only). Test infrastructure: `hpfem_add_test` takes a list of labels, long local runs are
  hidden Catch2 cases with the ctest label `validation-long` (excluded by the test presets),
  results as JSON lines in `benchmarks/results/`; `tests/convergence/tensor_mesh.hpp` builds
  conforming tensor-product meshes with interface-aligned, geometrically graded lines.
- `physics::BandStructure` (backlog): Floquet–Bloch band structures of photonic crystals.
  `assembly::bloch_constraints` now also exists for the H1 space, so the discrete gradient
  can be reduced consistently with the Nédélec space; the new
  `solvers::complex_eigenpairs_near_gauged` projects every Arnoldi vector onto the
  B-orthogonal complement of the (Bloch-reduced) gradients, which removes the kernel of the
  complex Hermitian curl–curl pencil without spurious modes. `bands(k)` returns the
  wavenumbers k0 of the lowest bands at a Bloch wave vector (ascending, with residuals) and
  `path(corners, segments)` walks Γ–X–M–Γ. Unit tests (empty lattice |k + G|, zero at Γ,
  symmetry in k, dielectric rods lower the bands, H1 Bloch constraints, gauged solver
  against a dense reference) and the convergence test `empty_lattice_bands`
  (exponential in p, rate 2p in h). Python: `BandStructure2D/3D`, `BandStructureSetup2D/3D`,
  `Bands2D/3D`. Theory: docs/theory/maxwell.md, section "Band structures".
- `adaptivity::hypercircle_estimate` (backlog): dual (magnetic) formulation and the
  hypercircle / Prager–Synge error bound. `dual_form` turns the per-cell Maxwell form into
  the dual curl–curl problem (materials swapped, PEC and PMC exchanged), `dual_solution`
  solves it on an H1 space (2D, scalar curl) or a Nédélec space (3D); the
  constitutive-relation estimate of the primal/dual pair is a guaranteed upper bound of the
  energy error for the coercive problem (k² < 0, real materials) without any constant and
  the Ladevèze estimator for the time-harmonic one. `assembly::ScalarForm` gained
  `diffusion_tensor` and `gradient_source`. Convergence test `hypercircle_bound` (bound holds
  on every mesh, rate p, effectivity 1.0–1.1 with the dual order p + 1), unit tests
  (exact pair gives η = 0, 3D cube, lossy problem), Python `dual_solution`,
  `hypercircle_estimate`, `HypercircleEstimate`. Theory: docs/theory/error-estimation.md,
  section "Dual formulation and guaranteed bounds".
- `physics::TimeDomain` (backlog): transient Maxwell solver. The second-order wave
  equation for E with real materials, conductivity by tag, PEC walls and the first-order
  Silver–Müller absorbing boundary (tangential boundary mass with the local wave impedance)
  is integrated with the implicit Newmark-β scheme (trapezoidal rule by default:
  unconditionally stable, second order, energy conserving); the Newmark operator is
  factorised once. Sources are a current density J(x) times a `TimeSignal` with analytic
  derivative (`gaussian_pulse`, `modulated_gaussian`), `run` calls an observer per step,
  `energy` gives the discrete energy. Convergence test `time_domain_cavity` (TE11 mode:
  order 2 in dt, order p in h, energy drift 1e-15), unit tests (signals, validation,
  conductivity decay, 3D cavity, pulse leaving a strip through the absorbing ends), Python
  `TimeDomain2D/3D`, `TimeDomainSetup2D/3D`, `TimeState2D/3D`, `TimeSignal`. Theory:
  docs/theory/maxwell.md, section "Time domain".

## [0.2.0] — 2026-10-03
Second release: everything from the Nédélec elements to the multiphysics step —
milestones M3 to M9 of `docs/roadmap.md`. Highlights: hierarchical Nédélec spaces with the
gauged Maxwell eigensolver, time-harmonic scattering with PML, Bloch periodicity and curved
elements, waveguide modes, residual and goal-oriented error estimation with hp-adaptivity
(exponential convergence on the L-shape and the plasmonic wedge), MUMPS, static
condensation, OpenMP and parameter sweeps, the complete Python API (`hpfem` with units,
material library, project files and command line, meshio / pyvista / matplotlib interop,
notebooks), the resonance solver for quasi-normal modes, ten application examples and the
optical–thermal coupling with carrier-generation export. Convergence tests #1–#7 of
CLAUDE.md §8 plus the Fabry–Pérot resonance and heat-conduction tests are part of the CI.
### Added
- `examples/ring_resonator/run.py` (M8): ring resonator in the 2D effective-index model —
  resonance wavelengths and Q from `Resonance2D` on the ring-plus-bus structure, bus
  transmission from a Gaussian line current normalised by the bare bus, dips at the
  eigenmode resonances. M8 complete.
- `hpfem.pv` (M9): carrier-generation profiles for device solvers — generation rate
  G = q / (ħω) per cell (`physics::absorbed_power_per_cell`, exact cell integrals) and as
  an H1 field, spectral weighting of monochromatic solutions by a spectral irradiance,
  volume-weighted depth profiles G(z) and export as meshio cell data / CSV;
  `Mesh.cell_volumes`. `examples/solar_cell_texture`: light trapping of a textured silicon
  cell with the generation profile and the temperature rise. M9 complete.
- `physics::ThermoOptical` (M9): the optical-thermal feedback loop — scattering solution →
  absorbed power → temperature → permittivity εr(T) = εr(T0) + dεr/dT (T − T0) per cell →
  scattering again, until the temperature settles (fixed-point iteration with optional
  under-relaxation, history of the temperature changes). `materials::MaterialMap::set_cell`
  overrides the material of single cells. Python: `ThermoOptical2D/3D`,
  `ThermoOpticalSetup`, `ThermoOpticalState`, `MaterialMap.set_cell`.
- `physics::Thermal` (M9): steady heat conduction −∇·(κ∇T) = q on the H1 space of the
  optical mesh with conductivities by cell tag, fixed temperatures on tagged facets,
  adiabatic walls and hanging-node constraints; `physics::absorbed_power_load` assembles
  the right-hand side of the absorbed optical power ωε₀/2 Im(εr)|E|² cell by cell (exact,
  interface jumps kept), `absorbed_power_density` gives the H1 interpolant for export.
  `assembly::ScalarForm::source_reference` (source in reference coordinates of the cell).
  Convergence test `heat_conduction`: temperature of a damped wave in a lossy slab against
  the closed form, exponential in p. Python: `Thermal2D/3D`, `ThermalSetup`,
  `absorbed_power_load`, `absorbed_power_density`. Theory page `docs/theory/multiphysics.md`.
- `examples/euv_mask/run.py` (M8): 3D EUV mask unit cell — tantalum pad on a Mo/Si
  multilayer at 13.5 nm under 6° incidence, Bloch-periodic in x and y with PML above and in
  the substrate, Kuhn tetrahedra with nodes on every layer interface, reflectivity from the
  scattered-field flux (bare mirror against the transfer matrix) and the near field as VTK.
- `physics::gaussian_current` (Python `gaussian_current`): the volume source iωμ₀J of a
  dipole smeared over a normalised Gaussian, evaluated in C++ — a point or line emitter
  inside a structure for the total-field formulation (a Python callback would serialise the
  parallel assembly on the GIL).
- `examples/quantum_dot_purcell/run.py` (M8): Purcell factor and beta factor of a line
  dipole in a GaAs/AlAs DBR ridge cavity over the wavelength — the scattered field of the
  analytic dipole field of the host (no singularity discretised), the emitted power as the
  Poynting flux of the total field through a circle around the dipole; the dipole in front
  of a PEC mirror checks the post-processing against the image solution.
- `solvers::complex_eigenpairs_near`: eigenpairs of a complex pencil closest to a complex
  shift (shift-invert Arnoldi with explicit restarts on a direct factorisation; lossy media,
  PML, complex frequencies), scaled internally like the real solvers.
- `physics::Resonance` (`ResonanceSetup`, `ResonantMode`): quasi-normal modes of open
  structures with PEC and PML — complex ω, resonance wavelength, Q = Re ω / (−2 Im ω) and the
  field; hanging-node meshes supported. Convergence test `fabry_perot_resonance` (exact
  complex Fabry–Pérot resonances, exponential in p), Python bindings (`Resonance2D/3D`,
  `complex_eigenpairs_near`), project files with `"problem": "resonance"`.
- `examples/vcsel_cavity/run.py` (M8): GaAs/AlAs quarter-wave DBR cavity at 850 nm on an
  interface-aligned strip mesh, resonance wavelength and Q against the transfer-matrix pole
  of the same stack, Q over the number of top pairs, mode profile.
- `examples/metasurface_unitcell/run.py` (M8): zeroth-order transmission and phase of a
  TiO₂ ridge on fused silica over the ridge width (Bloch unit cell, PML, phase relative to
  the bare substrate), with `python/tests/test_examples.py` as the regression layer of the
  Python examples.
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
- `hpfem.units` (length, time, frequency and energy multipliers; conversions between
  wavelength, angular frequency, frequency, photon energy and wavenumber) and
  `hpfem.materials` (dispersive materials `Tabulated`, `Sellmeier`, `DrudeLorentz` /
  `Drude`, `Constant` with `at(omega)` → core `Material`; library Si, SiO2, Au, Ag, Al, TiO2,
  GaAs, MAPbI3, water, air from the refractiveindex.info database with the original
  references in `python/hpfem/data/*.csv`).
- `hpfem.project`: JSON / YAML project files (problem type, mesh generator or Gmsh file with
  regions, order, spectral sweep, materials by tag from the library, source, PML, boundary
  names, Bloch pairs, solver, outputs: cross-sections, far field, point values, fluxes,
  diffraction efficiencies, estimate, VTK) with `load` / `validate` / `run`;
  `hpfem.cli` and the `hpfem` console script (`run`, `validate`, `info`, `materials`);
  `examples/*/project.json`.
- `examples/notebooks/`: three Jupyter notebooks (Mie cylinder, hp-adaptivity on the
  L-shape, gold nanowire absorption spectrum) committed without outputs and executed cell
  by cell in the Python test suite. M7 complete.
- `hpfem::parallel_for` rethrows an exception of a loop body on the calling thread (the
  remaining indices are skipped) instead of terminating the process.
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

### Fixed
- `solvers::gauged_curl_curl_eigenpairs` and `generalized_eigenpairs_near` scale the pencil
  to O(1) matrices internally: on SI meshes (mass entries ~ 1e-14, eigenvalues ~ 1e13)
  Spectra's absolute thresholds made the Lanczos iteration stop early with non-converged,
  run-to-run varying eigenvalues. Unit test: eigenvalues of the 1 µm box equal those of
  the unit box times 1e12 and are reproducible.
- MinGW builds link the GCC runtime (libstdc++, libgcc, winpthread) statically
  (`HPFEM_STATIC_RUNTIME`, default ON), so test executables no longer crash with
  `STATUS_ENTRYPOINT_NOT_FOUND` when an older `libstdc++-6.dll` (Git for Windows) is
  first on `PATH`.

## [0.1.0] — 2026-10-02
### Added
- Project scaffold: CMake presets, CI matrix, clang-format/tidy, pre-commit, devcontainer.
- `hpfem::version()`, core types/constants/error handling with tests.
- Python package skeleton (scikit-build-core + pybind11).
- Documentation site with theory pages (Maxwell, Nédélec, hp-adaptivity, error
  estimation, PML), architecture, roadmap and ADRs 0001–0005.

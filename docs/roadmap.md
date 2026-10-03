# Roadmap

The project task list. Claude Code picks the lowest unchecked item of the current
milestone (CLAUDE.md §11). Each item is done when the Definition of Done (CLAUDE.md §7)
holds. Estimated effort is in rough "focused sessions".

## M0 — Scaffold ✅
- [x] CMake + presets, strict warnings, FetchContent deps
- [x] CI matrix (gcc/clang/asan), clang-format, clang-tidy, docs deploy
- [x] Python package skeleton (scikit-build-core + pybind11)
- [x] Docs site, theory pages, ADRs, this roadmap

## M1 — Mesh infrastructure (≈ 6) ✅
- [x] `mesh::Mesh<Dim>`: vertices, cells (tri/tet), **derived edges and faces** with
      consistent global orientation (lowest-vertex-first rule, ADR-0003)
- [x] entity connectivity tables (cell→edge, cell→face, face→cell, edge→cells)
- [x] boundary and material tags (physical groups)
- [x] Gmsh `.msh` v4 reader (ASCII), simple structured generators for tests
- [x] affine geometry mapping: Jacobian, det, inverse-transpose, per cell
- [x] curved (order-2) geometry hook (interface only; implementation in M4)
- [x] uniform refinement (red refinement in 2D and 3D, Bey's rule) → prepares M5
- [x] unit tests: Euler characteristic, orientation consistency, boundary extraction
- [x] VTK (`.vtu`) export of mesh + cell data

## M2 — Scalar FEM on the infrastructure (≈ 4) ✅
- [x] `fespace::ReferenceElement<Dim>` with **fixed local numbering** (theory/nedelec.md)
- [x] Gauss–Jacobi quadrature on triangle/tetrahedron, exactness tests up to order 20
- [x] hierarchical H1 Lagrange basis, arbitrary order p (Legendre/Jacobi based)
- [x] `fespace::DofMap`: vertex/edge/face/cell DoFs, variable p per entity
- [x] global sparse assembly (COO → CSR), element loop with `std::span`
- [x] Dirichlet via DoF elimination
- [x] solver interface + Eigen SparseLU backend
- [x] **convergence test #1**: Poisson, manufactured solution, rate p+1 in L2, p in H1

## M3 — Nédélec elements and Maxwell eigenproblems (≈ 8) ✅
- [x] hierarchical H(curl) Nédélec-I basis, 2D (p ≤ 6) and 3D (p ≤ 4), per
      Schöberl–Zaglmayr (theory/nedelec.md)
- [x] orientation-aware local→global mapping of edge/face functions
- [x] curl–curl stiffness and mass matrices with complex tensor ε, μ
- [x] PEC boundary condition, PMC (natural)
- [x] Maxwell eigenproblem: shift-invert Arnoldi (Spectra) on `(S − σM)`
- [x] gauge handling / kernel filtering (discrete gradients); spurious-mode test
- [x] **convergence test #2**: PEC box eigenvalues, rate 2p; zero spurious modes
- [x] field evaluation at arbitrary points (point location + reference-coordinate inversion)
- [x] VTK export of vector fields (cell-averaged + high-order via subdivision)

## M4 — Time-harmonic scattering & waveguides (≈ 10) ✅
- [x] `physics::Scattering`: total/scattered-field formulation, plane-wave and
      dipole sources (theory/maxwell.md)
- [x] PML as complex coordinate stretching in boundary layer cells (theory/pml.md),
      polynomial profile, adaptive σ from wavelength and cell size
- [x] Bloch-periodic constraints (`E(x+a) = e^{ik·a} E(x)`) via constrained DoFs
- [x] curved elements (isoparametric order 2) + PML-compatible quadrature
- [x] post-processing: Poynting flux through surfaces, absorption, scattering
      cross-section, far field (Stratton–Chu), Fourier/diffraction coefficients
- [x] `physics::PropagatingMode`: 2D waveguide cross-section eigenproblem for
      effective index (quadratic → linearized)
- [x] **convergence test #3**: PML reflection < 1e-6 for plane wave
- [x] **convergence test #4**: Mie cylinder (2D) cross section vs. series
- [x] **convergence test #5**: slab waveguide n_eff
- [x] **convergence test #6**: lamellar grating efficiencies vs. RCWA
- [x] examples: `cavity_modes`, `mie_cylinder`, `slab_waveguide`, `lamellar_grating`

## M5 — Adaptivity: a-posteriori estimation and hp-refinement (≈ 10) ✅
- [x] residual-based estimator for curl–curl (theory/error-estimation.md):
      element residual + tangential-curl jump + normal-displacement jump
- [x] Dörfler marking
- [x] conforming h-refinement with **hanging edges/faces → constrained DoFs**
      (one-irregular rule), re-numbering, transfer of solution (prolongation)
- [x] p-refinement: per-entity order increase, minimum rule on shared entities
- [x] hp-decision: Legendre-coefficient decay / analyticity estimate per element
      (plus the error-prediction strategy, which is the default: the decay of the
      Galerkin solution misjudges unresolved corner cells)
- [x] goal-oriented (dual-weighted) estimator for a scalar quantity of interest
      (e.g. a Fourier coefficient) — needed for scatterometry accuracy claims
- [x] **convergence test #7**: exponential convergence at re-entrant corner (L-shape,
      then plasmonic wedge) — `adaptive_hp_refinement` and `plasmonic_wedge`
- [x] example `plasmonic_dimer`

## M6 — Solvers & performance (≈ 8) ✅ (apart from the optional MPI item)
- [x] MUMPS / PARDISO backend behind `solvers::DirectSolver` — MUMPS (sequential complex
      build) behind `solvers::LinearSolver` / `DirectSolverBackend` (ADR-0007); PARDISO
      can follow as another backend
- [x] static condensation of interior (cell-bubble) DoFs
- [x] OpenMP parallel assembly (colouring or per-thread COO buffers) — per-thread buffers,
      also the estimator loops
- [x] parameter sweeps / reduced basis hooks (frequency, angle, geometry parameters) —
      `physics::ScatteringOperator` / `solve_many` / `plane_wave_sweep` (one factorisation
      per frequency), `solvers::ReducedBasis` (affine frequency sweeps); geometry
      parameters come with the Python layer (M7)
- [ ] optional: MPI domain decomposition (own ADR before starting)
- [x] benchmarks recorded in `benchmarks/results/` (`bench_assembly_solve`)

## M7 — Python API & usability (≈ 6) ✅
- [x] bindings for mesh, spaces, materials, problems, solvers, post-processing —
      `python/bindings/bind_*.cpp` (pybind11, `<Name>2D` / `<Name>3D`, NumPy / SciPy at
      the boundary, callbacks with the GIL, exceptions mapped), `python/tests/`,
      `docs/python.md`
- [x] `hpfem.units` (nm, µm, eV, THz → SI) and `hpfem.materials` library
      (Si, SiO₂, Au, Ag, Al, TiO₂, GaAs, perovskite — tabulated n,k with sources) —
      `python/hpfem/units.py`, `python/hpfem/materials.py` + `python/hpfem/data/*.csv`
      (refractiveindex.info, CC0), Sellmeier / Drude–Lorentz / tabulated models
- [x] JSON/YAML project files (`hpfem run project.json`) + CLI — `hpfem.project`
      (scattering / waveguide / cavity, sweeps, outputs), `hpfem.cli` (`run`, `validate`,
      `info`, `materials`), `examples/*/project.json`
- [x] meshio / pyvista interop, matplotlib helpers — `python/hpfem/interop.py`
      (`to_meshio` / `from_meshio` / `field_to_meshio`, `to_pyvista` / `field_to_pyvista`
      on the subdivided mesh, `plot_mesh` / `plot_field` / `plot_convergence` /
      `plot_far_field`), `python/tests/test_interop.py` (commit 2265608)
- [x] Jupyter example notebooks — `examples/notebooks/` (Mie cylinder, hp-adaptivity on
      the L-shape, gold nanowire spectrum), executed by `python/tests/test_notebooks.py`

## M8 — Application examples (≈ 6) ✅
- [x] `metasurface_unitcell`: phase/transmission map over pillar diameter —
      `examples/metasurface_unitcell/run.py` (TiO₂ ridges on SiO₂, Bloch unit cell with
      PML, zeroth-order transmission and phase over the width), regression test in
      `python/tests/test_examples.py`
- [x] `vcsel_cavity`: DBR micro-cavity mode, Q-factor, resonance wavelength —
      `solvers::complex_eigenpairs_near` (complex shift-invert Arnoldi),
      `physics::Resonance` (quasi-normal modes with PML), convergence test
      `fabry_perot_resonance` (exact complex Fabry–Pérot resonances),
      `examples/vcsel_cavity/run.py` (GaAs/AlAs DBR cavity against the transfer-matrix pole)
- [x] `quantum_dot_purcell`: dipole in micropillar, Purcell factor, β-factor —
      `examples/quantum_dot_purcell/run.py` (line dipole in a DBR ridge cavity, scattered
      field of the analytic dipole, emitted power from the flux around the dipole, Purcell
      and beta spectra; validated against the image dipole in front of a PEC mirror)
- [x] `euv_mask`: 3D absorber on multilayer, oblique incidence, near-field export —
      `examples/euv_mask/run.py` (Ta pad on a Mo/Si mirror at 13.5 nm and 6°, Bloch unit cell
      in x and y, structured Kuhn tetrahedra on the layer interfaces, reflectivity against
      the transfer matrix of the bare mirror, VTK near field)
- [x] `ring_resonator`: coupling + resonance (2D effective-index model first) —
      `examples/ring_resonator/run.py` (quasi-normal modes of the ring coupled to the bus by
      `Resonance2D`, transmission spectrum from a Gaussian current in the bus; the dip sits
      at the eigenmode resonance)

## M9 — Multiphysics (≈ 8) ✅
- [x] absorbed-power density → heat-conduction solve on the same mesh (H1) —
      `physics::absorbed_power_load` / `absorbed_power_density`, `physics::Thermal`
      (κ by tag, fixed temperatures, hanging nodes), convergence test `heat_conduction`
      (damped wave in a lossy slab against the closed-form temperature), Python bindings,
      `docs/theory/multiphysics.md`
- [x] temperature-dependent ε feedback loop — `physics::ThermoOptical` (fixed-point
      iteration with relaxation, complex dεr/dT by tag, per-cell `MaterialMap` overrides),
      unit tests (uncoupled limit, self-consistent fixed point, linear response, relaxation)
- [x] carrier-generation profile export for PV device solvers — `hpfem.pv`
      (generation rate per cell and as a field, spectral weighting, depth profiles,
      meshio / CSV export), `physics::absorbed_power_per_cell`, Beer–Lambert check,
      `examples/solar_cell_texture`

## M10 — Validation against the literature
Independent, published reference values instead of our own implementations; every
benchmark is a convergence test with the labels `convergence` and `validation`, the long
local runs carry `validation-long` and are stored under `benchmarks/results/`. Results and
assessment: `docs/validation.md`.
- [x] A. rib waveguide (Vassallo, Opt. Quantum Electron. 29, 95, 1997, Table I / MTRM):
      quasi-TE and quasi-TM effective indices for five lateral thicknesses within the
      four-digit reference — `tests/convergence/rib_waveguide.cpp`, test helper
      `tests/convergence/tensor_mesh.hpp` (graded tensor meshes), PR #58
- [ ] B. metallic lamellar grating, H parallel to the ridges (Granet & Guizal, J. Opt. Soc.
      Am. A 13, 1019, 1996, Table 1): Littrow and specular efficiencies for three depths
- [ ] C. Mie sphere (Bohren & Huffman ch. 4): `physics::mie_sphere` (coefficients,
      efficiencies, fields inside and outside for complex ε), `mesh::box_with_ball`,
      3D convergence test `mie_sphere` (dielectric and absorbing sphere), Python bindings
- [ ] D. layered background for the scattered-field formulation (ADR-0008) and the
      slit–groove benchmark in silver (Besbes et al., J. Eur. Opt. Soc. Rapid Publ. 2,
      07022, 2007; Burger et al., Proc. SPIE 8880, 88801Z, 2013)
- [ ] gold sphere dimer (Hoffmann et al., Proc. SPIE 7390, 73900J, 2009; 80 nm spheres,
      1 nm gap, 632 nm, reference |E|² at the gap centre = 5.47624·10⁵ V²/m² for
      |E_inc| = 1 V/m) — **blocked**: the permittivity of gold used in the source is not
      stated; not to be attempted with a self-chosen ε

## Backlog / ideas
- [x] dual H-formulation for guaranteed error bounds (`adaptivity::dual_solution` on the
  H1 / Nédélec dual space, `hypercircle_estimate` with the Prager–Synge bound for the
  coercive problem; convergence test `hypercircle_bound`, Python `dual_solution` /
  `hypercircle_estimate`)
- [x] Floquet-Bloch band structure solver (`physics::BandStructure`: Bloch constraints with
  complex phases on the Nédélec and H1 spaces, gauged complex shift-invert Arnoldi,
  bands per wave vector and along Γ–X–M paths; convergence test on the empty lattice,
  `BandStructure2D/3D` in Python)
- [x] transient (time-domain) solver via implicit time stepping (`physics::TimeDomain`:
  Newmark-β on the second-order E wave equation, conductivity, PEC, first-order absorbing
  boundary, current sources with time signals; convergence test `time_domain_cavity`,
  Python `TimeDomain2D/3D`)
- [x] GPU backend: cuDSS direct solver behind `solvers::LinearSolver` (ADR-0008, measured on
  the RTX 3090: factorisation 2×, repeated solves 30–140× against MUMPS;
  `DirectSolverBackend::kCudss` opt-in, separately built `hpfem_gpu` library in `gpu/` loaded
  at run time behind `HPFEM_ENABLE_CUDA`, `LinearSolver::solve_many`); GPU assembly deferred
  (FP64 rate of consumer GPUs, MSVC-only toolchain)
- [x] `physics::PropagatingMode<2>` on adaptive meshes: apply the hanging-node constraints
  as `Resonance` does (found during M10 validation; fixed in PR #63)

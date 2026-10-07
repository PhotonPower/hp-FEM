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
- [x] B. metallic lamellar grating, H parallel to the ridges (Granet & Guizal, J. Opt. Soc.
      Am. A 13, 1019, 1996, Table 1): Littrow and specular efficiencies — h = 0.1 and 4.8 µm
      within 3·10⁻⁴ with the validated fill factor 0.5, h = 1 µm (resonant slot) reported
      only; PML needs 3 µm / order 4 for a 10⁻⁶ energy balance —
      `tests/convergence/metal_grating.cpp`, PR #64
- [x] C. Mie sphere (Bohren & Huffman ch. 4): `physics::mie_sphere` (coefficients,
      efficiencies, fields inside and outside for complex ε; eight digits against
      miepython), `mesh::box_with_ball`, 3D convergence test `mie_sphere` (dielectric and
      absorbing sphere, converging in p to the 10⁻³ level that the quadratic geometry and
      SparseLU allow), Python bindings — PR #67
- [x] D. layered background for the scattered-field formulation (ADR-0009:
      `physics::LayerStack`, `ScatteringSetup::background`) and the slit–groove benchmark in
      silver (Besbes et al., J. Eur. Opt. Soc. Rapid Publ. 2, 07022, 2007; Burger et al.,
      Proc. SPIE 8880, 88801Z, 2013): S/S₀ within 1.4·10⁻⁵ of Burger and 1.8·10⁻⁵ of Besbes,
      the 0.1 % difference between the sources is the substrate permittivity; the 10⁻⁶
      target is left by the surface-plasmon truncation of the PML — PRs #65, #68
- [ ] gold sphere dimer (Hoffmann et al., Proc. SPIE 7390, 73900J, 2009; 80 nm spheres,
      1 nm gap, 632 nm, reference |E|² at the gap centre = 5.47624·10⁵ V²/m² for
      |E_inc| = 1 V/m) — **blocked**: the permittivity of gold used in the source is not
      stated; not to be attempted with a self-chosen ε

## M11 — Axisymmetric (2.5D) solver (ADR-0010)
Bodies of revolution on the meridian mesh, one 2D problem per azimuthal order m.
- [x] forms of order m (`assembly::assemble_axisymmetric`: (E_r, E_z) in Nédélec,
  v = −i r E_φ in H1, diagonal tensors in (r, φ, z), order-m gradient for the gauge),
  axis conditions per m, `physics::AxisymmetricCavity`; convergence test on the PEC
  cylinder against the Bessel zeros (rate 2p for m = 0, 1, 2); `docs/theory/axisymmetric.md`
- [x] resonances with the cylindrical PML (`physics::AxisymmetricResonance`, Teixeira–Chew
  tensors as material, complex gauged eigensolver); convergence test on the quasi-normal
  modes of a dielectric sphere (Mie poles, exponential in p)
- [x] scattering with the axial plane wave (`physics::AxisymmetricScattering`,
  `axial_plane_wave`, m = ±1) and the power flux through surfaces of revolution
  (`axisymmetric_poynting_flux`); convergence test against the Mie cross-section of a
  sphere (exponential in p)
- [x] Python bindings (`AxisymmetricCavity`, `AxisymmetricResonance`,
  `AxisymmetricScattering`, `axial_plane_wave`, `axisymmetric_poynting_flux`)
- [x] dipole sources on the axis (`axisymmetric_gaussian_dipole`, total-field formulation
  with `AxisymmetricScatteringSetup::current`); convergence test against the Larmor power
  of the smeared dipole, Purcell peak at the TM_1 Mie pole
- [x] the micropillar quantum-dot example (`examples/micropillar_qd`: resonance of the
  fundamental m = 1 mode, Purcell and beta factor of the in-plane dipole, regression test)
- [x] far field from the m contributions (`axisymmetric_far_field`: Stratton–Chu with the
  analytic azimuthal integration; Larmor pattern of the dipole, Mie cross-section from the
  far field)
- [x] oblique incidence by the sum over the orders (`oblique_plane_wave`, `scatter_orders`,
  `superpose_far_field`; sphere at 50° vs Mie, exponential in p)
- [x] hp-adaptivity on the meridian plane (`adaptivity::axisymmetric_residual_estimate`,
  hanging-node constraints in the three problem classes, `AxisymmetricScattering::estimate` /
  `error`); convergence test at a re-entrant PEC edge, exponential in N^(1/3)

## M12 — Ports, modal expansion, sensitivities
Integrated photonics beyond the effective-index model and the tools around the solvers.
- [x] waveguide ports with modal excitation and S-parameters in 2D
  (`physics::WaveguidePort`, `PortModes<2>`: TM slab modes on the port edges by a 1D p-FEM,
  low-rank modal boundary term in `Scattering`, `port_coefficients`, `s_parameters`);
  convergence test `waveguide_port` (slab section: S21 and |S11| exponential in p),
  Python `WaveguidePort` / `PortModes2D` / `s_parameters`
- [x] 3D ports: cross-section modes from `PropagatingMode` on the extracted port mesh
  (`PortModes<3>`: frame, planar section with the inside cells' tags and orders, PEC rim,
  n × (μ⁻¹ curl E) = (∇ₜE_z − iβEₜ)/μᵣ, modal powers, orientation-independent signs);
  convergence test `waveguide_port_3d` (rectangular waveguide TE10: S21 and |S11| decay
  with p), `PortModes3D` / `s_parameters` for `Scattering3D` in Python
- [ ] the ring resonator with ports in 3D (SOI cross-section) as the example, S-parameters
  of a directional coupler against coupled-mode theory (needs the GPU solver for the
  mesh sizes involved)
- [ ] modal expansion by Riesz projection on the resonance solver (`physics::RieszProjection`,
  `AxisymmetricRieszProjection`: residues of the resolvent on circles around the
  quasi-normal modes plus a background contour, spectra of linear observables as sums over
  modes; verification against the direct solution and the micropillar Purcell spectrum)
- [ ] sensitivities: material derivatives of observables by the adjoint solve (`dwr` adjoint
  reused), then shape derivatives (Hadamard formula with the interface jumps, ADR on the
  geometry parametrisation); verification against finite differences

## M13 — Conical incidence and the E_z polarisation (2.5D)
From the user test report of 5 October 2026 (`spec-m12-ez-polarisation-2d.md`): the missing
polarisation of the 2D solver and the longitudinal wavenumber for the microscope model.
- [x] conical forms (`assembly::assemble_conical`: in-plane E in Nédélec, v = −i E_z in H1,
  longitudinal wavenumber β, diagonal tensors, `conical_gradient`), `physics::ConicalScattering`
  (PEC, Bloch on both spaces, PML tensors Λ = diag(s_y/s_x, s_x/s_y, s_x s_y), hanging nodes,
  scattered- and total-field formulation, layered background with `layered_conical_wave`),
  `conical_plane_wave` / `conical_polarisation`, `conical_poynting_flux`, `conical_fourier_coefficients`
  (any line orientation) and `conical_diffraction_efficiencies` with complex vector amplitudes;
  at β = 0 the solver is the E_z ("TE") polarisation. Unit tests (gradient kernel, decoupling,
  manufactured solutions, flat interface under conical incidence vs the stack), convergence
  tests `conical_mie_cylinder_ez` (series) and `conical_lamellar_grating_ez` (in-test TE RCWA,
  layered background); theory section `docs/theory/maxwell.md#conical-incidence`
- [ ] Python bindings (`ConicalScattering`, `layered_conical_wave`, orders), `hpfem.project` keys
  `"polarisation": "Ez" | "Hz"` and `"azimuth"` (conical), example (Si ridge in E_z, R1 Ag case)
- [x] conical grating validation at β ≠ 0 against the conical RCWA of the GUI work
  (`conical_grating_validation`: glass lamellar grating, s at θ = 40°, φ = 30° and p at
  θ = 50°, φ = 30° with the order m = −2 evanescent in air; 3.5e-5 and 4e-6 at p = 4,
  energy balance 1e-6)
- [ ] the acceptance case R1 of the report (Ag lamellar grating, 50°, TE, and the conical
  Ag / Si cases of `gui-support-features.md` section 3) as a `validation-long` test
- [ ] later: hp-adaptivity for the conical solver (estimator of the mode equation), E_z
  resonances and band structures, scalar `ScatteringEz` (H1 only) when the DoF count matters

## M14 — Accuracy infrastructure for oblique incidence and gratings
From the same report (`spec-m14a/b/c`); A and C by the gpu agent, B by dev.
- [x] M14-A PML for oblique incidence (`PmlProfile::for_angle`, `PmlBox::max_resolution` /
  `resolution_limit` / `recommended_thickness(profile, p)`, under-resolution warning in
  `Scattering`, R0^(cos θ / 2) documented in `docs/theory/pml.md`; convergence test
  `flat_surface_fresnel`: Si and Ag half spaces at 10-70° against Fresnel; Python / project
  keys `theta_max` / `target`)
- [ ] M14-B hp-adaptivity on a plasmonic grating (Bloch + layered background + PML): marking
  symmetrised across periodic faces, estimator options, convergence test on the Ag grating
- [x] M14-C post-processing for gratings (`diffraction_orders` on an `OrderLine` of any
  orientation with the incident wave subtracted, complex vector amplitudes; `power_balance` /
  `absorbed_power(problem, solution)` of the total field; `Surface::plane`; `total_field` on
  stack interfaces by the cell side; `incident_wave` on the stack wave, setup and problem;
  convergence test `grating_postprocessing`: the report's Si grating to 1e-6 of the RCWA,
  lossless balance 2e-6; convergence test #6 moved to the layered background)

## M15 — Features requested by the GUI work (FEM model builder)
From the user's GUI work of 6 October 2026 (details, API proposals, acceptance data and the measurements
behind them in [`gui-support-features.md`](gui-support-features.md); IDs F0–F16 as there). The GUI builds
a periodic unit cell, meshes it with Gmsh and runs `ConicalScattering` / `Scattering2D`; every item
removes glue code or a workaround on its side. Priorities P1 > P2 > P3. Overlaps: F1 contains the
unchecked "hp-adaptivity for the conical solver" of M13, F16 the "marking symmetrised across periodic
faces" of M14-B.
- [x] F0 (P1) conventions of the conical API documented in `layered_conical_wave` (s = k × ŷ / |·|
  = (−sin φ, 0, cos φ), p = k̂ × s, phase 1 at the origin on the top interface, scaled against
  physical components, solver frame against the literature frame, `ky` is the normal
  wavenumber) and tested at five (θ, φ) pairs to 1e-10
- [x] F16 stage 1 (P1): symmetric refinement across a Bloch pair in the adaptive mesh
  (`AdaptiveMesh::set_periodic`, mirrored after the closure in `refine` / `hp_refine`; 2D
  and 3D unit tests, Python `set_periodic`)
- [ ] F16 stage 2 (P1): non-matching (mortar-type) Bloch coupling for the Nédélec and H1 traces
  with different levels and orders on the two faces, and the periodic facets in the residual
  estimator. Observed before stage 1: `bloch_constraints: 14 master facets but 16 slave
  facets` after a closure refinement on an unstructured Gmsh mesh
- [x] F1 stage 1 (P1) hp-adaptivity for `ConicalScattering`: `adaptivity::conical_residual_estimate`
  (residual of the coupled system with the β terms, PML tensors and the E_z / εE_z interface
  conditions), `ConicalScattering::estimate` / `error`, Python bindings and hp loop; convergence
  tests `conical_hp_corner` (re-entrant PEC corner at β = 1.3, exponential, b = 0.26) and
  `conical_grating_hp` (acceptance (a) and (d): Ag grating TM 50° from p = 4, ΔR−1 = −4e-6,
  ΔR0 = −2.2e-4 at 88 k DoFs, `validation-long`; docs/validation.md E)
- [ ] F1 stage 2 (P1): goal-oriented estimator for the diffraction orders (`GoalEstimate` with the
  conical vector amplitudes; the energy-norm loop leaves unmarked cells at their initial order,
  so the PML / air order has to be chosen by hand today), corner pre-refinement
  (`refine_at_corners`), generator `hpfem.adaptive_solve` that streams steps, acceptance (b) TE
  Ag 50° and (c) conical φ = 40°
- [x] dimensionally consistent Gauss-law terms in the three residual estimators
  (`EstimatorOptions::length_scale`, default ℓ = 1/k; in SI units the unscaled terms dominated η
  by 1/(kh)², docs/validation.md E)
- [ ] F2 (P1) one-call periodic scattering API (`hpfem.grating.solve`: stack interfaces snapped to
  mesh lines, PML from the largest order angle, orders in cover and substrate, power balance)
- [ ] F3 (P1) vectorised field sampling and triangulated field export as NumPy
  (`solution.sample(points)`, `solution.triangulate(subdivisions)`), for both solvers
- [ ] F4 (P1) exact absorbed power per material tag and per cell (`absorbed_power`,
  `absorption_density`) by volume quadrature
- [ ] F5 (P1) job runner / CLI with a stable JSON schema and JSON-lines events
  (`python -m hpfem.run job.json`), `hpfem.version_info()`
- [ ] F6 (P1) mesh module: unit-cell mesher (`hpfem.meshing`), `mesh.report()`,
  `mesh.check_periodic`, `read_gmsh` reading `$Periodic`
- [ ] F7 (P1) structured diagnostics (`problem.validate()`): interface off the mesh lines, untagged
  cells, missing periodic partner, under-resolved or thin PML, too few elements per wavelength,
  material outside its data range, lossy incidence medium, PEC wall too close in a lossy substrate
- [ ] F8 (P2) sweep acceleration: `LinearSolver.refactorize` reusing the symbolic analysis, affine
  assembly per material tag, `solve_sweep` with processes
- [ ] F9 (P2) progress callback, cancellation, timing breakdown, `estimate_memory`
- [ ] F10 (P2) conical equivalents of `diffraction_orders` / `power_balance` (flux based, complex
  vector amplitudes)
- [ ] F11 (P2) isolated scatterers for the conical solver: cross sections, far field, automatic
  closed measurement contour
- [ ] F12 (P2) H field and Poynting vector of the conical solution
- [ ] F13 (P2) dispersive materials directly in the setup (`setup.set_frequency`), explicit
  out-of-range policy, Drude–Lorentz fit helper
- [ ] F14 (P3) high-level eigenproblems on the periodic-cell front end (resonances, bands)
- [ ] F15 (P3) distribution: Windows/Linux wheels, `pip install hpfem[gui]`, `hpfem-gui` entry point

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
- [x] GPU backend for *hp*-adaptive systems: cuDSS pivots statically and perturbed tiny
  pivots on systems with hanging nodes and high orders even after the scaling to
  max |a_ij| = 1 (L-shape test: 103 of 16 359 and 2 639 of 19 723 pivots; plasmonic wedge
  28 / 1 591 / 16 356), so `kAuto` sent them to MUMPS / SparseLU. Resolved by the diagonal
  equilibration D A D (d_i = 1/√|a_ii|, row norm where the diagonal is tiny) inside the GPU
  library: 0 perturbed pivots in every step of both tests, convergence identical to the CPU;
  cuDSS matching and reordering options changed nothing, and accepting perturbed
  factorisations with iterative refinement had lost accuracy at 2 639 pivots. The refusal of
  perturbed factorisations and the CPU fallback stay as the safety net; a residual-checked
  acceptance remains the option should future systems still trip the static pivoting.
- [x] `physics::PropagatingMode<2>` on adaptive meshes: apply the hanging-node constraints
  as `Resonance` does (found during M10 validation; fixed in PR #63)

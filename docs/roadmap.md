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
- [x] gold sphere dimer (Hoffmann et al., Proc. SPIE 7390, 73900J, 2009; 80 nm spheres,
      1 nm gap, 632 nm, reference |E|² at the gap centre = 5.47624·10⁵ V²/m² for
      |E_inc| = 1 V/m) — computed under the documented assumption of Johnson & Christy gold
      (the source does not state its ε): `examples/gold_dimer` with the axisymmetric solver
      (orders m = 0, ±1), converged to five digits at 2.9724·10⁵ (−46 %); the ε scan shows the
      reference reproduced for ε ≈ −10.3 + 0.8i, a gold with 35 % lower loss, so the
      material datum, not the solver, sets the deviation — `docs/validation.md` G,
      `benchmarks/results/2026-10-08-validation-gold-dimer.json` (maintainer's decision of
      2026-10-08 to replace the five-digit comparison by the sensitivity statement)

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
- [x] the ring resonator with ports in 3D (SOI cross-section): done as stage A, the 3D
  directional coupler against coupled-mode theory (`examples/directional_coupler_3d` with the
  mesh / port builder, the coupled-mode reference and the CPU regression test by dev; the
  cuDSS production runs at p = 1–3 and 2–30 µm, `ScatteringOperator::solve_port` for one
  factorisation per S-matrix, `--cell-z`, the p-convergence table and `docs/validation.md`
  section F by gpu). The full ring in 3D (stage B) is in the backlog by the maintainer's
  decision of 2026-10-08
- [x] modal expansion by Riesz projection on the resonance solver (`physics::RieszProjection`,
  `AxisymmetricRieszProjection`: residues of the resolvent on circles around the
  quasi-normal modes plus a background contour, spectra of linear observables as sums over
  modes; verification against the direct solution and the micropillar Purcell spectrum)
  (`riesz_projection.hpp`, `test_riesz_projection.cpp`, `examples/micropillar_qd`,
  `docs/theory/maxwell.md#modal-expansion-by-riesz-projection`)
- [x] sensitivities, material derivatives of observables by the adjoint solve
  (`physics/sensitivity.hpp`: `adjoint_solution` / `material_sensitivity` and the conical
  pair, dQ/dε_tag = k0² ∫_tag E·z, holomorphic; `hpfem.grating.sensitivity` for the
  efficiencies; verified against finite differences to 1e-6)
- [x] sensitivities, shape derivatives (ADR-0011: the discrete adjoint on the mesh,
  `shape_gradient` / `shape_derivative` and the conical pair, parameters as mesh velocity
  fields, `region_normal_velocity`, `move_nodes`; `hpfem.grating.shape_sensitivity`;
  verified against finite differences of the solve on moved meshes in 2D, conical and 3D)

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
- [x] Python bindings (`ConicalScattering`, `layered_conical_wave`, orders); the project-file
  keys for the conical case are superseded by `hpfem.grating.solve` and the job runner
  (`hpfem.run`, M15 F2 / F5: `"polarisation": "s" | "p"`, `theta_deg`, `phi_deg`)
- [x] example (Si ridge in E_z, the R1 Ag case) as a job file under `examples/`
  (`examples/lamellar_grating/jobs/`: Si TM, Si conical, Ag TE = R1; regression test in
  `test_examples.py`)
- [x] conical grating validation at β ≠ 0 against the conical RCWA of the GUI work
  (`conical_grating_validation`: glass lamellar grating, s at θ = 40°, φ = 30° and p at
  θ = 50°, φ = 30° with the order m = −2 evanescent in air; 3.5e-5 and 4e-6 at p = 4,
  energy balance 1e-6)
- [x] the acceptance case R1 of the report (Ag lamellar grating, 50°, TE, and the conical
  Ag / Si cases of `gui-support-features.md` section 3) as a `validation-long` test
  (`conical_grating_hp`, cases (a) TM Ag, (b) TE Ag, (c) conical Si, M15 F1 stage 2;
  `docs/validation.md` section E)
- [x] hp-adaptivity for the conical solver (M15 F1: residual and goal-oriented estimators of
  the coupled system), E_z resonances and the open-cell band structure (M15 F14,
  `physics::ConicalResonance`, `grating.resonances` / `bands`)
- [x] scalar E_z path (`ConicalScatteringSetup::scalar_ez`: the H1 block alone at β = 0 with
  an E_z-only excitation, identical solution, `grating.solve(scalar="auto")`) in place of a
  separate `ScatteringEz` solver

## M14 — Accuracy infrastructure for oblique incidence and gratings
From the same report (`spec-m14a/b/c`); A and C by the gpu agent, B by dev.
- [x] M14-A PML for oblique incidence (`PmlProfile::for_angle`, `PmlBox::max_resolution` /
  `resolution_limit` / `recommended_thickness(profile, p)`, under-resolution warning in
  `Scattering`, R0^(cos θ / 2) documented in `docs/theory/pml.md`; convergence test
  `flat_surface_fresnel`: Si and Ag half spaces at 10-70° against Fresnel; Python / project
  keys `theta_max` / `target`)
- [x] M14-B hp-adaptivity on a plasmonic grating (Bloch + layered background + PML): done as
  M15 F1 stage 2 (`conical_grating_hp`, estimator options, the Ag grating to the hp reference)
  with M15 F16 (non-matching Bloch coupling) in place of the symmetrised marking
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
- [x] F16 stage 2 (P1): non-matching Bloch coupling for the Nédélec and H1 traces (facets
  grouped by overlap; the coarser facet carries the trace truncated to the common order, the
  finer facets interpolate it, exact for different levels and orders on the two faces; unrelated
  layouts by interpolation with a warning), `assembly::PeriodicLocator`, and the Bloch facets in
  the residual estimators (`periodic` argument); `test_periodic_nonmatching.cpp`, case (d) of
  `conical_grating_hp` (ridge off centre, faces refined independently). Observed before stage
  1: `bloch_constraints: 14 master facets but 16 slave facets` after a closure refinement on an
  unstructured Gmsh mesh
- [x] F1 stage 1 (P1) hp-adaptivity for `ConicalScattering`: `adaptivity::conical_residual_estimate`
  (residual of the coupled system with the β terms, PML tensors and the E_z / εE_z interface
  conditions), `ConicalScattering::estimate` / `error`, Python bindings and hp loop; convergence
  tests `conical_hp_corner` (re-entrant PEC corner at β = 1.3, exponential, b = 0.26) and
  `conical_grating_hp` (acceptance (a) and (d): Ag grating TM 50° from p = 4, ΔR−1 = −5e-6,
  ΔR0 = −2.2e-4 at 88 k DoFs, `validation-long`; docs/validation.md E)
- [x] F1 stage 2 (P1): goal-oriented estimator for the conical solver
  (`physics::conical_dwr_estimate`, `conical_point_functional`, `conical_order_functional`,
  `adaptivity::conical_weighted_residual`; convergence test `conical_goal_oriented`), corner
  pre-refinement `adaptivity::refine_at_points`, generator `hpfem.adaptive_solve` that streams
  steps and stops on a tolerance, acceptance (b) TE Ag 50° and (c) conical TM Si 50°/40° in the
  long variant of `conical_grating_hp` (docs/validation.md E)
- [x] dimensionally consistent Gauss-law terms in the three residual estimators
  (`EstimatorOptions::length_scale`, default ℓ = 1/k; in SI units the unscaled terms dominated η
  by 1/(kh)², docs/validation.md E)
- [x] F2 (P1) one-call periodic scattering API: `hpfem.grating.solve` (interfaces snapped onto
  mesh vertices with `Mesh::set_vertex`, PML from the largest propagating-order angle, measurement
  lines between structure and PML, reflected / transmitted orders with vector amplitudes,
  absorbed power per tag, power balance, `field(points)`, timing, `GratingError` diagnostics);
  `test_grating_solve.py` against the conical RCWA references. The six GUI presets of
  `fem_worker.py` are not in this repository; the worker can be reduced to JSON handling on top
- [x] F3 (P1) vectorised field sampling and triangulated field export as NumPy
  (`physics/field_sampling.hpp`: `sample_field` / `triangulate_field` with `SamplingOptions`
  for E, H and the Poynting vector, grouped per cell; `solution.sample(points)`,
  `solution.triangulate(subdivisions)` for both solvers)
- [x] F4 (P1) exact absorbed power per material tag and per cell (`physics/absorption.hpp`:
  `absorbed_power_by_tag`, `absorption_density`; `solution.absorbed_power()` with `by_tag` /
  `per_cell`) by volume quadrature
- [x] F5 (P1) job runner / CLI with a stable JSON schema and JSON-lines events
  (`python -m hpfem.run`, `hpfem.run.run_job`, `hpfem.version_info`, schema version 1 for grating
  jobs with sweeps, maps, cancellation; the six GUI presets are not in the repository)
  (`python -m hpfem.run job.json`), `hpfem.version_info()`
- [x] F6 (P1) mesh module: unit-cell mesher (`hpfem.meshing`), `mesh.report()`,
  `mesh.check_periodic`, `read_gmsh` reading `$Periodic`
- [x] F7 (P1) structured diagnostics (`hpfem.diagnostics`, `grating.validate`, run by
  `grating.solve`): interface off the mesh lines, untagged
  cells, missing periodic partner, under-resolved or thin PML, too few elements per wavelength,
  material outside its data range, lossy incidence medium, PEC wall too close in a lossy substrate
- [x] F8 (P2) sweep acceleration: `LinearSolver::refactorize` reusing the symbolic analysis
  (SparseLU, MUMPS, cuDSS), `physics::ConicalSweep` (affine operator per material tag with a
  pattern cache), `hpfem.sweep.solve_sweep`; 2.1× over 50 points on cuDSS
- [x] F9 (P2) progress callback, cancellation, timing breakdown, `estimate_memory`
- [x] F10 (P2) conical equivalents of `diffraction_orders` / `power_balance` (flux based, complex
  vector amplitudes)
- [x] F11 (P2) isolated scatterers for the conical solver: cross sections, far field, automatic
  closed measurement contour
- [x] F12 (P2) H field and Poynting vector of the conical solution (`curl_field`, `h_field`,
  `poynting` and the incident counterparts, also on the layered background)
- [x] F13 (P2) dispersive materials directly in the setup (`materials.DispersiveMap.apply(setup,
  omega)`, library names, numbers and core materials accepted; `grating.solve` takes dispersive
  dicts), explicit
  out-of-range policy, Drude–Lorentz fit helper
- [x] F14 (P3) high-level eigenproblems on the periodic-cell front end (resonances, bands)
- [x] F15 (P3) distribution: Windows/Linux wheels, `pip install hpfem[gui]`, `hpfem-gui` entry point

## M16 — Optimisation, calibration and uncertainty quantification
Proposal of 9 October 2026, revised after the review of PR #134; scope, rationale, API sketches
and validation plan in [`optimisation-uq-features.md`](optimisation-uq-features.md) (IDs S0–S9
as there). The adjoint sensitivities of M12 (material and shape) are the foundation. Everything
is Python (`hpfem.opt`) on the existing C++ core, except the eigenvalue derivatives of S4. Own
code only where the project has an advantage (gradients, Laplace via the Jacobian, DWR);
gradient-enhanced / multi-fidelity / multi-objective BO, NUTS and further UQ methods come from
mature libraries as optional extras (BoTorch, emcee, SALib) with an ADR note. Flagship
problems: a 2D grating and a metasurface unit cell (the 3D ring later). Priorities P1 > P2 > P3;
effort in focused sessions.
- [x] S0 (P1, ≈ 1) ADR-0012 (accepted, `docs/adr/0012-optimisation-and-uq.md`): scope of
  `hpfem.opt`, dependencies (NumPy/SciPy required, own light Gaussian-process code,
  BoTorch / emcee / SALib only optional), study file format, the evaluator contract and the
  **morphing-versus-remeshing strategy**: a reference mesh is morphed with `move_nodes`
  within a parameter range, remeshing only when the quality guard trips, so that the
  objective stays consistent with the adjoint gradient
- [x] S1 (P1, ≈ 4) gradient infrastructure: geometry parameters (radius, width, height,
  position) mapped to mesh velocity fields; Jacobian of several observables in the **direct
  (tangent) mode** (one solve per parameter) and the **adjoint mode** (one solve per
  observable) on the same factorisation, the cheaper one chosen from the counts; derivatives
  with respect to frequency and angle; mesh-quality guard; finite-difference checks on the
  morphed mesh (`LinearSolver::solve_transposed`, `KeptFactorisation`, residual derivatives
  and `grating.jacobian`, `hpfem.opt` parameters / `shape_velocity` / `Morph`,
  `conical_parameter_tangent` with the derivative of the Bloch constraints for `"theta"`,
  `"phi"`, `"omega"`, `"wavelength"`)
- [x] S2 (P1, ≈ 3) study framework: design space (continuous, integer, categorical,
  constraints), evaluation cache, resume, JSON-lines result store, job-runner integration with
  events and cancellation; batch proposals are evaluated sequentially, several processes only
  with an explicit thread budget (OpenMP, one cuDSS GPU, per-process factorisations on Windows) (`hpfem.opt.study`: `DesignSpace`, `Study`, `*.study.jsonl`;
  `hpfem.opt.evaluator`: `Evaluation`, `FunctionEvaluator`; events and cancellation as
  callbacks, the `hpfem.run` tasks follow in S9; `workers > 1` not yet)
- [x] S3 (P1, ≈ 4) classical optimisers and the Laplace approximation: L-BFGS-B with
  gradients, Nelder–Mead, differential evolution (SciPy wrappers), Gauss–Newton /
  Levenberg–Marquardt with the Jacobian of S1, parameter covariance from the Fisher
  information; end-to-end showcase: Si-grating reconstruction (CD, height, side-wall angle)
  with uncertainties from synthetic data (`hpfem.opt.minimize` / `fit` / `laplace`, MGH17 to
  1e-6; `GratingEvaluator`; `examples/grating_reconstruction`)
- [x] S4 (P2, ≈ 3–4) eigenvalue derivatives of resonances and bands (C++): non-Hermitian
  problem with PML and losses (left eigenvector; with Bloch / conical incidence the solution
  at −k), nonlinear eigenproblem for dispersive ε(ω) (extra dε/dω term), complex Q;
  verified against finite differences (`physics/eigen_sensitivity.hpp`,
  `grating.resonance_sensitivity` / `refine_resonance`, `physics/band_sensitivity.hpp`,
  `group_velocity`)
- [x] S5 (P2, ≈ 3) Bayesian optimisation: own light Gaussian process (Matérn ARD, noise),
  expected improvement and LCB, constraints, gradient-enhanced GP with the adjoint
  derivatives; multi-fidelity and multi-objective via the optional BoTorch extra. The DWR
  estimate is used as a fidelity indicator or for adaptive refinement until the error is small
  against the noise, **not** as independent GP noise (hypothesis, tested in S8) (`hpfem.opt.gp`,
  `hpfem.opt.bo`; the BoTorch drivers are experimental, not run yet)
- [x] S6 (P2, ≈ 2) parameter retrieval beyond Laplace: Bayesian least squares on the surrogate,
  MCMC through the optional emcee extra (posterior against the Laplace result)
- [x] S7 (P2, ≈ 2) uncertainty propagation and sensitivity analysis: linearised propagation
  from the Jacobian (own), Monte Carlo on the surrogate, Sobol' indices by one route (GP
  surrogate, SALib optional), fabrication-tolerance example (`hpfem.opt.uq`; SALib not
  needed: the own estimators are verified against the analytic Ishigami indices)
- [ ] S8 (P2, ongoing) validation: Branin / Rosenbrock for BO, Ishigami for Sobol', NIST MGH17
  for the reconstruction uncertainties, linear-Gaussian problems (MCMC = Laplace), gradient-based
  against gradient-free BO on the 2D grating and the metasurface unit cell, test of the DWR
  hypothesis (fidelity indicator against independent noise). Done (`docs/validation.md` H, I,
  `benchmarks/opt_validation.py`): BO with/without gradients against Nelder–Mead, differential
  evolution and L-BFGS-B on the Si grating, and the DWR estimate as fidelity indicator
  (effectivity, the κσ rule holds, DWR as noise rejected). Open: the metasurface unit cell,
  multi-fidelity BO with the DWR indicator against single-fidelity BO (needs BoTorch), and a
  rerun of H and I after the PML fix #156 (the studies are self-consistent: data and model
  share the PML)
- [ ] S9 (P3, ≈ 3) GUI and job schema: tasks `optimize`, `reconstruct`, `uq`; study view in
  `hpfem-gui` (history, Pareto front, Sobol' bars)

## Backlog / ideas
- [ ] the full ring resonator with ports in 3D (M12 stage B): needs an hp-mesh that is fine
  only across the cores and in the coupling region, p = 3 there for the resonance widths
  (the coarse axial mesh of the long couplers loses the phase as h_z^(2p), docs/validation.md F)
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

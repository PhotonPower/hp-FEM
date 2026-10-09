# Changelog
All notable changes to this project are documented here (Keep a Changelog, SemVer).

## [Unreleased]
### Added
- Milestone M16 (optimisation, calibration and uncertainty quantification) in the roadmap,
  with the specification `docs/optimisation-uq-features.md` (S0–S9).
- ADR-0012 (M16 S0): scope of `hpfem.opt` (NumPy/SciPy required; BoTorch, emcee, SALib as
  optional extras), the real-valued evaluator contract, morphing of a reference mesh with a
  quality guard, adjoint and direct sensitivities on a kept factorisation with
  `LinearSolver::solve_transposed` (amends ADR-0007), the JSON-lines study store and job
  schema version 2.
- `LinearSolver::solve_transposed` / `solve_transposed_many` (M16 S1, ADR-0012): Aᵀx = b on
  the factorisation of A for the adjoint sensitivities; LDLᵀ paths solve with A, SparseLU and
  MUMPS transpose their factors, cuDSS factorises Aᵀ on demand; Python bindings.
- Kept factorisation (M16 S1, ADR-0012): `keep_factorisation` in `ScatteringSetup` /
  `ConicalScatteringSetup` and `hpfem.grating.solve` keeps the factorised system in the
  solution (`physics::KeptFactorisation`): tangent solves A s = r and adjoint solves Aᵀz = q
  on full-size vectors, the adjoint the exact transpose of the solve (condensation, ports,
  Bloch / hanging-node constraints, scalar E_z path); `adjoint_solution` /
  `conical_adjoint_solution` overloads with the solution, used by the shape derivatives and
  `grating.sensitivity`; `StaticCondensation::condense_load_transposed` /
  `recover_transposed`.
- Direct mode of the sensitivities (M16 S1, ADR-0012): residual derivatives at fixed
  coefficients for materials (`material_residual_derivative`, conical) and shapes
  (`shape_residual_derivative`, conical: one directional difference per moving cell) and the
  functional term (`functional_shape_derivative`, conical), so that de/dp = A⁻¹r on the kept
  factorisation; `hpfem.grating.jacobian(result, parameters, observables, mode)` with the
  automatic choice between direct and adjoint mode.
- `hpfem.opt` (M16 S1, ADR-0012 §3): `MaterialParameter`, `GeometryParameter` (`field`,
  `trapezoid_parameters`: CD, height, side-wall angle), `shape_velocity` (exact boundary
  motion of a shape, fixed interfaces, harmonic extension), `Morph` (reference mesh moved to
  new parameter values, `cell_quality` guard with `MeshQualityError`).
- Frequency and angle derivatives (M16 S1, `physics/parameter_sensitivity.hpp`):
  `conical_parameter_tangent` (de/dθ for a parameter of the whole setup from the problems at
  θ ± h: residual derivative along the transported coefficients plus the derivative of the
  Bloch constraints (∂P)ᴴρ₀, one solve on the kept factorisation), `conical_residual`,
  `conical_transported_solution`, `KeptFactorisation::solve_many(loads, system_loads)`,
  `ConicalScattering::system_dofs` / `system_constraints`; `hpfem.grating.jacobian` columns
  `"theta"`, `"phi"`, `"omega"`, `"wavelength"` (with the explicit dependence of the order
  post-processing and dispersive materials), `GratingResult.inputs`.
- Study framework in `hpfem.opt` (M16 S2, ADR-0012 §2, §5, §7): `DesignSpace` with
  `Continuous` (optionally logarithmic), `Integer` and `Categorical` parameters (the S1
  parameters count as continuous), `LinearConstraint` / `NonlinearConstraint`, unit-cube
  encoding and Latin-hypercube / Sobol' sampling; the evaluator contract (`Evaluation`,
  `Evaluator`, `FunctionEvaluator`, complex observables split into Re / Im); `Study` with
  sequential evaluation, a cache keyed by the scaled parameters, the fidelity and the hash of
  the evaluator settings, the append-only JSON-lines store (`*.study.jsonl`: header,
  evaluation, proposal, state, remesh and note lines), resume without re-evaluation,
  recorded failures, progress events and cancellation (`hpfem.Cancelled`), history and best
  point.
- Optimisers and least-squares reconstruction (`hpfem.opt`, M16 S3): `minimize` runs
  L-BFGS-B (with the gradient from the evaluator's Jacobian), Nelder–Mead and seeded
  differential evolution on a `Study` (points cached, stored and replayed; cancel and resume
  from `state` checkpoints; failed points get a failure value; a remesh restarts L-BFGS-B);
  `fit` is our own Levenberg–Marquardt / Gauss–Newton on `W^{1/2}(y − y_meas)` with bounds by
  active set and projection; `laplace` gives the Laplace covariance `(JᵀWJ)⁻¹` (scaled by
  χ²_red when σ is unknown), standard errors, correlations and an `IdentifiabilityWarning`
  for rank-deficient or ill-conditioned problems, all in SI. Validated on NIST StRD MGH17
  from both starts: parameters and certified standard deviations to 1e-6.
- Scatterometry evaluator (`hpfem.opt.GratingEvaluator`, `Configuration`, M16 S3): the
  efficiencies of a grating under several measurement configurations as an evaluator of the
  `Morph` geometry parameters and `MaterialParameter`s, with the Jacobian along the morph
  velocities on the kept factorisation; PML boxes and measurement lines fixed on the
  reference mesh, a frequency-dependent stack, optional remeshing, the order as fidelity.
- `examples/grating_reconstruction` (M16 S3 showcase): CD, height and side-wall angle of a
  silicon line grating reconstructed from synthetic spectroscopic R0 data (s and p, 65°,
  400–700 nm, noise 0.002; data one order higher on a mesh at the true geometry) by
  Levenberg–Marquardt with Laplace uncertainties; README and regression test.

- Resonance derivatives (M16 S4, `physics/eigen_sensitivity.hpp`): dω/dp, dQ/dp and
  dλ/dp of quasi-normal modes from the left eigenvector — the mode itself for `Resonance`
  (complex symmetric pencil, also with hanging nodes), transposed inverse iteration on the
  Bloch-reduced pencil of `ConicalResonance` — for the permittivity of a tag, a mesh velocity,
  β and the Bloch wavenumber (the complex dispersion of leaky modes); the dispersive
  denominator `2ω/c0² − ∂λ̂/∂ω` (`resonance_derivative_from`). `ConicalResonance::constraints`
  / `reduced_pencil`. Python: the bindings, `hpfem.grating.resonance_sensitivity` and
  `hpfem.grating.refine_resonance` (Newton to the self-consistent resonance of dispersive
  models at the complex ω); `DrudeLorentz` and `Constant` accept a complex ω (analytic
  continuation). Verified against re-solved modes to six digits and against the exact
  Fabry–Pérot resonance of a (dispersive) slab.
- Band derivatives (M16 S4, written by the helper agent `bands`): `physics::band_permittivity_derivative`,
  `band_permeability_derivative`, `band_shape_derivative` (mesh velocity, V = 0 on the
  periodic faces) and `band_wave_vector_derivative` / `group_velocity` give dk0/dp and dω/dp
  of every band of a `BandStructure` without another eigensolve (Hellmann–Feynman on the
  Hermitian Bloch pencil; the wave-vector derivative differentiates the Bloch prolongation).
  Degenerate bands are resolved as clusters (one-sided branch slopes, `multiplicity`).
  `BandStructureSetup::keep_modes` keeps the M-normalised modes in `Bands::modes`. Python:
  `hpfem.band_*_derivative`, `hpfem.group_velocity`, `BandDerivative`. Verified against
  re-solved bands to 1e-8 and against c0(k+G)/(n|k+G|) in a uniform medium.

- Bayesian optimisation in `hpfem.opt` (M16 S5, written by the helper agent `bo`): own light
  Gaussian process `GaussianProcess` / `MultiOutputGP` (Matérn 5/2, 3/2 and squared-exponential
  ARD kernels, GLS constant mean, learned or fixed noise, hyperparameters by the log marginal
  likelihood with analytic gradients, derivative observations for a gradient-enhanced
  surrogate) and `bayesian_optimize` driving a `Study` (log expected improvement or lower
  confidence bound, known design-space constraints and `OutcomeConstraint`s by the probability
  of feasibility, `use_gradients=True` from the evaluator's Jacobian, failure policy, remesh
  counting, checkpoints with an exact resume after a cancellation; the DWR estimate is not used
  as noise, ADR-0012 §6); `pareto_front` / `non_dominated`; experimental BoTorch drivers
  (optional extra `opt-bo`, not run yet) `pareto_optimize` (qLogNEHVI) and
  `multi_fidelity_optimize`. Branin to 1e-3 within 40 evaluations; gradient-enhanced BO reaches
  Hartmann-3 to 1e-3 in 8–13 evaluations where plain BO needs 20 or more.
- Posterior beyond the Laplace approximation in `hpfem.opt` (M16 S6): `sample` (MCMC with emcee,
  optional extra `opt-mcmc`, also in `dev`) on the posterior of a `fit` with the χ²_red noise
  scaling of `laplace`, uniform prior on the bounds and optional Gaussian priors;
  `build_surrogate` (gradient-enhanced `MultiOutputGP` per observable from Latin-hypercube
  evaluations with the Jacobian around the optimum, its variance in the likelihood, a
  validation error in noise standard deviations); `PosteriorResult` with `compare()` against
  Laplace. Linear-Gaussian, skewed (against a grid posterior) and surrogate-vs-direct tests;
  `examples/grating_reconstruction/run.py --posterior`.
- Uncertainty propagation and sensitivity analysis in `hpfem.opt` (M16 S7): `Normal` /
  `Uniform` inputs, `linear_propagation` (delta method with the Jacobian, variance shares),
  `build_global_surrogate` (gradient-enhanced GP per observable over the input box with active
  learning), `monte_carlo` and `sobol_indices` (Saltelli first-order, Jansen total, bootstrap
  intervals) on the surrogate or any function; verified on the Ishigami function (analytic
  indices) and closed-form linear models. `examples/fabrication_tolerance`: CD / height /
  side-wall angle tolerances of a silicon grating propagated to its reflectance spectrum.

### Fixed
- Shape derivatives of grating efficiencies along a mesh velocity that deforms the cells at a
  measurement line lying on mesh facets (the default midway line of structured cells): the
  order amplitude sampled on facets is not differentiable there (the normal Nédélec
  component jumps across facets), the Jacobian of a p-polarised line grating was off by a
  factor of ten for the height and by 1e-3 for the side-wall angle. `shape_velocity` / `Morph`
  take `band=(y_low, y_high)` to keep the lines and the PML out of the deformation;
  `grating.jacobian` and `grating.shape_sensitivity` refuse such velocities
  (`grating.deformed_line_cells`). The S1 morph test now uses the band and checks the
  derivatives to 1e-6 instead of 1e-3 (the "flaky" failure under load was this).
- `hpfem.opt.Morph` was linear in the parameters (`x_ref + Σ (p − p_ref) V_p`, ADR-0012 §3):
  exact for each parameter alone but not jointly (the wall of a trapezoid moves with
  `h cot α`), which biased the side-wall angle of the reconstruction showcase by 2.5 standard
  errors for every noise realisation. The boundary nodes now go to their place on the exact
  changed shape and the rest follows by the same harmonic extension; `velocity_at(values,
  name)` is the velocity at the evaluated point (used by `GratingEvaluator`); ADR-0012 §3
  amended. An unbounded side-wall angle gets the typical magnitude 1 rad (the period made its
  difference step 4e-13 rad, lost in the round-off of the node coordinates).

## [0.4.0] — 2026-10-08
Fourth release: the GUI support milestone M15 complete, the sensitivities of M12, and the
distribution. Highlights: hp-adaptivity for the conical solver with the residual and the
goal-oriented (dual-weighted residual) estimators, Bloch-periodic coupling of non-matching
meshes (`assembly::bloch_constraints`: the coarse trace truncated to the common order,
the fine side interpolated), and the Ag
grating of the acceptance test at 5·10⁻⁴ (F1, F16); the one-call grating API
`hpfem.grating.solve` with PML design, snapping, diagnostics, flux balance, dispersive
materials and the scalar E_z path, the unit-cell mesher on Gmsh, the job runner with a JSON
schema and JSON-lines events, progress callbacks, cancellation, timing and a memory estimate
(F2, F5–F7, F9, F13); the conical post-processing (orders on any line, cross-sections, far
field) and the conical resonances with the periodic-cell eigen front end (F10, F11, F14);
the field sampling, exact absorption, H field / Poynting vector and the sweep acceleration
of the gpu agent (F3, F4, F8, F12); material and shape sensitivities by the adjoint solve
(ADR-0011, verified against finite differences); the 3D directional coupler with modal
ports against coupled-mode theory on the cuDSS backend; the gold-dimer benchmark under the
documented Johnson & Christy assumption; and the wheels workflow (manylinux and Windows
wheels for the python.org CPython, `pip install hpfem[gui]`, `hpfem-gui`). CI runs in about
30 minutes with OpenBLAS for MUMPS and `-O1` under ASan.
### Changed
- The Gauss-law terms of the three residual estimators (`residual_estimate`,
  `axisymmetric_residual_estimate`, `conical_residual_estimate`) are multiplied by the
  squared length scale ℓ² = 1/k² (`EstimatorOptions::length_scale`, 0 = 1/k from `k_squared`):
  the divergence residual carries one inverse length more than the curl–curl residual, so in
  SI units the unscaled terms dominated η by 1/(kh)² ≈ 1e15 (docs/validation.md E). Problems
  with k = 1 (the L-shape tests) are unchanged; η of SI-scale problems is now a usable error
  measure and the marking weighs the Maxwell and Gauss residuals alike.

### Added
- `examples/gold_dimer` (M10 benchmark of Hoffmann et al. 2009 under the documented
  Johnson & Christy assumption): the 80 nm gold dimer with a 1 nm gap on a Gmsh meridian mesh,
  the axisymmetric solver for the orders m = 0, ±1 of the plane wave perpendicular to the
  axis, |E|² at the gap centre converged to five digits in p (2.9724e5 against the published
  5.47624e5), the permittivity sensitivity by finite differences and an ε scan locating the
  reference at ε ≈ −10.3 + 0.8i; `docs/validation.md` section G, benchmark record,
  regression test (needs gmsh).
- `examples/directional_coupler_3d` (M12, stage A of the 3D ring resonator with the gpu
  agent): two SOI strip waveguides between four 3D modal ports on half faces, structured
  box mesh with tagged cores, the coupled-mode reference κ = (β_even − β_odd) / 2 from
  `PropagatingMode` on the two-core section, `run.py --length/--gap/--order/--cell/--backend`
  with a results JSON, README and a quick CPU regression test (50 k DoFs, 3 s, lossless to
  1e-5).
- Production runs of the 3D directional coupler on cuDSS and their validation (M12 stage A,
  `docs/validation.md` section F, `benchmarks/results/2026-10-08-VR-directional-coupler-3d.json`):
  p = 1–3 at 2 µm and p = 2 at 10, 20 and 30 µm against coupled-mode theory (P_cross to 1e-5 at
  p = 3, supermode phase errors 5e-4 rad; 1.3–2 M DoFs in the hybrid memory mode);
  `run.py --cell-z` / `Geometry.cell_z` for a separate axial cell size (the phase error of the
  coarse axial mesh grows as h_z^(2p)).
- `physics::s_parameters` with one factorisation per S-matrix:
  `ScatteringOperator::solve_port(port, mode)` (the port terms are part of the operator,
  `Scattering::add_port_terms` is public), 3.5× faster on the coupler; iterative refinement in
  the cuDSS solve (`CUDSS_CONFIG_IR_N_STEPS`, env `HPFEM_GPU_IR_STEPS`, GPU API v5 with
  `hpfem_gpu_refactorize`).
- Sweep acceleration (M15 F8): `LinearSolver::refactorize` reusing the symbolic analysis
  (SparseLU `analyzePattern`, MUMPS job 2, cuDSS refactorisation phase), `physics::ConicalSweep`
  (affine assembly per material tag on a cached union pattern, PML map and reduced scatter
  targets, parallel `assemble_conical` over cells, scatter-based `Constraints::reduce`),
  `hpfem.sweep.solve_sweep`; 2.1× over 50 frequency points (197 k DoFs, cuDSS).
- H field and Poynting vector of the conical solution (M15 F12): `curl_field`, `h_field`,
  `poynting`, `incident_curl` / `incident_h_field` of the conical plane wave and of the
  layered conical wave (`LayeredConicalWave::field_curl`), `ConicalScatteringSetup::incident_curl`.
- Exact absorbed power per tag and per cell (M15 F4, `physics/absorption.hpp`):
  `absorbed_power_by_tag` (`AbsorbedPower{total, by_tag, per_cell, of_tag()}`),
  `absorption_density` (`AbsorptionDensity<Dim>`) by volume quadrature; Python
  `solution.absorbed_power(by_tag=, per_cell=)`.
- Vectorised field sampling and triangulated export (M15 F3, `physics/field_sampling.hpp`):
  `sample_field` (`SamplingOptions{quantity, scattered, bloch_wrap, interface_side}` for E, H
  and the Poynting vector, points grouped per cell, allocation-free kernels) and
  `triangulate_field` (`TriangulatedField<Dim>`); Python `solution.sample(points)` /
  `solution.triangulate(subdivisions)` returning NumPy arrays.
- Shape derivatives by the discrete adjoint on the mesh (M12, ADR-0011,
  `physics/shape_sensitivity.hpp`): `shape_gradient` / `conical_shape_gradient` (dQ/dx of
  every geometry node by central differences of the element integrals, in parallel),
  `shape_derivative` / `conical_shape_derivative` (one adjoint solve, the gradient paired
  with a mesh velocity, plus the derivative of the functional along it), parameters as mesh
  velocity fields (`region_normal_velocity`, `move_nodes`, `num_geometry_nodes`),
  `assembly::element_conical` factored out of the conical assembler; Python bindings and
  `grating.shape_sensitivity(result, velocity, order)`; verified against finite differences
  of the solve on moved meshes (disc radius in-plane and conical on a second-order mesh, ball
  radius in 3D, ridge height of the glass grating).
- Material sensitivities by the adjoint solve (M12, `physics/sensitivity.hpp`):
  `adjoint_solution` / `material_sensitivity` for `Scattering<Dim>` and
  `conical_adjoint_solution` / `conical_material_sensitivity` for the conical solver, the
  holomorphic derivative dQ/dε_tag = k0² ∫_tag E·z of any linear goal (constraints and PEC
  handled as in the DWR estimator), verified against finite differences; Python bindings,
  `grating.sensitivity(result, tag, order, side)` for the derivatives of an efficiency with
  respect to Re ε and Im ε; `ConicalScattering.transverse_dofs` / `longitudinal_dofs`.
- Scalar E_z path of the conical solver (`ConicalScatteringSetup::scalar_ez`): at β = 0 with
  an E_z-only excitation only the H1 block is constrained, factorised and solved (about a
  third of the DoFs), the in-plane coefficients are zero and all post-processing applies;
  `hpfem.grating.solve(scalar="auto")` takes it for s at φ = 0 (`result.scalar`). Replaces
  the planned scalar `ScatteringEz` solver.
- Job files of the grating acceptance cases (`examples/lamellar_grating/jobs/`: Si TM 50°,
  Si conical 50° / 40°, Ag TE 50° = case R1) with the RCWA references in the README and a
  regression test; the Ag E_z case agrees to 3e-6 at uniform p = 4.
- Distribution (M15 F15): `.github/workflows/wheels.yml` builds the sdist, `manylinux_2_28`
  x86_64 wheels (cibuildwheel) and Windows `win_amd64` wheels for the python.org CPython with
  the MSYS2 UCRT64 GCC (SparseLU build, static GCC runtime) for CPython 3.10–3.13, attaches
  them to `v*` releases; `pip install "hpfem[gui]"` (Streamlit, Gmsh, Matplotlib, meshio,
  PyYAML) and the `hpfem-gui` entry point (`hpfem.gui`) that starts the Streamlit app with
  the interpreter that has hpfem.
- Conical resonances and the periodic-cell eigen front end (M15 F14):
  `physics::ConicalResonance` (`conical_resonance.hpp`: the pencil S(β) − k0² M with PEC,
  PML, Bloch constraints and the gradient kernel projected out, modes with complex ω, Q and
  block coefficients, `field` / `h_field` / `poynting`, `sample_field` / `triangulate_field`
  overloads), `conical_material_form`; Python `hpfem.ConicalResonance`,
  `grating.resonances` / `grating.bands` (`ResonanceResult`, `ResonantMode.field`), job-runner
  tasks `resonances` and `bands` (events `mode` / `point`, maps per mode); unit tests (PEC
  square waveguide at β ≠ 0, Bloch strip), convergence test `conical_resonance` (Fabry–Pérot
  slab between PMLs), docs (maxwell.md "Conical resonances", python.md).
- Progress, cancellation, timing and memory estimate (M15 F9): `core/progress.hpp`
  (`ProgressCallback`, `ProgressEvent`, `Timing`, `Cancelled`, `ProgressReporter`);
  `ScatteringSetup::progress` / `ConicalScatteringSetup::progress` called at every phase of
  `solve` (assembly, constraints, factorisation, solve, post, done), cancellation between the
  phases, `solution.timing` with the seconds per phase; `solvers::estimate_memory` (DoFs,
  nonzeros, factor entries and bytes from fitted fill-in laws, record
  `benchmarks/results/2026-10-08-fill-in.jsonl`), `LinearSolver::factor_entries()`;
  Python: `hpfem.ProgressEvent`, `hpfem.Cancelled`, `hpfem.estimate_memory`,
  `grating.solve(progress=, cancel=)`, `grating.estimate_memory`, job-runner events
  `estimate` and `progress`, cancellation inside a solve, `timing` per point.
- Conical post-processing (M15 F10 / F11, `physics/conical_postprocess.hpp`):
  `conical_diffraction_orders` on any `OrderLine`, `to_literature_frame` /
  `from_literature_frame`, `conical_curl_of`, the flux-based `conical_power_balance`
  (reflected field through a cover line, transmitted through a substrate line, volumetric
  absorption; `grating.solve` reports it as `result.flux_balance`), `conical_cross_sections`
  (σ_sca by flux, σ_abs volumetric, σ_ext) and `ConicalFarField` (far field of the 2.5D field
  with the transverse wavenumber k_t, 3D Stratton–Chu integrated along z; equals `FarField<2>`
  at β = 0 and the flux at β ≠ 0); Python bindings; convergence test `conical_cross_sections`
  (Mie cylinder E_z and in-plane to the series by flux and far field).
- Dispersive materials in the setup (M15 F13): `materials.DispersiveMap` (models, core
  materials, library names or numbers by tag; `at(omega)` freezes a `MaterialMap`,
  `apply(setup, omega)` sets frequency and materials in one call, `range` of the tabulated
  data), the out-of-range policy of `Tabulated` (`"error"` / `"clamp"` with a warning,
  `materials.with_policy`), and `materials.fit_drude_lorentz` (passive Drude + n Lorentz poles
  fitted to tabulated n, k by least squares in log parameters, with the maximal relative error).
- Job runner (M15 F5): `python -m hpfem.run job.json` runs a grating job from a JSON document
  (schema version 1: model, mesh (structured / gmsh / file), materials, stack, incidence, sweep,
  solver, maps) with JSON-lines events (start, mesh, diagnostics, point, map, cancelled, error,
  done), `results.json` and `maps_<i>_<j>.npz`, cooperative cancellation by SIGTERM / SIGINT or a
  cancel file; `hpfem.run.run_job` with callbacks, `hpfem.version_info()`, and
  `hpfem.meshing.structured_unit_cell` (a Gmsh-free structured mesher of a `UnitCell`).
- Structured diagnostics (M15 F7): `hpfem.diagnostics` with `Diagnostic(code, severity, text,
  hint)` and the checks of the GUI wish list (invalid curved cells, poor angles, untagged cells,
  tags without material, stack interfaces off the mesh lines, missing periodic partners and
  non-identical faces, under-resolved or thin PML, too few elements per wavelength for the
  order, tabulated materials outside their range, lossy incidence medium, grazing orders, a PEC
  wall within six decay lengths in a lossy substrate); `grating.validate` and
  `diagnostics.validate_scattering` run them, `grating.solve` runs them first (`check`) and
  keeps the warnings in `result.diagnostics`; dispersive materials are accepted by
  `grating.solve`.
- Mesh module (M15 F6): `hpfem.meshing` builds grating / metasurface unit cells with the Gmsh
  Python API (`UnitCell` with slabs and rectangle / trapezoid / ellipse / polygon shapes copied
  by the period and clipped, material tags by priority, side physical groups, `$Periodic`,
  element sizes per tag from the wavelength and the metal decay length, interface refinement,
  curved cells for ellipses); `mesh::report` / `mesh_report` (counts, angles, aspect ratio,
  edge lengths, invalid curved cells, untagged cells, tags, hanging entities) and
  `mesh::check_periodic`; `read_gmsh_with_periodic` / `read_gmsh_periodic` read the `$Periodic`
  section into (master, slave, shift) links. `gmsh` is a dev extra; the meshing test skips
  without it.
- `hpfem.grating.solve` (M15 F2), the one-call periodic scattering API on the conical solver:
  snapping of the stack interfaces onto mesh vertices (`Mesh::set_vertex`, bound as
  `set_vertex`), PML designed from the largest propagating-order angle (`PmlProfile.for_angle`,
  reference index `min(n_cover, n_substrate)`), measurement lines between structure and PML,
  `GratingResult` with reflected / transmitted orders (efficiencies, vector amplitudes,
  wavenumbers), absorbed power per tag, power-balance residual, `field(points)` and timing;
  `GratingError` names straddling cells and bad options; `python/tests/test_grating_solve.py`.
- Non-matching Bloch-periodic coupling (M15 F16 stage 2): `assembly::bloch_constraints` no
  longer requires identical meshes on the two sides. Facets are grouped by overlap after the
  shift; the coarser facet of a group carries the trace of the coupled space truncated to the
  lowest order in the group (minimum rule), its surplus modes are zero and the finer facets'
  DoFs interpolate the trace, which is exact for different orders and for one-sided refinement
  (either side finer); unrelated layouts fall back to interpolation of the master trace (warning).
  `assembly::PeriodicLocator` finds the master cell under a point of a slave facet, and the
  residual estimators (`residual_estimate`, `conical_residual_estimate`, new `periodic`
  argument passed by `Scattering::estimate` / `ConicalScattering::estimate`) include the jump
  across the Bloch faces, credited to both sides. `AdaptiveMesh::set_periodic` is now optional;
  the grating hp test runs case (d) with the ridge 25 nm off centre and independently refined
  faces, and `hpfem.adaptive_solve` drops the order equalisation. A shift that moves the
  slave facets off the master line is still rejected.
- Goal-oriented hp-adaptivity for the conical solver (M15 F1 stage 2):
  `physics::conical_dwr_estimate` (DWR with the adjoint of the enriched coupled system, Bloch
  and hanging constraints, PEC) with `conical_point_functional` and `conical_order_functional`
  (vector amplitude of a diffraction order along a polarisation vector, the linearised
  efficiency), `adaptivity::conical_weighted_residual` (identity Σ r_K(W) = ℓ(W) − a(E_h, W)
  verified), an H1 `assembly::point_functional`, pre-refinement `adaptivity::refine_at_points`,
  the Python generator `hpfem.adaptive_solve` (streams DoFs, η, observables, goal error; stops on
  a tolerance; keeps paired Bloch cells at equal orders) and the bindings; convergence test
  `conical_goal_oriented` (goal error 35× below the energy-driven loop at 8 k DoFs); the
  acceptance cases (b) TE Ag 50° and (c) conical TM Si 50°/40° in `conical_grating_hp`.
- hp-adaptivity for the conical solver (M15 F1 stage 1): `adaptivity::conical_residual_estimate`,
  the residual estimator of the coupled 2.5D system (Cartesian curl with ∂_z = iβ, tangential
  and normal-flux jumps including the E_z / εE_z interface conditions, hanging facets),
  `ConicalScattering::estimate` / `error` with Python bindings (`ConicalError`); convergence
  tests `conical_hp_corner` (manufactured gradient mode at a re-entrant PEC corner, β = 1.3,
  effectivity 3.3–5.2, exponential in N^{1/3}) and `conical_grating_hp` (Ag lamellar grating,
  TM 50°, against the F1 references: ΔR−1 = −5e-6 and ΔR0 = −2.2e-4 at 88 k DoFs in the long
  variant, uniform meshes stagnate at 5e-3); the hp loop from Python in `test_conical.py`.
- `mesh::AdaptiveMesh::set_periodic` (M15 F16 stage 1): refinement is mirrored across declared
  Bloch face pairs after the closure, so the leaf facets of both faces stay identical and the
  Bloch constraints can be rebuilt after every adaptive step (the GUI's workaround
  `symmetrise_periodic` becomes unnecessary); unit tests in 2D and 3D, Python binding.
- Conical validation and conventions (M13, M15 F0): `conical_grating_validation` checks the
  conical solver at β ≠ 0 against the conical RCWA references of `docs/gui-support-features.md`
  (glass lamellar grating: s at 40°/30° to 3.5e-5, p at 50°/30° to 4e-6 including the order that
  is evanescent in air but propagates in glass; energy balance 1e-6); the s / p amplitude,
  phase and frame conventions of `layered_conical_wave` are documented and tested.
- Conical incidence and the E_z polarisation (M13, from the user test report): the 2.5D
  solver `physics::ConicalScattering` for z-invariant structures with the longitudinal
  wavenumber β (in-plane E in Nédélec, v = −i E_z in H1, `assembly::assemble_conical` with the
  exact gradient kernel; PEC, Bloch on both spaces, PML tensors, hanging nodes, scattered- and
  total-field formulation, layered background through `layered_conical_wave` from the 3D
  stack solution), `conical_plane_wave` / `conical_polarisation`, the conical Poynting flux,
  Fourier coefficients along any line and diffraction efficiencies with complex vector
  amplitudes. At β = 0 it is the missing E_z ("TE") polarisation of 2D gratings. Unit tests,
  the convergence tests `conical_mie_cylinder_ez` (series value to 4e-5 at p = 3) and
  `conical_lamellar_grating_ez` (in-test TE RCWA, |ΔR| 2e-6 and |ΔT| 2e-5 at p = 4 with the
  substrate as layered background), theory section with the caveat on E_z fluxes at
  material interfaces.
- Grating post-processing (M14-C, from the user test report): `physics::diffraction_orders`
  takes the orders on an `OrderLine` of any orientation (origin as phase reference, tangent
  along the period, normal away from the structure) by composite Gauss-Legendre blocks and
  subtracts an incident wave, so the reflected orders of a total field on a `LayerStack`
  background follow from the new `Scattering::incident_wave` (`LayeredPlaneWave::incident_wave`,
  `ScatteringSetup::incident_wave`); the result carries the complex vector amplitude of every
  order. `Surface::plane` builds flux surfaces on coordinate planes, `power_balance` gives
  the energy balance of a periodic problem (incident power per period, reflected flux of
  total minus incident wave, transmitted flux, absorbed power of the total field by
  `absorbed_power(problem, solution)`) with the relative residual as reference-free quality
  indicator, and `total_field` on an interface of the layered background evaluates the side
  of the located cell. The silicon lamellar grating of the report gives R0 = 0.143380 and
  R-1 = 0.142383 at p = 5 against the RCWA 0.143381 / 0.142382 (the report's +1.2e-4 was its
  extraction), balance residual 1.7e-6; the lossless grating closes R + T = 1 to 2e-6 once
  the PML is designed for the steepest propagating order on its side. Convergence test #6
  now runs on the layered background with the orders from `diffraction_orders` (its
  substrate PML was under-resolved, |ks|h = 9.4: two units at R0 = 1e-6 bring the
  transmitted orders from 1e-3 to 2.6e-4 at p = 3). Python bindings, project keys
  `outputs.diffraction.line` / `balance`, theory section in `docs/theory/maxwell.md`.
- Angle-aware PML design (M14-A, from the user test report): `PmlProfile::for_angle(theta_max,
  target, r_amplitude)` chooses the normal-incidence round-trip reflection R0 so that the
  field reflected by the far wall at the angle theta, R0^(cos theta / 2), stays below the
  target for a structure of amplitude reflection r; `PmlBox::max_resolution(h, n)` reports
  the largest |k s| h of the layers, `PmlBox::resolution_limit(p)` the rule of thumb (3 for
  p >= 4, 0.75 p below) and `recommended_thickness(k0, n, h, profile, p)` the thickness that
  keeps a profile resolved; `Scattering` warns when a PML is under-resolved with the index
  meshed in the layer. Convergence test `flat_surface_fresnel` (silicon and silver half
  spaces at 10-70 degrees against Fresnel), `docs/theory/pml.md` section on oblique
  incidence, Python `PmlProfile.for_angle` in degrees and the project keys
  `pml.profile.theta_max` / `target` with a warning for steep angles on a plain reflection.
- Modal expansion by Riesz projection (`physics::RieszProjection<Dim>`,
  `AxisymmetricRieszProjection`, engine `RieszProjectionBase` on a `RieszPencil`): the
  residues of the frozen-PML resonance pencil from trapezoidal circles around the listed
  poles, group contours, and an ellipse background whose integrand has the modal parts
  subtracted (otherwise the enclosed poles cap the convergence rate), each with a half-rule
  convergence check; `spectrum()` / `expand()` / `direct()` reconstruct a solution from the
  modal and background parts. Tests: strip identity to 1e-8 at 20 frequencies, 0.4^N
  convergence, sphere in 2.5D; Python bindings; the micropillar example's `modal_spectrum()`
  gives a Purcell factor of 5.03 against 5.14 of the sweep (2 % from the frozen PML).
- Batched loads and Dirichlet data for sweeps: `assembly::assemble_maxwell_loads` assembles
  the loads of all incident fields in one pass over the cells (geometry, quadrature and
  basis functions once per cell, cell colouring instead of per-thread copies of the n x k
  result), the hierarchical interpolation samples several functions at once
  (`interpolate` / `tangential_dirichlet_values` on a span of functions,
  `Scattering::dirichlet_many`), and `ScatteringOperator::solve_many` uses both: the
  100-angle sweep at 194 k DoFs takes 0.78 s instead of 3.2 s on cuDSS, the same saving on
  every backend; columns agree with the single-field paths to 1e-14. The measurement behind
  it (`bench_sweep_shares`: device-resident sweep vectors would have saved 0.1 s) and
  `HPFEM_GPU_TIMING=1` (upload / solve / download split of a GPU solve) are recorded;
  the Linux build of the GPU library is prepared in `gpu/` (RPATH, README) but untested.
- hp-adaptivity on the meridian plane (M11, last item): `adaptivity::axisymmetric_residual_estimate`
  (the r-weighted residual estimator of the mode equation, cylindrical curl and divergence,
  hanging facets), hanging-node constraints in `AxisymmetricCavity`, `AxisymmetricResonance`
  and `AxisymmetricScattering` (block constraints restricted to the free DoFs, gauge gradient
  by restriction), `axisymmetric_error` and `AxisymmetricScattering::estimate` / `error`;
  unit tests, the convergence test `axisymmetric_hp_corner` (manufactured re-entrant PEC edge,
  error ~ exp(-0.27 N^(1/3)), effectivity 3.5-5.6) and Python bindings. M11 is complete.
- Shift-invert Arnoldi with the Krylov basis on the device: `solvers::DeviceArnoldi` (GPU
  library API version 4, `hpfem_gpu_arnoldi_*` and rectangular device matrices) keeps the
  basis on the GPU, computes w = P K^-1 B v_j there (cuDSS solve, device products, the gauge
  projection with a second cuDSS factorisation), orthogonalises by classical Gram-Schmidt
  applied twice with reduction and update kernels and forms restart and Ritz vectors as
  V_m c; `complex_eigenpairs_near` / `complex_eigenpairs_near_gauged` run on a `KrylovBasis`
  abstraction (host and device, same algorithm) and take the device whenever the shift is
  factorised by cuDSS and the memory estimate fits (`HPFEM_GPU_ARNOLDI=0` keeps the host
  basis). Measured with 24 Krylov vectors, factorisation included: `Resonance` 369 k DoFs
  5.5 s to 3.5 s, gauged `BandStructure` 369 k DoFs 9.2 s to 6.4 s against the host basis
  with cuDSS solves (`bench_device_arnoldi`).
- Python: `hpfem.DeviceMatrix`, `hpfem.DeviceStepper`, `LinearSolver.backend` and
  `complex_eigenpairs_near_gauged`; `python/tests/test_gpu.py` (skipped without the GPU
  library).
- `physics::scatter_orders` / `oblique_plane_wave` / `superpose_far_field` (M11 backlog):
  oblique incidence on bodies of revolution by the sum over the azimuthal orders
  (Jacobi–Anger expansion of the plane wave, s and p polarisation, power-based stopping
  criterion, far-field superposition); sphere at 50° reproduces the Mie cross-section
  with exponential convergence in p; Python bindings.
- Waveguide ports with modal excitation and S-parameters in 2D (M12): `physics::WaveguidePort`
  on `ScatteringSetup::ports`, `PortModes<2>` (TM slab modes of the port cross-section by a
  hierarchical 1D p-FEM on the port edges, bi-orthogonal functionals and powers, orientation-
  independent sign convention), the low-rank modal boundary term and excitation in
  `Scattering::solve`, `Scattering::port_coefficients` and `physics::s_parameters`
  (power-normalised S-matrix, one solve per propagating channel); unit tests (parallel-plate
  modes, slab dispersion, straight guides: transmission phases, no reflection, symmetric
  unitary S), the convergence test `waveguide_port` (S21 error 1e-2 → 2e-11 and |S11| 6e-3 →
  2e-12 for p = 1…5), Python bindings and test, theory section
  `docs/theory/maxwell.md#waveguide-ports-and-s-parameters`.
- 3D waveguide ports (M12): `PortModes<3>` extracts the planar port cross-section as a 2D
  mesh in the port frame (tags, orders, PEC rim), takes its guided modes from
  `PropagatingMode` and builds the modal boundary term from
  n × (μ⁻¹ curl E) = (∇ₜE_z − iβEₜ)/μᵣ with the surface quadrature of the port facets;
  `Scattering<3>` with ports and `s_parameters<3>`. `PropagatingMode` drops the numerically
  zero eigenvalue of the gradient kernel by a relative threshold (it slipped through as a
  "guided" mode with β ≈ 1e-6). Unit tests (rectangular waveguide TE10: β, mode field and
  sign on both ports, S21 = e^{iβL}), the convergence test `waveguide_port_3d`, Python
  bindings (`PortModes3D`, `transverse_field`) and test, theory section.

### Fixed
- The MinGW Python extension is linked without debug info (`-Wl,--strip-debug`): with the
  release-with-asserts flags the module had grown past half a gigabyte and the Windows loader
  refused it ("not a valid Win32 application"); it is now 17 MB.
- A layered background rejected meshes whose lines sit on the interface only up to rounding
  (defect D2 of the user report): the interface tolerance of `Scattering` and
  `ConicalScattering` was relative to the stack thickness, which is zero for a bare substrate;
  it is now relative to the mesh extent along the normal.

## [0.3.0] — 2026-10-04
Third release: the backlog of 0.2.0 and two new milestones. Highlights: the axisymmetric
(2.5D) solver of ADR-0010 — eigenmodes, quasi-normal modes with the cylindrical PML, scattering
by the axial plane wave, dipole sources on the axis and the far field, each verified against an
analytic reference (Bessel zeros, Mie poles, Mie cross-section, Larmor power), with the
micropillar quantum-dot example; the GPU backend of ADR-0008 — cuDSS direct solver behind
`LinearSolver` as a separately built C-ABI library (nvcc + MSVC, cuDSS 0.8; without it 0.3.0
behaves as before), `solve_many`, automatic `kAuto` selection above 10 000 unknowns,
complex-symmetric LDL^T with symmetry detection, hybrid memory for factors beyond the device
memory, diagonal equilibration for hp-adaptive systems and the Newmark time stepper on the
device (164 k DoF: 3.5 ms per step against 43 ms with MUMPS); the validation milestone M10
against the literature (rib waveguide, metallic grating, Mie sphere, slit–groove in silver)
with the layered background of ADR-0009; and the earlier backlog items: Floquet–Bloch band
structures, the dual formulation with the guaranteed hypercircle bound, and the transient
Maxwell solver. Convergence tests #1–#7 plus the new ones (band structure, hypercircle,
transient cavity, axisymmetric cavity / sphere resonances / Mie sphere / dipole) run in CI.
### Added
- Newmark loop of `physics::TimeDomain` on the GPU: with the cuDSS backend the state stays
  on the device, every step is two device-resident sparse products (own SpMV kernel), the
  fused vector updates and the solve, with only the load scalar crossing per step; the
  state is downloaded for observer calls and at the end (`HPFEM_GPU_STEPPER=0` keeps the
  host loop). Same recursion as `step_reduced`, identical results. 164 k DoFs: 3.5 ms per
  step against 8.3 ms with the host loop and 43 ms on MUMPS. C interface API version 3
  (`hpfem_gpu_matrix_*`, `hpfem_gpu_stepper_*`), `solvers::DeviceMatrix`,
  `solvers::DeviceStepper`, `LinearSolver::backend()`.
- `physics::TimeDomain::run` works on the reduced (free-DoF) vectors and rebuilds the full
  state only for the observer and at the end (`step_reduced`); 164 k DoF, p = 2 on the CPU:
  54 → 44 ms per step. The reduced step is the hook for the GPU time stepper.
- GPU backend for hp-adaptive systems: the GPU library equilibrates the matrix diagonally
  (`D (sA) D`, `d_i = 1/√|a_ii|`) before the cuDSS factorisation, which removes the
  perturbed pivots of systems with hanging nodes and high orders entirely (0 in every step
  of the L-shape and plasmonic-wedge tests against up to 16 356 before), so these systems
  no longer fall back to the CPU; with every system on cuDSS the C++ and Python suites pass
  without a refused factorisation. `HPFEM_GPU_EQUILIBRATE=0` switches it off for
  comparisons; the self-test covers rows scaled 1e-8 … 1e8.
- cuDSS hybrid memory mode for factors beyond the device memory: the GPU library decides
  per factorisation from the peak-memory estimates after the analysis (factors partly in
  host memory above ~90 % of the free device memory; `HPFEM_GPU_HYBRID` forces it), fails
  with the numbers if even the host memory would not suffice (then `kAuto` takes MUMPS),
  reports mode and memory through `hpfem_gpu_factor_info2` (C API version 2, version-1
  libraries still load) and `LinearSolver::details()` (also MUMPS). Measured with the new
  `bench_hybrid_memory`: 1.28 M unknowns with 23 GB of factors in 105 s against 186 s for
  sequential MUMPS (`benchmarks/results/2026-10-03-VR-hybrid-memory.json`). Python:
  `LinearSolver.details`.
- `solvers::Symmetry::kDetect`: cuDSS and MUMPS measure the asymmetry of a matrix once per
  factorisation and take the LDLᵀ path when it is below 1e-12 (upper triangle as given,
  debug log line); the problem classes (`Scattering`, `ScatteringOperator`, `TimeDomain`,
  the complex shift-invert eigensolvers and with them `Resonance`, `BandStructure` and the
  axisymmetric problems, `AxisymmetricScattering`, `Thermal`, hypercircle and goal-oriented
  solves) pass it, so symmetric curl–curl systems get the half-cost factorisation and Bloch
  phases or non-symmetric tensors the general one. `asymmetry` without a transpose,
  `detect_symmetry`; Python `Symmetry.DETECT`, `detect_symmetry`.
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
- `physics::axisymmetric_far_field` (M11): near-to-far transform of one azimuthal order on
  a surface of revolution with the analytic azimuthal integration (Bessel functions J_m,
  J_m±1), pattern F_θ/F_φ(θ) and radiated power; unit tests (Larmor pattern of the axial
  dipole, power balance, sphere scattering vs flux and Mie), far-field column in the Mie
  convergence test, Python `axisymmetric_far_field`. M11 complete.
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

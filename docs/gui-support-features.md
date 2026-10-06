# hp-FEM feature wishlist from the GUI work ("FEM model builder")

Status: proposal of 6 October 2026, verified against `main` at fa8f6bf (merge of #97, conical incidence and E_z).
Audience: the development agents. Tracked in `docs/roadmap.md` as milestone M15; the IDs F0 to F16 below are used there.

The GUI sources this document refers to (`fem_app.py`, `fem_geometry.py`, `fem_worker.py`, `fem_post.py`, `fem_run.py`, `fem_materials.py`) and the conical RCWA reference
(`rcwa_conical.py`, source of the numbers in section 3) are kept outside this repository by the maintainer; the descriptions of the workarounds are meant to be readable without them.

## 0. Context: what the GUI does today and where it hurts

The GUI (Streamlit) lets a user enter a periodic unit cell (materials, layer stack, shapes, illumination, sweep), meshes it with Gmsh, runs hp-FEM and shows
spectra, diffraction orders, field maps and absorption. Architecture: the app writes `job.json` + `mesh.msh` into a folder; a worker (`fem_worker.py`, runs in the
Python that has `hpfem`) solves with `ConicalScattering` and writes `results.json` + `maps_<i>.npz`; the app plots. Files and what they work around:

| file | content | lines of workaround |
|---|---|---|
| `fem_geometry.py` | model description, layer/shape geometry, Gmsh meshing (OCC fragments, periodic copies, material assignment, size fields, order 2), mesh statistics | ~560 (of which ~330 meshing) |
| `fem_worker.py` | stack + PML design + Bloch + orders on a line + maps by point loops | ~240 |
| `fem_post.py` | absorbed power by raster integration, maps, plots | ~320 |
| `fem_run.py` | background process, log parsing | ~95 |

Everything in the table is logic that belongs next to the solver (testable, reusable by scripts, CI and other front-ends). Features below are ordered by what
they buy: **better** (accuracy, robustness), **faster**, **simpler** (less glue, fewer ways to get it wrong). Where a feature exists partly, the item says which part.

## 1. Overview

| ID | Feature | Better | Faster | Simpler | Effort | Prio | Exists? |
|---|---|:-:|:-:|:-:|:-:|:-:|---|
| F0 | Document and test the conventions of the conical API | x | | x | S | P1 | no |
| F1 | hp-adaptivity for `ConicalScattering` (TE, TM, conical) with goal-oriented estimator | x | x | x | L | P1 | in-plane only (`Scattering::estimate`, `GoalEstimate`, `HpStep`); hanging-node constraints already in `ConicalScattering` |
| F2 | One-call periodic scattering API (`hpfem.grating.solve`) | x | | x | M | P1 | pieces: `PmlProfile.for_angle`, `conical_*orders` |
| F3 | Vectorised field sampling and triangulated field export as NumPy | | x | x | M | P1 | `FieldExporter` writes .vtu only |
| F4 | Exact absorbed power per material tag / per cell | x | | x | S | P1 | no (`conical_poynting_flux` only, with the E_z interface caveat) |
| F5 | Job runner / CLI with a stable JSON schema and JSON-lines events | | | x | M | P1 | no |
| F6 | Mesh module: unit-cell mesher, mesh report, periodic check, `$Periodic` import | x | | x | M | P1 | `read_gmsh` skips `$Periodic`, no quality report |
| F7 | Structured diagnostics (`problem.validate()`) | x | | x | S | P1 | partly (`PmlBox.max_resolution`, `recommended_thickness`) |
| F8 | Sweep acceleration: reuse symbolic factorisation, affine assembly, parallel sweeps | | x | | M | P2 | `LinearSolver` factorises once per matrix, `solve_many` for loads only |
| F9 | Progress callback, cancellation, timing breakdown, memory estimate | | x | x | S | P2 | `set_num_threads`, `available_backends` |
| F10 | Conical equivalents of `diffraction_orders` / `power_balance` (flux based) | x | | | M | P2 | in-plane only |
| F11 | Isolated scatterers (non-periodic) for the conical solver: cross sections, far field | x | | | M | P2 | in-plane `cross_sections`, `pattern` |
| F12 | H field and Poynting vector of the conical solution | | | x | S | P2 | no |
| F13 | Dispersive materials directly in the setup, out-of-range policy | | | x | S | P2 | `Dispersive.at(omega)` per point |
| F14 | High-level eigenproblems for the GUI (resonances, bands) | x | | | L | P3 | `Resonance`, `BandStructure` exist |
| F15 | Distribution: pip wheels / installer for Windows, `hpfem-gui` entry point | | | x | M | P3 | MSYS2 build only |
| F16 | Periodic boundaries without identical meshes on both sides (non-matching Bloch coupling) and an estimator that includes the periodic facets | x | x | x | L | P1 | no: `bloch_constraints` pairs facets one to one; the residual estimator skips periodic facets |

## 2. Items

### F0. Document and test the conventions of the conical API (P1, S)
Found while writing `run_conical_stack.py` and the worker; the code cannot tell, so both normalise at run time.
* The sign and phase of the s and p amplitude of `layered_conical_wave` (amplitude 1.0): is the s field `+(−sin φ, cos φ, 0)` or `−…` in the BEM frame (x periodic, y invariant, z normal), is the phase 1 at the top interface? The worker divides a global factor `g = vdot(e_ref, E_inc)` out; a documented, tested convention removes this.
* `total_field` returns the physical `(E_x, E_y, E_z)` (the test subtracts `(i0, i1, 1j*i2)` from the scaled incident vector), `conical_plane_wave` takes physical amplitudes and stores `(E0x, E0y, −iE0z)`: state this in the docstrings.
* Frames: solver x periodic, y normal, z invariant; BEM/RCWA literature: x periodic, y invariant, z normal. Provide `hpfem.frames.to_literature(E)` or at least a documented mapping.
* `wave.ky` is the normal wavenumber (name clash with the literature's k_y = beta).
Acceptance: unit test that the incident field of `layered_conical_wave` for s and p at several (theta, phi) equals the documented vectors to 1e-12.

### F1. hp-adaptivity for `ConicalScattering` (P1, L)
Why: sharp metal corners under TM converge slowly on uniform meshes. Measured with the in-plane solver (Ag grating, TM, 50°, 405 nm, period 400 nm, ridge 200 nm, height 148 nm, layered background, PML R0 = 1e-16): uniform meshes stagnate at ΔR0 −3.4e-3 … −5.0e-3 and ΔR−1 +6.1e-3 … +8.7e-3 for 65–130 k DoFs; the hp-adaptive run (22 steps, `run_R3_adaptive.py`) reaches ΔR−1 +2.0e-5 at 66 k DoFs (ΔR0 +2.3e-4) with error ≈ exp(−0.4…−0.5 N^(1/3)). The new solver cannot do this: no `estimate`, no `error`, no AdaptiveMesh coupling in the bindings, although `ConicalScattering` already carries hanging-node constraints.
Proposal:
* `ConicalScattering::estimate(solution, options) -> adaptivity::Estimate`: residual indicators for the coupled system (E_t Nédélec, v = −iE_z in H1), including the beta-dependent terms, PML tensors and the interface condition on E_z / epsilon E_z; element indicators comparable with `Estimate` of the in-plane solver (so `HpStep` and `SmoothnessOptions` apply unchanged).
* Goal-oriented variant for diffraction orders: reuse `GoalEstimate` with `fourier_coefficient_functional`, extended to the vector amplitudes of the conical orders. The GUI wants "stop when |ΔR_m| < tol".
* Corner pre-refinement from geometry (defect D7): `adaptivity::refine_at_corners(mesh, tags, levels)` so the first steps do not refine smooth regions.
* Periodic faces: see F16. Until F16 exists the marking must be symmetric across a Bloch pair (defect D4); our workaround is in `fem_worker.py` (`periodic_defects`, `symmetrise_periodic`).
* Python: a generator `hpfem.adaptive_solve(problem_factory, goal, tol, max_dofs)` that yields per step (DoFs, estimator, observables R_m, T_m) so a GUI can stream convergence and stop on tolerance.
Acceptance: (a) beta = 0, TM, reproduces the in-plane adaptive result (ΔR−1 ≤ 5e-5, ΔR0 ≤ 5e-4 at ≤ 100 k DoFs); (b) TE, Ag, 50° (reference below); (c) conical, φ = 40°, with the references of section 3; (d) exponential convergence rate in N^(1/3) on the Ag case.

### F2. One-call periodic scattering API (P1, M)
The worker repeats this on every call: snap stack interfaces to mesh vertices (a mismatch of 1e-22 is rejected: defect D2), design the PML from the largest propagating order angle (`for_angle`, capped, with `n_ref = min(index above, index below)`), place the measurement lines in the cover and substrate, build `PeriodicPair2D` with the Bloch phase, extract R_m, T_m, energy balance.
Proposal: `hpfem.grating.solve(mesh, materials, stack, polarisation, theta, phi, omega, order, *, pml=..., bottom="pml"|"pec", orders_max=3)` returning a `GratingResult` with `R_orders`, `T_orders` (efficiencies and complex vector amplitudes), `A`, `power_balance_residual`, `wave`, `solution`, `field(points)`, timing. Tolerance or automatic snapping of stack interfaces to mesh lines (default 1e-9 relative to the period). Tag convention as in F6.
Acceptance: reproduces `fem_worker.py` on the six GUI presets; the worker then shrinks to JSON handling.

### F3. Vectorised field sampling and triangulated export as NumPy (P1, M)
Today a map costs one Python call to `total_field(solution, locator, x)` per pixel (10⁴ to 7·10⁴ per map, each with point location). Not yet measured on the real solver; expected to dominate the post-processing time.
Proposal (both for `Scattering` and `ConicalScattering`):
* `solution.sample(points)` with `points` of shape (N, 2): returns (N, 3) complex E (and optionally H), cell search and evaluation in C++ with OpenMP, Bloch wrapping for points outside the cell, an option to treat points on an interface as belonging to the cell above.
* `solution.triangulate(subdivisions=n)` returns `points`, `triangles`, `values` (complex, per component) of every element sampled on a regular sub-triangulation (curved cells included), plus `cell_tag` per triangle, ready for `matplotlib.tri`. `FieldExporter` (currently `.vtu` only) should reuse the same sampler.
Acceptance: ≥ 50× faster than the Python loop for 10⁵ points, identical values to 1e-12.

### F4. Exact absorbed power per material tag and per cell (P1, S)
The GUI integrates Q = ½ ω ε0 Im(ε) |E|² on the map raster; at sharp material boundaries this is wrong by about one pixel: measured against the energy balance (Si grating, converged RCWA field, so purely the quadrature error) the ratio field/balance is 1.08 on a 121 × 401 grid and 1.04 on 241 × 1201. The GUI shows a normalised column as a workaround.
Proposal: `absorbed_power(solution, tags=None) -> {tag: W/m}` and `absorption_density(solution) -> per cell and quadrature point`, for `ConicalScattering` and `Scattering`, volumetric quadrature of the Joule heating on the mesh, optional normalisation by the incident power per period (`S_inc = ½ n cos θ / Z0` for |E0| = 1 V/m). This avoids the E_z flux caveat of `conical_poynting_flux`.
Acceptance: Σ over tags = 1 − R − T to 1e-4 for a lossless substrate on the lamellar test (reference below); per-tag values equal the closed-form absorption of the flat stack for a layered background without structure.

### F5. Job runner / CLI with a stable JSON schema and JSON-lines events (P1, M)
`python -m hpfem.run job.json` with the schema of our `job.json` (model, layout, incidence, sweep, solver, maps, pscan; version field, room for `dim`), progress and results as JSON lines on stdout (`{"event":"point","i":3,"n":11,"R":…}`), cooperative cancellation (SIGTERM or a cancel file), results as `results.json` and `maps_*.npz`. Also `hpfem.version_info()` (version, git hash, build flags, backends).
Why: a GUI, batch scripts, CI and remote runs then share one entry point; `fem_worker.py` and the log parsing in `fem_run.py` disappear.
Acceptance: our six presets run through the CLI with identical results to the worker.

### F6. Mesh module (P1, M)
* `hpfem.meshing.PeriodicCell` (or `unit_cell_mesh(model_dict)`): the logic of `fem_geometry.build_mesh`: layers and substrate as slabs, shapes (rectangle, trapezoid, circle, ellipse, polygon) with periodic copies clipped to the cell, material by priority via the fragment map, physical groups (surfaces = material tags, curves = left/right/bottom/top), `setPeriodic`, sizes from wavelength per material (`N ≈ 12/p` elements per wavelength) and for metals from the field decay length 1/(k0 κ), interface refinement, order 2 for curved shapes. Must call `gmsh.initialize(interruptible=False)` (otherwise Gmsh fails in any non-main thread: found with Streamlit). Tests: material areas against a raster map to 1e-3 (our check), periodic node match, no sliver triangles for shapes touching an interface tangentially (we warn and embed 1–2 nm).
* `mesh.report()`: cells, min/mean angle, aspect ratio, edge length statistics, curved-element Jacobian validity, untagged cells (`kNoTag`), tags without material, periodic match.
* `mesh.check_periodic(left_tag, right_tag, shift) -> max mismatch`, and `read_gmsh` reading the `$Periodic` section so that pairs and shift need not be passed by hand (with F16 the pairing needs no matching nodes at all).
Acceptance: the six presets mesh to the same cell counts (±5 %) as `fem_geometry.py`; invalid curved elements are reported before assembly.

### F7. Structured diagnostics (P1, S)
`problem.validate() -> list[Diagnostic(code, severity, text, hint)]`, called by `solve`. Cases that cost us time: stack interface not on a mesh line (D2), untagged cells, tag without material, missing periodic partner, PML under-resolved (`PmlBox::max_resolution`) or thinner than `recommended_thickness`, too few elements per wavelength for the chosen p, tabulated material outside its range (today a late `ValueError`), lossy incidence medium, grazing orders that make the PML ineffective, bottom PEC wall at less than six decay lengths in a lossy substrate (the field reaches a wall 400 nm below Si at 405 nm with about 20 % of its surface amplitude, which falsifies R and A). The GUI maps codes to German text and shows them before the run.

### F8. Sweep acceleration (P2, M)
A 50-point spectrum at ~140 k DoFs is the common job. Proposals: (a) `LinearSolver.refactorize(matrix)` that reuses the symbolic analysis (MUMPS analysis phase) because the sparsity pattern does not change along a sweep; (b) affine assembly: element matrices per material tag assembled once (curl-curl S(beta), mass M_tag), combined per omega as S − k0² Σ ε_tag M_tag, PML part recomputed; (c) `solve_sweep(factory, values, processes=n)` with one thread per process, which usually beats threads in the direct solver; (d) angle sweeps at fixed omega: reuse the assembled operator when only the incident field and the Bloch phase change (the Bloch constraint changes the reduced matrix, so measure first).
Acceptance: 50 points at 100 k DoFs at least 2× faster than the naive loop; results identical to 1e-10.

### F9. Progress, cancellation, timing, memory estimate (P2, S)
`solve(progress=callback)` with phases (assembly, analysis, factorisation, solve, post), a cancellation flag checked between phases, `solution.timing`, and `estimate_memory(mesh, p, backend)` so the GUI can warn before a run (we estimate DoFs ≈ 1.4·0.9·(p+1)(p+2)·cells: replace by the library's number).

### F10. Conical equivalents of the in-plane post-processing (P2, M)
`diffraction_orders`, `power_balance` (flux based, M14-C) exist for the in-plane solver. For `ConicalScattering`: orders on any line with complex vector amplitudes in the BEM frame, flux-based efficiencies (volumetric where possible), `PowerBalance` with the relative residual as the reference-free quality indicator. The GUI shows ΣR + ΣT + A − 1 per point.

### F11. Isolated scatterers for the conical solver (P2, M)
GUI phase two: nanowires and particles, not only periodic cells. Python API for a finite cross-section in a homogeneous or layered background with an automatic closed measurement contour built from a tagged region (`Surface2D.around_cells` does the pieces), σ_sca, σ_ext, σ_abs and efficiencies, far-field pattern (the in-plane `cross_sections` and `pattern` exist). Acceptance: Mie cylinder TE and TM (series) to 1e-4; wire on a substrate against a published value.

### F12. H field and Poynting vector (P2, S)
`solution.h_field(points)` and `poynting(points)` for the conical solution (curl of E is available from the Nédélec part) for energy-flow maps and Poynting-based checks.

### F13. Dispersive materials in the setup (P2, S)
`setup.materials.set(tag, hpfem.materials.get("Ag"))` and `setup.set_frequency(omega)`; today the user loops and calls `.at(omega)` per tag. Policy for out-of-range wavelengths as an explicit option (error, clamp with warning). A Drude–Lorentz fit helper for tabulated n, k (smooth in omega, useful for adaptive runs and eigenproblems). Anisotropic diagonal ε as a later extension.

### F14. High-level eigenproblems (P3, L)
Resonances (complex frequency, Q, mode maps) and band structures through the same job schema (Bloch + layered + PML), returning mode fields as in F3. `Resonance` and `BandStructure` exist; the missing part is the periodic-cell front end.

### F15. Distribution (P3, M)
The MSYS2 build is the largest barrier for users. CI artefacts as wheels (`hpfem-<ver>-win_amd64.whl`, SparseLU build, MUMPS where licence and size allow), Linux wheels, `pip install hpfem[gui]` pulling Gmsh and Streamlit, and an `hpfem-gui` entry point that starts the app. The GUI can then drop the separate "Python with hpfem" setting.

### F16. Periodic boundaries without identical meshes on both sides (P1, L)
User wish (6 Oct 2026): the library should not insist on periodic meshes. Today `bloch_constraints` (`src/assembly/periodic.cpp`, `match_facets`) pairs every slave facet with the master facet at its shifted
centroid and throws unless the counts agree and every position matches ("the two sides must be meshed identically").
What goes wrong in practice: the first hp-adaptive run of the GUI on a Gmsh mesh stopped after step 6 with `bloch_constraints: 14 master facets (tag 101) but 16 slave facets (tag 102)`. The 1-irregularity rule of the
refinement split a boundary cell next to a marked cell on one side only (cells at the faces are never marked themselves, but they are refined by closure); on an unstructured mesh the neighbourhoods of the two faces are
not mirror images, so the refinement differs. The earlier structured-mesh run only worked because that mesh was symmetric.
Our workaround (`fem_worker.py`: `periodic_defects`, `symmetrise_periodic`) compares the y-intervals of the facets of both sides after every `hp_refine` and refines the coarser side by extra `hp_refine` calls until they
match. It needs `facets_with_tag`, `facet_vertices`, `facet_cells`; it refines cells the estimator never marked (wasted DoFs); and it can fail to converge (the worker then stops with the previous step). It does not
touch the polynomial orders at the faces; whether `bloch_constraints` handles different orders on the two sides is not documented (we never create them: the cells at the faces keep the start order).
Proposal, in two stages:
* **Stage 1 (small): symmetric refinement inside the library.** `AdaptiveMesh2D.set_periodic(master_tag, slave_tag, shift)` (and 3D): `refine` / `hp_refine` mirror every refinement of a face cell onto its partner, including the
  closure, so the two sides stay identical by construction. Removes the workaround; no change to the constraint code.
* **Stage 2 (the actual wish): non-matching periodic coupling.** Replace the one-to-one facet pairing by a mortar-type constraint: the trace of E_slave on the slave face equals phase times the L2 (or dual-mortar)
  projection of the master trace, assembled over the overlap of master and slave facets, for the tangential Nédélec trace and for the H1 trace of the conical solver, with different refinement levels and different
  polynomial orders on the two sides. Benefits: independent refinement of both sides (half the wasted DoFs), Gmsh and other meshes without `setPeriodic`, meshes of rotated or sheared lattices, and in 3D crossed
  gratings, where matching face meshes are much harder to produce. Needs a stable choice of the multiplier space (e.g. the coarser trace space, or a dual basis) so the discrete problem stays well posed; check
  against the matching case.
* **Estimator on periodic facets.** The residual estimator documents that natural and periodic facets are not accounted for (`residual_estimator.hpp`). With the phase, the jump of the tangential trace and of the
  normal flux across a Bloch pair belongs into the indicators of both adjacent cells. Then the cells at the faces can be marked like any other and the GUI can drop the exclusion mask (cells at the faces are never refined
  today: a singular corner within a few elements of the face, e.g. a ridge edge near x = 0, is refined only from the inside).
Acceptance: (a) the lamellar test (Si and Ag, 400 nm period) solved with the two faces refined to different levels and with different p gives R_m, T_m equal to the matching-mesh result to 1e-8 (same space for the
fine side) or within the discretisation error and converging with refinement; (b) a Gmsh mesh generated without `setPeriodic` works; (c) energy balance ΣR + ΣT + A − 1 unchanged; (d) hp-adaptive run on a Gmsh mesh with
a ridge edge 25 nm from the face (not mirror symmetric) converges like the symmetric structured mesh (Ag, TM, 50°: ΔR−1 ≤ 5e-5 at ≤ 100 k DoFs); (e) Bloch phase with kx ≠ 0 and with β ≠ 0.

## 3. Reference data for acceptance tests (all computed in this session)

Geometry of the grating cases: period 400 nm, ridge 200 nm, height 148 nm, λ = 405 nm, substrate = ridge material, air above, ridges centred at x = 0.
Reference solver: `rcwa_conical.py` (conical RCWA with Li factorisation; checked against Fresnel for flat surfaces at any azimuth to 1e-16, against the in-plane TM and TE RCWA at φ = 0 to 1e-11, energy conservation of lossless gratings to 1e-14; fields at φ = 0 to 4e-9). Values are 1/N extrapolated from 150 and 250 harmonics unless noted.

| case | R0 | R−1 | note |
|---|---|---|---|
| TM, Si (ε = 29.6345 + 2.7721i), θ = 50°, φ = 0 | 0.143381 | 0.142382 | converged in N |
| TE, Ag (ε = −4.6631 + 0.2160i), θ = 50°, φ = 0 | 0.319215 | 0.643575 | absorbed 0.03721 |
| TM, Si, θ = 50°, φ = 40° | 0.142373 | 0.179169 | conical |
| TE, Ag, θ = 50°, φ = 40° | 0.636653 | 0.310883 | conical, slow 1/N convergence (nh = 60 gives R0 0.6349) |
| TE, glass (ε = 2.25), θ = 40°, φ = 30° (nh = 60) | 0.039485 | 0.011312 | lossless: ΣR + ΣT = 1 to 1e-14, also transmitted orders |
| TM, glass, θ = 50°, φ = 30° (nh = 60) | 0.010957 | 0.014330 | transmitted: m = 0 0.875831, −1 0.092560, −2 0.006322 (evanescent in air, propagating in glass: easy to miss in an energy balance) |
| TM, Ag, θ = 50°, φ = 0 (in-plane reference) | 0.77960 ± 6e-5 | 0.07795 ± 2e-5 | the hp case of F1 |

Suggested tolerances: |ΔR_m| ≤ 5e-5 for Si and glass at ≤ 100 k DoFs (p = 5), ≤ 5e-4 for Ag cases (RCWA reference itself is only good to ~1e-4).

## 4. What would help the GUI most, in this order
F1 (accuracy for metals and the "stop at tolerance" mode) together with F16 (without it F1 needs a workaround on every Gmsh mesh), F3 (maps), F2 + F5 + F6 (less glue, one supported path), F4 and F7 (trustworthy absorption and early, readable errors), then F8 and F9 (spectra), then F10–F13, F14, F15.

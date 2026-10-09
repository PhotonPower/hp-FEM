# 0012 — Optimisation, calibration and UQ in `hpfem.opt` on the discrete sensitivities

**Status:** accepted (amends [0007](0007-direct-solver-backends.md))
**Date:** 2026-10-09

## Context

Milestone M16 (`docs/optimisation-uq-features.md`, items S0–S9) adds optimisation, parameter
retrieval with uncertainties and uncertainty propagation. Its premise is that the solver is not a
black box: M12 gives derivatives of observables with respect to permittivities
(`physics/sensitivity.hpp`) and node coordinates (ADR-0011), the DWR estimator gives an error
estimate per goal, and hp-refinement gives fidelity levels. Looking at the code as it is on
`main` after 0.4.0:

- `adjoint_solution` / `conical_adjoint_solution` assemble the system again and call
  `solvers::solve_direct`, i.e. every adjoint solve is a new factorisation. "One extra solve on
  the existing factorisation" is not true yet.
- `solvers::LinearSolver` (ADR-0007) has `solve` and `solve_many`, no solve with Aᵀ. The
  complex-symmetric systems (curl–curl with symmetric tensors, PML, Dirichlet elimination,
  condensation, hanging nodes) have Aᵀ = A; with Bloch phases they do not.
- A geometry parameter is a mesh velocity `V` that the user builds by hand
  (`region_normal_velocity`, or an array). The Gmsh route (`meshing.unit_cell_mesh`) produces a
  new topology for every parameter value; `meshing.structured_unit_cell` keeps the topology for
  fixed `nx` and `rows`.
- Only material derivatives have an analytic residual derivative (`-k0² M_tag`); shape
  derivatives use central differences of element integrals (ADR-0011). Frequency and angle
  derivatives do not exist.
- The job runner `hpfem.run` reads schema version 1 only and rejects anything else.
- NumPy and SciPy are already required dependencies of the Python package.

The decisions below are expensive to reverse: the evaluator contract and the study file format
are what every optimiser, the GUI and stored studies depend on, and the solver interface change
touches all backends.

## Decision

### 1. Package and dependencies

- A pure-Python subpackage `hpfem.opt` (modules `parameters`, `evaluator`, `study`, `optimize`,
  `lsq`, `gp`, `bo`, `posterior`, `uq`), imported explicitly (`import hpfem.opt`), not by
  `import hpfem`. New C++ only for section 4 (residual derivatives, kept factorisation,
  transposed solve) and for the eigenvalue derivatives of S4.
- Required: NumPy and SciPy, as today. Own code where the project has an advantage: parameters
  and morphing, the derivatives, Gauss–Newton / Levenberg–Marquardt, the Laplace approximation,
  a light Gaussian process (Matérn 5/2 ARD, noise, derivative observations), the study framework.
- Optional extras in `pyproject.toml`: `opt-bo = ["botorch"]` (multi-fidelity, multi-objective,
  gradient-enhanced BO on PyTorch), `opt-mcmc = ["emcee"]`, `opt-sa = ["SALib"]`. They are
  imported lazily inside the functions that need them; a missing extra raises an `ImportError`
  that names the extra. None of them is load-bearing for P1. The README dependency table lists
  them when they are added (CLAUDE.md §10). The CI `python` job installs the light extras
  (emcee, SALib); BoTorch paths are tested where it is installed and skipped otherwise.

### 2. Evaluator contract

An evaluator maps parameters to **real** observables:

```python
@dataclass
class Evaluation:
    params: dict[str, float]          # SI values, keyed by parameter name
    values: np.ndarray                # (m,) real observables
    jacobian: np.ndarray | None       # (m, n) d values / d params, columns in parameter order
    error: np.ndarray | None          # (m,) DWR estimate of |error| per observable, or None
    cost: float                       # wall time in seconds
    fidelity: dict                    # e.g. {"order": 4, "level": 0}
    mesh_id: int                      # which reference mesh (section 3); changes on remesh
    status: str = "ok"                # "ok" | "failed" | "cancelled"
    meta: dict = field(default_factory=dict)   # dofs, backend, timings

class Evaluator(Protocol):
    parameters: list[Parameter]
    observables: list[str]            # names, length m
    def __call__(self, params, *, jacobian: bool = False, fidelity: dict | None = None,
                 cancel: Callable[[], bool] | None = None) -> Evaluation: ...
```

- Complex quantities (amplitudes, eigenvalues) enter as two observables `Re` and `Im`; the
  holomorphic derivative gives both columns. Optimisers, least squares and Gaussian processes
  work on real vectors only.
- Parameters are SI internally; `hpfem.units` converts at the user boundary, as everywhere.
- A failed solve returns `status="failed"` with NaN values instead of raising, so that a study
  survives a bad point; the optimiser decides (penalty, shrink step, skip).
- `error` is a magnitude bound, not a standard deviation (section 6).

### 3. Shape parameters morph a reference mesh

- `Parameter(name, lower, upper, unit, scale, velocity)`. `velocity(mesh, value)` returns the
  node field `V = ∂x/∂p` of ADR-0011 for the reference mesh. Three sources:
  (a) builders on tagged regions (`region_normal_velocity`, translation, and the affine stretch of
  a ridge for width and height); (b) a fixed-topology mesher: if the mesher returns meshes of
  identical topology for `p ± h` (the structured unit-cell mesher with fixed `nx` and `rows`),
  `V` is the central difference of the node coordinates, exact for any parameter the mesher maps
  smoothly; (c) a user array.
- A study keeps a **reference mesh** at `p_ref`. An evaluation at `p` uses
  `x(p) = x_ref + Σ_i (p_i − p_ref,i) V_i` (`move_nodes`), or the fixed-topology mesher directly
  when one is available. Cell tags, constraints and the DoF map stay those of the reference mesh,
  so the objective is a smooth function of `p` and its derivative is exactly the one of
  ADR-0011.
- `V` must vanish on Bloch faces, on facets with non-zero Dirichlet data and on the top and
  bottom of a grating cell (ADR-0011, `grating.shape_sensitivity`); `Parameter` checks this once
  per reference mesh and raises otherwise.
- **Quality guard:** after a move, every cell's shape quality (ratio of inscribed to
  circumscribed radius, normalised to 1 for the equilateral simplex) is compared with the
  reference. An inverted cell, or a quality below `0.3` of the reference value, rejects the
  move; the study then **remeshes** at `p` (new reference, new `mesh_id`) and records a `remesh`
  event. Local optimisers treat a remesh as a restart (L-BFGS memory, trust region); surrogates
  keep the points, and the jump is part of the discretisation error.
- Finite-difference checks of derivatives run on the morphed mesh, never across a remesh.

### 4. Derivatives: one mechanism, two modes, kept factorisation

- **Residual derivatives by directional differences.** For any parameter θ (material value,
  shape velocity, frequency, angle of incidence) the vector `r_θ = ∂b/∂θ − (∂A/∂θ) e` is
  computed at fixed `e` by central differences of the element residuals `b_K − A_K e_K` along θ
  (step `1e-6` relative, as ADR-0011), cell-parallel. The analytic material path stays where it
  exists; everything else uses this one routine. For a functional vector that depends on θ
  (moving cells, frequency-dependent orders) the term `(∂q/∂θ)ᵀ e` is added as in ADR-0011.
- **Adjoint mode:** `Aᵀ z_k = q_k` per observable, `dQ_k/dθ = z_kᵀ r_θ`. **Direct (tangent)
  mode:** `A s_θ = r_θ` per parameter, `dQ_k/dθ = q_kᵀ s_θ`. Both cost one solve per column on
  the forward factorisation plus cheap residual differences. The evaluator picks per
  factorisation (frequency, angle, polarisation) the mode with fewer solves:
  direct if `n ≤ m`, adjoint otherwise; both use `solve_many`. A test checks that the two modes
  agree to `1e-8` relative and both agree with finite differences of the solve.
- **Kept factorisation.** `Scattering` / `ConicalScattering` get an option to keep the
  `LinearSolver` after `solve` (memory: the factors stay alive until the result is dropped), and
  the sensitivity functions accept the kept solver instead of factorising again. The reduced
  system with hanging-node and Bloch constraints is the one that is kept, so the adjoint and the
  tangent solves see the same constraints as the forward solve.
- **Transposed solve (amends ADR-0007).** `LinearSolver` gets
  `solve_transposed(const Vector&)` and `solve_transposed_many`. Default implementation: if the
  factorised matrix was declared or detected complex-symmetric, it is `solve`; otherwise the
  backend uses its native transposed solve (Eigen SparseLU `transpose().solve()`, MUMPS
  `ICNTL(9) ≠ 1`), and a backend without one factorises Aᵀ once on demand (the FEM pattern is
  structurally symmetric, so `refactorize` applies) and keeps it. For a reciprocal medium with
  Bloch phases Aᵀ(k) = A(−k), which the tests use as a cross-check.
- Frequency and angle derivatives use the same residual differences (the incident field, the
  Bloch phases and the PML stretch all depend on ω and the angles); dispersive ε(ω) enters
  through the material model, no separate path.

### 5. Study store: JSON lines

- A study is an append-only JSON-lines file (`*.study.jsonl`, UTF-8, one object per line,
  flushed after every line). The first line is the header:

  ```json
  {"type": "study", "schema": 1, "hpfem": "<version>", "space": {...},
   "evaluator": {"name": "...", "settings": {...}, "hash": "<sha256 of the canonical settings>"},
   "created": "<ISO 8601>"}
  ```

- Later lines: `"evaluation"` (the fields of `Evaluation`, arrays as nested lists, NaN as
  `null`, floats written with `repr` so they round-trip), `"remesh"` (`mesh_id`, `params`,
  reason), `"state"` (optimiser checkpoint: method, iteration, internal state needed to resume),
  `"proposal"` (points proposed but not yet evaluated), `"note"`.
- **Resume** = read the file, rebuild the cache and the optimiser state from the last `state`
  line, re-evaluate open proposals. A truncated last line (crash during a write) is ignored.
- **Cache key:** parameter values normalised by their scale and rounded to 12 significant
  digits, plus `fidelity` and the evaluator hash. A changed evaluator setting therefore never
  reuses old values silently.
- Fields and meshes are never stored in the study; only observables, derivatives and metadata.
  The same events are streamed to the job runner and the GUI.

### 6. Statistics conventions

- Least squares minimise `½ ‖W^{1/2}(y(p) − y_meas)‖²` with `W = diag(1/σ²)` from the
  measurement uncertainties. The Laplace covariance is `(Jᵀ W J)⁻¹` at the optimum (scaled by the
  reduced χ² when σ is unknown), computed in scaled parameters and reported in SI.
- The DWR estimate is **not** used as Gaussian noise of a surrogate: the discretisation error is
  systematic, smooth in `p` and signed. It is used (a) to refine an evaluation adaptively until
  `error ≤ κ σ` with `κ = 0.1` by default, and (b) as the fidelity indicator for multi-fidelity
  BO. Whether (b) beats the plain alternatives is the hypothesis of S8; the result is reported
  in `docs/validation.md` either way.

### 7. Parallel evaluation and the job runner

- Evaluations run **sequentially** by default (OpenMP uses the cores, cuDSS the one GPU).
  `workers=k > 1` starts `k` spawned processes, each with `OMP_NUM_THREADS = cores // k`, after a
  memory check `k · estimate_memory(...) < available memory`; with the cuDSS backend `k` is
  forced to 1.
- `hpfem.run` moves to schema version 2 with the tasks `optimize`, `reconstruct` and `uq`;
  it keeps reading version 1 documents unchanged. Progress, cancellation and timing use the
  existing event mechanism; a study task writes its `*.study.jsonl` next to the result.

## Consequences

- New C++ in P1: the kept factorisation, `solve_transposed` in all three backends (with the
  on-demand Aᵀ fallback), and the residual-difference routine for arbitrary parameters. ADR-0007
  is amended by section 4; its status line and table entry say so.
- Keeping the factorisation holds the factors in memory as long as a result object lives; the
  Python API releases them with the result, and `estimate_memory` already predicts their size.
- Morphing limits a study to shape changes the reference mesh can follow; large moves go through
  a remesh, which local optimisers see as a restart. Studies on Gmsh meshes therefore work best
  with region-based velocities; the structured mesher gives exact velocities for free.
- The evaluator contract is real-valued; complex goals cost two observables but keep every
  optimiser and surrogate simple.
- JSON lines are human-readable, survive crashes and stream to the GUI, but large studies with
  big Jacobians grow linearly; a study of 10⁴ evaluations with 100 × 10 Jacobians is a few tens
  of MB, acceptable.
- Three optional extras widen the test matrix; the tests that need them skip cleanly without.
- The job schema version 2 needs the GUI to read version 2 results; version 1 stays valid.

## Alternatives considered

- **Black-box only** (no derivatives, as the commercial optimiser) — throws away the cheapest
  information the solver has; least-squares reconstruction and its uncertainties would need
  finite differences or sampling.
- **Remeshing at every evaluation, finite-difference gradients** — the objective jumps with the
  mesh at the size of the discretisation error, which breaks line searches and Gauss–Newton and
  makes finite differences meaningless at small steps.
- **Adjoint mode only** — scatterometry (many orders, wavelengths and angles; three to five
  parameters) would pay one solve per observable instead of one per parameter.
- **Factorising Aᵀ always / transposing the matrix in Python** — doubles the memory and the
  factorisation time in the common complex-symmetric case, where the existing factors suffice.
- **BoTorch (PyTorch) as a required dependency** — a large binary dependency for every user,
  including those who only want least squares; kept optional.
- **Own MCMC samplers, Sobol' estimators, polynomial chaos** — mature libraries exist; the
  project has no advantage there. Polynomial chaos is not planned (backlog).
- **SQLite or HDF5 for studies** — binary, harder to inspect and merge; SQLite needs locking
  for concurrent workers, HDF5 a dependency. JSON lines with a single writer (the study process)
  suffice.
- **Automatic differentiation of the assembly** — rejected in ADR-0011 for the same reasons.

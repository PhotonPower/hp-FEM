# Optimisation, calibration and uncertainty quantification (milestone M16)

Status: proposal of 9 October 2026, revised after the review of PR #134. Written against `main` after release 0.4.0.
Audience: the development agents. Tracked in `docs/roadmap.md` as milestone M16; the IDs S0 to S9 below are used there.

## 1. Where we stand

| Capability | State |
|---|---|
| Material derivatives of observables (dQ/dε per tag, Re and Im ε) by the adjoint solve | done (M12, `physics/sensitivity.hpp`, `grating.sensitivity`) |
| Shape derivatives by the discrete adjoint on the mesh (mesh velocity fields) | done (M12, ADR-0011, `physics/shape_sensitivity.hpp`, `grating.shape_sensitivity`) |
| Derivatives of resonances / eigenvalues | missing |
| Derivatives with respect to frequency and angle | missing |
| Geometry parameter → mesh velocity mapping, Jacobian of several observables | missing (the velocity field is built by hand today) |
| Optimisers (local, global, Bayesian, multi-objective) | missing |
| Parameter retrieval with uncertainties (least squares, Laplace, MCMC) | missing |
| Uncertainty propagation, Sobol' indices, global surrogates | missing |

## 2. Reference: what the established tool offers

JCMsuite is the model for the solver; its optimisation and analysis side is a separate product,
JCMoptimizer. From its public product pages (jcmwave.com, `JCMsuite/Technology` and
`JCMoptimizer`) the feature groups are:

- optimisation of expensive black-box functions: Bayesian optimisation as the main method, plus
  downhill simplex, particle swarm, differential evolution and L-BFGS-B; physics-informed and
  multi-objective variants (Pareto front);
- model calibration: reconstruction of material and shape parameters from measured data together
  with their measurement uncertainties, by Bayesian least squares and Markov-chain Monte Carlo;
- sensitivity analysis (local and global, variance based / Sobol'), active learning of global
  surrogate models, on Gaussian processes and neural-network ensembles;
- scripting from Python and MATLAB, a browser dashboard for studies, benchmarking of studies
  against each other.

Only the feature descriptions are used here; nothing of that software is copied. The tool treats
the simulation as a black box. We do not have to.

## 3. Where the project has an advantage

1. **Derivatives are available.** The adjoint mode costs one extra solve per observable
   (independent of the number of parameters), the direct (tangent) mode one extra solve per
   parameter (independent of the number of observables), both on the factorisation that is
   already there. Gradient-based optimisers, Gauss–Newton for least squares and
   gradient-enhanced Gaussian processes follow directly. Scatterometry has many observables
   (orders × wavelengths × angles × polarisations) and few parameters (CD, height, side-wall
   angle), which favours the direct mode; optimisation of a single figure of merit favours the
   adjoint.
2. **Jacobian = Fisher information.** Parameter uncertainties of a reconstruction come from the
   Jacobian without any sampling (Laplace approximation); sampling is only needed for
   non-Gaussian posteriors. This is cheap enough to belong to the first priority.
3. **hp-adaptivity gives fidelity levels and an error indicator.** Different p / mesh levels are
   natural fidelities, and the goal-oriented (DWR) estimate says how far an evaluation is from
   converged. Caveat: the discretisation error is systematic, smooth in the parameters and
   signed, so it is *not* independent noise. The plan treats it as a fidelity indicator (or
   refines adaptively until the error is small against the measurement noise) and tests that
   hypothesis in S8 before anything is promised.

## 4. Scope and architecture

- A new Python subpackage `hpfem.opt`, no new C++ except the eigenvalue derivatives of S4.
- Own code only where we have an advantage: gradients, the Laplace approximation, the use of
  the DWR estimate. A light Gaussian process (Matérn ARD, noise, derivative observations) is
  our own and small (permissive licence, no PyTorch). Gradient-enhanced multi-fidelity and
  multi-objective BO, NUTS / ensemble MCMC, polynomial chaos and Sobol' estimators exist in
  mature libraries (BoTorch, emcee, SALib) and are optional extras. Any load-bearing dependency
  needs an ADR note (CLAUDE.md §10), hence S0. For Sobol' one route (GP surrogate) is enough.
- **Morphing versus remeshing.** If every parameter value got a new mesh, the objective would
  contain remeshing jumps and be inconsistent with the adjoint gradient, which hurts L-BFGS,
  Gauss–Newton and the finite-difference tests. ADR-0012 fixes the strategy: a reference mesh is
  morphed with `move_nodes` within a parameter range, remeshing happens only when the quality
  guard trips, and the finite-difference checks run on the morphed mesh.
- Studies are described by a design space and an evaluator
  `f(params) -> value(s) [, gradient / Jacobian] [, error estimate]`; results go to a JSON-lines
  store so that a study can be resumed, inspected and compared.
- Parallelism: OpenMP already uses the cores and cuDSS runs on one GPU; on Windows every spawned
  process reloads the library and holds its own factorisation. Batch proposals are therefore
  evaluated sequentially; several processes only with an explicit thread budget.
- The job runner (`hpfem.run`, schema version 1 → 2) gets the tasks `optimize`, `reconstruct`
  and `uq`, reusing events and cancellation (S9).
- Flagship problems for development and validation: a 2D grating and a metasurface unit cell.
  The full 3D ring resonator (M12 stage B) is not a target: it is not solved itself and too
  expensive to validate an optimiser on.

## 5. Work packages

### S0 — ADR-0012 (P1, ≈ 1 session)
Scope, dependencies, study file format, the evaluator contract (value, gradient or Jacobian,
error estimate, cost) and the morphing-versus-remeshing strategy of section 4. Status
*proposed* until the maintainer accepts it.

### S1 — Gradient infrastructure (P1, ≈ 4 sessions)
- `Parameter` objects for geometry (radius, width, height, position, rounding) that produce the
  mesh velocity field and the Jacobian column; composable with `region_normal_velocity` /
  `move_nodes`; valid within a parameter range of the morphed reference mesh.
- Jacobian of several observables in both modes (direct and adjoint) on one factorisation; the
  cheaper mode is chosen from the numbers of parameters and observables.
- Derivatives with respect to frequency and incidence angles.
- Mesh-quality guard: detect inverted or poor cells after a shape change and report the
  admissible step or trigger the remeshing of the strategy above.
- Tests: every derivative against central finite differences of the full solve on the morphed
  mesh (as for M12).

### S2 — Study framework (P1, ≈ 3 sessions)
Design space (continuous, integer, categorical, linear and nonlinear constraints), evaluation
cache keyed by parameters, resume, JSON-lines result store, progress and cancellation hooks of
M15 F9. Batch proposals run sequentially; several processes only with an explicit thread budget.

### S3 — Classical optimisers and Laplace approximation (P1, ≈ 4 sessions)
L-BFGS-B with gradients, Nelder–Mead and differential evolution (SciPy wrappers),
Gauss–Newton and Levenberg–Marquardt with the Jacobian of S1. The parameter covariance
follows from the Fisher information of that Jacobian. End-to-end showcase of the first
priority: scatterometry of a Si grating, reconstructing CD, height and side-wall angle with
their uncertainties from synthetic data (measured data and literature papers do not go into the
public repository).

### S4 — Eigenvalue derivatives (P2, ≈ 3–4 sessions)
The P1 goals do not need them, and they are harder than they look:
- with PML and losses the eigenproblem is non-Hermitian and needs the left eigenvector; with
  Bloch / conical incidence the left vector is the solution at −k;
- dispersive ε(ω) makes the problem nonlinear in ω, which adds a dε/dω term;
- complex ω gives the Q factor and its derivative.
Applies to `ConicalResonance`, `Resonance` and the band-structure solver. Verified against
finite differences.

### S5 — Bayesian optimisation (P2, ≈ 3 sessions)
Own light Gaussian process (Matérn ARD kernel, noise, hyperparameters by marginal likelihood),
expected improvement and lower confidence bound, constraints, gradient-enhanced GP with the
derivatives of S1. Multi-fidelity and multi-objective (Pareto front) through the optional
BoTorch extra. The DWR estimate enters as a fidelity indicator or as the stopping criterion of
adaptive refinement, not as independent heteroscedastic noise (see section 3).

### S6 — Parameter retrieval beyond Laplace (P2, ≈ 2 sessions)
Bayesian least squares on the surrogate and MCMC through the optional emcee extra, with the
posterior compared against the Laplace result of S3 (non-Gaussian cases only need sampling).

### S7 — Uncertainty propagation and sensitivity analysis (P2, ≈ 2 sessions)
Linearised (delta-method) propagation from the Jacobian, Monte Carlo on the surrogate, Sobol'
indices by one route (GP surrogate; SALib optional). Example: fabrication tolerances (for
example a few nm of line width) propagated to diffraction efficiencies.

### S8 — Validation (P2, ongoing)
Each item is a regression or convergence test with the labels of CLAUDE.md §8:
- Branin and Rosenbrock for the Bayesian optimiser, known optimum within tolerance;
- Ishigami function for Sobol' indices (analytic values);
- NIST MGH17 for the reconstruction uncertainties (certified values are public);
- linear-Gaussian problems where MCMC and the Laplace approximation must agree;
- gradient-based against gradient-free BO on the 2D grating and the metasurface unit cell,
  recorded in `benchmarks/results/` and assessed in `docs/validation.md`;
- the DWR hypothesis: does the error estimate work better as a fidelity indicator than as
  independent noise? Reported either way.

### S9 — GUI and job schema (P3, ≈ 3 sessions)
Tasks `optimize`, `reconstruct`, `uq` in the job schema; a study view in `hpfem-gui` (history,
Pareto front, Sobol' bars, parameter posteriors).

## 6. Priorities and effort

P1 = S0 to S3 (≈ 12 sessions): derivatives, study framework, optimisers and the Laplace
reconstruction with uncertainties as the end-to-end result. P2 = S4 to S8 (≈ 10–11 sessions):
eigenvalue derivatives, our own Bayesian optimiser, MCMC, uncertainty propagation and validation.
P3 = S9. Total roughly 25 sessions (down from 30 by using optional libraries).

## 7. Decisions of the review (PR #134)

- Own light GP, BoTorch optional: agreed.
- Synthetic data only in the repository: agreed.
- Full 3D ring (M12 stage B) as target application: no; flagships are the 2D grating and the
  metasurface unit cell, the ring later.

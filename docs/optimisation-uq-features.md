# Optimisation, calibration and uncertainty quantification (milestone M16)

Status: proposal of 9 October 2026, written against `main` after release 0.4.0.
Audience: the development agents. Tracked in `docs/roadmap.md` as milestone M16; the IDs S0 to S8 below are used there.

## 1. Where we stand

| Capability | State |
|---|---|
| Material derivatives of observables (dQ/dε per tag, Re and Im ε) by the adjoint solve | done (M12, `physics/sensitivity.hpp`, `grating.sensitivity`) |
| Shape derivatives by the discrete adjoint on the mesh (mesh velocity fields) | done (M12, ADR-0011, `physics/shape_sensitivity.hpp`, `grating.shape_sensitivity`) |
| Derivatives of resonances / eigenvalues, with respect to frequency and angle | missing |
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

## 3. What we can do better

1. **Adjoint gradients are cheap.** One extra solve (reusing the factorisation) gives the gradient
   with respect to any number of material and shape parameters. Gradient-based optimisers,
   Gauss–Newton for least squares and gradient-enhanced Gaussian processes follow directly.
2. **The error estimator gives a noise model.** The goal-oriented (DWR) estimate of each
   evaluation is a per-point discretisation uncertainty for the surrogate (heteroscedastic noise),
   so a cheap, coarse evaluation is weighted correctly instead of being trusted blindly.
3. **hp-adaptivity gives fidelity levels.** Different p / mesh levels are natural fidelities for
   multi-fidelity Bayesian optimisation.
4. **Jacobian = Fisher information.** Parameter uncertainties of a reconstruction come from the
   adjoint Jacobian without any sampling (Laplace approximation); sampling is only needed for
   non-Gaussian posteriors.

## 4. Scope and architecture

- A new Python subpackage `hpfem.opt`, no new C++ except the eigenvalue derivatives of S1.
- Dependencies: NumPy and SciPy required; the Gaussian-process code is our own and small
  (permissive licence, no PyTorch); BoTorch / GPyTorch only as an optional extra. Any
  load-bearing dependency needs an ADR note (CLAUDE.md §10), hence S0.
- Studies are described by a design space and an evaluator
  `f(params) -> value(s) [, gradient] [, error estimate]`; results go to a JSON-lines store so
  that a study can be resumed, inspected and compared.
- The job runner (`hpfem.run`, schema version 1 → 2) gets the tasks `optimize`, `reconstruct`
  and `uq`, reusing events and cancellation (S8).

## 5. Work packages

### S0 — ADR-0012 (P1, ≈ 1 session)
Decide scope, dependencies, the study file format and the evaluator contract (value, gradient,
error estimate, cost). Status *proposed* until the maintainer accepts it.

### S1 — Gradient infrastructure (P1, ≈ 4 sessions)
- `Parameter` objects for geometry (radius, width, height, position, rounding) that produce the
  mesh velocity field and the Jacobian column; composable with `region_normal_velocity` /
  `move_nodes`.
- Jacobian of several observables (for example all diffraction orders × polarisations) with one
  factorisation and one adjoint solve per observable.
- Derivatives with respect to frequency and incidence angles.
- Eigenvalue derivatives: dω/dε and dω/d shape of resonances (`ConicalResonance`, `Resonance`)
  and bands, including the complex Q.
- Mesh-quality guard: detect inverted or poor cells after a shape change and either remesh or
  report the admissible step.
- Tests: every derivative against central finite differences of the full solve (as for M12).

### S2 — Study framework (P1, ≈ 3 sessions)
Design space (continuous, integer, categorical, linear and nonlinear constraints), evaluation
cache keyed by parameters, parallel evaluation, resume, JSON-lines result store, progress and
cancellation hooks of M15 F9.

### S3 — Classical optimisers (P1, ≈ 3 sessions)
L-BFGS-B with adjoint gradients; Nelder–Mead, differential evolution and particle swarm
(SciPy or small own implementations); Gauss–Newton and Levenberg–Marquardt with the adjoint
Jacobian for least squares. All driven by the study framework.

### S4 — Bayesian optimisation (P2, ≈ 6 sessions)
- Gaussian process with Matérn ARD kernel and noise; hyperparameters by marginal likelihood.
- Acquisition: expected improvement and lower confidence bound; batch proposals; constraints.
- Gradient-enhanced GP using the adjoint derivatives as derivative observations.
- Heteroscedastic noise from the DWR estimate; multi-fidelity over p or mesh level.
- Multi-objective optimisation with Pareto-front output.

### S5 — Parameter retrieval (P2, ≈ 5 sessions)
Laplace approximation (covariance from the Fisher information of the adjoint Jacobian), Bayesian
least squares on the surrogate, MCMC (ensemble sampler; NUTS where gradients exist). Example:
scatterometry of a Si grating, reconstructing CD, height and side-wall angle with their
uncertainties from synthetic data (the realistic data case stays with the maintainer; measured
data and literature papers do not go into the public repository).

### S6 — Uncertainty propagation and sensitivity analysis (P2, ≈ 5 sessions)
Linearised (delta-method) propagation from the adjoint gradients; (quasi-)Monte Carlo on the
surrogate; polynomial chaos or stochastic collocation; first-order and total Sobol' indices;
active learning of a global surrogate (sampling where the GP is most uncertain). Example:
fabrication tolerances (for example a few nm of line width) propagated to diffraction
efficiencies.

### S7 — Validation (P2, ongoing)
Each item is a regression or convergence test with the labels of CLAUDE.md §8:
- Branin and Rosenbrock for the Bayesian optimiser, known optimum within tolerance;
- Ishigami function for Sobol' indices (analytic values);
- NIST MGH17 for the reconstruction uncertainties (certified values are public);
- linear-Gaussian problems where MCMC and the Laplace approximation must agree;
- adjoint-gradient BO against gradient-free BO on a grating or metasurface problem, recorded in
  `benchmarks/results/` and assessed in `docs/validation.md`.

### S8 — GUI and job schema (P3, ≈ 3 sessions)
Tasks `optimize`, `reconstruct`, `uq` in the job schema; a study view in `hpfem-gui` (history,
Pareto front, Sobol' bars, parameter posteriors).

## 6. Priorities and effort

P1 = S0, S1, S2, S3: they sit directly on the existing adjoint and give usable optimisation and
least-squares reconstruction early. P2 = S4, S5, S6, S7: the differentiating methods
(gradient-enhanced GP, DWR noise) and the uncertainty analysis. P3 = S8 and the multi-fidelity /
multi-objective parts of S4. Total roughly 30 focused sessions.

## 7. Open points for the maintainer

- Own GP code versus an optional BoTorch dependency (proposal: own code, BoTorch optional).
- Whether S5 uses real measurement data anywhere in the repository (proposal: synthetic only).
- Whether the full 3D ring resonator (M12 stage B) is a target application for S4, since it is
  the most expensive evaluator and benefits most from gradients and multi-fidelity.

"""Least-squares reconstruction and the Laplace approximation (M16 S3, ADR-0012 §6).

**Problem.** Given measurements ``y_meas`` (m,) with standard deviations ``σ`` and an
evaluator ``y(p)`` with its Jacobian, :func:`fit` minimises

.. math:: F(p) = \\tfrac12 \\lVert r(p) \\rVert^2, \\qquad
   r(p) = W^{1/2} (y(p) - y_\\mathrm{meas}), \\qquad W = \\mathrm{diag}(1/σ^2),

over the free (continuous, not ``fixed``) parameters inside their bounds. ``W`` makes
observables of very different magnitudes comparable; ``sigma="relative"`` takes
``σ_i = |y_meas,i|`` (relative errors of unknown common size).

**Algorithm.** In the unit-cube coordinates ``u`` of the design space (scaled parameters,
:class:`hpfem.opt.DesignSpace`), with ``J_u = W^{1/2} (∂y/∂p) (∂p/∂u)``, ``g = J_uᵀ r``:

- *Levenberg–Marquardt* (default): the step on the free set solves
  ``min ‖J δ + r‖² + μ ‖D^{1/2} δ‖²`` by a least-squares solve of the stacked system
  ``[J; (μD)^{1/2}] δ = [−r; 0]`` (no normal equations), with Moré's scaling ``D`` = the
  running maximum of ``diag(J_uᵀ J_u)``. The gain ratio ``ρ`` = actual / predicted decrease of
  ``F`` decides: ``ρ > 1e-4`` accepts the step and sets ``μ ← μ · max(1/3, 1 − (2ρ − 1)³)``
  (Nielsen), otherwise ``μ ← μ ν``, ``ν ← 2ν``. ``μ₀ = 1e-3``.
- *Gauss–Newton* (``method="gn"``): the minimum-norm solution of ``J δ = −r`` with step
  halving until ``F`` decreases.

**Bounds** are handled by an active set and projection: a parameter at a bound whose
gradient points outwards is held there (its step is zero); the step of the others is
computed, the trial point is projected onto the box and the predicted decrease is evaluated
for the projected step. Linear and nonlinear constraints of the design space are not modelled:
an infeasible trial point is rejected like a failed evaluation.

**Convergence** (any of): the scaled gradient ``max_j |g_j| / (‖J_j‖ ‖r‖) ≤ gtol`` on the free
set (MINPACK's cosine test), the step ``‖δu‖ ≤ xtol (‖u‖ + xtol)``, or both the actual and the
predicted relative decrease of ``F`` below ``ftol``; a zero residual. Failed or infeasible
trial points (ADR-0012 §2) count as rejected steps (``μ`` grows, the step shrinks); more
than ``max_failures`` stop the fit. A **remesh** at a trial point (ADR-0012 §3) is a restart:
the point is accepted (the new reference mesh was built there) and ``μ``, ``ν`` are reset.

**Laplace approximation** (:func:`laplace`). At the optimum the posterior of the parameters
(flat prior) is approximated by a Gaussian with covariance

.. math:: C = s^2 \\, (J^T W J)^{-1}, \\qquad s^2 = χ^2_\\mathrm{red} =
   \\frac{\\lVert W^{1/2}(y - y_\\mathrm{meas}) \\rVert^2}{m - n}

with ``s² = χ²_red`` when ``σ`` is unknown (``sigma=None`` or ``"relative"``, as the NIST StRD
certified standard deviations) and ``s² = 1`` when ``σ`` is given (``χ²_red`` is then the
goodness of fit, ≈ 1 for a consistent model). It is computed from the SVD of the
column-equilibrated ``W^{1/2} J`` (scaled parameters) and reported in SI: standard errors
``sqrt(diag C)``, the correlation matrix, the rank and the condition number. Singular values
below ``rcond`` times the largest make the problem rank deficient: the parameters with a
component above 0.1 in a null-space vector are reported as not identifiable (standard error
``inf``, covariance and correlation ``NaN``) and an :class:`IdentifiabilityWarning` is issued;
a condition number above ``max_condition`` gives the warning as well. The approximation is
not reliable for parameters at a bound (they are listed in ``at_bounds``, with a warning).
"""

from __future__ import annotations

import hashlib
import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import hpfem
from hpfem.opt.evaluator import Evaluation
from hpfem.opt.optimize import Problem, _Stop, as_study
from hpfem.opt.study import DesignSpace, EvaluationFailed, History, Study

_BOUND = 1e-12  # unit coordinates closer than this to 0 or 1 are at the bound


class IdentifiabilityWarning(UserWarning):
    """The Jacobian at the optimum is rank deficient or ill-conditioned: some parameters (or
    combinations) are not determined by the data."""


# --- Laplace approximation ---------------------------------------------------------------------


@dataclass
class Laplace:
    """Gaussian (Laplace) approximation of the parameter posterior at a least-squares optimum.

    ``covariance`` ``(n, n)``, ``std`` ``(n,)`` and ``correlation`` ``(n, n)`` in the units of
    the Jacobian's parameters (SI); ``chi2`` = ``‖W^{1/2} r‖²``, ``dof`` = ``m − rank``,
    ``chi2_red`` = ``chi2 / dof``; ``scaled`` whether the covariance was multiplied by
    ``factor`` = ``chi2_red`` (σ unknown); ``rank``, ``condition`` and ``singular_values`` of the
    column-equilibrated weighted Jacobian; ``non_identifiable`` the names of the parameters
    in its null space; ``warnings`` the messages issued."""

    names: list[str]
    covariance: np.ndarray
    std: np.ndarray
    correlation: np.ndarray
    chi2: float
    chi2_red: float
    dof: int
    scaled: bool
    factor: float
    rank: int
    condition: float
    singular_values: np.ndarray
    non_identifiable: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def errors(self) -> dict[str, float]:
        """Standard errors by parameter name."""
        return dict(zip(self.names, (float(s) for s in self.std), strict=True))

    @property
    def identifiable(self) -> bool:
        """Full rank: every parameter is determined by the data."""
        return self.rank == len(self.names)


def _sigma_array(sigma, m: int) -> np.ndarray | None:
    if sigma is None:
        return None
    s = np.broadcast_to(np.asarray(sigma, dtype=float), (m,)).copy()
    if not np.all(np.isfinite(s) & (s > 0)):
        raise ValueError("sigma: the standard deviations must be positive and finite")
    return s


def laplace(jacobian, residual, sigma=None, *, names: Sequence[str] | None = None,
            scale_covariance: bool | None = None, rcond: float = 1e-8,
            max_condition: float = 1e8, warn: bool = True) -> Laplace:  # fmt: skip
    """The Laplace approximation at a least-squares optimum, usable without a study.

    ``jacobian`` ``(m, n)`` is ``∂y/∂p`` at the optimum (any units, the results are in the
    same ones), ``residual`` ``(m,)`` is ``y(p) − y_meas`` (unweighted), ``sigma`` ``None``,
    a number or ``(m,)`` standard deviations of the measurements. ``scale_covariance``
    (default: ``sigma is None``) multiplies ``(JᵀWJ)⁻¹`` by ``χ²_red``. ``rcond`` is the
    relative singular-value threshold of the rank, ``max_condition`` the condition number
    above which the problem counts as ill-conditioned; both warn with
    :class:`IdentifiabilityWarning` unless ``warn=False``. See the module docstring."""
    jac = np.asarray(jacobian, dtype=float)
    if jac.ndim == 1:
        jac = jac[:, None]
    if jac.ndim != 2:
        raise ValueError(f"laplace: jacobian of shape {jac.shape}, expected (m, n)")
    m, n = jac.shape
    r = np.asarray(residual, dtype=float).ravel()
    if r.shape != (m,):
        raise ValueError(f"laplace: residual of length {len(r)} for a ({m}, {n}) Jacobian")
    if not (np.all(np.isfinite(jac)) and np.all(np.isfinite(r))):
        raise ValueError("laplace: non-finite Jacobian or residual")
    names = [f"p{j}" for j in range(n)] if names is None else [str(s) for s in names]
    if len(names) != n:
        raise ValueError(f"laplace: {len(names)} names for {n} parameters")
    s = _sigma_array(sigma, m)
    if scale_covariance is None:
        scale_covariance = s is None
    w = np.ones(m) if s is None else 1.0 / s
    a = w[:, None] * jac
    rw = w * r
    chi2 = float(rw @ rw)
    norms = np.linalg.norm(a, axis=0)
    c = np.where(norms > 0, norms, 1.0)
    _, sv, vt = np.linalg.svd(a / c, full_matrices=False)
    largest = float(sv[0]) if sv.size else 0.0
    rank = int(np.sum(sv > rcond * largest)) if largest > 0 else 0
    condition = largest / float(sv[-1]) if sv.size and sv[-1] > 0 else math.inf
    dof = m - rank
    chi2_red = chi2 / dof if dof > 0 else math.nan
    factor = chi2_red if scale_covariance else 1.0
    v = vt.T
    inverse = (v[:, :rank] / sv[:rank] ** 2) @ v[:, :rank].T
    cov = inverse / np.outer(c, c) * factor
    bad = np.sum(v[:, rank:] ** 2, axis=1) > 0.01 if rank < n else np.zeros(n, dtype=bool)
    cov[bad, :] = np.nan
    cov[:, bad] = np.nan
    cov[bad, bad] = np.inf
    std = np.sqrt(np.diag(cov))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.outer(std, std)
    good = np.isfinite(std) & (std > 0)
    corr[~good, :] = np.nan
    corr[:, ~good] = np.nan
    corr[good, good] = 1.0
    non_identifiable = [nm for nm, b in zip(names, bad, strict=True) if b]
    messages = []
    if rank < n:
        messages.append(
            f"laplace: the weighted Jacobian has rank {rank} < {n} (singular values below "
            f"{rcond:g} of the largest); not identifiable: {non_identifiable} (standard error "
            "inf)"
        )
    elif condition > max_condition:
        messages.append(
            f"laplace: ill-conditioned weighted Jacobian (condition number {condition:.3g} "
            f"> {max_condition:g} after column scaling); the parameters are nearly "
            "non-identifiable, their uncertainties large and strongly correlated"
        )
    if scale_covariance and dof <= 0:
        messages.append(
            f"laplace: no degrees of freedom left ({m} observables, rank {rank}) to estimate "
            "the noise level; covariance NaN"
        )
    if warn:
        for message in messages:
            warnings.warn(message, IdentifiabilityWarning, stacklevel=2)
    return Laplace(
        names=names,
        covariance=cov,
        std=std,
        correlation=corr,
        chi2=chi2,
        chi2_red=chi2_red,
        dof=dof,
        scaled=bool(scale_covariance),
        factor=factor,
        rank=rank,
        condition=condition,
        singular_values=sv,
        non_identifiable=non_identifiable,
        warnings=messages,
    )


# --- fit -------------------------------------------------------------------------------------


@dataclass
class FitResult:
    """Result of :func:`fit`.

    ``params`` the best point (SI, every parameter including the fixed ones); ``names`` /
    ``x`` the free parameters and their SI values; ``std`` their standard errors,
    ``covariance`` and ``correlation`` (SI, :class:`Laplace` in ``laplace``); ``cost`` =
    ``½‖W^{1/2}(y − y_meas)‖²``, ``chi2`` = ``2 cost``, ``chi2_red``, ``dof``; ``values`` the
    fitted observables ``y(p)``, ``residual`` = ``y − y_meas``, ``jacobian`` ``∂y/∂p`` (SI, free
    columns), ``sigma`` the standard deviations used (``None``: unweighted); ``observables``
    their names; ``at_bounds`` the free parameters at a bound; ``iterations`` (trial steps),
    ``evaluations`` (calls, cache hits included), ``new_evaluations``, ``cache_hits``,
    ``failures``, ``restarts`` (remeshes); ``success`` / ``message``; ``warnings`` (Laplace and
    bounds); the ``study`` and the final ``evaluation``."""

    params: dict[str, Any]
    names: list[str]
    x: np.ndarray
    std: np.ndarray
    covariance: np.ndarray
    correlation: np.ndarray
    cost: float
    chi2: float
    chi2_red: float
    dof: int
    values: np.ndarray
    residual: np.ndarray
    jacobian: np.ndarray
    sigma: np.ndarray | None
    observables: list[str]
    at_bounds: list[str]
    laplace: Laplace
    iterations: int
    evaluations: int
    new_evaluations: int
    cache_hits: int
    failures: int
    restarts: int
    success: bool
    message: str
    method: str
    warnings: list[str]
    study: Study
    evaluation: Evaluation
    fidelity: dict

    @property
    def errors(self) -> dict[str, float]:
        """Standard errors by parameter name (SI)."""
        return self.laplace.errors

    def history(self, status: str | None = "ok") -> History:
        """The study's evaluations at this fidelity (:meth:`Study.history`)."""
        return self.study.history(status, fidelity=self.fidelity)

    def summary(self) -> str:
        """A table of the parameters with standard errors, and the fit statistics."""
        lines = [f"{'parameter':<16} {'value':>16} {'std error':>12} {'rel':>9}"]
        for name, x, s in zip(self.names, self.x, self.std, strict=True):
            rel = abs(s / x) if x != 0 else math.inf
            lines.append(f"{name:<16} {x:>16.9g} {s:>12.4g} {rel:>9.2e}")
        lines.append(
            f"cost {self.cost:.6g}, chi2_red {self.chi2_red:.6g} (dof {self.dof}), "
            f"{self.evaluations} evaluations, {self.message}"
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"FitResult({dict(zip(self.names, self.x, strict=True))}, std={self.std}, "
            f"chi2_red={self.chi2_red:.6g}, success={self.success}, {self.message!r})"
        )


_FIT_METHODS = {
    "lm": "levenberg-marquardt",
    "levenberg-marquardt": "levenberg-marquardt",
    "gn": "gauss-newton",
    "gauss-newton": "gauss-newton",
}


@dataclass
class _Point:
    u: np.ndarray
    params: dict
    evaluation: Evaluation
    y: np.ndarray
    r: np.ndarray  # weighted residual
    jx: np.ndarray  # dy/dp, SI, free columns, unweighted
    ju: np.ndarray  # weighted, unit coordinates
    cost: float
    remeshed: bool


class _Fit:
    def __init__(self, problem: Problem, rows: list[int], y_meas: np.ndarray,
                 w: np.ndarray, max_evaluations: int | None, max_failures: int):  # fmt: skip
        self.problem = problem
        self.study = problem.study
        self.rows = rows
        self.y_meas = y_meas
        self.w = w
        self.max_evaluations = max_evaluations
        self.max_failures = int(max_failures)
        self.nfev = 0
        self.failures = 0
        self.restarts = 0
        self.mesh_id: int | None = None
        self.size0 = len(self.study)
        self.hits0 = self.study.cache_hits

    def cost_of(self, values: np.ndarray) -> float:
        r = self.w * (np.asarray(values)[self.rows] - self.y_meas)
        c = 0.5 * float(r @ r)
        return c if math.isfinite(c) else math.inf

    def point(self, u: np.ndarray) -> tuple[_Point | None, str]:
        if self.max_evaluations is not None and self.nfev >= self.max_evaluations:
            raise _Stop(f"maximum number of evaluations ({self.max_evaluations}) reached")
        self.nfev += 1
        before = self.problem.sizes()
        params, e = self.problem.evaluate(u, True)
        if e is None:
            return None, "infeasible point"
        if not e.ok:
            return None, str(e.meta.get("error", e.status))
        y = e.values[self.rows]
        jx = self.problem.free_jacobian(e)[self.rows]
        r = self.w * (y - self.y_meas)
        if not (np.all(np.isfinite(r)) and np.all(np.isfinite(jx))):
            return None, "non-finite values or Jacobian"
        ju = self.w[:, None] * jx * self.problem.dxdu(params)
        remeshed = self.problem.remeshed(before, e, self.mesh_id)
        if self.mesh_id is None or remeshed:
            self.mesh_id = e.mesh_id
        return _Point(u, params, e, y, r, jx, ju, 0.5 * float(r @ r), remeshed), ""

    def failed(self, why: str) -> None:
        self.failures += 1
        if self.failures > self.max_failures:
            raise _Stop(f"stopped after {self.failures} failed or infeasible points ({why})")


def _rows(observables, names: list[str]) -> list[int]:
    if observables is None:
        return list(range(len(names)))
    rows = []
    for o in observables:
        if isinstance(o, str):
            if o not in names:
                raise ValueError(f"fit: unknown observable {o!r} ({names})")
            rows.append(names.index(o))
        else:
            i = int(o)
            if not -len(names) <= i < len(names):
                raise ValueError(f"fit: observable index {i} for {len(names)} observables")
            rows.append(i % len(names))
    if len(set(rows)) != len(rows):
        raise ValueError("fit: an observable is selected twice")
    return rows


def fit(study, y_meas, sigma=None, *, x0=None, fixed: Mapping[str, Any] | None = None,
        observables: Sequence[int | str] | None = None, method: str = "lm",
        fidelity: Mapping | None = None, max_iterations: int = 200,
        max_evaluations: int | None = None, xtol: float = 1e-10, gtol: float = 1e-10,
        ftol: float = 1e-14, scale_covariance: bool | None = None, rcond: float = 1e-8,
        max_condition: float = 1e8, max_failures: int = 20, max_restarts: int = 10,
        callback: Callable[[dict], Any] | None = None,
        space: DesignSpace | Sequence | None = None,
        path: str | Path | None = None) -> FitResult:  # fmt: skip
    """Least-squares reconstruction of the free parameters from measured observables, with
    the Laplace uncertainties (see the module docstring for the algorithm and conventions).

    ``study`` is a :class:`~hpfem.opt.Study` or an evaluator / function (a study is made with
    ``space`` and the store ``path``, in memory without); the evaluator must return the
    Jacobian (it is called with ``jacobian=True`` at every point). ``y_meas`` are the measured
    values of ``observables`` (indices or names, default all, in order); ``sigma`` their
    standard deviations (a number or an array; ``None``: unweighted, ``"relative"``:
    ``|y_meas|``). ``x0`` the start (dict of the free parameters in SI or a sequence of their
    values; ``None``: the stored evaluation with the smallest cost, else the centre of the
    box); ``fixed`` parameters held at a value; ``method`` ``"lm"`` (Levenberg–Marquardt) or
    ``"gn"`` (Gauss–Newton). ``max_iterations`` bounds the trial steps, ``max_evaluations``
    the evaluations; ``xtol``, ``gtol``, ``ftol`` are the convergence tolerances;
    ``scale_covariance`` (default: σ unknown), ``rcond`` and ``max_condition`` go to
    :func:`laplace`. Every evaluation goes through the study (cached, stored, resumable); a
    ``"state"`` checkpoint (``x``, ``params``, ``cost``, ``mu``, the hash of ``y_meas``) is
    written after every accepted step. ``callback(info)`` gets ``iteration``, ``params``,
    ``cost`` after every accepted step; a true return value stops the fit. A failed start
    point raises :class:`~hpfem.opt.EvaluationFailed`; ``hpfem.Cancelled`` propagates after a
    ``"cancelled"`` checkpoint."""
    study = as_study(study, space, path)
    problem = Problem(study, fixed, fidelity)
    rows = _rows(observables, list(study.observables))
    y_meas = np.asarray(y_meas, dtype=float).ravel()
    if len(y_meas) != len(rows):
        raise ValueError(f"fit: {len(y_meas)} measured values for {len(rows)} observables")
    if not np.all(np.isfinite(y_meas)):
        raise ValueError("fit: y_meas must be finite")
    if isinstance(sigma, str):
        if sigma != "relative":
            raise ValueError(f"fit: sigma {sigma!r}, use a number, an array, None or 'relative'")
        if np.any(y_meas == 0):
            raise ValueError("fit: sigma='relative' needs non-zero measured values")
        s = np.abs(y_meas)
        unknown_noise = True
    else:
        s = _sigma_array(sigma, len(rows))
        unknown_noise = s is None
    if scale_covariance is None:
        scale_covariance = unknown_noise
    w = np.ones(len(rows)) if s is None else 1.0 / s
    key = str(method).lower().replace(" ", "-").replace("_", "-")
    if key not in _FIT_METHODS:
        raise ValueError(f"fit: method {method!r}, use 'lm' or 'gn'")
    method = _FIT_METHODS[key]
    lm = method == "levenberg-marquardt"
    run = _Fit(problem, rows, y_meas, w,
               None if max_evaluations is None else int(max_evaluations), max_failures)  # fmt: skip
    data_hash = hashlib.sha256(
        np.concatenate([y_meas, w, np.asarray(rows, dtype=float)]).tobytes()
    ).hexdigest()[:16]
    names = [study.observables[i] for i in rows]
    mu0 = 1e-3
    mu, nu = mu0, 2.0
    nit = 0
    success, message, status = False, "", "done"
    cur: _Point | None = None

    def checkpoint(state_status: str) -> None:
        state: dict[str, Any] = {"status": state_status, "free": problem.names,
                                 "fixed": problem.fixed, "fidelity": problem.fidelity,
                                 "observables": names, "data": data_hash, "mu": mu, "nu": nu,
                                 "nfev": run.nfev, "failures": run.failures,
                                 "restarts": run.restarts}  # fmt: skip
        if cur is not None:
            state.update(x=cur.u, params=cur.params, cost=cur.cost)
        study.checkpoint(method, nit, state)

    try:
        u0 = problem.start(x0, run.cost_of)
        cur, why = run.point(u0)
        if cur is None:
            raise EvaluationFailed(f"fit: the evaluation at the start point failed ({why})")
        scaling = np.zeros(problem.n)
        while True:
            if nit >= max_iterations:
                message = f"maximum number of iterations ({max_iterations}) reached"
                break
            g = cur.ju.T @ cur.r
            rnorm = float(np.linalg.norm(cur.r))
            if rnorm == 0.0:
                success, message = True, "zero residual"
                break
            outward = ((cur.u <= _BOUND) & (g > 0)) | ((cur.u >= 1.0 - _BOUND) & (g < 0))
            free = ~outward
            norms = np.linalg.norm(cur.ju, axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                cosine = np.where(norms > 0, np.abs(g) / (norms * rnorm), 0.0)
            if not free.any() or float(np.max(cosine[free])) <= gtol:
                success, message = True, "converged: gradient (gtol)"
                break
            scaling = np.maximum(scaling, norms**2)
            d = np.where(scaling > 0, scaling, 1.0)
            jf = cur.ju[:, free]
            nit += 1
            if lm:
                a = np.vstack([jf, np.diag(np.sqrt(mu * d[free]))])
                b = np.concatenate([-cur.r, np.zeros(int(free.sum()))])
            else:
                a, b = jf, -cur.r
            delta = np.zeros(problem.n)
            delta[free] = np.linalg.lstsq(a, b, rcond=None)[0]
            alpha = 1.0
            converged_cost = False
            while True:  # Gauss-Newton step halving; LM: a single pass
                trial_u = np.clip(cur.u + alpha * delta, 0.0, 1.0)
                step = trial_u - cur.u
                if float(np.linalg.norm(step)) <= xtol * (float(np.linalg.norm(cur.u)) + xtol):
                    trial = None
                    message = "converged: step (xtol)"
                    break
                predicted = cur.cost - 0.5 * float(np.sum((cur.r + cur.ju @ step) ** 2))
                trial, why = run.point(trial_u)
                if trial is not None and trial.remeshed:
                    break
                if trial is None:
                    run.failed(why)
                    rho = -math.inf
                else:
                    actual = cur.cost - trial.cost
                    rho = actual / predicted if predicted > 0 else -math.inf
                if lm or rho > 1e-4:
                    break
                alpha *= 0.5
                if alpha < 1e-10:
                    trial = None
                    message = "Gauss-Newton line search found no decrease"
                    break
            if trial is None and message:
                success = message.startswith("converged")
                break
            if trial is not None and trial.remeshed:  # restart on the new reference mesh
                cur = trial
                mu, nu = mu0, 2.0
                run.restarts += 1
                checkpoint("restart")
                if run.restarts > max_restarts:
                    message = f"stopped after {max_restarts} restarts on remeshes"
                    break
                continue
            if rho > 1e-4:
                converged_cost = (
                    actual <= ftol * cur.cost and predicted <= ftol * cur.cost and rho <= 2.0
                )
                cur = trial
                if lm:
                    mu *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
                    nu = 2.0
                checkpoint("running")
                if callback is not None and callback(
                    {"iteration": nit, "params": cur.params, "cost": cur.cost}
                ):
                    message = "stopped by the callback"
                    break
                if converged_cost:
                    success, message = True, "converged: relative cost decrease (ftol)"
                    break
            else:
                mu *= nu
                nu *= 2.0
                if mu > 1e30:
                    message = "no further decrease (damping exhausted)"
                    break
    except hpfem.Cancelled:
        checkpoint("cancelled")
        raise
    except _Stop as stop:
        message, status = str(stop), "stopped"
        if cur is None:
            checkpoint(status)
            raise EvaluationFailed(f"fit: {stop}") from None
    if not success and status == "done":
        status = "stopped"
    checkpoint(status)

    lap = laplace(cur.jx, cur.y - y_meas, s, names=problem.names,
                  scale_covariance=scale_covariance, rcond=rcond, max_condition=max_condition,
                  warn=False)  # fmt: skip
    messages = list(lap.warnings)
    at_bounds = [
        n for n, ui in zip(problem.names, cur.u, strict=True)
        if ui <= _BOUND or ui >= 1.0 - _BOUND
    ]  # fmt: skip
    if at_bounds:
        messages.append(
            f"fit: parameters at a bound {at_bounds}; the Laplace approximation is not "
            "reliable for them"
        )
    for text in messages:
        warnings.warn(text, IdentifiabilityWarning, stacklevel=2)
    return FitResult(
        params=dict(cur.params),
        names=list(problem.names),
        x=problem.x(cur.params),
        std=lap.std,
        covariance=lap.covariance,
        correlation=lap.correlation,
        cost=cur.cost,
        chi2=lap.chi2,
        chi2_red=lap.chi2_red,
        dof=lap.dof,
        values=cur.y,
        residual=cur.y - y_meas,
        jacobian=cur.jx,
        sigma=s,
        observables=names,
        at_bounds=at_bounds,
        laplace=lap,
        iterations=nit,
        evaluations=run.nfev,
        new_evaluations=len(study) - run.size0,
        cache_hits=study.cache_hits - run.hits0,
        failures=run.failures,
        restarts=run.restarts,
        success=success,
        message=message,
        method=method,
        warnings=messages,
        study=study,
        evaluation=cur.evaluation,
        fidelity=dict(problem.fidelity),
    )


__all__ = ["FitResult", "IdentifiabilityWarning", "Laplace", "fit", "laplace"]

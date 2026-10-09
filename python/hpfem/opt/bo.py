"""Bayesian optimisation driving a study (M16 S5, ADR-0012 §1, §5, §6, §7).

:func:`bayesian_optimize` minimises (or maximises) an objective over the free continuous
parameters of a :class:`hpfem.opt.Study` with the Gaussian process of :mod:`hpfem.opt.gp` as
surrogate. Every point goes through the study: cached, appended to the JSON-lines store,
cancellable (``hpfem.Cancelled``) and resumable. Coordinates, objectives and fixed parameters
follow :func:`hpfem.opt.minimize`: the free parameters in the unit-cube coordinates of the
design space (logarithmic for ``log=True``), the objective an observable (index or name) or a
function of the observable vector returning ``F`` or ``(F, dF/dy)``; integer and categorical
parameters must be fixed.

**Loop.** An initial design of ``n_initial`` points (default ``2 (d + 1)``; Latin hypercube or
scrambled Sobol' from :meth:`DesignSpace.sample`, seeded, infeasible points rejected) is
evaluated; then, one point per iteration (sequential, ADR-0012 §7): fit the surrogate(s) on
*every* successful evaluation of the study with the same fixed values and fidelity (also those
of earlier runs or other optimisers — the surrogate keeps all points, ADR-0012 §3), maximise
the acquisition, record the point as a ``"proposal"``, evaluate it, write a ``"state"``
checkpoint. Exact repeats (unit-cube distance below ``1e-6`` to a known point) are never
proposed; the next-best candidate is taken instead.

**Acquisition** (``F`` to be minimised, posterior mean ``μ`` and standard deviation ``σ`` of
the surrogate of ``F``, incumbent ``F*`` = the best observed feasible value):

- expected improvement (``"ei"``, default), maximised in log form,
  ``EI = σ h(z)``, ``z = (F* − ξ − μ)/σ``, ``h(z) = z Φ(z) + φ(z)``, with ``log h`` evaluated
  stably for ``z ≪ 0`` (Ament et al. 2023, "LogEI": ``log h = −z²/2 − ½ log 2π +
  log1mexp(log(erfcx(−z/√2) |z|) + ½ log(π/2))``, and the asymptote
  ``−z²/2 − ½ log 2π − 2 log|z|`` below ``z = −1/√ε``) — EI underflows to 0 far from the
  incumbent, its logarithm does not, so the multi-start optimiser still sees a slope;
- lower confidence bound (``"lcb"``), minimise ``μ − κ σ`` (``kappa = 2``).

**Constraints.** Known constraints (the linear and nonlinear constraints of the design space)
restrict the acquisition optimisation: candidates are drawn feasible, the local optimisation
runs SLSQP with the constraints (L-BFGS-B in the box without constraints), and an infeasible
result is replaced by the best feasible candidate; infeasible points are never evaluated.
Unknown (expensive) constraints are :class:`OutcomeConstraint` s ``lower ≤ g(y) ≤ upper`` on
an observable or a function of the values; each gets its own GP and enters EI as the
probability of feasibility, ``log EI + Σ log P(lower ≤ g ≤ upper)`` (Gardner et al. 2014);
until a feasible point is observed the acquisition is ``Σ log P`` alone. LCB does not combine
with outcome constraints (``ValueError``).

**Gradients.** ``use_gradients=True`` evaluates with ``jacobian=True`` and fits the
gradient-enhanced GP: ``dF/du = (dF/dy)ᵀ J_free · dp/du`` (the chain rule of
:func:`~hpfem.opt.minimize`); evaluations without a Jacobian enter with values only. The
acquisition is always optimised with the analytic gradients of the GP prediction.

**Stopping.** ``max_evaluations`` objective calls (initial design included, cache hits
included, as :func:`~hpfem.opt.minimize`), ``patience`` iterations without a better incumbent,
``tol`` on the predicted gain (EI: the maximal expected (feasible) improvement; LCB:
``F* − min LCB``), ``max_failures`` failed evaluations, or a true return of ``callback(info)``.

**Failures** (``status="failed"``, non-finite objective): the point stays in the surrogate of
the objective with the *worst* successful value (no derivatives), which steers the search away
without distorting the scale; it is never proposed again (the study would return the cached
failure). Outcome-constraint surrogates use the successful points only.

**Remesh** (ADR-0012 §3): a surrogate keeps every point; a remesh is counted (``remeshes``)
and noted in the checkpoints, the jump of the size of the discretisation error is absorbed by
the learned noise of the GP (``noise="learn"``, the default).

**DWR estimate** (ADR-0012 §6): ``Evaluation.error`` is not used — neither as GP noise nor
otherwise; the systematic discretisation error is controlled by the evaluator's fidelity.

**Checkpoints and resume.** A ``"state"`` record (method ``"bayes-ei"`` / ``"bayes-lcb"``)
after the initial design and after every iteration holds the status (``"running"``,
``"done"``, ``"stopped"``, ``"cancelled"``), the phase, the seed, the counts, the
hyperparameters of the surrogates (the warm start of the next fit), the incumbent and the last
predicted gain. Calling :func:`bayesian_optimize` again on a study whose last checkpoint is a
running or cancelled run of the same method and problem resumes it: the open proposal is
evaluated first, the iteration counter, the budget used and the seed are those of the
checkpoint, and every random choice is seeded by ``(seed, iteration)``, so the resumed run
proposes the points the uninterrupted run would have. Otherwise a new run starts (its
initial design comes from the cache when the same seed was used before).

**Limits.** The GP is dense: a few hundred evaluations (``n (d + 1)`` observations of a few
hundred with gradients). One point per iteration (no batch acquisition).

**Optional BoTorch path** (extra ``opt-bo``, imported lazily): :func:`pareto_optimize`
(multi-objective, qLogNEHVI) and :func:`multi_fidelity_optimize` (discrete fidelities,
cost-aware multi-fidelity knowledge gradient) drive the same study. Experimental: written
against BoTorch's documented API but not yet run (BoTorch is installed neither on the
development machine nor in CI; the tests skip). :func:`pareto_front` /
:func:`non_dominated` (no BoTorch) give the Pareto front of any study.
"""

from __future__ import annotations

import inspect
import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.optimize
import scipy.special

import hpfem
from hpfem.opt.evaluator import Evaluation
from hpfem.opt.gp import GaussianProcess
from hpfem.opt.optimize import OptimizeResult, Problem, _Objective, as_study
from hpfem.opt.study import DesignSpace, EvaluationFailed, Study, _qmc_engine, to_jsonable

ACQUISITIONS = ("ei", "lcb")
"""Acquisition functions of :func:`bayesian_optimize`."""

_C1 = 0.5 * math.log(2.0 * math.pi)
_C2 = 0.5 * math.log(0.5 * math.pi)
_SQRT2 = math.sqrt(2.0)
_FAR = -1.0 / math.sqrt(np.finfo(float).eps)
_MIN_DISTANCE = 1e-6


# --- acquisition functions ---------------------------------------------------------------------


def _log1mexp(x: np.ndarray) -> np.ndarray:
    """``log(1 − exp(x))`` for ``x < 0``."""
    x = np.asarray(x, dtype=float)
    small = x > -math.log(2.0)
    out = np.empty_like(x)
    out[small] = np.log(-np.expm1(x[small]))
    out[~small] = np.log1p(-np.exp(x[~small]))
    return out


def _log_h(z) -> np.ndarray:
    """``log(z Φ(z) + φ(z))``, stable for all ``z`` (LogEI)."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    upper = z > -1.0
    zu = z[upper]
    out[upper] = np.log(zu * scipy.special.ndtr(zu) + np.exp(-0.5 * zu * zu - _C1))
    mid = ~upper & (z > _FAR)
    zm = z[mid]
    out[mid] = (
        -0.5 * zm * zm
        - _C1
        + _log1mexp(np.log(scipy.special.erfcx(-zm / _SQRT2) * np.abs(zm)) + _C2)
    )
    far = ~upper & ~mid
    zf = z[far]
    out[far] = -0.5 * zf * zf - _C1 - 2.0 * np.log(np.abs(zf))
    return out


def log_expected_improvement(mean, std, best: float, xi: float = 0.0, *, gradient: bool = False):
    """``log EI`` of minimisation, ``EI = E[max(best − ξ − f, 0)]`` for ``f ~ N(mean, std²)``
    (stable for tiny ``EI``); with ``gradient`` also ``(d/d mean, d/d std)`` of ``log EI``."""
    mean = np.asarray(mean, dtype=float)
    std = np.maximum(np.asarray(std, dtype=float), 1e-300)
    z = (best - xi - mean) / std
    log_h = _log_h(z)
    value = np.log(std) + log_h
    if not gradient:
        return value
    d_mean = -np.exp(scipy.special.log_ndtr(z) - log_h) / std
    d_std = np.exp(-0.5 * z * z - _C1 - log_h) / std
    return value, d_mean, d_std


def expected_improvement(mean, std, best: float, xi: float = 0.0) -> np.ndarray:
    """``EI = σ (z Φ(z) + φ(z))``, ``z = (best − ξ − mean)/σ`` (minimisation)."""
    return np.exp(log_expected_improvement(mean, std, best, xi))


def lower_confidence_bound(mean, std, kappa: float = 2.0) -> np.ndarray:
    """``LCB = mean − κ std`` (minimised)."""
    return np.asarray(mean, dtype=float) - kappa * np.asarray(std, dtype=float)


def _log_pof_interval(a, b, std, gradient: bool):
    """``log(Φ(b) − Φ(a))`` for ``a < b`` (and its derivatives), stable in both tails."""
    log_phi = lambda t: -0.5 * t * t - _C1  # noqa: E731
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        lb, la = scipy.special.log_ndtr(b), scipy.special.log_ndtr(a)
        lnb, lna = scipy.special.log_ndtr(-b), scipy.special.log_ndtr(-a)
        value = np.where(
            b <= 0.0,
            lb + _log1mexp(np.minimum(la - lb, -1e-300)),
            np.where(
                a >= 0.0,
                lna + _log1mexp(np.minimum(lnb - lna, -1e-300)),
                np.log1p(-(np.exp(la) + np.exp(lnb))),
            ),
        )
        if not gradient:
            return value
        rb, ra = np.exp(log_phi(b) - value), np.exp(log_phi(a) - value)
        return value, (ra - rb) / std, (a * ra - b * rb) / std


def log_probability_of_feasibility(
    mean, std, lower: float = -math.inf, upper: float = math.inf, *, gradient: bool = False
):
    """``log P(lower ≤ g ≤ upper)`` for ``g ~ N(mean, std²)`` (stable in the tails); with
    ``gradient`` also ``(d/d mean, d/d std)``."""
    mean = np.asarray(mean, dtype=float)
    std = np.maximum(np.asarray(std, dtype=float), 1e-300)
    log_phi = lambda t: -0.5 * t * t - _C1  # noqa: E731
    lo, hi = float(lower), float(upper)
    b = (hi - mean) / std if math.isfinite(hi) else np.full_like(mean, np.inf)
    a = (lo - mean) / std if math.isfinite(lo) else np.full_like(mean, -np.inf)
    if math.isfinite(hi) and math.isfinite(lo):
        return _log_pof_interval(a, b, std, gradient)
    if math.isfinite(hi):
        value = scipy.special.log_ndtr(b)
        if not gradient:
            return value
        r = np.exp(log_phi(b) - value)
        return value, -r / std, -b * r / std
    if math.isfinite(lo):
        value = scipy.special.log_ndtr(-a)
        if not gradient:
            return value
        r = np.exp(log_phi(a) - value)
        return value, r / std, a * r / std
    zero = np.zeros_like(mean)
    return (zero, zero, zero) if gradient else zero


# --- problem pieces ----------------------------------------------------------------------------


@dataclass
class OutcomeConstraint:
    """An unknown (expensive) constraint ``lower ≤ g(y) ≤ upper`` on the observables: ``g`` is
    an observable (index or name) or a function of the values vector returning ``g`` or
    ``(g, dg/dy)``. Modelled by its own Gaussian process; :func:`bayesian_optimize` weights
    the expected improvement with the probability of feasibility."""

    observable: int | str | Callable
    lower: float = -math.inf
    upper: float = math.inf
    name: str = ""

    def __post_init__(self):
        self.lower, self.upper = float(self.lower), float(self.upper)
        if not self.lower < self.upper:
            raise ValueError(f"OutcomeConstraint {self.name}: lower must be < upper")
        if not (math.isfinite(self.lower) or math.isfinite(self.upper)):
            raise ValueError(f"OutcomeConstraint {self.name}: give a finite lower or upper bound")

    def satisfied(self, g: float) -> bool:
        slack = 1e-12 * max(1.0, abs(g))
        return self.lower - slack <= g <= self.upper + slack


@dataclass(repr=False)
class BOResult(OptimizeResult):
    """Result of :func:`bayesian_optimize` (and of the BoTorch drivers): the fields of
    :class:`~hpfem.opt.OptimizeResult` (``iterations`` = acquisition steps after the initial
    design, ``restarts`` = 0) plus the final surrogate ``gp`` of the objective (on the free
    parameters' unit-cube coordinates, objective sign as minimised: negated for
    ``maximize``), the ``constraint_gps`` of the outcome constraints, ``acquisition`` (the
    predicted gain per iteration: EI, or ``F* − min LCB``), ``remeshes`` and ``feasible``
    (whether ``params`` satisfies the outcome constraints)."""

    gp: GaussianProcess | None = None
    constraint_gps: list = field(default_factory=list)
    acquisition: np.ndarray = field(default_factory=lambda: np.zeros(0))
    remeshes: int = 0
    feasible: bool = True


class _Stop(Exception):
    """Ends the run: ``status`` and message."""

    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


def _seed(seed) -> int:
    if seed is None:
        return int(np.random.SeedSequence().entropy % (2**63))
    if isinstance(seed, np.random.Generator):
        return int(seed.integers(2**63 - 1))
    return int(seed)


def _initial_design(problem: Problem, n: int, method: str, rng: np.random.Generator) -> list:
    """``n`` feasible points of the free parameters (the fixed ones at their values)."""
    space = problem.space
    if not problem.fixed:
        return space.sample(n, method, seed=rng)
    if method not in ("lhs", "sobol", "random"):
        raise ValueError(f"initial design {method!r}, use 'lhs', 'sobol' or 'random'")
    out, drawn = [], 0
    while len(out) < n:
        batch = max(2 * (n - len(out)), 16)
        if method == "random":
            u = rng.random((batch, problem.n))
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                u = _qmc_engine(method, problem.n, rng).random(batch)
        for row in u:
            params = problem.params(row)
            if problem.feasible(params):
                out.append(params)
                if len(out) == n:
                    break
        drawn += batch
        if len(out) < n and drawn > 1000 * n + 1000:
            raise ValueError(f"initial design: only {len(out)} of {n} feasible points")
    return out


def _matches(problem: Problem, e: Evaluation, fidelity: Mapping | None = None) -> bool:
    """A recorded (not cancelled) evaluation at ``fidelity`` (default: the problem's) with
    the problem's fixed values."""
    fidelity = problem.fidelity if fidelity is None else dict(fidelity)
    if e.status == "cancelled" or e.fidelity != fidelity:
        return False
    space = problem.space
    return all(space[n].key(e.params[n]) == space[n].key(v) for n, v in problem.fixed.items())


class _Data:
    """The training data of a problem gathered from its study."""

    def __init__(self, u, f, df, g, dg, failed, evaluations):
        self.u, self.f, self.df = u, f, df  # f NaN for failures
        self.g, self.dg = g, dg  # (n, c), (n, c, d)
        self.failed = failed
        self.evaluations = evaluations


# --- the run -----------------------------------------------------------------------------------


class _BayesRun:
    def __init__(
        self,
        problem: Problem,
        goal: _Objective,
        constraints: list[OutcomeConstraint],
        acquisition: str,
        *,
        use_gradients: bool,
        xi: float,
        kappa: float,
        max_evaluations: int,
        patience: int | None,
        tol: float | None,
        max_failures: int,
        callback,
        gp_options: Mapping,
        raw_samples: int | None,
        num_restarts: int,
    ):
        self.problem = problem
        self.study = problem.study
        self.goal = goal
        self.constraints = constraints
        self.constraint_goals = [
            _Objective(c.observable, self.study.observables, False) for c in constraints
        ]
        self.kind = acquisition
        self.method = f"bayes-{acquisition}"
        self.use_gradients = bool(use_gradients)
        self.xi, self.kappa = float(xi), float(kappa)
        self.max_evaluations = int(max_evaluations)
        self.patience = None if patience is None else int(patience)
        self.tol = None if tol is None else float(tol)
        self.max_failures = int(max_failures)
        self.callback = callback
        self.gp_options = dict(gp_options or {})
        d = problem.n
        self.raw_samples = int(raw_samples) if raw_samples else min(256 * d, 2048)
        self.num_restarts = int(num_restarts)
        self.nfev = 0
        self.nit = 0
        self.failures = 0
        self.remeshes = 0
        self.stale = 0
        self.size0 = len(self.study)
        self.hits0 = self.study.cache_hits
        self.thetas: list = []
        self.gains: list[float] = []
        self.gp: GaussianProcess | None = None
        self.constraint_gps: list[GaussianProcess] = []
        self.phase = "initial"
        self.initial: list[dict] = []
        self.seed = 0
        self.message = ""

    # data

    def _matches(self, e: Evaluation) -> bool:
        return _matches(self.problem, e)

    def data(self) -> _Data:
        chosen: dict[str, Evaluation] = {}
        space = self.problem.space
        for e in self.study.evaluations:
            if not self._matches(e):
                continue
            key = repr(space.key(e.params))
            old = chosen.get(key)
            if old is not None:
                if old.ok and not e.ok:
                    continue
                if old.ok and e.ok and old.jacobian is not None and e.jacobian is None:
                    continue
            chosen[key] = e
        evals = list(chosen.values())
        n, d, c = len(evals), self.problem.n, len(self.constraints)
        u = np.zeros((n, d))
        f = np.full(n, np.nan)
        df = np.full((n, d), np.nan)
        g = np.full((n, c), np.nan)
        dg = np.full((n, c, d), np.nan)
        failed = np.zeros(n, dtype=bool)
        for r, e in enumerate(evals):
            u[r] = self.problem.encode(e.params)
            if not e.ok:
                failed[r] = True
                continue
            grad = self.use_gradients and e.jacobian is not None
            value, dfdy = self.goal(e.values, grad)
            if not math.isfinite(value):
                failed[r] = True
                continue
            f[r] = value
            chain = None
            if grad:
                chain = self.problem.free_jacobian(e) * self.problem.dxdu(e.params)
                df[r] = dfdy @ chain
            for j, cg in enumerate(self.constraint_goals):
                gj, dgdy = cg(e.values, grad)
                g[r, j] = gj
                if grad and math.isfinite(gj):
                    dg[r, j] = dgdy @ chain
        df[~np.isfinite(df)] = np.nan
        dg[~np.isfinite(dg)] = np.nan
        return _Data(u, f, df, g, dg, failed, evals)

    def incumbent(self, data: _Data) -> int | None:
        best, best_f = None, math.inf
        for r in range(len(data.f)):
            if data.failed[r]:
                continue
            if not all(c.satisfied(data.g[r, j]) for j, c in enumerate(self.constraints)):
                continue
            if data.f[r] < best_f:
                best, best_f = r, data.f[r]
        return best

    # surrogates

    def _model(self, rng: np.random.Generator) -> GaussianProcess:
        options = dict(self.gp_options)
        options.setdefault("seed", int(rng.integers(2**31 - 1)))
        options.setdefault("restarts", 1)  # plus the warm start and the prior centre
        return GaussianProcess(**options)

    def fit(self, data: _Data, rng: np.random.Generator) -> None:
        ok = ~data.failed
        if not np.any(ok):
            raise _Stop("stopped", "no successful evaluation to fit the surrogate")
        f = data.f.copy()
        f[data.failed] = np.max(data.f[ok])  # failures: the worst successful value
        df = data.df if self.use_gradients else None
        warm = list(self.thetas) + [None] * (1 + len(self.constraints) - len(self.thetas))
        self.gp = self._model(rng).fit(data.u, f, df, theta=_theta(warm[0]))
        self.constraint_gps = []
        for j in range(len(self.constraints)):
            rows = ok & np.isfinite(data.g[:, j])
            dg = data.dg[rows, j] if self.use_gradients else None
            model = self._model(rng).fit(
                data.u[rows], data.g[rows, j], dg, theta=_theta(warm[j + 1])
            )
            self.constraint_gps.append(model)
        self.thetas = [self.gp.theta.tolist()] + [m.theta.tolist() for m in self.constraint_gps]

    # acquisition

    def acquisition(self, u: np.ndarray, best: float | None, grad: bool):
        """Acquisition values (q,) to maximise and, with ``grad``, their gradients (q, d)."""
        gp = self.gp
        floor = (1e-12 * gp.y_scale) ** 2
        if grad:
            mean, var, dmean, dvar = gp.predict(u, grad=True)
        else:
            mean, var = gp.predict(u)
        var_c = np.maximum(var, floor)
        std = np.sqrt(var_c)
        dstd = (
            np.where(var > floor, 1.0, 0.0)[:, None] * dvar / (2.0 * std[:, None]) if grad else None
        )
        if self.kind == "lcb":
            value = -(mean - self.kappa * std)
            return value, (-(dmean - self.kappa * dstd) if grad else None)
        value = np.zeros(len(u))
        gradient = np.zeros_like(u) if grad else None
        if best is not None:
            if grad:
                v, dm, ds = log_expected_improvement(mean, std, best, self.xi, gradient=True)
                gradient += dm[:, None] * dmean + ds[:, None] * dstd
            else:
                v = log_expected_improvement(mean, std, best, self.xi)
            value += v
        for c, model in zip(self.constraints, self.constraint_gps, strict=True):
            if grad:
                m, s2, dm_c, ds2 = model.predict(u, grad=True)
            else:
                m, s2 = model.predict(u)
            fl = (1e-12 * model.y_scale) ** 2
            s2c = np.maximum(s2, fl)
            s = np.sqrt(s2c)
            if grad:
                v, dm, ds = log_probability_of_feasibility(m, s, c.lower, c.upper, gradient=True)
                dsc = np.where(s2 > fl, 1.0, 0.0)[:, None] * ds2 / (2.0 * s[:, None])
                gradient += dm[:, None] * dm_c + ds[:, None] * dsc
            else:
                v = log_probability_of_feasibility(m, s, c.lower, c.upper)
            value += v
        return value, gradient

    def gain(self, value: float, best: float | None) -> float:
        """The predicted gain of the chosen point (objective units)."""
        if best is None:
            return math.nan
        if self.kind == "lcb":
            return float(best + value)  # F* − LCB
        return float(math.exp(value))

    def _known_constraints(self):
        space = self.problem.space
        if not space.constraints:
            return None

        def values(u):
            params = self.problem.params(np.clip(u, 0.0, 1.0))
            out = []
            for c in space.constraints:
                v = np.atleast_1d(np.asarray(c.value(params), dtype=float))
                lo = np.broadcast_to(np.asarray(c.lower, dtype=float), v.shape)
                hi = np.broadcast_to(np.asarray(c.upper, dtype=float), v.shape)
                out.extend((v - lo)[np.isfinite(lo)])
                out.extend((hi - v)[np.isfinite(hi)])
            return np.array(out, dtype=float)

        return [{"type": "ineq", "fun": values}]

    def propose(self, data: _Data, best_row: int | None, rng: np.random.Generator):
        """The next point (unit coordinates) and its acquisition value."""
        d = self.problem.n
        best = None if best_row is None else float(data.f[best_row])
        engine = _qmc_engine("sobol", d, rng)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            candidates = [engine.random(self.raw_samples)]
        if best_row is not None:  # local candidates around the incumbent
            centre = data.u[best_row]
            k = max(self.raw_samples // 8, 8)
            for scale in (0.1, 0.01, 0.001):
                candidates.append(np.clip(centre + scale * rng.standard_normal((k, d)), 0.0, 1.0))
        raw = np.concatenate(candidates)
        constraints = self._known_constraints()
        if constraints is not None:
            raw = raw[[self.problem.feasible(self.problem.params(r)) for r in raw]]
            if not len(raw):
                raise _Stop("stopped", "no feasible candidate for the acquisition optimisation")
        values, _ = self.acquisition(raw, best, False)
        order = np.argsort(-values)
        starts = []
        for i in order:
            if all(np.max(np.abs(raw[i] - s)) > 1e-3 for s in starts):
                starts.append(raw[i])
            if len(starts) == self.num_restarts:
                break

        def negative(v):
            a, g = self.acquisition(np.clip(v, 0.0, 1.0)[None, :], best, True)
            return -float(a[0]), -g[0]

        found = [(float(values[i]), raw[i]) for i in order[: 4 * self.num_restarts]]
        for start in starts:
            try:
                if constraints is None:
                    res = scipy.optimize.minimize(
                        negative,
                        start,
                        jac=True,
                        method="L-BFGS-B",
                        bounds=[(0.0, 1.0)] * d,
                        options={"maxiter": 200},
                    )
                else:
                    res = scipy.optimize.minimize(
                        negative,
                        start,
                        jac=True,
                        method="SLSQP",
                        bounds=[(0.0, 1.0)] * d,
                        constraints=constraints,
                        options={"maxiter": 200},
                    )
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                continue
            u = np.clip(np.asarray(res.x, dtype=float), 0.0, 1.0)
            if np.isfinite(res.fun) and (
                constraints is None or self.problem.feasible(self.problem.params(u))
            ):
                found.append((-float(res.fun), u))
        found.sort(key=lambda t: -t[0])
        for value, u in found:
            if len(data.u) == 0 or np.min(np.max(np.abs(data.u - u), axis=1)) > _MIN_DISTANCE:
                return u, value
        # every candidate repeats a known point: explore with a random feasible point
        for _ in range(1000):
            u = rng.random(d)
            if self.problem.feasible(self.problem.params(u)):
                return u, float(self.acquisition(u[None, :], best, False)[0][0])
        raise _Stop("stopped", "no new feasible point to propose")

    # evaluations

    def evaluate(self, params: dict, source: str) -> Evaluation:
        if self.nfev >= self.max_evaluations:
            raise _Stop("done", f"maximum number of evaluations ({self.max_evaluations}) reached")
        before = len(self.study.remeshes)
        self.study.propose(
            [params], jacobian=self.use_gradients, fidelity=self.problem.fidelity, source=source
        )
        e = self.study.evaluate(params, jacobian=self.use_gradients, fidelity=self.problem.fidelity)
        self.nfev += 1
        self.count(e, before)
        return e

    def count(self, e: Evaluation, remeshes_before: int) -> None:
        self.remeshes += len(self.study.remeshes) - remeshes_before
        bad = not e.ok or not math.isfinite(self.goal(e.values, False)[0])
        if bad:
            self.failures += 1
            if self.failures > self.max_failures:
                raise _Stop("stopped", f"stopped after {self.failures} failed evaluations")

    def run_initial(self) -> None:
        todo = [p for p in self.initial]
        remaining = self.max_evaluations - self.nfev
        if len(todo) > remaining:
            todo = todo[:remaining]
        before = len(self.study.remeshes)
        evaluations = self.study.run(
            todo,
            jacobian=self.use_gradients,
            fidelity=self.problem.fidelity,
            source=f"{self.method}-initial",
        )
        self.nfev += len(evaluations)
        for e in evaluations:
            self.count(e, before)
            before = len(self.study.remeshes)

    def evaluate_open(self) -> None:
        """The open proposals of this problem (a cancelled iteration), one iteration each."""
        for p in self.study.open_proposals():
            params = p["params"]
            probe = Evaluation(params, [], fidelity=p["fidelity"])
            if not _matches(self.problem, probe) or not self.problem.feasible(params):
                continue
            self.evaluate(params, self.method)
            self.nit += 1

    # checkpoints

    def state(self, status: str, data: _Data | None = None) -> dict:
        state: dict[str, Any] = {
            "status": status,
            "phase": self.phase,
            "seed": self.seed,
            "free": self.problem.names,
            "fixed": self.problem.fixed,
            "fidelity": self.problem.fidelity,
            "nfev": self.nfev,
            "failures": self.failures,
            "remeshes": self.remeshes,
            "stale": self.stale,
            "theta": self.thetas,
            "gain": self.gains[-1] if self.gains else None,
            "use_gradients": self.use_gradients,
        }
        if self.phase == "initial":
            state["initial"] = self.initial
        if data is not None:
            row = self.incumbent(data)
            if row is not None:
                e = data.evaluations[row]
                state["best"] = {
                    "x": data.u[row],
                    "params": e.params,
                    "value": self.goal.sign * float(data.f[row]),
                }
        return state

    def checkpoint(self, status: str, data: _Data | None = None) -> None:
        self.study.checkpoint(self.method, self.nit, self.state(status, data))

    def restore(self, record: Mapping) -> bool:
        """Takes over a running or cancelled checkpoint of the same problem."""
        state = record.get("state")
        if record.get("method") != self.method or not isinstance(state, Mapping):
            return False
        if state.get("status") not in ("running", "cancelled"):
            return False
        if (
            state.get("free") != self.problem.names
            or state.get("fidelity") != self.problem.fidelity
        ):
            return False
        if to_jsonable(state.get("fixed")) != to_jsonable(self.problem.fixed):
            return False
        if bool(state.get("use_gradients")) != self.use_gradients:
            return False
        self.seed = int(state["seed"])
        self.nit = int(record.get("iteration", 0))
        self.nfev = int(state.get("nfev", 0))
        self.failures = int(state.get("failures", 0))
        self.remeshes = int(state.get("remeshes", 0))
        self.stale = int(state.get("stale", 0))
        self.thetas = list(state.get("theta") or [])
        self.phase = str(state.get("phase", "iterate"))
        if self.phase == "initial":
            self.initial = [self.problem.space.validate(p) for p in state.get("initial", [])]
        return True

    # the loop

    def loop(self) -> None:
        if self.phase == "initial":
            self.run_initial()
            self.phase = "iterate"
            self.checkpoint("running", self.data())
        else:
            self.evaluate_open()
        while True:
            data = self.data()
            best_row = self.incumbent(data)
            if self.nfev >= self.max_evaluations:
                raise _Stop(
                    "done", f"maximum number of evaluations ({self.max_evaluations}) reached"
                )
            rng = np.random.default_rng([self.seed, self.nit + 1])
            try:
                self.fit(data, rng)
            except (np.linalg.LinAlgError, ValueError) as error:
                raise _Stop("stopped", f"surrogate fit failed: {error}") from error
            u, value = self.propose(data, best_row, rng)
            best = None if best_row is None else float(data.f[best_row])
            gain = self.gain(value, best)
            self.gains.append(gain)
            if self.tol is not None and math.isfinite(gain) and gain < self.tol:
                raise _Stop("done", f"predicted gain {gain:.3g} below tol={self.tol:g}")
            params = self.problem.params(u)
            e = self.evaluate(params, self.method)
            self.nit += 1
            after = self.data()
            new_row = self.incumbent(after)
            improved = new_row is not None and (best is None or after.f[new_row] < best)
            self.stale = 0 if improved else self.stale + 1
            self.checkpoint("running", after)
            if self.callback is not None:
                info = {
                    "iteration": self.nit,
                    "params": params,
                    "value": self.goal.sign * self.goal(e.values, False)[0] if e.ok else math.nan,
                    "evaluations": self.nfev,
                    "gain": gain,
                }
                if new_row is not None:
                    info["best_params"] = after.evaluations[new_row].params
                    info["best_value"] = self.goal.sign * float(after.f[new_row])
                if self.callback(info):
                    raise _Stop("stopped", "stopped by the callback")
            if self.patience is not None and self.stale >= self.patience:
                raise _Stop("done", f"no improvement in {self.patience} iterations")

    def result(self, status: str) -> BOResult:
        data = self.data()
        row = self.incumbent(data)
        feasible = row is not None
        if row is None:  # no feasible point: the best successful one
            ok = np.flatnonzero(~data.failed)
            if not len(ok):
                raise EvaluationFailed(
                    f"bayesian_optimize: no successful evaluation ({self.message})"
                )
            row = int(ok[np.argmin(data.f[ok])])
        e = data.evaluations[row]
        params = dict(e.params)
        return BOResult(
            params=params,
            value=self.goal.sign * float(data.f[row]),
            evaluation=e,
            success=status == "done" and feasible,
            message=self.message if feasible else f"{self.message} (no feasible point)",
            method=self.method,
            names=list(self.problem.names),
            x=self.problem.x(params),
            iterations=self.nit,
            evaluations=self.nfev,
            new_evaluations=len(self.study) - self.size0,
            cache_hits=self.study.cache_hits - self.hits0,
            failures=self.failures,
            infeasible=0,
            restarts=0,
            study=self.study,
            fidelity=dict(self.problem.fidelity),
            gp=self.gp,
            constraint_gps=list(self.constraint_gps),
            acquisition=np.array(self.gains, dtype=float),
            remeshes=self.remeshes,
            feasible=feasible,
        )


def _theta(value):
    return None if value is None else np.asarray(value, dtype=float)


def _acquisition_name(acquisition: str) -> str:
    key = str(acquisition).lower().replace("-", "").replace("_", "")
    aliases = {
        "ei": "ei",
        "logei": "ei",
        "expectedimprovement": "ei",
        "lcb": "lcb",
        "lowerconfidencebound": "lcb",
        "ucb": "lcb",
    }
    if key not in aliases:
        raise ValueError(
            f"bayesian_optimize: acquisition {acquisition!r}, use one of {ACQUISITIONS}"
        )
    return aliases[key]


def bayesian_optimize(
    study,
    objective: int | str | Callable = 0,
    *,
    acquisition: str = "ei",
    use_gradients: bool = False,
    constraints: Sequence[OutcomeConstraint] = (),
    fixed: Mapping[str, Any] | None = None,
    maximize: bool = False,
    fidelity: Mapping | None = None,
    max_evaluations: int = 50,
    n_initial: int | None = None,
    initial: str = "lhs",
    initial_points: Sequence[Mapping[str, Any]] = (),
    seed=0,
    xi: float = 0.0,
    kappa: float = 2.0,
    patience: int | None = None,
    tol: float | None = None,
    max_failures: int = 20,
    gp_options: Mapping | None = None,
    raw_samples: int | None = None,
    num_restarts: int = 8,
    resume: bool = True,
    callback: Callable[[dict], Any] | None = None,
    space: DesignSpace | Sequence | None = None,
    path: str | Path | None = None,
) -> BOResult:
    """Bayesian optimisation of ``objective`` over a study's design space (module docstring).

    ``study``: a :class:`~hpfem.opt.Study`, or an evaluator / function with ``space`` and the
    store ``path``. ``objective``: an observable (index or name) or a function of the values
    returning ``F`` or ``(F, dF/dy)``; ``maximize`` flips the sign. ``acquisition``: ``"ei"``
    (log expected improvement, ``xi`` its margin) or ``"lcb"`` (``kappa``). ``use_gradients``
    fits the gradient-enhanced GP from the evaluator's Jacobian. ``constraints``:
    :class:`OutcomeConstraint` s on the observables (known constraints come from the design
    space). ``fixed``, ``fidelity``: as :func:`~hpfem.opt.minimize`. ``max_evaluations``: the
    budget of objective calls including the ``n_initial`` points (default ``2 (d + 1)``) of the
    ``initial`` design (``"lhs"``, ``"sobol"``, ``"random"``) and the ``initial_points`` given
    (dicts with every parameter, SI). ``seed`` (int, ``None`` for a random one) seeds the design,
    the hyperparameter restarts and the acquisition candidates. ``patience``, ``tol``,
    ``max_failures``, ``callback(info)`` stop the run. ``gp_options`` go to
    :class:`~hpfem.opt.GaussianProcess` (``kernel``, ``noise``, ``restarts``, bounds);
    ``raw_samples`` (default ``min(256 d, 2048)``) Sobol' candidates and ``num_restarts``
    local optimisations of the acquisition per iteration. ``resume=False`` ignores a running
    or cancelled checkpoint. ``hpfem.Cancelled`` propagates after a ``"cancelled"``
    checkpoint."""
    study = as_study(study, space, path)
    problem = Problem(study, fixed, fidelity)
    goal = _Objective(objective, study.observables, maximize)
    kind = _acquisition_name(acquisition)
    constraints = list(constraints)
    for c in constraints:
        if not isinstance(c, OutcomeConstraint):
            raise TypeError(f"constraints: {c!r} is not an OutcomeConstraint")
    if kind == "lcb" and constraints:
        raise ValueError("bayesian_optimize: outcome constraints need acquisition='ei'")
    n_init = 2 * (problem.n + 1) if n_initial is None else int(n_initial)
    if n_init < 0 or int(max_evaluations) < 1:
        raise ValueError("n_initial must be >= 0 and max_evaluations >= 1")
    given = [problem.space.check(p) for p in initial_points]
    for p in given:
        if not _matches(problem, Evaluation(p, [], fidelity=problem.fidelity)):
            raise ValueError(f"initial point {p} does not have the fixed values {problem.fixed}")
    if n_init + len(given) > int(max_evaluations):
        raise ValueError(
            f"the initial design ({n_init + len(given)} points) exceeds "
            f"max_evaluations={max_evaluations}"
        )
    run = _BayesRun(
        problem,
        goal,
        constraints,
        kind,
        use_gradients=use_gradients,
        xi=xi,
        kappa=kappa,
        max_evaluations=int(max_evaluations),
        patience=patience,
        tol=tol,
        max_failures=max_failures,
        callback=callback,
        gp_options=gp_options or {},
        raw_samples=raw_samples,
        num_restarts=num_restarts,
    )
    resumed = resume and study.state is not None and run.restore(study.state)
    if not resumed:
        run.seed = _seed(seed)
        rng = np.random.default_rng([run.seed, 0])
        design = _initial_design(problem, n_init, initial, rng) if n_init else []
        run.initial = given + design
        if not run.initial and not any(run._matches(e) and e.ok for e in study.evaluations):
            raise ValueError("bayesian_optimize: no initial design and no stored evaluation")
        run.checkpoint("running")
    status = "done"
    try:
        run.loop()
    except hpfem.Cancelled:
        run.checkpoint("cancelled", run.data())
        raise
    except _Stop as stop:
        status, run.message = stop.status, str(stop)
    run.checkpoint(status, run.data())
    return run.result(status)


# --- Pareto front (no BoTorch) -----------------------------------------------------------------


def non_dominated(values, maximize: bool | Sequence[bool] = False) -> np.ndarray:
    """Boolean mask of the non-dominated rows of ``values`` (k, p) (minimisation per column,
    or maximisation where ``maximize``); rows with NaN are dominated."""
    y = np.array(values, dtype=float)
    if y.ndim != 2:
        raise ValueError(f"non_dominated: values of shape {y.shape}, expected (k, p)")
    sign = np.where(np.broadcast_to(np.asarray(maximize, dtype=bool), (y.shape[1],)), -1.0, 1.0)
    y = y * sign
    finite = np.all(np.isfinite(y), axis=1)
    mask = finite.copy()
    for i in np.flatnonzero(finite):
        others = y[finite]
        dominated = np.any(np.all(others <= y[i], axis=1) & np.any(others < y[i], axis=1))
        mask[i] = not dominated
    return mask


@dataclass
class ParetoFront:
    """Non-dominated evaluations of a study: ``params`` (SI), ``values`` (k, p) of the
    objectives as given (not negated), ``evaluations``, ``index`` into
    :attr:`Study.evaluations`, the objective ``names``; sorted by the first objective."""

    params: list[dict]
    values: np.ndarray
    evaluations: list[Evaluation]
    index: np.ndarray
    names: list[str]

    def __len__(self) -> int:
        return len(self.params)


def pareto_front(
    study: Study,
    objectives: Sequence[int | str | Callable],
    *,
    maximize: bool | Sequence[bool] = False,
    fidelity: Mapping | None = None,
    constraints: Sequence[OutcomeConstraint] = (),
) -> ParetoFront:
    """The Pareto front of the successful evaluations of ``study`` (at ``fidelity`` if given)
    for several ``objectives`` (observables or functions of the values), minimised or
    maximised per ``maximize``; evaluations violating an :class:`OutcomeConstraint` are left
    out. Exact for any study size (``O(k²)``)."""
    objectives = list(objectives)
    if not objectives:
        raise ValueError("pareto_front: no objectives")
    goals = [_Objective(o, study.observables, False) for o in objectives]
    cgoals = [(c, _Objective(c.observable, study.observables, False)) for c in constraints]
    rows, values = [], []
    for i, e in enumerate(study.evaluations):
        if not e.ok or (fidelity is not None and e.fidelity != dict(fidelity)):
            continue
        if not all(c.satisfied(cg(e.values, False)[0]) for c, cg in cgoals):
            continue
        rows.append(i)
        values.append([g(e.values, False)[0] for g in goals])
    names = [
        o
        if isinstance(o, str)
        else (study.observables[o] if isinstance(o, int) else getattr(o, "__name__", f"f{j}"))
        for j, o in enumerate(objectives)
    ]
    if not rows:
        return ParetoFront([], np.zeros((0, len(goals))), [], np.zeros(0, dtype=int), names)
    y = np.array(values, dtype=float)
    mask = non_dominated(y, maximize)
    keep = np.flatnonzero(mask)
    keep = keep[np.argsort(y[keep, 0], kind="stable")]
    evals = study.evaluations
    index = np.array(rows, dtype=int)[keep]
    return ParetoFront(
        [dict(evals[i].params) for i in index], y[keep], [evals[i] for i in index], index, names
    )


# --- optional BoTorch path (extra "opt-bo") ----------------------------------------------------


def _require_botorch():
    try:
        import botorch  # noqa: F401
        import torch
    except ImportError as error:
        raise ImportError(
            "hpfem.opt: multi-objective and multi-fidelity Bayesian optimisation need BoTorch, "
            "the optional extra 'opt-bo' (pip install \"hpfem[opt-bo]\")"
        ) from error
    return torch


def _feasible_fallback(
    problem: Problem,
    u: np.ndarray,
    rng: np.random.Generator,
    score: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """``u`` if feasible, else the best feasible of random candidates by ``score``."""
    if problem.feasible(problem.params(u)):
        return u
    raw = rng.random((1024, problem.n))
    raw = raw[[problem.feasible(problem.params(r)) for r in raw]]
    if not len(raw):
        raise EvaluationFailed("no feasible candidate found for the BoTorch proposal")
    return raw[int(np.argmax(score(raw)))]


@dataclass
class ParetoResult:
    """Result of :func:`pareto_optimize`: the Pareto ``front`` of the study, the ``study``,
    ``evaluations`` (objective calls of the run), ``new_evaluations``, ``iterations``,
    ``message`` and the final BoTorch ``model``."""

    front: ParetoFront
    study: Study
    evaluations: int
    new_evaluations: int
    iterations: int
    message: str
    model: Any = None


def pareto_optimize(
    study,
    objectives: Sequence[int | str | Callable],
    *,
    maximize: bool | Sequence[bool] = False,
    ref_point=None,
    fixed: Mapping[str, Any] | None = None,
    fidelity: Mapping | None = None,
    max_evaluations: int = 40,
    n_initial: int | None = None,
    initial: str = "sobol",
    seed=0,
    num_restarts: int = 10,
    raw_samples: int = 512,
    callback: Callable[[dict], Any] | None = None,
    space: DesignSpace | Sequence | None = None,
    path: str | Path | None = None,
) -> ParetoResult:
    """Multi-objective Bayesian optimisation with BoTorch (optional extra ``opt-bo``): one
    ``SingleTaskGP`` per objective (``ModelListGP``, standardised outcomes) and the noisy
    log expected hypervolume improvement (``qLogNoisyExpectedHypervolumeImprovement``, or
    ``qNoisyExpectedHypervolumeImprovement`` on BoTorch versions without it), one point per
    iteration through the study. ``ref_point``: the reference point in objective units (as
    given; default: the worst observed value of each objective minus 10 % of its range).
    Known constraints by rejection (an infeasible optimum of the acquisition is replaced by the
    best feasible random candidate). Every stored evaluation of the problem is training data,
    so a rerun on the stored study continues from it (the trajectory is not replayed)."""
    torch = _require_botorch()
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import ModelListGP, SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import SumMarginalLogLikelihood

    try:
        from botorch.acquisition.multi_objective.logei import (
            qLogNoisyExpectedHypervolumeImprovement as Acquisition,
        )
    except ImportError:  # BoTorch < 0.9
        from botorch.acquisition.multi_objective.monte_carlo import (
            qNoisyExpectedHypervolumeImprovement as Acquisition,
        )

    study = as_study(study, space, path)
    problem = Problem(study, fixed, fidelity)
    p = len(objectives)
    maxes = np.broadcast_to(np.asarray(maximize, dtype=bool), (p,))
    goals = [
        _Objective(o, study.observables, bool(m)) for o, m in zip(objectives, maxes, strict=True)
    ]
    seed = _seed(seed)
    rng = np.random.default_rng([seed, 0])
    torch.manual_seed(seed)
    size0 = len(study)
    n_init = 2 * (problem.n + 1) if n_initial is None else int(n_initial)
    nfev, nit, message, model = 0, 0, "", None
    method = "botorch-qnehvi"
    try:
        design = _initial_design(problem, min(n_init, max_evaluations), initial, rng)
        study.run(design, fidelity=problem.fidelity, source=f"{method}-initial")
        nfev = len(design)
        while nfev < max_evaluations:
            x, y = [], []
            for e in study.evaluations:
                if not _matches(problem, e) or not e.ok:
                    continue
                f = [g(e.values, False)[0] for g in goals]
                if all(math.isfinite(v) for v in f):
                    x.append(problem.encode(e.params))
                    y.append([-v for v in f])  # BoTorch maximises
            if len(x) < 2:
                message = "fewer than two successful evaluations"
                break
            tx = torch.tensor(np.array(x), dtype=torch.double)
            ty = torch.tensor(np.array(y), dtype=torch.double)
            models = [
                SingleTaskGP(tx, ty[:, i : i + 1], outcome_transform=Standardize(m=1))
                for i in range(p)
            ]
            model = ModelListGP(*models)
            fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
            if ref_point is None:
                lo, hi = ty.min(dim=0).values, ty.max(dim=0).values
                ref = lo - 0.1 * (hi - lo).clamp_min(1e-12)
            else:
                sign = torch.tensor(np.where(maxes, 1.0, -1.0), dtype=torch.double)
                ref = sign * torch.tensor(np.asarray(ref_point, dtype=float), dtype=torch.double)
            acqf = Acquisition(
                model=model, ref_point=ref.tolist(), X_baseline=tx, prune_baseline=True
            )
            bounds = torch.stack(
                [
                    torch.zeros(problem.n, dtype=torch.double),
                    torch.ones(problem.n, dtype=torch.double),
                ]
            )
            candidate, _ = optimize_acqf(
                acqf,
                bounds=bounds,
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options={"batch_limit": 5, "maxiter": 200},
            )
            u = candidate.detach().cpu().numpy()[0]

            def score(c, acqf=acqf):
                with torch.no_grad():
                    t = torch.tensor(c, dtype=torch.double).unsqueeze(1)
                    return acqf(t).cpu().numpy()

            u = _feasible_fallback(problem, np.clip(u, 0.0, 1.0), rng, score)
            params = problem.params(u)
            study.propose([params], fidelity=problem.fidelity, source=method)
            e = study.evaluate(params, fidelity=problem.fidelity)
            nfev += 1
            nit += 1
            study.checkpoint(
                method,
                nit,
                {
                    "status": "running",
                    "nfev": nfev,
                    "seed": seed,
                    "free": problem.names,
                    "fixed": problem.fixed,
                    "fidelity": problem.fidelity,
                },
            )
            if callback is not None and callback(
                {"iteration": nit, "params": params, "values": e.values, "evaluations": nfev}
            ):
                message = "stopped by the callback"
                break
        else:
            message = f"maximum number of evaluations ({max_evaluations}) reached"
    except hpfem.Cancelled:
        study.checkpoint(
            method,
            nit,
            {
                "status": "cancelled",
                "nfev": nfev,
                "seed": seed,
                "free": problem.names,
                "fixed": problem.fixed,
                "fidelity": problem.fidelity,
            },
        )
        raise
    study.checkpoint(
        method,
        nit,
        {
            "status": "done",
            "nfev": nfev,
            "seed": seed,
            "free": problem.names,
            "fixed": problem.fixed,
            "fidelity": problem.fidelity,
        },
    )
    front = pareto_front(study, objectives, maximize=maximize, fidelity=problem.fidelity)
    return ParetoResult(front, study, nfev, len(study) - size0, nit, message, model)


def multi_fidelity_optimize(
    study,
    objective: int | str | Callable = 0,
    *,
    fidelities: Sequence[Mapping],
    costs: Sequence[float] | None = None,
    maximize: bool = False,
    fixed: Mapping[str, Any] | None = None,
    max_evaluations: int = 40,
    max_cost: float | None = None,
    n_initial: int | None = None,
    initial: str = "sobol",
    seed=0,
    num_fantasies: int = 64,
    num_restarts: int = 10,
    raw_samples: int = 512,
    space: DesignSpace | Sequence | None = None,
    path: str | Path | None = None,
) -> BOResult:
    """Multi-fidelity Bayesian optimisation with BoTorch (optional extra ``opt-bo``).

    ``fidelities``: the discrete fidelity levels (dicts passed to the evaluator, e.g.
    ``[{"order": 2}, {"order": 3}, {"order": 4}]``), cheapest first; the last one is the target.
    Level ``l`` gets the fidelity coordinate ``s = l / (L − 1)``. The model is a
    ``SingleTaskMultiFidelityGP`` on ``(u, s)``; the acquisition the cost-aware
    multi-fidelity knowledge gradient (``qMultiFidelityKnowledgeGradient`` with
    ``InverseCostWeightedUtility`` and an ``AffineFidelityCostModel`` fitted to ``costs`` per
    level, or by default to the measured mean ``Evaluation.cost`` per level), optimised over
    the levels with ``optimize_acqf_mixed`` (BoTorch's discrete-fidelity tutorial). The initial
    design cycles through the levels. The DWR estimate is not used (ADR-0012 §6; whether it
    beats this as a fidelity indicator is the S8 hypothesis). Stops after ``max_evaluations``
    or when the summed cost model reaches ``max_cost``; finally the maximiser of the posterior
    mean at the target fidelity is evaluated there, and the result is the best evaluation at
    the target fidelity."""
    torch = _require_botorch()
    from botorch.acquisition import PosteriorMean
    from botorch.acquisition.cost_aware import InverseCostWeightedUtility
    from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
    from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
    from botorch.acquisition.utils import project_to_target_fidelity
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskMultiFidelityGP
    from botorch.models.cost import AffineFidelityCostModel
    from botorch.models.transforms.outcome import Standardize
    from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed
    from gpytorch.mlls import ExactMarginalLogLikelihood

    levels = [dict(f) for f in fidelities]
    if len(levels) < 2:
        raise ValueError("multi_fidelity_optimize: give at least two fidelity levels")
    study = as_study(study, space, path)
    target = levels[-1]
    problem = Problem(study, fixed, target)
    goal = _Objective(objective, study.observables, maximize)
    d, n_levels = problem.n, len(levels)
    s_of = [i / (n_levels - 1) for i in range(n_levels)]
    seed = _seed(seed)
    rng = np.random.default_rng([seed, 0])
    torch.manual_seed(seed)
    size0, hits0 = len(study), study.cache_hits
    method = "botorch-mfkg"
    n_init = 2 * (d + 1) if n_initial is None else int(n_initial)
    nfev, nit, spent, message = 0, 0, 0.0, ""

    def data():
        x, y, cost = [], [], {i: [] for i in range(n_levels)}
        for e in study.evaluations:
            if not e.ok or e.fidelity not in levels or not _matches(problem, e, e.fidelity):
                continue
            f = goal(e.values, False)[0]
            if not math.isfinite(f):
                continue
            level = levels.index(e.fidelity)
            x.append(np.append(problem.encode(e.params), s_of[level]))
            y.append([-f])
            if math.isfinite(e.cost):
                cost[level].append(e.cost)
        return np.array(x), np.array(y), cost

    def cost_model(measured):
        if costs is not None:
            c = np.asarray(costs, dtype=float)
            known = np.arange(n_levels)
        else:
            known = np.array([i for i in range(n_levels) if measured[i]])
            c = np.array([np.mean(measured[i]) for i in known])
        if len(known) >= 2:
            a = np.column_stack([np.ones(len(known)), np.array(s_of)[known]])
            c0, w = np.linalg.lstsq(a, c, rcond=None)[0]
        else:
            c0, w = 1.0, 9.0
        c0 = max(float(c0), 1e-3 * max(float(np.max(c)) if len(c) else 1.0, 1e-12))
        return c0, max(float(w), 0.0)

    def evaluate(params, level, source):
        nonlocal nfev, spent
        study.propose([params], fidelity=levels[level], source=source)
        e = study.evaluate(params, fidelity=levels[level])
        nfev += 1
        c0, w = cost_model(data()[2])
        spent += c0 + w * s_of[level]
        return e

    def state(status):
        return {
            "status": status,
            "nfev": nfev,
            "seed": seed,
            "free": problem.names,
            "fixed": problem.fixed,
            "fidelities": levels,
            "spent": spent,
        }

    try:
        design = _initial_design(problem, min(n_init, max_evaluations), initial, rng)
        for i, params in enumerate(design):
            evaluate(params, i % n_levels, f"{method}-initial")
        bounds = torch.stack(
            [torch.zeros(d + 1, dtype=torch.double), torch.ones(d + 1, dtype=torch.double)]
        )
        while nfev < max_evaluations - 1 and (max_cost is None or spent < max_cost):
            x, y, measured = data()
            tx = torch.tensor(x, dtype=torch.double)
            ty = torch.tensor(y, dtype=torch.double)
            keyword = (
                "data_fidelities"
                if "data_fidelities" in inspect.signature(SingleTaskMultiFidelityGP).parameters
                else "data_fidelity"
            )
            fid = [d] if keyword == "data_fidelities" else d
            model = SingleTaskMultiFidelityGP(
                tx, ty, outcome_transform=Standardize(m=1), **{keyword: fid}
            )
            fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
            c0, w = cost_model(measured)
            utility = InverseCostWeightedUtility(
                cost_model=AffineFidelityCostModel(fidelity_weights={d: w}, fixed_cost=c0)
            )

            def project(t):
                return project_to_target_fidelity(X=t, target_fidelities={d: 1.0})

            current = FixedFeatureAcquisitionFunction(
                acq_function=PosteriorMean(model), d=d + 1, columns=[d], values=[1.0]
            )
            _, current_value = optimize_acqf(
                current,
                bounds=bounds[:, :-1],
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options={"batch_limit": 10, "maxiter": 200},
            )
            kg = qMultiFidelityKnowledgeGradient(
                model=model,
                num_fantasies=num_fantasies,
                current_value=current_value,
                cost_aware_utility=utility,
                project=project,
            )
            candidate, _ = optimize_acqf_mixed(
                kg,
                bounds=bounds,
                fixed_features_list=[{d: s} for s in s_of],
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options={"batch_limit": 5, "maxiter": 200},
            )
            c = candidate.detach().cpu().numpy()[0]
            level = int(np.argmin(np.abs(np.array(s_of) - c[d])))

            def score(raw, current=current):
                with torch.no_grad():
                    return current(torch.tensor(raw, dtype=torch.double).unsqueeze(1)).cpu().numpy()

            u = _feasible_fallback(problem, np.clip(c[:d], 0.0, 1.0), rng, score)
            evaluate(problem.params(u), level, method)
            nit += 1
            study.checkpoint(method, nit, state("running"))
        else:
            message = "budget reached"
        # the recommendation: maximiser of the posterior mean at the target fidelity
        x, y, _ = data()
        keyword = (
            "data_fidelities"
            if "data_fidelities" in inspect.signature(SingleTaskMultiFidelityGP).parameters
            else "data_fidelity"
        )
        model = SingleTaskMultiFidelityGP(
            torch.tensor(x, dtype=torch.double),
            torch.tensor(y, dtype=torch.double),
            outcome_transform=Standardize(m=1),
            **{keyword: [d] if keyword == "data_fidelities" else d},
        )
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        current = FixedFeatureAcquisitionFunction(
            acq_function=PosteriorMean(model), d=d + 1, columns=[d], values=[1.0]
        )
        rec, _ = optimize_acqf(
            current,
            bounds=bounds[:, :-1],
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={"batch_limit": 10, "maxiter": 200},
        )

        def score_rec(raw, current=current):
            with torch.no_grad():
                return current(torch.tensor(raw, dtype=torch.double).unsqueeze(1)).cpu().numpy()

        u = _feasible_fallback(
            problem, np.clip(rec.detach().cpu().numpy()[0], 0.0, 1.0), rng, score_rec
        )
        evaluate(problem.params(u), n_levels - 1, f"{method}-recommendation")
    except hpfem.Cancelled:
        study.checkpoint(method, nit, state("cancelled"))
        raise
    study.checkpoint(method, nit, state("done"))
    best = study.best(lambda v: goal(v, False)[0], fidelity=target)
    if best is None:
        raise EvaluationFailed("multi_fidelity_optimize: no successful evaluation at the target")
    params = dict(best.params)
    return BOResult(
        params=params,
        value=goal.sign * goal(best.values, False)[0],
        evaluation=best,
        success=True,
        message=message or "done",
        method=method,
        names=list(problem.names),
        x=problem.x(params),
        iterations=nit,
        evaluations=nfev,
        new_evaluations=len(study) - size0,
        cache_hits=study.cache_hits - hits0,
        failures=len(study.failed),
        infeasible=0,
        restarts=0,
        study=study,
        fidelity=dict(target),
    )


__all__ = [
    "ACQUISITIONS",
    "BOResult",
    "OutcomeConstraint",
    "ParetoFront",
    "ParetoResult",
    "bayesian_optimize",
    "expected_improvement",
    "log_expected_improvement",
    "log_probability_of_feasibility",
    "lower_confidence_bound",
    "multi_fidelity_optimize",
    "non_dominated",
    "pareto_front",
    "pareto_optimize",
]

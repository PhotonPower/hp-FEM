"""Classical optimisers driving a study (M16 S3, ADR-0012 §2, §3, §5).

:func:`minimize` runs L-BFGS-B (with the gradient from the evaluator's Jacobian), Nelder–Mead
or differential evolution (seeded) from SciPy on a :class:`hpfem.opt.Study`: every point goes
through :meth:`Study.evaluate`, so it is cached, appended to the JSON-lines store, replayed
without evaluating again when the same optimisation runs on the stored study, and the run is
cancellable (``hpfem.Cancelled``, raised by the study with the store consistent).

**Coordinates.** The optimisers work on the *free* parameters (every continuous parameter of
the design space that is not in ``fixed``) in the unit-cube coordinates of the design space
(:meth:`DesignSpace.encode`, logarithmic for ``log=True``), with bounds ``[0, 1]``. The
gradient is transformed by the chain rule, ``dF/du_j = dF/dp_j · dp_j/du_j`` with
``dp/du = upper − lower`` (linear) or ``p · ln(upper/lower)`` (logarithmic). Integer and
categorical parameters must be fixed.

**Objective.** An observable (index or name) or a function of the observable vector ``y``.
A function may return ``F`` or ``(F, dF/dy)``; without ``dF/dy`` L-BFGS-B differentiates it
numerically with respect to ``y`` (central differences, cheap: the objective is a function of
the values, not of the solve), and the gradient is ``dF/dp = (dF/dy)ᵀ J``.

**Constraints.** Bounds are the box ``[0, 1]``. Linear and nonlinear constraints of the
design space go to SciPy's differential evolution as constraints; L-BFGS-B and Nelder–Mead do
not support them, there an infeasible point is not evaluated and gets the failure value
(an extreme barrier).

**Failures** (ADR-0012 §2: ``status="failed"``, NaN values; also non-finite objectives or
gradients and infeasible points) are not fatal. The optimiser sees a *failure value*
``F_pen = F_worst + (F_worst − F_best) + max(1, |F_worst|)`` over the successful evaluations
of the run (or ``failure_value``), larger than every value seen, so the step is rejected:
the L-BFGS-B line search backtracks (its gradient is the last good gradient plus the slope
that leads from the last good point to ``F_pen``), the Nelder–Mead simplex contracts away and
the differential-evolution trial loses. More than ``max_failures`` failed evaluations stop
the run (infeasible points cost nothing and are only counted); a failure at the start point
of L-BFGS-B or Nelder–Mead stops it at once.

**Remesh** (ADR-0012 §3). When an evaluation remeshes (a new ``"remesh"`` record, or a new
evaluation on a different ``mesh_id``), L-BFGS-B restarts with an empty memory from that
point — the new reference mesh was built there; going back to the last iterate would leave
the morphing range of the new reference and remesh again — and tracks the best point from
the restart on (values on different meshes differ by the discretisation error, which would
mislead the curvature pairs and the line search).
``max_restarts`` bounds the number of restarts. Nelder–Mead and differential evolution only
compare values; a jump of the size of the discretisation error does not invalidate their
state, so they keep their simplex or population and every point.

**Checkpoints and resume.** After every iteration (generation) a ``"state"`` record
(:meth:`Study.checkpoint`) stores the method, the iteration, the current and the best point
(SI and unit coordinates), the number of evaluations, restarts and failures, the status
(``"running"``, ``"done"``, ``"stopped"``, ``"cancelled"``) and, for differential evolution,
the population. Resuming a cancelled run: call :func:`minimize` again on the stored study —
with the same ``x0`` (and ``seed``) the deterministic optimisers replay every point from the
cache and continue where they stopped; with ``x0=None`` L-BFGS-B and Nelder–Mead restart from
the best stored point and differential evolution continues from the population of its last
checkpoint.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.optimize

import hpfem
from hpfem.opt.evaluator import Evaluation
from hpfem.opt.study import (
    Continuous,
    DesignSpace,
    EvaluationFailed,
    History,
    Study,
    StudyError,
)

METHODS = ("L-BFGS-B", "Nelder-Mead", "differential-evolution")
"""Methods of :func:`minimize`."""

_ALIASES = {
    "l-bfgs-b": "L-BFGS-B",
    "lbfgsb": "L-BFGS-B",
    "lbfgs": "L-BFGS-B",
    "nelder-mead": "Nelder-Mead",
    "neldermead": "Nelder-Mead",
    "nm": "Nelder-Mead",
    "differential-evolution": "differential-evolution",
    "differential_evolution": "differential-evolution",
    "de": "differential-evolution",
}


class _Stop(Exception):
    """Ends a run early (budget, failures); the message is the result's message."""


class _Restart(Exception):
    """A remesh: restart the local optimiser at ``u``."""

    def __init__(self, u: np.ndarray):
        super().__init__("remesh")
        self.u = u


# --- shared with hpfem.opt.lsq ----------------------------------------------------------------


def as_study(target, space: DesignSpace | Sequence | None = None,
             path: str | Path | None = None) -> Study:  # fmt: skip
    """``target`` if it is a :class:`Study` (then ``space`` and ``path`` must be ``None``),
    else a new study of the evaluator (or function, then ``space`` is required) ``target``
    with the store ``path`` (``None``: in memory)."""
    if isinstance(target, Study):
        if space is not None or path is not None:
            raise ValueError("space and path are only for an evaluator, not for a study")
        study = target
    else:
        study = Study(target, path, space)
    if study.evaluator is None:
        raise StudyError("the study was opened read-only (Study.load): give an evaluator")
    return study


class Problem:
    """The free parameters of a study in unit-cube coordinates (used by :func:`minimize` and
    :func:`hpfem.opt.fit`): ``fixed`` parameters keep their value, the others must be
    continuous."""

    def __init__(self, study: Study, fixed: Mapping[str, Any] | None,
                 fidelity: Mapping | None):  # fmt: skip
        self.study = study
        self.space = study.space
        fixed = dict(fixed or {})
        unknown = set(fixed) - set(self.space.names)
        if unknown:
            raise ValueError(f"fixed: unknown parameters {sorted(unknown)} ({self.space.names})")
        self.fixed = {n: self.space[n].check(v) for n, v in fixed.items()}
        self.free: list[Continuous] = []
        for p in self.space.parameters:
            if p.name in self.fixed:
                continue
            if not isinstance(p, Continuous):
                raise ValueError(
                    f"parameter {p.name} is {p.kind}: the optimisers work on continuous "
                    f"parameters, fix it with fixed={{{p.name!r}: value}}"
                )
            self.free.append(p)
        if not self.free:
            raise ValueError("no free parameter: every parameter is fixed")
        self.names = [p.name for p in self.free]
        self.index = [self.space.names.index(n) for n in self.names]
        self.fidelity = dict(fidelity or {})
        self.n = len(self.free)

    def params(self, u) -> dict[str, Any]:
        """The full point (space order, SI) at unit coordinates ``u`` of the free parameters
        (clipped to [0, 1])."""
        values = dict(self.fixed)
        for p, ui in zip(self.free, np.asarray(u, dtype=float), strict=True):
            values[p.name] = p.decode(ui)
        return {n: values[n] for n in self.space.names}

    def encode(self, params: Mapping[str, Any]) -> np.ndarray:
        """Unit coordinates of the free parameters of ``params``."""
        return np.array([p.encode(p.check(params[p.name])) for p in self.free])

    def x(self, params: Mapping[str, Any]) -> np.ndarray:
        """SI values of the free parameters of ``params``."""
        return np.array([float(params[n]) for n in self.names])

    def dxdu(self, params: Mapping[str, Any]) -> np.ndarray:
        """``dp/du`` of the free parameters at ``params``."""
        out = []
        for p in self.free:
            if p.log:
                out.append(float(params[p.name]) * math.log(p.upper / p.lower))
            else:
                out.append(p.upper - p.lower)
        return np.array(out)

    def feasible(self, params: Mapping[str, Any]) -> bool:
        return self.space.violation(params) == 0.0

    def evaluate(self, u, jacobian: bool) -> tuple[dict, Evaluation | None]:
        """The point at ``u`` and its evaluation through the study, ``None`` when the point
        violates a constraint (not evaluated)."""
        params = self.params(u)
        if not self.feasible(params):
            return params, None
        return params, self.study.evaluate(params, jacobian=jacobian, fidelity=self.fidelity)

    def free_jacobian(self, evaluation: Evaluation) -> np.ndarray:
        """Columns of the free parameters of the evaluation's Jacobian (SI)."""
        if evaluation.jacobian is None:
            raise ValueError(
                f"the evaluator returned no Jacobian at {evaluation.params} (jacobian=True)"
            )
        return evaluation.jacobian[:, self.index]

    def matches(self, evaluation: Evaluation) -> bool:
        """A successful evaluation of this problem (fixed values and fidelity)."""
        if not evaluation.ok or evaluation.fidelity != self.fidelity:
            return False
        return all(
            self.space[n].key(evaluation.params[n]) == self.space[n].key(v)
            for n, v in self.fixed.items()
        )

    def start(self, x0, cost: Callable[[np.ndarray], float] | None) -> np.ndarray:
        """Unit coordinates of the start point: ``x0`` (dict with the free names, SI, or a
        sequence of their SI values); ``None``: the stored evaluation of this problem with
        the smallest ``cost(values)``, else the centre of the box."""
        if x0 is None:
            best, best_cost = None, math.inf
            if cost is not None:
                for e in self.study.evaluations:
                    if self.matches(e):
                        c = float(cost(e.values))
                        if c < best_cost:
                            best, best_cost = e, c
            if best is not None:
                return self.encode(best.params)
            u = np.full(self.n, 0.5)
        elif isinstance(x0, Mapping):
            missing = [n for n in self.names if n not in x0]
            if missing:
                raise ValueError(f"x0: missing free parameters {missing}")
            u = self.encode(x0)
        else:
            x = np.asarray(x0, dtype=float).ravel()
            if len(x) != self.n:
                raise ValueError(f"x0: {len(x)} values for the free parameters {self.names}")
            u = self.encode(dict(zip(self.names, x, strict=True)))
        if not self.feasible(self.params(u)):
            raise ValueError(f"the start point {self.params(u)} violates a constraint")
        return u

    def remeshed(self, before: tuple[int, int], evaluation: Evaluation | None,
                 mesh_id: int | None) -> bool:  # fmt: skip
        """Whether the last call of :meth:`evaluate` (study sizes ``before``) remeshed: a new
        ``"remesh"`` record, or a new evaluation on another mesh than ``mesh_id``."""
        if len(self.study.remeshes) > before[1]:
            return True
        new = len(self.study) > before[0]
        return bool(
            new and evaluation is not None and mesh_id is not None
            and evaluation.ok and evaluation.mesh_id != mesh_id
        )  # fmt: skip

    def sizes(self) -> tuple[int, int]:
        return len(self.study), len(self.study.remeshes)


# --- objective -------------------------------------------------------------------------------


class _Objective:
    """``F(y)`` to minimise and ``dF/dy``."""

    def __init__(self, objective, observables: Sequence[str], maximize: bool):
        self.sign = -1.0 if maximize else 1.0
        self.m = len(observables)
        if isinstance(objective, str):
            if objective not in observables:
                raise ValueError(f"objective: unknown observable {objective!r} ({observables})")
            objective = list(observables).index(objective)
        if isinstance(objective, (int, np.integer)) and not isinstance(objective, bool):
            column = int(objective)
            if not -self.m <= column < self.m:
                raise ValueError(f"objective: observable index {column} for {self.m} observables")
            self.column, self.function = column % self.m, None
        elif callable(objective):
            self.column, self.function = None, objective
        else:
            raise TypeError("objective: an observable index or name, or a function of the values")

    def __call__(self, values: np.ndarray, gradient: bool) -> tuple[float, np.ndarray | None]:
        """``(s F, s dF/dy)`` with ``s = -1`` when maximising."""
        if self.function is None:
            g = None
            if gradient:
                g = np.zeros(len(values))
                g[self.column] = self.sign
            return self.sign * float(values[self.column]), g
        out = self.function(np.array(values))
        if isinstance(out, tuple):
            f, g = float(out[0]), np.asarray(out[1], dtype=float).ravel()
            if g.shape != np.shape(values):
                raise ValueError(f"objective: dF/dy of shape {g.shape} for {len(values)} values")
            return self.sign * f, (self.sign * g if gradient else None)
        f = float(out)
        if not gradient:
            return self.sign * f, None
        g = np.zeros(len(values))
        for i in range(len(values)):
            h = 1e-6 * max(1.0, abs(values[i]))
            plus, minus = np.array(values), np.array(values)
            plus[i] += h
            minus[i] -= h
            g[i] = (float(self.function(plus)) - float(self.function(minus))) / (2 * h)
        return self.sign * f, self.sign * g


# --- result ----------------------------------------------------------------------------------


@dataclass
class OptimizeResult:
    """Result of :func:`minimize`.

    ``params`` is the best point (SI, every parameter including the fixed ones), ``value``
    the objective there (as given, not negated for ``maximize``), ``evaluation`` its
    :class:`~hpfem.opt.Evaluation`; ``names`` / ``x`` the free parameters and their SI values.
    ``evaluations`` counts the objective calls of the run (cache hits included),
    ``new_evaluations`` the evaluator calls, ``cache_hits`` the calls answered from the
    study's cache, ``failures`` the failed evaluations, ``infeasible`` the points rejected
    for violating a constraint (not evaluated), ``restarts`` the restarts
    after a remesh, ``iterations`` the optimiser iterations (generations). ``raw`` is SciPy's
    last result (unit coordinates)."""

    params: dict[str, Any]
    value: float
    evaluation: Evaluation | None
    success: bool
    message: str
    method: str
    names: list[str]
    x: np.ndarray
    iterations: int
    evaluations: int
    new_evaluations: int
    cache_hits: int
    failures: int
    infeasible: int
    restarts: int
    study: Study
    fidelity: dict
    raw: Any = None

    def history(self, status: str | None = "ok") -> History:
        """The study's evaluations at this fidelity (:meth:`Study.history`)."""
        return self.study.history(status, fidelity=self.fidelity)

    def __repr__(self) -> str:
        return (
            f"OptimizeResult({self.method}: value={self.value:.10g} at {self.params}, "
            f"{self.evaluations} evaluations ({self.new_evaluations} new), "
            f"success={self.success}, {self.message!r})"
        )


# --- the run ---------------------------------------------------------------------------------


class _Run:
    def __init__(self, problem: Problem, objective: _Objective, method: str,
                 max_evaluations: int | None, failure_value: float | None, max_failures: int,
                 callback: Callable[[dict], Any] | None):  # fmt: skip
        self.problem = problem
        self.study = problem.study
        self.objective = objective
        self.method = method
        self.max_evaluations = max_evaluations
        self.failure_value = None if failure_value is None else float(failure_value)
        self.max_failures = int(max_failures)
        self.callback = callback
        self.nfev = 0
        self.nit = 0
        self.failures = 0
        self.infeasible = 0
        self.restarts = 0
        self.restart_on_remesh = method == "L-BFGS-B"
        self.segment_nfev = 0
        self.mesh_id: int | None = None
        self.size0 = len(self.study)
        self.hits0 = self.study.cache_hits
        self.best: tuple[np.ndarray, float, Evaluation] | None = None  # u, F (minimised), e
        self.worst = -math.inf
        self.last: tuple[np.ndarray, float, np.ndarray | None] | None = None  # u, F, grad
        self.current: tuple[np.ndarray, float] | None = None
        self.success = False
        self.message = ""
        self.raw = None
        self.population = None

    # objective calls

    def penalty(self, u: np.ndarray, gradient: bool) -> tuple[float, np.ndarray | None]:
        if self.failure_value is not None:
            f = self.objective.sign * self.failure_value
        elif self.best is None:
            f = 1e30
        else:
            best = self.best[1]
            f = self.worst + (self.worst - best) + max(1.0, abs(self.worst))
        g = None
        if gradient:
            g = np.zeros(len(u))
            if self.last is not None and self.last[2] is not None:
                d = u - self.last[0]
                dd = float(d @ d)
                g = self.last[2].copy()
                if dd > 0:
                    g += 2.0 * (f - self.last[1]) * d / dd
        return f, g

    def fail(self, u: np.ndarray, gradient: bool, why: str,
             infeasible: bool = False) -> tuple[float, np.ndarray | None]:  # fmt: skip
        if infeasible:
            self.infeasible += 1
        else:
            self.failures += 1
            if self.failures > self.max_failures:
                raise _Stop(f"stopped after {self.failures} failed evaluations ({why})")
        if self.best is None and self.method != "differential-evolution":
            raise _Stop(f"the evaluation at the start point failed ({why})")
        return self.penalty(u, gradient)

    def fun(self, u, gradient: bool) -> tuple[float, np.ndarray | None]:
        if self.max_evaluations is not None and self.nfev >= self.max_evaluations:
            raise _Stop(f"maximum number of evaluations ({self.max_evaluations}) reached")
        self.nfev += 1
        self.segment_nfev += 1
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        before = self.problem.sizes()
        params, e = self.problem.evaluate(u, gradient)
        if e is None:
            return self.fail(u, gradient, "infeasible point", infeasible=True)
        if not e.ok:
            return self.fail(u, gradient, e.meta.get("error", e.status))
        f, dfdy = self.objective(e.values, gradient)
        if not math.isfinite(f):
            return self.fail(u, gradient, "non-finite objective")
        g = None
        if gradient:
            jac = self.problem.free_jacobian(e)
            g = (dfdy @ jac) * self.problem.dxdu(params)
            if not np.all(np.isfinite(g)):
                return self.fail(u, gradient, "non-finite gradient")
        remeshed = self.problem.remeshed(before, e, self.mesh_id)
        if remeshed or self.mesh_id is None:
            self.mesh_id = e.mesh_id
        restart = remeshed and self.restart_on_remesh and self.segment_nfev > 1
        if restart:
            self.best, self.worst = None, -math.inf
        self.worst = max(self.worst, f)
        if self.best is None or f < self.best[1]:
            self.best = (u, f, e)
        self.last = (u, f, g)
        if restart:
            raise _Restart(u)
        return f, g

    # iterations and checkpoints

    def state(self, status: str) -> dict:
        state: dict[str, Any] = {"status": status, "free": self.problem.names,
                                 "fixed": self.problem.fixed,
                                 "fidelity": self.problem.fidelity, "nfev": self.nfev,
                                 "restarts": self.restarts, "failures": self.failures,
                                 "infeasible": self.infeasible}  # fmt: skip
        if self.current is not None:
            u, f = self.current
            state.update(x=u, params=self.problem.params(u), value=self.objective.sign * f)
        if self.best is not None:
            u, f, _ = self.best
            state["best"] = {"x": u, "params": self.problem.params(u),
                             "value": self.objective.sign * f}  # fmt: skip
        if self.population is not None:
            state["population"], state["population_energies"] = self.population
        return state

    def checkpoint(self, status: str) -> None:
        self.study.checkpoint(self.method, self.nit, self.state(status))

    def iteration(self, intermediate_result) -> None:
        self.nit += 1
        self.current = (np.array(intermediate_result.x, dtype=float),
                        float(intermediate_result.fun))  # fmt: skip
        if "population" in intermediate_result:
            self.population = (np.array(intermediate_result.population),
                               np.array(intermediate_result.population_energies))  # fmt: skip
        self.checkpoint("running")
        if self.callback is not None:
            u, f = self.current
            info = {"iteration": self.nit, "params": self.problem.params(u),
                    "value": self.objective.sign * f, "evaluations": self.nfev}  # fmt: skip
            if self.callback(info):
                raise StopIteration

    # drivers

    def local(self, u0: np.ndarray, max_restarts: int, solve: Callable) -> None:
        u = u0
        while True:
            self.segment_nfev = 0
            try:
                self.raw = solve(u)
                self.success = bool(self.raw.success)
                self.message = str(self.raw.message)
                return
            except _Restart as restart:
                self.restarts += 1
                self.checkpoint("restart")
                if self.restarts > max_restarts:
                    self.message = f"stopped after {max_restarts} restarts on remeshes"
                    return
                u = restart.u
                self.last = None

    def lbfgsb(self, u0: np.ndarray, options: dict, max_restarts: int) -> None:
        opts = {"ftol": 1e-13, "gtol": 1e-9, "maxiter": 1000}
        opts.update(options)

        def solve(u):
            return scipy.optimize.minimize(
                lambda v: self.fun(v, True), u, jac=True, method="L-BFGS-B",
                bounds=[(0.0, 1.0)] * self.problem.n, options=opts, callback=self.iteration,
            )  # fmt: skip

        self.local(u0, max_restarts, solve)

    def nelder_mead(self, u0: np.ndarray, options: dict, max_restarts: int) -> None:
        opts = {"xatol": 1e-8, "fatol": 1e-12, "maxiter": 1000 * self.problem.n,
                "adaptive": self.problem.n > 2}  # fmt: skip
        size = float(options.pop("simplex_size", 0.1))
        opts.update(options)

        def solve(u):
            if "initial_simplex" not in options:
                simplex = [u]
                for i in range(self.problem.n):
                    v = u.copy()
                    v[i] += size if u[i] + size <= 1.0 else -size
                    simplex.append(v)
                opts["initial_simplex"] = np.array(simplex)
            return scipy.optimize.minimize(
                lambda v: self.fun(v, False)[0], u, method="Nelder-Mead",
                bounds=[(0.0, 1.0)] * self.problem.n, options=opts, callback=self.iteration,
            )  # fmt: skip

        self.local(u0, max_restarts, solve)

    def constraints(self):
        space = self.problem.space
        if not space.constraints:
            return ()
        centre = self.problem.params(np.full(self.problem.n, 0.5))
        lower, upper = [], []
        for c in space.constraints:
            k = np.size(c.value(centre))
            lower.append(np.broadcast_to(np.asarray(c.lower, dtype=float), (k,)))
            upper.append(np.broadcast_to(np.asarray(c.upper, dtype=float), (k,)))

        def values(u):
            params = self.problem.params(u)
            return np.concatenate([np.atleast_1d(c.value(params)) for c in space.constraints])

        return (scipy.optimize.NonlinearConstraint(values, np.concatenate(lower),
                                                   np.concatenate(upper)),)  # fmt: skip

    def differential_evolution(self, x0, options: dict, seed) -> None:
        kwargs: dict[str, Any] = {"maxiter": 1000, "popsize": 15, "tol": 0.01, "polish": False,
                                  "init": "latinhypercube", "updating": "immediate",
                                  "workers": 1}  # fmt: skip
        state = self.study.state
        if (
            x0 is None
            and state is not None
            and state.get("method") == self.method
            and isinstance(state.get("state"), Mapping)
            and state["state"].get("status") in ("running", "cancelled", "stopped")
            and state["state"].get("free") == self.problem.names
            and state["state"].get("population") is not None
            and "init" not in options
        ):
            kwargs["init"] = np.asarray(state["state"]["population"], dtype=float)
            self.nit = int(state.get("iteration", 0))
        kwargs.update(options)
        if x0 is not None:
            kwargs["x0"] = x0
        keyword = "rng" if "rng" in inspect.signature(
            scipy.optimize.differential_evolution).parameters else "seed"  # fmt: skip
        constraints = self.constraints()
        if constraints:
            kwargs["constraints"] = constraints
        self.raw = scipy.optimize.differential_evolution(
            lambda v: self.fun(v, False)[0], [(0.0, 1.0)] * self.problem.n,
            callback=self.iteration, **{keyword: seed}, **kwargs,
        )  # fmt: skip
        self.success = bool(self.raw.success)
        self.message = str(self.raw.message)

    def result(self) -> OptimizeResult:
        if self.best is None:
            raise _Stop("no successful evaluation")
        u, f, e = self.best
        params = self.problem.params(u)
        return OptimizeResult(
            params=params,
            value=self.objective.sign * f,
            evaluation=e,
            success=self.success,
            message=self.message,
            method=self.method,
            names=list(self.problem.names),
            x=self.problem.x(params),
            iterations=self.nit,
            evaluations=self.nfev,
            new_evaluations=len(self.study) - self.size0,
            cache_hits=self.study.cache_hits - self.hits0,
            failures=self.failures,
            infeasible=self.infeasible,
            restarts=self.restarts,
            study=self.study,
            fidelity=dict(self.problem.fidelity),
            raw=self.raw,
        )


def _method(method: str) -> str:
    key = str(method).lower().replace(" ", "-")
    if key not in _ALIASES:
        raise ValueError(f"minimize: method {method!r}, use one of {METHODS}")
    return _ALIASES[key]


def minimize(study, objective: int | str | Callable = 0, *, method: str = "L-BFGS-B",
             x0=None, fixed: Mapping[str, Any] | None = None, maximize: bool = False,
             fidelity: Mapping | None = None, max_evaluations: int | None = None,
             options: Mapping | None = None, seed=None, failure_value: float | None = None,
             max_failures: int = 20, max_restarts: int = 10,
             callback: Callable[[dict], Any] | None = None,
             space: DesignSpace | Sequence | None = None,
             path: str | Path | None = None) -> OptimizeResult:  # fmt: skip
    """Minimises (or maximises) an objective over a study's design space.

    ``study`` is a :class:`~hpfem.opt.Study` or an evaluator / function (a study is made with
    ``space`` and the store ``path``, in memory without). ``objective`` is an observable
    (index or name) or a function of the observable vector returning ``F`` or
    ``(F, dF/dy)``. ``method``: ``"L-BFGS-B"`` (gradient from the Jacobian, the evaluator is
    called with ``jacobian=True``), ``"Nelder-Mead"`` or ``"differential-evolution"``
    (seeded by ``seed``, an int or a ``numpy.random.Generator``; ``polish=False`` by default —
    polish with L-BFGS-B from ``result.params`` when gradients are available). ``x0``: dict
    with the free parameters (SI) or a sequence of their values; ``None`` resumes (module
    docstring). ``fixed`` holds parameters at a value (integer and categorical ones must be
    fixed). ``fidelity`` is passed to every evaluation. ``options`` go to SciPy (unit
    coordinates; defaults L-BFGS-B ``ftol=1e-13, gtol=1e-9``, Nelder–Mead ``xatol=1e-8,
    fatol=1e-12`` plus ``simplex_size=0.1``, differential evolution ``popsize=15, tol=0.01``).
    ``max_evaluations`` bounds the objective calls (cache hits included). ``callback(info)``
    gets ``iteration``, ``params``, ``value``, ``evaluations`` after every iteration; a true
    return value stops the run. Failures, constraints, remeshes and checkpoints: see the module
    docstring. ``hpfem.Cancelled`` propagates after a ``"cancelled"`` checkpoint."""
    study = as_study(study, space, path)
    problem = Problem(study, fixed, fidelity)
    goal = _Objective(objective, study.observables, maximize)
    method = _method(method)
    options = dict(options or {})
    if max_evaluations is not None and int(max_evaluations) < 1:
        raise ValueError("max_evaluations must be at least 1")
    run = _Run(problem, goal, method, None if max_evaluations is None else int(max_evaluations),
               failure_value, max_failures, callback)  # fmt: skip

    def cost(values):
        f = goal(values, False)[0]
        return f if math.isfinite(f) else math.inf

    status = "done"
    try:
        if method == "differential-evolution":
            u0 = None if x0 is None else problem.start(x0, None)
            run.differential_evolution(u0, options, seed)
        else:
            u0 = problem.start(x0, cost)
            if method == "L-BFGS-B":
                run.lbfgsb(u0, options, max_restarts)
            else:
                run.nelder_mead(u0, options, max_restarts)
    except hpfem.Cancelled:
        run.checkpoint("cancelled")
        raise
    except _Stop as stop:
        run.success, run.message, status = False, str(stop), "stopped"
    if run.best is None:
        run.checkpoint(status)
        raise _no_success(run)
    run.current = run.current or (run.best[0], run.best[1])
    run.checkpoint(status)
    return run.result()


def _no_success(run: _Run) -> EvaluationFailed:
    return EvaluationFailed(f"minimize: no successful evaluation ({run.message})")


__all__ = ["METHODS", "OptimizeResult", "minimize"]

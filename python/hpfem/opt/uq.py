"""Uncertainty propagation and global sensitivity analysis (M16 S7, ADR-0012 §6).

**Inputs.** Independent uncertain parameters are given as ``{name: Normal(mean, std)}`` or
``{name: Uniform(lower, upper)}``; the other parameters of the study keep the values of
``fixed`` (or the centre of their bounds). Values are SI.

**Linearised propagation** (:func:`linear_propagation`). One evaluation with the Jacobian at
the input means — one tangent or adjoint solve per parameter on the kept factorisation
(M16 S1) — gives the output covariance :math:`C_y = J Σ J^\\top` (the delta method) and the
share of every input in the variance of every output, :math:`J_{ij}^2 σ_j^2 / \\mathrm{Var}\\,y_i`
(the first-order Sobol' index of the linearised model). It is exact for a linear model and the
right first look for small tolerances; :func:`monte_carlo` on a surrogate checks it.

**Global surrogate** (:func:`build_global_surrogate`). A Latin-hypercube design in the input
box (``mean ± width · std`` for a normal input, the bounds for a uniform one, clipped to the
design space) is evaluated through the study — with the Jacobian when ``gradients`` — and a
gradient-enhanced Gaussian process per observable is fitted (:class:`hpfem.opt.MultiOutputGP`,
no noise). ``active`` further points are added one by one where the surrogate is least sure:
the candidate (of a fresh Latin hypercube) with the largest predicted standard deviation
relative to the spread of the training values, maximised over the observables (active
learning of the global surrogate). ``Surrogate.max_relative_std`` reports that quantity for
the final model.

**Monte Carlo** (:func:`monte_carlo`). Input samples are pushed through the surrogate mean
(or any vectorised function): output mean, standard deviation, quantiles and the samples;
normal samples outside the surrogate box are clipped to it and counted.

**Sobol' indices** (:func:`sobol_indices`). With two independent sample matrices A and B
(scrambled Sobol' points mapped through the inverse distribution functions) and the matrices
:math:`A_B^{(i)}` (A with column i from B), the first-order index follows the estimator of
Saltelli et al. (2010), :math:`S_i = \\overline{f(B)\\,(f(A_B^{(i)}) - f(A))} / V`, the total
index Jansen's, :math:`S_{T,i} = \\tfrac12\\,\\overline{(f(A) - f(A_B^{(i)}))^2} / V`, with
:math:`V` the variance of :math:`f` over A and B; ``N (n + 2)`` model calls in total, on the
surrogate mean they cost nothing. Bootstrap resampling of the rows gives confidence intervals.
Verified on the Ishigami function (analytic indices) directly and through the surrogate.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.stats

from hpfem.opt.gp import MultiOutputGP
from hpfem.opt.optimize import as_study

_QUANTILES = (0.025, 0.16, 0.5, 0.84, 0.975)


@dataclass(frozen=True)
class Normal:
    """A normally distributed input (SI)."""

    mean: float
    std: float

    def __post_init__(self):
        if not self.std > 0:
            raise ValueError(f"Normal: std must be positive, not {self.std}")

    @property
    def variance(self) -> float:
        return float(self.std) ** 2

    def box(self, width: float) -> tuple[float, float]:
        return self.mean - width * self.std, self.mean + width * self.std

    def ppf(self, u: np.ndarray) -> np.ndarray:
        return self.mean + self.std * scipy.stats.norm.ppf(u)


@dataclass(frozen=True)
class Uniform:
    """A uniformly distributed input on ``[lower, upper]`` (SI)."""

    lower: float
    upper: float

    def __post_init__(self):
        if not self.upper > self.lower:
            raise ValueError(f"Uniform: upper must exceed lower ({self.lower}, {self.upper})")

    @property
    def mean(self) -> float:
        return 0.5 * (self.lower + self.upper)

    @property
    def variance(self) -> float:
        return (self.upper - self.lower) ** 2 / 12.0

    def box(self, width: float) -> tuple[float, float]:
        return self.lower, self.upper

    def ppf(self, u: np.ndarray) -> np.ndarray:
        return self.lower + (self.upper - self.lower) * u


Distribution = Normal | Uniform


class _Setup:
    """The study, the uncertain inputs (names in the order of ``inputs``), their columns in
    the evaluator's Jacobian, the observable rows and the values of the other parameters."""

    def __init__(self, study, inputs: Mapping[str, Distribution], observables, fixed, fidelity):
        self.study = as_study(study)
        space = self.study.space
        names = list(space.names)
        unknown = set(inputs) - set(names)
        if unknown:
            raise ValueError(f"uncertain inputs {sorted(unknown)} are not parameters of the study")
        if not inputs:
            raise ValueError("no uncertain inputs")
        self.inputs = dict(inputs)
        self.names = list(inputs)
        self.columns = [names.index(n) for n in self.names]
        bounds = space.bounds()
        self.bounds = bounds[self.columns]
        all_observables = list(self.study.evaluator.observables)
        if observables is None:
            self.rows = list(range(len(all_observables)))
        else:
            self.rows = [o if isinstance(o, int) else all_observables.index(o) for o in observables]
        self.observables = [all_observables[r] for r in self.rows]
        base = {n: 0.5 * (lo + hi) for n, (lo, hi) in zip(names, bounds, strict=True)}
        base.update(dict(fixed or {}))
        self.params = base
        self.fidelity = dict(fidelity) if fidelity else None

    def at(self, x: np.ndarray) -> dict[str, Any]:
        out = dict(self.params)
        for name, value in zip(self.names, x, strict=True):
            out[name] = float(value)
        return out

    def evaluate(self, x: np.ndarray, jacobian: bool):
        evaluation = self.study.evaluate(self.at(x), jacobian=jacobian, fidelity=self.fidelity)
        if not evaluation.ok:
            return None
        values = np.asarray(evaluation.values, dtype=float)[self.rows]
        if not jacobian:
            return values, None
        jac = np.asarray(evaluation.jacobian, dtype=float)[np.ix_(self.rows, self.columns)]
        return values, jac

    def box(self, width: float) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = zip(*(self.inputs[n].box(width) for n in self.names), strict=True)
        lower = np.maximum(np.array(lo, dtype=float), self.bounds[:, 0])
        upper = np.minimum(np.array(hi, dtype=float), self.bounds[:, 1])
        if np.any(upper <= lower):
            raise ValueError("the input box does not intersect the bounds of the design space")
        return lower, upper


# --- linearised propagation ----------------------------------------------------------------------


@dataclass
class LinearPropagation:
    """Delta-method propagation at the input means: ``values`` (m,) the nominal outputs,
    ``jacobian`` (m, n), ``covariance`` (m, m) = ``J Σ Jᵀ``, ``std`` (m,), ``contributions``
    (m, n) the share of each input in each output variance (rows sum to 1 for independent
    inputs), for the ``observables`` and ``inputs`` (names)."""

    observables: list[str]
    inputs: list[str]
    values: np.ndarray
    jacobian: np.ndarray
    covariance: np.ndarray
    std: np.ndarray
    contributions: np.ndarray


def linear_propagation(study, inputs: Mapping[str, Distribution], *, observables=None,
                       fixed: Mapping[str, Any] | None = None,
                       fidelity: Mapping | None = None) -> LinearPropagation:  # fmt: skip
    """Linearised (delta-method) propagation of independent input uncertainties to the
    observables (module docstring): one evaluation with the Jacobian at the input means.
    ``study`` is a study or an evaluator; ``observables`` indices or names (default all);
    ``fixed`` the values of the other parameters."""
    setup = _Setup(study, inputs, observables, fixed, fidelity)
    mean = np.array([setup.inputs[n].mean for n in setup.names], dtype=float)
    out = setup.evaluate(mean, jacobian=True)
    if out is None:
        raise RuntimeError("linear_propagation: the evaluation at the input means failed")
    values, jac = out
    var_in = np.array([setup.inputs[n].variance for n in setup.names], dtype=float)
    parts = jac**2 * var_in
    covariance = (jac * var_in) @ jac.T
    var_out = np.diag(covariance)
    with np.errstate(invalid="ignore", divide="ignore"):
        contributions = np.where(var_out[:, None] > 0, parts / var_out[:, None], 0.0)
    return LinearPropagation(setup.observables, setup.names, values, jac, covariance,
                             np.sqrt(var_out), contributions)  # fmt: skip


# --- global surrogate ----------------------------------------------------------------------------


@dataclass
class GlobalSurrogate:
    """Gaussian-process model of the ``observables`` over the box ``[lower, upper]`` of the
    ``inputs`` (SI): :meth:`predict` gives mean and variance (q, m) at points (q, n),
    :meth:`__call__` the mean alone. ``x`` / ``y`` the training points and values,
    ``max_relative_std`` the largest predicted standard deviation relative to the spread of
    the training values over the last candidate pool (the surrogate's own uncertainty)."""

    inputs: list[str]
    observables: list[str]
    lower: np.ndarray
    upper: np.ndarray
    model: MultiOutputGP
    x: np.ndarray
    y: np.ndarray
    max_relative_std: float = math.nan

    def encode(self, x) -> np.ndarray:
        return (np.atleast_2d(np.asarray(x, dtype=float)) - self.lower) / (self.upper - self.lower)

    def predict(self, x) -> tuple[np.ndarray, np.ndarray]:
        return self.model.predict(self.encode(x))

    def __call__(self, x) -> np.ndarray:
        return self.predict(x)[0]

    @property
    def points(self) -> int:
        return len(self.x)


def _relative_std(surrogate: GlobalSurrogate, candidates: np.ndarray) -> np.ndarray:
    _, var = surrogate.predict(candidates)
    spread = np.std(surrogate.y, axis=0)
    spread = np.where(spread > 0, spread, 1.0)
    return np.max(np.sqrt(var) / spread, axis=1)


def build_global_surrogate(study, inputs: Mapping[str, Distribution], *, points: int | None = None,
                           active: int = 0, width: float = 4.0, gradients: bool = True,
                           candidates: int = 512, observables=None,
                           fixed: Mapping[str, Any] | None = None, fidelity: Mapping | None = None,
                           kernel: str = "matern52", seed=0) -> GlobalSurrogate:  # fmt: skip
    """Global surrogate of the observables over the input box (module docstring): ``points``
    Latin-hypercube evaluations (default ``4n + 4``; with the Jacobian if ``gradients``)
    through the study, then ``active`` points by active learning (``candidates`` per step).
    Failed evaluations are skipped."""
    setup = _Setup(study, inputs, observables, fixed, fidelity)
    n = len(setup.names)
    points = 4 * n + 4 if points is None else int(points)
    lower, upper = setup.box(width)
    scale = upper - lower
    rng = np.random.default_rng(seed)
    lhs = scipy.stats.qmc.LatinHypercube(d=n, seed=rng)
    xs, ys, dys = [], [], []

    def add(x: np.ndarray) -> None:
        out = setup.evaluate(x, jacobian=gradients)
        if out is None:
            return
        xs.append(np.asarray(x, dtype=float))
        ys.append(out[0])
        if gradients:
            dys.append(out[1] * scale)

    def fitted() -> GlobalSurrogate:
        if len(xs) < 2:
            raise ValueError("build_global_surrogate: fewer than two successful evaluations")
        model = MultiOutputGP(kernel=kernel, noise=0.0, seed=seed)
        x = np.array(xs)
        model.fit((x - lower) / scale, np.array(ys), np.array(dys) if gradients else None)
        return GlobalSurrogate(list(setup.names), list(setup.observables), lower, upper, model,
                               x, np.array(ys))  # fmt: skip

    for u in lhs.random(points):
        add(lower + u * scale)
    surrogate = fitted()
    for _ in range(int(active)):
        pool = lower + lhs.random(candidates) * scale
        add(pool[int(np.argmax(_relative_std(surrogate, pool)))])
        surrogate = fitted()
    pool = lower + lhs.random(candidates) * scale
    surrogate.max_relative_std = float(_relative_std(surrogate, pool).max())
    return surrogate


# --- Monte Carlo ---------------------------------------------------------------------------------


@dataclass
class MonteCarloResult:
    """Output statistics of ``samples`` input draws: ``mean``, ``std`` (m,), ``quantiles``
    {q: (m,)}, ``outputs`` (N, m) and ``inputs`` (N, n); ``clipped`` the number of draws
    moved into the surrogate box; ``surrogate_std`` the mean predicted surrogate standard
    deviation per output (0 for a plain function)."""

    observables: list[str]
    mean: np.ndarray
    std: np.ndarray
    quantiles: dict[float, np.ndarray]
    outputs: np.ndarray
    inputs: np.ndarray
    clipped: int = 0
    surrogate_std: np.ndarray = field(default_factory=lambda: np.zeros(0))


def _sample_inputs(inputs: Mapping[str, Distribution], u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    return np.column_stack([dist.ppf(u[:, j]) for j, dist in enumerate(inputs.values())])


def _model(model, inputs: Mapping[str, Distribution]) -> tuple[Callable, list[str], Any]:
    if isinstance(model, GlobalSurrogate):
        if list(inputs) != model.inputs:
            raise ValueError(
                f"the inputs {list(inputs)} do not match the surrogate's {model.inputs}"
            )
        return model, model.observables, model
    return model, [], None


def monte_carlo(model, inputs: Mapping[str, Distribution], *, samples: int = 10000,
                seed=0) -> MonteCarloResult:  # fmt: skip
    """Monte-Carlo propagation through a :class:`GlobalSurrogate` (its mean) or a function
    ``f(x)`` of the input samples ``x`` (N, n) returning (N,) or (N, m)."""
    function, observables, surrogate = _model(model, inputs)
    rng = np.random.default_rng(seed)
    x = _sample_inputs(inputs, rng.random((int(samples), len(inputs))))
    clipped = 0
    surrogate_std = np.zeros(0)
    if surrogate is not None:
        inside = np.all((x >= surrogate.lower) & (x <= surrogate.upper), axis=1)
        clipped = int(np.count_nonzero(~inside))
        x = np.clip(x, surrogate.lower, surrogate.upper)
        y, var = surrogate.predict(x)
        surrogate_std = np.sqrt(var).mean(axis=0)
    else:
        y = np.asarray(function(x), dtype=float)
    y = y.reshape(len(x), -1)
    return MonteCarloResult(
        observables or [f"y{i}" for i in range(y.shape[1])], y.mean(axis=0), y.std(axis=0, ddof=1),
        {q: np.quantile(y, q, axis=0) for q in _QUANTILES}, y, x, clipped, surrogate_std,
    )  # fmt: skip


# --- Sobol' indices ------------------------------------------------------------------------------


@dataclass
class SobolResult:
    """First-order ``first`` and total ``total`` Sobol' indices (m, n) of the ``observables``
    with respect to the ``inputs``, their bootstrap confidence half-widths ``first_conf`` /
    ``total_conf`` (95 %), the output ``variance`` (m,) and the number of base ``samples``."""

    observables: list[str]
    inputs: list[str]
    first: np.ndarray
    total: np.ndarray
    first_conf: np.ndarray
    total_conf: np.ndarray
    variance: np.ndarray
    samples: int

    def table(self, observable: int | str = 0) -> str:
        i = self.observables.index(observable) if isinstance(observable, str) else int(observable)
        lines = [f"{'input':<14} {'S_i':>8} {'±':>6} {'S_Ti':>8} {'±':>6}"]
        for j, name in enumerate(self.inputs):
            lines.append(f"{name:<14} {self.first[i, j]:8.4f} {self.first_conf[i, j]:6.3f} "
                         f"{self.total[i, j]:8.4f} {self.total_conf[i, j]:6.3f}")  # fmt: skip
        return "\n".join(lines)


def _sobol_estimates(fa, fb, fab):
    """First (Saltelli 2010) and total (Jansen) indices from f(A), f(B) (N, m) and f(A_B^i)
    (n, N, m)."""
    variance = np.var(np.concatenate([fa, fb]), axis=0)
    variance = np.where(variance > 0, variance, np.nan)
    first = np.mean(fb[None] * (fab - fa[None]), axis=1) / variance
    total = 0.5 * np.mean((fa[None] - fab) ** 2, axis=1) / variance
    return first.T, total.T, variance


def sobol_indices(model, inputs: Mapping[str, Distribution], *, samples: int = 4096,
                  bootstrap: int = 200, seed=0) -> SobolResult:  # fmt: skip
    """First-order and total Sobol' indices (module docstring) of a :class:`GlobalSurrogate`
    (its mean; normal inputs clipped to its box) or a vectorised function ``f(x)`` (N, n) →
    (N,) or (N, m); ``samples`` base points (rounded up to a power of two for the scrambled
    Sobol' sequence), ``bootstrap`` resamples for the confidence half-widths."""
    function, observables, surrogate = _model(model, inputs)
    n = len(inputs)
    m_exp = max(1, math.ceil(math.log2(max(int(samples), 2))))
    rng = np.random.default_rng(seed)
    points = scipy.stats.qmc.Sobol(d=2 * n, scramble=True, seed=rng).random_base2(m_exp)
    a = _sample_inputs(inputs, points[:, :n])
    b = _sample_inputs(inputs, points[:, n:])
    big_n = len(a)
    blocks = [a, b]
    for i in range(n):
        ab = a.copy()
        ab[:, i] = b[:, i]
        blocks.append(ab)
    x = np.concatenate(blocks)
    if surrogate is not None:
        x = np.clip(x, surrogate.lower, surrogate.upper)
    y = np.asarray(function(x), dtype=float).reshape(len(x), -1)
    fa, fb = y[:big_n], y[big_n : 2 * big_n]
    fab = y[2 * big_n :].reshape(n, big_n, -1)
    first, total, variance = _sobol_estimates(fa, fb, fab)
    boot_first, boot_total = [], []
    for _ in range(int(bootstrap)):
        idx = rng.integers(0, big_n, big_n)
        f, t, _ = _sobol_estimates(fa[idx], fb[idx], fab[:, idx])
        boot_first.append(f)
        boot_total.append(t)
    z = 1.959963984540054
    first_conf = z * np.std(boot_first, axis=0) if bootstrap else np.full_like(first, np.nan)
    total_conf = z * np.std(boot_total, axis=0) if bootstrap else np.full_like(total, np.nan)
    return SobolResult(observables or [f"y{i}" for i in range(first.shape[0])], list(inputs),
                       first, total, first_conf, total_conf, variance, big_n)  # fmt: skip


__all__ = [
    "GlobalSurrogate",
    "LinearPropagation",
    "MonteCarloResult",
    "Normal",
    "SobolResult",
    "Uniform",
    "build_global_surrogate",
    "linear_propagation",
    "monte_carlo",
    "sobol_indices",
]

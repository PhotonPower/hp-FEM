"""Parameter retrieval beyond the Laplace approximation (M16 S6, ADR-0012 §6).

**Posterior.** For a least-squares reconstruction (:func:`hpfem.opt.fit`) with measurements
``y_meas``, standard deviations ``σ`` and the model ``y(p)`` the posterior of the free
parameters is

.. math:: \\log π(p \\mid y) =
   -\\tfrac12 \\sum_i \\frac{(y_i(p) - y_{\\mathrm{meas},i})^2}{s^2σ_i^2}
   + \\log π_0(p) + \\mathrm{const},

with ``s² = χ²_red`` of the fit when ``σ`` is unknown (the scaling of :func:`hpfem.opt.laplace`;
a plug-in noise level) and ``s² = 1`` otherwise, and the prior ``π₀`` uniform on the bounds of
the design space, optionally times independent Gaussians ``(mean, std)`` per parameter.
:func:`sample` draws from it with the affine-invariant ensemble sampler of **emcee**
(Foreman-Mackey et al. 2013), the optional extra ``opt-mcmc`` (ADR-0012 §1). The Laplace
approximation of the fit is the Gaussian of this posterior at its mode; the samples show
where it is not (skewed or banana-shaped posteriors, parameters near a bound).

**Surrogate** (Bayesian least squares on the surrogate). A sampler needs 10⁴–10⁵ posterior
evaluations, far too many solves. :func:`build_surrogate` therefore evaluates the model at a
Latin-hypercube design in the box ``x̂ ± width · std`` around the optimum (clipped to the
bounds; the optimum itself included) **with the Jacobian** — one tangent or adjoint solve per
parameter on the kept factorisation (M16 S1) — and fits a gradient-enhanced Gaussian process
per observable (:class:`hpfem.opt.MultiOutputGP`, Matérn 5/2, inputs scaled to the box, no
noise: the evaluations are deterministic). The surrogate's own uncertainty enters the
likelihood (``s²σ_i² + var_i(p)``, with the matching log-determinant), so that the posterior
widens where the surrogate is unsure instead of trusting it blindly; ``validation`` extra
points check it (``Surrogate.validation``: the largest error over ``sσ``). The training and
validation evaluations go through the study (cached, stored, resumable). ``surrogate=False``
samples the model directly: the evaluator is called at every step **without** the study's
cache and store (10⁴–10⁵ calls would flood the store) — only for cheap evaluators.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.stats

from hpfem.opt.gp import MultiOutputGP
from hpfem.opt.lsq import FitResult, Laplace

_QUANTILES = (0.025, 0.16, 0.5, 0.84, 0.975)


def _emcee():
    try:
        import emcee
    except ImportError as error:  # the optional extra
        raise ImportError(
            "hpfem.opt.sample needs emcee: pip install 'hpfem[opt-mcmc]' (or pip install emcee)"
        ) from error
    return emcee


class PosteriorWarning(UserWarning):
    """The chain is too short for its autocorrelation time, the acceptance is outside the
    usual range, or the surrogate error is not small against the noise."""


@dataclass
class _Problem:
    """What the posterior of a fit needs: free parameters, their bounds, the fixed values,
    the observable rows, the data, the noise and the study."""

    names: list[str]
    lower: np.ndarray
    upper: np.ndarray
    columns: list[int]
    rows: list[int]
    params: dict[str, Any]
    y_meas: np.ndarray
    sigma: np.ndarray
    x_hat: np.ndarray
    std: np.ndarray
    study: Any
    fidelity: dict

    @classmethod
    def from_fit(cls, result: FitResult) -> _Problem:
        study = result.study
        space_names = list(study.space.names)
        bounds = study.space.bounds()
        columns = [space_names.index(n) for n in result.names]
        observables = list(study.evaluator.observables)
        rows = [observables.index(o) for o in result.observables]
        y_meas = np.asarray(result.values, dtype=float) - np.asarray(result.residual, dtype=float)
        sigma = np.ones(len(rows)) if result.sigma is None else np.asarray(result.sigma, float)
        if result.laplace.scaled:
            sigma = sigma * math.sqrt(result.chi2_red)
        std = np.asarray(result.std, dtype=float)
        if not np.all(np.isfinite(std) & (std > 0)):
            raise ValueError(
                "posterior: the Laplace standard errors are not finite (non-identifiable "
                f"parameters {result.laplace.non_identifiable}); fix them or add a prior"
            )
        return cls(list(result.names), bounds[columns, 0], bounds[columns, 1], columns, rows,
                   dict(result.params), y_meas, sigma, np.asarray(result.x, dtype=float), std,
                   study, dict(result.fidelity or {}))  # fmt: skip

    def params_at(self, x: np.ndarray) -> dict[str, Any]:
        out = dict(self.params)
        for name, value in zip(self.names, x, strict=True):
            out[name] = float(value)
        return out

    def evaluate(self, x: np.ndarray, jacobian: bool, direct: bool = False):
        params = self.params_at(x)
        fidelity = self.fidelity or None
        if direct:  # the evaluator itself, bypassing the study's cache and store
            evaluation = self.study.evaluator(params, jacobian=jacobian, fidelity=fidelity)
        else:
            evaluation = self.study.evaluate(params, jacobian=jacobian, fidelity=fidelity)
        if not evaluation.ok:
            return None
        values = np.asarray(evaluation.values, dtype=float)[self.rows]
        if not jacobian:
            return values, None
        jac = np.asarray(evaluation.jacobian, dtype=float)[np.ix_(self.rows, self.columns)]
        return values, jac


# --- surrogate -----------------------------------------------------------------------------------


@dataclass
class Surrogate:
    """Gradient-enhanced Gaussian-process model of the fitted observables on the box
    ``[lower, upper]`` of the free parameters ``names`` (SI). :meth:`predict` gives mean and
    variance (q, m) at points (q, n); ``validation`` is the largest ``|y - ŷ| / (sσ)`` at the
    validation points (``rms_validation`` their root mean square), ``points`` the number of
    training evaluations."""

    names: list[str]
    lower: np.ndarray
    upper: np.ndarray
    model: MultiOutputGP
    points: int
    validation: float = math.nan
    rms_validation: float = math.nan

    def encode(self, x) -> np.ndarray:
        return (np.atleast_2d(np.asarray(x, dtype=float)) - self.lower) / (self.upper - self.lower)

    def predict(self, x) -> tuple[np.ndarray, np.ndarray]:
        mean, var = self.model.predict(self.encode(x))
        return mean, var


def build_surrogate(result: FitResult, *, width: float = 4.0, points: int | None = None,
                    validation: int | None = None, seed=0,
                    kernel: str = "matern52") -> Surrogate:  # fmt: skip
    """Surrogate of the observables of ``result`` around its optimum (module docstring):
    ``points`` (default ``2n + 2``) Latin-hypercube evaluations with the Jacobian in
    ``x̂ ± width · std`` (clipped to the bounds) plus the optimum, ``validation`` (default
    ``n + 2``) further points without the Jacobian to measure the surrogate error. Failed
    evaluations are skipped (a warning)."""
    problem = _Problem.from_fit(result)
    n = len(problem.names)
    points = 2 * n + 2 if points is None else int(points)
    validation = n + 2 if validation is None else int(validation)
    lower = np.maximum(problem.lower, problem.x_hat - width * problem.std)
    upper = np.minimum(problem.upper, problem.x_hat + width * problem.std)
    if np.any(upper <= lower):
        raise ValueError("build_surrogate: the box around the optimum is empty")
    rng = np.random.default_rng(seed)
    sampler = scipy.stats.qmc.LatinHypercube(d=n, seed=rng)
    design = [problem.x_hat] + list(lower + sampler.random(points) * (upper - lower))
    xs, ys, dys, failed = [], [], [], 0
    scale = upper - lower
    for x in design:
        out = problem.evaluate(np.asarray(x), jacobian=True)
        if out is None:
            failed += 1
            continue
        xs.append((np.asarray(x) - lower) / scale)
        ys.append(out[0])
        dys.append(out[1] * scale)  # dy/du in the box coordinates
    if failed:
        warnings.warn(f"build_surrogate: {failed} training evaluations failed", PosteriorWarning,
                      stacklevel=2)  # fmt: skip
    if len(xs) < 2:
        raise ValueError("build_surrogate: fewer than two successful training evaluations")
    model = MultiOutputGP(kernel=kernel, noise=0.0, seed=seed)
    model.fit(np.array(xs), np.array(ys), np.array(dys))
    surrogate = Surrogate(list(problem.names), lower, upper, model, len(xs))
    if validation > 0:
        checks = lower + sampler.random(validation) * scale
        errors = []
        for x in checks:
            out = problem.evaluate(x, jacobian=False)
            if out is None:
                continue
            mean, _ = surrogate.predict(x)
            errors.append(np.abs(mean[0] - out[0]) / problem.sigma)
        if errors:
            errors = np.concatenate(errors)
            surrogate.validation = float(errors.max())
            surrogate.rms_validation = float(np.sqrt(np.mean(errors**2)))
            if surrogate.validation > 0.3:
                warnings.warn(
                    f"build_surrogate: the surrogate error reaches {surrogate.validation:.2g} "
                    "noise standard deviations; add points or shrink the box", PosteriorWarning,
                    stacklevel=2,
                )  # fmt: skip
    return surrogate


# --- sampling ------------------------------------------------------------------------------------


@dataclass
class PosteriorResult:
    """Posterior samples of the free parameters ``names`` (SI): ``samples`` (N, n) after
    burn-in and thinning, ``log_prob`` (N,); ``mean``, ``std``, ``covariance``,
    ``correlation``; ``quantiles`` {q: (n,)} for 2.5, 16, 50, 84 and 97.5 %; ``acceptance``
    (mean acceptance fraction), ``autocorr`` (integrated autocorrelation time per parameter,
    steps; NaN if the chain is too short to estimate it), ``walkers``, ``steps``, ``burn``,
    ``thin``; ``noise_scale`` ``s``; the ``laplace`` of the fit, its optimum ``x_hat``; the
    ``surrogate`` (or ``None``) and the ``warnings`` issued."""

    names: list[str]
    samples: np.ndarray
    log_prob: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    covariance: np.ndarray
    correlation: np.ndarray
    quantiles: dict[float, np.ndarray]
    acceptance: float
    autocorr: np.ndarray
    walkers: int
    steps: int
    burn: int
    thin: int
    noise_scale: float
    laplace: Laplace
    x_hat: np.ndarray
    surrogate: Surrogate | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def errors(self) -> dict[str, float]:
        """Posterior standard deviations by parameter name (SI)."""
        return dict(zip(self.names, (float(s) for s in self.std), strict=True))

    def compare(self) -> list[dict[str, float]]:
        """Per parameter: the Laplace mode and standard error, the posterior mean and standard
        deviation, the shift of the mean in Laplace standard errors and the ratio of the
        standard deviations (1 for a Gaussian posterior)."""
        rows = []
        for i, name in enumerate(self.names):
            s_l = float(self.laplace.std[i])
            rows.append({
                "name": name, "laplace": float(self.x_hat[i]), "laplace_std": s_l,
                "mean": float(self.mean[i]), "std": float(self.std[i]),
                "shift": float((self.mean[i] - self.x_hat[i]) / s_l),
                "ratio": float(self.std[i] / s_l),
            })  # fmt: skip
        return rows

    def summary(self) -> str:
        """A table of Laplace against posterior per parameter, and the chain statistics."""
        lines = [f"{'parameter':<14} {'Laplace':>14} {'± std':>10} {'posterior':>14} {'± std':>10}"
                 f" {'shift':>7} {'ratio':>6}"]  # fmt: skip
        for r in self.compare():
            lines.append(f"{r['name']:<14} {r['laplace']:>14.7g} {r['laplace_std']:>10.3g} "
                         f"{r['mean']:>14.7g} {r['std']:>10.3g} {r['shift']:>7.2f} "
                         f"{r['ratio']:>6.2f}")  # fmt: skip
        tau = np.nanmax(self.autocorr) if np.any(np.isfinite(self.autocorr)) else math.nan
        lines.append(f"{len(self.samples)} samples, {self.walkers} walkers x {self.steps} steps "
                     f"(burn {self.burn}), acceptance {self.acceptance:.2f}, autocorrelation "
                     f"{tau:.3g} steps" + (f", surrogate error {self.surrogate.validation:.2g} σ"
                                           if self.surrogate is not None else ""))  # fmt: skip
        return "\n".join(lines)


def sample(result: FitResult, *, surrogate: Surrogate | bool = True,
           prior: Mapping[str, tuple[float, float]] | None = None, walkers: int | None = None,
           steps: int = 3000, burn: int | None = None, thin: int = 1, seed=0,
           surrogate_variance: bool = True, progress: bool = False) -> PosteriorResult:  # fmt: skip
    """Samples the posterior of the free parameters of a fit (module docstring) with emcee.

    ``surrogate``: ``True`` builds one with :func:`build_surrogate` (default options), a
    :class:`Surrogate` is used as given, ``False`` evaluates the model at every step (cheap
    evaluators only). ``prior``: Gaussian ``(mean, std)`` per parameter name (SI) on top of the
    uniform prior on the bounds. ``walkers`` (default ``max(4n, 16)``, even), ``steps`` per
    walker, ``burn`` (default ``steps // 4``), ``thin``; the walkers start from the Laplace
    Gaussian of the fit (inside the bounds). ``surrogate_variance`` adds the surrogate's
    variance to the noise. Warnings (:class:`PosteriorWarning`) when the chain is shorter than
    50 autocorrelation times or the acceptance is below 0.15."""
    emcee = _emcee()
    problem = _Problem.from_fit(result)
    n = len(problem.names)
    unknown = set(prior or {}) - set(problem.names)
    if unknown:
        raise ValueError(f"sample: prior for unknown or fixed parameters {sorted(unknown)}")
    prior_index = [
        (problem.names.index(k), float(m), float(s)) for k, (m, s) in (prior or {}).items()
    ]
    if surrogate is True:
        surrogate = build_surrogate(result, seed=seed)
    elif surrogate is False:
        surrogate = None
    walkers = max(4 * n, 16) if walkers is None else int(walkers)
    walkers += walkers % 2
    burn = steps // 4 if burn is None else int(burn)
    if not 0 <= burn < steps:
        raise ValueError("sample: burn must lie in [0, steps)")
    s2 = problem.sigma**2

    def log_prior(x: np.ndarray) -> np.ndarray:
        inside = np.all((x >= problem.lower) & (x <= problem.upper), axis=1)
        out = np.where(inside, 0.0, -np.inf)
        for i, m, s in prior_index:
            out = out - 0.5 * ((x[:, i] - m) / s) ** 2
        return out

    def log_prob(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        lp = log_prior(x)
        ok = np.isfinite(lp)
        if not ok.any():
            return lp
        if surrogate is not None:
            mean, var = surrogate.predict(x[ok])
            total = s2 + (var if surrogate_variance else 0.0)
            ll = -0.5 * np.sum((mean - problem.y_meas) ** 2 / total + np.log(total / s2), axis=1)
        else:
            ll = np.empty(int(ok.sum()))
            for j, xj in enumerate(x[ok]):
                out = problem.evaluate(xj, jacobian=False, direct=True)
                ll[j] = (
                    -np.inf if out is None else -0.5 * np.sum((out[0] - problem.y_meas) ** 2 / s2)
                )
        lp[ok] = lp[ok] + ll
        return lp

    rng = np.random.default_rng(seed)
    cov = np.asarray(result.covariance, dtype=float)
    start = rng.multivariate_normal(problem.x_hat, cov, size=4 * walkers)
    start = start[np.all((start >= problem.lower) & (start <= problem.upper), axis=1)]
    start = start[np.isfinite(log_prob(start))][:walkers]
    if len(start) < walkers:
        raise ValueError("sample: could not place the walkers inside the bounds around the optimum")
    sampler = emcee.EnsembleSampler(walkers, n, log_prob, vectorize=True)
    # reproducible moves without touching numpy's global generator (emcee keeps a
    # RandomState in _random; its public random_state setter only takes a state tuple)
    sampler._random = np.random.RandomState(int(rng.integers(2**31)))
    sampler.run_mcmc(start, steps, progress=progress)
    chain = sampler.get_chain(discard=burn, thin=thin, flat=True)
    log_p = sampler.get_log_prob(discard=burn, thin=thin, flat=True)
    issued: list[str] = []
    try:
        tau = np.asarray(sampler.get_autocorr_time(discard=burn, quiet=True), dtype=float)
    except Exception:  # emcee raises for chains far too short to estimate tau
        tau = np.full(n, math.nan)
    if np.any(np.isfinite(tau)) and (steps - burn) < 50 * np.nanmax(tau):
        issued.append(f"the chain ({steps - burn} steps after burn-in) is shorter than 50 "
                      f"autocorrelation times ({np.nanmax(tau):.3g} steps)")  # fmt: skip
    acceptance = float(np.mean(sampler.acceptance_fraction))
    if acceptance < 0.15:  # high acceptance is normal for the stretch move in low dimension
        issued.append(f"acceptance fraction {acceptance:.2f} below 0.15")
    for message in issued:
        warnings.warn(f"sample: {message}", PosteriorWarning, stacklevel=2)
    covariance = np.atleast_2d(np.cov(chain, rowvar=False))
    std = np.sqrt(np.diag(covariance))
    return PosteriorResult(
        names=list(problem.names), samples=chain, log_prob=log_p, mean=chain.mean(axis=0),
        std=std, covariance=covariance, correlation=covariance / np.outer(std, std),
        quantiles={q: np.quantile(chain, q, axis=0) for q in _QUANTILES}, acceptance=acceptance,
        autocorr=tau, walkers=walkers, steps=steps, burn=burn, thin=thin,
        noise_scale=float(math.sqrt(result.chi2_red)) if result.laplace.scaled else 1.0,
        laplace=result.laplace, x_hat=problem.x_hat, surrogate=surrogate, warnings=issued,
    )  # fmt: skip


__all__ = ["PosteriorResult", "PosteriorWarning", "Surrogate", "build_surrogate", "sample"]

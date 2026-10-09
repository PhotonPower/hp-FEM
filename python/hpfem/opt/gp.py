"""A light Gaussian process with derivative observations (M16 S5, ADR-0012 §1, §6).

NumPy / SciPy only, meant for the surrogates of :mod:`hpfem.opt` (Bayesian optimisation,
later the Bayesian least squares and the Sobol' indices on the surrogate): a few hundred
observations (plain) or ``n (d + 1)`` of a few hundred (gradient-enhanced). Everything is
dense and ``O(N³)`` in the number ``N`` of observations — a Cholesky factorisation per
likelihood evaluation and an explicit inverse for the likelihood gradient; ``N ≈ 2000`` takes
seconds per hyperparameter fit and is the practical limit.

**Model.** ``f ~ GP(m, k)`` on inputs ``x ∈ R^d`` (the unit cube of a
:class:`~hpfem.opt.DesignSpace` in :mod:`hpfem.opt.bo`) with a stationary kernel with ARD
length scales ``ℓ_i`` and signal variance ``σ_f²``,

    s = Σ_i (x_i − x'_i)² / ℓ_i²,   r = √s,   k(x, x') = σ_f² φ(s),

- Matérn 5/2 (default): ``φ = (1 + √5 r + 5 r²/3) exp(−√5 r)``,
- Matérn 3/2: ``φ = (1 + √3 r) exp(−√3 r)``,
- squared exponential: ``φ = exp(−s/2)``,

a constant mean ``m`` (``mean="constant"``, profiled out by generalised least squares,
``m̂ = hᵀK⁻¹y / hᵀK⁻¹h`` with ``h`` = 1 for values and 0 for derivatives; a plug-in estimate,
its uncertainty is not added to the predictive variance) or ``m = 0`` (``mean="zero"``, after
centring), and homoscedastic Gaussian noise ``σ_n²`` on every observation, learned
(``noise="learn"``, the default) or fixed (a number; ``0`` means only the jitter).
Observations are standardised internally: ``y ← (y − ȳ) / s_y``, derivatives ``← ∂y / s_y``;
``σ_f²``, ``σ_n²`` and the bounds below refer to the standardised data, the predictions and
:attr:`GaussianProcess.hyperparameters` are in the units of ``y``.

**Derivative observations** (gradient-enhanced GP, "GEK"). Values ``y_a = f(x_a)`` and partial
derivatives ``g_{a,i} = ∂f/∂x_i(x_a)`` are jointly Gaussian; with ``W_i = (x_i − x'_i)/ℓ_i²``
and ``φ', φ''`` the derivatives of ``φ`` with respect to ``s``,

    cov(f(x),     f(x'))      = σ_f² φ
    cov(f(x),     ∂_j f(x'))  = −2 σ_f² φ' W_j
    cov(∂_i f(x), f(x'))      =  2 σ_f² φ' W_i
    cov(∂_i f(x), ∂_j f(x'))  = −σ_f² (4 φ'' W_i W_j + 2 φ' δ_ij / ℓ_i²)

(all kernels are at least twice differentiable at ``r = 0``; the ``1/r`` factors of the
Matérn derivatives always multiply a power of ``W`` that vanishes faster). Observation vector
``[y_1 … y_n, g_{1,1} … g_{1,d}, g_{2,1} … g_{n,d}]`` (values first, then the gradients point
by point); a NaN derivative (e.g. a fixed parameter, an evaluation without Jacobian) is simply
left out. Derivatives are taken in the coordinates of ``x`` — :mod:`hpfem.opt.bo` chain-rules
the evaluator's Jacobian (SI) to the unit cube.

**Hyperparameters** ``θ = (log ℓ_1 … log ℓ_d, log σ_f² [, log σ_n²])`` maximise the log
posterior ``log p(y | θ) + log p(θ)`` with

    log p(y | θ) = −½ rᵀK⁻¹r − ½ log det K − (N/2) log 2π,   r = y − m h,
    ∂ log p / ∂θ_k = ½ tr((ααᵀ − K⁻¹) ∂K/∂θ_k),   α = K⁻¹ r

(the GLS mean maximises the likelihood for every θ, so its derivative drops out) by L-BFGS-B
in log space from several starts: the previous optimum (warm start), the prior centre and
``restarts`` random points of the box (seeded). Bounds (standardised data, unit-cube inputs):
``ℓ ∈ [0.01, 100]``, ``σ_f² ∈ [1e-4, 1e4]``, ``σ_n² ∈ [1e-10, 1]``. Priors (``prior=True``):
``log ℓ_i ~ N(log(0.5 √d), 1.5²)`` and ``log σ_f² ~ N(0, 2²)``, weak enough to be overruled by
a few points; the noise has no prior (only its bounds), so noise-free smooth data drive it to
its lower bound. ``K`` is factorised by Cholesky with a jitter ``1e-10`` (standardised units)
on the diagonal, raised tenfold up to ``1e-4`` until the factorisation succeeds
(:attr:`GaussianProcess.jitter_`).

**Prediction** at ``x*``: ``μ = m + k*ᵀα``, ``v = σ_f² − k*ᵀK⁻¹k*`` (clipped at 0), the
covariance of a few points, and the gradients with respect to ``x*``,
``∇μ = (∂k*/∂x*)ᵀα`` (the posterior mean of ``∇f``) and ``∇v = −2 (∂k*/∂x*)ᵀK⁻¹k*``, needed by
the acquisition optimisation.

The DWR error estimate of an evaluation is **not** observation noise (ADR-0012 §6: the
discretisation error is systematic, smooth and signed); nothing in this module reads it.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.linalg
import scipy.optimize

KERNELS = ("matern52", "matern32", "se")
"""Kernels of :class:`GaussianProcess`."""

_ALIASES = {
    "matern52": "matern52",
    "matern-5/2": "matern52",
    "matern5/2": "matern52",
    "matern32": "matern32",
    "matern-3/2": "matern32",
    "matern3/2": "matern32",
    "se": "se",
    "rbf": "se",
    "squared-exponential": "se",
    "squared_exponential": "se",
}

_LOG_2PI = math.log(2.0 * math.pi)


def _kernel(name: str) -> str:
    key = str(name).lower().replace(" ", "")
    if key not in _ALIASES:
        raise ValueError(f"GaussianProcess: kernel {name!r}, use one of {KERNELS}")
    return _ALIASES[key]


def _radial(kernel: str, s: np.ndarray) -> tuple[np.ndarray, ...]:
    """``φ(s), φ'(s), φ''(s), φ'''(s)`` (derivatives with respect to ``s = r²``); the factors
    ``1/r`` are set to 0 at ``r = 0`` (they always multiply a vanishing power of ``W``)."""
    s = np.maximum(s, 0.0)
    if kernel == "se":
        e = np.exp(-0.5 * s)
        return e, -0.5 * e, 0.25 * e, -0.125 * e
    r = np.sqrt(s)
    positive = r > 0.0
    rinv = np.where(positive, 1.0 / np.where(positive, r, 1.0), 0.0)
    if kernel == "matern52":
        c = math.sqrt(5.0)
        e = np.exp(-c * r)
        return ((1.0 + c * r + (c * c / 3.0) * s) * e, -(c * c / 6.0) * (1.0 + c * r) * e,
                (c**4 / 12.0) * e, -(c**5 / 24.0) * e * rinv)  # fmt: skip
    c = math.sqrt(3.0)
    e = np.exp(-c * r)
    return ((1.0 + c * r) * e, -(c * c / 2.0) * e, (c**3 / 4.0) * e * rinv,
            -(c**3 / 8.0) * (c * r + 1.0) * e * rinv**3)  # fmt: skip


class _Pair:
    """Distances of two point sets and the radial functions (shared by the covariance and its
    length-scale derivatives)."""

    def __init__(self, kernel: str, x1: np.ndarray, x2: np.ndarray, ls: np.ndarray):
        self.n1, self.n2, self.d = len(x1), len(x2), x1.shape[1]
        self.inv2 = 1.0 / ls**2
        delta = x1[:, None, :] - x2[None, :, :]
        self.w = delta * self.inv2  # (n1, n2, d)
        self.q = delta * delta * self.inv2
        self.p = _radial(kernel, self.q.sum(axis=-1))

    def _join(self, ff, fg, gf, gg, g1: bool, g2: bool) -> np.ndarray:
        n1, n2, d = self.n1, self.n2, self.d
        rows = [ff]
        if g2:
            rows = [np.concatenate([ff, fg.reshape(n1, n2 * d)], axis=1)]
        if g1:
            bottom = gf.transpose(0, 2, 1).reshape(n1 * d, n2)
            if g2:
                bottom = np.concatenate([bottom, gg.transpose(0, 2, 1, 3).reshape(n1 * d, n2 * d)],
                                        axis=1)  # fmt: skip
            rows.append(bottom)
        return np.concatenate(rows, axis=0) if len(rows) > 1 else rows[0]

    def covariance(self, g1: bool, g2: bool) -> np.ndarray:
        """``φ``-part of the covariance (without ``σ_f²``), with derivative rows (``g1``) and
        columns (``g2``)."""
        p0, p1, p2, _ = self.p
        w = self.w
        fg = -2.0 * p1[..., None] * w if g2 else None
        gf = 2.0 * p1[..., None] * w if g1 else None
        gg = None
        if g1 and g2:
            gg = -(4.0 * p2[..., None, None] * w[..., :, None] * w[..., None, :]
                   + 2.0 * p1[..., None, None] * np.diag(self.inv2))  # fmt: skip
        return self._join(p0, fg, gf, gg, g1, g2)

    def length_derivative(self, k: int, g1: bool, g2: bool) -> np.ndarray:
        """Derivative of :meth:`covariance` with respect to ``log ℓ_k``."""
        p0, p1, p2, p3 = self.p
        w, qk = self.w, self.q[..., k]
        ff = -2.0 * p1 * qk
        fg = gf = gg = None
        if g1 or g2:
            t = 4.0 * (p2 * qk)[..., None] * w
            t[..., k] += 4.0 * p1 * w[..., k]
            fg, gf = t, -t
        if g1 and g2:
            ww = w[..., :, None] * w[..., None, :]
            gg = 8.0 * (p3 * qk)[..., None, None] * ww
            gg[..., k, :] += 8.0 * p2[..., None] * ww[..., k, :]
            gg[..., :, k] += 8.0 * p2[..., None] * ww[..., :, k]
            gg += 4.0 * (p2 * qk)[..., None, None] * np.diag(self.inv2)
            gg[..., k, k] += 4.0 * p1 * self.inv2[k]
        return self._join(ff, fg, gf, gg, g1, g2)


def _as_points(x, d: int | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :] if d is not None and len(x) == d else x[:, None]
    if x.ndim != 2:
        raise ValueError(f"GaussianProcess: points must be (n, d), got shape {x.shape}")
    if d is not None and x.shape[1] != d:
        raise ValueError(f"GaussianProcess: points with {x.shape[1]} coordinates, model has {d}")
    if not np.all(np.isfinite(x)):
        raise ValueError("GaussianProcess: non-finite input coordinates")
    return x


class GaussianProcess:
    """Gaussian-process regression with optional derivative observations (module docstring).

    ``kernel``: ``"matern52"`` (default), ``"matern32"`` or ``"se"``. ``mean``:
    ``"constant"`` (GLS) or ``"zero"``. ``noise``: ``"learn"`` or a fixed variance in the
    units of ``y²`` (``0``: jitter only). ``length_scale_bounds``, ``signal_bounds``,
    ``noise_bounds``: hyperparameter boxes (unit-cube inputs, standardised data); ``prior``:
    the weak log-normal priors on ``ℓ`` and ``σ_f²``; ``restarts``: random starts of the
    hyperparameter fit besides the warm start and the prior centre; ``seed`` seeds them;
    ``jitter``: initial diagonal jitter (standardised units).

    Fit with :meth:`fit`, predict with :meth:`predict`; :meth:`log_marginal_likelihood`
    exposes the objective of the hyperparameter fit and its gradient."""

    def __init__(self, kernel: str = "matern52", *, mean: str = "constant", noise="learn",
                 length_scale_bounds: tuple[float, float] = (1e-2, 1e2),
                 signal_bounds: tuple[float, float] = (1e-4, 1e4),
                 noise_bounds: tuple[float, float] = (1e-10, 1.0), prior: bool = True,
                 restarts: int = 3, seed=0, jitter: float = 1e-10):  # fmt: skip
        self.kernel = _kernel(kernel)
        if mean not in ("constant", "zero"):
            raise ValueError(f"GaussianProcess: mean {mean!r}, use 'constant' or 'zero'")
        self.mean = mean
        if isinstance(noise, str):
            if noise != "learn":
                raise ValueError(f"GaussianProcess: noise {noise!r}, use 'learn' or a variance")
            self.noise = "learn"
        else:
            if not float(noise) >= 0.0:
                raise ValueError("GaussianProcess: a fixed noise variance must be >= 0")
            self.noise = float(noise)
        for label, (lo, hi) in (("length_scale_bounds", length_scale_bounds),
                                ("signal_bounds", signal_bounds),
                                ("noise_bounds", noise_bounds)):  # fmt: skip
            if not 0.0 < float(lo) < float(hi) < math.inf:
                raise ValueError(f"GaussianProcess: {label} must satisfy 0 < lower < upper")
        self.length_scale_bounds = (float(length_scale_bounds[0]), float(length_scale_bounds[1]))
        self.signal_bounds = (float(signal_bounds[0]), float(signal_bounds[1]))
        self.noise_bounds = (float(noise_bounds[0]), float(noise_bounds[1]))
        self.prior = bool(prior)
        self.restarts = int(restarts)
        self.seed = seed
        self.jitter = float(jitter)
        self.theta: np.ndarray | None = None
        """log hyperparameters ``(log ℓ, log σ_f² [, log σ_n²])`` after :meth:`fit`"""
        self.jitter_ = math.nan
        """the jitter of the last factorisation (standardised units)"""
        self.fit_info: dict = {}
        """``starts``, ``best`` (log posterior), ``converged`` of the last hyperparameter fit"""
        self._fitted = False

    # --- data --------------------------------------------------------------------------------

    @property
    def learns_noise(self) -> bool:
        return self.noise == "learn"

    @property
    def num_observations(self) -> int:
        """Number ``N`` of observations (values plus finite derivatives)."""
        return int(self._mask.sum()) if self._fitted else 0

    def _set_data(self, x, y, dy) -> None:
        x = _as_points(x)
        n, d = x.shape
        y = np.asarray(y, dtype=float).ravel()
        if len(y) != n:
            raise ValueError(f"GaussianProcess: {len(y)} values for {n} points")
        if not np.all(np.isfinite(y)):
            raise ValueError("GaussianProcess: non-finite values (leave failed points out)")
        if n < 1:
            raise ValueError("GaussianProcess: no data")
        grad = dy is not None
        if grad:
            dy = np.asarray(dy, dtype=float)
            if dy.shape != (n, d):
                raise ValueError(f"GaussianProcess: derivatives of shape {dy.shape}, expected "
                                 f"{(n, d)}")  # fmt: skip
            if not np.any(np.isfinite(dy)):
                grad, dy = False, None
        mean = float(y.mean())
        scale = float(y.std())
        if not scale > 1e-300 * max(1.0, abs(mean)):
            scale = max(abs(mean), 1.0)
        self.x, self.n, self.d = x, n, d
        self.y_raw, self.dy_raw = y, dy
        self.y_mean, self.y_scale = mean, scale
        self.grad = grad
        obs = (y - mean) / scale
        mask = np.ones(n, dtype=bool)
        if grad:
            flat = dy.ravel() / scale
            mask = np.concatenate([mask, np.isfinite(flat)])
            obs = np.concatenate([obs, np.where(np.isfinite(flat), flat, 0.0)])
        self._mask = mask
        self._obs = obs[mask]
        h = np.zeros(len(mask))
        h[:n] = 1.0
        self._h = h[mask]

    # --- hyperparameters ---------------------------------------------------------------------

    def _bounds(self) -> list[tuple[float, float]]:
        b = [tuple(math.log(v) for v in self.length_scale_bounds)] * self.d
        b.append(tuple(math.log(v) for v in self.signal_bounds))
        if self.learns_noise:
            b.append(tuple(math.log(v) for v in self.noise_bounds))
        return b

    def _prior_centre(self) -> np.ndarray:
        centre = [math.log(0.5 * math.sqrt(self.d))] * self.d + [0.0]
        if self.learns_noise:
            centre.append(math.log(max(self.noise_bounds[0], 1e-8)))
        lo, hi = np.array(self._bounds()).T
        return np.clip(np.array(centre), lo, hi)

    def _log_prior(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        grad = np.zeros_like(theta)
        if not self.prior:
            return 0.0, grad
        d = self.d
        mu = np.array([math.log(0.5 * math.sqrt(d))] * d + [0.0])
        sd = np.array([1.5] * d + [2.0])
        z = (theta[: d + 1] - mu) / sd
        grad[: d + 1] = -z / sd
        return float(-0.5 * z @ z), grad

    def _split(self, theta: np.ndarray) -> tuple[np.ndarray, float, float]:
        d = self.d
        ls = np.exp(theta[:d])
        sf2 = math.exp(theta[d])
        if self.learns_noise:
            sn2 = math.exp(theta[d + 1])
        else:
            sn2 = self.noise / self.y_scale**2
        return ls, sf2, sn2

    def _factor(self, k: np.ndarray) -> tuple[np.ndarray, float]:
        jitter = self.jitter
        diag = np.arange(len(k))
        while True:
            a = k.copy()
            a[diag, diag] += jitter
            try:
                return scipy.linalg.cholesky(a, lower=True, check_finite=False), jitter
            except np.linalg.LinAlgError:
                if jitter >= 1e-4:
                    raise
                jitter *= 10.0

    def _likelihood(self, theta: np.ndarray, gradient: bool):
        """Log marginal likelihood (standardised data) and its gradient; also returns the
        factorisation for :meth:`fit`."""
        ls, sf2, sn2 = self._split(theta)
        pair = _Pair(self.kernel, self.x, self.x, ls)
        g = self.grad
        mask = self._mask
        kc = pair.covariance(g, g)[np.ix_(mask, mask)]
        k = sf2 * kc
        k[np.diag_indices_from(k)] += sn2
        chol, jitter = self._factor(k)
        obs, h = self._obs, self._h
        if self.mean == "constant":
            kinv_h = scipy.linalg.cho_solve((chol, True), h, check_finite=False)
            m = float(kinv_h @ obs / (h @ kinv_h))
        else:
            m = 0.0
        r = obs - m * h
        alpha = scipy.linalg.cho_solve((chol, True), r, check_finite=False)
        n_obs = len(obs)
        value = (
            -0.5 * float(r @ alpha) - float(np.log(np.diag(chol)).sum()) - 0.5 * n_obs * _LOG_2PI
        )
        state = (chol, jitter, m, alpha, ls, sf2, sn2)
        if not gradient:
            return value, None, state
        kinv = scipy.linalg.cho_solve((chol, True), np.eye(n_obs), check_finite=False)
        a = np.outer(alpha, alpha) - kinv
        grad = np.zeros(len(theta))
        for i in range(self.d):
            dk = pair.length_derivative(i, g, g)[np.ix_(mask, mask)]
            grad[i] = 0.5 * sf2 * float(np.sum(a * dk))
        grad[self.d] = 0.5 * sf2 * float(np.sum(a * kc))
        if self.learns_noise:
            grad[self.d + 1] = 0.5 * sn2 * float(np.trace(a))
        return value, grad, state

    def log_marginal_likelihood(self, theta=None, *, gradient: bool = False,
                                prior: bool = False):  # fmt: skip
        """The log marginal likelihood of the standardised data at ``theta`` (log
        hyperparameters; default: the fitted ones), plus the log prior if ``prior``; with
        ``gradient`` a tuple ``(value, d value / d theta)``."""
        if not self._fitted:
            raise RuntimeError("GaussianProcess: fit first")
        theta = self.theta if theta is None else np.asarray(theta, dtype=float)
        value, grad, _ = self._likelihood(theta, gradient)
        if prior:
            lp, lg = self._log_prior(theta)
            value += lp
            if gradient:
                grad = grad + lg
        return (value, grad) if gradient else value

    def _negative_posterior(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        try:
            value, grad, _ = self._likelihood(theta, True)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return 1e25, np.zeros_like(theta)
        lp, lg = self._log_prior(theta)
        total = value + lp
        if not math.isfinite(total) or not np.all(np.isfinite(grad)):
            return 1e25, np.zeros_like(theta)
        return -total, -(grad + lg)

    def _optimize(self, warm: np.ndarray | None) -> np.ndarray:
        bounds = self._bounds()
        lo, hi = np.array(bounds).T
        starts = []
        if warm is not None and len(warm) == len(bounds):
            starts.append(np.clip(warm, lo, hi))
        starts.append(self._prior_centre())
        rng = np.random.default_rng(self.seed)
        for _ in range(self.restarts):
            u = rng.random(len(bounds))
            start = self._prior_centre()
            # random length scales and signal variance; the noise starts low
            start[: self.d + 1] = lo[: self.d + 1] + u[: self.d + 1] * (hi - lo)[: self.d + 1]
            starts.append(start)
        best, best_value, converged = None, math.inf, 0
        for start in starts:
            result = scipy.optimize.minimize(self._negative_posterior, start, jac=True,
                                             method="L-BFGS-B", bounds=bounds,
                                             options={"maxiter": 200})  # fmt: skip
            converged += bool(result.success)
            if result.fun < best_value:
                best, best_value = np.array(result.x), float(result.fun)
        if best is None or best_value >= 1e25:
            raise np.linalg.LinAlgError("GaussianProcess: no hyperparameters with a factorisable "
                                        "covariance")  # fmt: skip
        self.fit_info = {"starts": len(starts), "best": -best_value, "converged": converged}
        return best

    # --- fit and predict ---------------------------------------------------------------------

    def fit(self, x, y, dy=None, *, theta=None, optimize: bool = True) -> GaussianProcess:
        """Conditions the process on values ``y`` (n,) at ``x`` (n, d) and optionally partial
        derivatives ``dy`` (n, d) (NaN entries are left out). With ``optimize`` the
        hyperparameters are fitted (warm start: ``theta`` or the previous optimum); otherwise
        ``theta`` (or the previous / prior-centre values) is used as given. Returns ``self``."""
        warm = self.theta if theta is None else np.asarray(theta, dtype=float)
        self._set_data(x, y, dy)
        self._fitted = True
        expected = self.d + 1 + int(self.learns_noise)
        if warm is not None and len(warm) != expected:
            if theta is not None:
                raise ValueError(f"GaussianProcess: theta of length {len(warm)}, expected "
                                 f"{expected}")  # fmt: skip
            warm = None
        if optimize:
            self.theta = self._optimize(warm)
        else:
            self.theta = self._prior_centre() if warm is None else warm
        _, _, state = self._likelihood(self.theta, False)
        self._chol, self.jitter_, self._m, self._alpha, self._ls, self._sf2, self._sn2 = state
        return self

    @property
    def hyperparameters(self) -> dict:
        """``length_scales`` (input units), ``signal_variance``, ``noise_variance`` and
        ``mean`` in the units of ``y``, ``jitter`` (standardised)."""
        if not self._fitted:
            raise RuntimeError("GaussianProcess: fit first")
        s2 = self.y_scale**2
        return {"length_scales": self._ls.copy(), "signal_variance": self._sf2 * s2,
                "noise_variance": self._sn2 * s2,
                "mean": self.y_mean + self.y_scale * self._m, "jitter": self.jitter_}  # fmt: skip

    def _cross(self, xs: np.ndarray, g1: bool) -> np.ndarray:
        pair = _Pair(self.kernel, xs, self.x, self._ls)
        return self._sf2 * pair.covariance(g1, self.grad)[:, self._mask]

    def predict(self, x, *, grad: bool = False, full_cov: bool = False):
        """Posterior at the points ``x`` (q, d) (a single point may be given as (d,)):
        ``(mean, var)`` of shape (q,); with ``full_cov`` ``(mean, cov)`` with ``cov`` (q, q);
        with ``grad`` ``(mean, var, dmean, dvar)``, the gradients (q, d) with respect to ``x``
        (``dmean`` is the posterior mean of ``∇f``). Units of ``y``."""
        if not self._fitted:
            raise RuntimeError("GaussianProcess: fit first")
        xs = _as_points(x, self.d)
        q = len(xs)
        if grad and full_cov:
            raise ValueError("GaussianProcess.predict: grad and full_cov are exclusive")
        kx = self._cross(xs, grad)
        k = kx[:q]
        mean = self._m + k @ self._alpha
        v = scipy.linalg.solve_triangular(self._chol, k.T, lower=True, check_finite=False)
        s, s2 = self.y_scale, self.y_scale**2
        if full_cov:
            kss = self._sf2 * _Pair(self.kernel, xs, xs, self._ls).covariance(False, False)
            cov = kss - v.T @ v
            return self.y_mean + s * mean, s2 * cov
        var = np.maximum(self._sf2 - np.sum(v * v, axis=0), 0.0)
        if not grad:
            return self.y_mean + s * mean, s2 * var
        dk = kx[q:].reshape(q, self.d, -1)  # d k(x*, X) / d x*_i
        dmean = dk @ self._alpha
        kinv_k = scipy.linalg.solve_triangular(self._chol.T, v, lower=False, check_finite=False)
        dvar = -2.0 * np.einsum("qin,nq->qi", dk, kinv_k)
        return self.y_mean + s * mean, s2 * var, s * dmean, s2 * dvar

    def sample(self, x, size: int = 1, seed=None) -> np.ndarray:
        """``size`` joint samples (size, q) of the posterior at the points ``x`` (q, d)."""
        mean, cov = self.predict(x, full_cov=True)
        cov = 0.5 * (cov + cov.T)
        scale = max(float(np.max(np.diag(cov), initial=0.0)), 1e-300)
        jitter = 1e-12 * scale
        while True:
            try:
                chol = np.linalg.cholesky(cov + jitter * np.eye(len(mean)))
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
                if jitter > 1e-2 * scale:
                    raise
        rng = np.random.default_rng(seed)
        return mean + rng.standard_normal((int(size), len(mean))) @ chol.T

    def __repr__(self) -> str:
        if not self._fitted:
            return f"GaussianProcess({self.kernel}, not fitted)"
        h = self.hyperparameters
        return (f"GaussianProcess({self.kernel}, n={self.n}, N={self.num_observations}, "
                f"length_scales={np.array2string(h['length_scales'], precision=3)}, "
                f"signal_variance={h['signal_variance']:.3g}, "
                f"noise_variance={h['noise_variance']:.3g})")  # fmt: skip


class MultiOutputGP:
    """Independent Gaussian processes, one per output column, each with its own
    hyperparameters (observables of a study have different length scales and magnitudes;
    shared hyperparameters would force one compromise on all of them). ``options`` go to every
    :class:`GaussianProcess` (``seed`` is offset by the output index)."""

    def __init__(self, **options):
        self.options = dict(options)
        self.models: list[GaussianProcess] = []

    def fit(self, x, y, dy=None, *, optimize: bool = True) -> MultiOutputGP:
        """``y`` (n, m) values, ``dy`` (n, m, d) derivatives or ``None``."""
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        m = y.shape[1]
        if len(self.models) != m:
            seed = self.options.get("seed", 0)
            self.models = []
            for j in range(m):
                options = dict(self.options)
                if isinstance(seed, (int, np.integer)):
                    options["seed"] = int(seed) + j
                self.models.append(GaussianProcess(**options))
        for j, model in enumerate(self.models):
            model.fit(x, y[:, j], None if dy is None else np.asarray(dy)[:, j, :],
                      optimize=optimize)  # fmt: skip
        return self

    def predict(self, x, *, grad: bool = False):
        """``(mean, var)`` of shape (q, m), with ``grad`` also ``(dmean, dvar)`` (q, m, d)."""
        parts = [model.predict(x, grad=grad) for model in self.models]
        out = [np.stack([p[i] for p in parts], axis=1) for i in range(len(parts[0]))]
        return tuple(out)

    def __len__(self) -> int:
        return len(self.models)


__all__ = ["KERNELS", "GaussianProcess", "MultiOutputGP"]

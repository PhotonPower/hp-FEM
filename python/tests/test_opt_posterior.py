"""Parameter posterior beyond the Laplace approximation (M16 S6, ADR-0012 §6): on a linear
Gaussian model the MCMC posterior (direct and on the gradient-enhanced surrogate) reproduces
the Laplace mean, standard errors and correlations; a Gaussian prior narrows it by the
expected factor; on a skewed one-parameter model the MCMC quantiles match a brute-force grid
posterior where the Laplace Gaussian does not; the surrogate of a two-parameter nonlinear
model reproduces the direct posterior and reports a small validation error; argument checks
and the message of the missing optional extra."""

import math
import sys

import numpy as np
import pytest

from hpfem.opt import (
    Continuous,
    DesignSpace,
    FunctionEvaluator,
    build_surrogate,
    fit,
    sample,
)


def linear_problem():
    rng = np.random.default_rng(11)
    t = np.linspace(0.0, 2.0, 12)
    design = np.column_stack([np.ones_like(t), t, t**2])
    truth = np.array([0.4, -1.3, 0.7])
    sigma = 0.05 * (1.0 + t)
    y_meas = design @ truth + sigma * rng.standard_normal(len(t))

    def model(c, jacobian=False):
        return design @ c, design

    space = DesignSpace([Continuous(n, -5.0, 5.0) for n in ("c0", "c1", "c2")])
    return FunctionEvaluator(model, space, len(t), as_array=True), y_meas, sigma


def check_gaussian(post, result, mean_tol=0.1, std_tol=0.1, corr_tol=0.1):
    for row in post.compare():
        assert abs(row["shift"]) < mean_tol, row
        assert abs(row["ratio"] - 1.0) < std_tol, row
    assert np.abs(post.correlation - result.correlation).max() < corr_tol


def test_linear_model_posterior_equals_the_laplace_approximation():
    pytest.importorskip("emcee")
    evaluator, y_meas, sigma = linear_problem()
    result = fit(evaluator, y_meas, sigma)
    direct = sample(result, surrogate=False, walkers=32, steps=2500, seed=1)
    check_gaussian(direct, result)
    assert 0.15 < direct.acceptance < 0.7
    assert direct.samples.shape[1] == 3 and direct.noise_scale == 1.0
    assert "posterior" in direct.summary()
    surrogate = build_surrogate(result, seed=2)
    assert surrogate.validation < 0.05  # a quadratic model with gradients: nearly exact
    on_surrogate = sample(result, surrogate=surrogate, walkers=32, steps=2600, seed=3)
    check_gaussian(on_surrogate, result)
    # a Gaussian prior with the Laplace width on each parameter: for the linear model the
    # posterior covariance is exactly (C^-1 + D)^-1 with D = diag(1 / std^2)
    prior = {n: (float(x), float(s)) for n, x, s in zip(result.names, result.x, result.std,
                                                        strict=True)}  # fmt: skip
    narrowed = sample(result, surrogate=surrogate, prior=prior, walkers=32, steps=2600, seed=4)
    exact = np.linalg.inv(np.linalg.inv(result.covariance) + np.diag(1.0 / result.std**2))
    np.testing.assert_allclose(narrowed.std, np.sqrt(np.diag(exact)), rtol=0.1)
    assert np.all(narrowed.std < result.std)


def test_skewed_posterior_matches_the_grid_where_laplace_does_not():
    pytest.importorskip("emcee")
    t = np.array([0.5, 1.0, 1.5])
    sigma = 0.08
    k_true = 1.2
    y_meas = np.exp(-k_true * t) + sigma * np.array([0.9, -1.1, 0.4])

    def model(p, jacobian=False):
        y = np.exp(-p[0] * t)
        return y, (-t * y)[:, None]

    space = DesignSpace([Continuous("k", 0.01, 8.0)])
    evaluator = FunctionEvaluator(model, space, len(t), as_array=True)
    result = fit(evaluator, y_meas, sigma)
    post = sample(result, surrogate=False, walkers=16, steps=4000, seed=5)
    # brute-force posterior on a fine grid (flat prior on the bounds)
    k = np.linspace(0.01, 8.0, 40001)
    chi2 = (((np.exp(-np.outer(k, t)) - y_meas) / sigma) ** 2).sum(axis=1)
    density = np.exp(-0.5 * (chi2 - chi2.min()))
    cdf = np.cumsum(density)
    cdf /= cdf[-1]
    grid_std = math.sqrt(
        np.sum(density * k**2) / density.sum() - (np.sum(density * k) / density.sum()) ** 2
    )
    # Monte-Carlo error of a quantile: about 0.04 posterior standard deviations for the
    # ~1500 effective samples of this chain (16 walkers x 3000 steps / tau ~ 30); 0.15 is ~4 of it
    for q in (0.16, 0.5, 0.84):
        grid_q = float(np.interp(q, cdf, k))
        assert abs(post.quantiles[q][0] - grid_q) < 0.15 * grid_std, (q, post.quantiles[q], grid_q)
    # the posterior is skewed: its mean lies well away from the Laplace mode
    assert post.compare()[0]["shift"] > 0.15


def test_surrogate_reproduces_the_direct_posterior_of_a_nonlinear_model():
    pytest.importorskip("emcee")
    t = np.linspace(0.0, 3.0, 10)
    sigma = 0.02

    def model(p, jacobian=False):
        a, k = p
        y = a * np.exp(-k * t)
        return y, np.column_stack([np.exp(-k * t), -a * t * np.exp(-k * t)])

    rng = np.random.default_rng(6)
    y_meas = model([1.5, 0.8])[0] + sigma * rng.standard_normal(len(t))
    space = DesignSpace([Continuous("a", 0.1, 5.0), Continuous("k", 0.05, 3.0)])
    evaluator = FunctionEvaluator(model, space, len(t), as_array=True)
    result = fit(evaluator, y_meas, sigma)
    surrogate = build_surrogate(result, points=10, seed=7)
    assert surrogate.points == 11 and surrogate.validation < 0.1
    direct = sample(result, surrogate=False, walkers=32, steps=3000, seed=8)
    on_surrogate = sample(result, surrogate=surrogate, walkers=32, steps=2600, seed=9)
    for a, b in zip(direct.compare(), on_surrogate.compare(), strict=True):
        assert abs(a["mean"] - b["mean"]) < 0.1 * a["laplace_std"]
        assert abs(b["std"] / a["std"] - 1.0) < 0.1


def test_argument_checks_and_the_missing_extra(monkeypatch):
    evaluator, y_meas, sigma = linear_problem()
    result = fit(evaluator, y_meas, sigma)
    monkeypatch.setitem(sys.modules, "emcee", None)  # import emcee -> ImportError
    with pytest.raises(ImportError, match="opt-mcmc"):
        sample(result, surrogate=False)
    monkeypatch.undo()
    pytest.importorskip("emcee")
    with pytest.raises(ValueError):
        sample(result, surrogate=False, prior={"c7": (0.0, 1.0)})
    with pytest.raises(ValueError):
        sample(result, surrogate=False, steps=10, burn=10)

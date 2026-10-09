"""Uncertainty propagation and Sobol' indices (M16 S7, ADR-0012 §6): the Saltelli / Jansen
estimators reproduce the analytic indices of the Ishigami function directly and through a
gradient-enhanced global surrogate built with active learning; on a linear model the
linearised propagation, Monte Carlo and the Sobol' indices agree exactly with the closed form;
on a nonlinear model Monte Carlo on the surrogate matches Monte Carlo on the function and the
linearised standard deviation is the small-tolerance limit; argument checks."""

import math

import numpy as np
import pytest

from hpfem.opt import (
    Continuous,
    DesignSpace,
    FunctionEvaluator,
    Normal,
    Uniform,
    build_global_surrogate,
    linear_propagation,
    monte_carlo,
    sobol_indices,
)

A, B = 7.0, 0.1  # Ishigami constants
PI = math.pi
V1 = 0.5 * (1 + B * PI**4 / 5) ** 2
V2 = A**2 / 8
V13 = B**2 * PI**8 * (1 / 18 - 1 / 50)
V = V1 + V2 + V13
ISHIGAMI_FIRST = np.array([V1 / V, V2 / V, 0.0])
ISHIGAMI_TOTAL = np.array([(V1 + V13) / V, V2 / V, V13 / V])
ISHIGAMI_INPUTS = {f"x{i}": Uniform(-PI, PI) for i in (1, 2, 3)}


def ishigami(x):
    x = np.atleast_2d(x)
    return np.sin(x[:, 0]) + A * np.sin(x[:, 1]) ** 2 + B * x[:, 2] ** 4 * np.sin(x[:, 0])


def ishigami_evaluator():
    def model(x, jacobian=False):
        x1, x2, x3 = x
        value = math.sin(x1) + A * math.sin(x2) ** 2 + B * x3**4 * math.sin(x1)
        grad = [
            math.cos(x1) * (1 + B * x3**4),
            2 * A * math.sin(x2) * math.cos(x2),
            4 * B * x3**3 * math.sin(x1),
        ]
        return [value], [grad]

    space = DesignSpace([Continuous(f"x{i}", -PI, PI) for i in (1, 2, 3)])
    return FunctionEvaluator(model, space, 1, as_array=True)


def test_sobol_indices_of_the_ishigami_function():
    result = sobol_indices(ishigami, ISHIGAMI_INPUTS, samples=2**14, seed=1)
    assert result.samples == 2**14
    np.testing.assert_allclose(result.first[0], ISHIGAMI_FIRST, atol=0.02)
    np.testing.assert_allclose(result.total[0], ISHIGAMI_TOTAL, atol=0.02)
    assert np.all(result.first_conf < 0.05) and np.all(result.total_conf < 0.05)
    assert result.variance[0] == pytest.approx(V, rel=0.02)
    assert "x3" in result.table()


def test_sobol_indices_through_the_actively_learned_surrogate():
    surrogate = build_global_surrogate(ishigami_evaluator(), ISHIGAMI_INPUTS, points=40,
                                       active=30, seed=2)  # fmt: skip
    # the largest relative std of a candidate pool is set by a few corners of the cube (0.32
    # here, mean 0.13); the indices are already within 0.006 of the analytic values
    assert surrogate.points == 70 and surrogate.max_relative_std < 0.5
    result = sobol_indices(surrogate, ISHIGAMI_INPUTS, samples=2**13, seed=3)
    np.testing.assert_allclose(result.first[0], ISHIGAMI_FIRST, atol=0.02)
    np.testing.assert_allclose(result.total[0], ISHIGAMI_TOTAL, atol=0.02)


def test_linear_model_propagation_monte_carlo_and_sobol_agree_with_the_closed_form():
    g = np.array([[2.0, -1.0, 0.5], [0.0, 3.0, 1.0]])
    offset = np.array([1.0, -2.0])

    def model(x, jacobian=False):
        return g @ x + offset, g

    space = DesignSpace([Continuous(n, -10.0, 10.0) for n in ("a", "b", "c")])
    evaluator = FunctionEvaluator(model, space, 2, as_array=True)
    inputs = {"a": Normal(0.5, 0.2), "b": Normal(-1.0, 0.1), "c": Uniform(0.0, 1.2)}
    var = np.array([0.04, 0.01, 0.12])
    lin = linear_propagation(evaluator, inputs)
    np.testing.assert_allclose(lin.values, g @ [0.5, -1.0, 0.6] + offset, rtol=1e-12)
    np.testing.assert_allclose(lin.covariance, (g * var) @ g.T, rtol=1e-12)
    np.testing.assert_allclose(lin.contributions.sum(axis=1), 1.0, rtol=1e-12)
    surrogate = build_global_surrogate(evaluator, inputs, seed=4)
    mc = monte_carlo(surrogate, inputs, samples=40000, seed=5)
    np.testing.assert_allclose(mc.mean, lin.values, atol=0.02 * lin.std.max())
    np.testing.assert_allclose(mc.std, lin.std, rtol=0.02)
    assert mc.clipped < 10 and np.all(mc.surrogate_std < 1e-3 * lin.std)
    # additive model: first-order = total = the linear contributions
    sobol = sobol_indices(surrogate, inputs, samples=2**13, seed=6)
    np.testing.assert_allclose(sobol.first, lin.contributions, atol=0.02)
    np.testing.assert_allclose(sobol.total, lin.contributions, atol=0.02)


def test_nonlinear_model_surrogate_monte_carlo_and_the_linear_limit():
    def model(x, jacobian=False):
        h, w = x
        y = [math.exp(-h) * math.cos(w), h * w**2]
        jac = [[-math.exp(-h) * math.cos(w), -math.exp(-h) * math.sin(w)], [w**2, 2 * h * w]]
        return y, jac

    space = DesignSpace([Continuous("h", 0.0, 3.0), Continuous("w", -1.0, 2.0)])
    evaluator = FunctionEvaluator(model, space, 2, as_array=True)
    inputs = {"h": Normal(1.0, 0.25), "w": Normal(0.7, 0.2)}

    def exact(x):
        return np.column_stack([np.exp(-x[:, 0]) * np.cos(x[:, 1]), x[:, 0] * x[:, 1] ** 2])

    surrogate = build_global_surrogate(evaluator, inputs, points=16, active=8, seed=7)
    on_surrogate = monte_carlo(surrogate, inputs, samples=40000, seed=8)
    direct = monte_carlo(exact, inputs, samples=40000, seed=8)
    np.testing.assert_allclose(on_surrogate.mean, direct.mean, rtol=0.01)
    np.testing.assert_allclose(on_surrogate.std, direct.std, rtol=0.02)
    for q in (0.16, 0.5, 0.84):
        np.testing.assert_allclose(on_surrogate.quantiles[q], direct.quantiles[q],
                                   atol=0.02 * direct.std.max())  # fmt: skip
    # the linearised std is the small-tolerance limit: within 2 % for tolerances ten times smaller
    small = {k: Normal(d.mean, d.std / 10) for k, d in inputs.items()}
    lin = linear_propagation(evaluator, small)
    np.testing.assert_allclose(lin.std, monte_carlo(exact, small, samples=40000, seed=9).std,
                               rtol=0.02)  # fmt: skip


def test_argument_checks():
    evaluator = ishigami_evaluator()
    with pytest.raises(ValueError):
        Normal(0.0, 0.0)
    with pytest.raises(ValueError):
        Uniform(1.0, 1.0)
    with pytest.raises(ValueError):
        linear_propagation(evaluator, {"x9": Normal(0.0, 1.0)})
    with pytest.raises(ValueError):
        linear_propagation(evaluator, {})
    surrogate = build_global_surrogate(evaluator, ISHIGAMI_INPUTS, points=8, seed=0)
    with pytest.raises(ValueError):
        monte_carlo(surrogate, {"x1": Uniform(-PI, PI)}, samples=10)

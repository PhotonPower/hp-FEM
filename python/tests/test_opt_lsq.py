"""Least-squares reconstruction and the Laplace approximation (M16 S3, ADR-0012 §6): an exact
linear model (fit and covariance against the closed form), the NIST StRD problem MGH17 from
both starts (certified parameters and standard deviations), observables of very different
magnitudes, bounds active at the solution, a non-identifiable parameter, the study
integration (store, replay, cancellation, resume), the failure policy and the remesh
restart."""

import math
import warnings

import numpy as np
import pytest

import hpfem
from hpfem.opt import (
    Continuous,
    DesignSpace,
    Evaluation,
    EvaluationFailed,
    FunctionEvaluator,
    IdentifiabilityWarning,
    Study,
    fit,
    laplace,
)

# --- NIST StRD MGH17 (https://www.itl.nist.gov/div898/strd/nls/data/mgh17.shtml) -------------
# y = b1 + b2 exp(-x b4) + b3 exp(-x b5), 33 observations, x = 0, 10, ..., 320.

MGH17_Y = np.array([
    8.440000e-01, 9.080000e-01, 9.320000e-01, 9.360000e-01, 9.250000e-01, 9.080000e-01,
    8.810000e-01, 8.500000e-01, 8.180000e-01, 7.840000e-01, 7.510000e-01, 7.180000e-01,
    6.850000e-01, 6.580000e-01, 6.280000e-01, 6.030000e-01, 5.800000e-01, 5.580000e-01,
    5.380000e-01, 5.220000e-01, 5.060000e-01, 4.900000e-01, 4.780000e-01, 4.670000e-01,
    4.570000e-01, 4.480000e-01, 4.380000e-01, 4.310000e-01, 4.240000e-01, 4.200000e-01,
    4.140000e-01, 4.110000e-01, 4.060000e-01,
])  # fmt: skip
MGH17_X = 10.0 * np.arange(33)
MGH17_B = np.array([3.7541005211e-01, 1.9358469127e00, -1.4646871366e00, 1.2867534640e-02,
                    2.2122699662e-02])  # fmt: skip
MGH17_SD = np.array([2.0723153551e-03, 2.2031669222e-01, 2.2175707739e-01, 4.4861358114e-04,
                     8.9471996575e-04])  # fmt: skip
MGH17_RSS = 5.4648946975e-05
MGH17_START1 = [50.0, 150.0, -100.0, 1.0, 2.0]
MGH17_START2 = [0.5, 1.5, -1.0, 0.01, 0.02]


def mgh17_model(b, jacobian=False):
    b1, b2, b3, b4, b5 = b
    e4, e5 = np.exp(-MGH17_X * b4), np.exp(-MGH17_X * b5)
    y = b1 + b2 * e4 + b3 * e5
    jac = np.column_stack([np.ones_like(MGH17_X), e4, e5, -MGH17_X * b2 * e4, -MGH17_X * b3 * e5])
    return y, jac


def mgh17_evaluator(bounds):
    space = DesignSpace([Continuous(f"b{i + 1}", lo, hi) for i, (lo, hi) in enumerate(bounds)])
    return FunctionEvaluator(mgh17_model, space, 33, as_array=True, name="MGH17")


@pytest.mark.parametrize("method", ["lm", "gn"])
def test_mgh17_from_start_2_matches_the_certified_values(method):
    evaluator = mgh17_evaluator([(-10, 10), (-10, 10), (-10, 10), (0, 1), (0, 1)])
    with warnings.catch_warnings():
        warnings.simplefilter("error", IdentifiabilityWarning)
        result = fit(evaluator, MGH17_Y, x0=MGH17_START2, method=method)
    assert result.success, result.message
    print(result.summary())
    np.testing.assert_allclose(result.x, MGH17_B, rtol=1e-6)
    np.testing.assert_allclose(result.std, MGH17_SD, rtol=1e-6)  # chi2_red-scaled Laplace
    assert result.chi2 == pytest.approx(MGH17_RSS, rel=1e-9)
    assert result.dof == 28 and result.chi2_red == pytest.approx(MGH17_RSS / 28, rel=1e-9)
    assert result.laplace.scaled and result.laplace.rank == 5 and not result.at_bounds
    assert result.errors["b4"] == result.std[3]
    np.testing.assert_allclose(np.diag(result.correlation), 1.0)
    assert result.correlation[1, 2] == pytest.approx(
        result.covariance[1, 2] / (result.std[1] * result.std[2])
    )


def test_mgh17_from_start_1():
    evaluator = mgh17_evaluator([(-100, 100), (-200, 200), (-200, 200), (0, 5), (0, 5)])
    with np.errstate(over="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("error", IdentifiabilityWarning)
        result = fit(evaluator, MGH17_Y, x0=MGH17_START1, max_iterations=2000)
    assert result.success, result.message
    np.testing.assert_allclose(result.x, MGH17_B, rtol=1e-6)
    np.testing.assert_allclose(result.std, MGH17_SD, rtol=1e-6)


def test_linear_model_fit_and_covariance_are_exact():
    rng = np.random.default_rng(7)
    t = np.linspace(0.0, 2.0, 12)
    design = np.column_stack([np.ones_like(t), t, t**2])
    truth = np.array([0.4, -1.3, 0.7])
    sigma = 0.01 * (1.0 + t)
    y_meas = design @ truth + sigma * rng.standard_normal(len(t))

    def model(c, jacobian=False):
        return design @ c, design

    space = DesignSpace([Continuous(n, -5.0, 5.0) for n in ("c0", "c1", "c2")])
    evaluator = FunctionEvaluator(model, space, len(t), as_array=True)
    weight = np.diag(1.0 / sigma**2)
    normal = design.T @ weight @ design
    c_exact = np.linalg.solve(normal, design.T @ weight @ y_meas)
    cov_exact = np.linalg.inv(normal)
    for method in ("lm", "gn"):
        result = fit(evaluator, y_meas, sigma, method=method)
        assert result.success, result.message
        np.testing.assert_allclose(result.x, c_exact, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(result.covariance, cov_exact, rtol=1e-9)  # sigma known
        assert not result.laplace.scaled and result.dof == len(t) - 3
        resid = result.residual / sigma
        assert result.chi2 == pytest.approx(resid @ resid, rel=1e-12)
    # unknown sigma: scaled by the reduced chi-square of the unweighted fit
    plain = fit(evaluator, y_meas)
    c_plain, rss = np.linalg.lstsq(design, y_meas, rcond=None)[:2]
    s2 = float(rss[0]) / (len(t) - 3)
    np.testing.assert_allclose(plain.x, c_plain, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(plain.covariance, s2 * np.linalg.inv(design.T @ design), rtol=1e-9)
    # the standalone Laplace approximation gives the same numbers
    lap = laplace(design, design @ c_exact - y_meas, sigma, names=["c0", "c1", "c2"])
    np.testing.assert_allclose(lap.covariance, cov_exact, rtol=1e-9)
    assert lap.errors["c1"] == pytest.approx(math.sqrt(cov_exact[1, 1]), rel=1e-9)
    assert lap.identifiable and lap.condition < 1e3


def test_observables_of_very_different_magnitudes():
    # two kinds of observables, about 1 and about 1e6: W balances them
    t = np.linspace(0.0, 1.0, 8)

    def model(p, jacobian=False):
        a, b = p
        y = np.concatenate([a + b * t, 1e6 * (a * b + t)])
        upper = np.column_stack([np.ones_like(t), t])
        lower = 1e6 * np.column_stack([np.full_like(t, b), np.full_like(t, a)])
        return y, np.vstack([upper, lower])

    space = DesignSpace([Continuous("a", 0.0, 2.0), Continuous("b", 0.0, 2.0)])
    evaluator = FunctionEvaluator(model, space, 16, as_array=True)
    truth = np.array([0.8, 1.25])
    y_true, jac_true = model(truth)
    noise = np.where(np.arange(16) % 2 == 0, 1.0, -1.0)
    sigma = np.concatenate([np.full(8, 1e-3), np.full(8, 1e3)])
    y_meas = y_true + 0.5 * sigma * noise
    result = fit(evaluator, y_meas, sigma)
    w = 1.0 / sigma
    expected = np.linalg.inv((w[:, None] * jac_true).T @ (w[:, None] * jac_true))
    assert result.success and np.allclose(result.x, truth, atol=5e-3)
    np.testing.assert_allclose(result.covariance, expected, rtol=1e-2)
    # relative errors of unknown size: sigma = |y_meas|, covariance scaled
    relative = fit(evaluator, y_meas, "relative")
    assert relative.laplace.scaled and np.allclose(relative.x, truth, atol=5e-3)
    np.testing.assert_array_equal(relative.sigma, np.abs(y_meas))
    # only the second group: it determines the product a b alone
    with pytest.warns(IdentifiabilityWarning, match="rank 1 < 2"):
        partial = fit(evaluator, y_meas[8:], observables=list(range(8, 16)), x0=[0.8, 1.0])
    assert partial.observables == [f"y{i}" for i in range(8, 16)]
    assert set(partial.laplace.non_identifiable) == {"a", "b"}


def test_bounds_active_at_the_solution():
    t = np.linspace(0.0, 1.0, 10)
    y_meas = 0.3 + 2.0 * t + 0.01 * np.sin(7 * t)  # unconstrained slope about 2

    def model(p, jacobian=False):
        return p[0] + p[1] * t, np.column_stack([np.ones_like(t), t])

    space = DesignSpace([Continuous("a", -1.0, 1.0), Continuous("b", 0.0, 1.5)])
    evaluator = FunctionEvaluator(model, space, len(t), as_array=True)
    for method in ("lm", "gn"):
        with pytest.warns(IdentifiabilityWarning, match="at a bound"):
            result = fit(evaluator, y_meas, x0=[0.0, 0.5], method=method)
        assert result.success, result.message
        assert result.params["b"] == 1.5 and result.at_bounds == ["b"]
        # the free parameter is optimal for the fixed slope
        assert result.params["a"] == pytest.approx(np.mean(y_meas - 1.5 * t), abs=1e-10)


def test_non_identifiable_parameter_gives_the_rank_warning():
    t = np.linspace(0.0, 1.0, 6)

    def model(p, jacobian=False):  # only a + b enters
        a, b, c = p
        return (a + b) * t + c, np.column_stack([t, t, np.ones_like(t)])

    space = DesignSpace([Continuous(n, -3.0, 3.0) for n in ("a", "b", "c")])
    evaluator = FunctionEvaluator(model, space, len(t), as_array=True)
    y_meas = 1.2 * t - 0.4 + 1e-3 * np.cos(5 * t)
    with pytest.warns(IdentifiabilityWarning, match="rank 2 < 3"):
        result = fit(evaluator, y_meas, x0=[0.1, 0.2, 0.0])
    assert result.success and result.laplace.rank == 2
    assert result.laplace.non_identifiable == ["a", "b"]
    assert np.isinf(result.std[:2]).all() and np.isfinite(result.std[2])
    assert result.params["a"] + result.params["b"] == pytest.approx(1.2, abs=2e-3)
    assert np.isnan(result.correlation[0, 2]) and result.correlation[2, 2] == 1.0
    # standalone: a zero column, an ill-conditioned pair
    with pytest.warns(IdentifiabilityWarning, match="not identifiable"):
        lap = laplace(np.column_stack([t, 0 * t]), 0.01 * t, names=["x", "y"])
    assert lap.non_identifiable == ["y"] and np.isfinite(lap.std[0])
    with pytest.warns(IdentifiabilityWarning, match="ill-conditioned"):
        laplace(np.column_stack([t, t + 1e-9 * t**2]), 0.01 * t, rcond=1e-14)


class Exponential:
    """y = a exp(-k x) as an evaluator that counts its calls, can fail for large k and can
    remesh when a point is far from its reference."""

    x = np.linspace(0.0, 3.0, 15)
    parameters = [Continuous("a", 0.1, 10.0), Continuous("k", 0.01, 10.0, log=True)]
    observables = [f"y{i}" for i in range(15)]

    def __init__(self, fail_above=None, remesh_distance=None):
        self.calls = 0
        self.fail_above = fail_above
        self.remesh_distance = remesh_distance
        self.reference = None
        self.mesh_id = 0

    def __call__(self, params, *, jacobian=False, fidelity=None, cancel=None):
        self.calls += 1
        a, k = params["a"], params["k"]
        if self.fail_above is not None and k > self.fail_above:
            raise RuntimeError(f"no convergence at k = {k}")
        meta = {}
        point = np.array([a, math.log(k)])
        if self.remesh_distance is not None:
            if self.reference is None:
                self.reference = point
            elif np.max(np.abs(point - self.reference)) > self.remesh_distance:
                self.reference, self.mesh_id = point, self.mesh_id + 1
                meta["remesh"] = {"mesh_id": self.mesh_id, "reason": "quality"}
        e = np.exp(-k * self.x)
        jac = np.column_stack([e, -a * self.x * e])
        return Evaluation(params, a * e, jac if jacobian else None, mesh_id=self.mesh_id,
                          meta=meta)  # fmt: skip


EXP_TRUE = (2.5, 0.8)
EXP_DATA = EXP_TRUE[0] * np.exp(-EXP_TRUE[1] * Exponential.x) * (1 + 0.01 * np.sin(np.arange(15)))


def test_fit_drives_a_study_with_store_replay_cancellation_and_resume(tmp_path):
    path = tmp_path / "fit.study.jsonl"
    reference = fit(Exponential(), EXP_DATA, 0.02, x0={"a": 1.0, "k": 3.0})
    assert reference.success and np.allclose(reference.x, EXP_TRUE, rtol=0.05)
    # log-scaled k: the covariance is in SI (not in log k)
    assert reference.covariance.shape == (2, 2) and reference.std[1] < 0.1
    # cancelled after 4 evaluations: the store is consistent
    evaluator = Exponential()
    study = Study(evaluator, path, cancel=lambda: evaluator.calls >= 4)
    with pytest.raises(hpfem.Cancelled):
        fit(study, EXP_DATA, 0.02, x0={"a": 1.0, "k": 3.0})
    stored = Study.load(path)
    assert len(stored) == 4 and stored.state["state"]["status"] == "cancelled"
    assert stored.state["method"] == "levenberg-marquardt"
    # rerun with the same start: replays the 4 points, then continues identically
    again = Exponential()
    result = fit(Study(again, path), EXP_DATA, 0.02, x0={"a": 1.0, "k": 3.0})
    assert again.calls == reference.new_evaluations - 4 and result.cache_hits == 4
    np.testing.assert_array_equal(result.x, reference.x)
    np.testing.assert_array_equal(result.covariance, reference.covariance)
    # without x0 the fit restarts from the best stored point and converges at once
    restart = Exponential()
    quick = fit(Study(restart, path), EXP_DATA, 0.02)
    assert restart.calls <= 1 and np.allclose(quick.x, reference.x, rtol=1e-8)
    assert quick.study.state["state"]["data"] == result.study.state["state"]["data"]
    history = quick.history()
    assert history.jacobian is not None and len(history) == len(quick.study)


def test_failed_trial_points_are_rejected_steps():
    # the start is far away; trial steps into k > 2 fail and are rejected
    evaluator = Exponential(fail_above=2.0)
    result = fit(evaluator, EXP_DATA, 0.02, x0={"a": 0.2, "k": 0.02})
    assert result.success, result.message
    assert np.allclose(result.x, EXP_TRUE, rtol=0.05)
    assert result.failures == len(result.study.failed) > 0
    # too many failures stop the fit; a failed start point raises
    stopped = fit(Exponential(fail_above=2.0), EXP_DATA, 0.02, x0={"a": 0.2, "k": 0.02},
                  max_failures=0)  # fmt: skip
    assert not stopped.success and "failed" in stopped.message
    with pytest.raises(EvaluationFailed, match="start point"):
        fit(Exponential(fail_above=0.5), EXP_DATA, 0.02, x0={"a": 1.0, "k": 3.0})


def test_remesh_restarts_the_fit():
    evaluator = Exponential(remesh_distance=0.5)
    result = fit(evaluator, EXP_DATA, 0.02, x0={"a": 9.0, "k": 0.05})
    assert result.success, result.message
    assert result.restarts >= 1 and result.restarts == len(result.study.remeshes)
    assert np.allclose(result.x, EXP_TRUE, rtol=0.05)
    reference = fit(Exponential(), EXP_DATA, 0.02, x0={"a": 9.0, "k": 0.05})
    np.testing.assert_allclose(result.x, reference.x, rtol=1e-8)
    np.testing.assert_allclose(result.std, reference.std, rtol=1e-6)


def test_fixed_parameters_and_input_checks():
    evaluator = Exponential()
    result = fit(evaluator, EXP_DATA, 0.02, fixed={"a": 2.5}, x0=[1.0])
    assert result.names == ["k"] and result.params["a"] == 2.5
    assert result.covariance.shape == (1, 1) and result.success
    with pytest.raises(ValueError, match="measured values"):
        fit(evaluator, EXP_DATA[:3], 0.02)
    with pytest.raises(ValueError, match="sigma"):
        fit(evaluator, EXP_DATA, -1.0)
    with pytest.raises(ValueError, match="method"):
        fit(evaluator, EXP_DATA, method="newton")

"""Bayesian optimisation in hpfem.opt (M16 S5, ADR-0012): the acquisition formulas, Branin
with expected improvement and lower confidence bound, the gradient-enhanced variant on
Hartmann-3, known and unknown constraints, cancellation and resume from the store, the failure
policy, remeshes, the DWR estimate (ignored), the Pareto front and the optional BoTorch
path."""

import math
import sys

import numpy as np
import pytest

import hpfem
from hpfem.opt import (
    Categorical,
    Continuous,
    DesignSpace,
    Evaluation,
    FunctionEvaluator,
    LinearConstraint,
    OutcomeConstraint,
    Study,
    bayesian_optimize,
    expected_improvement,
    log_expected_improvement,
    log_probability_of_feasibility,
    lower_confidence_bound,
    non_dominated,
    pareto_front,
)


def branin(p):
    x, y = p["x"], p["y"]
    b, c, t = 5.1 / (4 * math.pi**2), 5 / math.pi, 1 / (8 * math.pi)
    return (y - b * x * x + c * x - 6) ** 2 + 10 * (1 - t) * math.cos(x) + 10


BRANIN_SPACE = DesignSpace([Continuous("x", -5.0, 10.0), Continuous("y", 0.0, 15.0)])
BRANIN_MINIMA = [(-math.pi, 12.275), (math.pi, 2.275), (9.42478, 2.475)]
BRANIN_MIN = 0.397887357729738

# Hartmann-3 on [0, 1]^3 with its gradient
_A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
_P = 1e-4 * np.array(
    [[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]]
)
_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
HARTMANN3_MIN = -3.86278


def hartmann3(p, jacobian=False):
    x = np.array([p["x0"], p["x1"], p["x2"]])
    e = np.exp(-np.sum(_A * (x - _P) ** 2, axis=1))
    f = -float(np.sum(_ALPHA * e))
    if not jacobian:
        return f
    return f, np.sum((_ALPHA * e)[:, None] * 2 * _A * (x - _P), axis=0)


def counted(function):
    calls = []

    def wrapped(p):
        calls.append(dict(p))
        return function(p)

    wrapped.calls = calls
    return wrapped


# --- acquisition formulas -----------------------------------------------------------------------


def _phi(z):
    return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def _cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def test_expected_improvement_and_lcb_against_hand_computed_values():
    # mean 1, std 2, best 0.5: z = -0.25, EI = 2 (z Φ(z) + φ(z))
    z = -0.25
    hand = 2.0 * (z * _cdf(z) + _phi(z))
    assert expected_improvement(1.0, 2.0, 0.5) == pytest.approx(hand, rel=1e-12)
    assert hand == pytest.approx(0.5726894, rel=1e-6)
    # margin xi shifts the incumbent
    assert expected_improvement(1.0, 2.0, 0.7, xi=0.2) == pytest.approx(hand, rel=1e-12)
    # far above the incumbent EI underflows, its logarithm follows the asymptote
    # h(z) = φ(z)/z² (1 − 3/z² + 15/z⁴ − ...) for z → −∞
    for z, rel in ((-12.0, 1e-7), (-40.0, 1e-12), (-1e3, 1e-12), (-1e9, 1e-12)):
        x = -z
        series = -0.5 * x * x - 0.5 * math.log(2 * math.pi) - 2 * math.log(x)
        series += math.log1p(-3 / x**2 + 15 / x**4 - 105 / x**6 + 945 / x**8 - 10395 / x**10)
        assert log_expected_improvement(x, 1.0, 0.0) == pytest.approx(series, rel=rel)
    # in between, the direct formula (one digit lost to cancellation at z = -3)
    for z in (-3.0, -0.5, 0.0, 2.0):
        direct = math.log(z * _cdf(z) + _phi(z))
        assert log_expected_improvement(-z, 1.0, 0.0) == pytest.approx(direct, rel=1e-12)
    assert expected_improvement(40.0, 1.0, 0.0) == 0.0  # underflow, but log EI is finite
    # continuity across the switch at z = -1
    a, b = log_expected_improvement([1.0 - 1e-12, 1.0 + 1e-12], 1.0, 0.0)
    assert a == pytest.approx(b, abs=1e-10)
    # gradient against finite differences
    for mean, std in ((0.3, 0.7), (3.0, 0.2), (-1.0, 0.5)):
        _, dm, ds = log_expected_improvement(mean, std, 0.4, gradient=True)
        h = 1e-6
        fm = (
            log_expected_improvement(mean + h, std, 0.4)
            - log_expected_improvement(mean - h, std, 0.4)
        ) / (2 * h)
        fs = (
            log_expected_improvement(mean, std + h, 0.4)
            - log_expected_improvement(mean, std - h, 0.4)
        ) / (2 * h)
        assert dm == pytest.approx(fm, rel=1e-6) and ds == pytest.approx(fs, rel=1e-6)
    np.testing.assert_allclose(lower_confidence_bound([1.0, 2.0], [0.5, 0.1], 2.0), [0.0, 1.8])


def test_probability_of_feasibility_against_hand_computed_values():
    m, s = 0.2, 0.5
    assert math.exp(log_probability_of_feasibility(m, s, upper=1.0)) == pytest.approx(_cdf(1.6))
    assert math.exp(log_probability_of_feasibility(m, s, lower=0.0)) == pytest.approx(_cdf(0.4))
    two = _cdf(1.6) - _cdf(-0.4)
    assert math.exp(log_probability_of_feasibility(m, s, -0.0, 1.0)) == pytest.approx(two)
    # far tails stay finite: P(g <= -40 σ) and P(30 σ <= g <= 31 σ)
    assert log_probability_of_feasibility(0.0, 1.0, upper=-40.0) == pytest.approx(
        -804.608, abs=1e-2
    )
    assert np.isfinite(log_probability_of_feasibility(0.0, 1.0, 30.0, 31.0))
    for lower, upper in ((-math.inf, 1.0), (0.0, math.inf), (0.0, 1.0), (2.0, 3.0), (-3.0, -2.0)):
        _, dm, ds = log_probability_of_feasibility(m, s, lower, upper, gradient=True)
        h = 1e-6

        def f(mm, ss, lower=lower, upper=upper):
            return log_probability_of_feasibility(mm, ss, lower, upper)

        assert dm == pytest.approx((f(m + h, s) - f(m - h, s)) / (2 * h), rel=1e-5)
        assert ds == pytest.approx((f(m, s + h) - f(m, s - h)) / (2 * h), rel=1e-5)


# --- Branin, Hartmann-3 --------------------------------------------------------------------------


def test_branin_with_expected_improvement_is_seeded():
    f = counted(branin)
    result = bayesian_optimize(f, space=BRANIN_SPACE, max_evaluations=40, seed=0)
    assert result.value == pytest.approx(BRANIN_MIN, abs=1e-3)
    assert min(np.hypot(result.x[0] - a, result.x[1] - b) for a, b in BRANIN_MINIMA) < 0.05
    assert result.method == "bayes-ei" and result.success and result.feasible
    assert result.evaluations == result.new_evaluations == len(f.calls) == 40
    assert result.iterations == 40 - 6  # 2 (d + 1) initial points
    assert len(result.acquisition) == result.iterations and result.gp.n == 39
    state = result.study.state
    assert state["method"] == "bayes-ei" and state["state"]["status"] == "done"
    assert state["state"]["best"]["value"] == result.value
    # no point is evaluated twice
    keys = {tuple(BRANIN_SPACE.key(p)) for p in f.calls}
    assert len(keys) == len(f.calls)
    # the same seed proposes the same points
    again = counted(branin)
    bayesian_optimize(again, space=BRANIN_SPACE, max_evaluations=12, seed=0)
    assert again.calls == f.calls[:12]


def test_branin_with_lower_confidence_bound():
    result = bayesian_optimize(
        branin, space=BRANIN_SPACE, acquisition="lcb", kappa=2.0, max_evaluations=40, seed=0
    )
    assert result.value == pytest.approx(BRANIN_MIN, abs=1e-3)
    assert result.method == "bayes-lcb"
    assert np.all(np.isfinite(result.acquisition))


def _first_within(result, tol):
    best = np.minimum.accumulate([e.values[0] for e in result.study.evaluations])
    return next((i + 1 for i, v in enumerate(best) if v - HARTMANN3_MIN < tol), None)


def test_gradient_enhanced_bo_needs_fewer_evaluations_on_hartmann3():
    space = DesignSpace([Continuous(f"x{i}", 0.0, 1.0) for i in range(3)])
    plain = bayesian_optimize(
        FunctionEvaluator(hartmann3, space), max_evaluations=15, seed=0, n_initial=8
    )
    enhanced = bayesian_optimize(
        FunctionEvaluator(hartmann3, space),
        max_evaluations=15,
        seed=0,
        n_initial=4,
        use_gradients=True,
    )
    # evaluations with the Jacobian; the surrogate has n (d + 1) observations
    assert all(e.jacobian is not None for e in enhanced.study.evaluations)
    assert enhanced.gp.num_observations == 4 * enhanced.gp.n
    assert enhanced.value - HARTMANN3_MIN < 1e-3
    assert plain.value - HARTMANN3_MIN > 1e-3  # the plain variant is not there yet
    assert _first_within(enhanced, 1e-2) < (_first_within(plain, 1e-2) or 16)


# --- constraints ---------------------------------------------------------------------------------


def test_known_linear_and_unknown_outcome_constraints():
    space = DesignSpace(
        [Continuous("x", -2.0, 2.0), Continuous("y", -2.0, 2.0)],
        [LinearConstraint({"x": 1.0, "y": 1.0}, upper=1.0)],
    )

    def model(p):  # objective and an "expensive" constraint observable g = x
        return [(p["x"] - 1.0) ** 2 + (p["y"] - 1.0) ** 2, p["x"]]

    f = FunctionEvaluator(model, space, ["f", "g"])
    result = bayesian_optimize(
        f, "f", constraints=[OutcomeConstraint("g", upper=0.3)], max_evaluations=20, seed=1
    )
    # the optimum on x = 0.3, x + y = 1
    assert result.feasible and result.success
    assert result.params["x"] == pytest.approx(0.3, abs=1e-3)
    assert result.params["y"] == pytest.approx(0.7, abs=1e-3)
    assert result.value == pytest.approx(0.58, abs=1e-3)
    assert result.evaluation.values[1] <= 0.3
    assert all(space.feasible(e.params) for e in result.study.evaluations)
    assert len(result.constraint_gps) == 1
    with pytest.raises(ValueError, match="acquisition='ei'"):
        bayesian_optimize(f, "f", acquisition="lcb", constraints=[OutcomeConstraint("g", upper=0)])
    with pytest.raises(ValueError, match="finite"):
        OutcomeConstraint("g")


def test_maximize_fixed_categorical_log_scale_and_stopping_rules():
    space = DesignSpace([Continuous("k", 1e-2, 1e2, log=True), Categorical("mode", ("a", "b"))])

    def f(p):
        shift = 0.0 if p["mode"] == "a" else 0.5
        return -((math.log10(p["k"]) - shift) ** 2)

    result = bayesian_optimize(
        f, space=space, maximize=True, fixed={"mode": "b"}, max_evaluations=15, seed=2
    )
    assert result.params["mode"] == "b" and result.names == ["k"]
    assert math.log10(result.params["k"]) == pytest.approx(0.5, abs=2e-2)
    assert result.value == pytest.approx(0.0, abs=1e-3)
    with pytest.raises(ValueError, match="categorical"):
        bayesian_optimize(f, space=space)
    # tol on the predicted gain and patience end the run before the budget
    tol = bayesian_optimize(
        f, space=space, fixed={"mode": "a"}, maximize=True, max_evaluations=40, tol=1e-6, seed=2
    )
    assert tol.evaluations < 40 and "tol" in tol.message and tol.success
    patient = bayesian_optimize(
        f, space=space, fixed={"mode": "a"}, maximize=True, max_evaluations=40, patience=3, seed=2
    )
    assert patient.evaluations < 40 and "no improvement" in patient.message
    seen = []
    stopped = bayesian_optimize(
        f,
        space=space,
        fixed={"mode": "a"},
        maximize=True,
        max_evaluations=40,
        seed=2,
        callback=lambda info: seen.append(info) or len(seen) >= 2,
    )
    assert stopped.iterations == 2 and not stopped.success and "callback" in stopped.message
    assert {"iteration", "params", "value", "best_value", "gain"} <= set(seen[-1])
    with pytest.raises(ValueError, match="exceeds"):
        bayesian_optimize(f, space=space, fixed={"mode": "a"}, max_evaluations=3)


# --- store: cancel and resume --------------------------------------------------------------------


@pytest.mark.parametrize("cancel_after", [3, 11])  # in the initial design / in the loop
def test_cancellation_and_resume_continue_the_run(tmp_path, cancel_after):
    reference = counted(branin)
    expected = bayesian_optimize(reference, space=BRANIN_SPACE, max_evaluations=16, seed=5)
    path = tmp_path / "bo.study.jsonl"
    f = counted(branin)
    study = Study(f, path, BRANIN_SPACE, cancel=lambda: len(f.calls) >= cancel_after)
    with pytest.raises(hpfem.Cancelled):
        bayesian_optimize(study, max_evaluations=16, seed=5)
    stored = Study.load(path)
    assert len(stored) == cancel_after and stored.state["state"]["status"] == "cancelled"
    assert stored.state["method"] == "bayes-ei"
    assert len(Study(branin, path, BRANIN_SPACE).open_proposals()) >= 1
    # the rerun on the store resumes: the open proposal first, then the remaining iterations
    g = counted(branin)
    resumed = bayesian_optimize(Study(g, path, BRANIN_SPACE), max_evaluations=16, seed=99)
    assert f.calls + g.calls == reference.calls  # the points of the uninterrupted run
    assert resumed.params == expected.params and resumed.value == expected.value
    assert resumed.evaluations == 16 and resumed.iterations == expected.iterations
    # a finished run is not resumed: a new run starts, its design comes from the cache
    h = counted(branin)
    bayesian_optimize(Study(h, path, BRANIN_SPACE), max_evaluations=7, seed=5)
    assert len(h.calls) == 1


# --- failures, remeshes, DWR estimate ------------------------------------------------------------


def test_failed_evaluations_are_avoided_and_counted():
    def flaky(p):
        if p["x"] > 5.0:  # the "solver" fails in a region containing a minimum
            raise RuntimeError("solver diverged")
        return branin(p)

    f = counted(flaky)
    result = bayesian_optimize(f, space=BRANIN_SPACE, max_evaluations=40, seed=0)
    assert result.failures == len(result.study.failed) > 0
    assert result.params["x"] <= 5.0 and result.value == pytest.approx(BRANIN_MIN, abs=1e-2)
    keys = [tuple(BRANIN_SPACE.key(p)) for p in f.calls]
    assert len(set(keys)) == len(keys)  # a failed point is never proposed again
    # too many failures stop the run with the best point so far
    stopped = bayesian_optimize(
        flaky, space=BRANIN_SPACE, max_evaluations=40, seed=0, max_failures=1
    )
    assert not stopped.success and "failed" in stopped.message and stopped.failures == 2
    assert stopped.study.state["state"]["status"] == "stopped"


class Remeshing:
    """Branin whose 'reference mesh' is rebuilt when a point is far from it (ADR-0012 §3), with
    a small jump of the value (the discretisation error) per mesh."""

    parameters = [Continuous("x", -5.0, 10.0), Continuous("y", 0.0, 15.0)]
    observables = ["f"]

    def __init__(self):
        self.reference = np.array([2.5, 7.5])
        self.mesh_id = 0

    def __call__(self, params, *, jacobian=False, fidelity=None, cancel=None):
        x = np.array([params["x"], params["y"]])
        meta = {}
        if np.max(np.abs(x - self.reference)) > 6.0:
            self.reference, self.mesh_id = x, self.mesh_id + 1
            meta["remesh"] = {"mesh_id": self.mesh_id, "reason": "quality guard"}
        value = branin(params) + 1e-4 * math.sin(self.mesh_id)
        return Evaluation(
            params, [value], mesh_id=self.mesh_id, meta=meta, error=[1e3]
        )  # a (huge) DWR estimate


def test_remeshes_are_counted_and_the_dwr_estimate_is_not_noise():
    result = bayesian_optimize(Remeshing(), max_evaluations=30, seed=3)
    assert result.remeshes == len(result.study.remeshes) >= 1
    assert result.study.state["state"]["remeshes"] == result.remeshes
    assert result.value == pytest.approx(BRANIN_MIN, abs=1e-2)
    # Evaluation.error is ignored: with and without it the run proposes the same points
    with_error = FunctionEvaluator(lambda p: {"value": branin(p), "error": [10.0]}, BRANIN_SPACE)
    plain = FunctionEvaluator(branin, BRANIN_SPACE)
    a = bayesian_optimize(with_error, max_evaluations=12, seed=4)
    b = bayesian_optimize(plain, max_evaluations=12, seed=4)
    assert [e.params for e in a.study.evaluations] == [e.params for e in b.study.evaluations]
    assert all(e.error is not None for e in a.study.evaluations)
    np.testing.assert_array_equal(a.gp.theta, b.gp.theta)


# --- Pareto front --------------------------------------------------------------------------------


def test_pareto_front_of_a_study():
    y = np.array([[1.0, 5.0], [2.0, 3.0], [3.0, 4.0], [4.0, 1.0], [2.0, 3.0], [np.nan, 0.0]])
    np.testing.assert_array_equal(non_dominated(y), [True, True, False, True, True, False])
    np.testing.assert_array_equal(
        non_dominated(y, maximize=[False, True]), [True, False, False, False, False, False]
    )
    space = DesignSpace([Continuous("t", 0.0, 1.0)])
    study = Study(
        FunctionEvaluator(
            lambda p: [p["t"] ** 2, (1 - p["t"]) ** 2, p["t"]], space, ["a", "b", "t"]
        )
    )
    study.run([{"t": t} for t in (0.0, 0.25, 0.5, 0.75, 1.0)])
    study.run([{"t": 0.5}])  # cached
    front = pareto_front(study, ["a", "b"])
    assert len(front) == 5 and front.names == ["a", "b"]
    np.testing.assert_allclose(front.values[:, 0], [0.0, 0.0625, 0.25, 0.5625, 1.0])
    # another point on the front, a dominated point and an outcome constraint
    study.run([{"t": 0.6}])
    assert len(pareto_front(study, ["a", "b"])) == 6
    worse = Study(FunctionEvaluator(lambda p: [1.0, 1.0, 0.0], space, ["a", "b", "t"]))
    worse.run([{"t": 0.1}])
    assert len(pareto_front(worse, ["a", "b"])) == 1
    constrained = pareto_front(study, ["a", "b"], constraints=[OutcomeConstraint("t", upper=0.5)])
    assert max(p["t"] for p in constrained.params) <= 0.5 and len(constrained) == 3
    # t = 0 minimises a and maximises b: it dominates everything
    mixed = pareto_front(study, ["a", lambda v: v[1]], maximize=[False, True])
    assert mixed.params == [{"t": 0.0}] and mixed.names == ["a", "<lambda>"]


# --- optional BoTorch path -----------------------------------------------------------------------


def test_botorch_path_names_the_extra_when_missing(monkeypatch):
    from hpfem.opt import multi_fidelity_optimize, pareto_optimize

    monkeypatch.setitem(sys.modules, "botorch", None)
    with pytest.raises(ImportError, match="opt-bo"):
        pareto_optimize(branin, ["value", "value"], space=BRANIN_SPACE)
    with pytest.raises(ImportError, match="opt-bo"):
        multi_fidelity_optimize(branin, fidelities=[{}, {"order": 2}], space=BRANIN_SPACE)


def test_botorch_pareto_optimize():
    pytest.importorskip("botorch")
    from hpfem.opt import pareto_optimize

    space = DesignSpace([Continuous("t", 0.0, 1.0), Continuous("s", 0.0, 1.0)])
    f = FunctionEvaluator(
        lambda p: [p["t"] ** 2 + p["s"] ** 2, (1 - p["t"]) ** 2 + p["s"] ** 2], space, ["a", "b"]
    )
    result = pareto_optimize(f, ["a", "b"], max_evaluations=14, seed=0)
    assert result.evaluations == 14 and len(result.front) >= 3
    # the true front has s = 0
    assert np.median([p["s"] for p in result.front.params]) < 0.2


def test_botorch_multi_fidelity_optimize():
    pytest.importorskip("botorch")
    from hpfem.opt import multi_fidelity_optimize

    def f(p, fidelity=None):
        order = (fidelity or {}).get("order", 3)
        return branin(p) + 2.0 / order**2  # a discretisation error decaying with the order

    result = multi_fidelity_optimize(
        FunctionEvaluator(f, BRANIN_SPACE),
        fidelities=[{"order": 1}, {"order": 2}, {"order": 4}],
        costs=[1.0, 3.0, 10.0],
        max_evaluations=16,
        seed=0,
        num_fantasies=16,
        raw_samples=128,
        num_restarts=4,
    )
    assert result.fidelity == {"order": 4} and result.evaluation.fidelity == {"order": 4}
    assert result.value < 5.0

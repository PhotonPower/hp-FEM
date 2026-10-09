"""Classical optimisers driving a study (M16 S3, ADR-0012 §2, §3, §5): L-BFGS-B with the
gradient from the Jacobian on Rosenbrock (2D, 4D), Nelder-Mead on a smooth function,
differential evolution (seeded) on Branin, replay from the store, cancellation and resume,
the failure policy, constraints, the remesh restart and the objective forms."""

import json
import math

import numpy as np
import pytest

import hpfem
from hpfem.opt import (
    Categorical,
    Continuous,
    DesignSpace,
    Evaluation,
    EvaluationFailed,
    FunctionEvaluator,
    LinearConstraint,
    Study,
    minimize,
)


def rosenbrock_nd(x):
    x = np.asarray(x, dtype=float)
    value = float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))
    grad = np.zeros_like(x)
    grad[:-1] += -400.0 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2.0 * (1.0 - x[:-1])
    grad[1:] += 200.0 * (x[1:] - x[:-1] ** 2)
    return value, grad


class Rosenbrock:
    """An evaluator of the n-dimensional Rosenbrock function that counts its calls."""

    def __init__(self, n=2, fail=None):
        self.parameters = [Continuous(f"x{i}", -2.0, 2.0) for i in range(n)]
        self.observables = ["f"]
        self.name = "rosenbrock"
        self.settings = {"n": n}
        self.calls = 0
        self.fail = fail

    def __call__(self, params, *, jacobian=False, fidelity=None, cancel=None):
        self.calls += 1
        x = np.array([params[p.name] for p in self.parameters])
        if self.fail is not None and self.fail(x):
            return Evaluation.failure(params, "solver diverged", 1)
        value, grad = rosenbrock_nd(x)
        return Evaluation(params, [value], grad[None, :] if jacobian else None)


def branin(p):
    x, y = p["x"], p["y"]
    b, c, t = 5.1 / (4 * math.pi**2), 5 / math.pi, 1 / (8 * math.pi)
    return (y - b * x * x + c * x - 6) ** 2 + 10 * (1 - t) * math.cos(x) + 10


BRANIN_SPACE = [Continuous("x", -5.0, 10.0), Continuous("y", 0.0, 15.0)]
BRANIN_MINIMA = [(-math.pi, 12.275), (math.pi, 2.275), (9.42478, 2.475)]
BRANIN_MIN = 0.397887357729738


def test_lbfgsb_on_rosenbrock_2d_and_replay(tmp_path):
    path = tmp_path / "rb.study.jsonl"
    evaluator = Rosenbrock(2)
    study = Study(evaluator, path)
    result = minimize(study, "f", x0=[-1.2, 1.0])
    assert result.success, result.message
    assert np.allclose(result.x, [1.0, 1.0], atol=1e-5)
    assert result.value < 1e-10 and result.method == "L-BFGS-B"
    assert result.params == {"x0": result.x[0], "x1": result.x[1]}
    # every evaluation is in the study and the store, with its Jacobian
    assert evaluator.calls == result.new_evaluations == len(study) > 10
    assert result.evaluations == result.new_evaluations + result.cache_hits
    history = result.history()
    assert len(history) == len(study) and history.jacobian.shape == (len(study), 1, 2)
    assert study.state["method"] == "L-BFGS-B" and study.state["state"]["status"] == "done"
    assert study.state["state"]["best"]["value"] == result.value
    # the same optimisation on the stored study replays every point from the cache
    again = Rosenbrock(2)
    replay = minimize(Study(again, path), "f", x0=[-1.2, 1.0])
    assert again.calls == 0 and np.array_equal(replay.x, result.x)
    assert replay.cache_hits == replay.evaluations
    # without x0 it restarts from the best stored point and stops at once
    restart = minimize(Study(again, path), "f")
    assert again.calls <= 3 and np.allclose(restart.x, 1.0, atol=1e-5)


def test_lbfgsb_on_rosenbrock_4d_with_an_evaluator_and_a_path(tmp_path):
    evaluator = Rosenbrock(4)
    result = minimize(evaluator, x0=[-1.2, 1.0, -1.2, 1.0], path=tmp_path / "rb4.study.jsonl")
    assert result.success, result.message
    assert np.allclose(result.x, 1.0, atol=1e-4), result.x
    assert (tmp_path / "rb4.study.jsonl").exists()
    lines = (tmp_path / "rb4.study.jsonl").read_text(encoding="utf-8").splitlines()
    states = [json.loads(line) for line in lines if '"type": "state"' in line]
    assert len(states) == result.iterations + 1  # one per iteration plus the final one
    assert [s["iteration"] for s in states[:3]] == [1, 2, 3]


def test_nelder_mead_on_a_smooth_function():
    def f(p):
        x, y = p["x"], p["y"]
        return (x - 0.3) ** 2 + 2 * (y + 0.2) ** 2 + 0.5 * (x - 0.3) * (y + 0.2) + 1.0

    space = DesignSpace([Continuous("x", -1.0, 1.0), Continuous("y", -1.0, 1.0)])
    result = minimize(f, method="Nelder-Mead", space=space, x0={"x": 0.8, "y": 0.5})
    assert result.success, result.message
    assert np.allclose(result.x, [0.3, -0.2], atol=1e-6)
    assert result.value == pytest.approx(1.0, abs=1e-12)
    assert result.study.evaluations[0].params == {"x": 0.8, "y": 0.5}
    assert len(result.study) == result.new_evaluations


def test_differential_evolution_on_branin_is_seeded_and_replays(tmp_path):
    space = DesignSpace(BRANIN_SPACE)
    path = tmp_path / "branin.study.jsonl"
    options = {"popsize": 8, "tol": 1e-8, "maxiter": 200}
    result = minimize(branin, method="de", space=space, path=path, seed=3, options=options)
    assert result.value == pytest.approx(BRANIN_MIN, abs=1e-5)
    assert min(np.hypot(result.x[0] - a, result.x[1] - b) for a, b in BRANIN_MINIMA) < 1e-2
    assert result.method == "differential-evolution" and result.new_evaluations == len(result.study)
    state = result.study.state["state"]
    assert len(state["population"]) == 8 * 2 and state["status"] == "done"
    # same seed on the stored study: the identical run, entirely from the cache
    calls = []

    def counted(p):
        calls.append(p)
        return branin(p)

    again = minimize(counted, method="de", space=space, path=path, seed=3, options=options)
    assert not calls and np.array_equal(again.x, result.x)
    # another seed explores other points
    other = minimize(branin, method="de", space=space, seed=4, options=options)
    assert other.value == pytest.approx(BRANIN_MIN, abs=1e-5)


def test_cancellation_leaves_a_consistent_store_and_the_rerun_resumes(tmp_path):
    path = tmp_path / "cancel.study.jsonl"
    reference = minimize(Rosenbrock(2), x0=[-1.2, 1.0])
    evaluator = Rosenbrock(2)
    study = Study(evaluator, path, cancel=lambda: evaluator.calls >= 12)
    with pytest.raises(hpfem.Cancelled):
        minimize(study, x0=[-1.2, 1.0])
    stored = Study.load(path)
    assert len(stored) == 12 and stored.state["state"]["status"] == "cancelled"
    assert stored.state["method"] == "L-BFGS-B" and "best" in stored.state["state"]
    # the rerun with the same start replays the 12 stored points and continues
    rerun = Rosenbrock(2)
    result = minimize(Study(rerun, path), x0=[-1.2, 1.0])
    assert rerun.calls == reference.new_evaluations - 12
    np.testing.assert_array_equal(result.x, reference.x)
    assert len(Study.load(path)) == reference.new_evaluations


def test_differential_evolution_resumes_from_its_population(tmp_path):
    path = tmp_path / "de.study.jsonl"
    space = DesignSpace(BRANIN_SPACE)
    calls = []

    def f(p):
        calls.append(p)
        return branin(p)

    study = Study(f, path, space, cancel=lambda: len(calls) >= 40)
    with pytest.raises(hpfem.Cancelled):
        minimize(study, method="de", seed=1, options={"popsize": 8, "tol": 1e-8})
    state = Study.load(path).state
    assert state["state"]["status"] == "cancelled" and state["iteration"] >= 1
    population = np.array(state["state"]["population"])
    resumed = minimize(f, method="de", space=space, path=path, seed=2,
                       options={"popsize": 8, "tol": 1e-8})  # fmt: skip
    assert resumed.value == pytest.approx(BRANIN_MIN, abs=1e-5)
    assert population.shape == (16, 2)
    # it continued from the stored generation, not from a new initial population
    assert resumed.iterations > state["iteration"]
    first = resumed.study.evaluations[state["evaluations"]].params
    assert first != {"x": population[0][0], "y": population[0][1]}


def test_failure_policy_rejects_steps_and_stops_after_max_failures():
    # a region where the "solver" fails: L-BFGS-B backtracks out of it
    evaluator = Rosenbrock(2, fail=lambda x: x[0] > 1.3)
    result = minimize(evaluator, x0=[-1.2, 1.0])
    assert result.success, result.message
    assert np.allclose(result.x, 1.0, atol=1e-5)
    assert result.failures == len(result.study.failed) > 0
    assert all(math.isnan(e.values[0]) for e in result.study.failed)
    # Nelder-Mead with the same evaluator
    nm = minimize(Rosenbrock(2, fail=lambda x: x[0] > 1.3), method="nm", x0=[-1.2, 1.0],
                  options={"xatol": 1e-9, "fatol": 1e-14})  # fmt: skip
    assert np.allclose(nm.x, 1.0, atol=1e-4)
    # too many failures stop the run with the best point so far
    flaky = Rosenbrock(2, fail=lambda x: x[0] > -1.0)
    stopped = minimize(flaky, x0=[-1.2, 1.0], max_failures=3)
    assert not stopped.success and "failed" in stopped.message and stopped.failures == 4
    assert stopped.study.state["state"]["status"] == "stopped"
    # a failed start point is an error
    with pytest.raises(EvaluationFailed):
        minimize(Rosenbrock(2, fail=lambda x: True), x0=[0.0, 0.0])


def test_linear_constraints():
    def f(p):
        return (p["x"] - 1.0) ** 2 + (p["y"] - 1.0) ** 2

    space = DesignSpace(
        [Continuous("x", -2.0, 2.0), Continuous("y", -2.0, 2.0)],
        [LinearConstraint({"x": 1.0, "y": 1.0}, upper=1.0)],
    )
    de = minimize(f, method="de", space=space, seed=0, options={"tol": 1e-10, "popsize": 10})
    assert np.allclose(de.x, [0.5, 0.5], atol=2e-3) and space.feasible(de.params)
    assert all(space.feasible(e.params) for e in de.study.evaluations)
    nm = minimize(f, method="nm", space=space, x0=[0.0, 0.0])
    # an extreme barrier: Nelder-Mead creeps along the constraint
    assert np.allclose(nm.x, [0.5, 0.5], atol=2e-2) and space.feasible(nm.params)
    assert nm.infeasible > 0 and nm.failures == 0  # rejected without evaluation
    assert all(space.feasible(e.params) for e in nm.study.evaluations)


class Remeshing:
    """Rosenbrock with a 'reference mesh' that is rebuilt when a point is more than 0.5 away
    from where it was built (ADR-0012 §3); reports the remesh in ``meta``."""

    parameters = [Continuous("x", -2.0, 2.0), Continuous("y", -2.0, 2.0)]
    observables = ["f"]

    def __init__(self, start):
        self.reference = np.array(start, dtype=float)
        self.mesh_id = 0

    def __call__(self, params, *, jacobian=False, fidelity=None, cancel=None):
        x = np.array([params["x"], params["y"]])
        meta = {}
        if np.max(np.abs(x - self.reference)) > 0.5:
            self.reference, self.mesh_id = x, self.mesh_id + 1
            meta["remesh"] = {"mesh_id": self.mesh_id, "reason": "quality guard"}
        value, grad = rosenbrock_nd(x)
        return Evaluation(params, [value], grad[None, :] if jacobian else None,
                          mesh_id=self.mesh_id, meta=meta)  # fmt: skip


def test_remesh_restarts_the_local_optimisers():
    result = minimize(Remeshing([-1.2, 1.0]), x0=[-1.2, 1.0])
    assert result.restarts >= 1 and len(result.study.remeshes) == result.restarts
    assert result.success and np.allclose(result.x, 1.0, atol=1e-5)
    assert result.evaluation.mesh_id == result.study.evaluations[-1].mesh_id
    # Nelder-Mead keeps its simplex across remeshes
    nm = minimize(Remeshing([-1.2, 1.0]), method="nm", x0=[-1.2, 1.0],
                  options={"xatol": 1e-9, "fatol": 1e-14})  # fmt: skip
    assert nm.restarts == 0 and len(nm.study.remeshes) >= 1
    assert np.allclose(nm.x, 1.0, atol=1e-4)


def test_objective_forms_maximize_log_scale_fixed_and_budget():
    space = DesignSpace(
        [Continuous("k", 1e-3, 1e3, log=True), Continuous("a", -1.0, 1.0),
         Categorical("mode", ("te", "tm"))]
    )  # fmt: skip

    def model(p, jacobian=False):
        k, a = p["k"], p["a"]
        shift = 0.0 if p["mode"] == "te" else 0.5
        y = np.array([math.log10(k) - 1.0, a - shift])
        jac = np.array([[1.0 / (k * math.log(10)), 0.0, 0.0], [0.0, 1.0, 0.0]])
        return (y, jac) if jacobian else y

    f = FunctionEvaluator(model, space, ["log_k", "a"])

    with pytest.raises(ValueError, match="categorical"):
        minimize(f)

    def sum_of_squares(y):  # with its gradient
        return float(y @ y), 2 * y

    result = minimize(f, sum_of_squares, fixed={"mode": "tm"},
                      x0={"k": 1.0, "a": 0.0})  # fmt: skip
    assert result.params["k"] == pytest.approx(10.0, rel=1e-7)
    assert result.params["a"] == pytest.approx(0.5, abs=1e-7) and result.params["mode"] == "tm"
    assert result.names == ["k", "a"]
    # a plain function of the values (gradient by differences of the values) and maximize
    negative = minimize(f, lambda y: -float(y @ y), fixed={"mode": "te"},
                        maximize=True, x0={"k": 1.0, "a": 0.2})  # fmt: skip
    assert negative.params["k"] == pytest.approx(10.0, rel=1e-6)
    assert negative.params["a"] == pytest.approx(0.0, abs=1e-6)
    assert negative.value == pytest.approx(0.0, abs=1e-10)
    # the evaluation budget stops the run
    budget = minimize(f, sum_of_squares, fixed={"mode": "te"},
                      x0={"k": 900.0, "a": 0.9}, max_evaluations=3)  # fmt: skip
    assert not budget.success and budget.evaluations == 3 and "maximum" in budget.message
    # a callback can stop the run
    seen = []
    stopped = minimize(f, sum_of_squares, fixed={"mode": "te"},
                       x0={"k": 900.0, "a": 0.9},
                       callback=lambda info: seen.append(info) or len(seen) >= 2)  # fmt: skip
    assert stopped.iterations == 2 and seen[-1]["iteration"] == 2
    with pytest.raises(ValueError, match="method"):
        minimize(f, fixed={"mode": "te"}, method="newton")

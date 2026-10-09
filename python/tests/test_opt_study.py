"""Study framework (M16 S2, ADR-0012 §2, §5, §7): design spaces (validation, unit-cube
encoding, constraints, sampling), the evaluator contract and its normalisation, the cache key,
the JSON-lines store (round trip of complex values, arrays, NaN and infinity; truncated last
line), resume after a cancellation without evaluating again, failure recording, progress
events, an L-BFGS-B run on the Rosenbrock function driving a study, and a grating evaluator
with its Jacobian on a tiny structured cell."""

import json
import math

import numpy as np
import pytest
import scipy.optimize

import hpfem
from hpfem import grating, meshing, units
from hpfem.meshing import Shape, Slab, UnitCell
from hpfem.opt import (
    Categorical,
    Continuous,
    DesignSpace,
    Evaluation,
    EvaluationFailed,
    FunctionEvaluator,
    GeometryParameter,
    Integer,
    LinearConstraint,
    MaterialParameter,
    NonlinearConstraint,
    Study,
    StudyError,
    as_evaluation,
    complex_names,
    split_complex,
)
from hpfem.opt.study import from_jsonable, settings_hash, to_jsonable


def mixed_space(constraints=()):
    return DesignSpace(
        [
            Continuous("width", 100e-9, 300e-9, unit="m"),
            Continuous("rate", 1e-3, 1e3, log=True),
            Integer("layers", 1, 5),
            Categorical("material", ("air", "glass", 3)),
        ],
        constraints,
    )


def rosenbrock(params, jacobian=False):
    x, y = params["x"], params["y"]
    value = (1 - x) ** 2 + 100 * (y - x * x) ** 2
    if not jacobian:
        return value
    return value, np.array([-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)])


ROSENBROCK_SPACE = [Continuous("x", -2.0, 2.0), Continuous("y", -1.0, 3.0)]


class Counter:
    """A function evaluator that counts its calls."""

    def __init__(self, function):
        self.function = function
        self.calls = 0

    def __call__(self, params, jacobian=False):
        self.calls += 1
        return self.function(params, jacobian=jacobian)


# --- design space ------------------------------------------------------------------------------


def test_design_space_validates_points():
    space = mixed_space()
    assert space.names == ["width", "rate", "layers", "material"]
    point = space.validate({"material": "glass", "layers": 3.0, "rate": 1, "width": 200e-9})
    assert list(point) == space.names  # parameter order
    assert point["layers"] == 3 and isinstance(point["layers"], int)
    assert space.validate({"width": 300e-9 * (1 + 1e-15), "rate": 1.0, "layers": 1,
                           "material": 3})["width"] == 300e-9  # fmt: skip
    good = {"width": 2e-7, "rate": 1.0, "layers": 2, "material": "air"}
    bad = [
        {"rate": 1.0, "layers": 2, "material": "air"},  # missing
        {**good, "height": 1.0},  # unknown
        {**good, "width": 400e-9},  # outside the bounds
        {**good, "layers": 2.5},  # not an integer
        {**good, "layers": True},
        {**good, "material": "gold"},  # not a choice
        {**good, "material": 3.5},
        {**good, "width": math.nan},
        {**good, "width": "wide"},
    ]
    for params in bad:
        with pytest.raises(ValueError):
            space.validate(params)
    with pytest.raises(ValueError):
        Continuous("a", 1.0, 1.0)
    with pytest.raises(ValueError):
        Continuous("a", 0.0, math.inf)
    with pytest.raises(ValueError):
        Continuous("a", 0.0, 1.0, log=True)
    with pytest.raises(ValueError):
        Categorical("c", ("a", "a"))
    with pytest.raises(ValueError):
        Categorical("c", (1, True, object()))
    with pytest.raises(ValueError):
        DesignSpace([Continuous("a", 0, 1), Integer("a", 0, 1)])
    # the S1 parameters are continuous dimensions with their bounds and scale
    eps = MaterialParameter("eps", 3, "re", 2.0, 4.0)
    height = GeometryParameter.field("h", 0, "height", 100e-9, 200e-9, scale=10e-9)
    space = DesignSpace([eps, height])
    assert isinstance(space["eps"], Continuous) and space["eps"].typical == 2.0
    assert space["h"].scale == 10e-9 and space["h"].upper == 200e-9
    with pytest.raises(ValueError):
        DesignSpace([MaterialParameter("unbounded", 3)])
    wide = Continuous.from_parameter(MaterialParameter("unbounded", 3), lower=1.0, upper=9.0)
    assert (wide.lower, wide.upper) == (1.0, 9.0)


def test_encoding_round_trips():
    space = mixed_space()
    rng = np.random.default_rng(0)
    for u in rng.random((50, 4)):
        point = space.decode(u)
        assert space.feasible(point)
        again = space.decode(space.encode(point))
        assert again["layers"] == point["layers"] and again["material"] == point["material"]
        assert again["width"] == pytest.approx(point["width"], rel=1e-14)
        assert again["rate"] == pytest.approx(point["rate"], rel=1e-12)
        e = space.encode(point)
        assert e[0] == pytest.approx(u[0], abs=1e-12) and e[1] == pytest.approx(u[1], abs=1e-12)
        assert np.all((e >= 0) & (e <= 1))
        back = space.from_vector(space.to_vector(point))
        assert back["material"] == point["material"] and back["layers"] == point["layers"]
        assert back["width"] == point["width"]
    # the log scale: the geometric mean sits in the middle of the cube
    assert space["rate"].encode(1.0) == pytest.approx(0.5)
    assert space["rate"].decode(0.5) == pytest.approx(1.0)
    # every integer and every choice survives, and the cells are equal
    for k in range(1, 6):
        assert space["layers"].decode(space["layers"].encode(k)) == k
    assert [space["layers"].decode(u) for u in (0.0, 0.199, 0.2, 0.999, 1.0)] == [1, 1, 2, 5, 5]
    for c in ("air", "glass", 3):
        assert space["material"].decode(space["material"].encode(c)) == c
    assert space.decode([-1.0, 2.0, 0.5, 0.5])["width"] == 100e-9  # clipped
    assert space.to_vector({"width": 2e-7, "rate": 1.0, "layers": 2, "material": 3})[3] == 2.0
    np.testing.assert_array_equal(space.bounds(), [[100e-9, 300e-9], [1e-3, 1e3], [1, 5], [0, 2]])


def test_constraints_and_sampling():
    space = DesignSpace(
        [Continuous("x", 0.0, 1.0), Continuous("y", 0.0, 1.0), Integer("n", 1, 3)],
        [
            LinearConstraint({"x": 1.0, "y": 1.0}, upper=1.0, name="budget"),
            NonlinearConstraint(lambda p: p["x"] * p["y"], lower=0.01, name="area"),
        ],
    )
    assert space.feasible({"x": 0.5, "y": 0.5, "n": 1})  # on the linear bound
    assert not space.feasible({"x": 0.6, "y": 0.5, "n": 1})
    assert not space.feasible({"x": 0.0, "y": 0.5, "n": 1})
    assert not space.feasible({"x": 2.0, "y": 0.5, "n": 1})  # outside the bounds
    assert space.violation({"x": 0.75, "y": 0.5, "n": 1}) == pytest.approx(0.25)
    with pytest.raises(ValueError, match="budget"):
        space.check({"x": 0.75, "y": 0.5, "n": 1})
    with pytest.raises(ValueError, match="area"):
        space.check({"x": 0.0, "y": 0.5, "n": 1})
    a, lower, upper = space.linear_constraints()
    np.testing.assert_array_equal(a, [[1.0, 1.0, 0.0]])
    assert lower[0] == -math.inf and upper[0] == 1.0
    for method in ("lhs", "sobol", "random"):
        points = space.sample(20, method, seed=1)
        assert len(points) == 20 and all(space.feasible(p) for p in points)
    assert space.sample(5, seed=3) == space.sample(5, seed=3)  # reproducible
    # without constraints the Latin hypercube puts one point in every stratum
    plain = DesignSpace([Continuous("x", 0.0, 1.0), Continuous("y", 0.0, 1.0)])
    u = np.array([plain.encode(p) for p in plain.sample(10, "lhs", seed=2)])
    for column in u.T:
        assert sorted(np.floor(column * 10).astype(int)) == list(range(10))
    impossible = DesignSpace([Continuous("x", 0.0, 1.0)], [LinearConstraint({"x": 1.0}, 2.0)])
    with pytest.raises(ValueError):
        impossible.sample(2)
    with pytest.raises(ValueError):
        DesignSpace([Continuous("x", 0.0, 1.0)], [LinearConstraint({"z": 1.0}, upper=1.0)])
    # serialisation: everything but the nonlinear function comes back
    again = DesignSpace.from_dict(json.loads(json.dumps(to_jsonable(space.to_dict()))))
    assert again.to_dict() == space.to_dict()
    with pytest.raises(StudyError):
        again.check({"x": 0.5, "y": 0.4, "n": 1})  # the nonlinear function is not stored


# --- evaluator contract ---------------------------------------------------------------------------


def test_results_are_normalised_into_evaluations():
    p = {"x": 1.0, "y": 2.0}
    e = as_evaluation(3.0, p)
    assert e.ok and e.values.tolist() == [3.0] and e.jacobian is None and e.params == p
    e = as_evaluation((3.0, [1.0, 2.0]), p, num_params=2)  # value and gradient
    assert e.jacobian.shape == (1, 2) and e.gradient.tolist() == [1.0, 2.0]
    e = as_evaluation(([1.0, 2.0], np.eye(2), [0.1, 0.2]), p)
    assert e.error.tolist() == [0.1, 0.2] and e.jacobian.shape == (2, 2)
    e = as_evaluation({"value": 1.5, "gradient": [0.0, 1.0], "meta": {"dofs": 10}}, p)
    assert e.meta == {"dofs": 10} and e.gradient.tolist() == [0.0, 1.0]
    # complex observables: real and imaginary part, rows of the Jacobian alike
    e = as_evaluation(([1 + 2j, 3 - 1j], [[1j, 2.0], [0.5, -1j]]), p)
    assert e.values.tolist() == [1.0, 2.0, 3.0, -1.0]
    np.testing.assert_array_equal(e.jacobian, [[0, 2], [1, 0], [0.5, 0], [0, -1]])
    assert complex_names(["r", "t"]) == ["r.re", "r.im", "t.re", "t.im"]
    values, jac = split_complex(np.array([2.0]), None)
    assert values.tolist() == [2.0] and jac is None
    given = Evaluation({}, [1.0])
    assert as_evaluation(given, p) is given and given.params == p
    for bad in ({"jacobian": [1.0]}, {"values": 1, "extra": 2}, (1.0, 2.0, 3.0, 4.0)):
        with pytest.raises(ValueError):
            as_evaluation(bad, p)
    with pytest.raises(ValueError):
        as_evaluation((1.0, [1.0, 2.0, 3.0]), p, num_params=2)
    with pytest.raises(ValueError):
        as_evaluation([1.0, 2.0], p, num_values=1)
    with pytest.raises(TypeError):
        Evaluation(p, [1.0], jacobian=[[1j, 0.0]])
    with pytest.raises(ValueError):
        Evaluation(p, [1.0, 2.0], jacobian=[1.0, 2.0])  # a gradient needs a single value
    with pytest.raises(ValueError):
        Evaluation(p, [1.0], status="maybe")


def test_function_evaluator_passes_keywords_and_catches_failures():
    seen = {}

    def f(params, jacobian=False, fidelity=None, cancel=None):
        seen.update(jacobian=jacobian, fidelity=fidelity, cancel=cancel)
        if params["x"] < 0:
            raise RuntimeError("negative x")
        return [params["x"], 2 * params["x"]]

    evaluator = FunctionEvaluator(f, ROSENBROCK_SPACE, ["a", "b"], settings={"order": 3})
    flag = lambda: False  # noqa: E731
    e = evaluator({"x": 1.0, "y": 0.0}, jacobian=True, fidelity={"order": 2}, cancel=flag)
    assert seen == {"jacobian": True, "fidelity": {"order": 2}, "cancel": flag}
    assert e.ok and e.fidelity == {"order": 2} and e.cost >= 0
    e = evaluator({"x": -1.0, "y": 0.0})
    assert e.status == "failed" and np.all(np.isnan(e.values)) and len(e.values) == 2
    assert "negative x" in e.meta["error"] and "RuntimeError" in e.meta["traceback"]

    def cancelled(params):
        raise hpfem.Cancelled("stop")

    e = FunctionEvaluator(cancelled, ROSENBROCK_SPACE)({"x": 0.0, "y": 0.0})
    assert e.status == "cancelled"
    # a plain function of a vector, without keywords
    e = FunctionEvaluator(lambda x: x @ x, ROSENBROCK_SPACE, as_array=True)({"x": 3.0, "y": 4.0})
    assert e.values.tolist() == [25.0]
    wrong = FunctionEvaluator(lambda p: [1.0, 2.0], ROSENBROCK_SPACE)({"x": 0.0, "y": 0.0})
    assert wrong.status == "failed" and "2 values for 1 observables" in wrong.meta["error"]


# --- cache, store, resume ----------------------------------------------------------------------


def test_cache_key_is_robust_and_includes_fidelity_and_settings(tmp_path):
    counter = Counter(rosenbrock)
    space = DesignSpace(ROSENBROCK_SPACE)
    study = Study(FunctionEvaluator(counter, space, settings={"order": 3}), tmp_path / "a.jsonl")
    study.evaluate({"x": 0.3, "y": 0.0})
    study.evaluate({"x": 0.1 + 0.2, "y": -0.0})  # 0.30000000000000004 and -0.0
    study.evaluate({"x": 0.3 * (1 + 1e-14), "y": 1e-15})  # below 12 significant digits
    assert counter.calls == 1 and study.cache_hits == 2 and len(study) == 1
    study.evaluate({"x": 0.3 * (1 + 1e-9), "y": 0.0})
    assert counter.calls == 2
    study.evaluate({"x": 0.3, "y": 0.0}, fidelity={"order": 4})
    assert counter.calls == 3
    e = study.evaluate({"x": 0.3, "y": 0.0}, jacobian=True)  # cached without a Jacobian
    assert counter.calls == 4 and e.jacobian is not None
    assert study.evaluate({"x": 0.3, "y": 0.0}) is e  # the newer record serves both
    assert study.evaluate({"x": 0.3, "y": 0.0}, jacobian=True) is e and counter.calls == 4
    # changed evaluator settings: noted in the store, the old values are not reused
    again = Study(FunctionEvaluator(counter, space, settings={"order": 4}), tmp_path / "a.jsonl")
    assert again.notes[-1]["text"] == "evaluator settings changed"
    assert again.notes[-1]["evaluator"]["settings"] == {"order": 4}
    again.evaluate({"x": 0.3, "y": 0.0})
    assert counter.calls == 5
    third = Study(FunctionEvaluator(counter, space, settings={"order": 4}), tmp_path / "a.jsonl")
    assert len(third.notes) == 1  # no new note for the same settings
    third.evaluate({"x": 0.3, "y": 0.0})
    assert counter.calls == 5
    assert settings_hash({"a": 1, "b": [1.0, 2.0]}) == settings_hash({"b": [1.0, 2.0], "a": 1})


def test_store_round_trip_of_complex_values_arrays_nan_and_inf(tmp_path):
    space = mixed_space([LinearConstraint({"width": 1.0}, upper=250e-9, name="w")])

    def evaluator(params):
        return {
            "values": [params["width"] / 3, math.nan, math.inf],
            "jacobian": np.arange(12.0).reshape(3, 4) / 7,
            "error": [1e-3, math.nan, 0.0],
            "fidelity": {"order": 4},
            "mesh_id": 2,
            "meta": {
                "amplitude": 0.1 - 0.7j,
                "field": np.array([[1 + 1j, 2.0], [0.0, -1j]]),
                "orders": np.arange(3),
                "nested": {"ratio": 1 / 3, "flag": True, "none": None},
                "label": "ok",
            },
        }

    path = tmp_path / "round.study.jsonl"
    params = {"width": 2.123456789012345e-7, "rate": 0.1, "layers": 4, "material": 3}
    study = Study(FunctionEvaluator(evaluator, space, 3), path, meta={"user": "test"})
    original = study.evaluate(params)
    assert original.meta["fidelity_reported"] == {"order": 4} and original.fidelity == {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        json.loads(line)  # strict JSON: no NaN / Infinity literals
    assert "NaN" not in lines[1] and "Infinity" not in lines[1]
    header = json.loads(lines[0])
    assert header["type"] == "study" and header["schema"] == 1
    assert header["hpfem"] == hpfem.__version__ and header["meta"] == {"user": "test"}
    assert header["evaluator"]["observables"] == ["y0", "y1", "y2"]
    assert header["space"]["constraints"][0]["lower"] == "-inf"
    record = json.loads(lines[1])
    assert record["values"][1] is None and record["values"][2] == "inf"
    assert record["meta"]["amplitude"] == {"__complex__": [0.1, -0.7]}
    loaded = Study.load(path)
    assert loaded.space.to_dict() == space.to_dict()
    (e,) = loaded.evaluations
    assert e.params == params and type(e.params["layers"]) is int and e.params["material"] == 3
    assert e.params["width"] == params["width"]  # repr round trip, exact
    np.testing.assert_array_equal(e.values, original.values)  # NaN and inf included
    np.testing.assert_array_equal(e.jacobian, original.jacobian)
    np.testing.assert_array_equal(e.error, original.error)
    assert e.cost == original.cost and e.mesh_id == 2 and e.status == "ok"
    assert e.meta["amplitude"] == 0.1 - 0.7j
    np.testing.assert_array_equal(e.meta["field"], [[1 + 1j, 2.0], [0.0, -1j]])
    assert e.meta["orders"] == [0, 1, 2] and e.meta["nested"]["ratio"] == 1 / 3
    assert e.meta["nested"] == {"ratio": 1 / 3, "flag": True, "none": None}
    with pytest.raises(StudyError):
        loaded.evaluate(params)  # read-only
    assert from_jsonable(to_jsonable(np.array([1j, 2.0]))).tolist() == [1j, 2.0]
    # resuming with another design space is refused
    with pytest.raises(StudyError):
        Study(FunctionEvaluator(evaluator, mixed_space(), 3), path)
    with pytest.raises(FileExistsError):
        Study(FunctionEvaluator(evaluator, space, 3), path, resume=False)


def test_resume_after_cancel_evaluates_only_the_open_points(tmp_path):
    space = DesignSpace(ROSENBROCK_SPACE)
    points = space.sample(6, seed=4)
    path = tmp_path / "cancel.study.jsonl"
    counter = Counter(rosenbrock)
    events = []
    stop = {"after": 3}

    def cancel():
        return sum(e["event"] == "evaluation" for e in events) >= stop["after"]

    study = Study(counter, path, space, emit=events.append, cancel=cancel)
    study.checkpoint("test", 1, {"x": np.array([0.5, 0.25]), "sigma": 0.3, "z": 1 + 2j})
    with pytest.raises(hpfem.Cancelled):
        study.run(points, source="lhs")
    assert counter.calls == 3 and events[-1]["event"] == "cancelled"
    assert len(study.open_proposals()) == 3
    # a new session on the same file: the cache is rebuilt, nothing is evaluated again
    counter2 = Counter(rosenbrock)
    resumed = Study(counter2, path, space, emit=events.append)
    assert len(resumed) == 3 and events[-1]["event"] == "study" and events[-1]["open"] == 3
    assert resumed.state["method"] == "test" and resumed.state["iteration"] == 1
    assert resumed.state["state"]["z"] == 1 + 2j and resumed.state["state"]["x"] == [0.5, 0.25]
    open_points = [o["params"] for o in resumed.open_proposals()]
    assert open_points == points[3:]
    done = resumed.evaluate_open()
    assert counter2.calls == 3 and len(done) == 3 and resumed.open_proposals() == []
    everything = resumed.run(points)  # all from the cache
    assert counter2.calls == 3 and resumed.cache_hits == 6
    for e, p in zip(everything, points, strict=True):
        assert e.params == p and e.values[0] == pytest.approx(rosenbrock(p))
    # the history is in study order and the best point is the minimum
    history = resumed.history()
    assert history.x.shape == (6, 2) and history.values.shape == (6, 1)
    best = resumed.best()
    assert best.values[0] == history.values.min()


def test_truncated_last_line_is_dropped_on_resume(tmp_path):
    space = DesignSpace(ROSENBROCK_SPACE)
    path = tmp_path / "crash.study.jsonl"
    study = Study(rosenbrock, path, space)
    study.run([{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}])
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"type": "evaluation", "params": {"x": 0.5, "y"')  # crash during a write
    assert len(Study.load(path)) == 2  # read-only: ignored, file untouched
    resumed = Study(rosenbrock, path, space)
    assert len(resumed) == 2 and resumed.notes[-1]["text"].startswith("dropped a truncated")
    resumed.evaluate({"x": 0.5, "y": 0.5})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert all(json.loads(line) for line in lines)
    assert len(Study.load(path)) == 3
    # a broken line in the middle is an error, not a crash to repair
    lines.insert(2, "{broken")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(StudyError):
        Study.load(path)


def test_failures_are_recorded_and_the_study_goes_on(tmp_path):
    space = DesignSpace(ROSENBROCK_SPACE)
    calls = {"n": 0}

    def fragile(params):
        calls["n"] += 1
        if params["x"] > 1.5:
            raise ValueError(f"mesh inverted at x = {params['x']}")
        return params["x"] ** 2

    path = tmp_path / "fail.study.jsonl"
    study = Study(fragile, path, space)
    points = [{"x": 0.0, "y": 0.0}, {"x": 1.9, "y": 0.0}, {"x": 1.0, "y": 0.0}]
    results = study.run(points)
    assert [e.status for e in results] == ["ok", "failed", "ok"]
    assert math.isnan(results[1].values[0]) and "mesh inverted" in results[1].meta["error"]
    assert len(study.failed) == 1 and len(study.history()) == 2
    assert study.history(status=None).status == ["ok", "failed", "ok"]
    assert np.isnan(study.history("failed").values).all()
    study.evaluate(points[1])  # a cached failure is not repeated
    assert calls["n"] == 3
    retry = Study(fragile, path, space, retry_failed=True)
    assert retry.failed[0].meta["error"].startswith("ValueError")
    retry.evaluate(points[1])
    assert calls["n"] == 4 and len(retry.failed) == 2
    strict = Study(fragile, None, space, raise_errors=True)
    with pytest.raises(EvaluationFailed, match="mesh inverted"):
        strict.evaluate(points[1])
    assert len(strict.failed) == 1  # recorded before raising

    class Evaluator:  # an evaluator class that reports the failure itself
        parameters = ROSENBROCK_SPACE
        observables = ["a", "b"]
        settings = {"kind": "class"}

        def __call__(self, params, *, jacobian=False, fidelity=None, cancel=None):
            if params["x"] > 1.5:
                return Evaluation(params, [], status="failed", meta={"error": "no convergence"})
            remesh = {"mesh_id": 1, "reason": "quality"}
            return Evaluation(params, [1.0, 2.0], meta={"remesh": remesh})

    own = Study(Evaluator(), tmp_path / "own.study.jsonl")
    assert own.evaluate(points[1]).values.shape == (2,)  # padded with NaN
    own.evaluate(points[0])
    assert own.remeshes[0]["mesh_id"] == 1 and own.remeshes[0]["params"] == points[0]
    assert Study.load(tmp_path / "own.study.jsonl").remeshes[0]["reason"] == "quality"
    with pytest.raises(ValueError):
        own.evaluate({"x": 3.0, "y": 0.0})  # outside the space: nothing recorded
    assert len(own) == 2


def test_progress_events():
    space = DesignSpace(ROSENBROCK_SPACE)
    events = []
    study = Study(rosenbrock, None, space, emit=events.append)
    study.run([{"x": 0.0, "y": 0.0}, {"x": 0.5, "y": 0.0}, {"x": 0.0, "y": 0.0}])
    study.note("checked", value=1.5)
    kinds = [e["event"] for e in events]
    assert kinds == ["study", "proposal", "evaluation", "evaluation", "evaluation", "note"]
    assert len(events[1]["points"]) == 2  # the repeated point is proposed once
    evaluations = [e for e in events if e["event"] == "evaluation"]
    assert [(e["i"], e["n"], e["cached"], e["index"]) for e in evaluations] == [
        (0, 3, False, 0),
        (1, 3, False, 1),
        (2, 3, True, 0),
    ]
    for e in events:
        json.dumps(e, allow_nan=False)  # JSON-ready, as the job runner streams them
    assert evaluations[1]["values"] == [rosenbrock({"x": 0.5, "y": 0.0})]


def test_lbfgsb_on_rosenbrock_drives_a_study_and_replays_from_the_store(tmp_path):
    space = DesignSpace(ROSENBROCK_SPACE)
    path = tmp_path / "rosenbrock.study.jsonl"

    def optimise(study):
        def fun(x):
            e = study.evaluate(space.from_vector(x), jacobian=True)
            return e.values[0], e.gradient

        return scipy.optimize.minimize(
            fun, [-1.2, 1.0], jac=True, method="L-BFGS-B", bounds=space.bounds(),
            options={"gtol": 1e-10, "ftol": 1e-15},
        )  # fmt: skip

    counter = Counter(rosenbrock)
    study = Study(counter, path, space)
    result = optimise(study)
    assert np.allclose(result.x, [1.0, 1.0], atol=1e-5)
    best = study.best()
    assert best.values[0] < 1e-9 and best.params["x"] == pytest.approx(1.0, abs=1e-5)
    assert counter.calls == len(study) > 10
    # the same optimisation on the stored study replays every point from the cache
    replay = Counter(rosenbrock)
    again = optimise(Study(replay, path, space))
    assert replay.calls == 0 and np.array_equal(again.x, result.x)
    history = Study.load(path).history()
    assert history.jacobian.shape == (len(study), 1, 2)
    np.testing.assert_array_equal(history.x[-1], [p for p in study.evaluations[-1].params.values()])


# --- a real evaluator -----------------------------------------------------------------------------

NM = units.nm
PERIOD, HEIGHT = 400 * NM, 148 * NM
ROW = HEIGHT / 4
Y0, Y1 = -12 * ROW, 16 * ROW
SUB, LINE = 2, 3


class GratingEvaluator:
    """Zeroth-order reflection of a lamellar grating as a function of the real part of the
    line permittivity, with its derivative from the kept factorisation."""

    def __init__(self):
        cell = UnitCell(PERIOD, Y0, Y1, slabs=[Slab(SUB, Y0, 0.0)],
                        shapes=[Shape("rectangle", LINE, {"x": -100 * NM, "y": 0.0,
                                                          "width": 200 * NM,
                                                          "height": HEIGHT})])  # fmt: skip
        self.mesh = meshing.structured_unit_cell(
            cell, 8, [(Y0, 0.0, 12), (0.0, HEIGHT, 4), (HEIGHT, Y1, 12)]
        )
        glass = hpfem.Material.dielectric(1.5)
        self.stack = hpfem.LayerStack2D(hpfem.Material.dielectric(1.0), [], glass, 0.0)
        self.materials = {SUB: glass, LINE: hpfem.Material(eps_r=3.0)}
        self.parameters = [MaterialParameter("eps_line", LINE, "re", 2.0, 4.0)]
        self.observables = ["R0"]
        self.name = "lamellar R0"
        self.settings = {"order": 2, "wavelength": 500 * NM, "theta_deg": 30.0}
        self.solves = 0

    def __call__(self, params, *, jacobian=False, fidelity=None, cancel=None):
        materials = self.parameters[0].apply(self.materials, params["eps_line"])
        self.solves += 1
        omega = units.angular_frequency(wavelength=500 * NM)
        result = grating.solve(
            self.mesh, materials, self.stack, "s", 30 * units.deg, 0.0, omega, 2,
            pml={"top": 8 * ROW, "bottom": 8 * ROW}, orders_max=1, check=False,
            keep_factorisation=jacobian, cancel=cancel,
        )  # fmt: skip
        r0 = next(o for o in result.R_orders if o.m == 0).efficiency
        jac = None
        if jacobian:
            full, _, columns = grating.jacobian(result, [("eps", LINE)], observables=[("R", 0)])
            jac = full[:, [columns.index(f"eps[{LINE}].re")]]
        return Evaluation(params, [r0], jac, meta={"dofs": result.dofs})


def test_grating_evaluator_in_a_study(tmp_path):
    evaluator = GratingEvaluator()
    path = tmp_path / "grating.study.jsonl"
    study = Study(evaluator, path)
    assert study.space.names == ["eps_line"] and study.space["eps_line"].typical == 2.0
    e = study.evaluate({"eps_line": 3.0}, jacobian=True)
    h = 1e-4
    plus, minus = study.run([{"eps_line": 3.0 + h}, {"eps_line": 3.0 - h}])
    fd = (plus.values[0] - minus.values[0]) / (2 * h)
    assert e.ok and e.meta["dofs"] > 0 and 0 < e.values[0] < 1
    assert abs(e.gradient[0] - fd) < 1e-5 * abs(fd) + 1e-9, (e.gradient[0], fd)
    assert evaluator.solves == 3
    resumed = Study(GratingEvaluator(), path)
    assert resumed.evaluate({"eps_line": 3.0}, jacobian=True).gradient[0] == e.gradient[0]
    assert resumed.evaluator.solves == 0

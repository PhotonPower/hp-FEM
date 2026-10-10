"""hpfem.run schema version 2 (M16 S9): the study tasks "optimize", "reconstruct" and "uq" on a
cheap lamellar cell (order 2, two configurations), their events, the study store next to the
results (a rerun replays it), cancellation, the errors, and that version-1 documents still
report version 1."""

import json

import numpy as np
import pytest

from hpfem import run as runner

BASE = {
    "version": 2,
    "name": "lamellar study",
    "problem": "grating",
    "unit": 1e-9,
    "model": {
        "period": 400,
        "y_bottom": -800,
        "y_top": 950,
        "background_tag": 1,
        "slabs": [{"tag": 2, "y0": -800, "y1": 0}],
        "shapes": [
            {
                "kind": "rectangle",
                "tag": 3,
                "params": {"x": -100, "y": 0, "width": 200, "height": 148},
            },
        ],
    },  # fmt: skip
    "mesh": {"structured": {"nx": 16, "rows": [[-800, 0, 32], [0, 148, 6], [148, 950, 32]]}},
    "materials": {"1": "air", "2": {"n": [1.5, 0.0]}, "3": {"eps": [2.25, 0.0]}},
    "stack": {"incidence": "air", "substrate": {"n": [1.5, 0.0]}, "layers": [], "top": 0},
    "solver": {"order": 2, "pml": {"top": 419, "bottom": 419}, "orders_max": 2},
    "parameters": [
        {"name": "width", "shape": 0, "field": "width", "bounds": [170, 230]},
        {"name": "eps", "material": 3, "part": "re", "bounds": [2.0, 2.6]},
    ],
    "configurations": [
        {"wavelength": 405, "theta_deg": 40, "polarisation": "s", "orders": [["R", 0], ["T", 0]]},
        {"wavelength": 500, "theta_deg": 20, "polarisation": "p", "orders": [["R", 0]]},
    ],
}


def job(task, **block):
    return {**BASE, "task": task, task: block}


def test_optimize_writes_the_study_and_a_rerun_replays_it(tmp_path):
    events = []
    spec = job("optimize", objective=0, maximize=True, method="L-BFGS-B", max_evaluations=6,
               x0={"width": 200, "eps": 2.25})  # fmt: skip
    results = runner.run_job(spec, tmp_path, events.append)
    kinds = [e["event"] for e in events]
    assert kinds[:2] == ["start", "mesh"] and kinds[-1] == "done"
    assert events[0]["points"] is None and events[0]["version_info"]["schema"] == 2
    assert results["version"] == 2 and results["task"] == "optimize"
    assert results["observables"] == ["s 405nm 40deg: R0", "s 405nm 40deg: T0",
                                      "p 500nm 20deg: R0"]  # fmt: skip
    assert results["parameters"]["width"]["reference"] == pytest.approx(200.0)
    assert results["parameters"]["eps"]["reference"] == pytest.approx(2.25)
    opt = results["optimize"]
    first = results["points"][0]
    assert first["params"] == pytest.approx({"width": 200.0, "eps": 2.25})
    assert opt["value"] >= first["values"][0] - 1e-12  # maximised R0
    assert 170 <= opt["params"]["width"] <= 230 and 2.0 <= opt["params"]["eps"] <= 2.6
    evaluations = [e for e in events if e["event"] == "evaluation"]
    assert len(evaluations) == len(results["points"]) and not evaluations[0]["cached"]
    store = tmp_path / "lamellar_study.study.jsonl"
    assert results["study"] == str(store) and store.exists()
    header = json.loads(store.read_text(encoding="utf-8").splitlines()[0])
    assert header["type"] == "study"
    written = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert written["optimize"]["params"] == pytest.approx(opt["params"])
    # the same job again: every point comes from the store
    again = runner.run_job(spec, tmp_path)
    assert again["optimize"]["new_evaluations"] == 0
    assert again["optimize"]["params"] == pytest.approx(opt["params"])
    assert all(p["cached"] for p in again["points"])


def test_reconstruct_recovers_synthetic_parameters():
    truth = {"width": 207.0, "eps": 2.31}
    spec = job("reconstruct", synthetic={"params": truth, "noise": 1e-3, "seed": 3},
               x0={"width": 195, "eps": 2.2})  # fmt: skip
    results = runner.run_job(spec)
    rec = results["reconstruct"]
    assert rec["success"] and len(rec["measured"]) == 3
    for name, value in truth.items():
        assert abs(rec["params"][name] - value) < 4 * rec["std"][name] + 1e-9, name
        assert 0 < rec["std"][name] < 0.05 * value
    assert np.asarray(rec["correlation"]).shape == (2, 2)
    # the posterior needs emcee (optional extra opt-mcmc)
    try:
        import emcee  # noqa: F401
    except ImportError:
        with pytest.raises(runner.JobError, match="opt-mcmc"):
            runner.run_job(job("reconstruct", synthetic={"params": truth}, posterior={}))
    else:
        posterior = {"steps": 300, "walkers": 8}
        data = {"params": truth, "noise": 1e-3}
        post = runner.run_job(job("reconstruct", synthetic=data, posterior=posterior))
        post = post["reconstruct"]
        assert set(post["posterior"]["names"]) == set(truth)


def test_uq_linear_and_surrogate():
    inputs = {"width": {"normal": [200, 2]}, "eps": {"uniform": [2.2, 2.3]}}
    lin = runner.run_job(job("uq", inputs=inputs, propagation="linear", sobol=True))["uq"]
    assert lin["inputs"] == ["width", "eps"] and len(lin["values"]) == 3
    contributions = np.asarray(lin["contributions"])
    assert np.allclose(contributions.sum(axis=1), 1.0)
    assert np.all(np.asarray(lin["std"]) > 0)
    sur = runner.run_job(job("uq", inputs=inputs, propagation="surrogate", points=6,
                             samples=2000, sobol=True, sobol_samples=256))["uq"]  # fmt: skip
    # the surrogate's Monte Carlo agrees with the linearisation for these small inputs
    assert np.allclose(sur["mean"], lin["values"], atol=3 * max(lin["std"]))
    assert np.allclose(sur["std"], lin["std"], rtol=0.2, atol=1e-5)
    first = np.asarray(sur["sobol"]["first"])
    assert first.shape == (3, 2) and np.all(first > -0.05) and np.all(first < 1.05)


def test_cancellation_errors_and_version_one(tmp_path):
    events = []

    def cancel():  # after the first evaluation
        return any(e["event"] == "evaluation" for e in events)

    spec = job("optimize", objective="p 500nm 20deg: R0", method="Nelder-Mead",
               max_evaluations=10)  # fmt: skip
    results = runner.run_job(spec, None, events.append, cancel)
    assert results["cancelled"] and [e["event"] for e in events][-2:] == ["cancelled", "done"]
    with pytest.raises(runner.JobError, match="version 2"):
        runner.run_job({**job("optimize"), "version": 1})
    with pytest.raises(runner.JobError, match="objective"):
        runner.run_job(job("optimize", objective="R7"))
    with pytest.raises(runner.JobError, match="trapezoid|field|material"):
        runner.run_job({**job("uq", inputs={}), "parameters": [{"name": "x", "bounds": [0, 1]}]})
    with pytest.raises(runner.JobError, match="unknown parameter"):
        runner.run_job(job("uq", inputs={"height": {"normal": [148, 1]}}))
    with pytest.raises(runner.JobError, match="synthetic"):
        runner.run_job(job("reconstruct"))
    # version 1 documents keep reporting version 1 (the runner reads 1 and 2)
    assert runner.SUPPORTED_VERSIONS == (1, 2)
    v1 = {k: v for k, v in BASE.items() if k not in ("parameters", "configurations")}
    v1 = {**v1, "version": 1, "incidence": {"polarisation": "s", "theta_deg": 40,
                                            "wavelength": 405}}  # fmt: skip
    first = []
    out = runner.run_job(v1, None, first.append)
    assert out["version"] == 1 and out["version_info"]["schema"] == 1
    assert first[0]["points"] == 1 and first[0]["version_info"]["schema"] == 1

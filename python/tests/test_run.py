"""hpfem.run (M15 F5): a grating job through the runner with JSON-lines events, results and
maps on disk, cancellation and the command line."""

import json

import numpy as np
import pytest

import hpfem
from hpfem import run as runner
from hpfem import units

JOB = {
    "version": 1,
    "name": "glass lamellar",
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
            }
        ],
    },
    "mesh": {"structured": {"nx": 16, "rows": [[-800, 0, 32], [0, 148, 6], [148, 950, 32]]}},
    "materials": {"1": "air", "2": {"n": [1.5, 0.0]}, "3": {"eps": [2.25, 0.0]}},
    "stack": {"incidence": "air", "substrate": {"n": [1.5, 0.0]}, "layers": [], "top": 0},
    "incidence": {"polarisation": "p", "theta_deg": 50, "phi_deg": 30},
    "sweep": {"wavelength": [405]},
    "solver": {"order": 3, "pml": {"top": 419, "bottom": 419}, "orders_max": 2},
    "maps": [{"quantity": "E", "x": [-200, 200, 9], "y": [-300, 400, 8]}],
}


def test_job_reproduces_the_rcwa_and_writes_results_and_maps(tmp_path):
    events = []
    results = runner.run_job(JOB, tmp_path, events.append)
    kinds = [e["event"] for e in events]
    assert kinds[:2] == ["start", "mesh"]
    assert kinds[-1] == "done" and "point" in kinds and "map" in kinds
    assert events[0]["version_info"]["hpfem"] == hpfem.__version__
    point = results["points"][0]
    r = {o["m"]: o["efficiency"] for o in point["R_orders"] if o["propagating"]}
    t = {o["m"]: o["efficiency"] for o in point["T_orders"] if o["propagating"]}
    assert abs(r[0] - 0.010957) < 2e-3 and abs(t[0] - 0.875831) < 2e-3
    assert abs(point["balance"]) < 2e-3
    assert point["theta_deg"] == 50 and point["wavelength"] == pytest.approx(405e-9)
    written = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert written["points"][0]["R"] == point["R"]
    assert len(results["maps"]) == 1
    data = np.load(results["maps"][0]["file"])
    assert data["values"].shape == (8, 9, 3)
    assert np.isfinite(data["values"]).all()
    assert results["mesh"]["num_untagged"] == 0 and results["mesh"]["cell_tags"] == [1, 2, 3]


def test_sweep_cancellation_and_errors(tmp_path):
    job = dict(JOB, sweep={"theta_deg": {"start": 30, "stop": 60, "count": 3}}, maps=[])
    job["incidence"] = dict(JOB["incidence"], wavelength=405)
    events = []

    def cancel():  # cancel once the first point is done (polled between points and phases)
        return any(e["event"] == "point" for e in events)

    results = runner.run_job(job, None, events.append, cancel)
    assert results["cancelled"] and len(results["points"]) == 1
    assert [e["event"] for e in events][-2:] == ["cancelled", "done"]
    assert results["points"][0]["theta_deg"] == pytest.approx(30)
    # the estimate and the progress events of the solve, with the timing in the point
    estimate = next(e for e in events if e["event"] == "estimate")
    # the estimate counts the maps' DoFs, the point the free ones (PEC and Bloch eliminated)
    assert 0.95 < estimate["dofs"] / results["points"][0]["dofs"] < 1.05
    assert estimate["total_bytes"] > 0
    phases = [e["phase"] for e in events if e["event"] == "progress" and e["i"] == 0]
    assert phases == ["assembly", "constraints", "factorisation", "solve", "post", "done"]
    assert set(results["points"][0]["timing"]) >= {"setup", "solve", "solver.factorisation"}
    # cancellation inside a solve: no point is finished
    events = []

    def cancel_in_solve():
        return any(e["event"] == "progress" and e["phase"] == "solve" for e in events)

    results = runner.run_job(job, None, events.append, cancel_in_solve)
    assert results["cancelled"] and results["points"] == []
    assert [e["event"] for e in events][-2:] == ["cancelled", "done"]
    with pytest.raises(runner.JobError, match="version"):
        runner.run_job(dict(JOB, version=7))
    with pytest.raises(runner.JobError, match="material"):
        runner.run_job(dict(JOB, materials={"1": "unobtainium"}))
    with pytest.raises(runner.JobError, match="mesh"):
        runner.run_job(dict(JOB, mesh={}))


def test_command_line(tmp_path, capsys):
    job_file = tmp_path / "job.json"
    job = dict(JOB, maps=[])
    job_file.write_text(json.dumps(job), encoding="utf-8")
    code = runner.main([str(job_file), "--out", str(tmp_path / "out"), "--quiet"])
    assert code == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert lines[0]["event"] == "start" and lines[-1]["event"] == "done"
    assert (tmp_path / "out" / "results.json").exists()
    cancel_file = tmp_path / "cancel"
    cancel_file.write_text("x")
    code = runner.main([str(job_file), "--cancel-file", str(cancel_file)])
    assert code == 2
    assert runner.main([str(tmp_path / "missing.json")]) == 1
    info = runner.version_info()
    assert {"hpfem", "schema", "backends", "openmp", "gmsh"} <= set(info)
    assert units.deg > 0

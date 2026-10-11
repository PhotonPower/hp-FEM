"""The job-runner task ``emitter`` (schema 2, M17 S4): the cost estimate and the dry run, stage A
(Bloch array), stage B (single dipole: F_P, channels, aperture) and stage C (dP/dΩ on a grid
against the homogeneous medium), progress events, cancellation and the argument checks, on a
small homogeneous cell (accuracy: tests/test_dipole_emission.py and the long runs)."""

import copy
import math

import numpy as np
import pytest

import hpfem
from hpfem import grating, units
from hpfem.run import JobError, run_job

N = 1.5


def job(stage="B", **emitter):
    spec = {"position": [0.0, 100.0], "sigma": 60.0, "moment": "isotropic", "stage": stage,
            "wavelength": [1000.0], "scan": {"nodes": 2, "kx_nodes": 2, "angle_nodes": [2, 4],
                                             "symmetric": True, "beta_max_over_k0": 0.3 * N},
            "calibrate": True}  # fmt: skip
    spec.update(emitter)
    return {
        "version": 2, "task": "emitter", "unit": 1e-9,
        "model": {"period": 1000.0, "y_bottom": -2000.0, "y_top": 2000.0, "background_tag": 1},
        "mesh": {"structured": {"nx": 4, "rows": [[-2000.0, 2000.0, 16]]}},
        "materials": {"1": {"n": [N, 0.0]}},
        "stack": {"incidence": {"n": [N, 0.0]}, "substrate": {"n": [N, 0.0]}, "layers": [],
                  "top": 0},
        "solver": {"order": 2, "pml": {"top": 1000.0, "bottom": 1000.0}},
        "emitter": spec,
    }  # fmt: skip


@pytest.fixture(autouse=True)
def quiet():
    hpfem.set_log_level("warn")


def test_dry_run_gives_the_cost_without_solving():
    events = []
    results = run_job(job(dry_run=True, aperture=[0.5]), emit=events.append)
    cost = results["cost"]
    assert results["dry_run"] and results["points"] == []
    assert [e["event"] for e in events].count("emission_cost") == 1
    k0 = 2 * math.pi / 1e-6
    rule = grating.array_scan_rule(1e-6, k0, [N], beta_max=0.3 * N * k0, nodes=2, kx_nodes=2,
                                   symmetric=True, depth=0.5)  # fmt: skip
    assert cost["per_wavelength"]["samples"] == len(rule.weight)
    # channels: 2 θ × 2 φ (mirror symmetries of 4) × 2 sides × s, p; the cone: 2 θ × 2 φ × s, p
    assert cost["per_wavelength"]["plane_wave_solves"] == 16 + 8
    assert cost["solves"] == cost["per_wavelength"]["solves"] == len(rule.weight) + 24
    assert cost["dofs"] > 0 and cost["total_bytes"] > 0 and "DoFs" in cost["memory"]
    assert cost["seconds_per_sample"] > 0 and cost["seconds"] > 0


def test_stage_b_fractions_progress_and_aperture():
    events = []
    results = run_job(job(aperture=[0.5]), emit=events.append)
    (point,) = results["points"]
    fractions = point["fractions"]
    assert fractions["up"] + fractions["down"] + fractions["nonradiated"] == pytest.approx(1.0)
    assert point["purcell"] > 0 and point["n_host"] == pytest.approx(N)
    assert set(point["purcell_by_orientation"]) == {"x", "y", "z", "isotropic"}
    (cone,) = point["aperture"]
    assert cone["theta_max_deg"] == pytest.approx(math.degrees(math.asin(0.5 / N)))
    assert 0 < cone["fraction"] < fractions["up"]
    # the cone in the homogeneous medium: the isotropic pattern is uniform
    k0, sigma = 2 * math.pi / 1e-6, 60e-9
    uniform = N * k0**2 * hpfem.constants.Z0 * (2 / 3) * math.exp(-((N * k0 * sigma) ** 2))
    uniform /= 32 * math.pi**2
    solid_angle = 2 * math.pi * (1 - math.cos(math.asin(0.5 / N)))
    assert cone["power"] == pytest.approx(uniform * solid_angle, rel=3e-2)
    scan = [e for e in events if e["event"] == "progress" and e["phase"] == "scan"]
    assert len(scan) == scan[-1]["num_steps"] == point["samples"] + point["plane_wave_solves"]
    assert [e["event"] for e in events].count("point") == 1
    # the own moment: F_P of a moment along y equals the orientation y
    own = run_job(job(moment=[0.0, 2e-9, 0.0], aperture=[]))["points"][0]
    assert own["purcell"] == pytest.approx(point["purcell_by_orientation"]["y"], rel=1e-10)


def test_stage_c_matches_the_isotropic_dipole_in_a_homogeneous_medium():
    thetas, phis = [0.0, 30.0, 60.0], [0.0, 90.0, 180.0, 270.0]
    results = run_job(job("C", directions={"theta_deg": thetas, "phi_deg": phis,
                                           "sides": ["up", "down"]}, aperture=[1.2]))  # fmt: skip
    (point,) = results["points"]
    omega = units.angular_frequency(wavelength=1e-6)
    k0 = 2 * math.pi / 1e-6
    sigma = 60e-9
    expected = N * k0**2 * hpfem.constants.Z0 * (2 / 3) * math.exp(-((N * k0 * sigma) ** 2))
    expected /= 32 * math.pi**2
    for side in ("up", "down"):
        grid = np.asarray(point["dP_dOmega"][side])
        assert grid.shape == (len(thetas), len(phis))
        assert np.allclose(grid, expected, rtol=3e-2)
    assert point["P_bulk"] == pytest.approx(
        grating.dipole_bulk_power((1.0, 0.0, 0.0), omega, N, sigma), rel=1e-12
    )
    # the aperture from the grid: θ ≤ asin(1.2 / 1.5) = 53° covers the samples 0° and 30°
    up = [a for a in point["aperture"] if a["side"] == "up"][0]
    ring = expected * 2 * math.pi * np.sin(np.radians([0.0, 30.0]))
    assert up["power"] == pytest.approx(0.5 * ring.sum() * math.radians(30.0), rel=3e-2)


def test_stage_a_and_cancellation():
    results = run_job(job("A", kx_over_k0=0.2, beta_over_k0=0.1, moment="z"))
    (point,) = results["points"]
    assert point["stage"] == "A" and point["P_cell"] > 0
    assert point["kx"] == pytest.approx(0.2 * 2 * math.pi / 1e-6)
    assert point["orders_up"] and point["orders_down"]
    calls = []

    def cancel():
        calls.append(1)
        return len(calls) > 3

    cancelled = run_job(job(wavelength=[1000.0, 1100.0]), cancel=cancel)
    assert cancelled["cancelled"] and len(cancelled["points"]) < 2


def test_argument_checks():
    for bad in ({"stage": "D"}, {"moment": "q"}, {"moment": [1.0, 0.0]}):
        doc = job()
        doc["emitter"].update(bad)
        with pytest.raises(JobError):
            run_job(doc)
    doc = copy.deepcopy(job())
    doc["version"] = 1
    with pytest.raises(JobError, match="version 2"):
        run_job(doc)
    with pytest.raises(JobError):
        run_job(job("C"))  # stage C needs the directions

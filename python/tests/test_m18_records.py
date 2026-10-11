"""Regression against the stored M18 3D cross-checks (benchmarks/m18_validation_3d.py, long local
runs, docs/validation.md J): the record agrees with itself (3D at p = 3 against the 2.5D solution
of the same geometry), and the quick 2.5D nanohole reproduces the stored 3D transmission."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

import hpfem

ROOT = Path(__file__).resolve().parents[2]
RECORD = sorted((ROOT / "benchmarks" / "results").glob("*-validation-m18-3d.json"))


def record():
    if not RECORD:
        pytest.skip("no M18 3D record")
    return json.loads(RECORD[-1].read_text(encoding="utf-8"))["runs"]


def relative(a, b):
    return abs(a - b) / abs(b)


def test_stored_3d_cross_checks_agree_with_the_axisymmetric_solution():
    runs = record()
    for wavelength in ("550.0", "650.0"):
        three = runs[f"sphere_on_stack/3D/p3/{wavelength}"]
        axi = runs[f"sphere_on_stack/2.5D/{wavelength}"]
        assert relative(three["sigma_absorption_nm2"], axi["sigma_absorption_nm2"]) < 2e-3
        assert relative(three["sigma_scattering_nm2"], axi["sigma_scattering_nm2"]) < 5e-3
        for channel in ("up", "down", "lateral"):
            assert relative(three[f"sigma_{channel}_nm2"], axi[f"sigma_{channel}_nm2"]) < 5e-3
        for e3, e2 in zip(three["near_field_abs_E"], axi["near_field_abs_E"], strict=True):
            assert relative(e3, e2) < 1e-2
    for wavelength in ("600.0", "750.0", "900.0"):
        three = runs[f"nanohole/3D/p3/{wavelength}"]
        axi = runs[f"nanohole/2.5D/{wavelength}"]
        assert relative(three["T_over_T_geom"], axi["T_over_T_geom"]) < 2e-2
        assert relative(three["absorption_change_over_geom"],
                        axi["absorption_change_over_geom"]) < 3e-2  # fmt: skip
        assert abs(three["disc_stack_over_expected"] - 1) < 1e-4


def test_quick_axisymmetric_nanohole_reproduces_the_stored_3d_transmission():
    runs = record()
    path = ROOT / "examples" / "particle_on_substrate" / "run.py"
    spec = importlib.util.spec_from_file_location("particle_on_substrate_records", path)
    example = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = example
    spec.loader.exec_module(example)
    hpfem.set_log_level("warn")
    row = example.nanohole(True, wavelengths=[750e-9])["spectrum"][0]
    three = runs["nanohole/3D/p3/750.0"]
    assert relative(row["T_over_T_geom"], three["T_over_T_geom"]) < 3e-2
    assert relative(row["absorption_change_over_geom"], three["absorption_change_over_geom"]) < 5e-2

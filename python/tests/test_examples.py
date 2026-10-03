"""The Python example drivers in examples/ run in their quick configuration and give
physically consistent results (regression layer of CLAUDE.md §8 for the M8 examples)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

import hpfem

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def load_example(name: str):
    """Imports examples/<name>/run.py as a module."""
    path = EXAMPLES / name / "run.py"
    spec = importlib.util.spec_from_file_location(f"example_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def quiet():
    hpfem.set_log_level("warn")


def test_metasurface_unitcell_energy_balance_and_phase_coverage(tmp_path, monkeypatch):
    example = load_example("metasurface_unitcell")
    widths = np.array([0.25, 0.5, 0.75]) * example.PERIOD / example.units.nm
    elements = example.sweep(widths, cells_per_period=8, order=2, verbose=False)
    for element in elements:
        assert abs(element.transmission + element.reflection - 1.0) < 1e-2  # lossless
        assert 0 <= element.transmission <= 1.0 + 1e-3
    phases = np.unwrap([e.phase for e in elements])
    assert phases.max() - phases.min() > 0.5 * np.pi  # the ridge delays the wave noticeably
    monkeypatch.chdir(tmp_path)
    assert example.main(["--quick"]) == 0
    assert (tmp_path / "metasurface_unitcell.json").exists()


def test_vcsel_cavity_matches_the_transfer_matrix_pole():
    example = load_example("vcsel_cavity")
    result = example.compare(2, 6, cells_per_layer=2, order=3)
    assert abs(result.wavelength_nm - result.wavelength_tmm_nm) < 1e-3 * result.wavelength_tmm_nm
    assert abs(result.quality - result.quality_tmm) < 2e-2 * result.quality_tmm
    assert abs(result.wavelength_tmm_nm - 850.0) < 0.5  # designed for 850 nm
    more = example.compare(4, 6, cells_per_layer=2, order=3)
    assert more.quality > 1.5 * result.quality  # Q grows with the mirror pairs

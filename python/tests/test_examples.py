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


def test_quantum_dot_purcell_mirror_check_and_spectrum():
    example = load_example("quantum_dot_purcell")
    fem, exact = example.check_mirror(cells=32, order=3, sigma=0.06)
    assert abs(fem - exact) < 1e-2 * exact  # Gaussian current against the image dipole
    wavelengths = np.array([945.0, 950.0, 957.5]) * example.units.nm
    spec = example.spectrum(wavelengths, top_pairs=3, bottom_pairs=6, order=2, cell_fraction=0.1)
    assert max(spec.purcell) > 1.2  # enhancement near the cavity resonance
    assert all(0.0 < b <= 1.0 + 1e-6 for b in spec.beta_top)  # a fraction of the emitted power


def test_euv_mask_bare_mirror_against_the_transfer_matrix(tmp_path, monkeypatch):
    example = load_example("euv_mask")
    monkeypatch.chdir(tmp_path)
    result = example.run(quick=True)
    assert (
        abs(result.reflectivity_bare_fem - result.reflectivity_bare_tmm)
        < 0.1 * result.reflectivity_bare_tmm
    )
    assert (
        0.0 < result.reflectivity < result.reflectivity_bare_fem
    )  # the absorber pad darkens the cell
    assert (tmp_path / "euv_mask.vtu").stat().st_size > 1000


def test_ring_resonator_dip_at_the_eigenmode_resonance():
    example = load_example("ring_resonator")
    cell = 120 * example.units.nm
    res, _, _, _ = example.resonances(order=2, cell=cell, num_modes=1)
    assert len(res.wavelength_nm) == 1 and res.quality[0] > 20
    lam_res = res.wavelength_nm[0] * example.units.nm
    width = lam_res / res.quality[0]
    spec = example.transmission(
        [lam_res - 4 * width, lam_res, lam_res + 4 * width], order=2, cell=cell
    )
    assert spec.transmission[1] < 0.8 * min(spec.transmission[0], spec.transmission[2])  # the dip
    assert all(0.0 < t < 1.05 for t in spec.transmission)


def test_solar_cell_texture_traps_light_and_exports(tmp_path, monkeypatch):
    example = load_example("solar_cell_texture")
    monkeypatch.chdir(tmp_path)
    result = example.run(quick=True)
    for textured, flat in zip(result.absorbed_textured, result.absorbed_flat, strict=True):
        assert 0.0 < flat < 1.0 and 0.0 < textured < 1.0
    assert result.absorbed_textured[0] > result.absorbed_flat[0]  # light trapping at 600 nm
    assert result.generation_total > 0 and result.temperature_rise > 0
    assert (tmp_path / "solar_cell_texture_generation.vtu").stat().st_size > 1000
    profile = np.loadtxt(tmp_path / "solar_cell_texture_profile.csv", delimiter=",")
    assert profile.shape[1] == 2 and profile[:, 1].max() > 0


def test_micropillar_qd_purcell_peaks_at_the_resonance(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    example = load_example("micropillar_qd")
    result = example.run(quick=True)
    res, spec = result["resonance"], result["spectrum"]
    assert res["quality"] > 50
    assert abs(res["wavelength_nm"] - 940.0) < 15.0
    purcell = np.array(spec["purcell"])
    wavelengths = np.array(spec["wavelength_nm"])
    peak = wavelengths[np.argmax(purcell)]
    linewidth = res["wavelength_nm"] / res["quality"]
    assert abs(peak - res["wavelength_nm"]) < 1.5 * linewidth + 0.5 * (
        wavelengths[1] - wavelengths[0]
    )
    assert purcell.max() > 2.0
    assert purcell.max() > 2.0 * purcell.min()
    assert all(0.0 < b < 1.0 for b in spec["beta_top"])
    # the modal sum (Riesz projection on the resonance pencil) follows the sweep: same
    # peak, the fundamental mode carries the peak, every contour converged
    modal = result["modal"]
    modal_purcell = np.array(modal["purcell"])
    assert modal["convergence"] < 1e-3
    assert abs(wavelengths[np.argmax(modal_purcell)] - peak) <= wavelengths[1] - wavelengths[0]
    assert np.max(np.abs(modal_purcell - purcell) / purcell) < 0.1
    # the share of the fundamental mode is a Lorentzian peaking at the resonance on a
    # smooth background of order one (the emission into the leaky continuum)
    mode = np.array(modal["purcell_mode"])
    background = np.array(modal["purcell_background"])
    assert abs(wavelengths[np.argmax(mode)] - peak) <= wavelengths[1] - wavelengths[0]
    assert mode.max() > 0.3 * modal_purcell.max()
    assert mode.max() > 3.0 * mode.min()
    assert np.all(background > 0.5) and background.max() < 2.0 * background.min()

"""hpfem.units conversions and the hpfem.materials library against reference values."""

import numpy as np
import pytest

import hpfem
from hpfem import materials, units


def test_spectral_conversions_round_trip():
    omega = units.angular_frequency(wavelength=500 * units.nm)
    assert np.isclose(units.wavelength(omega) / units.nm, 500)
    assert np.isclose(units.photon_energy(omega) / units.eV, 2.4797, rtol=1e-4)
    assert np.isclose(units.frequency(omega) / units.THz, 599.58, rtol=1e-4)
    assert np.isclose(units.vacuum_wavenumber(omega) * 500 * units.nm, 2 * np.pi)
    assert np.isclose(units.angular_frequency(energy=2.4797 * units.eV), omega, rtol=1e-4)
    assert np.isclose(units.angular_frequency(frequency=599.58 * units.THz), omega, rtol=1e-4)
    assert np.isclose(units.angular_frequency(wavenumber=omega / hpfem.constants.c0), omega)
    lams = np.array([400, 800]) * units.nm
    assert units.angular_frequency(wavelength=lams).shape == (2,)
    with pytest.raises(ValueError):
        units.angular_frequency(wavelength=1e-6, energy=1.0)
    with pytest.raises(ValueError):
        units.angular_frequency()
    assert np.isclose(90 * units.deg, np.pi / 2)


def test_library_values_against_the_sources():
    lam = 589.3 * units.nm  # sodium D
    omega = units.angular_frequency(wavelength=lam)
    assert np.isclose(materials.get("SiO2").refractive_index(omega).real, 1.4584, atol=2e-4)
    assert np.isclose(materials.get("TiO2").refractive_index(omega).real, 2.613, atol=5e-3)
    assert np.isclose(materials.get("water").refractive_index(omega).real, 1.333, atol=1e-3)
    # Johnson & Christy gold around 1.96 eV (632.8 nm): n = 0.18, k = 3.43 (tabulated rows)
    au = materials.get("Au").refractive_index(units.angular_frequency(wavelength=632.8 * units.nm))
    assert abs(au.real - 0.18) < 0.03 and abs(au.imag - 3.43) < 0.1
    si = materials.get("Si").refractive_index(units.angular_frequency(wavelength=500 * units.nm))
    assert abs(si.real - 4.30) < 0.05 and 0.03 < si.imag < 0.06  # alpha ~ 1.1e4 /cm
    for name in ("Au", "Ag", "Al", "Si", "GaAs", "MAPbI3"):
        eps = materials.get(name).eps_r(omega)
        assert eps.imag > 0, name  # lossy: Im eps_r > 0 with exp(-i omega t)
    assert materials.get("Ag").eps_r(omega).real < -10  # noble metal in the visible
    assert (
        materials.get("MAPbI3").eps_r(units.angular_frequency(wavelength=900 * units.nm)).imag < 0.2
    )
    with pytest.raises(ValueError):
        materials.get("Si").eps_r(units.angular_frequency(wavelength=2 * units.um))
    with pytest.raises(ValueError):
        materials.get("TiO2").eps_r(units.angular_frequency(wavelength=300 * units.nm))
    with pytest.raises(KeyError):
        materials.get("unobtainium")
    assert "Johnson" in materials.get("Au").source and materials.get("Au").range[0] < 200 * units.nm


def test_models_and_core_material():
    omega = units.angular_frequency(wavelength=800 * units.nm)
    drude = materials.Drude(1.0, 1.37e16, 1.0e14)
    eps = drude.eps_r(omega)
    assert eps.real < 0 and eps.imag > 0
    dl = materials.DrudeLorentz(1.0, 1.37e16, 1.0e14, [(0.1, 3.0e15, 5.0e14)])
    assert dl.eps_r(omega) != eps
    table = materials.Tabulated([400e-9, 800e-9], [1.5, 1.6], [0.0, 0.1], name="t")
    assert np.isclose(
        table.refractive_index(units.angular_frequency(wavelength=600e-9)), 1.55 + 0.05j
    )
    mat = materials.get("Si").at(omega)
    assert isinstance(mat, hpfem.Material) and mat.eps_r.imag >= 0
    assert np.isclose(materials.get("Si").at_wavelength(800 * units.nm).eps_r, mat.eps_r)
    setup = hpfem.ScatteringSetup2D()
    setup.materials.set(2, materials.get("Au").at(omega))
    assert setup.materials.at(2).eps_r.real < 0
    assert isinstance(materials.plasma_frequency(5.9e28), float)
    with pytest.raises(ValueError):
        materials.Tabulated([800e-9, 400e-9], [1, 1], [0, 0])

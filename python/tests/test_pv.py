"""Carrier generation: per-cell rates sum to the absorbed power over the photon energy, the
depth profile of a damped wave follows Beer-Lambert, spectral weighting and export work."""

import numpy as np
import pytest

import hpfem
from hpfem import pv, units


def damped_wave(nx=8, ny=2, n=complex(3.5, 0.1), k0=4.0e6):
    """Strip [0, 1 um] x [0, 0.25 um] filled with a lossy medium, E_y = exp(i k0 n x)."""
    mesh = hpfem.rectangle(nx, ny, [0.0, 0.0], [1e-6, 0.25e-6])
    mesh_tag = 2
    for c in range(mesh.num_cells):
        mesh.set_cell_tag(c, mesh_tag)
    nd = hpfem.NedelecDofMap2D(mesh, 4)
    materials = hpfem.MaterialMap()
    materials.set(mesh_tag, hpfem.Material(n * n))
    kn = k0 * n
    e = hpfem.interpolate(nd, lambda x: np.array([0.0, np.exp(1j * kn * x[0])]))
    omega = k0 * hpfem.constants.c0
    return mesh, nd, materials, e, omega, n


def test_generation_per_cell_and_beer_lambert_profile():
    mesh, nd, materials, e, omega, n = damped_wave()
    g = pv.generation_per_cell(nd, e, omega, materials)
    assert g.shape == (mesh.num_cells,) and np.all(g > 0)
    total_photons = (g * mesh.cell_volumes).sum()
    absorbed = hpfem.absorbed_power(nd, e, omega, materials)
    assert np.isclose(total_photons * pv.HBAR * omega, absorbed, rtol=1e-10)
    # G(x) ~ exp(-alpha x), alpha = 2 k0 Im(n): the depth profile along x (axis 0)
    centres, profile = pv.depth_profile(mesh, g, axis=0, bins=8)
    alpha = 2 * omega / hpfem.constants.c0 * n.imag
    fitted = np.polyfit(centres, np.log(profile), 1)[0]
    assert abs(fitted + alpha) < 0.05 * alpha
    h1 = hpfem.DofMap2D(mesh, 4)
    field = pv.generation_field(nd, e, omega, materials, h1)
    locator = hpfem.PointLocator2D(mesh)
    value = hpfem.evaluate_h1(h1, field.astype(complex), locator, [0.5e-6, 0.1e-6]).real
    expected = (
        0.5
        * omega
        * hpfem.constants.eps0
        * (n * n).imag
        * np.exp(-alpha * 0.5e-6)
        / (pv.HBAR * omega)
    )
    assert abs(value - expected) < 1e-3 * expected


def test_spectral_weights_and_sum():
    wavelengths = np.array([500, 600, 700]) * units.nm
    irradiance = np.array([1.5, 1.4, 1.2]) * 1e9  # W/(m^2 m)
    weights = pv.spectral_weights(wavelengths, irradiance)
    intensity = hpfem.plane_wave_intensity(1.0, hpfem.Material.vacuum())
    assert np.isclose(weights[1], 1.4e9 * 100e-9 / intensity)
    assert np.isclose(weights[0], 1.5e9 * 100e-9 / intensity)  # end band gets the full spacing
    total = pv.spectral_sum([np.ones(3), 2 * np.ones(3), 3 * np.ones(3)], weights)
    assert np.allclose(total, weights[0] + 2 * weights[1] + 3 * weights[2])
    with pytest.raises(ValueError):
        pv.spectral_weights(wavelengths, irradiance[:2])
    with pytest.raises(ValueError):
        pv.spectral_sum([np.ones(3)], weights)


def test_export(tmp_path):
    pytest.importorskip("meshio")
    mesh, nd, materials, e, omega, _ = damped_wave(nx=4, ny=1)
    g = pv.generation_per_cell(nd, e, omega, materials)
    pv.write_generation(
        tmp_path / "g.vtu",
        mesh,
        g,
        absorbed=np.asarray(hpfem.absorbed_power_per_cell(nd, e, omega, materials)),
    )
    assert (tmp_path / "g.vtu").stat().st_size > 100
    centres, profile = pv.depth_profile(mesh, g, axis=0, bins=4)
    pv.write_profile(tmp_path / "g.csv", centres, profile)
    rows = np.loadtxt(tmp_path / "g.csv", delimiter=",")
    assert rows.shape == (4, 2) and np.allclose(rows[:, 1], profile)

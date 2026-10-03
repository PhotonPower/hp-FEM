"""Layer stacks from Python: Fresnel limit, energy balance, field continuity at an interface."""

import numpy as np
import pytest

import hpfem


def fresnel_p(n0, eps, theta):
    kz0 = n0 * np.cos(theta)
    kz1 = np.sqrt(eps - (n0 * np.sin(theta)) ** 2 + 0j)
    return abs((eps * kz0 - kz1) / (eps * kz0 + kz1)) ** 2


def test_layer_stack_fresnel_and_balance():
    k0 = 2 * np.pi / 600e-9
    silver = hpfem.Material(-15 + 1.2j, 1.0)
    half = hpfem.LayerStack2D(hpfem.Material.vacuum(), [], silver)
    wave = half.plane_wave(k0, 0.3)
    assert wave.reflectance == pytest.approx(fresnel_p(1.0, -15 + 1.2j, 0.3), rel=1e-12)
    assert wave.reflectance + wave.transmittance + wave.absorptance == pytest.approx(1.0)
    assert half.num_layers == 0 and half.region(-1e-9) == 1
    film = hpfem.LayerStack3D(
        hpfem.Material.vacuum(), [hpfem.Layer(silver, 400e-9)], hpfem.Material.dielectric(1.5)
    )
    for pol in (hpfem.Polarisation.S, hpfem.Polarisation.P):
        w = film.plane_wave(k0, 0.2, pol)
        assert w.absorptance > 0
        assert w.reflectance + w.transmittance + w.absorptance == pytest.approx(1.0, abs=1e-12)
        # tangential E continuous across the silver / substrate interface
        z = film.interface(1)
        e_above = w.field.value(np.array([0.1e-6, 0.0, z + 1e-15]))
        e_below = w.field.value(np.array([0.1e-6, 0.0, z - 1e-15]))
        assert np.allclose(e_above[:2], e_below[:2], rtol=1e-6, atol=1e-9 * np.linalg.norm(e_above))
        assert len(w.kz) == 3 and len(w.down) == 3
    with pytest.raises(ValueError):
        half.plane_wave(k0, 0.3, hpfem.Polarisation.S)

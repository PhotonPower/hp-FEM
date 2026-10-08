"""hpfem.sweep.solve_sweep: the points of a sweep in worker processes give the same results
as the loop in place."""

import numpy as np
import pytest

import hpfem
from hpfem import sweep

c0 = hpfem.constants.c0


def reflectance_of_slab(wavelength):
    """A picklable task: the reflectance of a thin film from the layer-stack solver."""
    k0 = 2 * np.pi / wavelength
    film = hpfem.Material(2.25 + 0.0j, 1.0)
    stack = hpfem.LayerStack2D(
        hpfem.Material.vacuum(), [hpfem.Layer(film, 0.2e-6)], hpfem.Material.dielectric(1.5)
    )
    return stack.plane_wave(k0, 0.3).reflectance


def test_solve_sweep_in_processes_matches_the_loop():
    wavelengths = [0.5e-6, 0.55e-6, 0.6e-6, 0.65e-6, 0.7e-6]
    in_place = sweep.solve_sweep(reflectance_of_slab, wavelengths, processes=1)
    assert in_place == [reflectance_of_slab(w) for w in wavelengths]
    pooled = sweep.solve_sweep(reflectance_of_slab, wavelengths, processes=2)
    assert pooled == in_place
    assert sweep.solve_sweep(reflectance_of_slab, [], processes=3) == []
    with pytest.raises(ValueError):
        sweep.solve_sweep(reflectance_of_slab, wavelengths, processes=0)

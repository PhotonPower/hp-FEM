"""Slab waveguide (convergence test #5 at small size): the fundamental TE mode's effective
index matches the root of the transcendental equation."""

import numpy as np
from conftest import BOX_SIDES

import hpfem

THICKNESS, CORE, CLAD, K0 = 1.0, 1.5, 1.0, 2.0


def slab_te_even():
    def f(n):
        kappa = K0 * np.sqrt(CORE**2 - n**2)
        gamma = K0 * np.sqrt(n**2 - CLAD**2)
        return kappa * np.tan(kappa * THICKNESS / 2) - gamma

    lo, hi = CLAD + 1e-12, CORE - 1e-12
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def test_effective_index_of_the_fundamental_mode():
    half_length = 12.0
    mesh = hpfem.rectangle(96, 2, [-half_length, 0.0], [half_length, 0.5])
    for c, x in enumerate(mesh.cell_centroids):
        if abs(x[0]) < THICKNESS / 2:
            mesh.set_cell_tag(c, 2)
    nd = hpfem.NedelecDofMap2D(mesh, 3)
    h1 = hpfem.DofMap2D(mesh, 3)
    setup = hpfem.WaveguideSetup()
    setup.omega = K0 * hpfem.constants.c0
    setup.materials.set(2, hpfem.Material.dielectric(CORE))
    setup.pec_tags = BOX_SIDES
    setup.num_modes = 2
    problem = hpfem.PropagatingMode(nd, h1, setup)
    assert np.isclose(problem.max_index, CORE)
    modes = problem.solve()
    assert len(modes) >= 1
    assert abs(modes[0].effective_index - slab_te_even()) < 1e-6
    # V = 1.12: a single guided TE mode; anything else is a box mode below the cladding index
    assert all(mode.effective_index < CLAD for mode in modes[1:])
    assert np.isclose(modes[0].beta, modes[0].effective_index * K0)
    assert modes[0].transverse.shape == (nd.num_dofs,)
    assert modes[0].longitudinal.shape == (h1.num_dofs,)

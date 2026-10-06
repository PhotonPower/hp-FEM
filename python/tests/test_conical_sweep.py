"""Sweep acceleration (M15 F8): ConicalSweep reproduces the per-point solves of a wavelength
sweep with a dispersive ridge and changing Bloch phases; LinearSolver.refactorize."""

import numpy as np
import scipy.sparse

import hpfem

c0 = hpfem.constants.c0
UM = 1e-6
PERIOD = 0.5 * UM


def grating_setup(stack, wavelength, angle, azimuth):
    k0 = 2 * np.pi / wavelength
    wave = hpfem.layered_conical_wave(stack, k0, angle, azimuth, hpfem.Polarisation.P)
    lam = wavelength / UM
    setup = hpfem.ConicalScatteringSetup()
    setup.omega = k0 * c0
    setup.beta = wave.beta
    setup.materials.set(2, hpfem.Material.dielectric(1.5))
    setup.materials.set(3, hpfem.Material(4.0 + 2.0 * (lam - 0.6) + 0.3j * lam, 1.0))
    setup.background = stack
    setup.incident = wave.field
    setup.pml = hpfem.PmlBox2D(
        [0.0, -0.5 * UM], [PERIOD, 0.25 * UM], [0.0, 0.0, 0.375 * UM, 0.375 * UM], k0
    )
    setup.pec_tags = [hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX]
    setup.periodic = [
        hpfem.PeriodicPair2D(
            hpfem.box_tag.X_MIN, hpfem.box_tag.X_MAX, [PERIOD, 0.0],
            hpfem.bloch_phase([wave.kx, 0.0], [PERIOD, 0.0]),
        )
    ]  # fmt: skip
    return setup


def test_conical_sweep_matches_the_per_point_solves():
    mesh = hpfem.rectangle(4, 12, [0.0, -0.875 * UM], [PERIOD, 0.625 * UM])
    for c, x in enumerate(mesh.cell_centroids):
        if x[1] < 0:
            mesh.set_cell_tag(c, 2)
        elif x[1] < 0.125 * UM and x[0] < 0.5 * PERIOD:
            mesh.set_cell_tag(c, 3)
    nd = hpfem.NedelecDofMap2D(mesh, 2)
    h1 = hpfem.DofMap2D(mesh, 2)
    stack = hpfem.LayerStack2D(hpfem.Material.vacuum(), [], hpfem.Material.dielectric(1.5))
    sweep = hpfem.ConicalSweep(nd, h1, grating_setup(stack, 0.6 * UM, 0.4, 0.3))
    assert sweep.num_groups == 3 and sweep.num_pml_cells > 0 and sweep.num_source_cells > 0
    for wavelength, angle in ((0.6 * UM, 0.4), (0.55 * UM, 0.7), (0.7 * UM, 0.4)):
        setup = grating_setup(stack, wavelength, angle, 0.3)
        direct = hpfem.ConicalScattering(nd, h1, setup).solve()
        swept = sweep.solve(setup)
        scale = np.hypot(np.linalg.norm(direct.transverse), np.linalg.norm(direct.longitudinal))
        assert np.linalg.norm(swept.transverse - direct.transverse) < 1e-8 * scale
        assert np.linalg.norm(swept.longitudinal - direct.longitudinal) < 1e-8 * scale
        assert swept.beta == setup.beta and swept.scattered
    assert sweep.timings.points == 3
    assert sweep.timings.total() > sweep.timings.setup > 0
    assert sweep.solver.name


def test_refactorize_matches_a_fresh_factorisation():
    n = 300
    rng = np.random.default_rng(5)
    rows = np.repeat(np.arange(n), 3)
    cols = (rows + rng.integers(0, n, rows.size)) % n
    rows = np.concatenate([rows, np.arange(n)])
    cols = np.concatenate([cols, np.arange(n)])

    def matrix(seed):
        g = np.random.default_rng(seed)
        values = g.standard_normal(rows.size) + 1j * g.standard_normal(rows.size)
        values[-n:] += 10.0  # dominant diagonal
        return scipy.sparse.csr_matrix((values, (rows, cols)), shape=(n, n))

    a1, a2 = matrix(1), matrix(2)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    for backend in hpfem.available_backends():
        solver = hpfem.make_direct_solver(backend)
        solver.factorize(a1)
        assert np.allclose(solver.solve(a1 @ x), x, atol=1e-10 * np.linalg.norm(x))
        solver.refactorize(a2)
        assert np.allclose(solver.solve(a2 @ x), x, atol=1e-10 * np.linalg.norm(x))
        fresh = hpfem.make_direct_solver(backend)
        fresh.factorize(a2)
        assert np.allclose(
            solver.solve(a2 @ x), fresh.solve(a2 @ x), atol=1e-12 * np.linalg.norm(x)
        )

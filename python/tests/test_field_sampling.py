"""Vectorised field sampling (M15 F3): identical to the point-wise loop, much faster, Bloch
wrapping, NaN outside, the triangulated field for matplotlib; Scattering2D and
ConicalScattering."""

import math
import time

import numpy as np
import pytest

import hpfem

c0 = hpfem.constants.c0


def grating_problem(p=2):
    """Bloch-periodic vacuum cell [0, 1] x [-1, 1] with a dielectric below y = 0."""
    k0 = 2 * np.pi / 0.6
    angle = np.pi / 6
    mesh = hpfem.rectangle(4, 16, [0.0, -1.0], [1.0, 1.0])
    for c, x in enumerate(mesh.cell_centroids):
        if x[1] < 0:
            mesh.set_cell_tag(c, 2)
    dofs = hpfem.NedelecDofMap2D(mesh, p)
    k = np.array([k0 * np.sin(angle), -k0 * np.cos(angle)])
    setup = hpfem.ScatteringSetup2D()
    setup.omega = k0 * c0
    setup.materials.set(2, hpfem.Material.dielectric(1.5))
    setup.incident = hpfem.plane_wave([np.cos(angle), np.sin(angle)], k)
    setup.formulation = hpfem.Formulation.SCATTERED_FIELD
    setup.pml = hpfem.PmlBox2D(
        [0.0, -0.6], [1.0, 0.6], [0.0, 0.0, 0.4, 0.4], k0, 1.0, hpfem.PmlProfile(2, 1e-8)
    )
    setup.pec_tags = [hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX]
    shift = np.array([1.0, 0.0])
    setup.periodic = [
        hpfem.PeriodicPair2D(
            hpfem.box_tag.X_MIN, hpfem.box_tag.X_MAX, shift, hpfem.bloch_phase(k, shift)
        )
    ]
    problem = hpfem.Scattering2D(dofs, setup)
    return mesh, problem, problem.solve()


def test_sample_matches_the_loop_and_is_much_faster():
    mesh, problem, solution = grating_problem()
    locator = hpfem.PointLocator2D(mesh)
    rng = np.random.default_rng(1)
    n = 20000
    points = np.column_stack([rng.uniform(0.0, 1.0, n), rng.uniform(-1.0, 1.0, n)])
    problem.sample(solution, locator, points[:100])  # warm-up (thread pool, first-call costs)
    fast = math.inf
    for _ in range(3):  # best of three: the CI runners are shared
        start = time.perf_counter()
        values, cells = problem.sample(solution, locator, points)
        fast = min(fast, time.perf_counter() - start)
    assert values.shape == (n, 2) and cells.shape == (n,)
    assert np.all(cells >= 0)
    m = 2000
    start = time.perf_counter()
    loop = np.array([problem.total_field(solution, locator, x) for x in points[:m]])
    slow = (time.perf_counter() - start) * n / m
    assert np.allclose(values[:m], loop, rtol=0, atol=1e-12 * np.abs(loop).max())
    print(
        f"sample: {fast * 1e3:.1f} ms for {n} points, loop {slow * 1e3:.0f} ms, {slow / fast:.0f}x"
    )
    # measured 45-80x with 16 threads and 10x single-threaded on the development machine, but
    # only 4.6-7.9x on the shared 4-core CI runners (2026-10-09, three runs), where the wall
    # clock of both paths depends on the load; 3x still separates the vectorised path from a
    # Python loop over points
    assert slow / fast > 3
    # scattered field, Bloch wrapping and NaN outside
    scattered, _ = problem.sample(solution, locator, points[:m], scattered=True)
    incident = np.array([problem.setup.incident.value(x) for x in points[:m]])
    assert np.allclose(scattered + incident, values[:m], atol=1e-12)
    phase = problem.setup.periodic[0].phase
    wrapped, wrapped_cells = problem.sample(solution, locator, points[:m] + [2.0, 0.0])
    assert np.allclose(wrapped, phase**2 * values[:m], atol=1e-12 * np.abs(loop).max())
    assert np.array_equal(wrapped_cells, cells[:m])
    outside, outside_cells = problem.sample(
        solution, locator, np.array([[1.5, 0.0], [0.5, 1.5]]), bloch_wrap=False
    )
    assert np.all(np.isnan(outside)) and np.all(outside_cells == -1)
    # the interface side: the cell above or below y = 0
    above = problem.sample(solution, locator, np.array([[0.37, 0.0]]), interface_side=+1)
    below = problem.sample(solution, locator, np.array([[0.37, 0.0]]), interface_side=-1)
    assert mesh.cell_centroid(int(above[1][0]))[1] > 0 > mesh.cell_centroid(int(below[1][0]))[1]
    assert abs(above[0][0, 0] - below[0][0, 0]) < 1e-10 * np.abs(loop).max()  # tangential
    with pytest.raises(ValueError):
        problem.sample(solution, locator, np.zeros((3, 3)))


def test_triangulate_for_matplotlib():
    mesh, problem, solution = grating_problem()
    tri = problem.triangulate(solution, 3)
    assert tri.points.shape == (6 * mesh.num_cells + 4 * mesh.num_cells, 2)
    assert tri.simplices.shape == (9 * mesh.num_cells, 3)
    assert tri.values.shape == (tri.points.shape[0], 2) and tri.values.dtype == np.complex128
    assert tri.cell.shape == (9 * mesh.num_cells,) and tri.tag.shape == tri.cell.shape
    assert set(tri.tag.tolist()) == {0, 2}
    assert len(tri) == 9 * mesh.num_cells
    # every sub-vertex value is the field of its parent cell at that point
    locator = hpfem.PointLocator2D(mesh)
    for s in range(0, len(tri), 37):
        for v in tri.simplices[s]:
            cell = int(tri.cell[s])
            xi = locator.reference_coordinates(cell, tri.points[v])
            assert xi is not None
            assert np.allclose(tri.values[v], problem.total_field(solution, cell, xi), atol=1e-12)
    mtri = pytest.importorskip("matplotlib.tri")
    triangulation = mtri.Triangulation(tri.points[:, 0], tri.points[:, 1], tri.simplices)
    assert triangulation.triangles.shape == tri.simplices.shape


def test_conical_sample_and_triangulate():
    k0, n2, period = 2 * np.pi, 1.6, 0.7
    angle, azimuth = np.radians(35.0), np.radians(50.0)
    stack = hpfem.LayerStack2D(
        hpfem.Material.dielectric(1.0), [], hpfem.Material.dielectric(n2), 0.0
    )
    mesh = hpfem.rectangle(4, 24, [0.0, -2.0], [period, 2.0])
    for c in range(mesh.num_cells):
        if mesh.cell_centroid(c)[1] < 0:
            mesh.set_cell_tag(c, 2)
    nd = hpfem.NedelecDofMap2D(mesh, 2)
    h1 = hpfem.DofMap2D(mesh, 2)
    wave = hpfem.layered_conical_wave(stack, k0, angle, azimuth, hpfem.Polarisation.P)
    setup = hpfem.ConicalScatteringSetup()
    setup.omega = k0 * c0
    setup.beta = wave.beta
    setup.materials.set(2, hpfem.Material.dielectric(n2))
    setup.background = stack
    setup.incident = wave.field
    setup.pml = hpfem.PmlBox2D([0.0, -1.0], [period, 1.0], [0.0, 0.0, 1.0, 1.0], k0)
    setup.pec_tags = [hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX]
    setup.periodic = [
        hpfem.PeriodicPair2D(
            hpfem.box_tag.X_MIN, hpfem.box_tag.X_MAX, [period, 0.0],
            hpfem.bloch_phase([wave.kx, 0.0], [period, 0.0]),
        )
    ]  # fmt: skip
    problem = hpfem.ConicalScattering(nd, h1, setup)
    solution = problem.solve()
    locator = hpfem.PointLocator2D(mesh)
    points = np.column_stack([np.linspace(0.05, 0.65, 30), np.linspace(-0.9, 0.9, 30)])
    values, cells = problem.sample(solution, locator, points)
    assert values.shape == (30, 3)
    loop = np.array([problem.total_field(solution, locator, x) for x in points])
    assert np.allclose(values, loop, atol=1e-12 * np.abs(loop).max())
    wrapped, _ = problem.sample(solution, locator, points - [period, 0.0])
    assert np.allclose(wrapped, values / setup.periodic[0].phase, atol=1e-12 * np.abs(loop).max())
    tri = problem.triangulate(solution, 2)
    assert tri.values.shape == (6 * mesh.num_cells, 3)
    assert tri.simplices.shape == (4 * mesh.num_cells, 3)

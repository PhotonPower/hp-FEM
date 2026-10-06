"""Absorbed power by volumetric quadrature (M15 F4): per tag, per cell and per quadrature
point, against the closed-form absorptance of a flat lossy film on a layered background,
for Scattering2D and ConicalScattering."""

import numpy as np
import pytest

import hpfem

c0 = hpfem.constants.c0
UM = 1e-6
PERIOD = 1.0 * UM
FILM = hpfem.Material(2.0 + 0.5j, 1.0)
GLASS = hpfem.Material.dielectric(1.5)


def film_mesh():
    mesh = hpfem.rectangle(4, 16, [0.0, -1.65 * UM], [PERIOD, 0.75 * UM])
    for c, x in enumerate(mesh.cell_centroids):
        if -0.15 * UM < x[1] < 0:
            mesh.set_cell_tag(c, 2)
        elif x[1] < -0.15 * UM:
            mesh.set_cell_tag(c, 3)
    return mesh


def pml(k0):
    return hpfem.PmlBox2D(
        [0.0, -1.2 * UM], [PERIOD, 0.45 * UM], [0.0, 0.0, 0.45 * UM, 0.3 * UM], k0, 1.0,
        hpfem.PmlProfile(2, 1e-10),
    )  # fmt: skip


def incident_power(angle):
    return hpfem.plane_wave_intensity(1.0, hpfem.Material.vacuum()) * np.cos(angle) * PERIOD


def test_absorbed_power_by_tag_scattering():
    k0 = 2 * np.pi / (0.6 * UM)
    angle = 0.35
    mesh = film_mesh()
    stack = hpfem.LayerStack2D(hpfem.Material.vacuum(), [hpfem.Layer(FILM, 0.15 * UM)], GLASS)
    wave = stack.plane_wave(k0, angle)
    dofs = hpfem.NedelecDofMap2D(mesh, 4)
    setup = hpfem.ScatteringSetup2D()
    setup.omega = k0 * c0
    setup.materials.set(2, FILM)
    setup.materials.set(3, GLASS)
    setup.background = stack
    setup.incident = wave.field
    setup.formulation = hpfem.Formulation.SCATTERED_FIELD
    setup.pml = pml(k0)
    setup.pec_tags = [hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX]
    k = np.array([k0 * np.sin(angle), -k0 * np.cos(angle)])
    shift = np.array([PERIOD, 0.0])
    setup.periodic = [
        hpfem.PeriodicPair2D(
            hpfem.box_tag.X_MIN, hpfem.box_tag.X_MAX, shift, hpfem.bloch_phase(k, shift)
        )
    ]
    problem = hpfem.Scattering2D(dofs, setup)
    solution = problem.solve()
    absorbed = hpfem.absorbed_power_by_tag(problem, solution, 6)
    assert set(absorbed.by_tag) == {2}
    assert absorbed.of_tag(2) == absorbed.total and absorbed.of_tag(3) == 0.0
    assert abs(absorbed.total / incident_power(angle) - wave.absorptance) < 2e-3
    assert absorbed.per_cell.shape == (mesh.num_cells,)
    assert np.isclose(absorbed.per_cell.sum(), absorbed.total, rtol=1e-12)
    assert np.all(absorbed.per_cell[mesh.cell_tags != 2] == 0.0)
    assert np.isclose(hpfem.absorbed_power(problem, solution, 6), absorbed.total, rtol=1e-12)
    density = hpfem.absorption_density(problem, solution, 6)
    assert np.isclose(density.total(), absorbed.total, rtol=1e-12)
    assert density.points.shape == (len(density), 2)
    assert np.all(density.density > 0) and np.all(mesh.cell_tags[density.cell] == 2)
    assert np.isclose(np.dot(density.weights, density.density), absorbed.total, rtol=1e-12)


def test_absorbed_power_by_tag_conical():
    k0 = 2 * np.pi / (0.6 * UM)
    angle, azimuth = 0.35, 0.6
    mesh = film_mesh()
    stack = hpfem.LayerStack2D(hpfem.Material.vacuum(), [hpfem.Layer(FILM, 0.15 * UM)], GLASS)
    nd = hpfem.NedelecDofMap2D(mesh, 4)
    h1 = hpfem.DofMap2D(mesh, 4)
    wave = hpfem.layered_conical_wave(stack, k0, angle, azimuth, hpfem.Polarisation.S)
    absorptance = 1.0 - wave.reflectance - wave.transmittance
    setup = hpfem.ConicalScatteringSetup()
    setup.omega = k0 * c0
    setup.beta = wave.beta
    setup.materials.set(2, FILM)
    setup.materials.set(3, GLASS)
    setup.background = stack
    setup.incident = wave.field
    setup.pml = pml(k0)
    setup.pec_tags = [hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX]
    setup.periodic = [
        hpfem.PeriodicPair2D(
            hpfem.box_tag.X_MIN, hpfem.box_tag.X_MAX, [PERIOD, 0.0],
            hpfem.bloch_phase([wave.kx, 0.0], [PERIOD, 0.0]),
        )
    ]  # fmt: skip
    problem = hpfem.ConicalScattering(nd, h1, setup)
    solution = problem.solve()
    absorbed = hpfem.absorbed_power_by_tag(problem, solution, 6)
    assert set(absorbed.by_tag) == {2}
    assert abs(absorbed.total / incident_power(angle) - absorptance) < 2e-3
    density = hpfem.absorption_density(problem, solution, 6)
    assert np.isclose(density.total(), absorbed.total, rtol=1e-12)
    with pytest.raises(TypeError):
        hpfem.absorbed_power_by_tag(problem, "no solution")

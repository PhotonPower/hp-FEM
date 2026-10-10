"""Post-processing of the axisymmetric solver on a layered background from Python (ADR-0014,
M18 S2): the bare stack's transmission through a disc and absorption in a cylinder summed over
the orders, and a hole in an absorbing film (absorbed power, change, channels)."""

import numpy as np
import pytest

import hpfem

K0 = 2 * np.pi


def film_stack():
    metal = hpfem.Material(-4.0 + 1.5j, 1.0)
    return hpfem.LayerStack3D(
        hpfem.Material.vacuum(),
        [hpfem.Layer(hpfem.Material.dielectric(1.5), 0.3), hpfem.Layer(metal, 0.1)],
        hpfem.Material.dielectric(1.45),
        0.0,
    )


def film_problem(hole, p=2):
    mesh = hpfem.rectangle(20, 30, [0.0, -1.5], [2.0, 1.5])
    s = film_stack()
    for c in range(mesh.num_cells):
        r, z = mesh.cell_centroid(c)
        region = s.region(z)
        mesh.set_cell_tag(c, 5 if hole and region == 2 and r < 0.3 else 10 + region)
    nd = hpfem.NedelecDofMap2D(mesh, p)
    h1 = hpfem.DofMap2D(mesh, p)
    setup = hpfem.AxisymmetricScatteringSetup()
    setup.omega = K0 * hpfem.constants.c0
    for j in range(s.num_layers + 2):
        setup.materials.set(10 + j, s.material(j))
    setup.materials.set(5, hpfem.Material.vacuum())
    setup.axis_tag = hpfem.box_tag.X_MIN
    setup.pml = hpfem.PmlBox2D([0.0, -1.0], [1.5, 1.0], [0.0, 0.5, 0.5, 0.5], K0)
    setup.background = s
    return mesh, nd, h1, setup, s


def test_bare_stack_disc_and_cylinder_sum_to_t_and_a():
    mesh, nd, h1, setup, s = film_problem(hole=False)
    theta, radius = 0.6, 1.0
    zero_e, zero_v = np.zeros(nd.num_dofs, complex), np.zeros(h1.num_dofs, complex)
    inside = np.array([mesh.cell_centroid(c)[0] < radius for c in range(mesh.num_cells)])
    transmitted = absorbed = 0.0
    for m in range(-12, 13):
        wave = hpfem.layered_axisymmetric_wave(s, K0, theta, "s", m)
        disc = hpfem.axisymmetric_disc_flux(
            nd, h1, zero_e, zero_v, m, setup.omega, setup.materials, -0.8, radius, -1,
            wave.value, wave.curl, 16,
        )  # fmt: skip
        assert disc.change() == pytest.approx(0.0, abs=1e-20)
        transmitted += disc.background
        power = hpfem.axisymmetric_absorbed_power(
            nd, h1, zero_e, zero_v, m, setup.omega, setup.materials, wave.value, setup.pml, 14
        )
        absorbed += np.asarray(power.per_cell)[inside].sum()
    reference = hpfem.layered_axisymmetric_wave(s, K0, theta, "s", 0)
    incident = np.cos(theta) * np.pi * radius**2 / (2 * hpfem.constants.Z0)
    # |m| <= 12 > k_rho R: the remaining orders are below 1e-5
    assert transmitted == pytest.approx(reference.transmittance * incident, rel=1e-4)
    assert absorbed == pytest.approx(reference.absorptance * incident, rel=1e-4)


def test_hole_in_a_film_absorption_change_and_channels():
    mesh, nd, h1, setup, s = film_problem(hole=True)
    wave = hpfem.layered_axisymmetric_wave(s, K0, 0.0, "p", 1)
    setup.azimuthal_order = 1
    setup.incident = wave.value
    problem = hpfem.AxisymmetricScattering(nd, h1, setup)
    field = problem.solve()
    body = problem.scatterer_cells()
    assert len(body) == 6 and all(mesh.cell_tag(int(c)) == 5 for c in body)
    total = np.asarray(problem.absorbed_power(field).per_cell)
    stack_only = np.asarray(problem.incident_absorbed_power().per_cell)
    assert total.sum() > 0 and stack_only.sum() > 0
    for c in range(mesh.num_cells):
        r, z = mesh.cell_centroid(c)
        mesh.set_cell_tag(c, 99 if r < 1.0 and -0.6 < z < 0.5 else mesh.cell_tag(c))
    surface = hpfem.Surface2D.around_cells(mesh, 99)
    channels = hpfem.axisymmetric_flux_channels(
        nd, h1, field.meridian, field.azimuthal, 1, setup.omega, setup.materials, surface, s
    )
    scattered = hpfem.axisymmetric_poynting_flux(
        nd, h1, field.meridian, field.azimuthal, 1, setup.omega, setup.materials, surface
    )
    assert channels.total() == pytest.approx(scattered, rel=1e-12)
    assert channels.up > 0 and channels.down > 0

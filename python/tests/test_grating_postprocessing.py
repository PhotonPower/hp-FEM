"""Grating post-processing from Python (M14-C): diffraction orders of an analytic
quasi-periodic field on a rotated line, the incident wave of a layered background, planar
surfaces and the power balance of a flat stack."""

import numpy as np
import pytest

import hpfem


def test_diffraction_orders_of_an_analytic_field():
    k = 2 * np.pi * 1.3
    kt0 = 0.2 * k
    rotation = 0.4
    tangent = np.array([np.cos(rotation), np.sin(rotation)])
    normal = np.array([-np.sin(rotation), np.cos(rotation)])
    origin = np.array([0.3, -0.7])
    amplitudes = {
        m: np.array([0.5 + 0.1 * m + 0.2j * m, -0.3 * m + (0.7 - 0.1 * m) * 1j])
        for m in range(-2, 3)
    }
    incident_amplitude = np.array([0.8, 0.6j])

    def kn(m):
        kt = kt0 + 2 * np.pi * m
        return np.sqrt(complex(k * k - kt * kt))

    def incident(x):
        t, n = np.dot(x - origin, tangent), np.dot(x - origin, normal)
        return incident_amplitude * np.exp(1j * (kt0 * t - kn(0) * n))

    def total(x):
        t, n = np.dot(x - origin, tangent), np.dot(x - origin, normal)
        e = incident(x)
        for m, a in amplitudes.items():
            e = e + a * np.exp(1j * ((kt0 + 2 * np.pi * m) * t + kn(m) * n))
        return e

    line = hpfem.OrderLine(origin, tangent, normal, 1.0)
    orders = hpfem.diffraction_orders(total, line, k, 1.0, kt0, kn(0).real, incident, 2)
    assert len(orders) == 5
    for o in orders:
        assert np.linalg.norm(o.amplitude - amplitudes[o.order]) < 1e-12
        assert o.propagating == (abs(o.order) <= 1)
        if o.propagating:
            expected = o.kn.real * np.linalg.norm(amplitudes[o.order]) ** 2 / kn(0).real
            assert abs(o.efficiency - expected) < 1e-12
    with pytest.raises(ValueError):
        bad = hpfem.OrderLine(origin, tangent, tangent, 1.0)
        hpfem.diffraction_orders(total, bad, k, 1.0, kt0, 1.0)


def test_layered_background_incident_wave_planes_and_balance():
    um = 1e-6
    k0 = 2 * np.pi / (0.852 * um)
    film = hpfem.Material(6.0 + 0j, 1.0)
    glass = hpfem.Material.dielectric(1.5)
    stack = hpfem.LayerStack2D(hpfem.Material.vacuum(), [hpfem.Layer(film, 0.2 * um)], glass)
    mesh = hpfem.rectangle(5, 12, [0.0, -1.6 * um], [1.0 * um, 0.8 * um])
    for c, x in enumerate(mesh.cell_centroids):
        if -0.2 * um < x[1] < 0:
            mesh.set_cell_tag(c, 2)
        elif x[1] < -0.2 * um:
            mesh.set_cell_tag(c, 3)
    dofs = hpfem.NedelecDofMap2D(mesh, 3)
    angle = 0.3
    wave = stack.plane_wave(k0, angle)
    setup = hpfem.ScatteringSetup2D()
    setup.omega = k0 * hpfem.constants.c0
    setup.materials.set(2, film)
    setup.materials.set(3, glass)
    setup.background = stack
    setup.incident = wave.field
    setup.incident_wave = wave.incident_wave
    setup.formulation = hpfem.Formulation.SCATTERED_FIELD
    setup.pml = hpfem.PmlBox2D(
        [0.0, -1.2 * um],
        [1.0 * um, 0.4 * um],
        [0.0, 0.0, 0.4 * um, 0.4 * um],
        k0,
        1.0,
        hpfem.PmlProfile(2, 1e-12),
    )
    setup.pec_tags = [hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX]
    problem = hpfem.Scattering2D(dofs, setup)
    x = np.array([0.37 * um, 0.3 * um])
    assert abs(np.linalg.norm(problem.incident_wave(x)) - 1.0) < 1e-9
    solution = problem.solve()
    reflection = hpfem.Surface2D.plane(mesh, 1, 0.2 * um, +1)
    transmission = hpfem.Surface2D.plane(mesh, 1, -0.6 * um, -1)
    assert len(reflection) == 5 and len(transmission) == 5
    with pytest.raises(ValueError):
        hpfem.Surface2D.plane(mesh, 1, 0.137 * um, +1)
    kn = k0 * np.cos(angle)
    balance = hpfem.power_balance(problem, solution, reflection, 1.0 * um, kn, 1.0, transmission)
    assert abs(balance.reflected / balance.incident - wave.reflectance) < 2e-2
    assert abs(balance.transmitted / balance.incident - wave.transmittance) < 2e-2
    assert balance.absorbed == 0.0
    assert abs(balance.relative_residual()) < 3e-2
    assert hpfem.absorbed_power(problem, solution) == 0.0
    # the reflected orders of the analytic stack field: order 0 carries the stack reflectance
    line = hpfem.OrderLine([0.0, 0.2 * um], [1.0, 0.0], [0.0, 1.0], 1.0 * um)
    exact = hpfem.diffraction_orders(
        wave.field.value, line, k0, 1.0, k0 * np.sin(angle), kn, wave.incident_wave.value, 1
    )
    assert abs(exact[1].efficiency - wave.reflectance) < 1e-10
    locator = hpfem.PointLocator2D(mesh)
    orders = hpfem.diffraction_orders(
        lambda p: problem.total_field(solution, locator, p),
        line,
        k0,
        1.0,
        k0 * np.sin(angle),
        kn,
        problem.incident_wave,
        1,
    )
    assert abs(orders[1].efficiency - wave.reflectance) < 2e-2

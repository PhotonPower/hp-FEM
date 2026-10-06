"""H field and Poynting vector of the conical solution (M15 F12): closed form of a plane wave,
the analytic curls of the built-in waves, sampling of H and S."""

import numpy as np
import pytest

import hpfem

c0 = hpfem.constants.c0
Z0 = hpfem.constants.Z0
mu0 = hpfem.constants.mu0


def test_plane_wave_h_and_poynting():
    mesh = hpfem.rectangle(3, 3)
    nd = hpfem.NedelecDofMap2D(mesh, 2)
    h1 = hpfem.DofMap2D(mesh, 2)
    k0 = 2.0
    k = np.array([1.0, 1.0, np.sqrt(k0**2 - 2.0)])
    e0 = hpfem.conical_polarisation(k, [1.0, 0.0, 0.0], hpfem.Polarisation.P)
    setup = hpfem.ConicalScatteringSetup()
    setup.omega = k0 * c0
    setup.beta = k[2]
    setup.pec_tags = [
        hpfem.box_tag.X_MIN,
        hpfem.box_tag.X_MAX,
        hpfem.box_tag.Y_MIN,
        hpfem.box_tag.Y_MAX,
    ]
    setup.incident = hpfem.conical_plane_wave(e0, k)
    by_differences = hpfem.ConicalScattering(nd, h1, setup)
    setup.incident_curl = hpfem.conical_plane_wave_curl(e0, k)
    problem = hpfem.ConicalScattering(nd, h1, setup)
    solution = problem.solve()
    locator = hpfem.PointLocator2D(mesh)
    x = np.array([0.2, 0.3])
    phase = np.exp(1j * (k[0] * x[0] + k[1] * x[1]))
    curl = 1j * np.cross(k, e0) * phase
    assert np.allclose(problem.incident_curl(x), curl, rtol=1e-12)
    assert np.allclose(by_differences.incident_curl(x), curl, rtol=1e-6)
    h = np.cross(k, e0) * phase / (setup.omega * mu0)
    assert np.allclose(problem.h_field(solution, locator, x), h, rtol=1e-9)
    assert np.allclose(problem.incident_h_field(x), h, rtol=1e-12)
    s = np.linalg.norm(e0) ** 2 / (2 * Z0) * k / np.linalg.norm(k)
    assert np.allclose(problem.poynting(solution, locator, x), s, rtol=1e-9)
    hit = locator.locate(x)
    assert np.allclose(problem.poynting(solution, hit.cell, hit.xi), s, rtol=1e-9)
    assert np.linalg.norm(problem.curl_field(solution, hit.cell, hit.xi)) < 1e-10 * np.linalg.norm(
        curl
    )
    assert problem.h_field(solution, locator, [3.0, 0.0]) is None
    assert problem.poynting(solution, locator, [3.0, 0.0]) is None
    # sampling of H and S: the same values, 3 columns each
    points = np.array([x, [0.71, 0.64], [0.5, 0.5]])
    h_sampled, cells = problem.sample(solution, locator, points, quantity="H")
    s_sampled, _ = problem.sample(solution, locator, points, quantity="S")
    assert h_sampled.shape == (3, 3) and s_sampled.shape == (3, 3)
    for i, p in enumerate(points):
        assert np.allclose(h_sampled[i], problem.h_field(solution, locator, p), rtol=1e-12)
        assert np.allclose(s_sampled[i].real, problem.poynting(solution, locator, p), rtol=1e-12)
    assert np.allclose(s_sampled.imag, 0.0)
    tri = problem.triangulate(solution, 2, quantity="S")
    assert tri.values.shape[1] == 3
    assert np.allclose(tri.values.real, s, rtol=1e-9)
    with pytest.raises(ValueError):
        problem.sample(solution, locator, points, quantity="B")
    with pytest.raises(ValueError):
        problem.sample(solution, locator, points, scattered=True, quantity="H")


def test_layered_wave_curls_and_energy_flow():
    um = 1e-6
    period = 0.7 * um
    k0 = 2 * np.pi / (0.6 * um)
    angle, azimuth = 0.5, 0.8
    glass = hpfem.Material.dielectric(1.6)
    stack = hpfem.LayerStack2D(hpfem.Material.vacuum(), [], glass)
    mesh = hpfem.rectangle(3, 16, [0.0, -1.65 * um], [period, 0.75 * um])
    for c in range(mesh.num_cells):
        if mesh.cell_centroid(c)[1] < 0:
            mesh.set_cell_tag(c, 3)
    nd = hpfem.NedelecDofMap2D(mesh, 2)
    h1 = hpfem.DofMap2D(mesh, 2)
    wave = hpfem.layered_conical_wave(stack, k0, angle, azimuth, hpfem.Polarisation.S)
    setup = hpfem.ConicalScatteringSetup()
    setup.omega = k0 * c0
    setup.beta = wave.beta
    setup.materials.set(3, glass)
    setup.background = stack
    setup.incident = wave.field
    setup.incident_curl = wave.field_curl
    setup.pml = hpfem.PmlBox2D(
        [0.0, -1.2 * um], [period, 0.45 * um], [0.0, 0.0, 0.45 * um, 0.3 * um], k0
    )
    setup.pec_tags = [hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX]
    setup.periodic = [
        hpfem.PeriodicPair2D(
            hpfem.box_tag.X_MIN, hpfem.box_tag.X_MAX, [period, 0.0],
            hpfem.bloch_phase([wave.kx, 0.0], [period, 0.0]),
        )
    ]  # fmt: skip
    problem = hpfem.ConicalScattering(nd, h1, setup)
    solution = problem.solve()  # a flat stack: the total field is the stack wave
    locator = hpfem.PointLocator2D(mesh)
    flux = hpfem.plane_wave_intensity(1.0, hpfem.Material.vacuum()) * np.cos(angle)
    above = problem.poynting(solution, locator, [0.2 * um, 0.3 * um])
    below = problem.poynting(solution, locator, [0.4 * um, -0.8 * um])
    assert np.isclose(above[1], -(1.0 - wave.reflectance) * flux, rtol=2e-3)
    assert np.isclose(below[1], -wave.transmittance * flux, rtol=2e-3)
    # the analytic curl of the incident wave against the finite-difference fallback
    x = np.array([0.33 * um, 0.4 * um])
    analytic = wave.incident_curl(x)
    alone = hpfem.ConicalScatteringSetup()
    alone.omega = setup.omega
    alone.beta = setup.beta
    alone.incident = wave.incident
    alone.pec_tags = setup.pec_tags
    fd = hpfem.ConicalScattering(nd, h1, alone).incident_curl(x)
    assert np.allclose(fd, analytic, rtol=1e-6)

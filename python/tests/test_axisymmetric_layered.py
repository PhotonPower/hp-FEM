"""Axisymmetric problems on a layered background from Python (ADR-0014, M18 S1): the order-m
stack wave against the phi-Fourier transform of the 3D stack field, its homogeneous limit and
the arguments, and a bare stack that scatters nothing while a hole in a layer does."""

import numpy as np
import pytest

import hpfem

c0 = hpfem.constants.c0
K0 = 2 * np.pi


def stack(top=0.0):
    metal = hpfem.Material(-4.0 + 1.5j, 1.0)
    return hpfem.LayerStack3D(
        hpfem.Material.vacuum(),
        [hpfem.Layer(hpfem.Material.dielectric(1.5), 0.3), hpfem.Layer(metal, 0.1)],
        hpfem.Material.dielectric(1.45),
        top,
    )


def phi_transform(field, r, z, m, n_phi=96):
    """Order m of the 3D field on the ring of radius r at height z, as (E_r, v, E_z)."""
    phi = 2 * np.pi * np.arange(n_phi) / n_phi
    values = np.array([field.value([r * np.cos(p), r * np.sin(p), z]) for p in phi])
    e_r = values[:, 0] * np.cos(phi) + values[:, 1] * np.sin(phi)
    e_phi = -values[:, 0] * np.sin(phi) + values[:, 1] * np.cos(phi)
    weight = np.exp(-1j * m * phi) / n_phi
    return np.array([weight @ e_r, -1j * r * (weight @ e_phi), weight @ values[:, 2]])


def test_layered_wave_matches_the_3d_stack_field():
    s = stack()
    for pol, enum in (("s", hpfem.Polarisation.S), ("p", hpfem.Polarisation.P)):
        wave3 = s.plane_wave(K0, 0.7, enum)
        for m in (-2, 0, 1, 3):
            wave = hpfem.layered_axisymmetric_wave(s, K0, 0.7, pol, m)
            for r, z in ((0.4, 0.2), (0.9, -0.1), (0.3, -0.35), (1.1, -0.8)):
                reference = phi_transform(wave3.field, r, z, m)
                assert np.allclose(wave.value(np.array([r, z])), reference, atol=1e-12)
    wave = hpfem.layered_axisymmetric_wave(s, K0, 0.7, hpfem.Polarisation.S, 0, side="bottom")
    assert 0 < wave.reflectance < 1 and wave.absorptance > 0
    assert wave.curl(np.array([0.5, 0.3])).shape == (3,)


def test_layered_wave_homogeneous_limit_and_arguments():
    n = 1.3
    uniform = hpfem.LayerStack3D(hpfem.Material.dielectric(n), [], hpfem.Material.dielectric(n))
    for theta in (0.0, 0.5):
        top = hpfem.layered_axisymmetric_wave(uniform, 3.0, theta, "p", 1)
        bottom = hpfem.layered_axisymmetric_wave(uniform, 3.0, theta, "p", 1, side="bottom")
        down = hpfem.oblique_plane_wave(1.0, 3.0 * n, np.pi - theta, hpfem.PlanePolarisation.P, 1)
        up = hpfem.oblique_plane_wave(1.0, 3.0 * n, theta, hpfem.PlanePolarisation.P, 1)
        x = np.array([0.6, -0.4])
        # p goes through Z0, rounded 3e-12 away from 1 / (c0 eps0) in the constants
        assert np.allclose(top.value(x), down(x), atol=1e-11)
        assert np.allclose(bottom.value(x), up(x), atol=1e-11)
    with pytest.raises(ValueError):
        hpfem.layered_axisymmetric_wave(uniform, 3.0, 0.2, "q", 0)
    with pytest.raises(ValueError):
        hpfem.layered_axisymmetric_wave(uniform, 3.0, 0.2, "s", 0, side="left")


def test_bare_stack_scatters_nothing_and_a_hole_scatters():
    mesh = hpfem.rectangle(20, 30, [0.0, -1.5], [2.0, 1.5])
    s = stack()
    for c in range(mesh.num_cells):
        mesh.set_cell_tag(c, 10 + s.region(mesh.cell_centroid(c)[1]))
    nd = hpfem.NedelecDofMap2D(mesh, 2)
    h1 = hpfem.DofMap2D(mesh, 2)
    setup = hpfem.AxisymmetricScatteringSetup()
    setup.omega = K0 * c0
    for j in range(s.num_layers + 2):
        setup.materials.set(10 + j, s.material(j))
    setup.axis_tag = hpfem.box_tag.X_MIN
    setup.azimuthal_order = 1
    setup.pml = hpfem.PmlBox2D([0.0, -1.0], [1.5, 1.0], [0.0, 0.5, 0.5, 0.5], K0)
    setup.background = s
    setup.incident = hpfem.layered_axisymmetric_wave(s, K0, 0.5, "s", 1).value
    field = hpfem.AxisymmetricScattering(nd, h1, setup).solve()
    assert np.linalg.norm(field.meridian) == 0 and np.linalg.norm(field.azimuthal) == 0
    # an air hole in the glass layer next to the axis
    for c in range(mesh.num_cells):
        r, z = mesh.cell_centroid(c)
        if r < 0.3 and -0.3 < z < 0:
            mesh.set_cell_tag(c, 5)
    setup.materials.set(5, hpfem.Material.vacuum())
    problem = hpfem.AxisymmetricScattering(nd, h1, setup)
    hole = next(c for c in range(mesh.num_cells) if mesh.cell_tag(c) == 5)
    assert problem.background_material(hole).eps_r == pytest.approx(2.25)
    assert np.linalg.norm(problem.solve().meridian) > 0
    # the interfaces must lie on mesh lines
    setup.background = stack(top=0.05)
    with pytest.raises(ValueError):
        hpfem.AxisymmetricScattering(nd, h1, setup)

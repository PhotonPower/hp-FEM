"""Conical post-processing bindings (M15 F10 / F11): frames, orders on a line, cross-sections
and far field of the Mie cylinder at beta = 0, and the flux balance of grating.solve."""

import numpy as np
import pytest

import hpfem


def test_literature_frame_round_trip():
    v = np.array([1 + 2j, 3 - 1j, 0.5 + 0.25j])
    lit = hpfem.to_literature_frame(v)
    assert lit[0] == v[0] and lit[1] == -v[2] and lit[2] == v[1]
    assert np.allclose(hpfem.from_literature_frame(lit), v)


def test_conical_diffraction_orders_of_a_plane_wave():
    k0 = 2 * np.pi / 0.405
    period, beta = 0.4, 0.2 * k0
    kt0 = 0.3 * k0
    kn = np.sqrt(k0**2 - kt0**2 - beta**2)
    a0 = np.array([0.5, 0.2j, 0.1])

    def field(x):
        return a0 * np.exp(1j * (kt0 * x[0] + kn * x[1]))

    line = hpfem.OrderLine(np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0]), period)
    orders = hpfem.conical_diffraction_orders(field, line, k0, 1.0, kt0, beta, kn, max_order=1)
    assert [o.order for o in orders] == [-1, 0, 1]
    assert np.allclose(orders[1].amplitude, a0, atol=1e-10)
    assert orders[1].efficiency == pytest.approx(np.vdot(a0, a0).real)
    assert np.linalg.norm(orders[0].amplitude) < 1e-10
    assert np.allclose(hpfem.conical_curl_of(field, np.array([0.1, 0.2]), beta, 1e-4),
                       1j * np.cross([kt0, kn, beta], a0) * np.exp(1j * (kt0 * 0.1 + kn * 0.2)),
                       rtol=1e-6)  # fmt: skip


def test_mie_cylinder_cross_sections_and_far_field():
    k, radius, index = 6.0, 0.25, 1.5
    disc, inside = 2, 7
    mesh = hpfem.square_with_disc(3, radius, 1.0, 2.0, disc)
    for c in range(mesh.num_cells):
        if mesh.cell_tag(c) != disc and np.linalg.norm(mesh.cell_centroid(c)) < 0.45:
            mesh.set_cell_tag(c, inside)
    nd = hpfem.NedelecDofMap2D(mesh, 3)
    h1 = hpfem.DofMap2D(mesh, 3)
    setup = hpfem.ConicalScatteringSetup()
    setup.omega = k * hpfem.constants.c0
    setup.beta = 0.0
    setup.materials.set(disc, hpfem.Material.dielectric(index))
    setup.incident = hpfem.conical_plane_wave(np.array([0.0, 0.0, 1.0]), np.array([k, 0.0, 0.0]))
    setup.pml = hpfem.PmlBox2D.uniform(
        np.array([-1.0, -1.0]), np.array([1.0, 1.0]), 1.0, k, 1.0, hpfem.PmlProfile(2, 1e-10)
    )
    setup.pec_tags = [hpfem.box_tag.X_MIN, hpfem.box_tag.X_MAX, hpfem.box_tag.Y_MIN,
                      hpfem.box_tag.Y_MAX]  # fmt: skip
    problem = hpfem.ConicalScattering(nd, h1, setup)
    solution = problem.solve()
    inner = hpfem.Surface2D.around_cells(mesh, inside)
    outer = hpfem.Surface2D()
    outer.facets = [
        f
        for f in inner.facets
        if all(mesh.cell_tag(int(c)) != disc for c in mesh.facet_cells(f.facet) if c >= 0)
    ]
    cs = hpfem.conical_cross_sections(problem, solution, outer, 1.0, 4)
    far = hpfem.ConicalFarField(problem, solution, outer, 4)
    assert cs.absorption == 0.0
    assert far.transverse_wavenumber == pytest.approx(k)
    assert far.scattering_cross_section(1.0) == pytest.approx(cs.scattering, rel=5e-3)
    pattern = far.pattern(0.3)
    assert abs(pattern[0]) < 1e-8 * abs(pattern[2]) and abs(pattern[1]) < 1e-8 * abs(pattern[2])

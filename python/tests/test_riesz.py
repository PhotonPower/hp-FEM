"""Riesz projection from Python: the modal expansion of a line-dipole source on the
Fabry-Perot strip reproduces the direct solution, the spectrum matrix sums to the total, and
the setup is validated."""

import numpy as np
import pytest

import hpfem

c0 = hpfem.constants.c0
INDEX = 3.5
THICKNESS = 1.0
MARGIN = 0.5
PML = 3.0
HEIGHT = 0.125
SLAB = 2


def fabry_perot(order):
    return complex(
        np.pi * order / (INDEX * THICKNESS),
        -np.log((INDEX + 1) / (INDEX - 1)) / (INDEX * THICKNESS),
    )


def strip(cells_per_unit=4, order=2):
    half = THICKNESS / 2 + MARGIN + PML
    nx = int(round(2 * half * cells_per_unit))
    mesh = hpfem.rectangle(nx, 1, [-half, 0.0], [half, HEIGHT])
    for c in range(mesh.num_cells):
        if abs(mesh.cell_centroids[c][0]) < THICKNESS / 2:
            mesh.set_cell_tag(c, SLAB)
    dofs = hpfem.NedelecDofMap2D(mesh, order)
    k4 = fabry_perot(4)
    setup = hpfem.ResonanceSetup2D()
    setup.target_omega = 0.97 * k4.real * c0
    setup.materials.set(SLAB, hpfem.Material.dielectric(INDEX))
    setup.pec_tags = [hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX]
    setup.pml = hpfem.PmlBox2D(
        [-(THICKNESS / 2 + MARGIN), 0.0], [THICKNESS / 2 + MARGIN, HEIGHT], [PML, PML, 0.0, 0.0],
        k4.real, 1.0, hpfem.PmlProfile(2, 1e-10),
    )  # fmt: skip
    setup.num_modes = 12
    setup.krylov_dimension = 48
    return mesh, dofs, setup


def test_riesz_projection_reproduces_the_direct_solution():
    mesh, dofs, setup = strip()
    resonance = hpfem.Resonance2D(dofs, setup)
    modes = resonance.solve()
    k4 = fabry_perot(4)
    riesz_setup = hpfem.RieszSetup()
    riesz_setup.poles = [m.omega for m in modes]
    riesz_setup.omega_min = (k4.real - 0.3) * c0
    riesz_setup.omega_max = (k4.real + 0.3) * c0
    riesz_setup.points_per_pole = 16
    riesz_setup.background_points = 32
    riesz = hpfem.RieszProjection2D(resonance, riesz_setup)
    centre, sigma = np.array([0.1, 0.06]), 0.05

    def current(x):
        g = np.exp(-np.sum((x - centre) ** 2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
        return np.array([0.0, g], dtype=complex)

    source = riesz.add_current(current)
    power = riesz.add_emitted_power(source)
    point = riesz.add_point_value([-0.2, 0.06], [0.0, 1.0])
    with pytest.raises(ValueError):
        riesz.total(0, 0, k4.real * c0)  # before run
    riesz.run()
    contours = riesz.contours
    assert contours[-1].kind == hpfem.RieszContour.Kind.BACKGROUND
    assert contours[-1].encloses(k4.real * c0)
    assert all(c.convergence < 1e-6 for c in contours)
    omegas = [(k4.real + d) * c0 for d in (-0.2, -0.1, 0.0, 0.1, 0.2)]
    spectrum = riesz.spectrum(source, power, omegas)
    assert spectrum.shape == (len(contours), len(omegas))
    for j, omega in enumerate(omegas):
        direct = riesz.direct(omega)
        assert direct.shape == (1, 2)
        total = riesz.total(source, power, omega)
        assert abs(total - direct[0, power]) < 1e-8 * abs(direct[0, power])
        assert abs(riesz.total(source, point, omega) - direct[0, point]) < 1e-8 * abs(
            direct[0, point]
        )
        assert abs(spectrum[:, j].sum() - total) < 1e-12 * abs(total)
        assert total.real > 0  # emitted power
    # the residue field of the fundamental mode is parallel to the eigenvector
    nearest = min(modes, key=lambda m: abs(m.omega / c0 - k4))
    row = next(i for i, c in enumerate(contours) if nearest.omega in c.poles)
    field = riesz.field(row, source, nearest.omega.real * c0)
    overlap = abs(np.vdot(nearest.field, field))
    assert overlap > (1 - 1e-6) * np.linalg.norm(nearest.field) * np.linalg.norm(field)
    with pytest.raises(ValueError):
        riesz.total(source, power, (k4.real + 2.0) * c0)  # outside the background contour


def test_riesz_setup_is_validated():
    _, dofs, setup = strip(order=1)
    resonance = hpfem.Resonance2D(dofs, setup)
    bad = hpfem.RieszSetup()
    bad.omega_min = 2.0
    bad.omega_max = 1.0
    with pytest.raises(ValueError):
        hpfem.RieszProjection2D(resonance, bad)
    bad.omega_max = 3.0
    bad.points_per_pole = 1
    with pytest.raises(ValueError):
        hpfem.RieszProjection2D(resonance, bad)

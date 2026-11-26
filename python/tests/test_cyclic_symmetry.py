"""Cyclic symmetry from Python (issue #244 S1+S2): the Mie cylinder under an oblique plane wave
solved on one quarter (C_4) reproduces the full-domain solution on the same mesh (fields in
every quadrant, cross-sections, absorbed power, far field); azimuthal parts sum to the field;
the rotated periodic pair and the hexagon mesh for C_6; the point inversion (3D) on the half
of a ball of a general anisotropic medium."""

import numpy as np
import pytest
from conftest import BOX_SIDES

import hpfem

K = 6.0
RADIUS = 0.25
PROFILE = hpfem.PmlProfile(2, 1e-6)
MASTER, SLAVE = 20, 21


def _setup():
    setup = hpfem.ScatteringSetup2D()
    setup.omega = K * hpfem.constants.c0
    setup.materials.set(2, hpfem.Material(complex(2.25, 0.3), 1.0))
    d = np.array([np.cos(0.37), np.sin(0.37)])
    setup.incident = hpfem.plane_wave(complex(0.7, 0.2) * np.array([-d[1], d[0]]), K * d)
    setup.formulation = hpfem.Formulation.SCATTERED_FIELD
    setup.pml = hpfem.PmlBox2D.uniform([-1.0, -1.0], [1.0, 1.0], 1.0, K, 1.0, PROFILE)
    setup.pec_tags = BOX_SIDES
    return setup


def test_azimuthal_parts_sum_to_the_field_and_the_cyclic_pair():
    field = hpfem.plane_wave([0.6, -0.8], [4.8, 3.6])
    sym = hpfem.CyclicSymmetry2D(3, center=[0.1, -0.2])
    parts = [hpfem.azimuthal_part(field, sym, m) for m in range(3)]
    x = np.array([0.3, 0.45])
    assert np.allclose(sum(p.value(x) for p in parts), field.value(x), atol=1e-14)
    w = np.exp(2j * np.pi / 3)
    r = sym.rotation(1)
    assert np.allclose(parts[1].value(sym.rotate(x)), w * r @ parts[1].value(x), atol=1e-13)
    pair = hpfem.cyclic_pair(MASTER, SLAVE, 6, 1)
    assert not pair.is_translation
    assert pair.phase == pytest.approx(np.exp(1j * np.pi / 3))
    assert np.allclose(pair.rotation, hpfem.rotation_matrix(np.pi / 3))
    assert hpfem.PeriodicPair2D(1, 2, [1.0, 0.0]).is_translation
    hexagon = hpfem.hexagon(3, 1.0, 1)
    assert hexagon.num_vertices == 37 and hexagon.num_cells == 54


def test_quarter_reproduces_the_full_problem():
    full = hpfem.square_with_disc(2, RADIUS, 1.0, 2.0, 2)
    quarter = hpfem.extract(full, lambda c: c[0] > 0 and c[1] > 0)
    sym = hpfem.CyclicSymmetry2D(4, master=MASTER, slave=SLAVE)
    assert hpfem.tag_sector_sides(quarter, sym, [1.0, 0.0]) == (16, 16)
    p = 2
    full_dofs = hpfem.NedelecDofMap2D(full, p)
    full_problem = hpfem.Scattering2D(full_dofs, _setup())
    full_solution = full_problem.solve()
    dofs = hpfem.NedelecDofMap2D(quarter, p)
    problem = hpfem.CyclicSymmetricScattering2D(dofs, _setup(), sym)
    assert problem.num_parts == 4 and problem.num_images == 4
    assert [problem.order(i) for i in range(4)] == [0, 1, 2, 3]
    # the order 3 = n - 1 reuses the transposed factors of order 1
    assert problem.num_factorisations == 3
    assert [problem.partner(i) for i in range(4)] == [-1, -1, -1, 1]
    solution = problem.solve()
    separate = hpfem.CyclicSymmetricScattering2D(dofs, _setup(), sym, pair_orders=False)
    assert separate.num_factorisations == 4
    for a, b in zip(solution.parts, separate.solve().parts, strict=True):
        assert np.abs(a.unknown - b.unknown).max() < 1e-10 * np.abs(b.unknown).max()

    points = np.array([[0.6, 0.3], [-0.4, 0.7], [-0.55, -0.5], [0.35, -0.8], [0.1, 0.05]])
    locator = hpfem.PointLocator2D(quarter)
    full_locator = hpfem.PointLocator2D(full)
    values = problem.field(solution, locator, points)
    expected = np.array([full_problem.total_field(full_solution, full_locator, x) for x in points])
    assert np.abs(values - expected).max() < 1e-9 * np.abs(expected).max()
    point, j = problem.fold([-0.4, 0.7])
    assert j == 1 and np.allclose(sym.rotate(point, j), [-0.4, 0.7])

    surface = hpfem.Surface2D.around_cells(quarter, 2)
    cs_full = hpfem.cross_sections(
        full_problem, full_solution, hpfem.Surface2D.around_cells(full, 2), 1.0
    )
    cs = problem.cross_sections(solution, surface, 1.0)
    assert cs.scattering == pytest.approx(cs_full.scattering, rel=1e-9)
    assert cs.absorption == pytest.approx(cs_full.absorption, rel=1e-9)
    assert problem.absorbed_power(solution) == pytest.approx(
        hpfem.absorbed_power(full_problem, full_solution), rel=1e-9
    )
    far_full = hpfem.FarField2D(
        full,
        hpfem.Surface2D.around_cells(full, 2),
        hpfem.discrete_field(full_dofs, full_solution.unknown),
        full_problem.setup.omega,
        hpfem.Material.vacuum(),
        6,
    )
    far = hpfem.CyclicSymmetricFarField2D(problem, solution, surface, 6)
    for phi in (0.2, 2.0, 4.1):
        d = [np.cos(phi), np.sin(phi)]
        assert np.allclose(far.pattern(d), far_full.pattern(d), rtol=1e-9, atol=1e-12)


def test_invalid_setups_are_refused():
    full = hpfem.square_with_disc(2, RADIUS, 1.0, 2.0, 2)
    quarter = hpfem.extract(full, lambda c: c[0] > 0 and c[1] > 0)
    sym = hpfem.CyclicSymmetry2D(4, master=MASTER, slave=SLAVE)
    hpfem.tag_sector_sides(quarter, sym, [1.0, 0.0])
    dofs = hpfem.NedelecDofMap2D(quarter, 1)
    with pytest.raises(ValueError):  # the quarter is too wide for C_6
        hpfem.CyclicSymmetricScattering2D(
            dofs, _setup(), hpfem.CyclicSymmetry2D(6, master=MASTER, slave=SLAVE)
        )
    setup = _setup()
    setup.pec_tags = BOX_SIDES + [MASTER]
    with pytest.raises(ValueError):
        hpfem.CyclicSymmetricScattering2D(dofs, setup, sym)


def test_point_inversion_halves_an_anisotropic_ball():
    k = 3.0
    full = hpfem.box_with_ball(1, 0.5, 1.0, 1.5, 2)
    half = hpfem.extract(full, lambda c: c[2] > 0)
    sym = hpfem.CyclicSymmetry3D.point_inversion([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], MASTER, SLAVE)
    assert sym.inversion and sym.order == 2
    assert np.allclose(sym.rotation(1), -np.eye(3))
    master, slave = hpfem.tag_sector_sides(half, sym, [1.0, 0.3, 0.0])
    assert master == slave > 0
    eps = np.array([[3.0 + 0.4j, 0.5, -0.3], [0.5, 2.2 + 0.2j, 0.4], [-0.3, 0.4, 4.0 + 0.3j]])
    setup = hpfem.ScatteringSetup3D()
    setup.omega = k * hpfem.constants.c0
    setup.materials.set(2, hpfem.Material.anisotropic(eps))
    d = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    a = np.cross(d, [0.0, 0.0, 1.0])
    a /= np.linalg.norm(a)
    setup.incident = hpfem.plane_wave(a + 0.5j * np.cross(d, a), k * d)
    setup.formulation = hpfem.Formulation.SCATTERED_FIELD
    setup.pml = hpfem.PmlBox3D.uniform([-1.0] * 3, [1.0] * 3, 0.5, k, 1.0, PROFILE)
    setup.pec_tags = BOX_SIDES + [hpfem.box_tag.Z_MIN, hpfem.box_tag.Z_MAX]
    full_problem = hpfem.Scattering3D(hpfem.NedelecDofMap3D(full, 1), setup)
    full_solution = full_problem.solve()
    problem = hpfem.CyclicSymmetricScattering3D(hpfem.NedelecDofMap3D(half, 1), setup, sym)
    assert problem.num_parts == 2 and problem.num_factorisations == 2
    solution = problem.solve()
    points = np.array([[0.2, 0.1, 0.3], [-0.3, 0.2, -0.1], [0.7, -0.6, -0.4]])
    values = problem.field(solution, hpfem.PointLocator3D(half), points)
    full_locator = hpfem.PointLocator3D(full)
    expected = np.array([full_problem.total_field(full_solution, full_locator, x) for x in points])
    assert np.abs(values - expected).max() < 1e-9 * np.abs(expected).max()
    point, j = problem.fold([-0.3, 0.2, -0.1])
    assert j == 1 and np.allclose(point, [0.3, -0.2, 0.1])
    assert problem.absorbed_power(solution) == pytest.approx(
        hpfem.absorbed_power(full_problem, full_solution), rel=1e-9
    )
    pair = hpfem.inversion_pair(MASTER, SLAVE, 1, [0.5, 0.0, -1.0])
    assert pair.phase == -1 and not pair.is_translation
    assert np.allclose(pair.rotation, -np.eye(3)) and np.allclose(pair.shift, [1.0, 0.0, -2.0])

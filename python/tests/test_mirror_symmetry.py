"""Mirror symmetry from Python (issue #244 S3): the Mie cylinder under an oblique plane wave on
the quarter domain reproduces the full-domain solution on the same mesh (fields in every
quadrant, cross-sections, absorbed power, far field); parity parts sum to the field."""

import numpy as np
import pytest
from conftest import BOX_SIDES

import hpfem

K = 6.0
RADIUS = 0.25
PROFILE = hpfem.PmlProfile(2, 1e-6)


def _setup(pml):
    setup = hpfem.ScatteringSetup2D()
    setup.omega = K * hpfem.constants.c0
    setup.materials.set(2, hpfem.Material(complex(2.25, 0.3), 1.0))
    d = np.array([np.cos(0.37), np.sin(0.37)])
    setup.incident = hpfem.plane_wave(complex(0.7, 0.2) * np.array([-d[1], d[0]]), K * d)
    setup.formulation = hpfem.Formulation.SCATTERED_FIELD
    setup.pml = pml
    setup.pec_tags = BOX_SIDES
    return setup


def test_parity_parts_sum_to_the_field():
    field = hpfem.plane_wave([0.6, -0.8], [4.8, 3.6])
    planes = [
        hpfem.MirrorPlane2D([0.1, 0.0], [1.0, 0.0], 10),
        hpfem.MirrorPlane2D([0.0, -0.2], [0.0, 1.0], 11),
    ]
    parts = [hpfem.parity_part(field, planes, [s0, s1]) for s0 in (1, -1) for s1 in (1, -1)]
    x = [0.3, 0.45]
    assert np.allclose(sum(p.value(x) for p in parts), field.value(x), atol=1e-14)
    odd_x = hpfem.parity_part(field, planes[:1], [-1])
    # odd under x -> 0.2 - x: n x E = E_y vanishes on the plane x = 0.1
    assert abs(odd_x.value([0.1, 0.3])[1]) < 1e-14
    assert np.allclose(planes[0].reflect([0.3, 0.45]), [-0.1, 0.45])
    with pytest.raises(ValueError):
        hpfem.parity_part(field, planes, [1, 0])


def test_quarter_domain_reproduces_the_full_problem():
    full = hpfem.square_with_disc(2, RADIUS, 1.0, 2.0, 2)
    quarter = hpfem.extract(full, lambda c: c[0] > 0 and c[1] > 0)
    planes = [
        hpfem.MirrorPlane2D([0.0, 0.0], [1.0, 0.0], 10),
        hpfem.MirrorPlane2D([0.0, 0.0], [0.0, 1.0], 11),
    ]
    for plane in planes:
        assert hpfem.tag_mirror_plane(quarter, plane) > 0
    p = 2
    full_dofs = hpfem.NedelecDofMap2D(full, p)
    full_problem = hpfem.Scattering2D(
        full_dofs, _setup(hpfem.PmlBox2D.uniform([-1.0, -1.0], [1.0, 1.0], 1.0, K, 1.0, PROFILE))
    )
    full_solution = full_problem.solve()
    dofs = hpfem.NedelecDofMap2D(quarter, p)
    sym = hpfem.MirrorSymmetricScattering2D(
        dofs,
        _setup(hpfem.PmlBox2D([0.0, 0.0], [1.0, 1.0], [0.0, 1.0, 0.0, 1.0], K, 1.0, PROFILE)),
        planes,
    )
    assert sym.num_parts == 4 and sym.num_images == 4
    solution = sym.solve()
    assert len(solution.parts) == 4

    points = np.array([[0.6, 0.3], [-0.4, 0.7], [-0.55, -0.5], [0.35, -0.8], [0.1, 0.05]])
    locator = hpfem.PointLocator2D(quarter)
    full_locator = hpfem.PointLocator2D(full)
    values = sym.field(solution, locator, points)
    expected = np.array([full_problem.total_field(full_solution, full_locator, x) for x in points])
    assert np.abs(values - expected).max() < 1e-9 * np.abs(expected).max()
    scattered = sym.field(solution, locator, points, scattered=True)
    incident = np.array([sym.incident_field(x) for x in points])
    assert np.allclose(scattered + incident, values, atol=1e-12)

    cs_full = hpfem.cross_sections(
        full_problem, full_solution, hpfem.Surface2D.around_cells(full, 2), 1.0
    )
    cs = sym.cross_sections(solution, hpfem.Surface2D.around_cells(quarter, 2), 1.0)
    assert cs.scattering == pytest.approx(cs_full.scattering, rel=1e-9)
    assert cs.absorption == pytest.approx(cs_full.absorption, rel=1e-9)
    assert sym.absorbed_power(solution) == pytest.approx(
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
    far = hpfem.MirrorSymmetricFarField2D(
        sym, solution, hpfem.Surface2D.around_cells(quarter, 2), 6
    )
    for phi in (0.2, 2.0, 4.1):
        d = [np.cos(phi), np.sin(phi)]
        assert np.allclose(far.pattern(d), far_full.pattern(d), rtol=1e-9, atol=1e-12)


def test_symmetric_excitation_needs_one_part_and_bad_setups_are_refused():
    full = hpfem.square_with_disc(2, RADIUS, 1.0, 2.0, 2)
    half = hpfem.extract(full, lambda c: c[1] > 0)
    plane = hpfem.MirrorPlane2D([0.0, 0.0], [0.0, 1.0], 10)
    hpfem.tag_mirror_plane(half, plane)
    dofs = hpfem.NedelecDofMap2D(half, 1)
    setup = _setup(hpfem.PmlBox2D([-1.0, 0.0], [1.0, 1.0], [1.0, 1.0, 0.0, 1.0], K, 1.0, PROFILE))
    setup.incident = hpfem.plane_wave([0.0, 1.0], [K, 0.0])  # E_y e^{ikx}: odd under y -> -y
    sym = hpfem.MirrorSymmetricScattering2D(dofs, setup, [plane])
    assert sym.num_parts == 1 and list(sym.parity(0)) == [-1]
    point, reflections = sym.fold([0.3, -0.4])
    assert np.allclose(point, [0.3, 0.4]) and reflections == 1
    setup.pec_tags = BOX_SIDES + [10]
    with pytest.raises(ValueError):
        hpfem.MirrorSymmetricScattering2D(dofs, setup, [plane])

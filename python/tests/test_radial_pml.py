"""Radial PML from Python (issue #244 S4): the circular, spherical and cylindrical layers, the
setup's pml accepting a PmlBox or a RadialPml, the C_6-symmetric polar_disc mesh, and a C_6
sector in a circular PML reproducing the full disc on the same mesh."""

import numpy as np
import pytest

import hpfem

K = 6.0
MASTER, SLAVE = 20, 21


def test_radial_layers_and_the_setup_property():
    pml = hpfem.RadialPml2D([0.0, 0.0], 1.0, 0.5, K, 1.0, hpfem.PmlProfile(2, 1e-6))
    assert pml.shape == "circular"
    assert not pml.in_layer([0.5, 0.5]) and pml.in_layer([1.0, 0.6])
    j = pml.jacobian([1.1, 0.4])
    assert np.allclose(j, j.T) and not np.allclose(j, np.eye(2))
    assert np.allclose(pml.jacobian([0.2, 0.1]), np.eye(2))
    sphere = hpfem.RadialPml3D([0.0, 0.0, 0.0], 1.0, 0.5, K)
    assert sphere.shape == "spherical"
    cylinder = hpfem.RadialPml3D.cylindrical([0, 0, 0], [0, 0, 2], 1.0, 0.5, -1.0, 1.0, 0.3, K)
    assert cylinder.shape == "cylindrical" and np.allclose(cylinder.axis, [0, 0, 1])
    assert cylinder.in_layer([0.1, 0.2, 1.1]) and not cylinder.in_layer([0.1, 0.2, 0.5])

    setup = hpfem.ScatteringSetup2D()
    assert setup.pml is None
    setup.pml = pml
    assert isinstance(setup.pml, hpfem.RadialPml2D)
    assert setup.pml.inner_radius == 1.0
    setup.pml = hpfem.PmlBox2D.uniform([-1.0, -1.0], [1.0, 1.0], 0.5, K)
    assert isinstance(setup.pml, hpfem.PmlBox2D)
    setup.pml = None
    assert setup.pml is None


def test_polar_disc_counts():
    mesh = hpfem.polar_disc(6, 2, [0.5, 1.0], [2, 0])
    rings = 4
    assert mesh.num_vertices == 1 + 6 * rings * (rings + 1) // 2
    assert mesh.num_cells == 6 * rings * rings
    with pytest.raises(ValueError):
        hpfem.polar_disc(6, 2, [1.0, 0.5])


def test_c6_sector_in_a_circular_pml_reproduces_the_full_disc():
    full = hpfem.polar_disc(6, 2, [0.25, 0.75, 1.25], [2, 0, 0])
    sector = hpfem.extract(full, lambda c: 0 < np.arctan2(c[1], c[0]) < np.pi / 3)
    sym = hpfem.CyclicSymmetry2D(6, master=MASTER, slave=SLAVE)
    assert hpfem.tag_sector_sides(sector, sym, [1.0, 0.0]) == (10, 10)

    def setup():
        s = hpfem.ScatteringSetup2D()
        s.omega = K * hpfem.constants.c0
        s.materials.set(2, hpfem.Material(complex(2.25, 0.3), 1.0))
        d = np.array([np.cos(0.37), np.sin(0.37)])
        s.incident = hpfem.plane_wave([-d[1], d[0]], K * d)
        s.formulation = hpfem.Formulation.SCATTERED_FIELD
        s.pml = hpfem.RadialPml2D([0.0, 0.0], 0.75, 0.5, K, 1.0, hpfem.PmlProfile(2, 1e-3))
        s.pec_tags = [hpfem.DISC_BOUNDARY]
        return s

    full_problem = hpfem.Scattering2D(hpfem.NedelecDofMap2D(full, 2), setup())
    full_solution = full_problem.solve()
    problem = hpfem.CyclicSymmetricScattering2D(hpfem.NedelecDofMap2D(sector, 2), setup(), sym)
    assert problem.num_parts == 6
    solution = problem.solve()
    cs_full = hpfem.cross_sections(
        full_problem, full_solution, hpfem.Surface2D.around_cells(full, 2), 1.0
    )
    cs = problem.cross_sections(solution, hpfem.Surface2D.around_cells(sector, 2), 1.0)
    assert cs.scattering == pytest.approx(cs_full.scattering, rel=1e-9)
    assert cs.absorption == pytest.approx(cs_full.absorption, rel=1e-9)

    # a box layer breaks the C_6 symmetry and is refused
    bad = setup()
    bad.pml = hpfem.PmlBox2D.uniform([-0.75, -0.75], [0.75, 0.75], 0.5, K)
    with pytest.raises(ValueError):
        hpfem.CyclicSymmetricScattering2D(hpfem.NedelecDofMap2D(sector, 1), bad, sym)

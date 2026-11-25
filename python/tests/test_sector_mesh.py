"""Sector meshing with Gmsh (issue #244 S5): one sector of a C_n-symmetric structure with
matching sides, solved with the cyclic symmetry in a radial PML: the Mie cylinder against the
series, a cluster of rods on C_6, C_3 and C_2 sectors (three different meshes) against each
other, the Mie sphere in 3D, the cylinder domain with slabs, and the refusal of a structure
without the symmetry."""

import math

import numpy as np
import pytest

import hpfem
from hpfem import meshing

NM = 1e-9
R = 250 * NM
K = 1.5 / R  # k R = 1.5


@pytest.fixture(scope="module", autouse=True)
def gmsh_available():
    gmsh = pytest.importorskip("gmsh")
    try:
        gmsh.initialize(interruptible=False)
        gmsh.finalize()
    except Exception as error:  # pragma: no cover - missing system libraries
        pytest.skip(f"gmsh cannot start: {error}")


def _solve_2d(sector, sizes, p, materials):
    mesh, sym = meshing.sector_mesh(sector, sizes)
    assert len(mesh.facets_with_tag(sector.master_tag)) == len(
        mesh.facets_with_tag(sector.slave_tag)
    )
    setup = hpfem.ScatteringSetup2D()
    setup.omega = K * hpfem.constants.c0
    for tag, material in materials.items():
        setup.materials.set(tag, material)
    d = np.array([math.cos(0.37), math.sin(0.37)])
    setup.incident = hpfem.plane_wave([-d[1], d[0]], K * d)
    setup.formulation = hpfem.Formulation.SCATTERED_FIELD
    setup.pml = hpfem.RadialPml2D(
        [0.0, 0.0], 1000 * NM, 1000 * NM, K, 1.0, hpfem.PmlProfile(2, 1e-10)
    )
    setup.pec_tags = [hpfem.DISC_BOUNDARY]
    problem = hpfem.CyclicSymmetricScattering2D(hpfem.NedelecDofMap2D(mesh, p), setup, sym)
    assert problem.num_parts == sector.order
    solution = problem.solve()
    return problem.cross_sections(solution, hpfem.Surface2D.around_cells(mesh, 3), 1.0)


def test_mie_cylinder_on_a_c6_sector_against_the_series():
    sector = meshing.Sector(
        order=6,
        radius=2000 * NM,
        shapes=[meshing.Shape("ellipse", 3, dict(x=0.0, y=0.0, rx=R, ry=R))],
        rings=[(1000 * NM, 1)],
    )
    cs = _solve_2d(sector, {1: 80 * NM, 3: 40 * NM}, 3, {3: hpfem.Material.dielectric(1.5)})
    exact = hpfem.mie_cylinder_scattering_width(K, R, 1.5)
    # 2.0e-5 here (p = 3, 935 cells); the PML and the discretisation, not the coupling
    assert abs(cs.scattering - exact) / exact < 1e-4


def test_rod_cluster_on_c6_c3_c2_sectors_agrees():
    rods = [
        meshing.Shape(
            "ellipse",
            3,
            dict(
                x=300 * NM * math.cos(j * math.pi / 3),
                y=300 * NM * math.sin(j * math.pi / 3),
                rx=80 * NM,
                ry=80 * NM,
            ),
        )
        for j in range(6)
    ]
    rods.append(meshing.Shape("ellipse", 3, dict(x=0.0, y=0.0, rx=100 * NM, ry=100 * NM)))
    lossy = {3: hpfem.Material(complex(4.0, 0.5), 1.0)}
    results = []
    for n in (6, 3, 2):
        sector = meshing.Sector(order=n, radius=2000 * NM, shapes=rods, rings=[(1000 * NM, 1)])
        results.append(_solve_2d(sector, {1: 80 * NM, 3: 30 * NM}, 3, lossy))
    # three different meshes of the same structure: 7e-7 apart here
    for cs in results[1:]:
        assert cs.scattering == pytest.approx(results[0].scattering, rel=1e-5)
        assert cs.absorption == pytest.approx(results[0].absorption, rel=1e-5)

    # three of the six rods are not C_6-symmetric
    with pytest.raises(ValueError, match="symmetric"):
        meshing.sector_mesh(
            meshing.Sector(order=6, radius=2000 * NM, shapes=rods[:3]), {1: 80 * NM}
        )


def test_sector_description_and_tags():
    sector = meshing.Sector(
        order=4,
        radius=1.0,
        dim=3,
        z_range=(-0.5, 0.5),
        rings=[(0.6, 7)],
        slabs=[meshing.Slab(8, -0.5, -0.2)],
        shapes=[meshing.Solid("cylinder", 9, dict(x=0.0, y=0.0, z=-0.2, r=0.2, height=0.3))],
        background_tag=6,
    )
    assert sector.tag_at(0.1, 0.0, 0.0) == 9  # the pillar
    assert sector.tag_at(0.4, 0.1, -0.3) == 8  # the substrate slab
    assert sector.tag_at(0.4, 0.1, 0.3) == 7  # the ring
    assert sector.tag_at(0.8, 0.0, 0.3) == 6  # background
    assert isinstance(sector.symmetry(), hpfem.CyclicSymmetry3D)
    for bad in (
        meshing.Sector(order=1, radius=1.0),
        meshing.Sector(order=4, radius=1.0, rings=[(1.5, 2)]),
        meshing.Sector(order=4, radius=1.0, slabs=[meshing.Slab(2, 0.0, 1.0)]),
        meshing.Sector(order=4, radius=1.0, shapes=[meshing.Solid("sphere", 2, dict(r=0.1))]),
        meshing.Sector(order=4, radius=1.0, master_tag=1),
    ):
        with pytest.raises(ValueError):
            bad.check()

    # the cylinder domain: pillar on a substrate slab, the ring as a mesh surface
    mesh, sym = meshing.sector_mesh(sector, {6: 0.25, 7: 0.25, 8: 0.2, 9: 0.1}, unit=1.0)
    assert sorted({mesh.cell_tag(c) for c in range(mesh.num_cells)}) == [6, 7, 8, 9]
    assert len(mesh.facets_with_tag(20)) == len(mesh.facets_with_tag(21)) > 0
    assert len(mesh.facets_with_tag(hpfem.DISC_BOUNDARY)) > 0
    # every vertex of the sector lies within 0 .. 90 degrees
    v = np.array([mesh.vertex(i) for i in range(mesh.num_vertices)])
    rho = np.hypot(v[:, 0], v[:, 1])
    phi = np.arctan2(v[:, 1], v[:, 0])[rho > 1e-9]
    assert phi.min() > -1e-9 and phi.max() < math.pi / 2 + 1e-9
    assert sym.order == 4


def test_mie_sphere_on_a_c6_sector():
    a = 100 * NM
    k = 2.0 / a
    sector = meshing.Sector(
        order=6,
        radius=3 * a,
        dim=3,
        shapes=[meshing.Solid("sphere", 3, dict(x=0.0, y=0.0, z=0.0, r=a))],
        rings=[(2 * a, 1)],
    )
    mesh, sym = meshing.sector_mesh(sector, {1: 0.5 * a, 3: 0.3 * a})
    setup = hpfem.ScatteringSetup3D()
    setup.omega = k * hpfem.constants.c0
    setup.materials.set(3, hpfem.Material.dielectric(2.0))
    d = np.array([math.sin(0.7) * math.cos(0.4), math.sin(0.7) * math.sin(0.4), math.cos(0.7)])
    pol = np.cross(d, [0.2, -1.0, 0.3])
    setup.incident = hpfem.plane_wave(pol / np.linalg.norm(pol), k * d)
    setup.formulation = hpfem.Formulation.SCATTERED_FIELD
    setup.pml = hpfem.RadialPml3D([0.0, 0.0, 0.0], 2 * a, a, k, 1.0, hpfem.PmlProfile(3, 1e-8))
    setup.pec_tags = [hpfem.DISC_BOUNDARY]
    problem = hpfem.CyclicSymmetricScattering3D(hpfem.NedelecDofMap3D(mesh, 1), setup, sym)
    assert problem.num_parts == 6
    cs = problem.cross_sections(problem.solve(), hpfem.Surface3D.around_cells(mesh, 3), 1.0)
    q = cs.scattering / (math.pi * a * a)
    mie = hpfem.mie_sphere(k, a, complex(4.0, 0.0)).scattering_efficiency()
    # p = 1 on a coarse mesh: 6.3e-2 here (p = 2: 1.9e-2, 38 s)
    assert abs(q - mie) / mie < 0.1

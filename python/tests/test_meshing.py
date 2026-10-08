"""hpfem.meshing (M15 F6): element sizes, the unit-cell mesher (needs the gmsh package,
skipped otherwise), the mesh report and the periodic check, and read_gmsh_periodic."""

import numpy as np
import pytest

import hpfem
from hpfem import grating, meshing, units

NM = units.nm
OMEGA = units.angular_frequency(wavelength=405 * NM)


def test_element_sizes_follow_wavelength_and_decay_length():
    glass = hpfem.Material.dielectric(1.5)
    silver = hpfem.Material()
    silver.eps_r = -4.6631 + 0.2160j
    sizes = meshing.element_sizes({1: hpfem.Material.vacuum(), 2: glass, 3: silver}, OMEGA, p=3)
    assert sizes[1] == pytest.approx(405 * NM / 4)
    assert sizes[2] == pytest.approx(405 * NM / 1.5 / 4)
    kappa = complex(silver.refractive_index).imag
    assert sizes[3] == pytest.approx(1.0 / (units.vacuum_wavenumber(OMEGA) * kappa))
    assert meshing.element_sizes({2: 1.5 + 0j}, OMEGA, 2, elements_per_wavelength=10)[
        2
    ] == pytest.approx(405 * NM / 15)


def test_unit_cell_tags_by_priority():
    cell = meshing.UnitCell(
        period=1.0,
        y_bottom=-1.0,
        y_top=1.0,
        slabs=[meshing.Slab(2, -1.0, 0.0)],
        shapes=[
            meshing.Shape("rectangle", 3, dict(x=-0.2, y=0.0, width=0.4, height=0.3)),
            meshing.Shape("ellipse", 4, dict(x=0.45, y=0.5, rx=0.1, ry=0.2)),
            meshing.Shape("trapezoid", 5, dict(x=0.0, y=-0.5, bottom=0.4, top=0.2, height=0.2)),
        ],
        background_tag=1,
    )
    assert cell.tag_at(0.0, 0.5) == 1
    assert cell.tag_at(0.0, -0.8) == 2
    assert cell.tag_at(0.1, 0.1) == 3
    assert cell.tag_at(0.46, 0.5) == 4
    assert cell.tag_at(-0.52, 0.5) == 4  # the periodic copy of the ellipse
    assert cell.tag_at(0.0, -0.45) == 5  # a shape wins over the slab
    assert cell.left == -0.5


def test_report_and_periodic_check_of_a_structured_cell():
    mesh = hpfem.rectangle(4, 6, [-0.5, -1.0], [0.5, 1.0])
    for c in range(mesh.num_cells):
        if np.mean([mesh.vertex(int(v)) for v in mesh.cell_vertices(c)], axis=0)[1] < 0:
            mesh.set_cell_tag(c, 2)
    r = meshing.report(mesh, {2: hpfem.Material.dielectric(1.5), 9: hpfem.Material.vacuum()})
    assert r["num_cells"] == 48
    assert r["min_angle"] == pytest.approx(np.arctan2(0.25, 1 / 3))  # 0.25 x 1/3 cells
    assert r["num_untagged"] == 24
    assert r["cell_tags"] == [2]
    assert r["tags_without_material"] == []
    assert r["materials_without_cells"] == [9]
    check = hpfem.check_periodic(mesh, hpfem.box_tag.X_MIN, hpfem.box_tag.X_MAX, [1.0, 0.0])
    assert check["identical"] and check["matched"] == 6
    assert (
        hpfem.check_periodic(mesh, hpfem.box_tag.X_MIN, hpfem.box_tag.X_MAX, [1.0, 0.05])[
            "unmatched_slave"
        ]
        == 6
    )


GMSH_PERIODIC_SQUARE = """$MeshFormat
4.1 0 8
$EndMeshFormat
$PhysicalNames
3
1 2 "right"
1 4 "left"
2 10 "domain"
$EndPhysicalNames
$Entities
4 4 1 0
1 0 0 0 0
2 1 0 0 0
3 1 1 0 0
4 0 1 0 0
1 0 0 0 1 0 0 0 2 1 -2
2 1 0 0 1 1 0 1 2 2 2 -3
3 0 1 0 1 1 0 0 2 3 -4
4 0 0 0 0 1 0 1 4 2 4 -1
1 0 0 0 1 1 0 1 10 4 1 2 3 4
$EndEntities
$Nodes
1 4 1 4
2 1 0 4
1
2
3
4
0 0 0
1 0 0
1 1 0
0 1 0
$EndNodes
$Elements
3 4 1 4
1 2 1 1
1 2 3
1 4 1 1
2 4 1
2 1 2 2
3 1 2 3
4 1 3 4
$EndElements
$Periodic
1
1 2 4
16 1 0 0 1 0 1 0 0 0 0 1 0 0 0 0 1
2
2 1
3 4
$EndPeriodic
"""


def test_read_gmsh_periodic_from_a_string():
    mesh, links = hpfem.read_gmsh_string_periodic(GMSH_PERIODIC_SQUARE, scale=2e-9, dim=2)
    assert mesh.num_cells == 2
    assert len(links) == 1
    master, slave, shift = links[0]
    assert (master, slave) == (4, 2)
    assert np.allclose(shift, [2e-9, 0.0])
    pair = hpfem.PeriodicPair2D(master, slave, shift, 1.0)
    assert pair.master == 4


def test_gmsh_unit_cell_meshes_and_solves():
    gmsh = pytest.importorskip("gmsh")
    try:
        gmsh.initialize(interruptible=False)
        gmsh.finalize()
    except Exception as error:  # pragma: no cover - missing system libraries
        pytest.skip(f"gmsh cannot start: {error}")
    glass = hpfem.Material.dielectric(1.5)
    cell = meshing.UnitCell(
        period=400 * NM,
        y_bottom=-800 * NM,
        y_top=950 * NM,
        slabs=[meshing.Slab(2, -800 * NM, 0.0)],
        shapes=[
            meshing.Shape("rectangle", 3, dict(x=-100 * NM, y=0.0, width=200 * NM, height=148 * NM))
        ],
        background_tag=1,
    )
    sizes = meshing.element_sizes({1: hpfem.Material.vacuum(), 2: glass, 3: glass}, OMEGA, p=3)
    mesh, links = meshing.unit_cell_mesh(cell, sizes)
    r = meshing.report(mesh, {1: hpfem.Material.vacuum(), 2: glass, 3: glass})
    assert r["num_untagged"] == 0 and r["tags_without_material"] == []
    assert sorted(r["facet_tags"]) == [1, 2, 3, 4]
    assert r["num_invalid"] == 0
    assert len(links) == 1 and (links[0][0], links[0][1]) == (
        hpfem.box_tag.X_MIN,
        hpfem.box_tag.X_MAX,
    )
    assert hpfem.check_periodic(mesh, links[0][0], links[0][1], links[0][2])["identical"]
    stack = hpfem.LayerStack2D(hpfem.Material.dielectric(1.0), [], glass, 0.0)
    result = grating.solve(
        mesh,
        {1: hpfem.Material.vacuum(), 2: glass, 3: glass},
        stack,
        "p",
        50 * units.deg,
        30 * units.deg,
        OMEGA,
        order=3,
        orders_max=2,
    )
    r = {o.m: o.efficiency for o in result.R_orders if o.propagating}
    assert abs(r[0] - 0.010957) < 3e-3
    assert abs(result.power_balance_residual) < 5e-3

"""Design parameters and mesh morphing (M16 S1, ADR-0012 §3): the CD / height / side-wall
angle of a trapezoidal line round-trip, the mesh velocity moves the nodes of the line's
boundary with it and keeps the cell boundary and the substrate interface fixed, the morphed
mesh keeps its tags and puts the boundary nodes on the new trapezoid, the quality guard stops
inverting moves, and the efficiency derivatives along the velocities (grating.jacobian) match
finite differences of solves on morphed meshes. The trapezoid cell is a structured mesh whose
columns follow the slanted walls (no Gmsh needed)."""

import math

import numpy as np
import pytest

import hpfem
from hpfem import grating, units
from hpfem.meshing import Shape, Slab, UnitCell
from hpfem.opt import (
    GeometryParameter,
    MaterialParameter,
    MeshQualityError,
    Morph,
    cell_quality,
    shape_velocity,
    trapezoid_parameters,
)

NM = units.nm
PERIOD = 400 * NM
HEIGHT = 148 * NM
OMEGA = units.angular_frequency(wavelength=405 * NM)
SUB, LINE = 2, 3
ROW = HEIGHT / 6
AIR_ROWS, SUB_ROWS, PML_ROWS = 15, 16, 17
Y0 = -(SUB_ROWS + PML_ROWS) * ROW
Y1 = (6 + AIR_ROWS + PML_ROWS) * ROW


def trapezoid_cell(bottom=220 * NM, top=180 * NM):
    shape = Shape("trapezoid", LINE, {"x": 0.0, "y": 0.0, "bottom": bottom, "top": top,
                                      "height": HEIGHT})  # fmt: skip
    return UnitCell(PERIOD, Y0, Y1, slabs=[Slab(SUB, Y0, 0.0)], shapes=[shape])


def trapezoid_mesh(cell, nx=16):
    """Structured cell whose columns |x| <= PERIOD/4 follow the walls of the trapezoid."""
    ny = SUB_ROWS + PML_ROWS + 6 + AIR_ROWS + PML_ROWS
    mesh = hpfem.rectangle(nx, ny, [-PERIOD / 2, Y0], [PERIOD / 2, Y1])
    p = cell.shapes[0].params
    for v in range(mesh.num_vertices):
        x, y = mesh.vertex(v)
        if 0.0 <= y <= HEIGHT + 1e-15:
            w = 0.5 * p["bottom"] + 0.5 * (p["top"] - p["bottom"]) * y / HEIGHT
            q = PERIOD / 4  # the grid column that becomes the wall
            if abs(x) <= q:
                xn = x * w / q
            else:
                xn = math.copysign(w + (abs(x) - q) * (PERIOD / 2 - w) / (PERIOD / 2 - q), x)
            mesh.set_vertex(v, [xn, y])
    for c in range(mesh.num_cells):
        centroid = np.mean([mesh.vertex(int(v)) for v in mesh.cell_vertices(c)], axis=0)
        mesh.set_cell_tag(c, cell.tag_at(*centroid))
    return mesh


def distance_to_polygon(points, polygon):
    corners = np.asarray(polygon, dtype=float)
    out = np.full(len(points), np.inf)
    for i in range(len(corners)):
        c0, c1 = corners[i], corners[(i + 1) % len(corners)]
        d = c1 - c0
        t = np.clip(((points - c0) @ d) / (d @ d), 0.0, 1.0)
        out = np.minimum(out, np.linalg.norm(points - (c0 + np.outer(t, d)), axis=1))
    return out


def test_trapezoid_parameters_round_trip():
    cell = trapezoid_cell()
    cd, height, angle = trapezoid_parameters(0)
    assert cd.get(cell) == pytest.approx(200 * NM)
    assert height.get(cell) == pytest.approx(HEIGHT)
    assert angle.get(cell) == pytest.approx(math.atan2(HEIGHT, 20 * NM))
    wider = cd.apply(cell, 210 * NM)
    assert cell.shapes[0].params["bottom"] == pytest.approx(220 * NM)  # the original is unchanged
    assert cd.get(wider) == pytest.approx(210 * NM)
    assert angle.get(wider) == pytest.approx(angle.get(cell))
    assert height.get(wider) == pytest.approx(HEIGHT)
    vertical = angle.apply(cell, math.pi / 2)
    p = vertical.shapes[0].params
    assert p["bottom"] == pytest.approx(p["top"]) and p["bottom"] == pytest.approx(200 * NM)
    width = GeometryParameter.field("height", 0, "height", 100 * NM, 200 * NM)
    assert width.typical(cell) == pytest.approx(100 * NM)
    eps = MaterialParameter("n2", LINE, "im", 0.0, 1.0)
    materials = {LINE: hpfem.Material(eps_r=2.25 + 0.1j)}
    assert eps.get(materials) == pytest.approx(0.1)
    changed = eps.apply(materials, 0.3)
    assert complex(changed[LINE].eps_r) == pytest.approx(2.25 + 0.3j)
    assert complex(materials[LINE].eps_r) == pytest.approx(2.25 + 0.1j)
    with pytest.raises(ValueError):
        MaterialParameter("bad", LINE, "abs")


def test_velocity_moves_the_boundary_and_keeps_the_interfaces():
    cell = trapezoid_cell()
    mesh = trapezoid_mesh(cell)
    x = np.asarray(mesh.vertices)
    boundary = np.zeros(mesh.num_vertices, dtype=bool)
    for f in mesh.boundary_facets:
        boundary[list(mesh.facet_vertices(int(f)))] = True
    on_line = distance_to_polygon(x, cell.shapes[0].polygon()) < 1e-9 * PERIOD
    on_substrate = (np.abs(x[:, 1]) < 1e-15) & ~on_line
    for parameter in trapezoid_parameters(0):
        v = shape_velocity(cell, mesh, parameter)
        assert v.shape == (hpfem.num_geometry_nodes(mesh), 2)
        assert np.all(v[boundary] == 0.0)
        assert np.all(v[on_substrate] == 0.0)  # the substrate interface stays put
        largest = np.abs(v[on_line]).max()  # m per unit of the parameter (m/rad for the angle)
        assert largest > 0.0
        free = ~(boundary | on_line | on_substrate)
        moving = np.linalg.norm(v[free], axis=1) > 1e-6 * largest  # harmonic extension
        assert np.count_nonzero(moving) > 0
    # the height at fixed CD and angle: the top edge rises by one and narrows by
    # (bottom - top) / (2 height) in total, the base widens by as much (on the substrate line)
    v = shape_velocity(cell, mesh, trapezoid_parameters(0)[1])
    p = cell.shapes[0].params
    slope = (p["bottom"] - p["top"]) / (4 * p["height"])  # per corner
    top = on_line & (np.abs(x[:, 1] - HEIGHT) < 1e-15)
    expected_top = np.column_stack([-x[top, 0] / (p["top"] / 2) * slope, np.ones(top.sum())])
    assert np.allclose(v[top], expected_top, atol=1e-6)
    base = on_line & (np.abs(x[:, 1]) < 1e-15)
    expected_base = np.column_stack([x[base, 0] / (p["bottom"] / 2) * slope, np.zeros(base.sum())])
    assert np.allclose(v[base], expected_base, atol=1e-6)
    # the plain field "height" keeps top and bottom: the top edge rises by one
    v = shape_velocity(cell, mesh, GeometryParameter.field("h", 0, "height"))
    assert np.allclose(v[top], [0.0, 1.0], atol=1e-6)
    # a shape that reaches the Bloch faces cannot move them
    full = {"x": -PERIOD / 2, "y": 0.0, "width": PERIOD, "height": HEIGHT}
    slab_like = UnitCell(PERIOD, Y0, Y1, slabs=[Slab(SUB, Y0, 0.0)],
                         shapes=[Shape("rectangle", LINE, full)])  # fmt: skip
    with pytest.raises(ValueError):
        shape_velocity(slab_like, trapezoid_mesh(cell),
                       GeometryParameter.field("w", 0, "width"))  # fmt: skip


def test_morph_puts_the_boundary_on_the_new_trapezoid_and_guards_quality():
    cell = trapezoid_cell()
    mesh = trapezoid_mesh(cell)
    parameters = trapezoid_parameters(0)
    morph = Morph(cell, mesh, parameters)
    x = np.asarray(mesh.vertices)
    on_line = distance_to_polygon(x, cell.shapes[0].polygon()) < 1e-9 * PERIOD
    values = {"cd": 205 * NM, "height": 150 * NM}  # linear in the corners: exact morph
    moved = morph.mesh_at(values)
    assert np.array_equal(np.asarray(moved.cell_tags), np.asarray(mesh.cell_tags))
    target = morph.cell_at(values).shapes[0].polygon()
    xm = np.asarray(moved.vertices)
    assert distance_to_polygon(xm[on_line], target).max() < 1e-12 * PERIOD
    assert morph.check(moved) > 0.3
    # the side-wall angle is nonlinear in the corners: second-order deviation
    step = 1e-3
    tilted = morph.mesh_at({"angle": morph.reference["angle"] + step})
    target = morph.cell_at({"angle": morph.reference["angle"] + step}).shapes[0].polygon()
    deviation = distance_to_polygon(np.asarray(tilted.vertices)[on_line], target).max()
    assert deviation < 10 * step**2 * HEIGHT
    # a collapsing line inverts cells: the guard refuses and reports
    with pytest.raises(MeshQualityError) as error:
        morph.mesh_at({"height": 2 * NM})
    assert error.value.inverted > 0 or error.value.ratio < 0.3
    with pytest.raises(ValueError):
        morph.mesh_at({"width": 1.0})
    assert np.all(cell_quality(mesh) > 0)


def test_derivatives_along_the_velocities_match_finite_differences():
    cell = trapezoid_cell()
    mesh = trapezoid_mesh(cell)
    morph = Morph(cell, mesh, trapezoid_parameters(0))
    glass = hpfem.Material.dielectric(1.5)
    stack = hpfem.LayerStack2D(hpfem.Material.dielectric(1.0), [], glass, 0.0)
    materials = {SUB: glass, LINE: hpfem.Material.dielectric(1.8)}
    pml = PML_ROWS * ROW
    common = dict(order=3, pml={"top": pml, "bottom": pml}, orders_max=2, check=False)

    def solve(m, keep=False):
        return grating.solve(m, materials, stack, "s", 40 * units.deg, 0.0, OMEGA,
                             keep_factorisation=keep, **common)  # fmt: skip

    def r0(res):
        return next(o for o in res.R_orders if o.m == 0).efficiency

    result = solve(mesh, keep=True)
    names = ["cd", "height", "angle"]
    jac, rows, _ = grating.jacobian(
        result, [("shape", morph.velocity(n)) for n in names], observables=[("R", 0)]
    )
    steps = {"cd": 1e-11, "height": 1e-11, "angle": 1e-5}
    for j, name in enumerate(names):
        ref = morph.reference[name]
        plus = r0(solve(morph.mesh_at({name: ref + steps[name]})))
        minus = r0(solve(morph.mesh_at({name: ref - steps[name]})))
        fd = (plus - minus) / (2 * steps[name])
        assert abs(jac[0, j] - fd) < 1e-3 * abs(fd), (name, jac[0, j], fd)

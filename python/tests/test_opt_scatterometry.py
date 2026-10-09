"""Scatterometry evaluator (M16 S3, hpfem.opt.GratingEvaluator): the efficiencies of a line
grating under two configurations as a function of CD, height, side-wall angle and the line's
permittivity; the values equal a plain grating.solve on the same frame, the Jacobian matches
finite differences of the evaluator on morphed meshes, the settings hash is reproducible, a
study caches it, and the quality guard remeshes (with a mesher) or fails the evaluation.
Structured cell without Gmsh: the line's vertical walls lie on column lines."""

import math

import numpy as np
import pytest

import hpfem
from hpfem import grating, meshing, units
from hpfem.meshing import Shape, Slab, UnitCell
from hpfem.opt import (
    Configuration,
    Continuous,
    DesignSpace,
    GratingEvaluator,
    MaterialParameter,
    Morph,
    Study,
    trapezoid_parameters,
)

NM = units.nm
PERIOD = 400 * NM
HEIGHT = 150 * NM
WIDTH = 200 * NM
SUB, LINE = 2, 3
ROW = HEIGHT / 6
AIR_ROWS, SUB_ROWS, PML_ROWS = 12, 12, 14
Y0 = -(SUB_ROWS + PML_ROWS) * ROW
Y1 = HEIGHT + (AIR_ROWS + PML_ROWS) * ROW


def line_cell(width=WIDTH, height=HEIGHT):
    shape = Shape("trapezoid", LINE, {"x": 0.0, "y": 0.0, "bottom": width, "top": width,
                                      "height": height})  # fmt: skip
    return UnitCell(PERIOD, Y0, Y1, slabs=[Slab(SUB, Y0, 0.0)], shapes=[shape], background_tag=1)


def line_mesh(cell):
    h = cell.shapes[0].params["height"]
    rows = [(Y0, 0.0, SUB_ROWS + PML_ROWS), (0.0, h, 6), (h, Y1, AIR_ROWS + PML_ROWS)]
    return meshing.structured_unit_cell(cell, 16, rows)


# the default measurement lines of the cell (midway to the PML, y = -150 nm and 300 nm) lie on
# mesh rows; the morph keeps the cells at them and everything beyond fixed (one row inside)
LINES = (-150 * NM, 300 * NM)
BAND = (LINES[0] + ROW, LINES[1] - ROW)


def evaluator(mesher=None, threshold=0.3, order=3, band=BAND):
    cell = line_cell()
    morph = Morph(cell, line_mesh(cell), trapezoid_parameters(0), quality_threshold=threshold,
                  band=band)  # fmt: skip
    glass = hpfem.Material.dielectric(1.5)
    stack = hpfem.LayerStack2D(hpfem.Material.dielectric(1.0), [], glass, 0.0)
    materials = {SUB: glass, LINE: hpfem.Material(eps_r=3.2 + 0.1j)}
    configurations = [
        Configuration(405 * NM, 40 * units.deg, 0.0, "s", (("R", 0), ("T", 0))),
        Configuration(500 * NM, 20 * units.deg, 0.0, "p", (("R", 0), ("R", -1))),
    ]
    return GratingEvaluator(
        morph, materials, stack, configurations,
        material_parameters=[MaterialParameter("eps_line", LINE, "re", 2.0, 5.0)],
        order=order, pml={"top": PML_ROWS * ROW, "bottom": PML_ROWS * ROW}, mesher=mesher,
        solve_options={"orders_max": 2},
    )  # fmt: skip


REFERENCE = {"cd": WIDTH, "height": HEIGHT, "angle": math.pi / 2, "eps_line": 3.2}


def space():
    return DesignSpace([
        Continuous("cd", 150 * NM, 250 * NM), Continuous("height", 50 * NM, 200 * NM),
        Continuous("angle", 1.3, 1.75), Continuous("eps_line", 2.0, 5.0),
    ])  # fmt: skip


def test_values_match_a_plain_solve_and_the_jacobian_matches_finite_differences():
    ev = evaluator()
    assert ev.names == ["cd", "height", "angle", "eps_line"]
    assert ev.observables == ["s 405nm 40deg: R0", "s 405nm 40deg: T0", "p 500nm 20deg: R0",
                              "p 500nm 20deg: R-1"]  # fmt: skip
    for name, value in ev.morph.reference.items():
        assert value == pytest.approx(REFERENCE[name], rel=1e-12)
    result = ev(REFERENCE, jacobian=True)
    assert result.ok and result.values.shape == (4,) and result.jacobian.shape == (4, 4)
    assert result.meta["quality"] == pytest.approx(1.0) and result.meta["dofs"] > 0
    # the first configuration by hand on the same frame
    frame = ev._frames[0]
    plain = grating.solve(ev.morph.mesh, ev.materials, ev.stack, "s", 40 * units.deg, 0.0,
                          units.angular_frequency(wavelength=405 * NM), 3, pml=frame["pml"],
                          cover_line=frame["cover_line"], substrate_line=frame["substrate_line"],
                          orders_max=2, check=False)  # fmt: skip
    r0 = next(o.efficiency for o in plain.R_orders if o.m == 0)
    t0 = next(o.efficiency for o in plain.T_orders if o.m == 0)
    assert result.values[0] == pytest.approx(r0, rel=1e-12)
    assert result.values[1] == pytest.approx(t0, rel=1e-12)
    # the Jacobian against central differences of the evaluator (morphed meshes)
    steps = {"cd": 1e-11, "height": 1e-11, "angle": 1e-5, "eps_line": 1e-4}
    for j, name in enumerate(ev.names):
        plus = ev({**REFERENCE, name: REFERENCE[name] + steps[name]}).values
        minus = ev({**REFERENCE, name: REFERENCE[name] - steps[name]}).values
        fd = (plus - minus) / (2 * steps[name])
        scale = np.abs(fd).max()
        assert scale > 0, name
        assert np.abs(result.jacobian[:, j] - fd).max() < 1e-5 * scale, (
            name,
            result.jacobian[:, j],
            fd,
        )
    # a lower order as fidelity
    low = ev(REFERENCE, fidelity={"order": 2})
    assert low.meta["order"] == 2 and np.abs(low.values - result.values).max() < 0.05


def test_settings_hash_is_reproducible_and_a_study_caches_the_evaluations(tmp_path):
    a, b = evaluator(), evaluator()
    path = tmp_path / "line.study.jsonl"
    study = Study(a, path, space())
    first = study.evaluate(REFERENCE)
    again = study.evaluate(REFERENCE)
    assert again is first and study.cache_hits == 1
    resumed = Study(b, path, space())  # a new evaluator with the same settings reuses the store
    assert resumed.evaluate(REFERENCE).values == pytest.approx(first.values, rel=0, abs=0)
    assert resumed.cache_hits == 1
    changed = Study(evaluator(order=2), path, space())  # a changed setting never reuses old values
    assert changed.evaluate(REFERENCE).meta["order"] == 2 and changed.cache_hits == 0


def test_quality_guard_remeshes_with_a_mesher_and_fails_without():
    far = {**REFERENCE, "height": 0.4 * HEIGHT}  # squeezes the line rows below the threshold
    plain = evaluator(threshold=0.6)
    failed = Study(plain, space=space()).evaluate(far)
    assert failed.status == "failed" and "remesh" in failed.meta["error"]
    remeshing = evaluator(mesher=line_mesh, threshold=0.6)
    study = Study(remeshing, space=space())
    result = study.evaluate(far, jacobian=True)
    assert result.ok and result.mesh_id == 1 and remeshing.mesh_id == 1
    assert len(study.remeshes) == 1 and study.remeshes[0]["mesh_id"] == 1
    assert remeshing.morph.reference["height"] == pytest.approx(0.4 * HEIGHT)
    assert result.meta["quality"] == pytest.approx(1.0)


def test_moving_mesh_lines_across_a_measurement_line_is_refused():
    # without the band the harmonic extension moves the mesh row of the cover line: the
    # p-polarised (Nedelec) order amplitude is not differentiable there
    with pytest.raises(ValueError, match="band"):
        evaluator(band=None)
    ev = evaluator()
    assert ev._frames[1]["cover_line"] == pytest.approx(LINES[1])
    assert ev._frames[1]["substrate_line"] == pytest.approx(LINES[0])
    with pytest.raises(ValueError, match="one cell row"):
        evaluator(band=LINES)  # fixed nodes on the lines are not enough
    free = Morph(ev.morph.cell, ev.morph.mesh, trapezoid_parameters(0))
    frame = ev._frames[1]
    config = ev.configurations[1]
    result = grating.solve(ev.morph.mesh, ev.materials, ev.stack, "p", config.theta, 0.0,
                           config.omega, 3, pml=frame["pml"], cover_line=frame["cover_line"],
                           substrate_line=frame["substrate_line"], orders_max=2, check=False,
                           keep_factorisation=True)  # fmt: skip
    with pytest.raises(grating.GratingError, match="not differentiable"):
        grating.jacobian(result, [("shape", free.velocity("height"))])
    with pytest.raises(grating.GratingError, match="not differentiable"):
        grating.shape_sensitivity(result, free.velocity("angle"))
    # the banded velocity vanishes on and beyond the lines
    y = np.asarray(ev.morph.mesh.vertices)[:, 1]
    outside = (y <= BAND[0] * (1 - 1e-9)) | (y >= BAND[1] * (1 - 1e-9))
    for p in ev.morph.parameters:
        assert not ev.morph.velocity(p.name)[: len(y)][outside].any()


def test_configuration_checks():
    with pytest.raises(ValueError):
        Configuration(405 * NM, 0.3, orders=(("X", 0),))
    with pytest.raises(ValueError):
        Configuration(405 * NM, 0.3, orders=())
    cell = line_cell()
    morph = Morph(cell, line_mesh(cell), trapezoid_parameters(0))
    stack = hpfem.LayerStack2D(hpfem.Material.dielectric(1.0), [], hpfem.Material.dielectric(1.5))
    with pytest.raises(ValueError, match="plain"):
        GratingEvaluator(morph, {LINE: hpfem.materials.Drude(1.0, 1e16, 1e14)}, stack,
                         [Configuration(405 * NM, 0.3)],
                         material_parameters=[MaterialParameter("e", LINE)])  # fmt: skip
    with pytest.raises(ValueError, match="no configurations"):
        GratingEvaluator(morph, {}, stack, [])

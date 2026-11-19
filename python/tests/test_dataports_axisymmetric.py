"""Emitter response tables of bodies of revolution by the axisymmetric FEM (issue #214, stage 3,
``hpfem.dataports.axisymmetric_emitter_response``): a flat stack against the semi-analytic
planar table, a hole in an absorbing film, the refusals."""

import json

import numpy as np
import pytest

import hpfem
from hpfem import dataports as dp

UM = 1e-6
LAM = 1.0 * UM
WATER = hpfem.Material.dielectric(1.33)
GLASS = hpfem.Material.dielectric(1.52)
FILM = hpfem.Material(complex(4.0, 1.0), 1.0)  # an absorbing film, 0.1 um
SOLUTION, SUBSTRATE, LAYER = 1, 2, 3
MATERIALS = {SOLUTION: WATER, SUBSTRATE: GLASS, LAYER: FILM}
STACK = hpfem.LayerStack2D(WATER, [hpfem.Layer(FILM, 0.1 * UM)], GLASS, 0.0)
PML = {"r": 0.5 * UM, "top": 0.5 * UM, "bottom": 0.5 * UM}


def _lines(segments):
    out = [segments[0][0]]
    for a, b, step in segments:
        out += list(np.linspace(a, b, max(1, round((b - a) / step)) + 1)[1:])
    return np.array(out) * UM


R_LINES = _lines([(0.0, 0.4, 0.05), (0.4, 1.6, 0.1)])
Z_LINES = _lines([(-1.5, -0.1, 0.1), (-0.1, 0.4, 0.05), (0.4, 1.4, 0.1)])


def _flat(rr, zz):
    return SUBSTRATE if zz < -0.1 * UM else (LAYER if zz < 0 else SOLUTION)


def _hole(rr, zz):
    if zz < -0.1 * UM:
        return SUBSTRATE
    return SOLUTION if (zz >= 0 or rr < 0.2 * UM) else LAYER


def test_meridian_mesh_tags_and_axis():
    mesh = dp.meridian_mesh(R_LINES, Z_LINES, _hole)
    assert mesh.num_cells == 2 * (len(R_LINES) - 1) * (len(Z_LINES) - 1)
    axis = [f for f in mesh.boundary_facets if mesh.facet_tag(int(f)) == 77]
    assert len(axis) == len(Z_LINES) - 1
    tags = set(np.asarray(mesh.cell_tags).tolist())
    assert tags == {SOLUTION, SUBSTRATE, LAYER}
    with pytest.raises(ValueError, match="start at 0"):
        dp.meridian_mesh(R_LINES + 1e-7, Z_LINES, _flat)


def test_flat_absorbing_film_against_the_planar_table():
    """A flat stack in the axisymmetric FEM against the semi-analytic planar table (stage 1):
    A_tot, A_rad and A_det of both orientations to 1e-2 at p = 3 on this coarse mesh (1.4e-4
    on the twice finer one for water on glass); the absorption of the film appears as
    A_tot − A_rad."""
    mesh = dp.meridian_mesh(R_LINES, Z_LINES, _flat)
    heights = [0.15 * UM, 0.3 * UM]
    fem = dp.axisymmetric_emitter_response(mesh, MATERIALS, STACK, LAM, heights, pml=PML,
                                           order=3, sigma=0.02 * UM)  # fmt: skip
    ref = dp.planar_emitter_response(STACK, LAM, heights)
    for name in ("a_tot", "a_rad"):
        got = np.diagonal(getattr(fem.table, name)[0, 0], axis1=1, axis2=2)
        want = np.diagonal(getattr(ref.table, name)[0, 0], axis1=1, axis2=2)
        assert np.abs(got / want - 1).max() < 1e-2, (name, got / want - 1)
    got = np.diagonal(fem.table.a_det[0, 0], axis1=2, axis2=3)
    want = np.diagonal(ref.table.a_det[0, 0], axis1=2, axis2=3)
    assert np.abs(got / want - 1).max() < 1e-2
    assert fem.convergence["energy"] > 0.1  # the film absorbs
    assert np.all(fem.table.a_tot[0, 0, :, 0, 1] == 0)  # diagonal on the axis
    conv = json.loads(fem.table.source["convergence"])
    assert conv["order"] == 3 and conv["dofs"] > 0
    assert "stack" not in fem.table.source  # a structure, not a planar stack (BindOptiX T4)


def test_hole_in_the_film():
    """An emitter in a hole of the absorbing film and one above it: a valid table (the checks of
    BindOptiX), the emitter in the hole couples more to the film than above it."""
    mesh = dp.meridian_mesh(R_LINES, Z_LINES, _hole)
    fem = dp.axisymmetric_emitter_response(mesh, MATERIALS, STACK, LAM,
                                           [-0.05 * UM, 0.15 * UM], pml=PML, order=3,
                                           sigma=0.008 * UM)  # fmt: skip
    t = fem.table
    assert t.n_host == pytest.approx(1.33)
    absorbed = np.diagonal(t.a_tot - t.a_rad, axis1=3, axis2=4)[0, 0]
    assert np.all(absorbed[0] > absorbed[1])
    data = json.loads(t.to_json())
    assert data["type"] == "EmitterResponseTable" and len(t.source["content_hash"]) == 64
    flat = dp.axisymmetric_emitter_response(dp.meridian_mesh(R_LINES, Z_LINES, _flat), MATERIALS,
                                            STACK, LAM, [0.15 * UM], pml=PML, order=3,
                                            sigma=0.008 * UM)  # fmt: skip
    assert abs(t.a_tot[0, 0, 1, 2, 2] / flat.table.a_tot[0, 0, 0, 2, 2] - 1) > 1e-3


def test_refusals():
    mesh = dp.meridian_mesh(R_LINES, Z_LINES, _flat)
    with pytest.raises(ValueError, match="reaches"):
        dp.axisymmetric_emitter_response(mesh, MATERIALS, STACK, LAM, [0.05 * UM], pml=PML,
                                         sigma=0.02 * UM)  # fmt: skip
    with pytest.raises(ValueError, match="PML"):
        dp.axisymmetric_emitter_response(mesh, MATERIALS, STACK, LAM, [1.2 * UM], pml=PML)
    with pytest.raises(ValueError, match="Jones"):
        dp.axisymmetric_emitter_response(
            mesh, MATERIALS, STACK, LAM, [0.3 * UM], pml=PML,
            detectors=[dp.Detector("a", "cover", jones=lambda t, p: np.eye(2))])  # fmt: skip
    with pytest.raises(ValueError, match="without a material"):
        dp.axisymmetric_emitter_response(mesh, {SOLUTION: WATER}, STACK, LAM, [0.3 * UM], pml=PML)

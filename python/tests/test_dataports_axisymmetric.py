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


def test_heights_on_one_factorisation_equal_separate_solves():
    """The heights after the first are solved on the kept factorisation with the load on the
    support of their Gaussian: the same table as one call per height."""
    mesh = dp.meridian_mesh(R_LINES, Z_LINES, _hole)
    heights = [0.15 * UM, 0.3 * UM]
    args = (mesh, MATERIALS, STACK, LAM)
    opts = {"pml": PML, "order": 2, "sigma": 0.02 * UM}
    both = dp.axisymmetric_emitter_response(*args, heights, **opts).table
    for i, h in enumerate(heights):
        one = dp.axisymmetric_emitter_response(*args, [h], **opts).table
        for name in ("a_tot", "a_rad", "a_det"):
            got, want = getattr(both, name)[0, 0, i], getattr(one, name)[0, 0, 0]
            assert np.abs(got - want).max() < 1e-10 * np.abs(want).max(), name


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


# --- field maps of bodies of revolution -----------------------------------------------------


def test_axisymmetric_field_map_without_a_body_equals_the_planar_map():
    """The flat film in the axisymmetric map against the exact stack field: normal incidence (the
    orders m = ±1) to 1e-7 and oblique incidence through the glass (|m| ≤ 4, Jacobi–Anger) to
    1e-4, with the same phase reference, medium labels and absorption density."""
    mesh = dp.meridian_mesh(R_LINES, Z_LINES, _flat)
    grid = dp.RegularGrid(([-0.1, 0.0, 0.1], [0.0, 0.08], [-0.1, -0.05, 0.0, 0.1, 0.3]))
    for k, pol, m_max in (([0.0, 0.0, -1.0], [1.0, 0.0, 0.0], None),
                          ([0.3, 0.0, np.sqrt(1 - 0.09)], [0.0, 1.0, 0.0], 4)):  # fmt: skip
        fem = dp.axisymmetric_field_map(mesh, MATERIALS, STACK, LAM, grid, k, pol, pml=PML,
                                        max_order=m_max)  # fmt: skip
        ref = dp.planar_field_map(STACK, LAM, grid, k, pol)
        tol = 1e-7 if m_max is None else 1e-4  # |m| <= 4: J_5 of k_rho r; 5e-7 at |m| <= 6
        assert np.abs(fem.e - ref.e).max() < tol * np.abs(ref.e).max()
        assert np.abs(fem.q_w_m3 - ref.q_w_m3).max() < tol * np.abs(ref.q_w_m3).max()
        assert fem.irradiance_w_m2 == pytest.approx(ref.irradiance_w_m2)
        assert fem.media == ("tag 1", "tag 3")
        assert np.array_equal(fem.medium[..., 2], np.zeros((3, 2), dtype=int))  # z = 0: water
    assert json.loads(fem.source["azimuthal_orders"]) == list(range(-4, 5))
    # s polarised: nine orders on five factorisations (|m| = 0 … 4, A_{-m} = D A_m D)
    assert (fem.source["solves"], fem.source["factorisations"]) == ("9", "5")


def test_axisymmetric_field_map_of_a_hole_is_rotation_covariant():
    """A hole in the absorbing film at normal incidence: the map of the y-polarised wave is the
    map of the x-polarised one turned by 90° about the axis (the FEM fields of the orders ±1
    summed and rotated), the field differs from the flat film near the hole, the points in the
    hole are water."""
    mesh = dp.meridian_mesh(R_LINES, Z_LINES, _hole)
    axis = [-0.1, 0.0, 0.1]
    grid = dp.RegularGrid((axis, axis, [-0.05, 0.0, 0.05]))
    ex = dp.axisymmetric_field_map(mesh, MATERIALS, STACK, LAM, grid, [0, 0, -1], [1, 0, 0],
                                   pml=PML, media={SOLUTION: "water", LAYER: "film"})  # fmt: skip
    ey = dp.axisymmetric_field_map(mesh, MATERIALS, STACK, LAM, grid, [0, 0, -1], [0, 1, 0],
                                   pml=PML)  # fmt: skip
    turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    for i in range(3):
        for j in range(3):
            # the point (x, y) turned by 90° is (−y, x): grid index (2 − j, i)
            rotated = ex.e[i, j] @ turn.T  # rows: the z points
            assert np.abs(ey.e[2 - j, i] - rotated).max() < 1e-9 * np.abs(ex.e).max()
    assert ex.media == ("water",)  # the grid lies in the hole and above it
    flat = dp.planar_field_map(STACK, LAM, grid, [0, 0, -1], [1, 0, 0])
    assert np.abs(ex.e - flat.e).max() > 1e-2 * np.abs(flat.e).max()


# --- emitters off the axis ------------------------------------------------------------------


def test_offaxis_sum_at_the_axis_equals_the_on_axis_table():
    """``radii=[0]``: the sum over the azimuthal orders stops after m = 1 (the higher orders of a
    dipole on the axis vanish) and gives the on-axis table."""
    mesh = dp.meridian_mesh(R_LINES, Z_LINES, _hole)
    args = (mesh, MATERIALS, STACK, LAM, [-0.05 * UM, 0.15 * UM])
    opts = {"pml": PML, "order": 2, "sigma": 0.008 * UM}
    on = dp.axisymmetric_emitter_response(*args, **opts).table
    general = dp.axisymmetric_emitter_response(*args, radii=[0.0], **opts)
    assert general.convergence["azimuthal_orders"] == 3
    for name in ("a_tot", "a_rad", "a_det"):
        got, want = getattr(general.table, name), getattr(on, name)
        assert np.abs(got - want).max() < 1e-10 * np.abs(want).max(), name


def test_offaxis_emitters_in_a_flat_film_equal_the_planar_table():
    """A flat stack is invariant under translation: an emitter at ρ0 = 0.1 µm gives the planar
    table (1e-2 at p = 3, as on the axis; p = 2 is off by 4 % on the axis too) with A_xz = 0;
    the orders converge."""
    mesh = dp.meridian_mesh(R_LINES, Z_LINES, _flat)
    heights = [0.15 * UM]
    fem = dp.axisymmetric_emitter_response(mesh, MATERIALS, STACK, LAM, heights, pml=PML,
                                           order=3, sigma=0.02 * UM, radii=[0.1 * UM],
                                           order_tolerance=1e-6)  # fmt: skip
    ref = dp.planar_emitter_response(STACK, LAM, heights)
    for name in ("a_tot", "a_rad"):
        want = np.diagonal(getattr(ref.table, name)[0, 0, 0])
        got = getattr(fem.table, name)[0, 0, 0]
        assert np.abs(np.diagonal(got) / want - 1).max() < 1e-2, name
        assert abs(got[0, 2]) < 1e-2 * got[0, 0]
        assert got[0, 1] == got[1, 2] == 0.0
    det = np.diagonal(fem.table.a_det[0, 0, 0], axis1=1, axis2=2)
    want = np.diagonal(ref.table.a_det[0, 0, 0], axis1=1, axis2=2)
    assert np.abs(det / want - 1).max() < 1e-2
    conv = fem.convergence
    assert 3 < conv["azimuthal_orders"] < 20
    assert conv["factorisations"] == conv["azimuthal_orders"] + 1
    assert fem.table.grid.axes_um[0] == pytest.approx([0.1])


def test_offaxis_emitter_near_the_edge_of_a_hole():
    """An emitter above the film next to the edge of the hole: the mirror symmetry φ → −φ leaves
    A_xz ≠ 0 and A_xx ≠ A_yy (both vanish on the axis), a valid table for BindOptiX."""
    mesh = dp.meridian_mesh(R_LINES, Z_LINES, _hole)
    fem = dp.axisymmetric_emitter_response(mesh, MATERIALS, STACK, LAM, [0.1 * UM], pml=PML,
                                           order=2, sigma=0.015 * UM, radii=[0.0, 0.15 * UM],
                                           order_tolerance=1e-6)  # fmt: skip
    a = fem.table.a_tot[:, 0, 0]
    assert abs(a[0, 0, 2]) < 1e-12 * a[0, 0, 0]  # on the axis: diagonal, A_xx = A_yy
    assert a[0, 0, 0] == pytest.approx(a[0, 1, 1], rel=1e-10)
    assert abs(a[1, 0, 2]) > 1e-3 * a[1, 0, 0]
    assert abs(a[1, 0, 0] / a[1, 1, 1] - 1) > 1e-3
    data = json.loads(fem.table.to_json())
    assert data["type"] == "EmitterResponseTable"
    with pytest.raises(ValueError, match="radii"):
        dp.axisymmetric_emitter_response(mesh, MATERIALS, STACK, LAM, [0.1 * UM], pml=PML,
                                         radii=[0.1 * UM, 0.0])  # fmt: skip
    with pytest.raises(ValueError, match="did not converge"):
        dp.axisymmetric_emitter_response(mesh, MATERIALS, STACK, LAM, [0.1 * UM], pml=PML,
                                         order=2, sigma=0.015 * UM, radii=[0.15 * UM],
                                         max_azimuthal_order=2)  # fmt: skip

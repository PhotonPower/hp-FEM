"""meshio / pyvista interop and the matplotlib helpers (optional packages skipped if absent)."""

import numpy as np
import pytest

import hpfem
from hpfem import interop

meshio = pytest.importorskip("meshio")


def rotation_field(x):
    return np.array([x[1], -x[0]])


def test_sampling_on_the_subdivision_is_exact_on_affine_cells():
    mesh = hpfem.rectangle(3, 2)
    nd = hpfem.NedelecDofMap2D(mesh, 2)
    e = hpfem.interpolate(nd, rotation_field)
    sub, values = interop.sample_on_subdivision(nd, e, 3)
    assert sub.mesh.num_cells == 9 * mesh.num_cells
    expected = np.stack([sub.mesh.vertices[:, 1], -sub.mesh.vertices[:, 0]], axis=1)
    assert np.allclose(values, expected, atol=1e-12)
    _, curls = interop.sample_on_subdivision(nd, e, 3, curl=True)
    assert np.allclose(curls[:, 0], -2.0, atol=1e-12)
    h1 = hpfem.DofMap2D(mesh, 2)
    u = hpfem.interpolate(h1, lambda x: x[0] ** 2 + x[1])
    sub2, scalars = interop.sample_on_subdivision(h1, u, 2)
    assert np.allclose(
        scalars, sub2.mesh.vertices[:, 0] ** 2 + sub2.mesh.vertices[:, 1], atol=1e-12
    )
    with pytest.raises(ValueError):
        interop.sample_on_subdivision(h1, u, 2, curl=True)


def test_meshio_round_trip_with_tags_and_facets():
    mesh = hpfem.rectangle(3, 3)
    mesh.set_cell_tag(0, 7)
    m = interop.to_meshio(mesh, cell_data={"eta": np.arange(mesh.num_cells, dtype=float)})
    assert m.points.shape == (mesh.num_vertices, 3)
    types = {b.type: b for b in m.cells}
    assert types["triangle"].data.shape == (mesh.num_cells, 3) and "line" in types
    assert m.cell_data["cell_tag"][0][0] == 7 and len(m.cell_data["eta"][0]) == mesh.num_cells
    back = interop.from_meshio(m)
    assert back.num_cells == mesh.num_cells and back.num_vertices == mesh.num_vertices
    assert back.cell_tags.tolist() == mesh.cell_tags.tolist()
    assert sorted(back.facet_tags[back.boundary_facets]) == sorted(
        mesh.facet_tags[mesh.boundary_facets]
    )
    scaled = interop.from_meshio(m, scale=1e-6)
    assert np.isclose(scaled.vertices.max(), 1e-6)


def test_meshio_second_order_and_3d():
    disc = hpfem.disc(2)
    m = interop.to_meshio(disc)
    assert {b.type for b in m.cells} >= {"triangle6", "line"}
    back = interop.from_meshio(m)
    assert back.geometry_order == 2 and back.num_cells == disc.num_cells
    assert np.allclose(np.sort(back.edge_nodes, axis=0), np.sort(disc.edge_nodes, axis=0))
    box = hpfem.box(2, 1, 1)
    m3 = interop.to_meshio(box)
    assert {b.type for b in m3.cells} == {"tetra", "triangle"}
    back3 = interop.from_meshio(m3)
    assert back3.num_cells == box.num_cells and back3.dim == 3
    assert len(back3.facets_with_tag(hpfem.box_tag.Z_MAX)) == len(
        box.facets_with_tag(hpfem.box_tag.Z_MAX)
    )


def test_field_to_meshio_point_data():
    mesh = hpfem.rectangle(2, 2)
    nd = hpfem.NedelecDofMap2D(mesh, 2)
    e = hpfem.interpolate(nd, rotation_field)
    m = interop.field_to_meshio(nd, e, 2, name="E", cell_data={"tag": mesh.cell_tags})
    assert m.point_data["E_re"].shape == (m.points.shape[0], 3)
    assert np.allclose(m.point_data["curl_E_re"], -2.0)
    assert len(m.cell_data["tag"][0]) == 4 * mesh.num_cells


def test_pyvista_grid():
    pyvista = pytest.importorskip("pyvista")
    mesh = hpfem.rectangle(2, 2)
    grid = interop.to_pyvista(mesh)
    assert isinstance(grid, pyvista.UnstructuredGrid) and grid.n_cells == mesh.num_cells
    nd = hpfem.NedelecDofMap2D(mesh, 2)
    field = interop.field_to_pyvista(nd, hpfem.interpolate(nd, rotation_field), 2)
    assert "E_abs" in field.point_data


def test_matplotlib_helpers(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mesh = hpfem.square_with_disc(2, 0.25, 1.0, 2.0, 2)
    ax = interop.plot_mesh(mesh)
    assert ax.get_aspect() == 1.0
    nd = hpfem.NedelecDofMap2D(mesh, 2)
    e = hpfem.interpolate(nd, rotation_field)
    for component in ("abs", "re", "im", "phase", "x"):
        interop.plot_field(nd, e, component=component, subdivisions=2, colorbar=False)
    with pytest.raises(ValueError):
        interop.plot_field(nd, e, component="z")
    interop.plot_convergence([10, 100, 1000], [1.0, 0.1, 0.01], label="p = 1")
    interop.plot_convergence([10, 100, 1000], [1.0, 0.1, 0.01], exponent=1 / 3)
    omega = 6.0 * hpfem.constants.c0
    far = hpfem.FarField2D(
        mesh,
        hpfem.Surface2D.around_cells(mesh, 2),
        hpfem.discrete_field(nd, e),
        omega,
        hpfem.Material.vacuum(),
        4,
    )
    ax = interop.plot_far_field(far, resolution=36)
    ax.figure.savefig(tmp_path / "far.png")
    plt.close("all")

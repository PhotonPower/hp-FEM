import numpy as np
import pytest

import hpfem

GMSH_TWO_TRIANGLES = """$MeshFormat
4.1 0 8
$EndMeshFormat
$PhysicalNames
2
1 7 "wall"
2 3 "glass"
$EndPhysicalNames
$Entities
0 1 1 0
1 0 0 0 1 0 0 1 7 0
1 0 0 0 1 1 0 1 3 1 1
$EndEntities
$Nodes
2 4 1 4
1 1 0 2
1
2
0 0 0
1 0 0
2 1 0 2
3
4
1 1 0
0 1 0
$EndNodes
$Elements
2 3 1 3
1 1 1 1
1 1 2
2 1 2 2
2 1 2 3
3 1 3 4
$EndElements
"""


def test_rectangle_counts_and_tags():
    m = hpfem.rectangle(3, 2)
    assert (m.num_vertices, m.num_edges, m.num_cells) == (12, 3 * 6 + 3 + 2, 12)
    assert m.vertices.shape == (12, 2) and m.cells.shape == (12, 3)
    assert m.num_boundary_facets == 2 * (3 + 2)
    assert set(m.facet_tags[m.boundary_facets]) == {1, 2, 3, 4}
    assert len(m.facets_with_tag(hpfem.box_tag.Y_MAX)) == 3
    assert m.is_conforming and m.geometry_order == 1
    c = m.cell_centroids
    assert c.shape == (12, 2) and np.all((c > 0) & (c < 1))
    m.set_cell_tag(0, 5)
    assert m.cells_with_tag(5).tolist() == [0]
    with pytest.raises(ValueError):
        hpfem.rectangle(0, 1)


def test_box_and_faces():
    m = hpfem.box(2, 1, 1)
    assert m.num_cells == 12 and m.dim == 3
    assert m.faces.shape[1] == 3 and m.num_facets == m.num_faces
    assert len(m.cell_faces(0)) == 4
    assert m.facet_cells(m.boundary_facets[0])[1] == hpfem.INVALID_INDEX


def test_mesh_from_arrays_and_connectivity():
    vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    cells = np.array([[0, 1, 2], [0, 2, 3]])
    m = hpfem.Mesh2D(vertices, cells, [1, 2])
    assert m.num_edges == 5 and m.cell_tags.tolist() == [1, 2]
    diagonal = m.edge_id(0, 2)
    assert sorted(m.facet_cells(diagonal)) == [0, 1]
    assert m.edge_id(1, 3) == hpfem.INVALID_INDEX
    assert m.tag_boundary(9) == 4
    with pytest.raises(ValueError):
        hpfem.Mesh2D(vertices, np.array([[0, 1, 7]]))


def test_read_gmsh_string():
    m = hpfem.read_gmsh_string(GMSH_TWO_TRIANGLES, scale=1e-9, dim=2)
    assert m.num_cells == 2 and m.cell_tags.tolist() == [3, 3]
    assert m.tag_name(2, 3) == "glass" and m.tag_by_name(1, "wall") == 7
    assert np.isclose(m.vertices.max(), 1e-9)
    assert len(m.facets_with_tag(7)) == 1


def test_disc_is_curved_and_refines():
    m = hpfem.disc(2, radius=0.5)
    assert m.geometry_order == 2 and m.edge_nodes.shape == (m.num_edges, 2)
    r = hpfem.refine_uniform(m)
    assert r.mesh.num_cells == 4 * m.num_cells
    assert r.parent_cell.shape == (r.mesh.num_cells,)
    assert r.mesh.geometry_order == 2


def test_extract_l_shape():
    square = hpfem.rectangle(4, 4, [-1.0, -1.0], [1.0, 1.0])
    l_shape = hpfem.extract(square, lambda x: not (x[0] > 0 and x[1] < 0))
    assert l_shape.num_cells == 24
    by_ids = hpfem.extract(square, square.cells_with_tag(0)[:6])
    assert by_ids.num_cells == 6


def test_adaptive_mesh_hanging_and_locator():
    adaptive = hpfem.AdaptiveMesh2D(hpfem.rectangle(2, 2))
    step = adaptive.refine([0])
    leaf = adaptive.mesh
    assert not leaf.is_conforming
    assert step.num_old_cells == 8 and step.num_cells == leaf.num_cells
    assert sum(step.refined(i) for i in range(step.num_cells)) >= 4
    assert adaptive.max_level == 1 and adaptive.level(0) in (0, 1)
    roles = {leaf.facet_hanging_role(f) for f in range(leaf.num_facets)}
    assert roles == {"none", "parent", "child"}
    locator = hpfem.PointLocator2D(leaf)
    hit = locator.locate([0.1, 0.1])
    assert hit is not None and 0 <= hit.cell < leaf.num_cells
    assert locator.locate([2.0, 0.0]) is None
    xi = locator.reference_coordinates(hit.cell, [0.1, 0.1])
    assert xi is not None and np.allclose(xi, hit.xi)

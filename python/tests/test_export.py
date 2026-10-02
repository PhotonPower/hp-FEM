import numpy as np

import hpfem


def test_vtk_writer_arrays(tmp_path):
    mesh = hpfem.rectangle(2, 2)
    writer = hpfem.VtkWriter2D(mesh, hpfem.VtkFormat.ASCII)
    writer.cell_scalars("ids", np.arange(mesh.num_cells))
    writer.cell_scalars("real", np.linspace(0, 1, mesh.num_cells))
    writer.cell_scalars("cplx", np.linspace(0, 1, mesh.num_cells) * (1 + 1j))
    writer.point_vectors("v", mesh.vertices)
    text = writer.to_string()
    for name in ("cell_tag", "ids", "real", "cplx_re", "cplx_im", "v"):
        assert f'Name="{name}"' in text
    file = tmp_path / "mesh.vtu"
    writer.write(file)
    assert file.read_text() == text
    hpfem.write_vtu_facets(mesh, tmp_path / "facets.vtu")
    assert (tmp_path / "facets.vtu").exists()


def test_field_exporter(tmp_path):
    mesh = hpfem.disc(2)
    nd = hpfem.NedelecDofMap2D(mesh, 2)
    h1 = hpfem.DofMap2D(mesh, 2)
    e = hpfem.interpolate(nd, lambda x: np.array([x[1], -x[0]]))
    u = hpfem.interpolate(h1, lambda x: x[0] ** 2)
    exporter = hpfem.FieldExporter2D(mesh, 2, hpfem.VtkFormat.ASCII)
    exporter.hcurl("E", nd, e).h1("u", h1, u).cell_scalars("eta", np.ones(mesh.num_cells))
    text = exporter.to_string()
    for name in ("E_re", "curl_E_re", "u_re", "eta"):
        assert f'Name="{name}"' in text
    exporter.write(tmp_path / "fields.vtu")
    averages = hpfem.cell_averages(nd, e)
    assert len(averages) == mesh.num_cells and averages[0].shape == (2,)
    assert hpfem.cell_averages(h1, u).shape == (mesh.num_cells,)
    # exact averages on affine cells: curl (y, -x) = -2
    square = hpfem.rectangle(2, 2)
    nd_square = hpfem.NedelecDofMap2D(square, 2)
    e_square = hpfem.interpolate(nd_square, lambda x: np.array([x[1], -x[0]]))
    curls = hpfem.cell_average_curls(nd_square, e_square)
    assert np.allclose([c[0] for c in curls], -2.0, atol=1e-10)

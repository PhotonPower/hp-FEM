"""Interoperability with meshio, pyvista and matplotlib.

Meshes convert both ways (``to_meshio`` / ``from_meshio``, ``to_pyvista``); discrete fields
are sampled on the n-fold subdivided mesh (:func:`hpfem.subdivide`), the piecewise-linear
carrier that shows high-order fields and their jumps, with ``field_to_meshio`` /
``field_to_pyvista``. The matplotlib helpers draw 2D meshes with tags, fields by component
(``tripcolor`` on the subdivision), convergence tables and far-field patterns. The optional
packages are imported lazily; ``ImportError`` names the missing one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from hpfem import _hpfem as core

# --- subdivision sampling (no optional dependency) --------------------------------------------


def _is_nedelec(dofs) -> bool:
    return type(dofs).__name__.startswith("NedelecDofMap")


def _dim(mesh) -> int:
    return int(mesh.dim)


def sample_on_subdivision(dofs, coefficients, subdivisions: int = 2, *, curl: bool = False):
    """``(subdivided, values)``: the n-fold subdivided mesh and the field (or its curl)
    at its vertices — complex ``(n,)`` for H1, ``(n, dim)`` (curl: ``(n, 1 | 3)``) for
    H(curl)."""
    sub = core.subdivide(dofs.mesh, int(subdivisions))
    if _is_nedelec(dofs):
        values = core.sample_hcurl(dofs, coefficients, sub.vertex_parent, sub.vertex_xi, curl)
    else:
        if curl:
            raise ValueError("sample_on_subdivision: an H1 field has no curl")
        values = core.sample_h1(dofs, coefficients, sub.vertex_parent, sub.vertex_xi)
    return sub, values


def _component(values: np.ndarray, component: str) -> np.ndarray:
    """Real scalar per vertex from complex values: 'abs' (|E|), 're'/'im' (first component
    or scalar), 'x'/'y'/'z' (modulus of that component), 'phase' (of the first component)."""
    values = np.asarray(values)
    vector = values.ndim == 2 and values.shape[1] > 1
    if component == "abs":
        return np.linalg.norm(values, axis=1) if vector else np.abs(values.reshape(-1))
    first = values[:, 0] if values.ndim == 2 else values
    if component == "re":
        return first.real
    if component == "im":
        return first.imag
    if component == "phase":
        return np.angle(first)
    if component in ("x", "y", "z"):
        index = "xyz".index(component)
        if not vector or index >= values.shape[1]:
            raise ValueError(
                f"component '{component}' not available for values of shape {values.shape}"
            )
        return np.abs(values[:, index])
    raise ValueError(f"unknown component '{component}' (abs | re | im | phase | x | y | z)")


# --- meshio ------------------------------------------------------------------------------------


def _meshio():
    try:
        import meshio
    except ImportError as error:  # pragma: no cover
        raise ImportError(
            "hpfem.interop needs the 'meshio' package (pip install meshio)"
        ) from error
    return meshio


def _cell_type(dim: int, order: int) -> str:
    return {(2, 1): "triangle", (2, 2): "triangle6", (3, 1): "tetra", (3, 2): "tetra10"}[
        (dim, order)
    ]


def _points_3d(points: np.ndarray) -> np.ndarray:
    out = np.zeros((points.shape[0], 3))
    out[:, : points.shape[1]] = points
    return out


def _edge_node_order(mesh, dim: int):
    """Local edge order of the Gmsh / meshio second-order simplices: triangle6 edges
    (0,1),(1,2),(2,0); tetra10 edges (0,1),(1,2),(2,0),(0,3),(2,3),(1,3) — as global edge ids
    per cell."""
    local = (
        [(0, 1), (1, 2), (2, 0)] if dim == 2 else [(0, 1), (1, 2), (2, 0), (0, 3), (2, 3), (1, 3)]
    )
    cells = mesh.cells
    return np.array([[mesh.edge_id(int(c[a]), int(c[b])) for a, b in local] for c in cells])


def to_meshio(mesh, cell_data: Mapping[str, np.ndarray] | None = None,
              point_data: Mapping[str, np.ndarray] | None = None):  # fmt: skip
    """``meshio.Mesh`` of a mesh (``triangle`` / ``tetra``, or ``triangle6`` / ``tetra10`` with
    the edge nodes of a second-order mesh); ``cell_tag`` is always a cell-data array, plus
    the facets as ``line`` / ``triangle`` cells with ``facet_tag`` for tagged facets."""
    meshio = _meshio()
    dim = _dim(mesh)
    order = int(mesh.geometry_order)
    points = mesh.vertices
    cells = mesh.cells.copy()
    if order == 2:
        points = np.vstack([points, mesh.edge_nodes])
        cells = np.hstack([cells, mesh.num_vertices + _edge_node_order(mesh, dim)])
    blocks = [(_cell_type(dim, order), cells)]
    data = {"cell_tag": [np.asarray(mesh.cell_tags, dtype=np.int32)]}
    for name, values in (cell_data or {}).items():
        data[name] = [np.asarray(values)]
    tagged = np.flatnonzero(mesh.facet_tags != core.NO_TAG)
    if len(tagged):
        facets = np.array([mesh.facet_vertices(int(f)) for f in tagged])
        blocks.append(("line" if dim == 2 else "triangle", facets))
        data["cell_tag"].append(np.zeros(len(tagged), dtype=np.int32))
        for name in data:
            if name != "cell_tag":
                data[name].append(np.full(len(tagged), np.nan))
        data["facet_tag"] = [np.zeros(len(cells), dtype=np.int32),
                             np.asarray(mesh.facet_tags[tagged], dtype=np.int32)]  # fmt: skip
    return meshio.Mesh(
        _points_3d(points), blocks, point_data=dict(point_data or {}), cell_data=data
    )


def from_meshio(mesh, dim: int | None = None, scale: float = 1.0):
    """``Mesh2D`` / ``Mesh3D`` from a ``meshio.Mesh``: the ``triangle`` / ``tetra`` (or
    ``triangle6`` / ``tetra10``) cells become the cells, ``gmsh:physical`` (or ``cell_tag``)
    the tags, ``line`` / ``triangle`` elements of the lower dimension tag the facets; the
    coordinates are multiplied by ``scale``. Unused points are dropped."""
    blocks = {b.type: np.asarray(b.data) for b in mesh.cells}
    if dim is None:
        dim = 3 if ("tetra" in blocks or "tetra10" in blocks) else 2
    cell_type = "tetra" if dim == 3 else "triangle"
    key = cell_type if cell_type in blocks else cell_type + ("10" if dim == 3 else "6")
    if key not in blocks:
        raise ValueError(f"from_meshio: no {cell_type} cells in the mesh")
    cells = blocks[key][:, : dim + 1]
    used, inverse = np.unique(cells, return_inverse=True)
    points = np.asarray(mesh.points, dtype=float)[used, :dim] * scale
    cells = inverse.reshape(cells.shape)
    tags = None
    for name in ("gmsh:physical", "cell_tag"):
        if name in mesh.cell_data:
            for block, values in zip(mesh.cells, mesh.cell_data[name], strict=True):
                if block.type == key:
                    tags = [int(t) for t in values]
    out = (core.Mesh2D if dim == 2 else core.Mesh3D)(points, cells, tags)
    facet_type = "line" if dim == 2 else "triangle"
    if facet_type in blocks and facet_type != key:
        facets = blocks[facet_type][:, :dim]
        renumber = {int(old): new for new, old in enumerate(used)}
        keep = [all(int(v) in renumber for v in f) for f in facets]
        facets = np.array([[renumber[int(v)] for v in f] for f in facets[keep]], dtype=np.int64)
        facet_tags = None
        for name in ("gmsh:physical", "cell_tag", "facet_tag"):
            if name in mesh.cell_data:
                for block, values in zip(mesh.cells, mesh.cell_data[name], strict=True):
                    if block.type == facet_type:
                        facet_tags = np.asarray(values)[keep]
        if facet_tags is not None and len(facets):
            out.set_facet_tags(facets, [int(t) for t in facet_tags])
    if key.endswith(("6", "10")):
        nodes = blocks[key][:, dim + 1 :]
        edge_nodes = np.zeros((out.num_edges, dim))
        local = (
            [(0, 1), (1, 2), (2, 0)]
            if dim == 2
            else [(0, 1), (1, 2), (2, 0), (0, 3), (2, 3), (1, 3)]
        )
        for verts, mid in zip(cells, nodes, strict=True):
            for (a, b), node in zip(local, mid, strict=True):
                edge_nodes[out.edge_id(int(verts[a]), int(verts[b]))] = (
                    mesh.points[node, :dim] * scale
                )
        out.set_edge_nodes(edge_nodes)
    return out


def field_to_meshio(dofs, coefficients, subdivisions: int = 2, name: str = "E",
                    cell_data: Mapping[str, np.ndarray] | None = None):  # fmt: skip
    """``meshio.Mesh`` of the subdivided mesh with the field as point data ``<name>_re`` /
    ``<name>_im`` (and its curl ``curl_<name>_*`` for H(curl)); parent-cell data is broadcast
    to the sub-cells."""
    sub, values = sample_on_subdivision(dofs, coefficients, subdivisions)
    values = np.asarray(values)
    point_data = {}
    if values.ndim == 2:
        point_data[f"{name}_re"] = _points_3d(values.real)
        point_data[f"{name}_im"] = _points_3d(values.imag)
        _, curls = sample_on_subdivision(dofs, coefficients, subdivisions, curl=True)
        curls = np.asarray(curls)
        if curls.shape[1] == 1:
            point_data[f"curl_{name}_re"] = curls[:, 0].real
            point_data[f"curl_{name}_im"] = curls[:, 0].imag
        else:
            point_data[f"curl_{name}_re"] = curls.real
            point_data[f"curl_{name}_im"] = curls.imag
    else:
        point_data[f"{name}_re"] = values.real
        point_data[f"{name}_im"] = values.imag
    broadcast = {k: np.asarray(v)[sub.parent_cell] for k, v in (cell_data or {}).items()}
    return to_meshio(sub.mesh, cell_data=broadcast, point_data=point_data)


# --- pyvista -----------------------------------------------------------------------------------


def to_pyvista(mesh, cell_data: Mapping[str, np.ndarray] | None = None,
               point_data: Mapping[str, np.ndarray] | None = None):  # fmt: skip
    """``pyvista.UnstructuredGrid`` of a mesh (affine cells; curved meshes are exported with
    straight edges) with ``cell_tag`` and the given arrays."""
    try:
        import pyvista
    except ImportError as error:  # pragma: no cover
        raise ImportError("to_pyvista needs the 'pyvista' package (pip install pyvista)") from error
    dim = _dim(mesh)
    cells = mesh.cells
    n = cells.shape[0]
    connectivity = np.hstack([np.full((n, 1), dim + 1, dtype=np.int64), cells]).ravel()
    celltype = pyvista.CellType.TRIANGLE if dim == 2 else pyvista.CellType.TETRA
    grid = pyvista.UnstructuredGrid(
        connectivity, np.full(n, celltype, dtype=np.uint8), _points_3d(mesh.vertices)
    )
    grid.cell_data["cell_tag"] = np.asarray(mesh.cell_tags, dtype=np.int32)
    for name, values in (cell_data or {}).items():
        grid.cell_data[name] = np.asarray(values)
    for name, values in (point_data or {}).items():
        grid.point_data[name] = np.asarray(values)
    return grid


def field_to_pyvista(dofs, coefficients, subdivisions: int = 2, name: str = "E"):
    """``pyvista.UnstructuredGrid`` of the subdivided mesh with the field as point data
    (``<name>_re``, ``<name>_im``, ``<name>_abs``)."""
    sub, values = sample_on_subdivision(dofs, coefficients, subdivisions)
    values = np.asarray(values)
    if values.ndim == 2:
        data = {f"{name}_re": _points_3d(values.real), f"{name}_im": _points_3d(values.imag)}
    else:
        data = {f"{name}_re": values.real, f"{name}_im": values.imag}
    data[f"{name}_abs"] = _component(values, "abs")
    return to_pyvista(sub.mesh, point_data=data)


# --- matplotlib --------------------------------------------------------------------------------


def _axes(ax):
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover
        raise ImportError("the plot helpers need 'matplotlib' (pip install matplotlib)") from error
    if ax is None:
        _, ax = plt.subplots()
    return ax


def plot_mesh(mesh, ax=None, *, tags: bool = True, facet_tags: bool = True, linewidth: float = 0.5):
    """Draws a 2D mesh: cells coloured by tag (``tags``), tagged facets in colour."""
    if _dim(mesh) != 2:
        raise ValueError("plot_mesh draws two-dimensional meshes")
    ax = _axes(ax)
    import matplotlib.tri as mtri

    tri = mtri.Triangulation(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.cells)
    if tags and np.any(mesh.cell_tags != core.NO_TAG):
        ax.tripcolor(tri, facecolors=mesh.cell_tags, cmap="Pastel1", alpha=0.8)
    ax.triplot(tri, color="k", linewidth=linewidth)
    if facet_tags:
        tagged = np.flatnonzero(mesh.facet_tags != core.NO_TAG)
        if len(tagged):
            import matplotlib.cm as cm

            unique = np.unique(mesh.facet_tags[tagged])
            colours = cm.tab10(np.linspace(0, 1, max(len(unique), 2)))
            for f in tagged:
                a, b = mesh.facet_vertices(int(f))
                x = mesh.vertices[[a, b]]
                colour = colours[np.searchsorted(unique, mesh.facet_tag(int(f)))]
                ax.plot(x[:, 0], x[:, 1], color=colour, linewidth=2 * linewidth + 1)
    ax.set_aspect("equal")
    return ax


def plot_field(
    dofs,
    coefficients,
    ax=None,
    *,
    component: str = "abs",
    subdivisions: int = 3,
    cmap: str = "viridis",
    colorbar: bool = True,
    mesh_lines: bool = False,
    **kwargs,
):
    """Draws a 2D field (``tripcolor`` with Gouraud shading on the subdivision): ``component``
    is 'abs', 're', 'im', 'phase', 'x' or 'y' (see ``_component``). Returns the axes."""
    if _dim(dofs.mesh) != 2:
        raise ValueError("plot_field draws two-dimensional fields")
    ax = _axes(ax)
    import matplotlib.tri as mtri

    sub, values = sample_on_subdivision(dofs, coefficients, subdivisions)
    scalar = _component(values, component)
    tri = mtri.Triangulation(sub.mesh.vertices[:, 0], sub.mesh.vertices[:, 1], sub.mesh.cells)
    image = ax.tripcolor(tri, scalar, shading="gouraud", cmap=cmap, **kwargs)
    if mesh_lines:
        parent = mtri.Triangulation(
            dofs.mesh.vertices[:, 0], dofs.mesh.vertices[:, 1], dofs.mesh.cells
        )
        ax.triplot(parent, color="w", linewidth=0.3, alpha=0.6)
    if colorbar:
        ax.figure.colorbar(image, ax=ax, label=component)
    ax.set_aspect("equal")
    return ax


def plot_convergence(
    num_dofs: Sequence[float],
    errors: Sequence[float],
    ax=None,
    *,
    label: str | None = None,
    exponent: float | None = None,
    **kwargs,
):
    """Error against #DoF: log–log by default; with ``exponent`` the abscissa is
    ``#DoF^exponent`` on a linear scale and the ordinate logarithmic (exponential
    convergence shows as a straight line, e.g. ``exponent=1/3`` for 2D hp-adaptivity)."""
    ax = _axes(ax)
    n = np.asarray(num_dofs, dtype=float)
    e = np.asarray(errors, dtype=float)
    if exponent is None:
        ax.loglog(n, e, marker="o", label=label, **kwargs)
        ax.set_xlabel("#DoF")
    else:
        ax.semilogy(n**exponent, e, marker="o", label=label, **kwargs)
        ax.set_xlabel(f"#DoF^{exponent:.3g}")
    ax.set_ylabel("error")
    ax.grid(True, which="both", alpha=0.3)
    if label:
        ax.legend()
    return ax


def plot_far_field(far, ax=None, *, resolution: int = 360, db: bool = False, **kwargs):
    """Polar plot of |F(phi)| of a 2D far field (``FarField2D``); ``db`` for 20 log10."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"})
    angles = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
    pattern = np.array([np.linalg.norm(far.pattern([np.cos(a), np.sin(a)])) for a in angles])
    if db:
        pattern = 20 * np.log10(np.maximum(pattern, 1e-300) / pattern.max())
    ax.plot(np.append(angles, angles[0]), np.append(pattern, pattern[0]), **kwargs)
    return ax


__all__ = [
    "sample_on_subdivision", "to_meshio", "from_meshio", "field_to_meshio", "to_pyvista",
    "field_to_pyvista", "plot_mesh", "plot_field", "plot_convergence", "plot_far_field",
]  # fmt: skip

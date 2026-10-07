"""Unit-cell meshing for gratings and metasurfaces (M15 F6) with the Gmsh Python API.

A :class:`UnitCell` describes one period: the cell box (``x0 .. x0 + period`` by
``y_bottom .. y_top``, PML space included), horizontal :class:`Slab` layers spanning the
period (substrate, films, cover) and :class:`Shape` scatterers (rectangle, trapezoid,
ellipse, polygon) that are copied by ±period and clipped to the cell. Materials are
cell tags: a shape wins over a slab, a later shape over an earlier one. The sides get the
facet tags ``box_tag.X_MIN`` / ``X_MAX`` / ``Y_MIN`` / ``Y_MAX``, so the mesh can go straight
into :func:`hpfem.grating.solve`. Element sizes per tag come from :func:`element_sizes`
(elements per wavelength and the field decay length in metals), interfaces are refined by a
factor, curved shapes get second-order (curved) cells.

The Gmsh part (:func:`write_unit_cell`) needs the ``gmsh`` package (``pip install gmsh``)
and does not import ``hpfem``; :func:`unit_cell_mesh` reads the written file with
``hpfem.read_gmsh_periodic``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

X_MIN, X_MAX, Y_MIN, Y_MAX = 1, 2, 3, 4  # hpfem.box_tag values


@dataclass
class Slab:
    """A horizontal layer over the full period, ``y0 <= y <= y1``."""

    tag: int
    y0: float
    y1: float


@dataclass
class Shape:
    """A scatterer. ``kind``: ``"rectangle"`` (``x``, ``y``, ``width``, ``height``: lower-left
    corner), ``"trapezoid"`` (``x``, ``y``, ``bottom``, ``top``, ``height``: centred at ``x``,
    base at ``y``), ``"ellipse"`` (``x``, ``y``, ``rx``, ``ry``: centre) or ``"polygon"``
    (``points``: list of (x, y), counter-clockwise)."""

    kind: str
    tag: int
    params: dict = field(default_factory=dict)

    def polygon(self) -> list[tuple[float, float]] | None:
        p = self.params
        if self.kind == "rectangle":
            x, y, w, h = p["x"], p["y"], p["width"], p["height"]
            return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        if self.kind == "trapezoid":
            x, y, b, t, h = p["x"], p["y"], p["bottom"], p["top"], p["height"]
            return [(x - b / 2, y), (x + b / 2, y), (x + t / 2, y + h), (x - t / 2, y + h)]
        if self.kind == "polygon":
            return [tuple(map(float, q)) for q in p["points"]]
        return None

    def contains(self, x: float, y: float) -> bool:
        if self.kind == "ellipse":
            p = self.params
            return ((x - p["x"]) / p["rx"]) ** 2 + ((y - p["y"]) / p["ry"]) ** 2 <= 1.0
        poly = self.polygon()
        inside = False
        n = len(poly)
        for i in range(n):
            (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
            if (y1 > y) != (y2 > y):
                xc = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x < xc:
                    inside = not inside
        return inside

    def bounds(self) -> tuple[float, float]:
        if self.kind == "ellipse":
            return self.params["x"] - self.params["rx"], self.params["x"] + self.params["rx"]
        xs = [q[0] for q in self.polygon()]
        return min(xs), max(xs)


@dataclass
class UnitCell:
    period: float
    y_bottom: float
    y_top: float
    slabs: list[Slab] = field(default_factory=list)
    shapes: list[Shape] = field(default_factory=list)
    x0: float | None = None
    """left edge; default ``-period / 2``"""
    background_tag: int = 0
    """tag of cells covered by no slab and no shape (``hpfem.kNoTag`` is 0 … use a real tag)"""

    @property
    def left(self) -> float:
        return -self.period / 2 if self.x0 is None else self.x0

    def tag_at(self, x: float, y: float) -> int:
        """Material tag at a point: the last shape containing it, else its slab, else the
        background."""
        for shape in reversed(self.shapes):
            for k in (-1, 0, 1):
                if shape.contains(x - k * self.period, y):
                    return shape.tag
        for slab in self.slabs:
            if slab.y0 <= y <= slab.y1:
                return slab.tag
        return self.background_tag


def element_sizes(
    materials: Mapping[int, object],
    omega: float,
    p: int,
    elements_per_wavelength: float | None = None,
    decay_lengths: float = 1.0,
) -> dict[int, float]:
    """Element size per tag [m]: the local wavelength λ0 / max(Re n, 1) over
    ``elements_per_wavelength`` (default ``12 / p``), and in absorbing media at most
    ``decay_lengths`` field decay lengths 1 / (k0 Im n). ``materials`` maps tags to objects
    with ``refractive_index`` (``hpfem.Material``) or to complex indices."""
    from hpfem import units

    k0 = float(units.vacuum_wavenumber(omega))
    wavelength = 2 * math.pi / k0
    per_wavelength = 12.0 / p if elements_per_wavelength is None else elements_per_wavelength
    out = {}
    for tag, material in materials.items():
        n = complex(getattr(material, "refractive_index", material))
        h = wavelength / (max(n.real, 1.0) * per_wavelength)
        if n.imag > 1e-12:
            h = min(h, decay_lengths / (k0 * n.imag))
        out[int(tag)] = float(h)
    return out


def write_unit_cell(
    cell: UnitCell,
    path: str | Path,
    sizes: Mapping[int, float],
    *,
    default_size: float | None = None,
    interface_factor: float = 0.5,
    periodic: bool = True,
    order: int | None = None,
    unit: float = 1e-9,
    verbose: bool = False,
) -> Path:
    """Meshes the unit cell with Gmsh (OpenCASCADE) and writes a MSH 4.1 ASCII file with
    physical groups (surfaces: material tags; curves: the four sides) and, with ``periodic``,
    the ``$Periodic`` section (right side = left side + period; optional since the Bloch
    coupling accepts non-matching faces). ``sizes`` are element sizes per tag [m]
    (``default_size`` for tags not listed), ``interface_factor`` scales the size at material
    interfaces, ``order`` 2 gives curved cells (the default when a shape is an ellipse). The
    model is built in units of ``unit`` metres (default nanometres: OpenCASCADE works with
    absolute tolerances of about 1e-7 model units); read the file with ``scale=unit``."""
    import gmsh  # the only place that needs it

    path = Path(path)
    u = float(unit)
    x0, x1 = cell.left / u, (cell.left + cell.period) / u
    y0, y1 = cell.y_bottom / u, cell.y_top / u
    period = cell.period / u
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add("unit_cell")
        occ = gmsh.model.occ
        base = occ.addRectangle(x0, y0, 0, period, y1 - y0)
        tools = []
        for slab in cell.slabs:
            lo, hi = max(slab.y0 / u, y0), min(slab.y1 / u, y1)
            if hi > lo:
                tools.append((2, occ.addRectangle(x0, lo, 0, period, hi - lo)))
        for shape in cell.shapes:
            lo, hi = (b / u for b in shape.bounds())
            copies = [k for k in (-1, 0, 1) if lo + k * period < x1 and hi + k * period > x0]
            for k in copies:
                dx = k * period
                if shape.kind == "ellipse":
                    p = shape.params
                    tools.append(
                        (2, occ.addDisk(p["x"] / u + dx, p["y"] / u, 0, p["rx"] / u, p["ry"] / u))
                    )
                else:
                    points = [occ.addPoint(px / u + dx, py / u, 0) for px, py in shape.polygon()]
                    lines = [
                        occ.addLine(points[i], points[(i + 1) % len(points)])
                        for i in range(len(points))
                    ]
                    loop = occ.addCurveLoop(lines)
                    tools.append((2, occ.addPlaneSurface([loop])))
        pieces, _ = occ.fragment([(2, base)], tools, removeObject=True, removeTool=True)
        occ.synchronize()
        # keep the pieces inside the cell, tag them by their centre of mass
        surfaces_by_tag: dict[int, list[int]] = {}
        eps = 1e-6 * period
        for dim, tag in gmsh.model.getEntities(2):
            cx, cy, _ = occ.getCenterOfMass(dim, tag)
            if cx < x0 - eps or cx > x1 + eps or cy < y0 - eps or cy > y1 + eps:
                occ.remove([(dim, tag)], recursive=True)
                continue
            surfaces_by_tag.setdefault(cell.tag_at(cx * u, cy * u), []).append(tag)
        occ.synchronize()
        for tag, surfaces in surfaces_by_tag.items():
            gmsh.model.addPhysicalGroup(2, surfaces, tag, f"material_{tag}")
        # the four sides and the material interfaces
        sides: dict[int, list[int]] = {X_MIN: [], X_MAX: [], Y_MIN: [], Y_MAX: []}
        interfaces: list[int] = []
        tol = 1e-6 * period
        for dim, tag in gmsh.model.getEntities(1):
            cx, cy, _ = occ.getCenterOfMass(dim, tag)
            p0 = gmsh.model.getValue(dim, tag, [0.0])  # a point of the curve, to tell a side
            if abs(cx - x0) < tol and abs(p0[0] - x0) < tol:
                sides[X_MIN].append(tag)
            elif abs(cx - x1) < tol and abs(p0[0] - x1) < tol:
                sides[X_MAX].append(tag)
            elif abs(cy - y0) < tol and abs(p0[1] - y0) < tol:
                sides[Y_MIN].append(tag)
            elif abs(cy - y1) < tol and abs(p0[1] - y1) < tol:
                sides[Y_MAX].append(tag)
            else:
                adjacent = gmsh.model.getAdjacencies(dim, tag)[0]
                tags = {
                    cell.tag_at(*(c * u for c in occ.getCenterOfMass(2, s)[:2])) for s in adjacent
                }
                if len(tags) > 1:
                    interfaces.append(tag)
        for side_tag, curves in sides.items():
            if curves:
                gmsh.model.addPhysicalGroup(
                    1,
                    curves,
                    side_tag,
                    {X_MIN: "left", X_MAX: "right", Y_MIN: "bottom", Y_MAX: "top"}[side_tag],
                )
        if periodic and sides[X_MIN] and sides[X_MAX]:
            affine = [1, 0, 0, period, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
            gmsh.model.mesh.setPeriodic(1, sides[X_MAX], sides[X_MIN], affine)
        # sizes: one constant field per tag, the minimum over them, refined at interfaces
        fields = []
        all_sizes = [float(h) / u for h in sizes.values()]
        if default_size is not None:
            all_sizes.append(float(default_size) / u)
        if not all_sizes:
            raise ValueError("write_unit_cell: give sizes per tag or a default_size")
        for tag, surfaces in surfaces_by_tag.items():
            h = sizes.get(tag, default_size)
            if h is None:
                raise ValueError(
                    f"write_unit_cell: no element size for tag {tag} and no default_size"
                )
            f = gmsh.model.mesh.field.add("Constant")
            gmsh.model.mesh.field.setNumbers(f, "SurfacesList", surfaces)
            gmsh.model.mesh.field.setNumber(f, "VIn", float(h) / u)
            gmsh.model.mesh.field.setNumber(f, "VOut", 1e300)
            fields.append(f)
        if interfaces and interface_factor < 1.0:
            h_min = min(all_sizes)
            dist = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(dist, "CurvesList", interfaces)
            gmsh.model.mesh.field.setNumber(dist, "Sampling", 100)
            thr = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(thr, "InField", dist)
            gmsh.model.mesh.field.setNumber(thr, "SizeMin", interface_factor * h_min)
            gmsh.model.mesh.field.setNumber(thr, "SizeMax", 1e300)
            gmsh.model.mesh.field.setNumber(thr, "DistMin", 0.5 * h_min)
            gmsh.model.mesh.field.setNumber(thr, "DistMax", 2.0 * h_min)
            fields.append(thr)
        combined = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(combined, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(combined)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
        curved = any(s.kind == "ellipse" for s in cell.shapes)
        gmsh.option.setNumber(
            "Mesh.ElementOrder", 2 if (order == 2 or (order is None and curved)) else 1
        )
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path


def unit_cell_mesh(
    cell: UnitCell, sizes: Mapping[int, float], path: str | Path | None = None, **options
):
    """:func:`write_unit_cell` followed by ``hpfem.read_gmsh_periodic``: returns
    ``(mesh, periodic_links)`` with the mesh in metres (the cell is given in metres) and the
    links as ``(master_tag, slave_tag, shift)``; ``path`` defaults to a temporary file."""
    import tempfile

    import hpfem

    if path is None:
        with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as handle:
            path = handle.name
    unit = float(options.get("unit", 1e-9))
    write_unit_cell(cell, path, sizes, **options)
    return hpfem.read_gmsh_periodic(str(path), unit, 2)


def report(mesh, materials: Mapping[int, object] | Sequence[int] | None = None) -> dict:
    """``hpfem.mesh_report`` plus ``tags_without_material`` (cell tags not in ``materials``)
    and ``materials_without_cells``."""
    import hpfem

    out = dict(hpfem.mesh_report(mesh))
    if materials is not None:
        known = {
            int(t) for t in (materials.keys() if isinstance(materials, Mapping) else materials)
        }
        present = {int(t) for t in out["cell_tags"]}
        out["tags_without_material"] = sorted(present - known)
        out["materials_without_cells"] = sorted(known - present)
    return out


__all__ = [
    "Shape",
    "Slab",
    "UnitCell",
    "element_sizes",
    "report",
    "unit_cell_mesh",
    "write_unit_cell",
]

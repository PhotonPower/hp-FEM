"""Crossed (doubly periodic) gratings in 3D (M21 S2, issue #237).

A :class:`CrossedGrating` is a stack of planar :class:`Layer` objects periodic in x and y with
the periods ``period = (px, py)``: the superstrate (incidence medium) above z = 0, the layers from
z = 0 downwards, the substrate below. Each layer has a background material and :class:`Shape`
inclusions (circle, ellipse, rectangle, polygon) that fill the full layer height: a hole is a
shape of the superstrate material, a pillar one of a solid. Shapes that cross the cell boundary
are continued periodically. Coordinates in metres, the cell spans ``[0, px] × [0, py]``.

:func:`solve` meshes the unit cell with Gmsh (:func:`unit_cell_mesh`: second-order tetrahedra for
curved shapes, identical meshes on opposite side faces), puts the homogeneous layers into the
layered background of the scattered-field formulation (so only the shapes scatter), adds PML above
and below and the Bloch phases of the incident wave in x and y, solves the s and p polarisation
with one factorisation and measures the reflected and transmitted orders (m, n) on planes in the
superstrate and the substrate (:func:`hpfem.diffraction_orders`). The result holds both
polarisations, so any other one follows without a new solve
(:meth:`CrossedResult.polarised`).

Conventions: the stack normal is z (FourierGrating uses y; the import of S3 converts), the
incident wave comes from above at the polar angle ``theta`` from the normal with its tangential
wavevector rotated by ``phi`` from +x (as ``LayerStack3D.plane_wave``); s has E perpendicular to
the plane of incidence, p in it, with the field directions and phases of
``LayerStack3D.plane_wave`` (at normal incidence and ``phi = 0``: s along -y, p along +x).
Time dependence exp(-iωt), angles in radians.

The Gmsh part needs the ``gmsh`` package (``pip install gmsh``).
"""

from __future__ import annotations

import math
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX = 1, 2, 3, 4, 5, 6  # hpfem.box_tag values

SUPERSTRATE_TAG = 1
SUBSTRATE_TAG = 2


def layer_tag(layer: int) -> int:
    """Cell tag of the background of layer ``layer`` (0 = top layer)."""
    return 10 + layer


def shape_tag(layer: int, shape: int) -> int:
    """Cell tag of shape ``shape`` of layer ``layer``."""
    return 100 * (layer + 1) + shape


class CrossedError(ValueError):
    """An invalid crossed-grating model or option."""


@dataclass
class Shape:
    """An inclusion filling the full height of its layer.

    ``kind``: ``"circle"`` (``center``, ``radius``), ``"ellipse"`` (``center``, ``rx``, ``ry``,
    ``angle``), ``"rectangle"`` (``center``, ``width`` along x, ``height`` along y, ``angle``) or
    ``"polygon"`` (``points``: (x, y) vertices, counter-clockwise). ``angle`` [rad] rotates about
    the centre. ``material``: an ``hpfem.Material``, a refractive index (complex for absorbing
    media), a name of :mod:`hpfem.materials` or a dispersive model with ``.at(omega)``."""

    kind: str
    material: object
    center: tuple[float, float] = (0.0, 0.0)
    radius: float | None = None
    rx: float | None = None
    ry: float | None = None
    width: float | None = None
    height: float | None = None
    points: Sequence[tuple[float, float]] | None = None
    angle: float = 0.0

    def __post_init__(self):
        need = {
            "circle": ("radius",),
            "ellipse": ("rx", "ry"),
            "rectangle": ("width", "height"),
            "polygon": ("points",),
        }
        if self.kind not in need:
            raise CrossedError(
                f"Shape: unknown kind {self.kind!r} (circle, ellipse, rectangle, polygon)"
            )
        for name in need[self.kind]:
            value = getattr(self, name)
            if value is None:
                raise CrossedError(f"Shape {self.kind}: {name} is missing")
            if name != "points" and not float(value) > 0:
                raise CrossedError(f"Shape {self.kind}: {name} must be positive")
        if self.kind == "polygon" and len(self.points) < 3:
            raise CrossedError("Shape polygon: at least three points")

    @property
    def curved(self) -> bool:
        return self.kind in ("circle", "ellipse")

    def outline(self, segments: int = 64) -> np.ndarray:
        """Vertices of the outline (curves sampled with ``segments`` points), shape (n, 2)."""
        cx, cy = map(float, self.center)
        if self.kind == "polygon":
            return np.asarray(self.points, dtype=float)
        if self.kind == "rectangle":
            local = 0.5 * np.array(
                [
                    [-self.width, -self.height],
                    [self.width, -self.height],
                    [self.width, self.height],
                    [-self.width, self.height],
                ]
            )
        else:
            rx, ry = (self.radius, self.radius) if self.kind == "circle" else (self.rx, self.ry)
            t = np.linspace(0.0, 2 * np.pi, segments, endpoint=False)
            local = np.column_stack([rx * np.cos(t), ry * np.sin(t)])
        c, s = math.cos(self.angle), math.sin(self.angle)
        return local @ np.array([[c, s], [-s, c]]) + np.array([cx, cy])

    def bounds(self) -> tuple[float, float, float, float]:
        """(x_min, x_max, y_min, y_max)."""
        if self.kind == "circle":
            cx, cy = map(float, self.center)
            r = float(self.radius)
            return cx - r, cx + r, cy - r, cy + r
        if self.kind == "ellipse":
            cx, cy = map(float, self.center)
            c, s = math.cos(self.angle), math.sin(self.angle)
            ex = math.hypot(self.rx * c, self.ry * s)
            ey = math.hypot(self.rx * s, self.ry * c)
            return cx - ex, cx + ex, cy - ey, cy + ey
        p = self.outline()
        return (
            float(p[:, 0].min()),
            float(p[:, 0].max()),
            float(p[:, 1].min()),
            float(p[:, 1].max()),
        )


@dataclass
class Layer:
    """A planar layer of ``thickness`` [m] of the ``material`` with :class:`Shape` inclusions
    (later shapes win where they overlap)."""

    thickness: float
    material: object
    shapes: list[Shape] = field(default_factory=list)


@dataclass
class CrossedGrating:
    """A doubly periodic stack: ``period = (px, py)`` [m], ``superstrate`` above z = 0, the
    ``layers`` from z = 0 downwards, the ``substrate`` below. The superstrate must be lossless
    (it carries the incident wave)."""

    period: tuple[float, float]
    superstrate: object
    substrate: object
    layers: list[Layer] = field(default_factory=list)

    def __post_init__(self):
        if len(self.period) != 2 or not all(float(p) > 0 for p in self.period):
            raise CrossedError("CrossedGrating: period must be two positive lengths (px, py)")
        for i, layer in enumerate(self.layers):
            if not float(layer.thickness) > 0:
                raise CrossedError(f"CrossedGrating: layer {i} needs a positive thickness")
            if len(layer.shapes) > 99:
                raise CrossedError(f"CrossedGrating: layer {i} has more than 99 shapes")

    @property
    def depth(self) -> float:
        """Total thickness of the layers [m]."""
        return float(sum(float(layer.thickness) for layer in self.layers))

    def interfaces(self) -> list[float]:
        """z of the interfaces from the top (0) to the bottom (-depth)."""
        z = [0.0]
        for layer in self.layers:
            z.append(z[-1] - float(layer.thickness))
        return z


# ---------------------------------------------------------------------------------- materials


def as_material(spec, omega: float):
    """An ``hpfem.Material`` at the angular frequency ``omega`` from a material, a refractive
    index (complex: n + i k with k > 0 absorbing), a name of :mod:`hpfem.materials` or a
    dispersive model with ``.at(omega)``."""
    import hpfem

    if isinstance(spec, hpfem.Material):
        return spec
    if hasattr(spec, "at"):
        return spec.at(omega)
    if isinstance(spec, str):
        return hpfem.materials.get(spec).at(omega)
    if isinstance(spec, (int, float, complex, np.number)):
        n = complex(spec)
        return hpfem.Material(n * n, 1.0)
    raise CrossedError(
        f"unknown material {spec!r}: give an hpfem.Material, a refractive index, a library name "
        "or a dispersive model"
    )


def _index(material) -> complex:
    return complex(material.refractive_index)


# ---------------------------------------------------------------------------------- mesh


@dataclass
class CellLayout:
    """Vertical layout of the computational cell [m]: the mesh spans ``z_bottom`` … ``z_top``;
    PML from ``pml_top`` up and from ``pml_bottom`` down; the order planes at ``z_above`` (in the
    superstrate) and ``z_below`` (in the substrate)."""

    z_top: float
    pml_top: float
    z_above: float
    z_below: float
    pml_bottom: float
    z_bottom: float

    def levels(self, grating: CrossedGrating) -> list[float]:
        """All horizontal mesh interfaces, descending."""
        return [self.z_top, self.pml_top, *grating.interfaces(), self.pml_bottom, self.z_bottom]


def _region_materials(grating: CrossedGrating, omega: float) -> dict[int, object]:
    mats = {
        SUPERSTRATE_TAG: as_material(grating.superstrate, omega),
        SUBSTRATE_TAG: as_material(grating.substrate, omega),
    }
    for i, layer in enumerate(grating.layers):
        mats[layer_tag(i)] = as_material(layer.material, omega)
        for j, shape in enumerate(layer.shapes):
            mats[shape_tag(i, j)] = as_material(shape.material, omega)
    return mats


def _matching(
    surfaces: dict[int, tuple], shift: np.ndarray, masters: list[int], slaves: list[int], tol: float
) -> tuple[list[int], list[int]]:
    """Pairs slave surfaces with the master surfaces at their position shifted back."""
    paired_slaves, paired_masters = [], []
    for s in slaves:
        target = np.asarray(surfaces[s]) - np.concatenate([shift, shift])
        best = min(masters, key=lambda m: np.abs(np.asarray(surfaces[m]) - target).max())
        if np.abs(np.asarray(surfaces[best]) - target).max() > tol:
            raise CrossedError(
                "unit_cell_mesh: the side faces are not periodic (a shape crosses the cell "
                "boundary without its periodic copy?)"
            )
        paired_slaves.append(s)
        paired_masters.append(best)
    return paired_slaves, paired_masters


def write_unit_cell(
    grating: CrossedGrating,
    layout: CellLayout,
    path: str | Path,
    sizes: dict[int, float],
    *,
    order: int | None = None,
    unit: float = 1e-9,
    verbose: bool = False,
) -> Path:
    """Meshes the unit cell with Gmsh (OpenCASCADE) and writes a MSH 4.1 ASCII file: volumes
    tagged ``SUPERSTRATE_TAG``, ``SUBSTRATE_TAG``, ``layer_tag(i)``, ``shape_tag(i, j)``; the six
    sides ``X_MIN`` … ``Z_MAX`` (``hpfem.box_tag``); the x and y sides periodic (identical meshes,
    ``$Periodic`` section). ``sizes`` are element sizes per tag [m], ``order`` 2 gives curved
    cells (the default with a circle or an ellipse). The model is built in units of ``unit``
    metres (OpenCASCADE tolerances are absolute); read the file with ``scale=unit``."""
    import gmsh

    path = Path(path)
    u = float(unit)
    px, py = (float(p) / u for p in grating.period)
    levels = [z / u for z in layout.levels(grating)]
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add("crossed_unit_cell")
        occ = gmsh.model.occ
        # one box per horizontal slab, top to bottom: tag of the slab
        boxes = []  # (volume, tag)
        slab_tags = [SUPERSTRATE_TAG, SUPERSTRATE_TAG]
        slab_tags += [layer_tag(i) for i in range(len(grating.layers))]
        slab_tags += [SUBSTRATE_TAG, SUBSTRATE_TAG]
        for (z1, z0), tag in zip(zip(levels[:-1], levels[1:], strict=True), slab_tags, strict=True):
            boxes.append((occ.addBox(0, 0, z0, px, py, z1 - z0), tag))
        # shapes with their periodic copies
        tools = []  # (volume, tag)
        interfaces = grating.interfaces()
        for i, layer in enumerate(grating.layers):
            z_top, z_bottom = interfaces[i] / u, interfaces[i + 1] / u
            height = z_top - z_bottom
            for j, shape in enumerate(layer.shapes):
                x0, x1, y0, y1 = (b / u for b in shape.bounds())
                for kx in (-1, 0, 1):
                    for ky in (-1, 0, 1):
                        dx, dy = kx * px, ky * py
                        if x0 + dx >= px or x1 + dx <= 0 or y0 + dy >= py or y1 + dy <= 0:
                            continue
                        volume = _shape_volume(occ, shape, u, z_bottom, height)
                        if dx or dy:
                            occ.translate([(3, volume)], dx, dy, 0)
                        tools.append((volume, shape_tag(i, j)))
        inputs = [(3, v) for v, _ in boxes] + [(3, v) for v, _ in tools]
        tags_of_input = [t for _, t in boxes] + [t for _, t in tools]
        _, out_map = occ.fragment(inputs[: len(boxes)], inputs[len(boxes) :])
        occ.synchronize()
        # owner of each piece: the last shape covering it, else its slab; pieces of shape copies
        # outside the cell belong to no slab and are removed
        in_box: set[int] = set()
        owner: dict[int, int] = {}
        for k, pieces in enumerate(out_map):
            for dim, piece in pieces:
                if dim != 3:
                    continue
                if k < len(boxes):
                    in_box.add(piece)
                    owner.setdefault(piece, tags_of_input[k])
                else:
                    owner[piece] = tags_of_input[k]  # shapes come after the slabs: they win
        outside = [(3, v) for v in owner if v not in in_box]
        if outside:
            occ.remove(outside, recursive=True)
            occ.synchronize()
        volumes_by_tag: dict[int, list[int]] = {}
        for v in sorted(in_box):
            volumes_by_tag.setdefault(owner[v], []).append(v)
        for tag, vols in volumes_by_tag.items():
            gmsh.model.addPhysicalGroup(3, vols, tag, f"region_{tag}")
        # the six sides
        tol = 1e-6 * max(px, py)
        sides: dict[int, list[int]] = {t: [] for t in (X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX)}
        boxes_of: dict[int, tuple] = {}
        for dim, s in gmsh.model.getEntities(2):
            bb = gmsh.model.getBoundingBox(dim, s)
            boxes_of[s] = bb
            x0, y0, z0, x1, y1, z1 = bb
            if abs(x0) < tol and abs(x1) < tol:
                sides[X_MIN].append(s)
            elif abs(x0 - px) < tol and abs(x1 - px) < tol:
                sides[X_MAX].append(s)
            elif abs(y0) < tol and abs(y1) < tol:
                sides[Y_MIN].append(s)
            elif abs(y0 - py) < tol and abs(y1 - py) < tol:
                sides[Y_MAX].append(s)
            elif abs(z0 - levels[-1]) < tol and abs(z1 - levels[-1]) < tol:
                sides[Z_MIN].append(s)
            elif abs(z0 - levels[0]) < tol and abs(z1 - levels[0]) < tol:
                sides[Z_MAX].append(s)
        names = {
            X_MIN: "x_min",
            X_MAX: "x_max",
            Y_MIN: "y_min",
            Y_MAX: "y_max",
            Z_MIN: "z_min",
            Z_MAX: "z_max",
        }
        for tag, surfaces in sides.items():
            if not surfaces:
                raise CrossedError(f"unit_cell_mesh: no surface on side {names[tag]}")
            gmsh.model.addPhysicalGroup(2, surfaces, tag, names[tag])
        for slave_side, master_side, shift in (
            (X_MAX, X_MIN, np.array([px, 0.0, 0.0])),
            (Y_MAX, Y_MIN, np.array([0.0, py, 0.0])),
        ):
            slaves, masters = _matching(
                boxes_of, shift, sides[master_side], sides[slave_side], 10 * tol
            )
            affine = [1, 0, 0, shift[0], 0, 1, 0, shift[1], 0, 0, 1, 0, 0, 0, 0, 1]
            gmsh.model.mesh.setPeriodic(2, slaves, masters, affine)
        # element sizes: one constant field per region
        fields = []
        for tag, vols in volumes_by_tag.items():
            if tag not in sizes:
                raise CrossedError(f"unit_cell_mesh: no element size for region {tag}")
            f = gmsh.model.mesh.field.add("Constant")
            gmsh.model.mesh.field.setNumbers(f, "VolumesList", vols)
            gmsh.model.mesh.field.setNumber(f, "VIn", float(sizes[tag]) / u)
            gmsh.model.mesh.field.setNumber(f, "VOut", 1e300)
            fields.append(f)
        combined = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(combined, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(combined)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        curved = any(s.curved for layer in grating.layers for s in layer.shapes)
        gmsh.option.setNumber(
            "Mesh.ElementOrder", 2 if (order == 2 or (order is None and curved)) else 1
        )
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
        gmsh.model.mesh.generate(3)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path


def _shape_volume(occ, shape: Shape, u: float, z0: float, height: float) -> int:
    """OpenCASCADE volume of a shape between z0 and z0 + height (model units)."""
    cx, cy = (float(c) / u for c in shape.center)
    if shape.kind == "circle":
        return occ.addCylinder(cx, cy, z0, 0, 0, height, float(shape.radius) / u)
    if shape.kind == "ellipse":
        rx, ry, angle = float(shape.rx) / u, float(shape.ry) / u, float(shape.angle)
        if rx < ry:  # OpenCASCADE wants the major radius first
            rx, ry, angle = ry, rx, angle + math.pi / 2
        disk = occ.addDisk(cx, cy, z0, rx, ry)
        if angle:
            occ.rotate([(2, disk)], cx, cy, z0, 0, 0, 1, angle)
        out = occ.extrude([(2, disk)], 0, 0, height)
    elif shape.kind == "rectangle":
        w, h = float(shape.width) / u, float(shape.height) / u
        box = occ.addBox(cx - w / 2, cy - h / 2, z0, w, h, height)
        if shape.angle:
            occ.rotate([(3, box)], cx, cy, z0, 0, 0, 1, float(shape.angle))
        return box
    else:
        points = [occ.addPoint(float(x) / u, float(y) / u, z0) for x, y in shape.points]
        lines = [occ.addLine(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]
        surface = occ.addPlaneSurface([occ.addCurveLoop(lines)])
        out = occ.extrude([(2, surface)], 0, 0, height)
    return next(tag for dim, tag in out if dim == 3)


def layout_for(
    grating: CrossedGrating,
    wavelength: float,
    sizes: dict[int, float],
    gap: float | None = None,
    pml: tuple[float, float] | None = None,
) -> CellLayout:
    """Default vertical layout: a gap of ``gap`` [m] (default half a vacuum wavelength, at least
    two cells) between the structure and each PML, the order planes in the middle of the gaps,
    PML thicknesses ``pml = (top, bottom)`` (default one local wavelength rounded up to whole
    cells)."""
    h_top, h_bottom = sizes[SUPERSTRATE_TAG], sizes[SUBSTRATE_TAG]
    g_top = max(0.5 * wavelength if gap is None else float(gap), 2 * h_top)
    g_bottom = max(0.5 * wavelength if gap is None else float(gap), 2 * h_bottom)
    if pml is None:
        t_top = t_bottom = None
    else:
        t_top, t_bottom = map(float, pml)
    if t_top is None:
        t_top = math.ceil(wavelength / h_top - 1e-9) * h_top
    if t_bottom is None:
        t_bottom = math.ceil(wavelength / h_bottom - 1e-9) * h_bottom
    depth = grating.depth
    return CellLayout(
        z_top=g_top + t_top,
        pml_top=g_top,
        z_above=0.5 * g_top,
        z_below=-depth - 0.5 * g_bottom,
        pml_bottom=-depth - g_bottom,
        z_bottom=-depth - g_bottom - t_bottom,
    )


def element_sizes(
    grating: CrossedGrating, omega: float, p: int, elements_per_wavelength: float | None = None
) -> dict[int, float]:
    """Element size per region tag [m] (:func:`hpfem.meshing.element_sizes`: the local
    wavelength over ``elements_per_wavelength``, default 8 / p in 3D, and the decay length in
    absorbing media); a layer's background is at most the size of its shapes' smallest
    dimension / 2."""
    from hpfem import meshing

    if elements_per_wavelength is None:
        elements_per_wavelength = 8.0 / p
    sizes = meshing.element_sizes(
        _region_materials(grating, omega), omega, p, elements_per_wavelength
    )
    for i, layer in enumerate(grating.layers):
        sizes[layer_tag(i)] = min(sizes[layer_tag(i)], float(layer.thickness))
        for j, shape in enumerate(layer.shapes):
            x0, x1, y0, y1 = shape.bounds()
            feature = 0.5 * min(x1 - x0, y1 - y0)
            sizes[shape_tag(i, j)] = min(sizes[shape_tag(i, j)], feature, float(layer.thickness))
            sizes[layer_tag(i)] = min(sizes[layer_tag(i)], max(feature, sizes[shape_tag(i, j)]))
    return sizes


def unit_cell_mesh(
    grating: CrossedGrating,
    layout: CellLayout,
    sizes: dict[int, float],
    path: str | Path | None = None,
    **options,
):
    """:func:`write_unit_cell` followed by ``hpfem.read_gmsh_periodic``: ``(mesh, links)`` in
    metres; ``path`` defaults to a temporary file."""
    import hpfem

    unit = float(options.get("unit", 1e-9))
    if path is None:
        with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as handle:
            path = handle.name
    write_unit_cell(grating, layout, path, sizes, **options)
    return hpfem.read_gmsh_periodic(str(path), unit, 3)


# ---------------------------------------------------------------------------------- solve


@dataclass
class CrossedOrder:
    """One order (m, n) of the reflected or transmitted field."""

    m: int
    n: int
    efficiency: float
    """power fraction of the incident wave (0 for evanescent orders)"""
    amplitude: np.ndarray
    """complex vector amplitude (E_x, E_y, E_z) on the order plane (phase reference at x = y = 0
    of the plane)"""
    k_tangential: np.ndarray
    """(k_x, k_y, 0) of the order [1/m]"""
    kn: complex
    """normal wavenumber in the medium of the plane, imaginary for evanescent orders"""
    propagating: bool
    direction: np.ndarray | None = None
    """unit propagation direction of a propagating order (reflected up, transmitted down),
    ``None`` for an evanescent one"""


@dataclass
class CrossedResult:
    """Orders of a crossed grating for one polarisation (default as solved) with the raw s and
    p amplitudes, so :meth:`polarised` gives any other polarisation without a new solve."""

    wavelength: float
    theta: float
    phi: float
    polarisation: tuple[complex, complex]
    """(a_s, a_p) of the incident wave, |a_s|² + |a_p|² = 1"""
    reflected: list[CrossedOrder]
    transmitted: list[CrossedOrder]
    absorption: float | None
    """absorbed power / incident power in the structure (None if not evaluated)"""
    num_dofs: int
    num_cells: int
    layout: CellLayout
    _raw: dict = field(repr=False, default_factory=dict)

    @property
    def R(self) -> float:  # noqa: N802 (the usual symbol)
        """Total reflectance ΣR."""
        return float(sum(o.efficiency for o in self.reflected))

    @property
    def T(self) -> float:  # noqa: N802
        """Total transmittance ΣT."""
        return float(sum(o.efficiency for o in self.transmitted))

    @property
    def balance(self) -> float | None:
        """ΣR + ΣT + A − 1 (a quality indicator: 0 up to the discretisation error)."""
        if self.absorption is None:
            return None
        return self.R + self.T + self.absorption - 1.0

    def order(self, m: int, n: int, side: str = "R") -> CrossedOrder:
        """The order (m, n) of the reflected (``"R"``) or transmitted (``"T"``) field."""
        orders = self.reflected if side == "R" else self.transmitted
        for o in orders:
            if o.m == m and o.n == n:
                return o
        raise KeyError(f"order ({m}, {n}) not computed (max_order too small)")

    def table(self, propagating_only: bool = True) -> list[dict]:
        """Rows ``{"side", "m", "n", "efficiency"}`` for printing or a data frame."""
        rows = []
        for side, orders in (("R", self.reflected), ("T", self.transmitted)):
            for o in orders:
                if o.propagating or not propagating_only:
                    rows.append({"side": side, "m": o.m, "n": o.n, "efficiency": o.efficiency})
        return rows

    def polarised(self, polarisation) -> CrossedResult:
        """The result for another incident polarisation (``"s"``, ``"p"`` or (a_s, a_p)) from
        the stored s and p solutions; the absorption needs the solve's problem objects and is
        recomputed only for absorbing structures."""
        return _assemble_result(self._raw, _polarisation(polarisation))

    def field(self, points, quantity: str = "E", scattered: bool = False) -> np.ndarray:
        """The field of this result's polarisation at ``points`` (n, 3) [m]: rows (E_x, E_y,
        E_z) [V/m] for an incident wave of amplitude 1 V/m (``quantity="H"``: the magnetic
        field [A/m]); the total field, or the scattered one against the layered background
        with ``scattered``. Points outside the cell in x or y are mapped back with the Bloch
        phase, points above or below the mesh give NaN. Phase reference of the incident wave:
        its value at z = 0 (the top of the layers)."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if points.shape[1] != 3:
            raise CrossedError("field: points must have shape (n, 3)")
        a_s, a_p = self.polarisation
        out = np.zeros((points.shape[0], 3), dtype=complex)
        for key, a in (("s", a_s), ("p", a_p)):
            if a == 0:
                continue
            values, _ = self._raw["problems"][key].sample(
                self._raw["solutions"][key],
                self._raw["locator"],
                points,
                scattered=scattered,
                quantity=quantity,
            )
            out += a * np.asarray(values)
        return out


def _polarisation(spec) -> tuple[complex, complex]:
    if isinstance(spec, str):
        key = spec.lower()
        if key in ("s", "te"):
            return (1.0 + 0j, 0j)
        if key in ("p", "tm"):
            return (0j, 1.0 + 0j)
        raise CrossedError(f"polarisation {spec!r}: 's', 'p' or (a_s, a_p)")
    a_s, a_p = (complex(a) for a in spec)
    norm = math.sqrt(abs(a_s) ** 2 + abs(a_p) ** 2)
    if norm == 0:
        raise CrossedError("polarisation (a_s, a_p) must not be zero")
    return (a_s / norm, a_p / norm)


def _orders_of(raw_s, raw_p, a_s, a_p, kn_incident, normal) -> list[CrossedOrder]:
    out = []
    for os_, op_ in zip(raw_s, raw_p, strict=True):
        amplitude = a_s * np.asarray(os_.amplitude) + a_p * np.asarray(op_.amplitude)
        efficiency = (
            float(os_.kn.real * np.vdot(amplitude, amplitude).real / kn_incident)
            if os_.propagating
            else 0.0
        )
        order = CrossedOrder(
            m=os_.m,
            n=os_.n,
            efficiency=efficiency,
            amplitude=amplitude,
            k_tangential=np.asarray(os_.k_tangential),
            kn=complex(os_.kn),
            propagating=bool(os_.propagating),
            direction=np.asarray(os_.direction(normal)) if os_.propagating else None,
        )
        out.append(order)
    return out


def _assemble_result(raw: dict, polarisation: tuple[complex, complex]) -> CrossedResult:
    a_s, a_p = polarisation
    kn = raw["kn_incident"]
    reflected = _orders_of(raw["R_s"], raw["R_p"], a_s, a_p, kn, raw["up"])
    transmitted = _orders_of(raw["T_s"], raw["T_p"], a_s, a_p, kn, raw["down"])
    absorption = raw["absorption"](a_s, a_p)
    return CrossedResult(
        wavelength=raw["wavelength"],
        theta=raw["theta"],
        phi=raw["phi"],
        polarisation=(a_s, a_p),
        reflected=reflected,
        transmitted=transmitted,
        absorption=absorption,
        num_dofs=raw["num_dofs"],
        num_cells=raw["num_cells"],
        layout=raw["layout"],
        _raw=raw,
    )


def solve(
    grating: CrossedGrating,
    wavelength: float,
    theta: float = 0.0,
    phi: float = 0.0,
    polarisation="s",
    *,
    order: int = 3,
    max_order: int | None = None,
    elements_per_wavelength: float | None = None,
    gap: float | None = None,
    pml: tuple[float, float] | None = None,
    pml_target: float = 1e-6,
    pml_profile=None,
    mesh=None,
    num_points: int | None = None,
    path: str | Path | None = None,
) -> CrossedResult:
    """Reflected and transmitted orders (m, n) of a crossed grating.

    ``wavelength`` [m] in vacuum; ``theta``, ``phi`` [rad] the polar angle from the normal and
    the azimuth of the tangential wavevector from +x; ``polarisation`` ``"s"``, ``"p"`` or
    (a_s, a_p). ``order`` is the polynomial order, ``elements_per_wavelength`` the mesh density
    (default 8 / order per local wavelength), ``max_order`` the largest |m|, |n| measured
    (default: all propagating orders plus one). ``gap`` [m] between structure and PML (default
    λ/2), ``pml = (top, bottom)`` thicknesses [m] (default one wavelength in whole cells) with a
    profile designed for the steepest propagating order and the reflection ``pml_target`` (at
    most as steep as R0 = 1e-10), or the ``hpfem.PmlProfile`` given as ``pml_profile``. For a
    convergence table keep ``elements_per_wavelength`` fixed and raise ``order``.
    ``mesh``: a ``(mesh, links, layout)`` triple from an earlier result's
    ``CrossedResult._raw["mesh"]`` to skip meshing (same grating, wavelength and order);
    ``path`` keeps the Gmsh file. ``num_points``: Gauss points per lattice direction on the order
    planes (default 8 per cell across the period, at least 8 per order)."""
    import hpfem

    if not wavelength > 0:
        raise CrossedError("solve: the wavelength must be positive")
    if not 0 <= theta < math.pi / 2:
        raise CrossedError("solve: theta must lie in [0, pi/2)")
    pol = _polarisation(polarisation)
    omega = 2 * math.pi * hpfem.constants.c0 / wavelength
    k0 = 2 * math.pi / wavelength
    mats = _region_materials(grating, omega)
    n_sup, n_sub = _index(mats[SUPERSTRATE_TAG]), _index(mats[SUBSTRATE_TAG])
    if abs(n_sup.imag) > 1e-12:
        raise CrossedError("solve: the superstrate must be lossless")
    if abs(n_sub.imag) > 1e-12 or not n_sub.real > 0:
        raise CrossedError(
            "solve: absorbing or metallic substrates are not supported yet (the transmitted orders "
            "are measured on a plane in the substrate)"
        )
    px, py = (float(p) for p in grating.period)
    kt = k0 * n_sup.real * math.sin(theta) * np.array([math.cos(phi), math.sin(phi), 0.0])
    # orders: all propagating in either half-space, plus one
    n_max = max(n_sup.real, n_sub.real)
    if max_order is None:
        reach = max(px, py) * (k0 * n_max + np.linalg.norm(kt)) / (2 * math.pi)
        max_order = int(math.floor(reach)) + 1
    # PML profile for the steepest propagating order
    theta_max = float(theta)
    for n_half in (n_sup.real, n_sub.real):
        for m in range(-max_order, max_order + 1):
            for n in range(-max_order, max_order + 1):
                k_t = np.linalg.norm(kt + 2 * math.pi * np.array([m / px, n / py, 0.0]))
                s = k_t / (k0 * n_half)
                if s < 1:
                    theta_max = max(theta_max, math.asin(s))
    theta_max = min(theta_max, math.radians(80))
    if pml_profile is None:
        # steeper than R0 = 1e-10 only costs resolution: the efficiencies of the hole array of
        # docs/python.md moved by 2e-4 and its balance got worse, not better
        profile = hpfem.PmlProfile.for_angle(math.degrees(theta_max), pml_target, 1.0, 2)
        profile = hpfem.PmlProfile(2, max(profile.reflection, 1e-10))
    else:
        profile = pml_profile

    if mesh is None:
        sizes = element_sizes(grating, omega, order, elements_per_wavelength)
        layout = layout_for(grating, wavelength, sizes, gap, pml)
        fem_mesh, links = unit_cell_mesh(grating, layout, sizes, path)
    else:
        fem_mesh, links, layout = mesh
        sizes = None

    stack = hpfem.LayerStack3D(
        mats[SUPERSTRATE_TAG],
        [
            hpfem.Layer(mats[layer_tag(i)], float(layer.thickness))
            for i, layer in enumerate(grating.layers)
        ],
        mats[SUBSTRATE_TAG],
        0.0,
    )
    waves = {
        "s": stack.plane_wave(k0, theta, hpfem.Polarisation.S, 1.0, phi),
        "p": stack.plane_wave(k0, theta, hpfem.Polarisation.P, 1.0, phi),
    }
    material_map = hpfem.MaterialMap(mats[SUPERSTRATE_TAG])
    for tag, material in mats.items():
        material_map.set(tag, material)
    n_ref = min(n_sup.real, n_sub.real)
    pml_box = hpfem.PmlBox3D(
        [0.0, 0.0, layout.pml_bottom],
        [px, py, layout.pml_top],
        [0.0, 0.0, 0.0, 0.0, layout.pml_bottom - layout.z_bottom, layout.z_top - layout.pml_top],
        k0,
        n_ref,
        profile,
    )
    pairs = []
    for master, slave, shift in links or [
        (X_MIN, X_MAX, [px, 0.0, 0.0]),
        (Y_MIN, Y_MAX, [0.0, py, 0.0]),
    ]:
        shift = np.asarray(shift, dtype=float)
        pairs.append(
            hpfem.PeriodicPair3D(
                int(master), int(slave), shift, complex(np.exp(1j * kt.dot(shift)))
            )
        )

    def setup_for(wave):
        s = hpfem.ScatteringSetup3D()
        s.omega = omega
        s.materials = material_map
        s.background = stack
        s.incident = wave.field
        s.incident_wave = wave.incident_wave
        s.formulation = hpfem.Formulation.SCATTERED_FIELD
        s.pml = pml_box
        s.incident_tags = [Z_MIN, Z_MAX]
        s.periodic = pairs
        return s

    dofs = hpfem.NedelecDofMap3D(fem_mesh, order)
    problems = {key: hpfem.Scattering3D(dofs, setup_for(wave)) for key, wave in waves.items()}
    operator = hpfem.ScatteringOperator3D(problems["s"])
    solutions = {"s": operator.solve(), "p": operator.solve(waves["p"].field)}
    locator = hpfem.PointLocator3D(fem_mesh)

    if num_points is None:
        h_plane = min(
            sizes[SUPERSTRATE_TAG] if sizes else max(px, py) / 4,
            sizes[SUBSTRATE_TAG] if sizes else max(px, py) / 4,
        )
        num_points = max(8 * (2 * max_order + 1), 8 * math.ceil(max(px, py) / h_plane))
    up = np.array([0.0, 0.0, 1.0])
    down = -up
    above = hpfem.OrderPlane([0.0, 0.0, layout.z_above], [px, 0.0, 0.0], [0.0, py, 0.0], up)
    below = hpfem.OrderPlane([0.0, 0.0, layout.z_below], [px, 0.0, 0.0], [0.0, py, 0.0], down)
    kn_incident = k0 * n_sup.real * math.cos(theta)
    raw = {
        "wavelength": wavelength,
        "theta": theta,
        "phi": phi,
        "kn_incident": kn_incident,
        "up": up,
        "down": down,
        "num_dofs": dofs.num_dofs,
        "num_cells": fem_mesh.num_cells,
        "layout": layout,
        "mesh": (fem_mesh, links, layout),
        "problems": problems,
        "solutions": solutions,
        "locator": locator,
        "dofs": dofs,
    }
    for key in ("s", "p"):
        problem, solution = problems[key], solutions[key]

        def total(x, problem=problem, solution=solution):
            e = problem.total_field(solution, locator, x)
            if e is None:
                raise CrossedError(f"order plane point {x} outside the mesh")
            return e

        raw[f"R_{key}"] = hpfem.diffraction_orders(
            total,
            above,
            k0,
            n_sup.real,
            kt,
            kn_incident,
            waves[key].incident_wave.value,
            max_order,
            num_points,
        )
        raw[f"T_{key}"] = hpfem.diffraction_orders(
            total, below, k0, n_sub.real, kt, kn_incident, None, max_order, num_points
        )
    incident_power = (
        hpfem.plane_wave_intensity(1.0, mats[SUPERSTRATE_TAG]) * math.cos(theta) * px * py
    )
    raw["absorption"] = _absorption_function(
        problems, solutions, waves, mats, incident_power, dofs, setup_for
    )
    return _assemble_result(raw, pol)


def _absorption_function(problems, solutions, waves, mats, incident_power, dofs, setup_for):
    """A function (a_s, a_p) -> absorbed / incident power: zero without absorbing regions; the
    pure polarisations from their problems, a mixture through a problem with the combined
    background field (Python callbacks over the absorbing cells only)."""
    import hpfem

    lossy = any(abs(complex(m.refractive_index).imag) > 1e-12 for m in mats.values())
    cache: dict = {}

    def absorption(a_s, a_p):
        if not lossy:
            return 0.0
        key = (complex(a_s), complex(a_p))
        if key in cache:
            return cache[key]
        if a_p == 0 or a_s == 0:
            pure = "s" if a_p == 0 else "p"
            power = hpfem.absorbed_power(problems[pure], solutions[pure])
        else:
            ws, wp = waves["s"].field, waves["p"].field
            field_ = hpfem.IncidentField3D(
                lambda x: a_s * np.asarray(ws.value(x)) + a_p * np.asarray(wp.value(x)),
                lambda x: a_s * np.asarray(ws.curl(x)) + a_p * np.asarray(wp.curl(x)),
            )
            setup = setup_for(waves["s"])
            setup.incident = field_
            problem = hpfem.Scattering3D(dofs, setup)
            unknown = a_s * solutions["s"].unknown + a_p * solutions["p"].unknown
            mixed = hpfem.ScatteringSolution3D(hpfem.Formulation.SCATTERED_FIELD, unknown)
            power = hpfem.absorbed_power(problem, mixed)
        cache[key] = float(power / incident_power)
        return cache[key]

    return absorption

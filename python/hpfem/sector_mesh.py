"""Sector meshing for structures with cyclic symmetry C_n (issue #244 S5) with the Gmsh Python
API.

A :class:`Sector` describes the full structure around the z axis (2D: around the origin) and
the domain: a disc of ``radius`` (2D), a ball (3D) or a cylinder between two heights (3D,
``z_range``). ``rings`` are circles / spheres / cylinders around the axis that become mesh lines
(material shells, the start of a :class:`hpfem.RadialPml2D` / ``RadialPml3D``), ``slabs`` are z
layers (3D), ``shapes`` are :class:`hpfem.meshing.Shape` (2D) or :class:`Solid` (3D) scatterers,
given for the **whole** structure (all n copies). :func:`write_sector` meshes the sector between
the angles 0 and 2π/n with Gmsh (OpenCASCADE) such that the two sides match under the rotation
(Gmsh ``setPeriodic``), which the rotated periodic coupling of
:class:`hpfem.CyclicSymmetricScattering2D` / ``3D`` needs; the cells carry the material tags,
the side at angle 0 the ``master_tag``, the side at 2π/n the ``slave_tag`` and the outer
boundary the ``boundary_tag`` (``hpfem.DISC_BOUNDARY`` by default, the PEC wall behind a PML).
The structure must have the symmetry: the material tag of every piece of the sector must be
the same at its n rotated copies (checked, ``ValueError``). :func:`sector_mesh` reads the file
back and returns the mesh and the matching :class:`hpfem.CyclicSymmetry2D` / ``3D``.

The Gmsh part needs the ``gmsh`` package (``pip install gmsh``); the model is built in units of
``unit`` metres (default nanometres), as in :func:`hpfem.meshing.write_unit_cell`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .meshing import Slab

DISC_BOUNDARY = 1  # hpfem.DISC_BOUNDARY


@dataclass
class Solid:
    """A 3D scatterer. ``kind``: ``"sphere"`` (``x``, ``y``, ``z``, ``r``: centre and radius),
    ``"cylinder"`` (``x``, ``y``, ``z``, ``r``, ``height``: axis along z from the base centre
    ``(x, y, z)``) or ``"box"`` (``x``, ``y``, ``z``, ``dx``, ``dy``, ``dz``: lower corner and
    edge lengths)."""

    kind: str
    tag: int
    params: dict = field(default_factory=dict)

    def contains(self, x: float, y: float, z: float) -> bool:
        p = self.params
        if self.kind == "sphere":
            return (x - p["x"]) ** 2 + (y - p["y"]) ** 2 + (z - p["z"]) ** 2 <= p["r"] ** 2
        if self.kind == "cylinder":
            inside = (x - p["x"]) ** 2 + (y - p["y"]) ** 2 <= p["r"] ** 2
            return inside and p["z"] <= z <= p["z"] + p["height"]
        if self.kind == "box":
            return (
                p["x"] <= x <= p["x"] + p["dx"]
                and p["y"] <= y <= p["y"] + p["dy"]
                and p["z"] <= z <= p["z"] + p["dz"]
            )
        raise ValueError(f"Solid: unknown kind {self.kind!r}")


@dataclass
class Sector:
    """One sector of a C_n-symmetric structure around the z axis (2D: the origin)."""

    order: int
    """n: the sector spans the angles 0 .. 2π/n"""
    radius: float
    """outer radius of the domain (disc, ball or cylinder) [m]"""
    shapes: list = field(default_factory=list)
    """:class:`hpfem.meshing.Shape` (2D) or :class:`Solid` (3D), the whole structure; a later
    shape wins over an earlier one, a shape over the slabs and rings"""
    rings: list[tuple[float, int]] = field(default_factory=list)
    """``(r, tag)``: the region within distance r of the axis (3D ball: of the origin) and
    outside the smaller rings carries ``tag``; every r becomes a mesh line (surface)"""
    slabs: list[Slab] = field(default_factory=list)
    """3D: layers ``Slab(tag, z0, z1)`` (the y range of :class:`Slab` read as z); a slab wins over
    the rings"""
    background_tag: int = 1
    """tag of the region covered by nothing else"""
    dim: int = 2
    z_range: tuple[float, float] | None = None
    """3D: the domain is the cylinder of ``radius`` between these heights; None: the ball"""
    master_tag: int = 20
    slave_tag: int = 21
    boundary_tag: int = DISC_BOUNDARY

    @property
    def angle(self) -> float:
        return 2.0 * math.pi / self.order

    def tag_at(self, x: float, y: float, z: float = 0.0) -> int:
        """Material tag at a point [m]."""
        for shape in reversed(self.shapes):
            inside = shape.contains(x, y, z) if self.dim == 3 else shape.contains(x, y)
            if inside:
                return shape.tag
        if self.dim == 3:
            for slab in self.slabs:
                if slab.y0 <= z <= slab.y1:
                    return slab.tag
        distance = math.hypot(x, y)
        if self.dim == 3 and self.z_range is None:
            distance = math.sqrt(x * x + y * y + z * z)
        for r, tag in sorted(self.rings):
            if distance <= r:
                return tag
        return self.background_tag

    def symmetry(self):
        """The matching ``hpfem.CyclicSymmetry2D`` / ``3D`` (axis z through the origin)."""
        import hpfem

        if self.dim == 2:
            return hpfem.CyclicSymmetry2D(self.order, master=self.master_tag, slave=self.slave_tag)
        return hpfem.CyclicSymmetry3D(self.order, master=self.master_tag, slave=self.slave_tag)

    def check(self) -> None:
        """Raises ValueError for an unusable description."""
        if self.order < 2:
            raise ValueError(f"Sector: the order {self.order} must be at least 2")
        if self.dim not in (2, 3):
            raise ValueError(f"Sector: dim must be 2 or 3, not {self.dim}")
        if not self.radius > 0:
            raise ValueError("Sector: the radius must be positive")
        if any(not 0 < r < self.radius for r, _ in self.rings):
            raise ValueError("Sector: every ring radius must lie between 0 and the radius")
        if self.dim == 2 and (self.slabs or self.z_range is not None):
            raise ValueError("Sector: slabs and z_range belong to 3D sectors")
        if self.z_range is not None and not self.z_range[1] > self.z_range[0]:
            raise ValueError("Sector: z_range must increase")
        tags = {self.master_tag, self.slave_tag, self.boundary_tag}
        if len(tags) != 3:
            raise ValueError("Sector: the master, slave and boundary tags must differ")
        for shape in self.shapes:
            if (self.dim == 3) != isinstance(shape, Solid):
                raise ValueError("Sector: 2D sectors take Shape, 3D sectors Solid")


def _rotate(x: float, y: float, angle: float) -> tuple[float, float]:
    c, s = math.cos(angle), math.sin(angle)
    return c * x - s * y, s * x + c * y


def write_sector(
    sector: Sector,
    path: str | Path,
    sizes: Mapping[int, float],
    *,
    default_size: float | None = None,
    interface_factor: float = 0.5,
    order: int = 2,
    unit: float = 1e-9,
    verbose: bool = False,
) -> Path:
    """Meshes the sector with Gmsh (OpenCASCADE) and writes a MSH 4.1 ASCII file with physical
    groups (cells: material tags; facets: the two sides and the outer boundary) and the
    ``$Periodic`` section of the rotation by 2π/n, so that the side at 2π/n is the rotated
    side at 0 node by node. ``sizes`` are element sizes per tag [m] (``default_size`` for the
    others), ``interface_factor`` scales the size at material interfaces, ``order`` 2 (the
    default; the outer boundary is curved) gives curved cells. Raises ValueError if the
    structure is not C_n-symmetric."""
    import gmsh  # the only place that needs it

    sector.check()
    path = Path(path)
    u = float(unit)
    dim = sector.dim
    a = sector.angle
    radius = sector.radius / u
    z_range = None if sector.z_range is None else (sector.z_range[0] / u, sector.z_range[1] / u)
    extent = radius if z_range is None else max(radius, abs(z_range[0]), abs(z_range[1]))
    big = 4.0 * extent
    tol = 1e-7 * extent
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add("sector")
        occ = gmsh.model.occ
        # the wedge 0 <= phi <= 2 pi / n (a fan polygon, extruded along z in 3D)
        z_low = -big if dim == 3 else 0.0
        fan = [occ.addPoint(0.0, 0.0, z_low)]
        steps = max(2, int(math.ceil(a / (math.pi / 8))))
        for k in range(steps + 1):
            t = a * k / steps
            fan.append(occ.addPoint(big * math.cos(t), big * math.sin(t), z_low))
        lines = [occ.addLine(fan[i], fan[(i + 1) % len(fan)]) for i in range(len(fan))]
        wedge = [(2, occ.addPlaneSurface([occ.addCurveLoop(lines)]))]
        if dim == 3:
            wedge = [e for e in occ.extrude(wedge, 0, 0, 2 * big) if e[0] == 3]
        # the domain and the tools
        if dim == 2:
            domain = [(2, occ.addDisk(0, 0, 0, radius, radius))]
        elif z_range is None:
            domain = [(3, occ.addSphere(0, 0, 0, radius))]
        else:
            domain = [(3, occ.addCylinder(0, 0, z_range[0], 0, 0, z_range[1] - z_range[0], radius))]
        piece, _ = occ.intersect(domain, wedge, removeObject=True, removeTool=True)
        tools = []
        for r, _ in sector.rings:
            r = r / u
            if dim == 2:
                tools.append((2, occ.addDisk(0, 0, 0, r, r)))
            elif z_range is None:
                tools.append((3, occ.addSphere(0, 0, 0, r)))
            else:
                tools.append((3, occ.addCylinder(0, 0, -big, 0, 0, 2 * big, r)))
        for slab in sector.slabs:
            z0, z1 = slab.y0 / u, slab.y1 / u
            tools.append((3, occ.addBox(-big, -big, z0, 2 * big, 2 * big, z1 - z0)))
        for shape in sector.shapes:
            p = {k: (v / u if isinstance(v, (int, float)) else v) for k, v in shape.params.items()}
            if dim == 3:
                if shape.kind == "sphere":
                    tools.append((3, occ.addSphere(p["x"], p["y"], p["z"], p["r"])))
                elif shape.kind == "cylinder":
                    tools.append(
                        (3, occ.addCylinder(p["x"], p["y"], p["z"], 0, 0, p["height"], p["r"]))
                    )
                elif shape.kind == "box":
                    tools.append((3, occ.addBox(p["x"], p["y"], p["z"], p["dx"], p["dy"], p["dz"])))
                else:
                    raise ValueError(f"write_sector: unknown solid {shape.kind!r}")
            elif shape.kind == "ellipse":
                tools.append((2, occ.addDisk(p["x"], p["y"], 0, p["rx"], p["ry"])))
            else:
                points = [occ.addPoint(px / u, py / u, 0) for px, py in shape.polygon()]
                loop = [
                    occ.addLine(points[i], points[(i + 1) % len(points)])
                    for i in range(len(points))
                ]
                tools.append((2, occ.addPlaneSurface([occ.addCurveLoop(loop)])))
        if tools:
            occ.fragment(piece, tools, removeObject=True, removeTool=True)
        occ.synchronize()

        def inside_domain(x: float, y: float, z: float) -> bool:
            if math.hypot(x, y) > tol and not -1e-9 <= math.atan2(y, x) <= a + 1e-9:
                return False
            if dim == 2 or z_range is not None:
                if math.hypot(x, y) > radius + tol:
                    return False
                return z_range is None or z_range[0] - tol <= z <= z_range[1] + tol
            return math.sqrt(x * x + y * y + z * z) <= radius + tol

        def inner_point(d: int, tag: int) -> tuple[float, float, float]:
            """A point inside the entity: its centre of mass, or a sample of its bounding box."""
            c = occ.getCenterOfMass(d, tag)
            if gmsh.model.isInside(d, tag, list(c)):
                return c
            x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(d, tag)
            m = 9
            for i in range(1, m):
                for j in range(1, m):
                    for k in range(1, m if d == 3 else 2):
                        q = (
                            x0 + (x1 - x0) * i / m,
                            y0 + (y1 - y0) * j / m,
                            z0 + (z1 - z0) * k / m if d == 3 else c[2],
                        )
                        if gmsh.model.isInside(d, tag, list(q)):
                            return q
            return c

        # keep the pieces of the sector, tag them, check the symmetry
        pieces_by_tag: dict[int, list[int]] = {}
        tag_of_piece: dict[int, int] = {}
        for d, t in gmsh.model.getEntities(dim):
            q = inner_point(d, t)
            if not inside_domain(*q):
                occ.remove([(d, t)], recursive=True)
                continue
            tag = sector.tag_at(q[0] * u, q[1] * u, q[2] * u)
            for j in range(1, sector.order):
                x, y = _rotate(q[0], q[1], j * a)
                other = sector.tag_at(x * u, y * u, q[2] * u)
                if other != tag:
                    raise ValueError(
                        f"write_sector: the structure is not C_{sector.order}-symmetric: tag "
                        f"{tag} at ({q[0] * u:.4g}, {q[1] * u:.4g}, {q[2] * u:.4g}) m but tag "
                        f"{other} at its rotation by {j}·2π/{sector.order}"
                    )
            pieces_by_tag.setdefault(tag, []).append(t)
            tag_of_piece[t] = tag
        occ.synchronize()
        if not pieces_by_tag:
            raise ValueError("write_sector: the sector is empty")
        for tag, pieces in pieces_by_tag.items():
            gmsh.model.addPhysicalGroup(dim, pieces, tag, f"material_{tag}")

        # facets: the two sides (planar, on the half-planes at 0 and 2 pi / n), the outer
        # boundary (one adjacent piece) and the material interfaces
        # (unit direction of the half-plane, its normal); for n = 2 both lie on one line
        halves = {
            sector.master_tag: ((1.0, 0.0), (0.0, 1.0)),
            sector.slave_tag: ((math.cos(a), math.sin(a)), (-math.sin(a), math.cos(a))),
        }

        def on_side(d: int, t: int, half: tuple) -> bool:
            direction, normal = half
            lo, hi = gmsh.model.getParametrizationBounds(d, t)
            samples = []
            for i in range(3):
                for j in range(3 if d == 2 else 1):
                    s = [lo[0] + (hi[0] - lo[0]) * (i + 0.5) / 3]
                    if d == 2:
                        s.append(lo[1] + (hi[1] - lo[1]) * (j + 0.5) / 3)
                    samples.extend(s)
            xyz = gmsh.model.getValue(d, t, samples)
            for k in range(0, len(xyz), 3):
                if abs(normal[0] * xyz[k] + normal[1] * xyz[k + 1]) > tol:
                    return False
                if direction[0] * xyz[k] + direction[1] * xyz[k + 1] < -tol:
                    return False
            return True

        sides: dict[int, list[int]] = {sector.master_tag: [], sector.slave_tag: []}
        boundary: list[int] = []
        interfaces: list[int] = []
        for d, t in gmsh.model.getEntities(dim - 1):
            adjacent = [s for s in gmsh.model.getAdjacencies(d, t)[0] if s in tag_of_piece]
            if not adjacent:
                continue
            side = next((k for k, h in halves.items() if on_side(d, t, h)), None)
            if side is not None and len(adjacent) == 1:
                sides[side].append(t)
            elif len(adjacent) == 1:
                boundary.append(t)
            elif len({tag_of_piece[s] for s in adjacent}) > 1:
                interfaces.append(t)
        if not sides[sector.master_tag] or not sides[sector.slave_tag]:
            raise ValueError("write_sector: no facets found on the sides of the sector")
        gmsh.model.addPhysicalGroup(dim - 1, sides[sector.master_tag], sector.master_tag, "master")
        gmsh.model.addPhysicalGroup(dim - 1, sides[sector.slave_tag], sector.slave_tag, "slave")
        if boundary:
            gmsh.model.addPhysicalGroup(dim - 1, boundary, sector.boundary_tag, "boundary")
        # each slave facet is the rotation of one master facet
        c, s = math.cos(a), math.sin(a)
        affine = [c, -s, 0, 0, s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        masters = {t: occ.getCenterOfMass(dim - 1, t) for t in sides[sector.master_tag]}
        pairs_slave, pairs_master = [], []
        for t in sides[sector.slave_tag]:
            cs = occ.getCenterOfMass(dim - 1, t)
            best = min(
                masters,
                key=lambda m: math.dist(
                    (*_rotate(masters[m][0], masters[m][1], a), masters[m][2]), cs
                ),
            )
            image = (*_rotate(masters[best][0], masters[best][1], a), masters[best][2])
            if math.dist(image, cs) > 1e-5 * extent:
                raise ValueError(
                    "write_sector: the two sides of the sector do not match under the rotation"
                )
            pairs_slave.append(t)
            pairs_master.append(best)
        if len(set(pairs_master)) != len(pairs_master) or len(masters) != len(pairs_slave):
            raise ValueError("write_sector: the two sides of the sector do not match")
        gmsh.model.mesh.setPeriodic(dim - 1, pairs_slave, pairs_master, affine)

        # sizes: one constant field per tag, the minimum over them, refined at interfaces
        all_sizes = [float(h) / u for h in sizes.values()]
        if default_size is not None:
            all_sizes.append(float(default_size) / u)
        if not all_sizes:
            raise ValueError("write_sector: give sizes per tag or a default_size")
        fields = []
        for tag, pieces in pieces_by_tag.items():
            h = sizes.get(tag, default_size)
            if h is None:
                raise ValueError(f"write_sector: no element size for tag {tag} and no default_size")
            f = gmsh.model.mesh.field.add("Constant")
            gmsh.model.mesh.field.setNumbers(
                f, "SurfacesList" if dim == 2 else "VolumesList", pieces
            )
            gmsh.model.mesh.field.setNumber(f, "VIn", float(h) / u)
            gmsh.model.mesh.field.setNumber(f, "VOut", 1e300)
            fields.append(f)
        if interfaces and interface_factor < 1.0:
            h_min = min(all_sizes)
            dist = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(
                dist, "CurvesList" if dim == 2 else "SurfacesList", interfaces
            )
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
        gmsh.option.setNumber("Mesh.ElementOrder", 2 if order == 2 else 1)
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
        gmsh.model.mesh.generate(dim)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path


def sector_mesh(
    sector: Sector, sizes: Mapping[int, float], path: str | Path | None = None, **options
):
    """:func:`write_sector` followed by ``hpfem.read_gmsh``: returns ``(mesh, symmetry)`` with
    the mesh in metres and the matching ``hpfem.CyclicSymmetry2D`` / ``3D``; ``path`` defaults
    to a temporary file."""
    import tempfile

    import hpfem

    if path is None:
        with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as handle:
            path = handle.name
    unit = float(options.get("unit", 1e-9))
    write_sector(sector, path, sizes, **options)
    return hpfem.read_gmsh(str(path), unit, sector.dim), sector.symmetry()


__all__ = ["Sector", "Solid", "sector_mesh", "write_sector"]

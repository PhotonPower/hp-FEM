"""Design parameters of a grating unit cell and the morphing of a reference mesh
(M16 S1, ADR-0012 §3).

A :class:`MaterialParameter` is the real or imaginary part of the relative permittivity of a
cell tag. A :class:`GeometryParameter` is a scalar of one :class:`hpfem.meshing.Shape` of a
:class:`hpfem.meshing.UnitCell` — a field such as ``height``, or a derived quantity such as the
mid-height width (CD) or the side-wall angle of a trapezoid (:func:`trapezoid_parameters`) —
given by a getter and a setter on the shape's ``params``.

The mesh velocity :math:`V = \\partial x/\\partial p` of a geometry parameter
(:func:`shape_velocity`) moves the nodes on the shape's boundary exactly as the boundary
moves (central differences of the corners of a polygon shape, of the boundary point at fixed
angle of an ellipse), keeps the cell boundary and every other material interface fixed and
extends the motion harmonically into the remaining nodes, so that large steps keep the cells
regular. :class:`Morph` keeps a reference mesh and moves it to new parameter values,
``x(p) = x_ref + Σ_i (p_i - p_ref,i) V_i``: the topology, cell tags and DoF numbering stay
those of the reference, the objective is smooth in ``p`` and consistent with the discrete
shape derivatives of ADR-0011. A move that inverts a cell or degrades a cell's quality below
``quality_threshold`` (0.3) of its reference value raises :class:`MeshQualityError`; the study
then remeshes (ADR-0012 §3). SI units throughout.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

import hpfem
from hpfem.meshing import Shape, UnitCell


class MeshQualityError(ValueError):
    """A morphed mesh has inverted cells or cells below the quality threshold."""

    def __init__(self, message: str, ratio: float, inverted: int):
        super().__init__(message)
        self.ratio = ratio
        """smallest quality relative to the reference cell"""
        self.inverted = inverted
        """number of inverted (or degenerate) cells"""


@dataclass
class MaterialParameter:
    """The real (``part="re"``) or imaginary (``"im"``) part of the relative permittivity of
    the cells with ``tag``."""

    name: str
    tag: int
    part: str = "re"
    lower: float = -math.inf
    upper: float = math.inf
    scale: float | None = None
    """typical magnitude for normalisation; default: ``upper - lower`` if finite, else 1"""

    def __post_init__(self):
        if self.part not in ("re", "im"):
            raise ValueError(f"MaterialParameter {self.name}: part={self.part!r}, use 're' or 'im'")

    def get(self, materials: Mapping[int, object]) -> float:
        eps = complex(materials[self.tag].eps_r)
        return eps.real if self.part == "re" else eps.imag

    def apply(self, materials: Mapping[int, object], value: float) -> dict:
        """A copy of the material map with the parameter set to ``value``."""
        out = dict(materials)
        old = materials[self.tag]
        eps = complex(old.eps_r)
        eps = complex(value, eps.imag) if self.part == "re" else complex(eps.real, value)
        out[self.tag] = hpfem.Material(eps_r=eps, mu_r=complex(old.mu_r))
        return out

    @property
    def typical(self) -> float:
        return _typical(self.scale, self.lower, self.upper)


@dataclass
class GeometryParameter:
    """A scalar of shape ``shape`` (index in ``UnitCell.shapes``): ``getter(params)`` reads it
    from the shape's ``params``, ``setter(params, value)`` returns new ``params``. Use the
    constructors :meth:`field` and :func:`trapezoid_parameters`."""

    name: str
    shape: int
    getter: Callable[[dict], float]
    setter: Callable[[dict, float], dict]
    lower: float = -math.inf
    upper: float = math.inf
    scale: float | None = None
    """typical magnitude for normalisation and the velocity step; default: ``upper - lower``
    if finite, else the cell period"""

    @classmethod
    def field(cls, name: str, shape: int, key: str, lower=-math.inf, upper=math.inf, scale=None):
        """The entry ``key`` of the shape's ``params`` (e.g. ``"height"``, ``"x"``, ``"rx"``)."""

        def setter(params, value):
            out = dict(params)
            out[key] = float(value)
            return out

        return cls(name, shape, lambda p: float(p[key]), setter, lower, upper, scale)

    def get(self, cell: UnitCell) -> float:
        return float(self.getter(self._shape(cell).params))

    def apply(self, cell: UnitCell, value: float) -> UnitCell:
        """A copy of the cell with the parameter set to ``value``."""
        out = copy.deepcopy(cell)
        shape = self._shape(out)
        shape.params = self.setter(dict(shape.params), float(value))
        return out

    def typical(self, cell: UnitCell) -> float:
        return _typical(self.scale, self.lower, self.upper, cell.period)

    def _shape(self, cell: UnitCell) -> Shape:
        if not 0 <= self.shape < len(cell.shapes):
            raise ValueError(f"GeometryParameter {self.name}: no shape {self.shape} in the cell")
        return cell.shapes[self.shape]


def _typical(scale, lower, upper, default=1.0) -> float:
    if scale is not None:
        return float(scale)
    if math.isfinite(lower) and math.isfinite(upper) and upper > lower:
        return float(upper - lower)
    return float(default)


def _trapezoid_to_cd(params: Mapping) -> tuple[float, float, float]:
    b, t, h = float(params["bottom"]), float(params["top"]), float(params["height"])
    return 0.5 * (b + t), h, math.atan2(h, 0.5 * (b - t))


def _trapezoid_from_cd(params: Mapping, cd: float, h: float, angle: float) -> dict:
    out = dict(params)
    half = h * math.cos(angle) / math.sin(angle)  # h / tan(angle), finite at 90 degrees
    out["bottom"], out["top"], out["height"] = cd + half, cd - half, h
    return out


_UNBOUNDED = (-math.inf, math.inf)


def trapezoid_parameters(
    shape: int, cd=_UNBOUNDED, height=_UNBOUNDED, angle=_UNBOUNDED, prefix: str = ""
) -> list[GeometryParameter]:
    """The scatterometry triple of a ``"trapezoid"`` shape: the mid-height width ``cd``
    (critical dimension), the ``height`` and the side-wall ``angle`` (radians, 90° for
    vertical walls), each changed with the other two fixed (centre ``x`` and base ``y``
    fixed). Bounds are ``(lower, upper)`` pairs; names ``prefix + "cd"`` etc."""

    def make(name, index, bounds):
        def getter(params):
            return _trapezoid_to_cd(params)[index]

        def setter(params, value):
            values = list(_trapezoid_to_cd(params))
            values[index] = float(value)
            return _trapezoid_from_cd(params, *values)

        return GeometryParameter(prefix + name, shape, getter, setter, *bounds)

    return [make("cd", 0, cd), make("height", 1, height), make("angle", 2, angle)]


# --- geometry nodes and mesh velocities ------------------------------------------------------


def _geometry_nodes(mesh) -> np.ndarray:
    """Coordinates of the geometry nodes: vertices, then the edge nodes of a second-order mesh."""
    nodes = np.asarray(mesh.vertices, dtype=float)
    if mesh.geometry_order == 2:
        nodes = np.vstack([nodes, np.asarray(mesh.edge_nodes, dtype=float)])
    return nodes


def _boundary_motion(shape: Shape, plus: Shape, minus: Shape, step: float, x: np.ndarray,
                     tol: float) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Velocity of the points ``x`` that lie on the boundary of ``shape`` and the mask of
    those points; ``plus`` / ``minus`` are the shape at the parameter ± ``step``."""
    v = np.zeros_like(x)
    on = np.zeros(len(x), dtype=bool)
    if shape.kind == "ellipse":
        p, pp, pm = shape.params, plus.params, minus.params
        u = (x[:, 0] - p["x"]) / p["rx"]
        w = (x[:, 1] - p["y"]) / p["ry"]
        on = np.abs(np.hypot(u, w) - 1.0) < tol / min(p["rx"], p["ry"])
        theta = np.arctan2(w, u)

        def point(q):
            return np.column_stack(
                [q["x"] + q["rx"] * np.cos(theta), q["y"] + q["ry"] * np.sin(theta)]
            )

        v[on] = ((point(pp) - point(pm)) / (2 * step))[on]
        return v, on
    corners = np.asarray(shape.polygon(), dtype=float)
    dc = (np.asarray(plus.polygon(), dtype=float) - np.asarray(minus.polygon(), dtype=float)) / (
        2 * step
    )
    n = len(corners)
    for i in range(n):
        c0, c1 = corners[i], corners[(i + 1) % n]
        d = c1 - c0
        length2 = float(d @ d)
        if length2 == 0.0:
            continue
        t = ((x - c0) @ d) / length2
        dist = np.linalg.norm(x - (c0 + np.outer(t, d)), axis=1)
        hit = (t >= -tol / math.sqrt(length2)) & (t <= 1 + tol / math.sqrt(length2)) & (dist < tol)
        tc = np.clip(t, 0.0, 1.0)[:, None]
        v[hit] = ((1 - tc) * dc[i] + tc * dc[(i + 1) % n])[hit]
        on |= hit
    return v, on


def shape_velocity(cell: UnitCell, mesh, parameter: GeometryParameter, step: float = 1e-6):
    """Mesh velocity ``V = dx/dp`` (array ``(hpfem.num_geometry_nodes(mesh), 2)``) of a
    geometry parameter on a mesh of ``cell``: the nodes on the boundary of the shape (and of
    its copies shifted by ± period) move with the boundary (central differences with
    ``step`` times the parameter's typical magnitude), the cell boundary and the other
    material interfaces stay fixed, the remaining vertices follow by the harmonic extension
    (graph Laplacian of the mesh edges), edge nodes off the shape by the mean of their
    vertices. Raises ``ValueError`` if a node on the cell boundary would move (the shape
    touches the Bloch faces or the top / bottom)."""
    shape = parameter._shape(cell)
    value = parameter.get(cell)
    h = step * parameter.typical(cell)
    plus = parameter._shape(parameter.apply(cell, value + h))
    minus = parameter._shape(parameter.apply(cell, value - h))
    x = _geometry_nodes(mesh)
    n_v = int(mesh.num_vertices)
    tol = 1e-9 * cell.period
    boundary = np.zeros(n_v, dtype=bool)
    for f in mesh.boundary_facets:
        boundary[list(mesh.facet_vertices(int(f)))] = True
    v = np.zeros_like(x)
    on = np.zeros(len(x), dtype=bool)
    for k in (-1, 0, 1):
        shift = np.array([k * cell.period, 0.0])
        vk, onk = _boundary_motion(shape, plus, minus, h, x - shift, tol)
        # checked per copy: a node on the cell boundary may lie on two copies at once
        if np.any(np.linalg.norm(vk[:n_v][onk[:n_v] & boundary], axis=1) > 0):
            raise ValueError(
                f"shape_velocity: parameter {parameter.name} moves nodes on the cell boundary "
                "(the shape touches the Bloch faces, the top or the bottom)"
            )
        v[onk] = vk[onk]
        on |= onk
    # fixed vertices: the cell boundary and every material interface
    tags = np.asarray(mesh.cell_tags)
    fixed = on[:n_v] | boundary
    for f in range(mesh.num_facets):
        cells = list(mesh.facet_cells(f))
        if len(cells) == 2 and min(cells) >= 0 and tags[cells[0]] != tags[cells[1]]:
            fixed[list(mesh.facet_vertices(f))] = True
    free = ~fixed
    if free.any():
        edges = np.asarray(mesh.edges, dtype=int)
        a, b = edges[:, 0], edges[:, 1]
        ones = np.ones(len(edges))
        adjacency = scipy.sparse.coo_matrix(
            (np.concatenate([ones, ones]), (np.concatenate([a, b]), np.concatenate([b, a]))),
            shape=(n_v, n_v),
        ).tocsr()
        laplacian = (
            scipy.sparse.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
        ).tocsr()
        idx_free, idx_fixed = np.flatnonzero(free), np.flatnonzero(fixed)
        l_ff = laplacian[idx_free][:, idx_free].tocsc()
        l_fb = laplacian[idx_free][:, idx_fixed]
        rhs = -(l_fb @ v[idx_fixed])
        v[idx_free] = np.column_stack(
            [scipy.sparse.linalg.spsolve(l_ff, rhs[:, d]) for d in range(2)]
        )
    if mesh.geometry_order == 2:
        edges = np.asarray(mesh.edges, dtype=int)
        mean = 0.5 * (v[edges[:, 0]] + v[edges[:, 1]])
        off = ~on[n_v:]
        v[n_v:][off] = mean[off]
    return v


def cell_quality(mesh) -> np.ndarray:
    r"""Signed shape quality of every cell (ADR-0012 §3): the ratio of the inscribed to the
    circumscribed radius of the vertex triangle, normalised by its equilateral value,
    :math:`2 r / R = 16 A^2 / (s\,abc)` with the perimeter s and the edge lengths a, b, c (1 for
    the equilateral triangle, 0 for a degenerate one), with the sign of the area (negative for
    a clockwise cell); :class:`Morph` compares it with the reference cell, so either
    orientation works."""
    x = np.asarray(mesh.vertices, dtype=float)
    cells = np.asarray(mesh.cells, dtype=int)
    p0, p1, p2 = x[cells[:, 0]], x[cells[:, 1]], x[cells[:, 2]]
    e1, e2, e3 = p1 - p0, p2 - p1, p0 - p2
    area = 0.5 * (e1[:, 0] * (-e3[:, 1]) - e1[:, 1] * (-e3[:, 0]))
    a_, b_, c_ = (np.linalg.norm(e, axis=1) for e in (e1, e2, e3))
    return 16.0 * area * np.abs(area) / ((a_ + b_ + c_) * a_ * b_ * c_)


class Morph:
    """A reference mesh of ``cell`` moved to new values of the geometry ``parameters``.

    ``mesh_at(values)`` returns a moved copy, ``x = x_ref + Σ (p - p_ref) V_p`` with the
    velocities of :func:`shape_velocity`, after checking it (:meth:`check`); parameters
    missing from ``values`` keep their reference values. ``cell_at(values)`` is the exact
    geometry (for a remesh). The velocities are those the derivatives need
    (``grating.jacobian(result, [("shape", morph.velocity(name)), ...])``)."""

    def __init__(self, cell: UnitCell, mesh, parameters: Sequence[GeometryParameter],
                 quality_threshold: float = 0.3, step: float = 1e-6):  # fmt: skip
        if mesh.dim != 2:
            raise ValueError("Morph: unit cells are two-dimensional")
        names = [p.name for p in parameters]
        if len(set(names)) != len(names):
            raise ValueError(f"Morph: duplicate parameter names {names}")
        self.cell = cell
        self.mesh = mesh
        self.parameters = list(parameters)
        self.quality_threshold = float(quality_threshold)
        self.reference = {p.name: p.get(cell) for p in self.parameters}
        self._velocity = {p.name: shape_velocity(cell, mesh, p, step) for p in self.parameters}
        self._quality = cell_quality(mesh)
        if np.any(self._quality == 0):
            raise ValueError("Morph: the reference mesh has degenerate cells")

    def velocity(self, name: str) -> np.ndarray:
        return self._velocity[name]

    def _deltas(self, values: Mapping[str, float]) -> dict[str, float]:
        unknown = set(values) - set(self.reference)
        if unknown:
            raise ValueError(f"Morph: unknown parameters {sorted(unknown)}")
        return {n: float(values.get(n, ref)) - ref for n, ref in self.reference.items()}

    def mesh_at(self, values: Mapping[str, float]):
        """The reference mesh moved to ``values``; raises :class:`MeshQualityError`."""
        moved = self.mesh.copy()
        for name, delta in self._deltas(values).items():
            if delta != 0.0:
                hpfem.move_nodes(moved, self._velocity[name], delta)
        self.check(moved)
        return moved

    def cell_at(self, values: Mapping[str, float]) -> UnitCell:
        """The unit cell with the parameters set to ``values`` (the exact geometry)."""
        cell = self.cell
        deltas = self._deltas(values)
        for p in self.parameters:
            cell = p.apply(cell, self.reference[p.name] + deltas[p.name])
        return cell

    def check(self, mesh) -> float:
        """Smallest cell quality of ``mesh`` relative to the reference; raises
        :class:`MeshQualityError` for inverted cells or a ratio below the threshold."""
        ratio = cell_quality(mesh) / self._quality
        inverted = int(np.count_nonzero(ratio <= 0))
        worst = float(ratio.min())
        if inverted or worst < self.quality_threshold:
            raise MeshQualityError(
                f"Morph: {inverted} inverted cells, smallest quality ratio {worst:.3g} "
                f"(threshold {self.quality_threshold:g}); remesh at these values",
                worst,
                inverted,
            )
        return worst

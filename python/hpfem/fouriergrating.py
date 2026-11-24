"""Import of FourierGrating (RCWA) models of crossed gratings (M21 S3, issue #237).

:func:`load` reads a FourierGrating configuration (``"dimension": "2D"``: periodic in x and z,
layered in y) into a :class:`hpfem.crossed.CrossedGrating` and the list of incidences of its
scan; :func:`solve` computes every incidence with :func:`hpfem.crossed.solve` and
:func:`write_efficiencies` writes them as FourierGrating's ``efficiencies.csv``, so both programs
can be compared order by order (``hpfem fg model.json``).

Frames (verified against FourierGrating's fields and efficiencies, docs/validation.md N):

* FourierGrating has the stack normal along y (substrate y < 0, layers listed from the bottom),
  hpfem along z (layers from z = 0 downwards). A point (x, y, z) of FourierGrating is
  (x, -z, y - H) in hpfem, H the total thickness of the layers (a proper rotation about x).
* Incidence: ``phi_deg`` is the polar angle (theta here), ``psi_deg`` the azimuth from x towards
  z, i.e. ``phi = -psi`` here; ``delta_deg`` mixes s and p as E = cos δ s + sin δ p, where
  FourierGrating's s is hpfem's s and its p is minus hpfem's p, so (a_s, a_p) = (cos δ, -sin δ).
* Orders: FourierGrating's (n, m) along (x, z) is the order (m, n) = (n, -m) here.

Supported: layers ``homogeneous``, ``contour2d`` (circle, ellipse, polygon_um; no holes, no
NURBS) and ``polygon2d`` (vertices_um, rectangle_um); materials ``{"n": x | [n, k]}`` and
``{"eps": x | [re, im]}``; incidence values as numbers, lists or ``{"from", "to", "steps"}``.
Other features (sweeps, beams, dispersive models, tensors, profiles, CAD) raise
:class:`FourierGratingError` with the JSON pointer.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from hpfem import crossed as cg

UM = 1e-6


class FourierGratingError(ValueError):
    """A FourierGrating configuration this importer cannot represent."""


@dataclass
class Incidence:
    """One point of the scan: hpfem's ``wavelength`` [m], ``theta``, ``phi`` [rad] and
    ``polarisation`` (a_s, a_p), with FourierGrating's values in ``fg``."""

    wavelength: float
    theta: float
    phi: float
    polarisation: tuple[float, float]
    fg: dict = field(default_factory=dict)


@dataclass
class Model:
    """An imported model: the grating, the incidences of the scan and the source dict."""

    grating: cg.CrossedGrating
    incidences: list[Incidence]
    source: dict
    thickness: float
    """total thickness of the layers [m] (FourierGrating's y = H is hpfem's z = 0)"""

    def fg_to_hpfem(self, points_um) -> np.ndarray:
        """FourierGrating points (n, 3) [µm] -> hpfem points [m]."""
        p = np.atleast_2d(np.asarray(points_um, dtype=float)) * UM
        return np.column_stack([p[:, 0], -p[:, 2], p[:, 1] - self.thickness])

    @staticmethod
    def hpfem_to_fg_field(e) -> np.ndarray:
        """hpfem field rows (E_x, E_y, E_z) -> FourierGrating's components (E_x, E_y, E_z)."""
        e = np.atleast_2d(np.asarray(e))
        return np.column_stack([e[:, 0], e[:, 2], -e[:, 1]])


def _values(spec, where: str) -> list[float]:
    if isinstance(spec, (int, float)):
        return [float(spec)]
    if isinstance(spec, Sequence) and not isinstance(spec, str):
        return [float(v) for v in spec]
    if isinstance(spec, Mapping) and {"from", "to", "steps"} <= set(spec):
        steps = int(spec["steps"])
        if steps < 1:
            raise FourierGratingError(f"{where}: steps must be >= 1")
        return [float(v) for v in np.linspace(float(spec["from"]), float(spec["to"]), steps)]
    raise FourierGratingError(f"{where}: a number, a list or {{'from', 'to', 'steps'}}")


def _complex(value, where: str) -> complex:
    if isinstance(value, (int, float)):
        return complex(value)
    if isinstance(value, Sequence) and not isinstance(value, str) and len(value) == 2:
        if all(isinstance(v, (int, float)) for v in value):
            return complex(float(value[0]), float(value[1]))
    raise FourierGratingError(
        f"{where}: only constant values (x or [re, im]) are supported, not models"
    )


def _material(materials: Mapping, name, where: str):
    import hpfem

    if not isinstance(name, str) or name not in materials:
        raise FourierGratingError(f"{where}: unknown material {name!r}")
    spec = materials[name]
    keys = set(spec) if isinstance(spec, Mapping) else set()
    if keys == {"n"}:
        n = _complex(spec["n"], f"/materials/{name}/n")
        return hpfem.Material(n * n, 1.0)
    if keys == {"eps"}:
        return hpfem.Material(_complex(spec["eps"], f"/materials/{name}/eps"), 1.0)
    raise FourierGratingError(
        f"/materials/{name}: only {{'n': ...}} or {{'eps': ...}} are supported (got {sorted(keys)})"
    )


def _wrap(value: float, period: float) -> float:
    """The shift k * period that moves ``value`` into [0, period)."""
    return -math.floor(value / period) * period


def _shape(region: Mapping, materials: Mapping, pz: float, where: str) -> cg.Shape:
    """A contour2d region or polygon2d inclusion in hpfem's frame (x, -z)."""
    if "holes" in region and region["holes"]:
        raise FourierGratingError(f"{where}: regions with holes are not supported")
    material = _material(materials, region.get("material"), f"{where}/material")
    kinds = [
        k for k in ("circle", "ellipse", "polygon_um", "vertices_um", "rectangle_um") if k in region
    ]
    if len(kinds) != 1:
        other = sorted(set(region) - {"material", "holes"})
        raise FourierGratingError(
            f"{where}: one of circle, ellipse, polygon_um, vertices_um, rectangle_um expected "
            f"(got {other}; NURBS is not supported)"
        )
    kind = kinds[0]
    geo = region[kind]
    if kind in ("circle", "ellipse", "rectangle_um"):
        cx, cz = (float(v) for v in (geo["center_um"] if kind != "rectangle_um" else geo["center"]))
        y = -cz * UM
        y += _wrap(y, pz)
        center = (cx * UM, y)
        if kind == "circle":
            return cg.Shape("circle", material, center=center, radius=float(geo["radius_um"]) * UM)
        if kind == "ellipse":
            rx, rz = (float(v) * UM for v in geo["radii_um"])
            angle = -math.radians(float(geo.get("angle_deg", 0.0)))
            return cg.Shape("ellipse", material, center=center, rx=rx, ry=rz, angle=angle)
        sx, sz = (float(v) * UM for v in geo["size"])
        return cg.Shape("rectangle", material, center=center, width=sx, height=sz)
    points = np.asarray(geo, dtype=float) * UM
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise FourierGratingError(f"{where}/{kind}: a list of at least three [x, z] points")
    xy = np.column_stack([points[:, 0], -points[:, 1]])
    xy[:, 1] += _wrap(float(xy[:, 1].mean()), pz)
    if _signed_area(xy) < 0:
        xy = xy[::-1]  # counter-clockwise in hpfem's frame (the map flips the orientation)
    return cg.Shape("polygon", material, points=[tuple(p) for p in xy])


def _signed_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _layer(spec: Mapping, materials: Mapping, pz: float, where: str) -> cg.Layer:
    kind = spec.get("type")
    if "thickness_um" not in spec:
        raise FourierGratingError(f"{where}: thickness_um is missing (type {kind!r})")
    thickness = float(spec["thickness_um"]) * UM
    if kind == "homogeneous":
        return cg.Layer(thickness, _material(materials, spec.get("material"), f"{where}/material"))
    if kind in ("contour2d", "polygon2d"):
        key = "regions" if kind == "contour2d" else "inclusions"
        background = _material(materials, spec.get("background"), f"{where}/background")
        shapes = [
            _shape(region, materials, pz, f"{where}/{key}/{i}")
            for i, region in enumerate(spec.get(key, []))
        ]
        return cg.Layer(thickness, background, shapes)
    raise FourierGratingError(
        f"{where}: layer type {kind!r} is not supported (homogeneous, contour2d, polygon2d)"
    )


def load(spec: str | Path | Mapping) -> Model:
    """Reads a FourierGrating configuration (a path or the parsed dict)."""
    if not isinstance(spec, Mapping):
        spec = json.loads(Path(spec).read_text(encoding="utf-8"))
    spec = dict(spec)
    if spec.get("dimension") != "2D":
        raise FourierGratingError(
            "/dimension: only crossed gratings ('2D') are imported; 1D gratings go to hpfem.grating"
        )
    for key in ("sweep", "aperiodic", "cylindrical", "substrate_structure", "beam"):
        if key in spec:
            raise FourierGratingError(f"/{key}: not supported by the import")
    period = spec.get("period_um", {})
    px, pz = float(period.get("x", 0)) * UM, float(period.get("z", 0)) * UM
    if not (px > 0 and pz > 0):
        raise FourierGratingError("/period_um: x and z must be positive")
    materials = spec.get("materials", {})
    superstrate = _material(materials, spec.get("superstrate"), "/superstrate")
    substrate = _material(materials, spec.get("substrate"), "/substrate")
    fg_layers = spec.get("layers", [])
    layers = [
        _layer(fg_layers[i], materials, pz, f"/layers/{i}") for i in reversed(range(len(fg_layers)))
    ]
    grating = cg.CrossedGrating((px, pz), superstrate, substrate, layers)

    inc = spec.get("incidence", {})
    if inc.get("type", "plane_wave") != "plane_wave":
        raise FourierGratingError(f"/incidence/type: {inc.get('type')!r} is not supported")
    scan = [
        _values(inc.get(key, default), f"/incidence/{key}")
        for key, default in (
            ("lambda_um", None),
            ("phi_deg", 0.0),
            ("psi_deg", 0.0),
            ("delta_deg", 0.0),
        )
    ]
    incidences = []
    for lam, phi_deg, psi_deg, delta_deg in itertools.product(*scan):
        delta = math.radians(delta_deg)
        incidences.append(
            Incidence(
                wavelength=lam * UM,
                theta=math.radians(phi_deg),
                phi=-math.radians(psi_deg),
                polarisation=(math.cos(delta), -math.sin(delta)),
                fg={
                    "lambda_um": lam,
                    "phi_deg": phi_deg,
                    "psi_deg": psi_deg,
                    "delta_deg": delta_deg,
                },
            )
        )
    return Model(grating, incidences, spec, grating.depth)


def solve(
    model: Model,
    *,
    order: int = 3,
    elements_per_wavelength: float | None = None,
    progress: Callable[[int, int, Incidence], None] | None = None,
    **options,
) -> list[tuple[Incidence, cg.CrossedResult]]:
    """Every incidence of the scan with :func:`hpfem.crossed.solve`: one solve per (wavelength,
    angles), the polarisations of the scan from it without a new solve, and the mesh reused
    while the wavelength stays the same. ``options`` go to :func:`hpfem.crossed.solve`."""
    out = []
    cache: dict = {}
    mesh = None
    mesh_wavelength = None
    for i, incidence in enumerate(model.incidences):
        if progress is not None:
            progress(i, len(model.incidences), incidence)
        key = (incidence.wavelength, incidence.theta, incidence.phi)
        if key not in cache:
            reuse = mesh if mesh_wavelength == incidence.wavelength else None
            cache[key] = cg.solve(
                model.grating,
                incidence.wavelength,
                incidence.theta,
                incidence.phi,
                incidence.polarisation,
                order=order,
                elements_per_wavelength=elements_per_wavelength,
                mesh=reuse,
                **options,
            )
            mesh, mesh_wavelength = cache[key]._raw["mesh"], incidence.wavelength
        out.append((incidence, cache[key].polarised(incidence.polarisation)))
    return out


def efficiency_rows(results: Sequence[tuple[Incidence, cg.CrossedResult]]) -> list[dict]:
    """Rows of FourierGrating's ``efficiencies.csv`` (propagating orders): ``lambda_um``,
    ``phi_deg``, ``psi_deg``, ``delta_deg``, ``side``, ``n``, ``m``, ``efficiency``."""
    rows = []
    for incidence, result in results:
        for side, orders in (("R", result.reflected), ("T", result.transmitted)):
            for o in orders:
                if not o.propagating:
                    continue
                rows.append(
                    {**incidence.fg, "side": side, "n": o.m, "m": -o.n, "efficiency": o.efficiency}
                )
    return rows


def write_efficiencies(results, path: str | Path) -> Path:
    """Writes :func:`efficiency_rows` as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = efficiency_rows(results)
    columns = ["lambda_um", "phi_deg", "psi_deg", "delta_deg", "side", "n", "m", "efficiency"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: (f"{v:.12g}" if isinstance(v, float) else v) for k, v in row.items()}
            )
    return path


def read_efficiencies(path: str | Path, threshold: float = 0.0) -> dict:
    """FourierGrating's (or :func:`write_efficiencies`') CSV as {(lambda_um, phi_deg, psi_deg,
    delta_deg, side, n, m): efficiency} for efficiencies above ``threshold``."""
    out = {}
    with Path(path).open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = float(row["efficiency"])
            if value <= threshold:
                continue
            key = (
                float(row["lambda_um"]),
                float(row["phi_deg"]),
                float(row["psi_deg"]),
                float(row["delta_deg"]),
                row["side"],
                int(row["n"]),
                int(row["m"]),
            )
            out[key] = value
    return out

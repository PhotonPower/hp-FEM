"""Structured diagnostics (M15 F7): checks that catch the mistakes which otherwise surface
as late exceptions or as wrong numbers, each as a :class:`Diagnostic` with a stable ``code``
(for a GUI to translate), a ``severity`` (``"error"``, ``"warning"``, ``"info"``), a ``text``
and a ``hint``. :func:`validate_scattering` checks a mesh with a ``ConicalScatteringSetup``
or ``ScatteringSetup2D``; :func:`hpfem.grating.validate` and ``grating.solve`` run the
grating-specific set (bottom wall, grazing orders, measurement lines) on top of it.

Codes: ``mesh_invalid_cells``, ``mesh_poor_angles``, ``mesh_untagged_cells``,
``tag_without_material``, ``material_without_cells``, ``interface_off_mesh``,
``periodic_partner_missing``, ``periodic_faces_differ``, ``pml_under_resolved``, ``pml_thin``,
``pml_missing``, ``too_few_elements_per_wavelength``, ``material_out_of_range``,
``lossy_incidence_medium``, ``grazing_order``, ``pec_wall_too_close``, ``setup``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

import hpfem
from hpfem import units


@dataclass
class Diagnostic:
    code: str
    severity: str
    text: str
    hint: str = ""

    def __str__(self) -> str:
        return f"[{self.severity}] {self.code}: {self.text}" + (
            f" ({self.hint})" if self.hint else ""
        )


class ValidationError(ValueError):
    """Raised by :func:`raise_on_errors` with every error-level diagnostic in its message."""

    def __init__(self, diagnostics: Sequence[Diagnostic]):
        self.diagnostics = list(diagnostics)
        super().__init__("; ".join(str(d) for d in self.diagnostics))


def errors(diagnostics: Sequence[Diagnostic]) -> list[Diagnostic]:
    return [d for d in diagnostics if d.severity == "error"]


def raise_on_errors(diagnostics: Sequence[Diagnostic]) -> None:
    bad = errors(diagnostics)
    if bad:
        raise ValidationError(bad)


# --- helpers ---------------------------------------------------------------------------------


def _cell_vertices(mesh, c):
    return np.array([mesh.vertex(int(v)) for v in mesh.cell_vertices(c)], dtype=float)


def _cell_size(vertices) -> float:
    """Longest edge of a simplex given by its vertex array."""
    n = len(vertices)
    return max(
        float(np.linalg.norm(vertices[i] - vertices[j])) for i in range(n) for j in range(i + 1, n)
    )


def _material_of(material_map, mesh, c):
    return material_map.of_cell(mesh, c)


def _index(material) -> complex:
    return complex(material.refractive_index)


# --- individual checks -----------------------------------------------------------------------


def validate_mesh(mesh, materials=None, min_angle: float = 5 * units.deg) -> list[Diagnostic]:
    """Invalid curved cells (error), angles below ``min_angle`` (warning), untagged cells
    (info) and, with ``materials`` (a ``MaterialMap``, a dict or a list of tags), tags without
    a material and materials without cells (warning / info)."""
    out = []
    r = hpfem.mesh_report(mesh)
    if r["num_invalid"] > 0:
        out.append(
            Diagnostic(
                "mesh_invalid_cells",
                "error",
                f"{r['num_invalid']} curved cell(s) with an inverted Jacobian, e.g. cell "
                f"{int(r['invalid_cells'][0])}",
                "re-mesh with smaller elements at the curved boundary or without second-order "
                "elements",
            )
        )
    if r["min_angle"] < min_angle:
        out.append(
            Diagnostic(
                "mesh_poor_angles",
                "warning",
                f"smallest angle {np.degrees(r['min_angle']):.1f} deg (aspect ratio up to "
                f"{r['max_aspect_ratio']:.1f})",
                "sliver cells hurt the conditioning; check the mesher's size fields",
            )
        )
    if r["num_untagged"] > 0:
        out.append(
            Diagnostic(
                "mesh_untagged_cells",
                "info",
                f"{r['num_untagged']} of {r['num_cells']} cells carry no tag and take the "
                "background material",
                "tag every region if the background is not what they should be",
            )
        )
    if materials is not None:
        if isinstance(materials, hpfem.MaterialMap):
            known = {int(t) for t in r["cell_tags"] if materials.has(int(t))}
        elif isinstance(materials, Mapping):
            known = {int(t) for t in materials}
        else:
            known = {int(t) for t in materials}
        present = {int(t) for t in r["cell_tags"]}
        missing = sorted(present - known)
        if missing:
            out.append(
                Diagnostic(
                    "tag_without_material",
                    "warning",
                    f"cell tag(s) {missing} have no material and take the background",
                    "add the tags to the material map",
                )
            )
        unused = sorted(known - present)
        if unused and not isinstance(materials, hpfem.MaterialMap):
            out.append(
                Diagnostic(
                    "material_without_cells",
                    "info",
                    f"material tag(s) {unused} occur in no cell",
                    "",
                )
            )
    return out


def validate_interfaces(mesh, stack, tolerance: float) -> list[Diagnostic]:
    """Cells straddling an interface of the layered background (error, the first three)."""
    out = []
    interfaces = [stack.interface(i) for i in range(stack.num_layers + 1)]
    for c in range(mesh.num_cells):
        ys = _cell_vertices(mesh, c)[:, 1]
        lo, hi = float(ys.min()), float(ys.max())
        for y in interfaces:
            if lo < y - tolerance and hi > y + tolerance:
                out.append(
                    Diagnostic(
                        "interface_off_mesh",
                        "error",
                        f"cell {c} (y from {lo:.6g} to {hi:.6g} m) straddles the stack interface "
                        f"at y = {y:.6g} m",
                        "put the interfaces on mesh lines (hpfem.grating.solve snaps vertices "
                        "within snap_tolerance x period)",
                    )
                )
                break
        if len(out) >= 3:
            break
    return out


def validate_periodic(mesh, pairs, tolerance: float = 1e-8) -> list[Diagnostic]:
    """For every ``(master_tag, slave_tag, shift)`` (or ``PeriodicPair2D``): a slave facet
    whose shifted image lies outside the master face is an error; faces that are not meshed
    identically are an info (the non-matching coupling handles them)."""
    out = []
    for pair in pairs:
        if hasattr(pair, "master"):
            master, slave, shift = (
                int(pair.master),
                int(pair.slave),
                np.asarray(pair.shift, dtype=float),
            )
        else:
            master, slave, shift = int(pair[0]), int(pair[1]), np.asarray(pair[2], dtype=float)
        masters = list(mesh.facets_with_tag(master))
        slaves = list(mesh.facets_with_tag(slave))
        if not masters or not slaves:
            out.append(
                Diagnostic(
                    "periodic_partner_missing",
                    "error",
                    f"periodic pair ({master}, {slave}): {len(masters)} master and "
                    f"{len(slaves)} slave facets",
                    "tag both faces",
                )
            )
            continue
        direction = shift / np.linalg.norm(shift)
        tangent = np.array([-direction[1], direction[0]]) if len(shift) == 2 else None
        if tangent is None:
            continue  # 3D: only the counts are checked here
        master_points = [
            np.asarray(mesh.vertex(int(v)), dtype=float)
            for f in masters
            for v in mesh.facet_vertices(f)
        ]
        master_t = [float(np.dot(x, tangent)) for x in master_points]
        master_d = float(np.mean([np.dot(x, direction) for x in master_points]))
        lo, hi = min(master_t), max(master_t)
        eps = tolerance * (hi - lo + 1e-300) + 1e-12 * np.linalg.norm(shift)
        for f in slaves:
            for v in mesh.facet_vertices(f):
                x = np.asarray(mesh.vertex(int(v)), dtype=float) - shift
                t = float(np.dot(x, tangent))
                off_line = abs(float(np.dot(x, direction)) - master_d) > eps
                if off_line or t < lo - eps or t > hi + eps:
                    out.append(
                        Diagnostic(
                            "periodic_partner_missing",
                            "error",
                            f"slave facet {f} (tag {slave}) does not land on the master face "
                            f"(tag {master}) after the shift",
                            "the two faces must cover the same interval",
                        )
                    )
                    break
            else:
                continue
            break
        check = hpfem.check_periodic(mesh, master, slave, shift, tolerance)
        if not check["identical"]:
            out.append(
                Diagnostic(
                    "periodic_faces_differ",
                    "info",
                    f"faces {master} / {slave} are not meshed identically "
                    f"({check['num_master']} vs {check['num_slave']} facets, "
                    f"{check['matched']} matched)",
                    "fine: the non-matching Bloch coupling interpolates the trace; identical "
                    "faces give the classic one-to-one constraints",
                )
            )
    return out


def _pml_cells(mesh, box):
    lower, upper = np.asarray(box.lower, dtype=float), np.asarray(box.upper, dtype=float)
    cells = []
    for c in range(mesh.num_cells):
        v = _cell_vertices(mesh, c)
        centroid = v.mean(axis=0)
        if np.any(centroid < lower) or np.any(centroid > upper):
            cells.append((c, _cell_size(v)))
    return cells


def validate_pml(mesh, box, p: int, index: float = 1.0) -> list[Diagnostic]:
    """PML layers under-resolved for order ``p`` (``max_resolution`` above
    ``resolution_limit(p)``) or thinner than ``recommended_thickness`` (warnings); no PML
    cells in the mesh (error)."""
    out = []
    cells = _pml_cells(mesh, box)
    if not cells:
        out.append(
            Diagnostic(
                "pml_missing",
                "error",
                "no cell lies in the PML layers",
                "the layers must lie inside the mesh (the box is the interior)",
            )
        )
        return out
    h = float(np.median([size for _, size in cells]))
    limit = hpfem.PmlBox2D.resolution_limit(int(p))
    resolution = box.max_resolution(h, float(index))
    if resolution > limit:
        out.append(
            Diagnostic(
                "pml_under_resolved",
                "warning",
                f"|k s| h = {resolution:.2f} in the PML (median cell {h:.3g} m) exceeds "
                f"{limit:.2f} for p = {p}",
                "thicker layers, smaller cells or a gentler profile (PmlProfile.for_angle with "
                "a larger target)",
            )
        )
    thicknesses = [float(t) for t in box.thickness]
    profile = getattr(box, "profile", None)
    try:
        recommended = (
            hpfem.PmlBox2D.recommended_thickness(box.k0, float(index), h, profile, int(p))
            if profile is not None
            else hpfem.PmlBox2D.recommended_thickness(box.k0, float(index), h, 0.5)
        )
    except Exception:  # an unresolvable layer: reported above
        recommended = 0.0
    thin = [t for t in thicknesses if 0 < t < recommended * (1 - 1e-9)]
    if thin:
        out.append(
            Diagnostic(
                "pml_thin",
                "warning",
                f"PML thickness {min(thin):.3g} m below the recommended {recommended:.3g} m "
                f"for p = {p}",
                "half a local wavelength rounded up to whole cells, more for steep profiles",
            )
        )
    return out


def validate_resolution(
    mesh, material_map, omega: float, orders, minimum: float = 6.0
) -> list[Diagnostic]:
    """Elements per local wavelength times the order, per material tag, below ``minimum``
    (warning): the discretisation cannot represent the field."""
    out = []
    k0 = float(units.vacuum_wavenumber(omega))
    wavelength = 2 * np.pi / k0
    sizes: dict[int, list[float]] = {}
    orders_list = (
        [int(orders)] * mesh.num_cells if np.isscalar(orders) else [int(q) for q in orders]
    )
    p_by_tag: dict[int, list[int]] = {}
    for c in range(mesh.num_cells):
        tag = int(mesh.cell_tag(c))
        sizes.setdefault(tag, []).append(_cell_size(_cell_vertices(mesh, c)))
        p_by_tag.setdefault(tag, []).append(orders_list[c])
    for tag, hs in sizes.items():
        material = material_map.of_tag(tag) if hasattr(material_map, "of_tag") else None
        if material is None:
            # resolve through any cell of the tag
            c = next(i for i in range(mesh.num_cells) if int(mesh.cell_tag(i)) == tag)
            material = material_map.of_cell(mesh, c)
        n = max(_index(material).real, 1.0)
        h = float(np.median(hs))
        p = int(np.median(p_by_tag[tag]))
        per_wavelength = wavelength / (n * h)
        if per_wavelength * p < minimum:
            out.append(
                Diagnostic(
                    "too_few_elements_per_wavelength",
                    "warning",
                    f"tag {tag}: {per_wavelength:.1f} elements per local wavelength at p = {p}",
                    f"aim for {12 / max(p, 1):.0f} elements per wavelength at p = {p} "
                    "(hpfem.meshing.element_sizes)",
                )
            )
    return out


def validate_materials(materials, omega: float) -> list[Diagnostic]:
    """Dispersive materials outside their tabulated range (error)."""
    out = []
    if not isinstance(materials, Mapping):
        return out
    lam = float(units.wavelength(omega))
    for tag, material in materials.items():
        rng = getattr(material, "range", None)
        if rng is None:
            continue
        lo, hi = rng
        if lam < lo * (1 - 1e-12) or lam > hi * (1 + 1e-12):
            out.append(
                Diagnostic(
                    "material_out_of_range",
                    "error",
                    f"tag {tag}: {getattr(material, 'name', 'material')} is tabulated for "
                    f"{lo / units.nm:.0f}-{hi / units.nm:.0f} nm, requested "
                    f"{lam / units.nm:.1f} nm",
                    "choose another data set or fit a Drude-Lorentz model",
                )
            )
    return out


def validate_stack(stack) -> list[Diagnostic]:
    """A lossy incidence medium (error); ``stack`` is a ``LayerStack2D`` or the incidence
    ``Material`` itself (the stack constructor already refuses a lossy medium)."""
    n = _index(getattr(stack, "incidence_medium", stack))
    if abs(n.imag) > 1e-12:
        return [
            Diagnostic(
                "lossy_incidence_medium",
                "error",
                f"the incidence medium has Im n = {n.imag:.3g}",
                "the incident plane wave must be defined in a lossless medium",
            )
        ]
    return []


def grazing_orders(
    k0: float,
    kx: float,
    index: float,
    period: float,
    orders_max: int,
    limit: float = 80 * units.deg,
):
    """Orders propagating at more than ``limit`` from the normal in a medium of index n."""
    grazing = []
    for m in range(-orders_max, orders_max + 1):
        kt = kx + 2 * np.pi * m / period
        s = abs(kt) / (k0 * index)
        if s < 1.0 and np.arcsin(s) > limit:
            grazing.append((m, float(np.degrees(np.arcsin(s)))))
    return grazing


def validate_orders(
    k0: float, kx: float, n_cover: float, n_sub: float | None, period: float, orders_max: int
) -> list[Diagnostic]:
    out = []
    for medium, n in (("cover", n_cover), ("substrate", n_sub)):
        if n is None:
            continue
        for m, angle in grazing_orders(k0, kx, n, period, orders_max):
            out.append(
                Diagnostic(
                    "grazing_order",
                    "warning",
                    f"order {m} propagates at {angle:.1f} deg in the {medium}: the PML absorbs "
                    "it poorly",
                    "a thicker PML or a profile designed for that angle (PmlProfile.for_angle); "
                    "the efficiency of a grazing order converges slowly",
                )
            )
    return out


def validate_bottom_wall(
    distance: float, n_sub: complex, k0: float, decay_lengths: float = 6.0
) -> list[Diagnostic]:
    """A PEC wall closer than ``decay_lengths`` field decay lengths to the structure in a lossy
    substrate (warning with the amplitude fraction reaching the wall)."""
    kappa = _index(n_sub).imag if hasattr(n_sub, "refractive_index") else complex(n_sub).imag
    if kappa <= 1e-12:
        return []
    delta = 1.0 / (k0 * kappa)
    if distance < decay_lengths * delta:
        return [
            Diagnostic(
                "pec_wall_too_close",
                "warning",
                f"the PEC wall lies {distance / delta:.1f} decay lengths below the structure: "
                f"{np.exp(-distance / delta) * 100:.0f} % of the surface amplitude reaches it",
                f"extend the substrate to at least {decay_lengths * delta:.3g} m or use a PML "
                "below",
            )
        ]
    return []


# --- combined ------------------------------------------------------------------------------


def validate_scattering(mesh, setup, orders, materials=None) -> list[Diagnostic]:
    """Checks a mesh against a ``ConicalScatteringSetup`` or ``ScatteringSetup2D``: the mesh
    (:func:`validate_mesh`), the stack interfaces, the periodic pairs, the PML and the
    resolution per tag; ``materials`` (dict) adds the range check of dispersive data and the
    tags-without-material check."""
    out = validate_mesh(mesh, materials if materials is not None else setup.materials)
    background = getattr(setup, "background", None)
    k0 = float(units.vacuum_wavenumber(setup.omega))
    if background is not None:
        out += validate_stack(background)
        vertices = np.array([mesh.vertex(v) for v in range(mesh.num_vertices)], dtype=float)
        extent = float(vertices[:, 1].max() - vertices[:, 1].min())
        out += validate_interfaces(mesh, background, 1e-9 * extent)
    if getattr(setup, "periodic", None):
        out += validate_periodic(mesh, setup.periodic)
    p = int(orders) if np.isscalar(orders) else int(np.median(orders))
    if getattr(setup, "pml", None) is not None:
        index = 1.0
        if background is not None:
            index = min(
                _index(background.incidence_medium).real,
                max(_index(background.substrate).real, 1.0),
            )
        out += validate_pml(mesh, setup.pml, p, index)
    out += validate_resolution(mesh, setup.materials, setup.omega, orders)
    if materials is not None:
        out += validate_materials(materials, setup.omega)
    del k0
    return out


__all__ = [
    "Diagnostic",
    "ValidationError",
    "errors",
    "grazing_orders",
    "raise_on_errors",
    "validate_bottom_wall",
    "validate_interfaces",
    "validate_materials",
    "validate_mesh",
    "validate_orders",
    "validate_periodic",
    "validate_pml",
    "validate_resolution",
    "validate_scattering",
    "validate_stack",
]

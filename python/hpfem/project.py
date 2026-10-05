"""Declarative project files: a JSON (or YAML) description of a simulation that
``hpfem run project.json`` (``hpfem.cli``) or :func:`run` turns into meshes, spaces,
problems and results without writing Python.

A project is a mapping with the keys below (lengths in ``length_unit``, the spectral
variable in its own unit; everything is converted to SI at this boundary):

.. code-block:: json

    {
      "problem": "scattering",          // "scattering" | "waveguide" | "cavity" | "resonance"
      "dim": 2,
      "length_unit": "um",
      "mesh": {"type": "square_with_disc", "n": 4, "radius": 0.25,
               "half_width": 1.0, "outer": 2.0, "inclusion_tag": 2},
      "order": 3,
      "wavelength": {"value": [1.0, 1.05], "unit": "um"},   // or "frequency" / "energy"
      "materials": {"background": "vacuum", "2": "Si"},      // library name, {"n": 1.5},
                                                              // {"eps_r": [re, im]}
      "formulation": "scattered_field",
      "source": {"type": "plane_wave", "angle": 0, "angle_unit": "deg",
                 "polarisation": "TE", "amplitude": 1.0},
      "pml": {"lower": [-1, -1], "upper": [1, 1], "thickness": 1.0,
              "profile": {"order": 2, "reflection": 1e-10}},
      "boundaries": {"pec": ["x_min", "x_max", "y_min", "y_max"], "incident": []},
      "periodic": [{"master": "y_min", "slave": "y_max", "shift": [0, 1]}],
      "solver": {"backend": "auto", "condense": true, "threads": 0},
      "outputs": {"cross_sections": {"around_tag": 2},
                  "far_field": {"around_tag": 2, "directions": 72},
                  "points": [[0.5, 0.1]],
                  "flux": [{"surface_tag": 1}],
                  "diffraction": {"x_above": 1.25, "x_below": -0.5, "orders": 1},
                  "vtk": {"file": "mie.vtu", "subdivisions": 3}}
    }

Mesh types: ``rectangle`` (nx, ny, lower, upper), ``box`` (nx, ny, nz, lower, upper),
``disc`` (n, center, radius, curved), ``ball``, ``square_with_disc`` (n, radius, half_width,
outer, inclusion_tag), ``gmsh`` (file, relative to the project file), ``extract`` is not
supported. Cells may be re-tagged by region with ``"regions": [{"tag": 2, "box": [lo, hi]}]``
(centroid inside the axis-aligned box). Boundary tags are integers or the side names
``x_min`` … ``z_max`` of the generators. A list in ``wavelength.value`` runs a sweep (one
factorisation per wavelength, re-tabulating the dispersive materials).

The results (:func:`run`) are JSON-serialisable: one entry per spectral point with the
requested outputs; complex numbers are ``[re, im]`` pairs.
"""

from __future__ import annotations

import json
import time
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from hpfem import _hpfem as hpfem  # the extension module (no circular import of the package)
from hpfem import materials as material_library
from hpfem import units

SIDE_TAGS = {
    "x_min": hpfem.box_tag.X_MIN,
    "x_max": hpfem.box_tag.X_MAX,
    "y_min": hpfem.box_tag.Y_MIN,
    "y_max": hpfem.box_tag.Y_MAX,
    "z_min": hpfem.box_tag.Z_MIN,
    "z_max": hpfem.box_tag.Z_MAX,
    "disc": hpfem.DISC_BOUNDARY,
}

LENGTH_UNITS = {"m": units.m, "cm": units.cm, "mm": units.mm, "um": units.um, "nm": units.nm}
SPECTRAL_UNITS = {
    "wavelength": {"m": units.m, "um": units.um, "nm": units.nm},
    "frequency": {"Hz": units.Hz, "GHz": units.GHz, "THz": units.THz, "PHz": units.PHz},
    "energy": {"J": units.J, "eV": units.eV, "meV": units.meV},
}


class ProjectError(ValueError):
    """A malformed project description (names the offending key)."""


# --- loading ---------------------------------------------------------------------------------


def load(path: str | Path) -> dict[str, Any]:
    """Reads a ``.json`` or ``.yaml`` / ``.yml`` project file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise ProjectError("YAML project files need the 'pyyaml' package") from error
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, Mapping):
        raise ProjectError(f"{path}: the project must be a mapping")
    data = dict(data)
    data.setdefault("_directory", str(path.parent))
    return data


def _get(mapping: Mapping, key: str, default=None, *, required=False, where="project"):
    if key in mapping:
        return mapping[key]
    if required:
        raise ProjectError(f"{where}: missing key '{key}'")
    return default


def _tag(value, where: str) -> int:
    if isinstance(value, str):
        if value not in SIDE_TAGS:
            raise ProjectError(
                f"{where}: unknown boundary name '{value}' (use {sorted(SIDE_TAGS)})"
            )
        return SIDE_TAGS[value]
    return int(value)


def _point(value, dim: int, scale: float, where: str) -> np.ndarray:
    point = np.asarray(value, dtype=float) * scale
    if point.shape != (dim,):
        raise ProjectError(f"{where}: expected {dim} coordinates, got {value}")
    return point


# --- spectral points -------------------------------------------------------------------------


def spectral_points(project: Mapping) -> list[float]:
    """Angular frequencies [rad/s] of the project's ``wavelength`` / ``frequency`` / ``energy``
    entry (a number or a list, with a ``unit``)."""
    keys = [k for k in SPECTRAL_UNITS if k in project]
    if len(keys) != 1:
        raise ProjectError("give exactly one of 'wavelength', 'frequency', 'energy'")
    key = keys[0]
    spec = project[key]
    if isinstance(spec, Mapping):
        value, unit = _get(spec, "value", required=True, where=key), _get(spec, "unit", "")
    else:
        value, unit = spec, ""
    if not unit:
        unit = {"wavelength": "m", "frequency": "Hz", "energy": "J"}[key]
    if unit not in SPECTRAL_UNITS[key]:
        raise ProjectError(f"{key}: unknown unit '{unit}' (use {sorted(SPECTRAL_UNITS[key])})")
    values = np.atleast_1d(np.asarray(value, dtype=float)) * SPECTRAL_UNITS[key][unit]
    return [float(units.angular_frequency(**{key: v})) for v in values]


# --- mesh ------------------------------------------------------------------------------------


def build_mesh(project: Mapping):
    """The mesh of the project (generator or Gmsh file, regions re-tagged)."""
    dim = int(_get(project, "dim", 2))
    if dim not in (2, 3):
        raise ProjectError("dim must be 2 or 3")
    scale = _length_scale(project)
    spec = _get(project, "mesh", required=True)
    kind = _get(spec, "type", required=True, where="mesh")
    where = f"mesh ({kind})"

    def pt(key, default):
        return _point(_get(spec, key, default), dim, scale, where)

    if kind == "rectangle" or kind == "box":
        if kind == "rectangle" and dim != 2 or kind == "box" and dim != 3:
            raise ProjectError(f"{where}: dimension mismatch with dim = {dim}")
        counts = [int(_get(spec, k, required=True, where=where)) for k in ("nx", "ny", "nz")[:dim]]
        lower = pt("lower", [0.0] * dim)
        upper = pt("upper", [1.0] * dim)
        mesh = (hpfem.rectangle if dim == 2 else hpfem.box)(*counts, lower, upper)
    elif kind in ("disc", "ball"):
        mesh = (hpfem.disc if dim == 2 else hpfem.ball)(
            int(_get(spec, "n", required=True, where=where)),
            pt("center", [0.0] * dim),
            float(_get(spec, "radius", 1.0)) * scale,
            bool(_get(spec, "curved", True)),
        )
    elif kind == "square_with_disc":
        if dim != 2:
            raise ProjectError(f"{where}: two-dimensional only")
        mesh = hpfem.square_with_disc(
            int(_get(spec, "n", required=True, where=where)),
            float(_get(spec, "radius", required=True, where=where)) * scale,
            float(_get(spec, "half_width", required=True, where=where)) * scale,
            float(_get(spec, "outer", required=True, where=where)) * scale,
            int(_get(spec, "inclusion_tag", 2)),
        )
    elif kind == "gmsh":
        file = Path(_get(spec, "file", required=True, where=where))
        if not file.is_absolute():
            file = Path(_get(project, "_directory", ".")) / file
        mesh = hpfem.read_gmsh(str(file), float(_get(spec, "scale", scale)), dim)
    else:
        raise ProjectError(f"mesh: unknown type '{kind}'")
    for region in _get(spec, "regions", []):
        tag = int(_get(region, "tag", required=True, where="mesh.regions"))
        lo, hi = (_point(v, dim, scale, "mesh.regions") for v in region["box"])
        centroids = mesh.cell_centroids
        inside = np.all((centroids >= lo) & (centroids <= hi), axis=1)
        for c in np.flatnonzero(inside):
            mesh.set_cell_tag(int(c), tag)
    return mesh


def _length_scale(project: Mapping) -> float:
    unit = _get(project, "length_unit", "m")
    if unit not in LENGTH_UNITS:
        raise ProjectError(f"length_unit: unknown unit '{unit}' (use {sorted(LENGTH_UNITS)})")
    return LENGTH_UNITS[unit]


# --- materials -------------------------------------------------------------------------------


def _dispersive(spec, where: str) -> material_library.Dispersive:
    if isinstance(spec, str):
        try:
            return material_library.get(spec)
        except KeyError as error:
            raise ProjectError(f"{where}: {error}") from None
    if isinstance(spec, Mapping):
        if "n" in spec:
            n = spec["n"]
            n = complex(n[0], n[1]) if isinstance(n, Sequence) else complex(n)
            return material_library.Constant(n * n, name=f"n={n}")
        if "eps_r" in spec:
            e = spec["eps_r"]
            e = complex(e[0], e[1]) if isinstance(e, Sequence) else complex(e)
            return material_library.Constant(e, complex(_get(spec, "mu_r", 1.0)), name=f"eps={e}")
        if "drude" in spec:
            d = spec["drude"]
            return material_library.Drude(
                float(d["eps_inf"]), float(d["omega_p"]), float(d["gamma"])
            )
    raise ProjectError(f"{where}: a material is a library name, {{'n': ...}} or {{'eps_r': ...}}")


def material_map(project: Mapping, omega: float) -> hpfem.MaterialMap:
    """The core materials of all tags at one frequency."""
    spec = _get(project, "materials", {})
    background = _dispersive(_get(spec, "background", "vacuum"), "materials.background")
    mapping = hpfem.MaterialMap(background.at(omega))
    for key, value in spec.items():
        if key == "background":
            continue
        mapping.set(int(key), _dispersive(value, f"materials.{key}").at(omega))
    return mapping


# --- scattering ------------------------------------------------------------------------------


def _plane_wave(project: Mapping, dim: int, omega: float, background_index: float):
    spec = _get(project, "source", {"type": "plane_wave"})
    kind = _get(spec, "type", "plane_wave")
    k = units.vacuum_wavenumber(omega) * background_index
    amplitude = float(_get(spec, "amplitude", 1.0))
    if kind == "plane_wave":
        if dim == 2:
            angle = float(_get(spec, "angle", 0.0))
            if _get(spec, "angle_unit", "deg") == "deg":
                angle *= units.deg
            direction = np.array([np.cos(angle), np.sin(angle)])
            polarisation = _get(spec, "polarisation", "TE")
            if isinstance(polarisation, str):
                # in-plane E perpendicular to k (the H_z polarisation of Scattering2D)
                polarisation = np.array([-direction[1], direction[0]])
        else:
            direction = np.asarray(_get(spec, "direction", [0, 0, 1]), dtype=float)
            direction /= np.linalg.norm(direction)
            polarisation = _get(spec, "polarisation", required=True, where="source")
        e0 = amplitude * np.asarray(polarisation, dtype=complex)
        return hpfem.plane_wave(e0, k * direction), amplitude
    if kind == "dipole":
        scale = _length_scale(project)
        position = _point(
            _get(spec, "position", required=True, where="source"), dim, scale, "source"
        )
        moment = np.asarray(_get(spec, "moment", required=True, where="source"), dtype=complex)
        return hpfem.dipole_field(position, moment, k), amplitude
    raise ProjectError(f"source: unknown type '{kind}'")


def _angle_against_layers(project: Mapping, thickness) -> float:
    """Largest angle (degrees) between the plane-wave direction and the normals of the sides
    that carry a layer; 0 without a plane wave or with layers on every side (a box around a
    scatterer sees every direction anyway)."""
    spec = _get(project, "source", {})
    if _get(spec, "type", "plane_wave") != "plane_wave" or not isinstance(thickness, Sequence):
        return 0.0
    angle = float(_get(spec, "angle", 0.0))
    if _get(spec, "angle_unit", "deg") != "deg":
        angle /= units.deg
    normals = []
    for side, t in enumerate(thickness):
        if float(t) > 0:
            normals.append(side // 2)  # axis of the side's normal
    if not normals or len(set(normals)) == len(thickness) // 2:
        return 0.0
    direction = np.array([np.cos(angle * units.deg), np.sin(angle * units.deg)])
    worst = 0.0
    for axis in set(normals):
        cosine = abs(direction[axis]) if axis < 2 else 0.0
        worst = max(worst, float(np.degrees(np.arccos(min(1.0, cosine)))))
    return worst


def _pml(project: Mapping, dim: int, omega: float, background_index: float):
    spec = _get(project, "pml")
    if spec is None:
        return None
    scale = _length_scale(project)
    lower = _point(_get(spec, "lower", required=True, where="pml"), dim, scale, "pml")
    upper = _point(_get(spec, "upper", required=True, where="pml"), dim, scale, "pml")
    thickness = _get(spec, "thickness", required=True, where="pml")
    profile_spec = _get(spec, "profile", {})
    if "theta_max" in profile_spec:
        # angle-aware design: R0 for the largest incidence angle (degrees) and the error target
        profile = hpfem.PmlProfile.for_angle(
            float(_get(profile_spec, "theta_max", required=True, where="pml.profile")),
            float(_get(profile_spec, "target", 1e-4)),
            float(_get(profile_spec, "r_amplitude", 1.0)),
            int(_get(profile_spec, "order", 2)),
        )
    else:
        profile = hpfem.PmlProfile(
            int(_get(profile_spec, "order", 3)), float(_get(profile_spec, "reflection", 1e-8))
        )
        angle = _angle_against_layers(project, thickness)
        if angle > 30.0 and profile.reflection > 1e-12:
            warnings.warn(
                f"pml.profile.reflection = {profile.reflection:g} is the round-trip reflection at "
                f"normal incidence; at {angle:.0f} deg against the layer normal it leaves about "
                f"{profile.reflection ** (np.cos(angle * units.deg) / 2):.1e} of the field at "
                "the far wall. Give pml.profile.theta_max (degrees) and target instead "
                "(PmlProfile.for_angle, docs/theory/pml.md).",
                stacklevel=2,
            )
    k0 = units.vacuum_wavenumber(omega)
    box = hpfem.PmlBox2D if dim == 2 else hpfem.PmlBox3D
    if isinstance(thickness, Sequence):
        return box(
            lower, upper, [float(t) * scale for t in thickness], k0, background_index, profile
        )
    return box.uniform(lower, upper, float(thickness) * scale, k0, background_index, profile)


def scattering_setup(project: Mapping, omega: float):
    """The ``ScatteringSetup`` of the project at one frequency."""
    dim = int(_get(project, "dim", 2))
    setup = (hpfem.ScatteringSetup2D if dim == 2 else hpfem.ScatteringSetup3D)()
    setup.omega = omega
    setup.materials = material_map(project, omega)
    background_index = float(np.real(setup.materials.background.refractive_index))
    setup.incident, _ = _plane_wave(project, dim, omega, background_index)
    formulation = _get(project, "formulation", "scattered_field")
    try:
        setup.formulation = {
            "scattered_field": hpfem.Formulation.SCATTERED_FIELD,
            "total_field": hpfem.Formulation.TOTAL_FIELD,
        }[formulation]
    except KeyError:
        raise ProjectError(
            f"formulation: '{formulation}' (scattered_field | total_field)"
        ) from None
    boundaries = _get(project, "boundaries", {})
    setup.pec_tags = [_tag(t, "boundaries.pec") for t in _get(boundaries, "pec", [])]
    setup.incident_tags = [_tag(t, "boundaries.incident") for t in _get(boundaries, "incident", [])]
    pml = _pml(project, dim, omega, background_index)
    if pml is not None:
        setup.pml = pml
    scale = _length_scale(project)
    pairs = []
    k_vector = _wave_vector(project, dim, omega, background_index)
    for entry in _get(project, "periodic", []):
        shift = _point(
            _get(entry, "shift", required=True, where="periodic"), dim, scale, "periodic"
        )
        pair_type = hpfem.PeriodicPair2D if dim == 2 else hpfem.PeriodicPair3D
        pairs.append(
            pair_type(_tag(entry["master"], "periodic"), _tag(entry["slave"], "periodic"), shift,
                      hpfem.bloch_phase(k_vector, shift))
        )  # fmt: skip
    setup.periodic = pairs
    solver = _get(project, "solver", {})
    backend = _get(solver, "backend", "auto")
    try:
        setup.solver = {
            "auto": hpfem.DirectSolverBackend.AUTO,
            "sparse_lu": hpfem.DirectSolverBackend.SPARSE_LU,
            "mumps": hpfem.DirectSolverBackend.MUMPS,
            "cudss": hpfem.DirectSolverBackend.CUDSS,
        }[backend]
    except KeyError:
        raise ProjectError(
            f"solver.backend: '{backend}' (auto | sparse_lu | mumps | cudss)"
        ) from None
    setup.condense = bool(_get(solver, "condense", True))
    return setup


def _wave_vector(project: Mapping, dim: int, omega: float, background_index: float):
    """Wave vector of the plane-wave source (zero for other sources), for Bloch phases."""
    spec = _get(project, "source", {"type": "plane_wave"})
    if _get(spec, "type", "plane_wave") != "plane_wave":
        return np.zeros(dim)
    k = units.vacuum_wavenumber(omega) * background_index
    if dim == 2:
        angle = float(_get(spec, "angle", 0.0))
        if _get(spec, "angle_unit", "deg") == "deg":
            angle *= units.deg
        return k * np.array([np.cos(angle), np.sin(angle)])
    direction = np.asarray(_get(spec, "direction", [0, 0, 1]), dtype=float)
    return k * direction / np.linalg.norm(direction)


def _complex(value) -> list[float]:
    value = complex(value)
    return [value.real, value.imag]


def _scattering_outputs(project: Mapping, mesh, dofs, problem, solution, omega: float) -> dict:
    dim = int(_get(project, "dim", 2))
    scale = _length_scale(project)
    outputs = _get(project, "outputs", {})
    setup = problem.setup
    amplitude = float(_get(_get(project, "source", {}), "amplitude", 1.0))
    surface_type = hpfem.Surface2D if dim == 2 else hpfem.Surface3D
    result: dict[str, Any] = {}
    if "cross_sections" in outputs:
        spec = outputs["cross_sections"]
        surface = surface_type.around_cells(
            mesh, int(_get(spec, "around_tag", required=True, where="outputs.cross_sections"))
        )
        cs = hpfem.cross_sections(
            problem, solution, surface, amplitude, int(_get(spec, "extra_order", 2))
        )
        result["cross_sections"] = {
            "scattering": cs.scattering,
            "absorption": cs.absorption,
            "extinction": cs.extinction,
        }
    if "far_field" in outputs:
        spec = outputs["far_field"]
        surface = surface_type.around_cells(
            mesh, int(_get(spec, "around_tag", required=True, where="outputs.far_field"))
        )
        far_type = hpfem.FarField2D if dim == 2 else hpfem.FarField3D
        field = hpfem.discrete_field(dofs, solution.unknown)
        far = far_type(
            mesh, surface, field, omega, setup.materials.background, int(_get(spec, "order", 8))
        )
        count = int(_get(spec, "directions", 72))
        entry: dict[str, Any] = {
            "scattering_cross_section": far.scattering_cross_section(amplitude)
        }
        if dim == 2:
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            entry["angles_deg"] = (angles / units.deg).tolist()
            entry["pattern_abs"] = [
                float(np.linalg.norm(far.pattern([np.cos(a), np.sin(a)]))) for a in angles
            ]
        result["far_field"] = entry
    if "points" in outputs:
        locator = (hpfem.PointLocator2D if dim == 2 else hpfem.PointLocator3D)(mesh)
        values = []
        for raw in outputs["points"]:
            x = _point(raw, dim, scale, "outputs.points")
            total = problem.total_field(solution, locator, x)
            values.append(None if total is None else [_complex(v) for v in total])
        result["points"] = values
    if "flux" in outputs:
        fluxes = []
        for spec in outputs["flux"]:
            if "surface_tag" in spec:
                surface = surface_type.boundary(mesh, _tag(spec["surface_tag"], "outputs.flux"))
            else:
                surface = surface_type.around_cells(
                    mesh, int(_get(spec, "around_tag", required=True, where="outputs.flux"))
                )
            fluxes.append(
                hpfem.poynting_flux(dofs, solution.unknown, omega, setup.materials, surface)
            )
        result["flux"] = fluxes
    if "diffraction" in outputs and dim == 2:
        spec = outputs["diffraction"]
        period = float(_get(spec, "period", required=True, where="outputs.diffraction")) * scale
        orders = int(_get(spec, "orders", 1))
        locator = hpfem.PointLocator2D(mesh)
        k_vector = _wave_vector(
            project, 2, omega, float(np.real(setup.materials.background.refractive_index))
        )
        points = int(_get(spec, "points", 64))
        k0 = units.vacuum_wavenumber(omega)
        kx_incident = abs(float(k_vector[0]))
        entry = {}
        for side, field in (("above", "scattered"), ("below", "total")):
            if f"x_{side}" not in spec:
                continue
            x0 = float(spec[f"x_{side}"]) * scale
            tag = spec.get(f"tag_{side}")
            index = float(
                np.real(
                    (
                        setup.materials.at(int(tag))
                        if tag is not None
                        else setup.materials.background
                    ).refractive_index
                )
            )
            sample = problem.scattered_field if field == "scattered" else problem.total_field
            coefficients = hpfem.fourier_coefficients(
                lambda x, sample=sample: sample(solution, locator, x),
                x0,
                0.0,
                period,
                float(k_vector[1]),
                orders,
                points,
            )
            efficiencies = hpfem.diffraction_efficiencies(
                coefficients, k0, index, period, float(k_vector[1]), kx_incident, amplitude
            )
            entry[side] = [
                {"order": o.order, "efficiency": o.efficiency, "propagating": o.propagating}
                for o in efficiencies
            ]
        if "line" in spec:
            # orders on a line of any orientation (diffraction_orders): the total field with
            # the incident wave subtracted (reflected orders) or the field alone
            line_spec = spec["line"]
            origin = _point(
                _get(line_spec, "origin", required=True, where="outputs.diffraction.line"),
                2,
                scale,
                "outputs.diffraction.line",
            )
            tangent = np.array(_get(line_spec, "tangent", [1.0, 0.0]), dtype=float)
            normal = np.array(_get(line_spec, "normal", [0.0, 1.0]), dtype=float)
            tangent /= np.linalg.norm(tangent)
            normal /= np.linalg.norm(normal)
            line = hpfem.OrderLine(origin, tangent, normal, period)
            index = float(
                _get(line_spec, "index", np.real(setup.materials.background.refractive_index))
            )
            k_tangential = float(np.dot(k_vector, tangent))
            kn_incident = abs(float(np.dot(k_vector, normal)))
            subtract = bool(_get(line_spec, "subtract_incident", True))
            orders_on_line = hpfem.diffraction_orders(
                lambda x: problem.total_field(solution, locator, x), line, k0, index,
                k_tangential, kn_incident, problem.incident_wave if subtract else None, orders,
                points, amplitude,
            )  # fmt: skip
            entry["line"] = [
                {
                    "order": o.order,
                    "efficiency": o.efficiency,
                    "propagating": o.propagating,
                    "amplitude": [_complex(v) for v in o.amplitude],
                }
                for o in orders_on_line
            ]
        if "balance" in spec:
            # flux-based energy balance: planes given by their coordinate along the normal axis
            balance_spec = spec["balance"]
            axis = int(_get(balance_spec, "axis", 1))
            reflection = hpfem.Surface2D.plane(
                mesh,
                axis,
                float(
                    _get(
                        balance_spec,
                        "reflection",
                        required=True,
                        where="outputs.diffraction.balance",
                    )
                )
                * scale,
                +1,
            )
            transmission = None
            if "transmission" in balance_spec:
                transmission = hpfem.Surface2D.plane(
                    mesh, axis, float(balance_spec["transmission"]) * scale, -1
                )
            balance = hpfem.power_balance(
                problem, solution, reflection, period, abs(float(k_vector[axis])), amplitude,
                transmission,
            )  # fmt: skip
            entry["balance"] = {
                "incident": balance.incident,
                "reflected": balance.reflected,
                "transmitted": balance.transmitted,
                "absorbed": balance.absorbed,
                "relative_residual": balance.relative_residual(),
            }
        result["diffraction"] = entry
    if "vtk" in outputs:
        spec = outputs["vtk"]
        file = Path(_get(spec, "file", "fields.vtu"))
        if not file.is_absolute():
            file = Path(_get(project, "_directory", ".")) / file
        exporter = (hpfem.FieldExporter2D if dim == 2 else hpfem.FieldExporter3D)(
            mesh, int(_get(spec, "subdivisions", 2))
        )
        exporter.hcurl("E", dofs, solution.unknown)
        if "estimate" in result:
            exporter.cell_scalars("eta", np.asarray(result["estimate"]["indicators"]))
        exporter.write(str(file))
        result["vtk"] = str(file)
    return result


def run_scattering(project: Mapping) -> dict[str, Any]:
    dim = int(_get(project, "dim", 2))
    mesh = build_mesh(project)
    order = int(_get(project, "order", 2))
    dofs = (hpfem.NedelecDofMap2D if dim == 2 else hpfem.NedelecDofMap3D)(mesh, order)
    problem_type = hpfem.Scattering2D if dim == 2 else hpfem.Scattering3D
    results = []
    for omega in spectral_points(project):
        start = time.perf_counter()
        problem = problem_type(dofs, scattering_setup(project, omega))
        solution = problem.solve()
        entry = {
            "omega": omega,
            "wavelength_nm": float(units.wavelength(omega) / units.nm),
            "num_dofs": dofs.num_dofs,
        }
        if _get(_get(project, "outputs", {}), "estimate", False):
            estimate = problem.estimate(solution)
            entry["estimate"] = {
                "total": estimate.total(),
                "indicators": estimate.indicators.tolist(),
            }
        entry.update(_scattering_outputs(project, mesh, dofs, problem, solution, omega))
        entry["seconds"] = time.perf_counter() - start
        results.append(entry)
    return {"problem": "scattering", "num_cells": mesh.num_cells, "results": results}


# --- waveguide and cavity ----------------------------------------------------------------------


def run_waveguide(project: Mapping) -> dict[str, Any]:
    if int(_get(project, "dim", 2)) != 2:
        raise ProjectError("waveguide: the cross-section is two-dimensional (dim = 2)")
    mesh = build_mesh(project)
    order = int(_get(project, "order", 2))
    nd = hpfem.NedelecDofMap2D(mesh, order)
    h1 = hpfem.DofMap2D(mesh, order)
    results = []
    for omega in spectral_points(project):
        setup = hpfem.WaveguideSetup()
        setup.omega = omega
        setup.materials = material_map(project, omega)
        setup.pec_tags = [
            _tag(t, "boundaries.pec") for t in _get(_get(project, "boundaries", {}), "pec", [])
        ]
        setup.num_modes = int(_get(project, "num_modes", 2))
        modes = hpfem.PropagatingMode(nd, h1, setup).solve()
        results.append({
            "omega": omega,
            "wavelength_nm": float(units.wavelength(omega) / units.nm),
            "num_dofs": nd.num_dofs + h1.num_dofs,
            "effective_index": [m.effective_index for m in modes],
            "beta": [m.beta for m in modes],
        })  # fmt: skip
    return {"problem": "waveguide", "num_cells": mesh.num_cells, "results": results}


def run_cavity(project: Mapping) -> dict[str, Any]:
    dim = int(_get(project, "dim", 2))
    mesh = build_mesh(project)
    order = int(_get(project, "order", 2))
    nd = (hpfem.NedelecDofMap2D if dim == 2 else hpfem.NedelecDofMap3D)(mesh, order)
    h1 = (hpfem.DofMap2D if dim == 2 else hpfem.DofMap3D)(mesh, order)
    # relative tensors: eigenvalues are k0^2 of the resonances (vacuum: omega^2 / c0^2)
    mat = material_map(project, 1.0)

    def form(c):
        m = mat.of_cell(mesh, c)
        eps, inv_mu = complex(m.eps_r), 1.0 / complex(m.mu_r)
        form_type = hpfem.MaxwellForm2D if dim == 2 else hpfem.MaxwellForm3D
        return form_type(
            inverse_permeability=lambda x: inv_mu * np.eye(1 if dim == 2 else 3, dtype=complex),
            permittivity=lambda x: eps * np.eye(dim, dtype=complex),
        )

    system = hpfem.assemble_maxwell(nd, form)
    gradient = hpfem.discrete_gradient(h1, nd)
    pec = [_tag(t, "boundaries.pec") for t in _get(_get(project, "boundaries", {}), "pec", [])]
    facets = [f for f in mesh.boundary_facets if not pec or mesh.facet_tag(f) in pec]
    constrained_nd = np.unique(np.concatenate([nd.facet_dofs(f) for f in facets])) if facets else []
    constrained_h1 = np.unique(np.concatenate([h1.facet_dofs(f) for f in facets])) if facets else []
    count = int(_get(project, "num_eigenvalues", 6))
    eigen = _get(project, "eigen", {})
    options = hpfem.EigenOptions(
        num_eigenvalues=count,
        shift=float(_get(eigen, "shift", -1.0)),
        krylov_dimension=int(_get(eigen, "krylov_dimension", 4 * count + 10)),
        tolerance=float(_get(eigen, "tolerance", 1e-10)),
    )
    result = hpfem.gauged_curl_curl_eigenpairs(
        system.stiffness, system.mass, gradient,
        hpfem.free_dofs(nd.num_dofs, constrained_nd), hpfem.free_dofs(h1.num_dofs, constrained_h1),
        options,
    )  # fmt: skip
    k0 = np.sqrt(np.abs(result.eigenvalues))
    return {
        "problem": "cavity",
        "num_cells": mesh.num_cells,
        "num_dofs": nd.num_dofs,
        "k0_squared": result.eigenvalues.tolist(),
        "wavelength_nm": (2 * np.pi / k0 / units.nm).tolist(),
        "frequency_THz": (k0 * hpfem.constants.c0 / (2 * np.pi) / units.THz).tolist(),
    }


def run_resonance(project: Mapping) -> dict[str, Any]:
    """Quasi-normal modes near the spectral point(s): wavelength, Q and complex omega."""
    dim = int(_get(project, "dim", 2))
    mesh = build_mesh(project)
    order = int(_get(project, "order", 2))
    dofs = (hpfem.NedelecDofMap2D if dim == 2 else hpfem.NedelecDofMap3D)(mesh, order)
    results = []
    for omega in spectral_points(project):
        setup = (hpfem.ResonanceSetup2D if dim == 2 else hpfem.ResonanceSetup3D)()
        setup.target_omega = omega
        setup.materials = material_map(project, omega)
        boundaries = _get(project, "boundaries", {})
        setup.pec_tags = [_tag(t, "boundaries.pec") for t in _get(boundaries, "pec", [])]
        background_index = float(np.real(setup.materials.background.refractive_index))
        pml = _pml(project, dim, omega, background_index)
        if pml is not None:
            setup.pml = pml
        setup.num_modes = int(_get(project, "num_modes", 4))
        eigen = _get(project, "eigen", {})
        setup.krylov_dimension = int(_get(eigen, "krylov_dimension", 0))
        setup.tolerance = float(_get(eigen, "tolerance", 1e-10))
        setup.max_iterations = int(_get(eigen, "max_iterations", 100))
        modes = (hpfem.Resonance2D if dim == 2 else hpfem.Resonance3D)(dofs, setup).solve()
        results.append({
            "target_omega": omega,
            "target_wavelength_nm": float(units.wavelength(omega) / units.nm),
            "num_dofs": dofs.num_dofs,
            "modes": [
                {
                    "omega": _complex(m.omega),
                    "wavelength_nm": m.wavelength / units.nm,
                    "quality": m.quality,
                    "residual": m.residual,
                }
                for m in modes
            ],
        })  # fmt: skip
    return {"problem": "resonance", "num_cells": mesh.num_cells, "results": results}


RUNNERS = {
    "scattering": run_scattering,
    "waveguide": run_waveguide,
    "cavity": run_cavity,
    "resonance": run_resonance,
}


def run(project: Mapping | str | Path) -> dict[str, Any]:
    """Runs a project (mapping or file path) and returns the JSON-serialisable results."""
    if not isinstance(project, Mapping):
        project = load(project)
    kind = _get(project, "problem", "scattering")
    if kind not in RUNNERS:
        raise ProjectError(f"problem: '{kind}' (use {sorted(RUNNERS)})")
    threads = int(_get(_get(project, "solver", {}), "threads", 0))
    if threads:
        hpfem.set_num_threads(threads)
    return RUNNERS[kind](project)


def validate(project: Mapping) -> None:
    """Builds the mesh and the setups of a project without solving (raises ProjectError)."""
    kind = _get(project, "problem", "scattering")
    if kind not in RUNNERS:
        raise ProjectError(f"problem: '{kind}' (use {sorted(RUNNERS)})")
    build_mesh(project)
    if kind == "cavity":  # no spectral variable: the eigenvalues are the frequencies
        material_map(project, 1.0)
        return
    for omega in spectral_points(project):
        if kind == "scattering":
            scattering_setup(project, omega)
        else:
            material_map(project, omega)


__all__ = ["ProjectError", "load", "run", "validate", "build_mesh", "scattering_setup",
           "material_map", "spectral_points", "RUNNERS"]  # fmt: skip

"""Job runner (M15 F5): ``python -m hpfem.run job.json [--out DIR] [--cancel-file F]``
runs a grating job described by a JSON document (schema version 1 below), streams
progress and results as JSON lines on stdout, writes ``results.json`` and ``maps_<i>.npz``
into the output directory, and stops cooperatively on SIGTERM / SIGINT or when the cancel
file appears. :func:`run_job` is the same from Python with an ``emit`` callback.

Schema (``"version": 1``, lengths in metres unless a ``"unit"`` factor is given, angles in
degrees):

.. code-block:: json

    {"version": 1, "problem": "grating", "unit": 1e-9,
     "model": {"period": 400, "y_bottom": -800, "y_top": 950, "background_tag": 1,
               "slabs": [{"tag": 2, "y0": -800, "y1": 0}],
               "shapes": [{"kind": "rectangle", "tag": 3,
                           "params": {"x": -100, "y": 0, "width": 200, "height": 148}}]},
     "mesh": {"structured": {"nx": 16, "rows": [[-800, 0, 32], [0, 148, 6], [148, 950, 32]]}},
     "materials": {"1": "air", "2": "glass", "3": {"eps": [2.25, 0.0]}},
     "stack": {"incidence": "air", "substrate": "glass", "layers": [], "top": 0},
     "incidence": {"polarisation": "p", "theta_deg": 50, "phi_deg": 30},
     "sweep": {"wavelength": {"start": 400, "stop": 410, "count": 3}},
     "solver": {"order": 3, "pml": {"top": 419, "bottom": 419}, "bottom": "pml",
                "orders_max": 2, "check": true},
     "maps": [{"quantity": "E", "x": [-200, 200, 81], "y": [-400, 600, 201]}]}

``mesh`` is either ``{"structured": ...}`` (:func:`hpfem.meshing.structured_unit_cell`),
``{"gmsh": {"sizes": {...}, "interface_factor": 0.5}}`` (:func:`hpfem.meshing.unit_cell_mesh`,
needs the gmsh package; ``"sizes"`` default: :func:`hpfem.meshing.element_sizes` at the first
sweep point) or ``{"file": "cell.msh", "scale": 1e-9}``. Materials are library names
(``hpfem.materials.get``), ``{"eps": [re, im]}``, ``{"n": [n, k]}`` or ``{"eps": x}``; ``sweep``
holds ``"wavelength"`` or ``"theta_deg"`` as a list of values or ``{"start", "stop", "count"}``.

``"task"`` selects what is computed: ``"scattering"`` (default, the sweep above),
``"resonances"`` (the quasi-normal modes of the cell closest to ``resonance.wavelength``:
``{"resonance": {"wavelength": 405, "num_modes": 4, "kx_over_g": 0.0, "beta": 0.0}}`` with
the Bloch wavenumber in units of 2π / period, or ``"theta_deg"`` for kx = k0 n sin(theta) at
the target; :func:`hpfem.grating.resonances`) and ``"bands"`` (the same for every Bloch
wavenumber of ``{"sweep": {"kx_over_g": {"start": 0, "stop": 0.5, "count": 6}}}``:
:func:`hpfem.grating.bands`). ``incidence`` is not needed for these tasks; the maps are
written per mode (``maps_<point>_<map>_<mode>.npz``, ``"modes": [0, 1]`` in a map spec
restricts them).

Events (one JSON object per line): ``start`` (job, version_info), ``mesh`` (report),
``estimate`` (predicted ``dofs``, ``matrix_nonzeros``, ``factor_entries``, ``total_bytes``,
``backend``, ``text`` of one factorisation), ``diagnostics`` (list of code / severity / text /
hint, per point only when they change), ``progress`` (``i``, ``phase``, ``step``,
``num_steps``, ``seconds`` at the start of every phase of a solve), ``point`` (``i``, ``n``,
``wavelength``, ``theta_deg``, ``R``, ``T``, ``A``, ``balance``, ``R_orders``, ``T_orders``,
``dofs``, ``seconds``, ``timing``), ``map`` (file), ``cancelled``, ``error`` (``text``) and
``done`` (``results`` file). Cancellation is checked between the points and between the
phases of a solve. The eigen tasks emit ``mode`` (``i``, ``m``, ``omega`` as [re, im],
``wavelength``, ``Q``, ``residual``) per mode and ``point`` (``i``, ``n``, ``kx``,
``kx_over_g``, ``beta``, ``modes``, ``dofs``, ``seconds``, ``timing``) per Bloch wavenumber;
the results hold them under ``"points"`` as well.

**Schema version 2** (M16 S9, ADR-0012 §7) adds the study tasks ``"optimize"``,
``"reconstruct"`` and ``"uq"`` (version-1 documents run unchanged; results and
``version_info`` report the document's version). They evaluate the efficiencies of the cell
under ``"configurations"`` as functions of ``"parameters"``
(:class:`hpfem.opt.GratingEvaluator` on a morphed reference mesh) through a study whose
store ``<name>.study.jsonl`` lies next to ``results.json`` — running the job again replays it
and continues:

.. code-block:: json

    {"version": 2, "task": "reconstruct", "model": {...}, "mesh": {...}, "materials": {...},
     "stack": {...}, "solver": {"order": 3, "pml": {"top": 419, "bottom": 419}},
     "parameters": [{"name": "cd", "shape": 0, "trapezoid": "cd", "bounds": [80, 120]},
                    {"name": "swa", "shape": 0, "trapezoid": "angle", "bounds": [80, 90]},
                    {"name": "w", "shape": 0, "field": "width", "bounds": [180, 220]},
                    {"name": "eps", "material": 3, "part": "re", "bounds": [2.0, 2.6]}],
     "configurations": [{"wavelength": 405, "theta_deg": 65, "phi_deg": 0,
                         "polarisation": "s", "orders": [["R", 0], ["T", 0]]}],
     "morph": {"band": [-300, 450], "quality_threshold": 0.3, "remesh": false},
     "reconstruct": {"measured": [0.1, 0.8], "sigma": 0.002, "x0": {"cd": 100},
                     "posterior": {"steps": 2000, "walkers": 16}}}

Parameter values (bounds, ``x0``, distributions, results) are in job units: lengths times
``unit``, angles in degrees, permittivities as they are. ``"optimize"``: ``objective`` (index
or label of an observable, default 0), ``maximize``, ``method`` (``"L-BFGS-B"``,
``"Nelder-Mead"``, ``"differential-evolution"`` or ``"bayesian"`` with ``acquisition`` and
``use_gradients``), ``max_evaluations``, ``x0``, ``seed`` (:func:`hpfem.opt.minimize`,
:func:`hpfem.opt.bayesian_optimize`). ``"reconstruct"``: ``measured`` values or
``synthetic`` ``{"params", "noise", "seed"}`` (data from the model itself), ``sigma`` (number,
list, ``"relative"`` or none), ``x0``, ``method``; the fit with its Laplace standard errors
and correlations (:func:`hpfem.opt.fit`) and, with ``posterior``, emcee samples
(:func:`hpfem.opt.sample`; needs the optional extra ``opt-mcmc``). ``"uq"``: ``inputs``
``{"name": {"normal": [mean, std]}}`` or ``{"uniform": [lower, upper]}``, the other
parameters at their reference values (or ``fixed``), ``propagation`` ``"linear"``
(:func:`hpfem.opt.linear_propagation`) or ``"surrogate"`` (``points``, ``active``,
``samples``: :func:`hpfem.opt.build_global_surrogate` and :func:`hpfem.opt.monte_carlo`),
``sobol`` (:func:`hpfem.opt.sobol_indices` on the surrogate; the variance shares of the
linearisation otherwise). Events: ``start`` (``points`` null), ``mesh``, ``estimate``, one
``evaluation`` per evaluation (``index``, ``params``, ``values``, ``status``, ``cached``,
``seconds``; also in ``results["points"]``), ``remesh``, ``cancelled`` and ``done``; the
results carry ``task``, ``study``, ``parameters`` (scale to SI and reference value),
``observables`` and the block of the task.

The task ``"emitter"`` (schema version 2, M17 S4, ADR-0013) computes the emission of a single
Gaussian dipole in the cell (periodic in x, invariant in z) over a wavelength sweep:

.. code-block:: json

    {"version": 2, "task": "emitter", "model": {...}, "mesh": {...}, "materials": {...},
     "stack": {...}, "solver": {"order": 3, "pml": {"top": 300, "bottom": 300}},
     "emitter": {"position": [0, 160], "sigma": 8, "moment": "isotropic", "stage": "B",
                 "wavelength": {"start": 600, "stop": 700, "count": 3},
                 "scan": {"nodes": 6, "kx_nodes": 4, "depth": 0.5, "angle_nodes": [8, 16],
                          "symmetric": false},
                 "aperture": [0.5], "calibrate": true, "dry_run": false}}

``moment``: ``"isotropic"`` (the mean of the three orientations), ``"x"``, ``"y"``, ``"z"``
(unit moments along the axes: x along the period, y the stack normal, z along the lines) or
``[px, py, pz]`` [A m]. ``stage``: ``"A"`` — the Bloch array at one ``kx_over_k0``,
``beta_over_k0`` (:func:`hpfem.grating.emit`, one solve per orientation); ``"B"`` — the single
dipole by array scanning (:func:`hpfem.grating.dipole_emission` with the ``scan`` options and
``beta_max_over_k0``): F_P, the fractions up / down / nonradiated of the emitted power and,
per numerical ``aperture``, the power collected above (:func:`hpfem.grating.emission_cone`,
``aperture_side`` ``"up"`` or ``"down"``); ``"C"`` — dP/dΩ on the grid ``directions``
(``theta_deg``, ``phi_deg``, ``sides``; :func:`hpfem.grating.emission_pattern`) with the power
into each ``aperture`` from the grid (trapezoidal rule over the sampled θ ≤ asin(NA / n) and
the φ samples as a uniform circle). Before the solves an ``emission_cost`` event (and
``results["cost"]``) gives the number of solves per wavelength and in total, the memory of one
factorisation and — with ``calibrate`` (one cell sample and one plane-wave solve) — the
predicted wall time; ``dry_run`` stops there. Events: ``start``, ``mesh``, ``estimate``,
``emission_cost``, ``progress`` (``i``, ``phase``, ``step``, ``num_steps``) per solve,
``point`` per wavelength, ``cancelled`` (checked between the solves) and ``done``; the results
carry ``task``, ``emitter`` (the specification in SI), ``cost`` and ``points``.
"""

from __future__ import annotations

import json
import math
import os
import platform
import signal
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

import hpfem
from hpfem import grating, materials, meshing, units

SCHEMA_VERSION = 2
SUPPORTED_VERSIONS = (1, 2)
"""schema versions this runner reads; version-1 documents run unchanged"""


class JobError(ValueError):
    """A problem with the job document."""


def version_info(schema: int = SCHEMA_VERSION) -> dict[str, Any]:
    """Version, platform, build features and solver backends (also for the ``start`` event);
    ``schema`` is the version of the document being run (default: the newest)."""
    try:
        import gmsh  # noqa: F401

        has_gmsh = True
    except Exception:
        has_gmsh = False
    return {
        "hpfem": hpfem.__version__,
        "schema": int(schema),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "openmp": bool(hpfem.has_openmp()),
        "threads": int(hpfem.num_threads()),
        "backends": [hpfem.backend_name(b) for b in hpfem.available_backends()],
        "gmsh": has_gmsh,
    }


# --- parsing ---------------------------------------------------------------------------------


def _get(mapping: Mapping, key: str, default=None, *, required=False, where="job"):
    if key in mapping:
        return mapping[key]
    if required:
        raise JobError(f"{where}: missing '{key}'")
    return default


def _material(spec, where: str) -> materials.Dispersive:
    if isinstance(spec, str):
        try:
            return materials.get(spec)
        except KeyError as error:
            raise JobError(f"{where}: unknown material '{spec}'") from error
    if isinstance(spec, Mapping):
        if "eps" in spec:
            value = spec["eps"]
            eps = (
                complex(value[0], value[1]) if isinstance(value, (list, tuple)) else complex(value)
            )
            return materials.Constant(eps, name=str(spec.get("name", "constant")))
        if "n" in spec:
            value = spec["n"]
            n = complex(value[0], value[1]) if isinstance(value, (list, tuple)) else complex(value)
            return materials.Constant(n * n, name=str(spec.get("name", "constant")))
    raise JobError(f"{where}: a material is a library name, {{'eps': [re, im]}} or {{'n': [n, k]}}")


def _sweep_values(spec, unit: float, where: str) -> list[float]:
    """A number, a list of values, or ``{"start": a, "stop": b, "count": n}``."""
    if isinstance(spec, (int, float)):
        return [float(spec) * unit]
    if isinstance(spec, Mapping):
        count = int(_get(spec, "count", required=True, where=where))
        start = float(_get(spec, "start", required=True, where=where))
        stop = float(_get(spec, "stop", start))
        return [float(v) * unit for v in np.linspace(start, stop, count)]
    values = [float(v) * unit for v in spec]
    if not values:
        raise JobError(f"{where}: empty sweep")
    return values


def _unit_cell(model: Mapping, unit: float) -> meshing.UnitCell:
    slabs = [
        meshing.Slab(int(s["tag"]), float(s["y0"]) * unit, float(s["y1"]) * unit)
        for s in _get(model, "slabs", [])
    ]
    shapes = []
    for sh in _get(model, "shapes", []):
        params = {}
        for key, value in dict(_get(sh, "params", {})).items():
            if key == "points":
                params[key] = [(float(q[0]) * unit, float(q[1]) * unit) for q in value]
            else:
                params[key] = float(value) * unit
        shapes.append(meshing.Shape(str(sh["kind"]), int(sh["tag"]), params))
    x0 = _get(model, "x0")
    return meshing.UnitCell(
        period=float(_get(model, "period", required=True, where="model")) * unit,
        y_bottom=float(_get(model, "y_bottom", required=True, where="model")) * unit,
        y_top=float(_get(model, "y_top", required=True, where="model")) * unit,
        slabs=slabs,
        shapes=shapes,
        x0=None if x0 is None else float(x0) * unit,
        background_tag=int(_get(model, "background_tag", 0)),
    )


def _stack(spec: Mapping, omega: float, unit: float) -> hpfem.LayerStack2D:
    incidence = _material(_get(spec, "incidence", required=True, where="stack"), "stack.incidence")
    substrate = _material(_get(spec, "substrate", required=True, where="stack"), "stack.substrate")
    layers = [
        hpfem.Layer(
            _material(layer["material"], "stack.layers").at(omega), float(layer["thickness"]) * unit
        )
        for layer in _get(spec, "layers", [])
    ]
    return hpfem.LayerStack2D(
        incidence.at(omega), layers, substrate.at(omega), float(_get(spec, "top", 0.0)) * unit
    )


def _build_mesh(
    job: Mapping, cell: meshing.UnitCell, material_specs, omega: float, unit: float, base: Path
):
    spec = _get(job, "mesh", required=True)
    if "file" in spec:
        path = Path(spec["file"])
        if not path.is_absolute():
            path = base / path
        mesh, _links = hpfem.read_gmsh_periodic(str(path), float(_get(spec, "scale", 1.0)), 2)
        return mesh
    if "structured" in spec:
        s = spec["structured"]
        rows = [
            (float(r[0]) * unit, float(r[1]) * unit, int(r[2]))
            for r in _get(s, "rows", required=True, where="mesh.structured")
        ]
        return meshing.structured_unit_cell(
            cell, int(_get(s, "nx", required=True, where="mesh.structured")), rows
        )
    if "gmsh" in spec:
        g = spec["gmsh"]
        sizes = _get(g, "sizes")
        if sizes is None:
            p = int(_get(_get(job, "solver", {}), "order", 4))
            sizes = meshing.element_sizes(
                {t: m.at(omega) for t, m in material_specs.items()}, omega, p
            )
        else:
            sizes = {int(t): float(h) * unit for t, h in sizes.items()}
        mesh, _links = meshing.unit_cell_mesh(
            cell,
            sizes,
            interface_factor=float(_get(g, "interface_factor", 0.5)),
            periodic=bool(_get(g, "periodic", True)),
        )
        return mesh
    raise JobError("mesh: give 'structured', 'gmsh' or 'file'")


def _grid(spec, unit: float):
    lo, hi, n = spec
    return np.linspace(float(lo) * unit, float(hi) * unit, int(n))


# --- the run ----------------------------------------------------------------------------------


class Cancelled(Exception):
    pass


def _diagnostic_dicts(found):
    return [{"code": d.code, "severity": d.severity, "text": d.text, "hint": d.hint} for d in found]


def run_job(
    job: Mapping,
    out_dir: str | Path | None = None,
    emit: Callable[[dict], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    base: str | Path | None = None,
) -> dict[str, Any]:
    """Runs a job document and returns the results dict (also written to ``results.json`` in
    ``out_dir`` when given, with the maps as ``maps_<point>_<map>.npz``). ``emit`` receives the
    events, ``cancel()`` is polled between points."""
    emit = emit or (lambda event: None)
    base = Path(base) if base is not None else Path.cwd()
    version = int(_get(job, "version", 1))
    if version not in SUPPORTED_VERSIONS:
        raise JobError(
            f"version {job.get('version')!r}: this runner reads schema versions "
            f"{', '.join(map(str, SUPPORTED_VERSIONS))}"
        )
    if _get(job, "problem", "grating") != "grating":
        raise JobError(f"problem {job.get('problem')!r}: only 'grating' jobs are supported")
    unit = float(_get(job, "unit", 1.0))
    model = _get(job, "model", required=True)
    cell = _unit_cell(model, unit)
    material_specs = {
        int(t): _material(m, f"materials[{t}]")
        for t, m in dict(_get(job, "materials", required=True)).items()
    }
    task = str(_get(job, "task", "scattering"))
    tasks = ("scattering", "resonances", "bands") + (V2_TASKS if version >= 2 else ())
    if task not in tasks:
        hint = " (this task needs version 2)" if task in V2_TASKS else ""
        raise JobError(f"task {task!r}: use {', '.join(repr(t) for t in tasks)}{hint}")
    incidence = _get(job, "incidence", required=task == "scattering") or {}
    polarisation = str(_get(incidence, "polarisation", "p"))
    theta0 = float(_get(incidence, "theta_deg", 0.0)) * units.deg
    phi = float(_get(incidence, "phi_deg", 0.0)) * units.deg
    sweep = _get(job, "sweep", {})
    resonance = _get(job, "resonance", required=task in ("resonances", "bands")) or {}
    emitter_spec = _get(job, "emitter", required=task == "emitter") or {}
    if task == "emitter":
        wavelengths = _sweep_values(_get(emitter_spec, "wavelength", required=True,
                                         where="emitter"), unit, "emitter.wavelength")  # fmt: skip
        thetas = [0.0] * len(wavelengths)
    elif task in STUDY_TASKS:
        configurations = _get(job, "configurations", required=True)
        if not configurations:
            raise JobError("configurations: at least one measurement configuration")
        wavelengths = [float(_get(configurations[0], "wavelength", required=True,
                                  where="configurations[0]")) * unit]  # fmt: skip
        thetas = [0.0]
    elif task != "scattering":
        wavelengths = [
            float(_get(resonance, "wavelength", required=True, where="resonance")) * unit
        ]
        thetas = [0.0]
    elif "wavelength" in sweep:
        wavelengths = _sweep_values(sweep["wavelength"], unit, "sweep.wavelength")
        thetas = [theta0] * len(wavelengths)
    elif "theta_deg" in sweep:
        thetas = [v * units.deg for v in _sweep_values(sweep["theta_deg"], 1.0, "sweep.theta_deg")]
        wavelength = float(_get(incidence, "wavelength", required=True, where="incidence")) * unit
        wavelengths = [wavelength] * len(thetas)
    else:
        wavelengths = [
            float(_get(incidence, "wavelength", required=True, where="incidence")) * unit
        ]
        thetas = [theta0]
    solver = _get(job, "solver", {})
    order = _get(solver, "order", 4)
    pml_spec = _get(solver, "pml")
    pml = (
        {k: float(v) * unit for k, v in pml_spec.items()} if isinstance(pml_spec, Mapping) else None
    )
    options = dict(
        bottom=str(_get(solver, "bottom", "pml")),
        orders_max=int(_get(solver, "orders_max", 3)),
        snap_tolerance=float(_get(solver, "snap_tolerance", 1e-9)),
        pml_target=float(_get(solver, "pml_target", 1e-6)),
        check=bool(_get(solver, "check", True)),
    )
    maps = _get(job, "maps", [])
    out = Path(out_dir) if out_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    emit(
        {
            "event": "start",
            "job": _get(job, "name", ""),
            "points": None if task in STUDY_TASKS else len(wavelengths),
            "version_info": version_info(version),
        }
    )
    omega0 = units.angular_frequency(wavelength=wavelengths[0])
    mesh = _build_mesh(job, cell, material_specs, omega0, unit, base)
    report = meshing.report(mesh, material_specs)
    emit(
        {
            "event": "mesh",
            **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in report.items()},
        }
    )
    try:
        estimate = grating.estimate_memory(mesh, order, options.get("solver"))
        emit(
            {
                "event": "estimate",
                "dofs": estimate.dofs,
                "matrix_nonzeros": estimate.matrix_nonzeros,
                "factor_entries": estimate.factor_entries,
                "total_bytes": estimate.total_bytes,
                "backend": hpfem.backend_name(estimate.backend),
                "text": estimate.describe(),
            }
        )
    except (ValueError, RuntimeError) as error:  # a problem the solve reports properly
        emit({"event": "estimate", "text": f"no estimate: {error}"})
    results: dict[str, Any] = {
        "version": version,
        "job": _get(job, "name", ""),
        "version_info": version_info(version),
        "mesh": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in report.items()},
        "points": [],
        "maps": [],
        "cancelled": False,
    }
    t_start = time.perf_counter()
    if task == "emitter":
        _run_emitter(job, emitter_spec, wavelengths, mesh, material_specs, unit, order, pml,
                     options, emit, cancel, results)  # fmt: skip
        return _finish(results, out, emit, t_start)
    if task in STUDY_TASKS:
        model = _study_model(job, cell, mesh, material_specs, unit, omega0)
        _run_study_task(job, task, model, out, emit, cancel, results)
        return _finish(results, out, emit, t_start)
    if task != "scattering":
        _run_modes(
            job, task, resonance, mesh, material_specs, unit, order, pml, options, maps, out,
            emit, cancel, results, t_start,
        )  # fmt: skip
        return _finish(results, out, emit, t_start)
    last_codes = None
    for i, (wavelength, theta) in enumerate(zip(wavelengths, thetas, strict=True)):
        if cancel is not None and cancel():
            results["cancelled"] = True
            emit({"event": "cancelled", "i": i, "n": len(wavelengths)})
            break
        omega = units.angular_frequency(wavelength=wavelength)
        stack = _stack(_get(job, "stack", required=True), omega, unit)
        t0 = time.perf_counter()

        def report(event, i=i):
            emit(
                {
                    "event": "progress",
                    "i": i,
                    "phase": event.phase,
                    "step": event.step,
                    "num_steps": event.num_steps,
                    "seconds": event.seconds,
                }
            )

        try:
            result = grating.solve(
                mesh, material_specs, stack, polarisation, theta, phi, omega, order, pml=pml,
                progress=report, cancel=cancel, **options,
            )  # fmt: skip
        except hpfem.Cancelled:
            results["cancelled"] = True
            emit({"event": "cancelled", "i": i, "n": len(wavelengths)})
            break
        found = result.diagnostics
        codes = [(d.code, d.text) for d in found]
        if codes != last_codes:
            emit({"event": "diagnostics", "i": i, "items": _diagnostic_dicts(found)})
            last_codes = codes
        point = {
            "i": i,
            "wavelength": wavelength,
            "theta_deg": float(theta / units.deg),
            "phi_deg": float(phi / units.deg),
            "R": result.R,
            "T": result.T,
            "A": result.A,
            "A_by_tag": {str(k): v for k, v in result.A_by_tag.items()},
            "balance": result.power_balance_residual,
            "R_orders": [
                {
                    "m": o.m,
                    "efficiency": o.efficiency,
                    "propagating": o.propagating,
                    "amplitude": [[a.real, a.imag] for a in o.amplitude],
                }
                for o in result.R_orders
            ],
            "T_orders": [
                {
                    "m": o.m,
                    "efficiency": o.efficiency,
                    "propagating": o.propagating,
                    "amplitude": [[a.real, a.imag] for a in o.amplitude],
                }
                for o in result.T_orders
            ],
            "dofs": result.dofs,
            "beta": result.wave.beta,
            "seconds": time.perf_counter() - t0,
            "timing": dict(result.timing),
            "diagnostics": _diagnostic_dicts(found),
        }
        results["points"].append(point)
        emit(
            {
                "event": "point",
                "n": len(wavelengths),
                **{k: v for k, v in point.items() if k != "diagnostics"},
            }
        )
        for j, spec in enumerate(maps):
            xs = _grid(_get(spec, "x", required=True, where="maps"), unit)
            ys = _grid(_get(spec, "y", required=True, where="maps"), unit)
            gx, gy = np.meshgrid(xs, ys)
            points = np.column_stack([gx.ravel(), gy.ravel()])
            quantity = str(_get(spec, "quantity", "E"))
            values = result.field(
                points, quantity=quantity, scattered=bool(_get(spec, "scattered", False))
            )
            values = values.reshape(len(ys), len(xs), 3)
            entry = {"point": i, "map": j, "quantity": quantity, "shape": [len(ys), len(xs), 3]}
            if out is not None:
                file = out / f"maps_{i}_{j}.npz"
                np.savez_compressed(
                    file, x=xs, y=ys, values=values, quantity=quantity, wavelength=wavelength
                )
                entry["file"] = str(file)
            results["maps"].append(entry)
            emit({"event": "map", **entry})
    return _finish(results, out, emit, t_start)


def _finish(results: dict, out: Path | None, emit, t_start: float) -> dict:
    results["seconds"] = time.perf_counter() - t_start
    if out is not None:
        file = out / "results.json"
        file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        results["file"] = str(file)
    emit(
        {
            "event": "done",
            "results": results.get("file", ""),
            "points": len(results["points"]),
            "cancelled": results["cancelled"],
        }
    )
    return results


def _mode_dict(mode) -> dict:
    return {
        "m": mode.index,
        "omega": [mode.omega.real, mode.omega.imag],
        "wavelength": mode.wavelength,
        "Q": mode.Q,
        "residual": mode.residual,
    }


def _run_modes(
    job, task, resonance, mesh, material_specs, unit, order, pml, options, maps, out, emit,
    cancel, results, t_start,
):  # fmt: skip
    """The eigen tasks: resonances at one Bloch wavenumber or along a sweep of them."""
    omega = units.angular_frequency(
        wavelength=float(_get(resonance, "wavelength", required=True, where="resonance")) * unit
    )
    stack = _stack(_get(job, "stack", required=True), omega, unit)
    period = float(_get(_get(job, "model", required=True), "period", required=True)) * unit
    g = 2 * np.pi / period
    if task == "bands":
        sweep = _get(job, "sweep", required=True)
        kx_over_g = _sweep_values(_get(sweep, "kx_over_g", required=True, where="sweep"), 1.0,
                                  "sweep.kx_over_g")  # fmt: skip
    elif "theta_deg" in resonance:
        k0 = units.vacuum_wavenumber(omega)
        n = complex(stack.incidence_medium.refractive_index).real
        kx_over_g = [k0 * n * np.sin(float(resonance["theta_deg"]) * units.deg) / g]
    else:
        kx_over_g = [float(_get(resonance, "kx_over_g", 0.0))]
    beta = float(_get(resonance, "beta", 0.0))
    num_modes = int(_get(resonance, "num_modes", 4))
    kwargs = dict(
        beta=beta,
        num_modes=num_modes,
        order=order,
        pml=pml,
        bottom=options["bottom"],
        snap_tolerance=options["snap_tolerance"],
        pml_target=options["pml_target"],
        krylov_dimension=int(_get(resonance, "krylov_dimension", 0)),
        tolerance=float(_get(resonance, "tolerance", 1e-10)),
        max_iterations=int(_get(resonance, "max_iterations", 100)),
    )
    if options.get("solver") is not None:
        kwargs["solver"] = options["solver"]
    results["task"] = task
    for i, value in enumerate(kx_over_g):
        if cancel is not None and cancel():
            results["cancelled"] = True
            emit({"event": "cancelled", "i": i, "n": len(kx_over_g)})
            break
        t0 = time.perf_counter()

        def report(event, i=i):
            emit(
                {
                    "event": "progress",
                    "i": i,
                    "phase": event.phase,
                    "step": event.step,
                    "num_steps": event.num_steps,
                    "seconds": event.seconds,
                }
            )

        try:
            result = grating.resonances(
                mesh, material_specs, stack, omega, kx=float(value) * g, progress=report,
                cancel=cancel, **kwargs,
            )  # fmt: skip
        except hpfem.Cancelled:
            results["cancelled"] = True
            emit({"event": "cancelled", "i": i, "n": len(kx_over_g)})
            break
        modes = [_mode_dict(m) for m in result.modes]
        for m in modes:
            emit({"event": "mode", "i": i, **m})
        point = {
            "i": i,
            "kx": result.kx,
            "kx_over_g": float(value),
            "beta": beta,
            "target_wavelength": float(2 * np.pi * hpfem.constants.c0 / omega),
            "modes": modes,
            "dofs": result.dofs,
            "seconds": time.perf_counter() - t0,
            "timing": dict(result.timing),
        }
        results["points"].append(point)
        emit({"event": "point", "n": len(kx_over_g), **point})
        for j, spec in enumerate(maps):
            xs = _grid(_get(spec, "x", required=True, where="maps"), unit)
            ys = _grid(_get(spec, "y", required=True, where="maps"), unit)
            gx, gy = np.meshgrid(xs, ys)
            points = np.column_stack([gx.ravel(), gy.ravel()])
            quantity = str(_get(spec, "quantity", "E"))
            wanted = _get(spec, "modes", list(range(len(result.modes))))
            for m in wanted:
                if int(m) >= len(result.modes):
                    continue
                mode = result.modes[int(m)]
                values = mode.field(points, quantity=quantity).reshape(len(ys), len(xs), 3)
                entry = {
                    "point": i,
                    "map": j,
                    "mode": int(m),
                    "quantity": quantity,
                    "shape": [len(ys), len(xs), 3],
                }
                if out is not None:
                    file = out / f"maps_{i}_{j}_{int(m)}.npz"
                    np.savez_compressed(
                        file, x=xs, y=ys, values=values, quantity=quantity,
                        omega=np.array([mode.omega]), wavelength=mode.wavelength,
                    )  # fmt: skip
                    entry["file"] = str(file)
                results["maps"].append(entry)
                emit({"event": "map", **entry})


# --- the emitter task of schema version 2 (M17 S4) ------------------------------------------------


def _listed(value) -> list:
    """A list of the value, or the value as a one-element list."""
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _emitter_moment(spec, where: str = "emitter.moment"):
    """The orientation key and the moment vector (None for "isotropic") of a moment spec."""
    if isinstance(spec, str):
        if spec == "isotropic":
            return "isotropic", None
        if spec in ("x", "y", "z"):
            return spec, np.eye(3, dtype=complex)["xyz".index(spec)]
        raise JobError(f"{where}: {spec!r}, use 'isotropic', 'x', 'y', 'z' or [px, py, pz]")
    vector = np.asarray([complex(*c) if isinstance(c, (list, tuple)) else complex(c)
                         for c in spec], dtype=complex)  # fmt: skip
    if vector.shape != (3,):
        raise JobError(f"{where}: three components [px, py, pz] needed")
    return "moment", vector


def _aperture_from_grid(theta, phi, d_omega, na: float, n: float) -> float:
    """Power into θ ≤ asin(NA / n) from dP/dΩ on a (θ, φ) grid: trapezoidal rule in θ over the
    samples inside, φ samples as a uniform circle."""
    theta_max = math.asin(min(1.0, na / n))
    theta = np.asarray(theta)
    ring = np.asarray(d_omega).mean(axis=1) * 2 * math.pi * np.sin(theta)  # ∫ dφ per θ
    inside = theta <= theta_max + 1e-12
    t, f = theta[inside], ring[inside]
    return float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(t))) if len(t) > 1 else 0.0


def _run_emitter(job, spec, wavelengths, mesh, material_specs, unit, order, pml, options, emit,
                 cancel, results) -> None:  # fmt: skip
    """The emitter task: stage A, B or C per wavelength, the cost estimate first."""
    stage = str(_get(spec, "stage", "B", where="emitter")).upper()
    if stage not in ("A", "B", "C"):
        raise JobError(f"emitter.stage {stage!r}: use 'A', 'B' or 'C'")
    key, vector = _emitter_moment(_get(spec, "moment", "isotropic", where="emitter"))
    position = [float(v) * unit for v in _get(spec, "position", required=True, where="emitter")]
    if len(position) != 2:
        raise JobError("emitter.position: (x, y) in the cell")
    sigma = float(_get(spec, "sigma", required=True, where="emitter")) * unit
    dipole = {"position": position, "sigma": sigma}
    if vector is not None:
        dipole["moment"] = vector
    scan = dict(_get(spec, "scan", {}, where="emitter"))
    scan_kwargs = dict(nodes=int(scan.get("nodes", 6)), kx_nodes=int(scan.get("kx_nodes", 4)),
                       depth=float(scan.get("depth", 0.5)),
                       angle_nodes=tuple(int(v) for v in scan.get("angle_nodes", (8, 16))),
                       symmetric=bool(scan.get("symmetric", False)))  # fmt: skip
    beta_max_over_k0 = scan.get("beta_max_over_k0")
    apertures = [float(v) for v in _listed(_get(spec, "aperture", [], where="emitter"))]
    aperture_side = str(_get(spec, "aperture_side", "up", where="emitter"))
    directions_spec = _get(spec, "directions", required=stage == "C", where="emitter") or {}
    thetas_deg = [float(v) for v in _listed(directions_spec.get("theta_deg", []))]
    phis_deg = [float(v) for v in _listed(directions_spec.get("phi_deg", [0.0]))]
    sides = [str(v) for v in _listed(directions_spec.get("sides", ["up"]))]
    plane_options = {k: v for k, v in options.items() if k != "check"}
    results["task"] = "emitter"
    results["emitter"] = {"stage": stage, "moment": key, "position": position, "sigma": sigma,
                          "wavelengths": list(wavelengths), "apertures": apertures,
                          "aperture_side": aperture_side, "scan": {**scan_kwargs,
                          "angle_nodes": list(scan_kwargs["angle_nodes"]),
                          "beta_max_over_k0": beta_max_over_k0}}  # fmt: skip
    moments = ["x", "y", "z"] if key == "isotropic" else [key]

    def stack_at(omega):
        return _stack(_get(job, "stack", required=True), omega, unit)

    # --- the cost estimate -----------------------------------------------------------------------
    omega0 = units.angular_frequency(wavelength=wavelengths[0])
    k0 = float(units.vacuum_wavenumber(omega0))
    estimate = grating.estimate_memory(mesh, order)
    per_wavelength, calibration = {}, {}
    if stage == "B":
        cost = grating.emission_cost(
            mesh, material_specs, stack_at(omega0), dipole, omega0, order=order, pml=pml,
            beta_max=None if beta_max_over_k0 is None else float(beta_max_over_k0) * k0,
            calibrate=bool(_get(spec, "calibrate", True, where="emitter")), **scan_kwargs,
            **options,
        )  # fmt: skip
        n_theta, n_phi = scan_kwargs["angle_nodes"]
        cone = len(apertures) * 2 * _cone_directions_count(n_theta, n_phi, scan_kwargs["symmetric"])
        per_wavelength = {"samples": cost.samples, "plane_wave_solves": cost.plane_wave_solves
                          + cone}  # fmt: skip
        calibration = {"seconds_per_sample": cost.seconds_per_sample,
                       "seconds_per_plane_wave": cost.seconds_per_plane_wave}  # fmt: skip
    elif stage == "A":
        per_wavelength = {"samples": len(moments), "plane_wave_solves": 0}
    else:
        per_wavelength = {
            "samples": 0,
            "plane_wave_solves": 2 * len(thetas_deg) * len(phis_deg) * len(sides),
        }
    solves = per_wavelength["samples"] + per_wavelength["plane_wave_solves"]
    seconds = None
    if calibration.get("seconds_per_sample") is not None:
        seconds = len(wavelengths) * (
            per_wavelength["samples"] * calibration["seconds_per_sample"]
            + per_wavelength["plane_wave_solves"] * (calibration["seconds_per_plane_wave"] or 0.0)
        )  # fmt: skip
    results["cost"] = {"stage": stage, "per_wavelength": {**per_wavelength, "solves": solves},
                       "wavelengths": len(wavelengths), "solves": solves * len(wavelengths),
                       "dofs": int(estimate.dofs), "total_bytes": int(estimate.total_bytes),
                       "memory": estimate.describe(), **calibration,
                       "seconds": seconds}  # fmt: skip
    emit({"event": "emission_cost", **results["cost"]})
    if bool(_get(spec, "dry_run", False, where="emitter")):
        results["dry_run"] = True
        return

    # --- the sweep -------------------------------------------------------------------------------
    for i, wavelength in enumerate(wavelengths):
        if cancel is not None and cancel():
            results["cancelled"] = True
            emit({"event": "cancelled", "i": i, "n": len(wavelengths)})
            return
        omega = units.angular_frequency(wavelength=wavelength)
        k0 = float(units.vacuum_wavenumber(omega))
        stack = stack_at(omega)
        t0 = time.perf_counter()

        def progress(step, num_steps, phase, i=i, t0=t0):
            emit({"event": "progress", "i": i, "phase": phase, "step": int(step),
                  "num_steps": int(num_steps), "seconds": time.perf_counter() - t0})  # fmt: skip

        try:
            if stage == "A":
                point = _emitter_stage_a(mesh, material_specs, stack, dipole, moments, omega,
                                         k0, spec, order, pml, plane_options, progress,
                                         cancel)  # fmt: skip
            elif stage == "B":
                point = _emitter_stage_b(mesh, material_specs, stack, dipole, key, omega, k0,
                                         scan_kwargs, beta_max_over_k0, apertures, aperture_side,
                                         order, pml, options, progress, cancel)  # fmt: skip
            else:
                point = _emitter_stage_c(mesh, material_specs, stack, dipole, key, omega,
                                         thetas_deg, phis_deg, sides, apertures, order, pml,
                                         options, progress, cancel)  # fmt: skip
        except hpfem.Cancelled:
            results["cancelled"] = True
            emit({"event": "cancelled", "i": i, "n": len(wavelengths)})
            return
        point = {"i": i, "wavelength": wavelength, **point, "seconds": time.perf_counter() - t0}
        results["points"].append(point)
        emit({"event": "point", "n": len(wavelengths), **point})


def _cone_directions_count(n_theta: int, n_phi: int, symmetric: bool) -> int:
    """Directions of :func:`hpfem.grating.emission_cone` (one θ panel; the φ reduction of the
    mirror symmetries as there)."""
    return len(grating._hemisphere_directions(n_theta, n_phi, symmetric, True, (), 0.5))


def _emitter_stage_a(mesh, material_specs, stack, dipole, moments, omega, k0, spec, order, pml,
                     options, progress, cancel) -> dict:  # fmt: skip
    """The Bloch array at (kx, β), averaged over the orientations."""
    kx = float(_get(spec, "kx_over_k0", 0.0, where="emitter")) * k0
    beta = float(_get(spec, "beta_over_k0", 0.0, where="emitter")) * k0
    fields = ("P_cell", "up", "down", "absorbed", "guided")
    sums = dict.fromkeys(fields, 0.0)
    orders_up, orders_down, leak = {}, {}, 0.0
    for j, key in enumerate(moments):
        if cancel is not None and cancel():
            raise hpfem.Cancelled("emitter cancelled")
        progress(j, len(moments), "emit")
        moment = dipole.get("moment") if key == "moment" else np.eye(3)["xyz".index(key)]
        r = grating.emit(mesh, material_specs, stack, {**dipole, "moment": moment}, omega, kx,
                         beta, order, pml=pml, **options)  # fmt: skip
        for f in fields:
            sums[f] += getattr(r, f) / len(moments)
        for o in r.orders_up:
            orders_up[o.m] = orders_up.get(o.m, 0.0) + o.power / len(moments)
        for o in r.orders_down:
            orders_down[o.m] = orders_down.get(o.m, 0.0) + o.power / len(moments)
        leak += (r.pml_leak or 0.0) / len(moments)
    return {"stage": "A", "kx": kx, "beta": beta, **sums, "pml_leak": leak,
            "orders_up": [[m, p] for m, p in sorted(orders_up.items())],
            "orders_down": [[m, p] for m, p in sorted(orders_down.items())]}  # fmt: skip


def _emitter_stage_b(mesh, material_specs, stack, dipole, key, omega, k0, scan_kwargs,
                     beta_max_over_k0, apertures, aperture_side, order, pml, options, progress,
                     cancel) -> dict:  # fmt: skip
    """The single dipole by array scanning, its channels and the collection apertures."""
    result = grating.dipole_emission(
        mesh, material_specs, stack, dipole, omega, order=order, pml=pml,
        beta_max=None if beta_max_over_k0 is None else float(beta_max_over_k0) * k0,
        progress=lambda i, n: progress(i, n, "scan"), cancel=cancel, **scan_kwargs, **options,
    )  # fmt: skip
    p_em = result.P_em[key]
    point = {"stage": "B", "purcell": result.purcell[key], "P_em": p_em,
             "P_bulk": result.P_bulk if key != "moment" else p_em / result.purcell[key],
             "n_host": result.n_host, "lossy": result.lossy, "samples": result.samples,
             "plane_wave_solves": result.directions,
             "purcell_by_orientation": dict(result.purcell),
             "timing": dict(result.timing)}  # fmt: skip
    if result.up is not None:
        point["fractions"] = {"up": result.up[key] / p_em, "down": result.down[key] / p_em,
                              "nonradiated": result.nonradiated[key] / p_em}  # fmt: skip
    point["aperture"] = []
    for na in apertures:
        cone = grating.emission_cone(
            mesh, material_specs, stack, dipole, omega, na, side=aperture_side,
            angle_nodes=scan_kwargs["angle_nodes"], symmetric=scan_kwargs["symmetric"],
            order=order, pml=pml, progress=lambda i, n, na=na: progress(i, n, f"aperture {na}"),
            cancel=cancel, **options,
        )  # fmt: skip
        power = cone.power[key]
        point["aperture"].append({"NA": na, "side": aperture_side, "theta_max_deg":
                                  math.degrees(cone.theta_max), "power": power,
                                  "fraction": power / p_em})  # fmt: skip
    return point


def _emitter_stage_c(mesh, material_specs, stack, dipole, key, omega, thetas_deg, phis_deg,
                     sides, apertures, order, pml, options, progress, cancel) -> dict:  # fmt: skip
    """dP/dΩ on the (θ, φ) grid of every side, and the apertures from the grid."""
    spec = {**dipole, "moment": key if key != "moment" else dipole["moment"]}
    directions = [(math.radians(t), math.radians(f), side)
                  for side in sides for t in thetas_deg for f in phis_deg]  # fmt: skip
    for side in sides:
        if side not in ("up", "down"):
            raise JobError(f"emitter.directions.sides: {side!r}, use 'up' or 'down'")
    pattern = grating.emission_pattern(mesh, material_specs, stack, spec, omega, directions,
                                       order=order, pml=pml,
                                       progress=lambda i, n: progress(i, n, "pattern"),
                                       cancel=cancel, **options)  # fmt: skip
    shape = (len(thetas_deg), len(phis_deg))
    grid, by_pol, n_side, apertures_out = {}, {}, {}, []
    for s_index, side in enumerate(sides):
        block = slice(s_index * shape[0] * shape[1], (s_index + 1) * shape[0] * shape[1])
        total = pattern.total[block].reshape(shape)
        grid[side] = total.tolist()
        by_pol[side] = {pol: pattern.dP_dOmega[block, j].reshape(shape).tolist()
                        for j, pol in enumerate(pattern.pol)}  # fmt: skip
        n_side[side] = float(pattern.n[block][0]) if shape[0] * shape[1] else None
        for na in apertures:
            power = _aperture_from_grid(np.radians(thetas_deg), np.radians(phis_deg), total, na,
                                        n_side[side])  # fmt: skip
            apertures_out.append({"NA": na, "side": side, "power": power,
                                  "per_P_bulk": power / pattern.P_bulk})  # fmt: skip
    return {"stage": "C", "theta_deg": thetas_deg, "phi_deg": phis_deg, "sides": sides,
            "dP_dOmega": grid, "dP_dOmega_by_pol": by_pol, "P_bulk": pattern.P_bulk,
            "n": n_side, "aperture": apertures_out, "timing": dict(pattern.timing)}  # fmt: skip


# --- the study tasks of schema version 2 -------------------------------------------------------

STUDY_TASKS = ("optimize", "reconstruct", "uq")
V2_TASKS = (*STUDY_TASKS, "emitter")


def _to_list(value):
    return np.asarray(value, dtype=float).tolist()


class _StudyModel:
    """The evaluator of a study task and the job-unit scales of its parameters."""

    def __init__(self, evaluator, scales: dict[str, float], reference: dict[str, float]):
        self.evaluator = evaluator
        self.names = list(scales)
        self.scales = scales
        self.reference = reference  # job units

    def to_job(self, params_si) -> dict[str, float]:
        return {n: float(params_si[n]) / self.scales[n] for n in self.names if n in params_si}

    def to_si(self, values, where: str) -> dict[str, float]:
        unknown = set(values) - set(self.names)
        if unknown:
            raise JobError(f"{where}: unknown parameters {sorted(unknown)}")
        return {n: float(v) * self.scales[n] for n, v in values.items()}


def _study_model(job: Mapping, cell, mesh, material_specs, unit: float, omega0: float):
    import dataclasses

    from hpfem import opt

    geometry, material_parameters, scales = [], [], {}
    for i, p in enumerate(_get(job, "parameters", required=True)):
        where = f"parameters[{i}]"
        name = str(_get(p, "name", required=True, where=where))
        bounds = _get(p, "bounds", required=True, where=where)
        if "material" in p:
            scale = 1.0
            material_parameters.append(
                opt.MaterialParameter(
                    name,
                    int(p["material"]),
                    str(_get(p, "part", "re")),
                    float(bounds[0]),
                    float(bounds[1]),
                )  # fmt: skip
            )
        elif "trapezoid" in p:
            key = str(p["trapezoid"])
            index = {"cd": 0, "height": 1, "angle": 2}.get(key)
            if index is None:
                raise JobError(f"{where}: trapezoid parameter {key!r}, use cd, height or angle")
            scale = units.deg if key == "angle" else unit
            base = opt.trapezoid_parameters(int(_get(p, "shape", required=True, where=where)))
            geometry.append(dataclasses.replace(base[index], name=name,
                                                lower=float(bounds[0]) * scale,
                                                upper=float(bounds[1]) * scale))  # fmt: skip
        elif "field" in p:
            scale = unit
            geometry.append(opt.GeometryParameter.field(
                name, int(_get(p, "shape", required=True, where=where)), str(p["field"]),
                float(bounds[0]) * scale, float(bounds[1]) * scale,
            ))  # fmt: skip
        else:
            raise JobError(f"{where}: give 'trapezoid', 'field' (with 'shape') or 'material'")
        if name in scales:
            raise JobError(f"{where}: duplicate parameter name {name!r}")
        scales[name] = scale
    morph_spec = _get(job, "morph", {})
    band = _get(morph_spec, "band")
    morph = opt.Morph(
        cell, mesh, geometry, quality_threshold=float(_get(morph_spec, "quality_threshold", 0.3)),
        band=None if band is None else (float(band[0]) * unit, float(band[1]) * unit),
    )  # fmt: skip
    evaluator_materials = dict(material_specs)
    for p in material_parameters:  # a material parameter needs a frequency-independent material
        spec = material_specs.get(p.tag)
        if not isinstance(spec, materials.Constant):
            raise JobError(f"parameter {p.name}: material {p.tag} must be given as eps or n")
        evaluator_materials[p.tag] = spec.at(omega0)
    configurations = []
    for i, c in enumerate(_get(job, "configurations", required=True)):
        where = f"configurations[{i}]"
        orders = [(str(o[0]), int(o[1])) for o in _get(c, "orders", [["R", 0]])]
        configurations.append(opt.Configuration(
            float(_get(c, "wavelength", required=True, where=where)) * unit,
            float(_get(c, "theta_deg", 0.0)) * units.deg,
            float(_get(c, "phi_deg", 0.0)) * units.deg,
            str(_get(c, "polarisation", "s")), tuple(orders),
        ))  # fmt: skip
    solver = _get(job, "solver", {})
    pml_spec = _get(solver, "pml")
    pml = (
        {k: float(v) * unit for k, v in pml_spec.items()} if isinstance(pml_spec, Mapping) else None
    )
    mesher = None
    mesh_spec = _get(job, "mesh", {})
    if bool(_get(morph_spec, "remesh", False)):
        if "structured" not in mesh_spec:
            raise JobError("morph.remesh: only structured meshes are rebuilt by the runner")
        s = mesh_spec["structured"]
        rows = [(float(r[0]) * unit, float(r[1]) * unit, int(r[2])) for r in s["rows"]]

        def mesher(new_cell, nx=int(s["nx"]), rows=rows):
            return meshing.structured_unit_cell(new_cell, nx, rows)

    evaluator = opt.GratingEvaluator(
        morph, evaluator_materials, lambda omega: _stack(job["stack"], omega, unit),
        configurations, material_parameters=material_parameters,
        order=int(_get(solver, "order", 3)), pml=pml, mesher=mesher,
        solve_options={"orders_max": int(_get(solver, "orders_max", 3)),
                       "bottom": str(_get(solver, "bottom", "pml"))},
        name=str(_get(job, "name", "grating")) or "grating",
    )  # fmt: skip
    reference = {name: morph.reference[name] / scales[name] for name in morph.reference}
    for p in material_parameters:
        reference[p.name] = p.get(evaluator_materials)
    return _StudyModel(evaluator, scales, reference)


def _distribution(spec, scale: float, where: str):
    from hpfem import opt

    if "normal" in spec:
        mean, std = spec["normal"]
        return opt.Normal(float(mean) * scale, float(std) * scale)
    if "uniform" in spec:
        lo, hi = spec["uniform"]
        return opt.Uniform(float(lo) * scale, float(hi) * scale)
    raise JobError(f"{where}: give {{'normal': [mean, std]}} or {{'uniform': [lower, upper]}}")


def _run_study_task(job, task, model: _StudyModel, out, emit, cancel, results) -> None:
    """optimize / reconstruct / uq on a study whose store lies next to the results."""
    from hpfem import opt

    path = None
    if out is not None:
        stem = "".join(
            ch if ch.isalnum() or ch in "-_" else "_" for ch in str(_get(job, "name", ""))
        )
        path = out / f"{stem or 'job'}.study.jsonl"

    def forward(event: Mapping) -> None:
        kind = event.get("event")
        if kind == "evaluation":
            entry = {
                "index": int(event.get("index", len(results["points"]))),
                "params": model.to_job(event.get("params", {})),
                "values": event.get("values"),
                "status": event.get("status", "ok"),
                "cached": bool(event.get("cached", False)),
                "seconds": event.get("cost"),
            }
            results["points"].append(entry)
            emit({"event": "evaluation", **entry})
        elif kind == "remesh":
            emit({"event": "remesh", "mesh_id": event.get("mesh_id"),
                  "params": model.to_job(event.get("params", {}))})  # fmt: skip

    study = opt.Study(model.evaluator, path, emit=forward, cancel=cancel)
    results["task"] = task
    results["study"] = None if path is None else str(path)
    results["parameters"] = {n: {"scale": model.scales[n], "reference": model.reference[n]}
                             for n in model.names}  # fmt: skip
    results["observables"] = list(model.evaluator.observables)
    spec = _get(job, task, {})
    try:
        if task == "optimize":
            results["optimize"] = _optimize(study, model, spec)
        elif task == "reconstruct":
            results["reconstruct"] = _reconstruct(study, model, spec)
        else:
            results["uq"] = _uq(study, model, spec)
    except hpfem.Cancelled:
        results["cancelled"] = True
        emit({"event": "cancelled", "i": len(results["points"]), "n": None})


def _objective(spec, observables):
    objective = _get(spec, "objective", 0)
    if isinstance(objective, str) and objective not in observables:
        raise JobError(f"optimize.objective {objective!r}: not one of {observables}")
    return objective


def _optimize(study, model: _StudyModel, spec: Mapping) -> dict:
    from hpfem import opt

    method = str(_get(spec, "method", "L-BFGS-B"))
    objective = _objective(spec, study.observables)
    x0 = _get(spec, "x0")
    common = dict(maximize=bool(_get(spec, "maximize", False)),
                  max_evaluations=_get(spec, "max_evaluations"))  # fmt: skip
    if method in ("bayesian", "bo"):
        result = opt.bayesian_optimize(
            study, objective, acquisition=str(_get(spec, "acquisition", "ei")),
            use_gradients=bool(_get(spec, "use_gradients", False)),
            seed=int(_get(spec, "seed", 0)), **{**common, "max_evaluations":
                                                int(common["max_evaluations"] or 30)},
        )  # fmt: skip
    else:
        result = opt.minimize(
            study, objective, method=method,
            x0=None if x0 is None else model.to_si(x0, "optimize.x0"),
            seed=_get(spec, "seed"), **common,
        )  # fmt: skip
    return {
        "method": method,
        "objective": objective,
        "params": model.to_job(result.params),
        "value": float(result.value),
        "values": _to_list(result.evaluation.values) if result.evaluation is not None else None,
        "success": bool(result.success),
        "message": str(result.message),
        "evaluations": int(result.evaluations),
        "new_evaluations": int(result.new_evaluations),
        "cache_hits": int(result.cache_hits),
    }


def _reconstruct(study, model: _StudyModel, spec: Mapping) -> dict:
    from hpfem import opt

    measured = _get(spec, "measured")
    synthetic = _get(spec, "synthetic")
    if (measured is None) == (synthetic is None):
        raise JobError("reconstruct: give either 'measured' values or 'synthetic' data")
    sigma = _get(spec, "sigma")
    out: dict[str, Any] = {}
    if synthetic is not None:  # data from the model itself, with Gaussian noise
        truth = model.to_si(_get(synthetic, "params", required=True, where="synthetic"),
                            "reconstruct.synthetic.params")  # fmt: skip
        clean = study.evaluate({**model.to_si(model.reference, "reference"), **truth}).values
        noise = float(_get(synthetic, "noise", 0.0))
        rng = np.random.default_rng(int(_get(synthetic, "seed", 0)))
        measured = clean + noise * rng.standard_normal(len(clean))
        sigma = noise if sigma is None and noise > 0 else sigma
        out["measured"] = _to_list(measured)
    if isinstance(sigma, list):
        sigma = np.asarray(sigma, dtype=float)
    x0 = _get(spec, "x0")
    result = opt.fit(
        study, np.asarray(measured, dtype=float), sigma,
        x0=None if x0 is None else model.to_si(x0, "reconstruct.x0"),
        method=str(_get(spec, "method", "lm")),
        max_iterations=int(_get(spec, "max_iterations", 200)),
    )  # fmt: skip
    names = list(result.names)
    std = {n: float(s) / model.scales[n] for n, s in zip(names, result.std, strict=True)}
    out.update({
        "params": model.to_job(result.params),
        "std": std,
        "correlation": np.asarray(result.correlation, dtype=float).tolist(),
        "names": names,
        "chi2_red": float(result.chi2_red),
        "cost": float(result.cost),
        "at_bounds": list(result.at_bounds),
        "warnings": [str(w) for w in result.warnings],
        "iterations": int(result.iterations),
        "evaluations": int(result.evaluations),
        "success": bool(result.success),
        "message": str(result.message),
        "values": _to_list(result.values),
        "residual": _to_list(result.residual),
    })  # fmt: skip
    posterior = _get(spec, "posterior")
    if posterior is not None:
        try:
            import emcee  # noqa: F401
        except ImportError as error:
            raise JobError(
                "reconstruct.posterior needs emcee, the optional extra opt-mcmc "
                "(pip install hpfem[opt-mcmc])"
            ) from error
        walkers = _get(posterior, "walkers")
        post = opt.sample(result, walkers=None if walkers is None else int(walkers),
                          steps=int(_get(posterior, "steps", 2000)),
                          seed=int(_get(posterior, "seed", 0)))  # fmt: skip
        scale = np.array([model.scales[n] for n in post.names])
        out["posterior"] = {
            "names": list(post.names),
            "mean": _to_list(np.asarray(post.mean) / scale),
            "std": _to_list(np.asarray(post.std) / scale),
            "quantiles": {
                str(q): _to_list(np.asarray(v) / scale) for q, v in post.quantiles.items()
            },
            "acceptance": float(post.acceptance),
            "autocorr": _to_list(post.autocorr),
            "walkers": int(post.walkers),
            "steps": int(post.steps),
        }
    return out


def _uq(study, model: _StudyModel, spec: Mapping) -> dict:
    from hpfem import opt

    inputs_spec = dict(_get(spec, "inputs", required=True, where="uq"))
    inputs = {}
    for name, dist in inputs_spec.items():
        if name not in model.scales:
            raise JobError(f"uq.inputs: unknown parameter {name!r}")
        inputs[name] = _distribution(dist, model.scales[name], f"uq.inputs.{name}")
    fixed = model.to_si({n: v for n, v in model.reference.items() if n not in inputs}, "reference")
    fixed.update(model.to_si(dict(_get(spec, "fixed", {})), "uq.fixed"))
    observables = _get(spec, "observables")
    propagation = str(_get(spec, "propagation", "linear"))
    out: dict[str, Any] = {"propagation": propagation, "inputs": list(inputs)}
    if propagation == "linear":
        lin = opt.linear_propagation(study, inputs, observables=observables, fixed=fixed)
        scale = np.array([model.scales[n] for n in lin.inputs])
        out.update({
            "observables": list(lin.observables),
            "values": _to_list(lin.values),
            "std": _to_list(lin.std),
            "covariance": np.asarray(lin.covariance).tolist(),
            "jacobian": (np.asarray(lin.jacobian) * scale).tolist(),  # per job unit
            "contributions": np.asarray(lin.contributions).tolist(),
        })  # fmt: skip
        if bool(_get(spec, "sobol", False)):  # linear model: the shares are the indices
            out["sobol"] = {"first": out["contributions"], "total": out["contributions"]}
        return out
    if propagation != "surrogate":
        raise JobError(f"uq.propagation {propagation!r}: use 'linear' or 'surrogate'")
    surrogate = opt.build_global_surrogate(
        study, inputs, points=_get(spec, "points"), active=int(_get(spec, "active", 0)),
        gradients=bool(_get(spec, "gradients", True)), observables=observables, fixed=fixed,
        seed=int(_get(spec, "seed", 0)),
    )  # fmt: skip
    mc = opt.monte_carlo(surrogate, inputs, samples=int(_get(spec, "samples", 10000)),
                         seed=int(_get(spec, "seed", 0)))  # fmt: skip
    out.update({
        "observables": list(mc.observables),
        "mean": _to_list(mc.mean),
        "std": _to_list(mc.std),
        "quantiles": {str(q): _to_list(v) for q, v in mc.quantiles.items()},
        "clipped": int(mc.clipped),
        "surrogate_std": _to_list(mc.surrogate_std),
    })  # fmt: skip
    if bool(_get(spec, "sobol", False)):
        sob = opt.sobol_indices(surrogate, inputs, samples=int(_get(spec, "sobol_samples", 4096)),
                                seed=int(_get(spec, "seed", 0)))  # fmt: skip
        out["sobol"] = {
            "first": np.asarray(sob.first).tolist(),
            "total": np.asarray(sob.total).tolist(),
            "first_conf": np.asarray(sob.first_conf).tolist(),
            "total_conf": np.asarray(sob.total_conf).tolist(),
            "samples": int(sob.samples),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m hpfem.run", description="run a grating job (JSON) with JSON-lines events"
    )
    parser.add_argument("job", help="job file (JSON)")
    parser.add_argument("--out", help="output directory for results.json and the maps")
    parser.add_argument("--cancel-file", help="stop after the current point once this file exists")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--quiet", action="store_true", help="library log level warn")
    args = parser.parse_args(argv)
    if args.quiet:
        hpfem.set_log_level("warn")
    if args.threads:
        hpfem.set_num_threads(args.threads)
    flag = {"cancel": False}

    def on_signal(_signum, _frame):
        flag["cancel"] = True

    for name in ("SIGTERM", "SIGINT"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), on_signal)

    def cancel() -> bool:
        return flag["cancel"] or (args.cancel_file is not None and os.path.exists(args.cancel_file))

    def emit(event: dict) -> None:
        sys.stdout.write(json.dumps(event) + "\n")
        sys.stdout.flush()

    try:
        job = json.loads(Path(args.job).read_text(encoding="utf-8"))
        results = run_job(job, args.out, emit, cancel, base=Path(args.job).resolve().parent)
    except Exception as error:  # the GUI reads the event, the exit code says it failed
        emit({"event": "error", "type": type(error).__name__, "text": str(error)})
        return 1
    return 2 if results["cancelled"] else 0


__all__ = [
    "JobError", "SCHEMA_VERSION", "STUDY_TASKS", "SUPPORTED_VERSIONS", "V2_TASKS", "main",
    "run_job",
    "version_info",
]  # fmt: skip

if __name__ == "__main__":
    sys.exit(main())

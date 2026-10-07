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
"""

from __future__ import annotations

import json
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

SCHEMA_VERSION = 1


class JobError(ValueError):
    """A problem with the job document."""


def version_info() -> dict[str, Any]:
    """Version, platform, build features and solver backends (also for the ``start`` event)."""
    try:
        import gmsh  # noqa: F401

        has_gmsh = True
    except Exception:
        has_gmsh = False
    return {
        "hpfem": hpfem.__version__,
        "schema": SCHEMA_VERSION,
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
    if int(_get(job, "version", 1)) != SCHEMA_VERSION:
        raise JobError(
            f"version {job.get('version')!r}: this runner reads schema version {SCHEMA_VERSION}"
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
    if task not in ("scattering", "resonances", "bands"):
        raise JobError(f"task {task!r}: use 'scattering', 'resonances' or 'bands'")
    incidence = _get(job, "incidence", required=task == "scattering") or {}
    polarisation = str(_get(incidence, "polarisation", "p"))
    theta0 = float(_get(incidence, "theta_deg", 0.0)) * units.deg
    phi = float(_get(incidence, "phi_deg", 0.0)) * units.deg
    sweep = _get(job, "sweep", {})
    resonance = _get(job, "resonance", required=task != "scattering") or {}
    if task != "scattering":
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
            "points": len(wavelengths),
            "version_info": version_info(),
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
        "version": SCHEMA_VERSION,
        "job": _get(job, "name", ""),
        "version_info": version_info(),
        "mesh": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in report.items()},
        "points": [],
        "maps": [],
        "cancelled": False,
    }
    t_start = time.perf_counter()
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


__all__ = ["JobError", "SCHEMA_VERSION", "main", "run_job", "version_info"]

if __name__ == "__main__":
    sys.exit(main())

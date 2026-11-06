"""M16 S8 validation studies on the quick silicon-grating cell of examples/grating_reconstruction
(spectroscopic R0, s and p, 65 deg, 400 / 550 / 700 nm; CD, height and side-wall angle). Not run
in CI; the numbers go to benchmarks/results and docs/validation.md (sections H and I).

``python benchmarks/opt_validation.py bo [--seeds 0 1 2] [--budget 40]``
    (a) gradient-based against gradient-free optimisation of a spectrum match
    F(p) = sum((R(p) - R_target)/sigma)^2 (target from the same model, so F_min = 0): Bayesian
    optimisation with and without the gradients of the evaluator (use_gradients), Nelder-Mead,
    differential evolution and L-BFGS-B; evaluations to reach F <= m (rms deviation of one
    noise sigma) and F <= 0.01 m, the parameter error, the cost.

``python benchmarks/opt_validation.py dwr [--seeds 0 1 2 3 4] [--kappa 0.1]``
    (b) the DWR hypothesis of ADR-0012 §6: the goal-oriented estimate of every efficiency
    (``conical_dwr_estimate`` of the linearised order functional) against the true
    discretisation error at p = 2, 3, 4 (effectivity), and reconstructions from synthetic data
    (p = 5 on a mesh at the true geometry plus noise) with a fixed order, with the DWR
    estimate as independent noise (sigma_eff^2 = sigma^2 + eta^2) and with the DWR estimate as
    fidelity indicator (the smallest p with max eta <= kappa sigma): bias and coverage of the
    estimates over the noise seeds, and the cost.

``python benchmarks/opt_validation.py meta [--seeds 0 1 2] [--budget 40]``
    (c) the methods of (a) on a metasurface unit cell: a TiO2 ridge on fused silica (period
    500 nm, 600 nm at 36.87 deg, the Littrow angle of the -1 order), width, height and side-wall
    angle chosen to maximise the mean of the -1 transmission efficiency for s and p (a
    polarisation-independent beam deflector); a multimodal landscape, with remeshing.

``python benchmarks/opt_validation.py mf [--seeds 0 1 2] [--cost 400]``
    (d) multi-fidelity optimisation of the spectrum match of (a) with the levels p = 2 and
    p = 3 under the same budget of solve time (seconds): single-fidelity BO at p = 2 and at
    p = 3, BoTorch's cost-aware multi-fidelity knowledge gradient (the optional extra opt-bo)
    and BO with the DWR estimate as fidelity indicator (every point at p = 2 with the DWR
    estimates of its efficiencies, refined to p = 3 unless the estimate shows that it cannot
    beat the best p = 3 value). The target is p = 5 on a mesh at the truth; the recommendation
    of every run is re-evaluated at p = 4.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

import hpfem
from hpfem import grating, materials, units
from hpfem.meshing import Shape, Slab, UnitCell
from hpfem.opt import (
    Configuration,
    Evaluation,
    GratingEvaluator,
    Morph,
    Study,
    fit,
    trapezoid_parameters,
)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "grating_reconstruction", ROOT / "examples" / "grating_reconstruction" / "run.py"
)
ex = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ex
spec.loader.exec_module(ex)

NM = units.nm
WAVELENGTHS = (400, 550, 700)
NAMES = ("cd", "height", "angle")
RESULTS = ROOT / "benchmarks" / "results"


def configurations():
    return [Configuration(w * NM, ex.THETA, 0.0, pol, (("R", 0),))
            for w in WAVELENGTHS for pol in ("s", "p")]  # fmt: skip


def evaluator(geometry, order, remesh=True):
    """The example's model with remeshing (the structured mesher of the example) when the
    morph's quality guard trips, so that optimisers can roam the whole box."""
    cell = ex.unit_cell(geometry, geometry["height"])
    mesh = ex.line_mesh(cell)
    band = (-(ex.SUB_ROWS - 1) * ex.ROW, geometry["height"] + (ex.AIR_ROWS // 2 - 1) * ex.ROW)
    parameters = trapezoid_parameters(0, cd=ex.BOUNDS["cd"], height=ex.BOUNDS["height"],
                                      angle=ex.BOUNDS["angle"])  # fmt: skip
    morph = Morph(cell, mesh, parameters, band=band)
    pml = {"top": ex.PML_TOP_ROWS * ex.ROW, "bottom": ex.PML_BOTTOM_ROWS * ex.ROW}
    si = hpfem.materials.get("Si")
    return GratingEvaluator(morph, {ex.SUB: si, ex.LINE: si}, ex.stack_at, configurations(),
                            order=order, pml=pml, mesher=ex.line_mesh if remesh else None,
                            solve_options={"orders_max": 1})  # fmt: skip


def info() -> dict:
    return {"hpfem": hpfem.__version__, "python": platform.python_version(),
            "platform": platform.platform(), "threads": int(hpfem.num_threads()),
            "date": time.strftime("%Y-%m-%d")}  # fmt: skip


def scaled(params) -> dict:
    """Parameters in nm / deg for the report."""
    return {"cd": params["cd"] / NM, "height": params["height"] / NM,
            "angle": math.degrees(params["angle"])}  # fmt: skip


# --- (a) gradient-based against gradient-free -------------------------------------------------


def run_bo(seeds, budget, order=2):
    import hpfem.opt as opt

    target = evaluator(ex.TRUE, order)(ex.TRUE).values
    m = len(target)
    sigma = ex.NOISE

    def objective(y):
        r = (np.asarray(y) - target) / sigma
        return float(r @ r), 2 * r / sigma

    lower = np.array([ex.BOUNDS[n][0] for n in NAMES])
    upper = np.array([ex.BOUNDS[n][1] for n in NAMES])

    def start(seed):
        u = np.random.default_rng(100 + seed).random(3)
        return dict(zip(NAMES, lower + u * (upper - lower), strict=True))

    methods = {
        "bo": lambda s, seed: opt.bayesian_optimize(s, objective, max_evaluations=budget,
                                                    seed=seed),  # fmt: skip
        "bo_gradients": lambda s, seed: opt.bayesian_optimize(
            s, objective, max_evaluations=budget, seed=seed, use_gradients=True),
        "nelder_mead": lambda s, seed: opt.minimize(s, objective, method="Nelder-Mead",
                                                    x0=start(seed), max_evaluations=budget),
        "differential_evolution": lambda s, seed: opt.minimize(
            s, objective, method="differential-evolution", seed=seed, max_evaluations=budget),
        "lbfgsb": lambda s, seed: opt.minimize(s, objective, method="L-BFGS-B", x0=start(seed),
                                               max_evaluations=budget),
    }  # fmt: skip
    runs = []
    for name, method in methods.items():
        for seed in seeds:
            ev = evaluator(ex.START, order)
            study = Study(ev)
            t0 = time.perf_counter()
            try:
                method(study, seed)
                message = "ok"
            except Exception as error:  # a failed run is a result as well
                message = f"{type(error).__name__}: {error}"
            seconds = time.perf_counter() - t0
            history, best, best_params = [], math.inf, None
            for e in study.evaluations:
                f = objective(e.values)[0] if e.ok else math.inf
                if f < best:
                    best, best_params = f, e.params
                history.append(best)

            def first(threshold, history=history):
                return next((i + 1 for i, f in enumerate(history) if f <= threshold), None)

            error = None
            if best_params is not None:
                error = {n: abs(best_params[n] - ex.TRUE[n]) for n in NAMES}
                error = {"cd_nm": error["cd"] / NM, "height_nm": error["height"] / NM,
                         "angle_deg": math.degrees(error["angle"])}  # fmt: skip
            run = {
                "method": name, "seed": seed, "evaluations": len(study.evaluations),
                "failed": len(study.failed), "remeshes": len(study.remeshes),
                "best_F": best, "evals_to_F_le_m": first(m), "evals_to_F_le_0.01m": first(0.01 * m),
                "best": None if best_params is None else scaled(best_params),
                "error": error, "seconds": seconds,
                "seconds_per_evaluation": seconds / max(len(study.evaluations), 1),
                "history": history, "message": message,
            }  # fmt: skip
            runs.append(run)
            print(f"{name:24s} seed {seed}: F_best {best:9.3g}, to F<=m {run['evals_to_F_le_m']},"
                  f" to F<=0.01m {run['evals_to_F_le_0.01m']}, {len(study.evaluations)} evals,"
                  f" {seconds:6.1f} s ({message})", flush=True)  # fmt: skip
    return {"study": "M16 S8 (a): gradient-based against gradient-free optimisation",
            "problem": {"cell": "examples/grating_reconstruction quick (Si line on Si)",
                        "wavelengths_nm": list(WAVELENGTHS), "theta_deg": 65, "order": order,
                        "observables": m, "sigma": sigma, "truth": scaled(ex.TRUE),
                        "bounds": {"cd_nm": [70, 130], "height_nm": [90, 150],
                                   "angle_deg": [78, 90]}, "budget": budget,
                        "objective": "F = sum(((R - R_target) / sigma)^2), R_target from the same "
                                     "model at the truth (F_min = 0)"},
            "info": info(), "runs": runs}  # fmt: skip


# --- (b) the DWR hypothesis ---------------------------------------------------------------------


def dwr_estimates(ev, params, order):
    """The efficiencies at params with their DWR estimates (signed estimate of the error and the
    bound sum |r_K|), from grating.solve on the evaluator's morphed mesh and frames."""
    geometry = {n: params[n] for n in NAMES}
    mesh = ev.morph.mesh_at(geometry)
    values, estimates, bounds = [], [], []
    for config, frame in zip(ev.configurations, ev._frames, strict=True):
        result = grating.solve(mesh, ev.materials, ev.stack_at(config.omega), config.polarisation,
                               config.theta, config.phi, config.omega, order, pml=frame["pml"],
                               cover_line=frame["cover_line"],
                               substrate_line=frame["substrate_line"], check=False,
                               **ev.options)  # fmt: skip
        for side, m in config.orders:
            orders = result.R_orders if side == "R" else result.T_orders
            values.append(next(o.efficiency for o in orders if o.m == m))
            functional, scale = grating._order_functional(result, m, side)
            est = hpfem.conical_dwr_estimate(result.problem, result.solution, functional)
            estimates.append(scale * complex(est.error).real)
            bounds.append(scale * float(est.total()))
    return np.array(values), np.array(estimates), np.array(bounds)


def paired_shift(runs, summary, reference="fixed_p4"):
    """The discretisation bias of every strategy: the shift of its estimate against the
    reference strategy on the same noise realisation (mean and spread over the seeds, in the
    parameter's units and in the strategy's own std), and the median wall time."""
    by_key = {(r["strategy"], r["seed"]): r for r in runs}
    for name, entry in summary.items():
        seeds = [r["seed"] for r in runs
                 if r["strategy"] == name and (reference, r["seed"]) in by_key]  # fmt: skip
        shift = np.array([[by_key[(name, k)]["estimate"][n] - by_key[(reference, k)]["estimate"][n]
                           for n in NAMES] for k in seeds])  # fmt: skip
        std = np.array([[by_key[(name, k)]["std"][n] for n in NAMES] for k in seeds])
        entry[f"shift_vs_{reference}"] = {
            "mean": dict(zip(NAMES, shift.mean(axis=0).tolist(), strict=True)),
            "spread": dict(zip(NAMES, shift.std(axis=0).tolist(), strict=True)),
            "mean_in_std": dict(zip(NAMES, (shift / std).mean(axis=0).tolist(), strict=True)),
        }  # fmt: skip
        runs_of = [by_key[(name, k)] for k in seeds]
        entry["median_seconds"] = float(np.median([r["seconds"] for r in runs_of]))
        entry["mean_evaluations"] = float(np.mean([r["evaluations"] for r in runs_of]))


def run_dwr(seeds, kappa):
    sigma = ex.NOISE
    # synthetic data: p = 5 on a mesh built at the true geometry
    t0 = time.perf_counter()
    clean = evaluator(ex.TRUE, 5, remesh=False)(ex.TRUE).values
    t_data = time.perf_counter() - t0
    # effectivity at the truth on the morphed model mesh: the DWR estimate against the true
    # discretisation error (against p = 5 on the same mesh)
    reference = evaluator(ex.START, 5, remesh=False)(ex.TRUE).values
    effectivity = {}
    eta_start = {}
    for p in (2, 3, 4):
        ev = evaluator(ex.START, p, remesh=False)
        values, est, bound = dwr_estimates(ev, ex.TRUE, p)
        error = reference - values
        effectivity[p] = {"true_error": error.tolist(), "dwr_estimate": est.tolist(),
                          "dwr_bound": bound.tolist(),
                          "max_true_error": float(np.abs(error).max()),
                          "max_estimate": float(np.abs(est).max()),
                          "corrected_error": float(np.abs(error - est).max())}  # fmt: skip
        _, est_start, _ = dwr_estimates(ev, ex.START, p)
        eta_start[p] = np.abs(est_start)
        print(f"p = {p}: max |true error| {np.abs(error).max():.3e}, max |DWR| "
              f"{np.abs(est).max():.3e}, after correction {np.abs(error - est).max():.3e}, "
              f"max eta at the start {np.abs(est_start).max():.3e}", flush=True)  # fmt: skip
    # the fidelity rule at the start point
    p_rule = next((p for p in (2, 3, 4) if eta_start[p].max() <= kappa * sigma), 4)
    strategies = {
        "fixed_p2": (2, sigma),
        "fixed_p3": (3, sigma),
        "fixed_p4": (4, sigma),
        "dwr_as_noise_p2": (2, np.sqrt(sigma**2 + eta_start[2] ** 2)),
        f"dwr_fidelity_p{p_rule}": (p_rule, sigma),
    }
    runs = []
    for name, (p, s) in strategies.items():
        for seed in seeds:
            measured = clean + sigma * np.random.default_rng(seed).standard_normal(len(clean))
            ev = evaluator(ex.START, p)
            t0 = time.perf_counter()
            result = fit(ev, measured, s, x0=ex.START)
            seconds = time.perf_counter() - t0
            dev = {n: (float(x) - ex.TRUE[n]) / float(std)
                   for n, x, std in zip(NAMES, result.x, result.std, strict=True)}  # fmt: skip
            runs.append({"strategy": name, "order": p, "seed": seed, "deviation_in_std": dev,
                         "std": scaled(dict(zip(NAMES, result.std, strict=True))),
                         "estimate": scaled(dict(zip(NAMES, result.x, strict=True))),
                         "chi2_red": float(result.chi2_red),
                         "evaluations": int(result.new_evaluations),
                         "seconds": seconds})  # fmt: skip
            print(f"{name:18s} seed {seed}: dev/std {[round(v, 2) for v in dev.values()]}, "
                  f"chi2_red {result.chi2_red:.2f}, {seconds:6.1f} s", flush=True)  # fmt: skip
    summary = {}
    for name in strategies:
        devs = np.array([[r["deviation_in_std"][n] for n in NAMES]
                         for r in runs if r["strategy"] == name])  # fmt: skip
        summary[name] = {"mean_deviation_in_std": devs.mean(axis=0).tolist(),
                         "rms_deviation_in_std": np.sqrt((devs**2).mean(axis=0)).tolist(),
                         "coverage_2std": float((np.abs(devs) <= 2).mean()),
                         "mean_seconds": float(np.mean([r["seconds"] for r in runs
                                                        if r["strategy"] == name]))}  # fmt: skip
    paired_shift(runs, summary)
    return {"study": "M16 S8 (b): the DWR estimate as fidelity indicator (ADR-0012 §6)",
            "problem": {"cell": "examples/grating_reconstruction quick (Si line on Si)",
                        "wavelengths_nm": list(WAVELENGTHS), "sigma": sigma, "kappa": kappa,
                        "truth": scaled(ex.TRUE), "start": scaled(ex.START),
                        "data": "p = 5 on a mesh at the true geometry plus Gaussian noise",
                        "data_seconds": t_data},
            "effectivity": {str(p): v for p, v in effectivity.items()},
            "eta_at_start": {str(p): v.tolist() for p, v in eta_start.items()},
            "rule_order": p_rule, "info": info(), "runs": runs, "summary": summary}  # fmt: skip


# --- (c) the metasurface unit cell ---------------------------------------------------------------

META_PERIOD = 500 * NM
META_WAVELENGTH = 600 * NM
META_THETA = math.asin(0.6)  # Littrow for the -1 order in air: sin(theta) = lambda / (2 P)
META_ROW = 20 * NM
META_AIR_ROWS, META_SUB_ROWS, META_PML_TOP_ROWS, META_PML_BOTTOM_ROWS = 10, 6, 20, 12
META_BOUNDS = {"cd": (60 * NM, 400 * NM), "height": (150 * NM, 600 * NM),
               "angle": (math.radians(80.0), math.radians(90.0))}  # fmt: skip
META_START = {"cd": 200 * NM, "height": 350 * NM, "angle": math.radians(88.0)}
META_SUB, META_RIDGE, META_AIR = 2, 3, 1


def meta_cell(geometry) -> UnitCell:
    """The deflector cell with the ridge at ``geometry``; the cell top is that of the tallest
    ridge of the box, so the measurement lines (fixed on the reference mesh) lie above every
    ridge."""
    h = geometry["height"]
    half = h / math.tan(geometry["angle"]) / 2
    params = {"x": 0.0, "y": 0.0, "bottom": geometry["cd"] + 2 * half,
              "top": geometry["cd"] - 2 * half, "height": h}  # fmt: skip
    shape = Shape("trapezoid", META_RIDGE, params)
    y0 = -(META_SUB_ROWS + META_PML_BOTTOM_ROWS) * META_ROW
    y1 = META_BOUNDS["height"][1] + (META_AIR_ROWS + META_PML_TOP_ROWS) * META_ROW
    return UnitCell(META_PERIOD, y0, y1, slabs=[Slab(META_SUB, y0, 0.0)], shapes=[shape],
                    background_tag=META_AIR)  # fmt: skip


def meta_mesh(cell: UnitCell, nx: int = 24):
    """Structured mesh of the deflector cell (the column line at P/4 bent onto the ridge wall,
    the ridge on rows of about META_ROW, the air rows stretched up to the fixed cell top)."""
    p = cell.shapes[0].params
    h = p["height"]
    rows = [(cell.y_bottom, 0.0, META_SUB_ROWS + META_PML_BOTTOM_ROWS),
            (0.0, h, max(4, round(h / META_ROW))),
            (h, cell.y_top, META_AIR_ROWS + META_PML_TOP_ROWS)]  # fmt: skip
    ys = [cell.y_bottom]
    for a, b, n in rows:
        ys += [a + (b - a) * k / n for k in range(1, n + 1)]
    half = META_PERIOD / 2
    xs = [-half + META_PERIOD * i / nx for i in range(nx + 1)]
    q = xs[nx // 2 + nx // 4]

    def wall(y):
        return 0.5 * p["bottom"] + 0.5 * (p["top"] - p["bottom"]) * y / h

    vertices = []
    for y in ys:
        for x in xs:
            if 0.0 <= y <= h:
                w = wall(y)
                x = x * w / q if abs(x) <= q else math.copysign(
                    w + (abs(x) - q) * (half - w) / (half - q), x)  # fmt: skip
            vertices.append((x, y))
    cells = []
    for j in range(len(ys) - 1):
        for i in range(nx):
            a = j * (nx + 1) + i
            cells += [(a, a + 1, a + nx + 2), (a, a + nx + 2, a + nx + 1)]
    tags = []
    for tri in cells:
        cx = sum(vertices[v][0] for v in tri) / 3
        cy = sum(vertices[v][1] for v in tri) / 3
        inside = 0 < cy < h and abs(cx) < wall(cy)
        tags.append(META_SUB if cy < 0 else (META_RIDGE if inside else META_AIR))
    mesh = hpfem.Mesh2D(vertices, cells, tags)
    tol = 1e-9 * META_PERIOD
    for f in mesh.boundary_facets:
        a, b = (mesh.vertex(int(v)) for v in mesh.facet_vertices(f))
        mx, my = 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])
        if abs(mx + half) < tol:
            mesh.set_facet_tag(f, hpfem.box_tag.X_MIN)
        elif abs(mx - half) < tol:
            mesh.set_facet_tag(f, hpfem.box_tag.X_MAX)
        elif abs(my - cell.y_bottom) < tol:
            mesh.set_facet_tag(f, hpfem.box_tag.Y_MIN)
        elif abs(my - cell.y_top) < tol:
            mesh.set_facet_tag(f, hpfem.box_tag.Y_MAX)
    return mesh


def meta_stack(omega: float):
    return hpfem.LayerStack2D(hpfem.Material.dielectric(1.0), [],
                              materials.get("SiO2").at(omega), 0.0)  # fmt: skip


def meta_evaluator(order: int = 2):
    """The deflector: the reference mesh at the tallest ridge (the lines stay above every
    ridge), morphed in a band one row inside the lines, remeshed when the guard trips."""
    h_max = META_BOUNDS["height"][1]
    cell = meta_cell({**META_START, "height": h_max})
    band = (-(META_SUB_ROWS // 2 - 1) * META_ROW, h_max + (META_AIR_ROWS // 2 - 1) * META_ROW)
    parameters = trapezoid_parameters(0, cd=META_BOUNDS["cd"], height=META_BOUNDS["height"],
                                      angle=META_BOUNDS["angle"])  # fmt: skip
    morph = Morph(cell, meta_mesh(cell), parameters, band=band)
    pml = {"top": META_PML_TOP_ROWS * META_ROW, "bottom": META_PML_BOTTOM_ROWS * META_ROW}
    configurations = [Configuration(META_WAVELENGTH, META_THETA, 0.0, pol,
                                    (("T", -1), ("T", 0), ("R", 0), ("R", -1)))
                      for pol in ("s", "p")]  # fmt: skip
    return GratingEvaluator(morph, {META_SUB: materials.get("SiO2"),
                                    META_RIDGE: materials.get("TiO2")},
                            meta_stack, configurations, order=order, pml=pml, mesher=meta_mesh,
                            solve_options={"orders_max": 2}, name="deflector")  # fmt: skip


def run_meta(seeds, budget, order=2):
    import hpfem.opt as opt

    def objective(y):  # minimise minus the mean -1 transmission of s and p
        g = np.zeros(len(y))
        g[0] = g[4] = -0.5
        return float(-0.5 * (y[0] + y[4])), g

    lower = np.array([META_BOUNDS[n][0] for n in NAMES])
    upper = np.array([META_BOUNDS[n][1] for n in NAMES])

    def start(seed):
        u = np.random.default_rng(100 + seed).random(3)
        return dict(zip(NAMES, lower + u * (upper - lower), strict=True))

    methods = {
        "bo": lambda s, seed: opt.bayesian_optimize(s, objective, max_evaluations=budget,
                                                    seed=seed),  # fmt: skip
        "bo_gradients": lambda s, seed: opt.bayesian_optimize(
            s, objective, max_evaluations=budget, seed=seed, use_gradients=True),
        "nelder_mead": lambda s, seed: opt.minimize(s, objective, method="Nelder-Mead",
                                                    x0=start(seed), max_evaluations=budget),
        "differential_evolution": lambda s, seed: opt.minimize(
            s, objective, method="differential-evolution", seed=seed, max_evaluations=budget),
        "lbfgsb": lambda s, seed: opt.minimize(s, objective, method="L-BFGS-B", x0=start(seed),
                                               max_evaluations=budget),
    }  # fmt: skip
    runs = []
    for name, method in methods.items():
        for seed in seeds:
            study = Study(meta_evaluator(order))
            t0 = time.perf_counter()
            try:
                method(study, seed)
                message = "ok"
            except Exception as error:  # a failed run is a result as well
                message = f"{type(error).__name__}: {error}"
            seconds = time.perf_counter() - t0
            history, best, best_params, best_values = [], -math.inf, None, None
            for e in study.evaluations:
                t = -objective(e.values)[0] if e.ok else -math.inf
                if t > best:
                    best, best_params, best_values = t, e.params, e.values
                history.append(best)
            run = {
                "method": name, "seed": seed, "evaluations": len(study.evaluations),
                "failed": len(study.failed), "remeshes": len(study.remeshes),
                "best_T": best, "best": None if best_params is None else scaled(best_params),
                "best_values": None if best_values is None else list(map(float, best_values)),
                "seconds": seconds,
                "seconds_per_evaluation": seconds / max(len(study.evaluations), 1),
                "history": history, "message": message,
            }  # fmt: skip
            runs.append(run)
            print(f"{name:24s} seed {seed}: T-1 best {best:.4f} at {run['best']}, "
                  f"{len(study.evaluations)} evals, {seconds:6.1f} s ({message})",
                  flush=True)  # fmt: skip
    best_known = max(r["best_T"] for r in runs)
    for r in runs:  # evaluations to within 0.01 / 0.05 of the best value any run found
        for gap in (0.01, 0.05):
            r[f"evals_to_within_{gap}"] = next(
                (i + 1 for i, t in enumerate(r["history"]) if t >= best_known - gap), None
            )
    return {"study": "M16 S8 (c): gradient-based against gradient-free optimisation of a "
                     "metasurface deflector",
            "problem": {"cell": "TiO2 ridge on SiO2, period 500 nm, air above",
                        "wavelength_nm": 600, "theta_deg": math.degrees(META_THETA),
                        "order": order, "objective": "maximise (T-1(s) + T-1(p)) / 2",
                        "bounds": {"cd_nm": [60, 400], "height_nm": [150, 600],
                                   "angle_deg": [80, 90]}, "budget": budget,
                        "start": scaled(META_START)},
            "best_known": best_known, "info": info(), "runs": runs}  # fmt: skip


# --- (d) multi-fidelity with the DWR estimate as fidelity indicator -------------------------------


class DwrScreened:
    """An evaluator of the spectrum match that solves every point at the cheap order with the
    DWR estimates of its efficiencies and refines to the target order only where the point may
    beat the best target-order value: with r = (y - y_target) / sigma and eta the signed DWR
    estimates, F = |r|^2 can change by at most dF = sum(2 |r| |eta| / sigma + (eta / sigma)^2)
    (to first order in the estimate); a point with F - dF > F_best keeps its cheap values. The
    surrogate sees the cheap values of screened points (their bias is small against their
    distance from the best) and the target values of the others."""

    def __init__(self, base: GratingEvaluator, target, sigma, cheap=2, fine=3):
        self.base, self.target, self.sigma = base, np.asarray(target), float(sigma)
        self.cheap, self.fine = int(cheap), int(fine)
        self.parameters, self.observables = base.parameters, base.observables
        self.names = base.names
        self.name = f"{base.name}-dwr-screened"
        self.settings = {**base.settings, "screen": {"cheap": cheap, "fine": fine}}
        self.best = math.inf

    def _f(self, y):
        r = (np.asarray(y) - self.target) / self.sigma
        return float(r @ r), r

    def __call__(self, params, *, jacobian=False, fidelity=None, cancel=None):
        t0 = time.perf_counter()
        ev = self.base
        geometry = {p.name: float(params[p.name]) for p in ev.morph.parameters}
        meta: dict = {}
        mesh = ev._mesh(geometry, meta)
        values, eta = [], []
        for config, frame in zip(ev.configurations, ev._frames, strict=True):
            result = grating.solve(mesh, ev.materials, ev.stack_at(config.omega),
                                   config.polarisation, config.theta, config.phi, config.omega,
                                   self.cheap, pml=frame["pml"], cover_line=frame["cover_line"],
                                   substrate_line=frame["substrate_line"], check=False,
                                   **ev.options)  # fmt: skip
            for side, m in config.orders:
                orders = result.R_orders if side == "R" else result.T_orders
                values.append(next(o.efficiency for o in orders if o.m == m))
                functional, scale = grating._order_functional(result, m, side)
                est = hpfem.conical_dwr_estimate(result.problem, result.solution, functional)
                eta.append(scale * complex(est.error).real)
        values, eta = np.array(values), np.array(eta)
        f, r = self._f(values)
        df = float(np.sum(2 * np.abs(r) * np.abs(eta) / self.sigma + (eta / self.sigma) ** 2))
        cheap_seconds = time.perf_counter() - t0
        meta.update({"eta_max": float(np.abs(eta).max()), "F_cheap": f, "dF": df,
                     "cheap_seconds": cheap_seconds})  # fmt: skip
        if f - df > self.best:
            meta.update({"order": self.cheap, "screened": True})
        else:
            fine = ev(params, fidelity={"order": self.fine}, cancel=cancel)
            values = fine.values
            self.best = min(self.best, self._f(values)[0])
            meta.update({"order": self.fine, "screened": False})
        return Evaluation(params=dict(params), values=values,
                          cost=time.perf_counter() - t0, mesh_id=ev.mesh_id,
                          meta=meta)  # fmt: skip


def run_mf(seeds, cost_budget, budget=200, only=None):
    import hpfem.opt as opt

    sigma = ex.NOISE
    target = evaluator(ex.TRUE, 5, remesh=False)(ex.TRUE).values

    def objective(y):
        r = (np.asarray(y) - target) / sigma
        return float(r @ r), 2 * r / sigma

    def f_of(values):
        return objective(values)[0]

    strategies = {
        "sf_p2": lambda s, seed: opt.bayesian_optimize(s, objective, max_evaluations=budget,
                                                       seed=seed, fidelity={"order": 2}),
        "sf_p3": lambda s, seed: opt.bayesian_optimize(s, objective, max_evaluations=budget,
                                                       seed=seed, fidelity={"order": 3}),
        # MF-KG stops by its own cost model (the measured solve seconds per level), so that its
        # recommendation, the maximiser of the posterior mean at p = 3, is still evaluated
        "mf_kg": lambda s, seed: opt.multi_fidelity_optimize(
            s, objective, fidelities=[{"order": 2}, {"order": 3}], max_evaluations=budget,
            max_cost=cost_budget, seed=seed, num_fantasies=32, raw_samples=256, num_restarts=6),
        "dwr_screened": lambda s, seed: opt.bayesian_optimize(s, objective,
                                                              max_evaluations=budget, seed=seed),
    }  # fmt: skip
    reference = evaluator(ex.START, 4)
    floor = {p: f_of(evaluator(ex.START, p)(ex.TRUE).values) for p in (2, 3, 4)}
    print(f"F at the truth on the morphed start mesh: {floor}", flush=True)
    runs = []
    for name, strategy in strategies.items():
        if only and name not in only:
            continue
        for seed in seeds:
            base = evaluator(ex.START, 3)
            model = DwrScreened(base, target, sigma) if name == "dwr_screened" else base
            box: dict = {}

            # the solve time of the evaluations so far (MF-KG: a safety margin for its final
            # recommendation)
            limit = cost_budget * (1.1 if name == "mf_kg" else 1.0)

            def over_budget(box=box, limit=limit):
                return sum(float(e.cost or 0.0) for e in box["study"].evaluations) >= limit

            study = box["study"] = Study(model, cancel=over_budget)
            t0 = time.perf_counter()
            try:
                strategy(study, seed)
                message = "evaluation limit"
            except hpfem.Cancelled:
                message = "cost budget reached"
            except Exception as error:  # a failed run is a result as well
                message = f"{type(error).__name__}: {error}"
            seconds = time.perf_counter() - t0
            # the recommendation: the best value at the target order p = 3 (sf_p2: at p = 2)
            trace, cost, best, best_params = [], 0.0, math.inf, None
            levels = []
            for e in study.evaluations:
                cost += float(e.cost or 0.0)
                order = int(e.meta.get("order", (e.fidelity or {}).get("order", 3)))
                levels.append(order)
                at_target = order == (2 if name == "sf_p2" else 3)
                if e.ok and at_target and f_of(e.values) < best:
                    best, best_params = f_of(e.values), e.params
                trace.append([cost, best])
            check = None
            if best_params is not None:
                ref = reference(best_params)
                check = {"F_p4": f_of(ref.values),
                         "error": {"cd_nm": abs(best_params["cd"] - ex.TRUE["cd"]) / NM,
                                   "height_nm": abs(best_params["height"] - ex.TRUE["height"]) / NM,
                                   "angle_deg": abs(math.degrees(best_params["angle"]
                                                                 - ex.TRUE["angle"]))}}  # fmt: skip
            run = {"strategy": name, "seed": seed, "evaluations": len(study.evaluations),
                   "evaluations_by_order": {str(p): levels.count(p) for p in sorted(set(levels))},
                   "failed": len(study.failed), "solve_seconds": cost, "wall_seconds": seconds,
                   "best_F": best, "best": None if best_params is None else scaled(best_params),
                   "check": check, "trace": trace, "message": message}  # fmt: skip
            runs.append(run)
            print(f"{name:14s} seed {seed}: F {best:9.3g} (p4 "
                  f"{check['F_p4'] if check else math.nan:9.3g}), {len(study.evaluations)} evals "
                  f"{run['evaluations_by_order']}, solve {cost:6.0f} s, wall {seconds:6.0f} s "
                  f"({message})", flush=True)  # fmt: skip
    return {"study": "M16 S8 (d): multi-fidelity BO and the DWR estimate as fidelity indicator",
            "problem": {"cell": "examples/grating_reconstruction quick (Si line on Si)",
                        "wavelengths_nm": list(WAVELENGTHS), "sigma": sigma,
                        "truth": scaled(ex.TRUE), "levels": [2, 3],
                        "target": "p = 5 on a mesh at the true geometry (no noise)",
                        "cost_budget_seconds": cost_budget, "F_floor": floor,
                        "objective": "F = sum(((R - R_target) / sigma)^2)"},
            "info": info(), "runs": runs}  # fmt: skip


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("study", choices=("bo", "dwr", "meta", "mf"))
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--budget", type=int, default=40)
    parser.add_argument("--kappa", type=float, default=0.1)
    parser.add_argument("--cost", type=float, default=400.0, help="mf: solve seconds per run")
    parser.add_argument("--strategies", nargs="+", help="mf: run only these strategies")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    hpfem.set_log_level("warn")
    if args.study == "bo":
        data = run_bo(args.seeds or [0, 1, 2], args.budget)
    elif args.study == "dwr":
        data = run_dwr(args.seeds or [0, 1, 2, 3, 4], args.kappa)
    elif args.study == "meta":
        data = run_meta(args.seeds or [0, 1, 2], args.budget)
    else:
        data = run_mf(args.seeds or [0, 1, 2], args.cost, only=args.strategies)
    out = args.out or RESULTS / f"{time.strftime('%Y-%m-%d')}-validation-opt-{args.study}.json"
    out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"written {out}")


if __name__ == "__main__":
    main()

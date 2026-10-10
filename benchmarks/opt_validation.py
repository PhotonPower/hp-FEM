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
from hpfem import grating, units
from hpfem.opt import Configuration, GratingEvaluator, Morph, Study, fit, trapezoid_parameters

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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("study", choices=("bo", "dwr"))
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--budget", type=int, default=40)
    parser.add_argument("--kappa", type=float, default=0.1)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    hpfem.set_log_level("warn")
    if args.study == "bo":
        data = run_bo(args.seeds or [0, 1, 2], args.budget)
    else:
        data = run_dwr(args.seeds or [0, 1, 2, 3, 4], args.kappa)
    out = args.out or RESULTS / f"{time.strftime('%Y-%m-%d')}-validation-opt-{args.study}.json"
    out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"written {out}")


if __name__ == "__main__":
    main()

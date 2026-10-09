"""Fabrication tolerances of a silicon line grating propagated to its reflectance (M16 S7).

The silicon line grating of ``examples/grating_reconstruction`` (period 300 nm, mid-height width
CD 100 nm, height 120 nm, side-wall angle 86 degrees, silicon from M. A. Green 2008) is made
with normally distributed fabrication errors: CD ± 2 nm, height ± 3 nm, side-wall angle ± 0.5
degrees (one standard deviation each, independent). How much does the zeroth-order reflectance
R0 (s and p, 65 degrees, 400-700 nm) scatter, and which tolerance dominates?

1. Linearised propagation (hpfem.opt.linear_propagation): one evaluation with the Jacobian at
   the nominal geometry, C_R = J Sigma J^T, and the share of every tolerance in the variance.
2. A global gradient-enhanced surrogate of the 14 reflectances over +-4 standard deviations
   (hpfem.opt.build_global_surrogate: Latin hypercube with the Jacobian, then active learning).
3. Monte Carlo on the surrogate (hpfem.opt.monte_carlo, 20000 samples): mean, standard
   deviation and the 2.5-97.5 % band of every reflectance, against the linearised values.
4. First-order and total Sobol' indices (hpfem.opt.sobol_indices) on the surrogate.

The model (mesh morphed by CD, height and angle; efficiencies and Jacobian on the kept
factorisation) is the one of the reconstruction example, imported from there. Run
``python examples/fabrication_tolerance/run.py [--quick]``; the results go to
``fabrication_tolerance.json``, the evaluations to ``fabrication_tolerance.study.jsonl``.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import hpfem
from hpfem import units
from hpfem.opt import (
    Configuration,
    Normal,
    Study,
    build_global_surrogate,
    linear_propagation,
    monte_carlo,
    sobol_indices,
)

NM = units.nm


def _reconstruction():
    """The model helpers of examples/grating_reconstruction/run.py."""
    path = Path(__file__).resolve().parent.parent / "grating_reconstruction" / "run.py"
    spec = importlib.util.spec_from_file_location("grating_reconstruction_model", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODEL = _reconstruction()
NOMINAL = MODEL.TRUE  # CD 100 nm, height 120 nm, angle 86 degrees
TOLERANCES = {"cd": 2 * NM, "height": 3 * NM, "angle": math.radians(0.5)}


@dataclass
class Result:
    observables: list
    nominal: list
    linear_std: list
    mc_mean: list
    mc_std: list
    mc_band: list
    """2.5 and 97.5 % quantiles per observable"""
    linear_share: list
    """(observables, parameters) shares of the linearised variance"""
    sobol_first: list
    sobol_total: list
    surrogate_points: int
    surrogate_max_relative_std: float
    evaluations: int
    seconds: float
    parameters: list


def run(quick: bool = False, seed: int = 3, out: str | Path = ".") -> Result:
    t0 = time.perf_counter()
    out = Path(out)
    wavelengths = [450, 600] if quick else [400, 450, 500, 550, 600, 650, 700]
    configurations = [
        Configuration(w * NM, MODEL.THETA, 0.0, pol, (("R", 0),))
        for w in wavelengths for pol in ("s", "p")
    ]  # fmt: skip
    order = 2 if quick else 3
    store = out / "fabrication_tolerance.study.jsonl"
    if store.exists():
        store.unlink()
    study = Study(MODEL.evaluator(NOMINAL, configurations, order), store)
    inputs = {name: Normal(NOMINAL[name], TOLERANCES[name]) for name in ("cd", "height", "angle")}
    linear = linear_propagation(study, inputs)
    points, active = (8, 0) if quick else (16, 8)
    surrogate = build_global_surrogate(study, inputs, points=points, active=active, seed=seed)
    mc = monte_carlo(surrogate, inputs, samples=20000, seed=seed)
    sobol = sobol_indices(surrogate, inputs, samples=2**12 if quick else 2**13, seed=seed)
    result = Result(
        observables=list(linear.observables),
        nominal=linear.values.tolist(),
        linear_std=linear.std.tolist(),
        mc_mean=mc.mean.tolist(),
        mc_std=mc.std.tolist(),
        mc_band=np.column_stack([mc.quantiles[0.025], mc.quantiles[0.975]]).tolist(),
        linear_share=linear.contributions.tolist(),
        sobol_first=sobol.first.tolist(),
        sobol_total=sobol.total.tolist(),
        surrogate_points=surrogate.points,
        surrogate_max_relative_std=float(surrogate.max_relative_std),
        evaluations=len(study.history(None).index),
        seconds=time.perf_counter() - t0,
        parameters=list(inputs),
    )
    with open(out / "fabrication_tolerance.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(result), handle, indent=2)
    return result


def _report(r: Result) -> None:
    print(f"{'observable':<18} {'nominal':>8} {'lin std':>8} {'MC std':>8} {'2.5 %':>8} "
          f"{'97.5 %':>8}  dominant (S_T)")  # fmt: skip
    for i, name in enumerate(r.observables):
        total = r.sobol_total[i]
        j = int(np.argmax(total))
        print(f"{name:<18} {r.nominal[i]:8.4f} {r.linear_std[i]:8.4f} {r.mc_std[i]:8.4f} "
              f"{r.mc_band[i][0]:8.4f} {r.mc_band[i][1]:8.4f}  {r.parameters[j]} "
              f"{total[j]:.2f}")  # fmt: skip
    print(f"surrogate {r.surrogate_points} points (max relative std "
          f"{r.surrogate_max_relative_std:.2g}), {r.evaluations} evaluations, "
          f"{r.seconds:.1f} s")  # fmt: skip


if __name__ == "__main__":
    hpfem.set_log_level("warn")
    _report(run(quick="--quick" in sys.argv))

"""Scatterometry: reconstruction of a silicon line grating with uncertainties (M16 S3).

Spectroscopic scatterometry measures the specular reflectance of a periodic structure over
wavelength and polarisation and infers its geometry by fitting a rigorous model. This example
does that end to end on synthetic data: a silicon line grating on silicon (period 300 nm,
mid-height width CD 100 nm, height 120 nm, side-wall angle 86 degrees) is "measured" as R0
for s and p polarisation at 65 degrees incidence over 400-700 nm, with Gaussian noise of
0.002 in reflectance. The data are computed one polynomial order higher than the model, on a
mesh built at the true geometry, so the fit sees a model error as in practice (no "inverse
crime"). The model morphs a reference mesh built at the start values (hpfem.opt.Morph) and
evaluates the efficiencies with their Jacobian along the morph velocities
(hpfem.opt.GratingEvaluator: one factorisation per wavelength and polarisation, the Jacobian
from the kept factors). Levenberg-Marquardt (hpfem.opt.fit) recovers CD, height and
side-wall angle; the Laplace approximation (J^T W J)^-1 at the optimum gives their standard
errors and correlations. With ``--posterior`` the posterior is sampled as well (M16 S6,
hpfem.opt.sample with the optional extra opt-mcmc): a gradient-enhanced surrogate of the 14
reflectances around the optimum, MCMC on it, and the comparison with the Laplace Gaussian.

Silicon from M. A. Green, Sol. Energy Mater. Sol. Cells 92, 1305 (2008) (hpfem.materials).
Run ``python examples/grating_reconstruction/run.py [--quick] [--posterior]``; results go to
``grating_reconstruction.json``, the study to ``grating_reconstruction.study.jsonl``.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

import hpfem
from hpfem import materials, units
from hpfem.meshing import Shape, Slab, UnitCell
from hpfem.opt import Configuration, GratingEvaluator, Morph, fit, sample, trapezoid_parameters

NM = units.nm
PERIOD = 300 * NM
TRUE = {"cd": 100 * NM, "height": 120 * NM, "angle": math.radians(86.0)}
START = {"cd": 112 * NM, "height": 108 * NM, "angle": math.radians(88.5)}
BOUNDS = {"cd": (70 * NM, 130 * NM), "height": (90 * NM, 150 * NM),
          "angle": (math.radians(78.0), math.radians(90.0))}  # fmt: skip
NOISE = 0.002  # standard deviation of the measured reflectance
THETA = 65 * units.deg
SUB, LINE, AIR = 2, 3, 1
ROW = 15 * NM  # row height of the structured mesh
AIR_ROWS, SUB_ROWS, PML_TOP_ROWS, PML_BOTTOM_ROWS = 10, 6, 20, 10


@dataclass
class Result:
    true: dict
    start: dict
    estimate: dict
    std: dict
    deviation_in_std: dict
    correlation: list
    chi2_red: float
    model_error: float
    """largest |R(order p) - R(order p + 1)| at the true geometry (the discretisation error
    the fit sees, compare with NOISE)"""
    iterations: int
    evaluations: int
    num_dofs: int
    wavelengths_nm: list
    seconds: float
    observables: list = field(default_factory=list)
    posterior: dict | None = None
    """with ``--posterior``: per parameter the posterior mean and std, the shift of the mean
    in Laplace standard errors and the std ratio; the surrogate error in noise standard
    deviations, the acceptance, the autocorrelation time and the number of surrogate points"""


def unit_cell(geometry: dict, height_reference: float) -> UnitCell:
    """The cell with the line at ``geometry`` (rows laid out for ``height_reference``)."""
    h = geometry["height"]
    half = h / math.tan(geometry["angle"]) / 2  # half of bottom - top
    shape = Shape("trapezoid", LINE, {"x": 0.0, "y": 0.0, "bottom": geometry["cd"] + 2 * half,
                                      "top": geometry["cd"] - 2 * half, "height": h})  # fmt: skip
    y0 = -(SUB_ROWS + PML_BOTTOM_ROWS) * ROW
    y1 = height_reference + (AIR_ROWS + PML_TOP_ROWS) * ROW
    return UnitCell(PERIOD, y0, y1, slabs=[Slab(SUB, y0, 0.0)], shapes=[shape], background_tag=AIR)


def line_mesh(cell: UnitCell, nx: int = 20, line_rows: int = 8):
    """Structured mesh whose columns |x| <= q bend onto the slanted walls of the line, the line
    on ``line_rows`` rows, the air rows stretched to keep the top of the cell fixed."""
    p = cell.shapes[0].params
    h = p["height"]
    y_bottom, y_top = cell.y_bottom, cell.y_top
    rows = [(y_bottom, 0.0, SUB_ROWS + PML_BOTTOM_ROWS), (0.0, h, line_rows),
            (h, y_top, AIR_ROWS + PML_TOP_ROWS)]  # fmt: skip
    ys = [y_bottom]
    for a, b, n in rows:
        ys += [a + (b - a) * k / n for k in range(1, n + 1)]
    xs = [-PERIOD / 2 + PERIOD * i / nx for i in range(nx + 1)]
    q = xs[nx // 2 + nx // 6]  # the column line that becomes the wall (about P/6)
    vertices = []
    for y in ys:
        for x in xs:
            if 0.0 <= y <= h:
                w = 0.5 * p["bottom"] + 0.5 * (p["top"] - p["bottom"]) * y / h
                if abs(x) <= q:
                    x = x * w / q
                else:
                    x = math.copysign(w + (abs(x) - q) * (PERIOD / 2 - w) / (PERIOD / 2 - q), x)
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
        tags.append(SUB if cy < 0 else (LINE if 0 < cy < h and _inside(p, cx, cy) else AIR))
    mesh = hpfem.Mesh2D(vertices, cells, tags)
    tol = 1e-9 * PERIOD
    for f in mesh.boundary_facets:
        a, b = (mesh.vertex(int(v)) for v in mesh.facet_vertices(f))
        mx, my = 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])
        if abs(mx + PERIOD / 2) < tol:
            mesh.set_facet_tag(f, hpfem.box_tag.X_MIN)
        elif abs(mx - PERIOD / 2) < tol:
            mesh.set_facet_tag(f, hpfem.box_tag.X_MAX)
        elif abs(my - y_bottom) < tol:
            mesh.set_facet_tag(f, hpfem.box_tag.Y_MIN)
        elif abs(my - y_top) < tol:
            mesh.set_facet_tag(f, hpfem.box_tag.Y_MAX)
    return mesh


def _inside(p: dict, x: float, y: float) -> bool:
    w = 0.5 * p["bottom"] + 0.5 * (p["top"] - p["bottom"]) * y / p["height"]
    return abs(x) < w


def stack_at(omega: float):
    """Air over silicon at the angular frequency ``omega``."""
    return hpfem.LayerStack2D(hpfem.Material.dielectric(1.0), [],
                              materials.get("Si").at(omega), 0.0)  # fmt: skip


def evaluator(geometry: dict, configurations, order: int):
    """The model: the reference mesh at ``geometry``, morphed by CD, height and angle."""
    cell = unit_cell(geometry, geometry["height"])
    mesh = line_mesh(cell)
    # the cover line lies midway between the line top and the top PML, on a mesh row: the
    # morph keeps the cells there, the PML and the substrate PML fixed (one row inside)
    band = (-(SUB_ROWS - 1) * ROW, geometry["height"] + (AIR_ROWS // 2 - 1) * ROW)
    parameters = trapezoid_parameters(0, cd=BOUNDS["cd"], height=BOUNDS["height"],
                                      angle=BOUNDS["angle"])  # fmt: skip
    morph = Morph(cell, mesh, parameters, band=band)
    pml = {"top": PML_TOP_ROWS * ROW, "bottom": PML_BOTTOM_ROWS * ROW}
    return GratingEvaluator(morph, {SUB: materials.get("Si"), LINE: materials.get("Si")},
                            stack_at, configurations, order=order, pml=pml,
                            solve_options={"orders_max": 1})  # fmt: skip


def run(
    quick: bool = False, seed: int = 7, out: str | Path = ".", posterior: bool = False
) -> Result:
    t0 = time.perf_counter()
    out = Path(out)
    wavelengths = [400, 550, 700] if quick else [400, 450, 500, 550, 600, 650, 700]
    configurations = [
        Configuration(w * NM, THETA, 0.0, pol, (("R", 0),))
        for w in wavelengths for pol in ("s", "p")
    ]  # fmt: skip
    order = 2 if quick else 3
    # synthetic measurement: one order higher, mesh built at the true geometry, plus noise
    truth = evaluator(TRUE, configurations, order + 1)
    clean = truth(TRUE).values
    model = evaluator(TRUE, configurations, order)(TRUE).values
    rng = np.random.default_rng(seed)
    measured = clean + NOISE * rng.standard_normal(len(clean))
    # the fit on the morphed reference mesh of the start geometry
    store = out / "grating_reconstruction.study.jsonl"
    if store.exists():
        store.unlink()
    ev = evaluator(START, configurations, order)
    result = fit(ev, measured, NOISE, x0=START, path=store)
    names = ["cd", "height", "angle"]
    estimate = dict(zip(names, (float(v) for v in result.x), strict=True))
    std = dict(zip(names, (float(s) for s in result.std), strict=True))
    out_result = Result(
        true=TRUE,
        start=START,
        estimate=estimate,
        std=std,
        deviation_in_std={n: (estimate[n] - TRUE[n]) / std[n] for n in names},
        correlation=np.asarray(result.correlation).tolist(),
        chi2_red=float(result.chi2_red),
        model_error=float(np.abs(model - clean).max()),
        iterations=int(result.iterations),
        evaluations=int(result.new_evaluations),
        num_dofs=int(result.evaluation.meta["dofs"]),
        wavelengths_nm=wavelengths,
        seconds=time.perf_counter() - t0,
        observables=list(ev.observables),
    )
    if posterior:
        post = sample(result, walkers=24, steps=3000, seed=seed)
        out_result.posterior = {
            "parameters": {r["name"]: {k: r[k] for k in ("mean", "std", "shift", "ratio")}
                           for r in post.compare()},
            "surrogate_error": float(post.surrogate.validation),
            "surrogate_points": int(post.surrogate.points),
            "acceptance": float(post.acceptance),
            "autocorr": float(np.nanmax(post.autocorr)),
        }  # fmt: skip
        out_result.seconds = time.perf_counter() - t0
    with open(out / "grating_reconstruction.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(out_result), handle, indent=2)
    return out_result


def _report(r: Result) -> None:
    scale = {"cd": (NM, "nm"), "height": (NM, "nm"), "angle": (units.deg, "deg")}
    print(f"{'':8} {'true':>9} {'start':>9} {'estimate':>10} {'std':>8} {'dev/std':>8}")
    for n, (s, unit) in scale.items():
        print(f"{n:8} {r.true[n] / s:9.3f} {r.start[n] / s:9.3f} {r.estimate[n] / s:10.3f} "
              f"{r.std[n] / s:8.3f} {r.deviation_in_std[n]:8.2f}  {unit}")  # fmt: skip
    print(f"chi2_red {r.chi2_red:.3f}, model error {r.model_error:.2e} (noise {NOISE}), "
          f"{r.iterations} iterations, {r.evaluations} evaluations, {r.num_dofs} DoFs, "
          f"{r.seconds:.1f} s")  # fmt: skip
    print("correlation (cd, height, angle):")
    for row in r.correlation:
        print("  " + " ".join(f"{c:7.3f}" for c in row))
    if r.posterior:
        print(f"{'':8} {'mean':>10} {'std':>8} {'shift':>7} {'ratio':>6}   (posterior, MCMC)")
        for n, (s, unit) in scale.items():
            p = r.posterior["parameters"][n]
            print(f"{n:8} {p['mean'] / s:10.3f} {p['std'] / s:8.3f} {p['shift']:7.2f} "
                  f"{p['ratio']:6.2f}  {unit}")  # fmt: skip
        post = r.posterior
        print(f"surrogate: {post['surrogate_points']} points, error "
              f"{post['surrogate_error']:.2g} noise std; acceptance {post['acceptance']:.2f}, "
              f"autocorrelation {post['autocorr']:.3g} steps")  # fmt: skip


if __name__ == "__main__":
    hpfem.set_log_level("warn")
    _report(run(quick="--quick" in sys.argv, posterior="--posterior" in sys.argv))

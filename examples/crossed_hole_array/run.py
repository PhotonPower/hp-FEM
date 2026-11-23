"""Hole array in a glass film on glass: a FourierGrating (RCWA) model recomputed with hp-FEM.

The crossed grating of ``hole_array_glass_r125.json`` (FourierGrating's configuration format:
period 0.5 µm in x and z, air holes of radius 0.125 µm in a 0.15 µm glass layer on glass,
λ = 0.405 µm, normal incidence, p polarisation) is imported with ``hpfem.fouriergrating``,
solved with ``hpfem.crossed`` for a sequence of polynomial orders on one mesh, and compared
order by order with FourierGrating's RCWA (Nx = Nz = 20, ``reference_rcwa_N20.csv``).

With ``--fg-nearfield DIR`` (the output directory of a FourierGrating run of the same model with
``"nearfield"`` enabled, containing ``nearfield_horizontal`` / ``nearfield_vertical``) the
near fields are compared on FourierGrating's cuts as well and plotted (needs matplotlib).

Run ``python examples/crossed_hole_array/run.py [--quick] [--fg-nearfield DIR]``; results go to
``crossed_hole_array.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import hpfem
from hpfem import fouriergrating as fgi

HERE = Path(__file__).resolve().parent
MODEL = HERE / "hole_array_glass_r125.json"
REFERENCE = HERE / "reference_rcwa_N20.csv"
NEARFIELD = HERE / "hole_array_glass_r125_nearfield.json"  # the FourierGrating run of the cuts


def compare_orders(results, reference: dict) -> dict:
    """Efficiencies of hp-FEM against the reference, keyed like FourierGrating's CSV."""
    rows = {
        (r["lambda_um"], r["phi_deg"], r["psi_deg"], r["delta_deg"], r["side"], r["n"], r["m"]): r[
            "efficiency"
        ]
        for r in fgi.efficiency_rows(results)
    }
    table = []
    for key, value in sorted(reference.items(), key=lambda kv: (kv[0][4], kv[0][5], kv[0][6])):
        table.append(
            {"side": key[4], "n": key[5], "m": key[6], "rcwa": value, "hpfem": rows.get(key, 0.0)}
        )
    return {
        "orders": table,
        "max_deviation": max(abs(t["hpfem"] - t["rcwa"]) for t in table),
        "sum_hpfem": float(sum(rows.values())),
    }


def compare_near_fields(model, result, fg_dir: Path, order: int, plot_dir: Path | None) -> dict:
    """Relative L2 deviation of E on FourierGrating's cuts (rows on the layer interfaces
    excluded: one-sided values), overall and away from the material edges."""
    out = {}
    for cut in ("horizontal", "vertical"):
        d = fg_dir / f"nearfield_{cut}"
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        a1 = np.array(meta["axes"]["axis1"]["values_um"])
        a2 = np.array(meta["axes"]["axis2"]["values_um"])
        cuts = json.loads(NEARFIELD.read_text(encoding="utf-8"))["nearfield"]
        fixed = cuts["horizontal_y_um"] if cut == "horizontal" else cuts["vertical_z_um"]
        A2, A1 = np.meshgrid(a2, a1, indexing="ij")  # FourierGrating stores [axis2, axis1]
        if cut == "horizontal":
            x, z, y = A1, A2, np.full_like(A1, fixed)
        else:
            x, y, z = A1, A2, np.full_like(A1, fixed)
        fg_e = np.stack([np.load(d / f"E{c}.npy") for c in "xyz"], axis=-1)
        points = model.fg_to_hpfem(np.column_stack([x.ravel(), y.ravel(), z.ravel()]))
        hp_e = model.hpfem_to_fg_field(result.field(points)).reshape(fg_e.shape)
        keep = ~(np.isclose(y, 0.0) | np.isclose(y, model.thickness / 1e-6))
        r = np.hypot(x - 0.25, z - 0.25) if cut == "horizontal" else np.abs(x - 0.25)
        far = keep & (np.abs(r - 0.125) > 0.02)
        if cut == "vertical":
            far &= (np.abs(y) > 0.015) & (np.abs(y - 0.15) > 0.015)

        out[cut] = {
            "relative_l2": _relative(fg_e, hp_e, keep),
            "relative_l2_away_from_edges": _relative(fg_e, hp_e, far),
        }
        if plot_dir is not None:
            _plot(cut, a1, a2, fg_e, hp_e, order, plot_dir / f"crossed_hole_array_{cut}.png")
    return out


def _relative(reference, value, mask) -> float:
    return float(np.linalg.norm(reference[mask] - value[mask]) / np.linalg.norm(reference[mask]))


def _plot(cut, a1, a2, fg_e, hp_e, order, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e2f = (np.abs(fg_e) ** 2).sum(-1)
    e2h = (np.abs(hp_e) ** 2).sum(-1)
    vmax = max(e2f.max(), e2h.max())
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    extent = [a1[0], a1[-1], a2[0], a2[-1]]
    label = "z" if cut == "horizontal" else "y"
    for a, data, title in (
        (ax[0], e2f, "RCWA (FourierGrating, N = 20) |E|²"),
        (ax[1], e2h, f"hp-FEM p = {order} |E|²"),
        (ax[2], np.abs(e2f - e2h), "| difference |"),
    ):
        image = a.imshow(
            data, origin="lower", extent=extent, aspect="auto", vmax=None if a is ax[2] else vmax
        )
        a.set_title(title)
        a.set_xlabel("x [µm]")
        a.set_ylabel(f"{label} [µm]")
        fig.colorbar(image, ax=a)
    where = "y = 0.075 µm, mid-layer" if cut == "horizontal" else "z = 0.25 µm, through the hole"
    fig.suptitle(f"Hole array, {cut} cut ({where}), p polarisation, 405 nm (FourierGrating frame)")
    fig.savefig(path, dpi=90)
    plt.close(fig)


def run(
    quick: bool = False,
    fg_nearfield: str | None = None,
    out: str | None = None,
    plot_dir: str | None = None,
) -> dict:
    model = fgi.load(MODEL)
    reference = fgi.read_efficiencies(REFERENCE)
    orders, density = ((2,), 3.0) if quick else ((3, 4), 2.5)
    record = {"model": MODEL.name, "elements_per_wavelength": density, "steps": []}
    print(
        f"{'p':>3} {'unknowns':>9} {'time':>7} {'R(0,0)':>9} {'T(0,0)':>9} {'max |dη|':>9} "
        f"{'balance':>9}"
    )
    for p in orders:
        start = time.perf_counter()
        results = fgi.solve(model, order=p, elements_per_wavelength=density)
        seconds = time.perf_counter() - start
        incidence, result = results[0]
        cmp = compare_orders(results, reference)
        step = {
            "order": p,
            "unknowns": result.num_dofs,
            "seconds": seconds,
            "balance": result.balance,
            **cmp,
        }
        if fg_nearfield:
            step["near_field"] = compare_near_fields(
                model, result, Path(fg_nearfield), p, Path(plot_dir) if plot_dir else None
            )
        record["steps"].append(step)
        r00 = next(t["hpfem"] for t in cmp["orders"] if (t["side"], t["n"], t["m"]) == ("R", 0, 0))
        t00 = next(t["hpfem"] for t in cmp["orders"] if (t["side"], t["n"], t["m"]) == ("T", 0, 0))
        print(
            f"{p:>3} {result.num_dofs:>9} {seconds:>6.0f}s {r00:>9.6f} {t00:>9.6f} "
            f"{cmp['max_deviation']:>9.1e} {result.balance:>9.1e}"
        )
        if "near_field" in step:
            for cut, v in step["near_field"].items():
                print(
                    f"    near field {cut}: relative L2 {v['relative_l2']:.1e} "
                    f"({v['relative_l2_away_from_edges']:.1e} away from the material edges)"
                )
    print("orders (RCWA | hp-FEM, last order):")
    for t in record["steps"][-1]["orders"]:
        print(f"  {t['side']} ({t['n']:+d},{t['m']:+d})  {t['rcwa']:.6f}  {t['hpfem']:.6f}")
    if out:
        Path(out).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="p = 2 on a coarse mesh (~15 s)")
    parser.add_argument("--fg-nearfield", help="FourierGrating near-field output directory")
    parser.add_argument("--plots", help="directory for the near-field plots")
    args = parser.parse_args(argv)
    hpfem.set_log_level("warn")
    run(args.quick, args.fg_nearfield, "crossed_hole_array.json", args.plots)
    return 0


if __name__ == "__main__":
    sys.exit(main())

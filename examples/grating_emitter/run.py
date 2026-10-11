"""Quantum dot above a dielectric grating: emission, extraction and Purcell factor (M17).

A quantum dot (a Gaussian dipole, isotropic: the mean of the three orientations) sits 50 nm above
the middle of a glass ridge of a glass grating on a glass substrate, in air. The grating is
periodic in x and invariant along the lines (z); the single emitter is not periodic, so its
emission is the array scanning of the cell problem over the Bloch wavenumber kx and the
wavenumber β along the lines (`hpfem.grating.dipole_emission`, ADR-0013, docs/theory/maxwell.md):

- the **Purcell factor** F_P = P_em / P_bulk (the emitted power over that of the same dipole in
  air);
- the **channels**: the fractions of the emitted power radiated up into the air and down into the
  glass, by reciprocity integrated over the half-spaces, and the remainder (absorbed or guided;
  here neither exists, so it measures the accuracy of the quadratures);
- the **extraction into a numerical aperture**: the power an objective of NA 0.5 above the sample
  collects (`hpfem.grating.emission_cone`), the LED / single-photon-source figure of merit.

The structure has no high-index layer and therefore no guided modes (their poles on the real kx
axis are not treated yet, ADR-0013 §4). Everything runs through the job runner task ``emitter``
(`hpfem.run.run_job`): first a ``dry_run`` with the cost estimate, then the sweep, for the grating
and for the flat glass surface (the same dipole 200 nm above plain glass) as the reference.

Run `python examples/grating_emitter/run.py [--quick]`; results go to `grating_emitter.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import hpfem
from hpfem.run import run_job

TAG_AIR, TAG_GLASS = 1, 2


def job(quick: bool, grating: bool = True, dry_run: bool = False) -> dict:
    """The job document (lengths in nm) of the grating or of the flat reference."""
    period, ridge_width, ridge_height = 500.0, 250.0, 150.0
    y_bottom, y_top, pml = -1500.0, 1700.0, 600.0
    h = 50.0 if quick else 25.0
    rows = [
        [y_bottom, 0.0, round(-y_bottom / (2 * h))],  # coarser in the glass
        [0.0, ridge_height, round(ridge_height / h)],
        [ridge_height, 400.0, round((400.0 - ridge_height) / h)],  # the dipole region
        [400.0, y_top, round((y_top - 400.0) / (2 * h))],
    ]
    shapes = []
    if grating:
        shapes = [{"kind": "rectangle", "tag": TAG_GLASS,
                   "params": {"x": -ridge_width / 2, "y": 0.0, "width": ridge_width,
                              "height": ridge_height}}]  # fmt: skip
    scan = ({"nodes": 3, "kx_nodes": 2, "depth": 0.5, "angle_nodes": [4, 8], "symmetric": True}
            if quick else
            {"nodes": 6, "kx_nodes": 4, "depth": 0.5, "angle_nodes": [8, 16],
             "symmetric": True})  # fmt: skip
    wavelengths = [650.0] if quick else {"start": 600.0, "stop": 800.0, "count": 3}
    return {
        "version": 2, "task": "emitter", "name": "grating" if grating else "flat", "unit": 1e-9,
        "model": {"period": period, "y_bottom": y_bottom, "y_top": y_top,
                  "background_tag": TAG_AIR,
                  "slabs": [{"tag": TAG_GLASS, "y0": y_bottom, "y1": 0.0}], "shapes": shapes},
        "mesh": {"structured": {"nx": round(period / h), "rows": rows}},
        "materials": {str(TAG_AIR): "air", str(TAG_GLASS): {"n": [1.5, 0.0]}},
        "stack": {"incidence": "air", "substrate": {"n": [1.5, 0.0]}, "layers": [], "top": 0},
        "solver": {"order": 2 if quick else 3, "pml": {"top": pml, "bottom": pml}},
        "emitter": {"position": [0.0, 200.0], "sigma": 30.0, "moment": "isotropic",
                    "stage": "B", "wavelength": wavelengths, "scan": scan, "aperture": [0.5],
                    "calibrate": True, "dry_run": dry_run},
    }  # fmt: skip


def summary(points) -> list[dict]:
    rows = []
    for p in points:
        f = p.get("fractions", {})
        rows.append({"wavelength_nm": p["wavelength"] * 1e9, "purcell": p["purcell"],
                     "up": f.get("up"), "down": f.get("down"),
                     "nonradiated": f.get("nonradiated"),
                     "na_0p5": p["aperture"][0]["fraction"] if p["aperture"] else None,
                     "samples": p["samples"], "seconds": p["seconds"]})  # fmt: skip
    return rows


def run(quick: bool = False, out: str | None = "grating_emitter.json", verbose=True) -> dict:
    hpfem.set_log_level("warn")

    def show(event):
        if verbose and event["event"] in ("emission_cost", "point", "cancelled"):
            text = {k: v for k, v in event.items() if k not in ("timing", "purcell_by_orientation")}
            print(json.dumps(text), flush=True)

    cost = run_job(job(quick, dry_run=True), emit=show)["cost"]
    result = {"quick": bool(quick), "cost": cost}
    for name, grating in (("grating", True), ("flat", False)):
        points = run_job(job(quick, grating=grating), emit=show)["points"]
        result[name] = summary(points)
    if out:
        Path(out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    if verbose:
        print(f"{'wl [nm]':>7} {'F_P grating':>12} {'F_P flat':>9} {'up grating':>11} "
              f"{'up flat':>8} {'NA0.5 grating':>14} {'NA0.5 flat':>11}")  # fmt: skip
        for g, f in zip(result["grating"], result["flat"], strict=True):
            print(f"{g['wavelength_nm']:7.0f} {g['purcell']:12.3f} {f['purcell']:9.3f} "
                  f"{g['up']:11.3f} {f['up']:8.3f} {g['na_0p5']:14.3f} "
                  f"{f['na_0p5']:11.3f}")  # fmt: skip
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="coarse mesh, p = 2, small rules")
    parser.add_argument("--out", default="grating_emitter.json")
    args = parser.parse_args(argv)
    run(args.quick, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

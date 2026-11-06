"""M19 S7e-2b: a single Gaussian dipole above a lossless tilted crystal slab that guides modes, by
array scanning with the guided-mode poles (``grating.dipole_emission``, crystals: β over
(−β_max, β_max), the full kx zone, the poles detected and bumped per β node, no β pre-pass for
the pinch points) against the 2D-spectrum reference on the deformed path
(``hpfem.planar.dipole_above_stack(depth=...)``, exact for the Gaussian emitter). A long local
run; the result goes to ``benchmarks/results/<date>-validation-m19-crystal-guided.json``.

    PYTHONPATH=python python benchmarks/m19_crystal_guided_scan.py [--workers 6] [--nodes 6]
"""

import argparse
import datetime
import json
import math
import pathlib
import time

import numpy as np

import hpfem
from hpfem import grating, units
from hpfem.planar import dipole_above_stack

UM = 1e-6
OMEGA = units.angular_frequency(wavelength=UM)
K0 = 2 * math.pi / UM
PERIOD, D, HEIGHT, SIGMA = 1.0 * UM, 0.25 * UM, 0.25 * UM, 0.04 * UM
AXIS = np.array([0.8, 0.0, 0.6])  # laboratory frame, z the normal
EPS = 3.2 * np.eye(3) + 0.9 * np.outer(AXIS, AXIS)
SWAP = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0.0]])  # laboratory <-> solver frame
MOMENTS = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1], "moment": [0.6, 0.5, 0.4]}


def cell(h=0.05 * UM):
    ny = round(4 * UM / h)
    mesh = hpfem.rectangle(round(PERIOD / h / 2), ny, [-PERIOD / 2, -2 * UM],
                           [PERIOD / 2, 2 * UM])  # fmt: skip
    for c in range(mesh.num_cells):
        y = np.mean([mesh.vertex(int(v))[1] for v in mesh.cell_vertices(c)])
        if -D < y < 0:
            mesh.set_cell_tag(c, 4)
        elif y < -D:
            mesh.set_cell_tag(c, 5)
    return mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--nodes", type=int, default=6)
    parser.add_argument("--kx-nodes", type=int, default=4)
    parser.add_argument("--order", type=int, default=4)
    args = parser.parse_args()
    crystal = hpfem.Material.anisotropic(EPS)
    glass = hpfem.Material.dielectric(1.45)
    stack = hpfem.AnisotropicStack(hpfem.Material.dielectric(1.0), [hpfem.Layer(crystal, D)],
                                   glass)  # fmt: skip
    references = {
        key: dipole_above_stack(
            stack, K0, HEIGHT, SWAP @ np.array(m, float), azimuths=48, depth=0.03, points=800
        )  # fmt: skip
        for key, m in MOMENTS.items()
    }
    print("reference", {k: round(v, 8) for k, v in references.items()}, flush=True)
    t0 = time.perf_counter()

    def progress(i, n):
        if i % 200 == 0:
            print(f"  {i} / {n} cell problems, {time.perf_counter() - t0:.0f} s", flush=True)

    result = grating.dipole_emission(
        cell(), {4: crystal, 5: glass}, stack,
        {"position": (0.05 * UM, HEIGHT), "sigma": SIGMA, "moment": MOMENTS["moment"]}, OMEGA,
        nodes=args.nodes, kx_nodes=args.kx_nodes, order=args.order,
        pml={"top": UM, "bottom": UM}, channels=False, guided=True, workers=args.workers,
        progress=progress,
    )  # fmt: skip
    seconds = time.perf_counter() - t0
    rows = {}
    for key in MOMENTS:
        fem = float(result.purcell[key])
        rows[key] = {"fem": fem, "reference": references[key],
                     "relative_error": fem / references[key] - 1}  # fmt: skip
        print(f"{key:>7} F = {fem:.8f}  reference {references[key]:.8f}  "
              f"error {fem / references[key] - 1:+.2e}", flush=True)  # fmt: skip
    print(f"{result.samples} samples, {len(result.guided_modes or [])} guided-pole bumps, "
          f"{seconds:.0f} s")  # fmt: skip
    out = {
        "benchmark": "M19 S7e-2b dipole above a guiding tilted crystal slab (array scan)",
        "date": datetime.date.today().isoformat(),
        "nodes": args.nodes, "kx_nodes": args.kx_nodes, "order": args.order,
        "workers": args.workers, "samples": int(result.samples), "seconds": seconds,
        "purcell": rows,
    }  # fmt: skip
    path = pathlib.Path(__file__).parent / "results"
    path = path / f"{out['date']}-validation-m19-crystal-guided.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

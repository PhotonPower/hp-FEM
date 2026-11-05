"""M19 S7c: a single Gaussian dipole above a tilted lossless crystal half-space by array scanning
(``grating.dipole_emission``, β over (−β_max, β_max), the full kx zone, crystal light lines)
against the 2D plane-wave-spectrum reference ``hpfem.planar.dipole_above_stack`` (exact for the
Gaussian emitter). A long local run (minutes with 4 workers); the result goes to
``benchmarks/results/<date>-validation-m19-crystal-scan.json``.

    PYTHONPATH=python python benchmarks/m19_crystal_dipole_scan.py [--workers 4] [--nodes 6]
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
PERIOD, HEIGHT, SIGMA = 1.0 * UM, 0.32 * UM, 0.05 * UM
AXIS = np.array([math.sin(0.7), 0.3, math.cos(0.7)])  # laboratory frame, z the normal
EPS = 2.4 * np.eye(3) + np.outer(AXIS, AXIS) / 1.09
SWAP = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0.0]])  # laboratory <-> solver frame
MOMENTS = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1], "moment": [0.6, 0.5, 0.4]}


def cell(h=0.1 * UM):
    ny = round(4 * UM / h)
    mesh = hpfem.rectangle(round(PERIOD / h), ny, [-PERIOD / 2, -2 * UM], [PERIOD / 2, 2 * UM])
    for c in range(mesh.num_cells):
        y = np.mean([mesh.vertex(int(v))[1] for v in mesh.cell_vertices(c)])
        if y < 0:
            mesh.set_cell_tag(c, 5)
    return mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--nodes", type=int, default=6)
    parser.add_argument("--kx-nodes", type=int, default=4)
    parser.add_argument("--order", type=int, default=4)
    args = parser.parse_args()
    crystal = hpfem.Material.anisotropic(EPS)
    stack = hpfem.AnisotropicStack(hpfem.Material.dielectric(1.0), [], crystal)
    t0 = time.perf_counter()
    result = grating.dipole_emission(
        cell(), {5: crystal}, stack,
        {"position": (0.05 * UM, HEIGHT), "sigma": SIGMA, "moment": MOMENTS["moment"]}, OMEGA,
        nodes=args.nodes, kx_nodes=args.kx_nodes, order=args.order,
        pml={"top": UM, "bottom": UM}, channels=False, workers=args.workers,
    )  # fmt: skip
    seconds = time.perf_counter() - t0
    rows = {}
    for key, m in MOMENTS.items():
        reference = dipole_above_stack(stack, K0, HEIGHT, SWAP @ np.array(m, float), azimuths=96)
        fem = float(result.purcell[key])
        rows[key] = {"fem": fem, "reference": reference, "relative_error": fem / reference - 1}
        print(
            f"{key:>7} F = {fem:.8f}  reference {reference:.8f}  error {fem / reference - 1:+.2e}"
        )
    print(f"{result.samples} samples, {seconds:.0f} s")
    out = {
        "benchmark": "M19 S7c dipole above a tilted crystal half-space (array scan)",
        "date": datetime.date.today().isoformat(),
        "nodes": args.nodes, "kx_nodes": args.kx_nodes, "order": args.order,
        "workers": args.workers, "samples": int(result.samples), "seconds": seconds,
        "purcell": rows,
    }  # fmt: skip
    path = (
        pathlib.Path(__file__).parent
        / "results"
        / (f"{out['date']}-validation-m19-crystal-scan.json")
    )
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

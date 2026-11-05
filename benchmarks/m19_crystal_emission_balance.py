"""M19 S7d: the power of a Gaussian dipole above a tilted lossless crystal layer on an isotropic
substrate (index 2, no guided modes), radiated into the cover and into the substrate by
reciprocity (``grating.emission_cone`` over the full half-spaces, 8×16 directions each), against
the total of the 2D-spectrum reference ``hpfem.planar.dipole_above_stack``: lossless and without
guided modes, up + down must equal the total. A local run (about 15 min).

    PYTHONPATH=python python benchmarks/m19_crystal_emission_balance.py
"""

import math
import time

import numpy as np

import hpfem
from hpfem import grating, units
from hpfem.planar import dipole_above_stack

UM = 1e-6
OMEGA = units.angular_frequency(wavelength=UM)
K0 = 2 * math.pi / UM
PERIOD, D, HEIGHT, SIGMA = 0.5 * UM, 0.2 * UM, 0.32 * UM, 0.05 * UM
AXIS = np.array([0.8, 0.0, 0.6])
EPS = 2.25 * np.eye(3) + 1.1 * np.outer(AXIS, AXIS)  # laboratory frame
SWAP = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0.0]])
MOMENT = np.array([0.6, 0.5, 0.4])  # solver frame


def main():
    crystal = hpfem.Material.anisotropic(EPS)
    substrate = hpfem.Material.dielectric(2.0)
    stack = hpfem.AnisotropicStack(hpfem.Material.dielectric(1.0), [hpfem.Layer(crystal, D)],
                                   substrate)  # fmt: skip
    mesh = hpfem.rectangle(5, 40, [-PERIOD / 2, -2 * UM], [PERIOD / 2, 2 * UM])
    for c in range(mesh.num_cells):
        y = np.mean([mesh.vertex(int(v))[1] for v in mesh.cell_vertices(c)])
        if -D < y < 0:
            mesh.set_cell_tag(c, 4)
        elif y < -D:
            mesh.set_cell_tag(c, 5)
    dipole = {"position": (0.05 * UM, HEIGHT), "moment": MOMENT, "sigma": SIGMA}
    t0 = time.perf_counter()
    total = 0.0
    for side, n in (("up", 1.0), ("down", 2.0)):
        cone = grating.emission_cone(mesh, {4: crystal, 5: substrate}, stack, dipole, OMEGA, n,
                                     side=side, angle_nodes=(8, 16), order=4,
                                     pml={"top": UM, "bottom": UM})  # fmt: skip
        total += cone.power["moment"]
        print(f"{side:>5}: {cone.power['moment']:.6e} W")
    ratio = total / grating.dipole_bulk_power(MOMENT, OMEGA, 1.0, SIGMA)
    reference = dipole_above_stack(stack, K0, HEIGHT, SWAP @ MOMENT, azimuths=96)
    print(f"(up + down) / P_bulk = {ratio:.8f}, reference F = {reference:.8f}, "
          f"error {ratio / reference - 1:+.2e}, {time.perf_counter() - t0:.0f} s")  # fmt: skip


if __name__ == "__main__":
    main()

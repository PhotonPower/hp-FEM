"""M19 S7e: the power of a Gaussian dipole above a lossless tilted crystal half-space radiated
into the cover (``grating.emission_cone``, isotropic side, 8×16 directions) and into the crystal
(``emission_cone(side="down")``: by reciprocity with the power-normalised crystal modes over the
in-plane wavevector, 8 points per panel and 16 azimuths), against the total of the 2D-spectrum
reference ``hpfem.planar.dipole_above_stack``: lossless and without guided modes, up + down must
equal the total. ``orders_max=0``: the flat stack excites no higher orders, so the check for
backward diffracted orders under the PML is not needed. A local run (about 20 min).

    PYTHONPATH=python python benchmarks/m19_crystal_substrate_balance.py
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
PERIOD, HEIGHT, SIGMA = 0.5 * UM, 0.32 * UM, 0.05 * UM
AXIS = np.array([math.sin(0.7), 0.3, math.cos(0.7)])
EPS = 2.4 * np.eye(3) + np.outer(AXIS, AXIS) / 1.09  # laboratory frame
SWAP = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0.0]])


def main():
    crystal = hpfem.Material.anisotropic(EPS)
    stack = hpfem.AnisotropicStack(hpfem.Material.dielectric(1.0), [], crystal)
    mesh = hpfem.rectangle(5, 40, [-PERIOD / 2, -2 * UM], [PERIOD / 2, 2 * UM])
    for c in range(mesh.num_cells):
        if np.mean([mesh.vertex(int(v))[1] for v in mesh.cell_vertices(c)]) < 0:
            mesh.set_cell_tag(c, 5)
    moment = np.array([0.6, 0.5, 0.4])  # solver frame
    dipole = {"position": (0.05 * UM, HEIGHT), "moment": moment, "sigma": SIGMA}
    common = dict(order=4, pml={"top": UM, "bottom": UM}, orders_max=0, angle_nodes=(8, 16))
    t0 = time.perf_counter()
    cones = {
        side: grating.emission_cone(
            mesh, {5: crystal}, stack, dipole, OMEGA, aperture, side=side, **common
        )  # fmt: skip
        for side, aperture in (("up", 1.0), ("down", 2.0))
    }
    for key, m in (("x", np.eye(3)[0]), ("y", np.eye(3)[1]), ("z", np.eye(3)[2]),
                   ("moment", moment)):  # fmt: skip
        total = cones["up"].power[key] + cones["down"].power[key]
        ratio = total / grating.dipole_bulk_power(m, OMEGA, 1.0, SIGMA)
        reference = dipole_above_stack(stack, K0, HEIGHT, SWAP @ m, azimuths=96)
        print(f"{key:>7}: (up + down)/P_bulk = {ratio:.8f}, F = {reference:.8f}, "
              f"error {ratio / reference - 1:+.2e}")  # fmt: skip
    print(f"{time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()

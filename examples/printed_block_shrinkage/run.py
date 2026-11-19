"""Shrinkage of a two-photon printed polymer block on glass (ADR-0016, M20).

A flat block of resin, printed on a glass substrate and bonded to it at its base, shrinks during
development and post-curing. The shrinkage s is an isotropic eigenstrain −s I; it follows from the
local degree of polymerisation, which a process model (Voxelspaetzle) delivers as a volume field.
Here a stand-in profile s(z) on a grid is used, rising from the base to the top (the top is less
cross-linked). The block is clamped at the base (the glass is rigid against the resin), its other
faces are free. One quarter is computed, with the symmetry planes x = 0 and y = 0.

Results: the displacement of the top face, the sag between the centre and the corner of the top
face (the shape error of a printed optical surface), the pull-in of the side walls at the top,
and the strain energy. Check inside the script (`check_layer`): with a uniform shrinkage the
centre of a wide block behaves like the laterally constrained layer, u_z = −s z (1 + ν)/(1 − ν).

Run `python examples/printed_block_shrinkage/run.py [--quick]`; results go to
`printed_block_shrinkage.json`, the top-face profile to `printed_block_shrinkage.png`.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass

import numpy as np

import hpfem
from hpfem import units

# resin: an assumed IP-Dip-like photoresist after development (stand-in values)
E_RESIN = 4.5e9  # Pa
NU_RESIN = 0.35
HALF_WIDTH = 20 * units.um  # the block is 40 µm × 40 µm
HEIGHT = 2 * units.um


@dataclass
class Result:
    order: int
    dofs: int
    shrinkage_base: float
    shrinkage_top: float
    uz_top_centre_nm: float
    uz_top_corner_nm: float
    sag_nm: float  # u_z(centre) − u_z(corner) of the top face; < 0: the centre sinks deeper
    pull_in_nm: float  # u_x at the middle of the side wall x = a, top edge (negative: inwards)
    strain_energy_pj: float
    profile_x_um: list
    profile_uz_nm: list


def solve_block(order: int = 2, n: int = 10, nz: int = 4, shrinkage_base: float = 0.02,
                shrinkage_top: float = 0.04) -> Result:  # fmt: skip
    """The quarter block [0, a] × [0, a] × [0, H] at polynomial order `order`."""
    mesh = hpfem.box(n, n, nz, [0.0, 0.0, 0.0], [HALF_WIDTH, HALF_WIDTH, HEIGHT])
    dofs = hpfem.DofMap3D(mesh, order)
    tag = hpfem.box_tag
    setup = hpfem.ElasticitySetup3D()
    setup.background = hpfem.ElasticMaterial(E_RESIN, NU_RESIN)
    setup.displacements = [
        hpfem.DisplacementCondition3D(tag.X_MIN, [True, False, False]),  # symmetry plane
        hpfem.DisplacementCondition3D(tag.Y_MIN, [False, True, False]),  # symmetry plane
        hpfem.DisplacementCondition3D(tag.Z_MIN, [True, True, True]),  # bonded to the glass
    ]
    # the shrinkage as a data port: a grid field, invariant in x and y, linear in z
    z = np.linspace(0.0, HEIGHT, 5)
    s = shrinkage_base + (shrinkage_top - shrinkage_base) * z / HEIGHT
    setup.set_shrinkage_grid(hpfem.GridField3D([[0.0], [0.0], list(z)], s))
    setup.static_condensation = True
    problem = hpfem.Elasticity3D(dofs, setup)
    u = problem.solve()
    xs = np.linspace(0.0, HALF_WIDTH, 21)
    top = np.column_stack([xs, np.zeros_like(xs), np.full_like(xs, HEIGHT)])
    uz = problem.displacements_at(u, top)[:, 2].real
    probes = np.array([[0.0, 0.0, HEIGHT], [HALF_WIDTH, HALF_WIDTH, HEIGHT],
                       [HALF_WIDTH, 0.0, HEIGHT]])  # fmt: skip
    d = problem.displacements_at(u, probes).real
    nm = 1e9
    return Result(
        order=order,
        dofs=problem.num_dofs,
        shrinkage_base=shrinkage_base,
        shrinkage_top=shrinkage_top,
        uz_top_centre_nm=float(d[0, 2] * nm),
        uz_top_corner_nm=float(d[1, 2] * nm),
        sag_nm=float((d[0, 2] - d[1, 2]) * nm),
        pull_in_nm=float(d[2, 0] * nm),
        strain_energy_pj=float(4 * problem.strain_energy(u) * 1e12),  # the whole block
        profile_x_um=(xs * 1e6).tolist(),
        profile_uz_nm=(uz * nm).tolist(),
    )


def check_layer(order: int = 2, n: int = 10, nz: int = 4, s: float = 0.03) -> tuple[float, float]:
    """Uniform shrinkage: u_z at the centre of the top face against the laterally constrained
    layer −s H (1 + ν)/(1 − ν) (the wide block is constrained by its base away from the walls)."""
    r = solve_block(order, n, nz, s, s)
    return r.uz_top_centre_nm, -s * HEIGHT * (1 + NU_RESIN) / (1 - NU_RESIN) * 1e9


def main(quick: bool = False) -> dict:
    hpfem.set_log_level("warn")
    order, n, nz = (2, 6, 4) if quick else (3, 6, 3)
    fem, layer = check_layer(order, n, nz)
    print(f"uniform shrinkage: u_z(top centre) = {fem:.2f} nm, constrained layer {layer:.2f} nm "
          f"({fem / layer - 1:+.2e})")  # fmt: skip
    result = solve_block(order, n, nz)
    print(f"graded shrinkage {result.shrinkage_base:.0%} → {result.shrinkage_top:.0%}, p = "
          f"{result.order}, {result.dofs} DoF:")  # fmt: skip
    r = result
    print(f"  top centre u_z = {r.uz_top_centre_nm:.1f} nm, corner {r.uz_top_corner_nm:.1f} nm, "
          f"sag {r.sag_nm:.1f} nm, wall pull-in {r.pull_in_nm:.1f} nm, energy "
          f"{r.strain_energy_pj:.3g} pJ")  # fmt: skip
    out = {"check_layer": {"fem_nm": fem, "layer_nm": layer}, "result": asdict(result)}
    with open("printed_block_shrinkage.json", "w") as f:
        json.dump(out, f, indent=1)
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(result.profile_x_um, result.profile_uz_nm)
        ax.set_xlabel("x [µm] (top face, y = 0)")
        ax.set_ylabel("u_z [nm]")
        ax.set_title("printed block: top face after shrinkage")
        fig.tight_layout()
        fig.savefig("printed_block_shrinkage.png", dpi=120)
    except ImportError:
        pass
    return out


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)

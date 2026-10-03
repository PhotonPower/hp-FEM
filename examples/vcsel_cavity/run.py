"""VCSEL micro-cavity: resonance wavelength and quality factor of a DBR cavity.

A vertical-cavity surface-emitting laser is a one-wavelength GaAs cavity between two
distributed Bragg reflectors of quarter-wave AlAs / GaAs pairs, designed for 850 nm. The
resonance is a quasi-normal mode: `hpfem.Resonance2D` solves the eigenproblem with PML
above (air) and below (substrate) and returns the complex frequency, from which the
resonance wavelength and Q = Re omega / (-2 Im omega) follow. The effective 2D model is a
narrow strip with PEC side walls (the planar, laterally uniform mode), so the result can
be checked against the transfer-matrix pole of the same layer stack, which this script
computes as well. Sweeping the number of top pairs shows the exponential growth of Q.

Run `python examples/vcsel_cavity/run.py [--quick]`; results go to `vcsel_cavity.json`,
the mode profile to `vcsel_cavity.png`.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass

import numpy as np

import hpfem
from hpfem import units

LAMBDA_DESIGN = 850 * units.nm
N_HIGH, N_LOW = 3.52, 2.95  # GaAs, AlAs near 850 nm (non-dispersive model)
N_SUBSTRATE = 3.52  # GaAs substrate below the bottom DBR
N_AIR = 1.0
TAG_HIGH, TAG_LOW, TAG_SUBSTRATE = 2, 3, 4


@dataclass
class Layer:
    index: float
    thickness: float  # [m]
    tag: int


def stack(top_pairs: int, bottom_pairs: int) -> list[Layer]:
    """Layers from the top (air side) down: top DBR, cavity, bottom DBR."""
    q_high = LAMBDA_DESIGN / (4 * N_HIGH)
    q_low = LAMBDA_DESIGN / (4 * N_LOW)
    layers = []
    for _ in range(top_pairs):
        layers += [Layer(N_HIGH, q_high, TAG_HIGH), Layer(N_LOW, q_low, TAG_LOW)]
    layers.append(Layer(N_HIGH, LAMBDA_DESIGN / N_HIGH, TAG_HIGH))  # one-wavelength cavity
    for _ in range(bottom_pairs):
        layers += [Layer(N_LOW, q_low, TAG_LOW), Layer(N_HIGH, q_high, TAG_HIGH)]
    return layers


# --- transfer-matrix reference ---------------------------------------------------------------


def transfer_matrix_pole(layers: list[Layer], k_guess: complex, n_top=N_AIR, n_bottom=N_SUBSTRATE):
    """Complex k0 of the resonance: zero of the (1, 1) element of the stack's transfer matrix
    (no incoming waves from either side), found by the secant method from k_guess."""

    def m11(k: complex) -> complex:
        # characteristic matrix for E_y / H_z at normal incidence (Born & Wolf convention)
        total = np.eye(2, dtype=complex)
        for layer in layers:
            phase = k * layer.index * layer.thickness
            m = np.array(
                [
                    [np.cos(phase), -1j * np.sin(phase) / layer.index],
                    [-1j * layer.index * np.sin(phase), np.cos(phase)],
                ]
            )
            total = total @ m
        # outgoing-wave condition on both sides: the coefficient of the incoming wave vanishes
        a = (total[0, 0] + total[0, 1] * n_bottom) * n_top + (total[1, 0] + total[1, 1] * n_bottom)
        return a

    k0, k1 = k_guess, k_guess * (1 + 1e-3)
    f0, f1 = m11(k0), m11(k1)
    for _ in range(100):
        k2 = k1 - f1 * (k1 - k0) / (f1 - f0)
        k0, f0, k1, f1 = k1, f1, k2, m11(k2)
        if abs(k1 - k0) < 1e-13 * abs(k1):
            break
    return k1


# --- finite-element model --------------------------------------------------------------------


def strip_mesh(nodes: np.ndarray, height: float, tags: list[int]):
    """Strip of one cell row over the 1D grid `nodes` (two triangles per interval, cell tags
    per interval), PEC-tagged on the long sides (y = 0 and y = height)."""
    n = len(nodes)
    vertices = np.vstack(
        [np.column_stack([nodes, np.zeros(n)]), np.column_stack([nodes, np.full(n, height)])]
    )
    cells, cell_tags = [], []
    for i in range(n - 1):
        cells.append([i, i + 1, n + i + 1])
        cells.append([i, n + i + 1, n + i])
        cell_tags += [tags[i], tags[i]]
    mesh = hpfem.Mesh2D(vertices, np.array(cells), cell_tags)
    bottom = np.array([[i, i + 1] for i in range(n - 1)])
    top = bottom + n
    mesh.set_facet_tags(bottom, [hpfem.box_tag.Y_MIN] * (n - 1))
    mesh.set_facet_tags(top, [hpfem.box_tag.Y_MAX] * (n - 1))
    return mesh


@dataclass
class Result:
    top_pairs: int
    bottom_pairs: int
    wavelength_nm: float
    quality: float
    wavelength_tmm_nm: float
    quality_tmm: float
    num_dofs: int


def resonance(top_pairs: int, bottom_pairs: int, cells_per_layer: int = 2, order: int = 3,
              margin_factor: float = 1.0):  # fmt: skip
    """FEM resonance of the stack: (mode, mesh, dofs, x-coordinates of the layer interfaces)."""
    layers = stack(top_pairs, bottom_pairs)
    k_design = 2 * np.pi / LAMBDA_DESIGN
    margin = margin_factor * LAMBDA_DESIGN  # homogeneous air / substrate before the PML
    pml = 2 * LAMBDA_DESIGN
    # x runs downward from the air side: air margin + PML above, substrate margin + PML below;
    # the 1D grid has nodes on every layer interface, cells_per_layer cells per layer
    total = sum(layer.thickness for layer in layers)
    boundaries = np.cumsum([0.0] + [layer.thickness for layer in layers])
    nodes = [-(margin + pml)]
    tags = []
    for x0, x1, tag in (
        [(-(margin + pml), -margin, 0), (-margin, 0.0, 0)]
        + [(boundaries[i], boundaries[i + 1], layer.tag) for i, layer in enumerate(layers)]
        + [
            (total, total + margin, TAG_SUBSTRATE),
            (total + margin, total + margin + pml, TAG_SUBSTRATE),
        ]
    ):
        count = max(cells_per_layer, int(np.ceil((x1 - x0) / (LAMBDA_DESIGN / 16))))
        for j in range(1, count + 1):
            nodes.append(x0 + (x1 - x0) * j / count)
            tags.append(tag)
    height = 0.05 * LAMBDA_DESIGN  # narrow strip: lateral modes far above the target
    mesh = strip_mesh(np.array(nodes), height, tags)
    dofs = hpfem.NedelecDofMap2D(mesh, order)
    setup = hpfem.ResonanceSetup2D()
    setup.target_omega = k_design * hpfem.constants.c0
    setup.materials.set(TAG_HIGH, hpfem.Material.dielectric(N_HIGH))
    setup.materials.set(TAG_LOW, hpfem.Material.dielectric(N_LOW))
    setup.materials.set(TAG_SUBSTRATE, hpfem.Material.dielectric(N_SUBSTRATE))
    setup.pec_tags = [hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX]
    setup.pml = hpfem.PmlBox2D(
        [-margin, 0.0], [total + margin, height], [pml, pml, 0.0, 0.0], k_design, 1.0,
        hpfem.PmlProfile(2, 1e-10),
    )  # fmt: skip
    setup.num_modes = 3
    setup.krylov_dimension = 40
    problem = hpfem.Resonance2D(dofs, setup)
    modes = problem.solve()
    return modes[0], mesh, dofs, boundaries


def compare(top_pairs: int, bottom_pairs: int, **kwargs) -> Result:
    mode, _, dofs, _ = resonance(top_pairs, bottom_pairs, **kwargs)
    k_fem = mode.omega / hpfem.constants.c0
    k_tmm = transfer_matrix_pole(stack(top_pairs, bottom_pairs), k_fem)
    return Result(
        top_pairs, bottom_pairs, mode.wavelength / units.nm, mode.quality,
        2 * np.pi / k_tmm.real / units.nm, k_tmm.real / (-2 * k_tmm.imag), dofs.num_dofs,
    )  # fmt: skip


def main(argv=None) -> int:
    hpfem.set_log_level("warn")
    quick = "--quick" in (argv or sys.argv[1:])
    bottom = 8 if quick else 14
    tops = [2, 4] if quick else [4, 6, 8, 10]
    print(
        f"GaAs / AlAs DBR cavity, lambda_design = {LAMBDA_DESIGN / units.nm:.0f} nm, "
        f"{bottom} bottom pairs"
    )
    print(f"{'top':>4} {'lambda [nm]':>12} {'Q':>10} {'lambda TMM':>12} {'Q TMM':>10} {'DoF':>7}")
    results = []
    for top in tops:
        r = compare(top, bottom)
        results.append(r)
        print(
            f"{top:>4} {r.wavelength_nm:>12.3f} {r.quality:>10.1f} {r.wavelength_tmm_nm:>12.3f} "
            f"{r.quality_tmm:>10.1f} {r.num_dofs:>7}"
        )
    with open("vcsel_cavity.json", "w", encoding="utf-8") as out:
        json.dump([asdict(r) for r in results], out, indent=2)
    # mode profile of the last configuration
    mode, mesh, dofs, boundaries = resonance(tops[-1], bottom)
    locator = hpfem.PointLocator2D(mesh)
    x = np.linspace(mesh.vertices[:, 0].min(), mesh.vertices[:, 0].max(), 2000)
    profile = []
    for xi in x:
        value = hpfem.evaluate_hcurl(
            dofs, mode.field, locator, [xi, 0.5 * mesh.vertices[:, 1].max()]
        )
        profile.append(abs(value[1]) if value is not None else np.nan)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.plot(x / units.nm, profile / np.nanmax(profile))
        for b in boundaries:
            ax.axvline(b / units.nm, color="0.8", linewidth=0.5)
        ax.set_xlabel("depth from the top DBR surface [nm]")
        ax.set_ylabel("|E_y| / max")
        ax.set_title(
            f"{tops[-1]} top / {bottom} bottom pairs: "
            f"lambda = {mode.wavelength / units.nm:.2f} nm, Q = {mode.quality:.0f}"
        )
        fig.tight_layout()
        fig.savefig("vcsel_cavity.png", dpi=120)
    except ImportError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

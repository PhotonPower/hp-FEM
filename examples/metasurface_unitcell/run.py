"""Metasurface unit cell: transmission and phase of a dielectric ridge over its width.

A periodic array of TiO2 ridges (height h, period a < lambda) on fused silica is lit at
normal incidence from the air side (light travels in -x, the substrate fills x < 0, the
ridges stand on it in 0 < x < h). Each width w gives the complex zeroth-order transmission
t(w); a meta-lens is a map of such elements chosen for their phase arg t(w). In the
effective 2D model the "pillars" are ridges (invariant in z) and the in-plane electric
field (H_z polarisation) is solved; the Bloch-periodic unit cell with PML above and
below is the lamellar-grating setup of `tests/convergence/lamellar_grating.cpp`.

Run `python examples/metasurface_unitcell/run.py` (about a minute); results go to
`metasurface_unitcell.json`, the phase map to `metasurface_unitcell.png`.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass

import numpy as np

import hpfem
from hpfem import materials, units

PERIOD = 350 * units.nm  # subwavelength in air and in the substrate (lambda / n = 411 nm)
WAVELENGTH = 600 * units.nm
# the geometry in multiples of a / 4, so that ridge, margins and PML fall on cell boundaries
# of the structured mesh for any multiple of 4 cells per period (a PML shorter than about
# 1.5 wavelengths leaves a few per cent in the energy balance)
HEIGHT = 1.75 * PERIOD  # 612.5 nm
MARGIN = 1.5 * PERIOD  # homogeneous region between the ridge and the PML
PML = 2.75 * PERIOD
TAG_SUBSTRATE, TAG_RIDGE = 2, 3


@dataclass
class Element:
    width_nm: float
    transmission: float  # |t|^2 (power, zeroth order)
    reflection: float
    phase: float  # arg t relative to the bare substrate [rad]
    num_dofs: int


def unit_cell(width: float, cells_per_period: int = 12, order: int = 3):
    """Solves one unit cell: (problem, solution, mesh, dofs)."""
    k0 = units.vacuum_wavenumber(units.angular_frequency(wavelength=WAVELENGTH))
    omega = k0 * hpfem.constants.c0
    cell = PERIOD / cells_per_period
    x_bottom, x_top = -(MARGIN + PML), HEIGHT + MARGIN + PML
    nx = round((x_top - x_bottom) / cell)
    mesh = hpfem.rectangle(nx, cells_per_period, [x_bottom, 0.0], [x_top, PERIOD])
    for c, x in enumerate(mesh.cell_centroids):
        if x[0] < 0:
            mesh.set_cell_tag(c, TAG_SUBSTRATE)
        elif x[0] < HEIGHT and abs(x[1] - PERIOD / 2) < width / 2:
            mesh.set_cell_tag(c, TAG_RIDGE)
    dofs = hpfem.NedelecDofMap2D(mesh, order)
    setup = hpfem.ScatteringSetup2D()
    setup.omega = omega
    setup.materials.set(TAG_SUBSTRATE, materials.get("SiO2").at(omega))
    setup.materials.set(TAG_RIDGE, materials.get("TiO2").at(omega))
    k = np.array([-k0, 0.0])  # normal incidence from +x
    setup.incident = hpfem.plane_wave([0.0, 1.0], k)
    setup.formulation = hpfem.Formulation.SCATTERED_FIELD
    setup.pml = hpfem.PmlBox2D(
        [-MARGIN, 0.0], [HEIGHT + MARGIN, PERIOD], [PML, PML, 0.0, 0.0], k0, 1.0,
        hpfem.PmlProfile(2, 1e-10),  # normal incidence; oblique: PmlProfile.for_angle
    )  # fmt: skip
    setup.pec_tags = [hpfem.box_tag.X_MIN, hpfem.box_tag.X_MAX]
    shift = np.array([0.0, PERIOD])
    setup.periodic = [
        hpfem.PeriodicPair2D(
            hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX, shift, hpfem.bloch_phase(k, shift)
        )
    ]
    problem = hpfem.Scattering2D(dofs, setup)
    return problem, problem.solve(), mesh, dofs


def orders(problem, solution, mesh, dofs, cells_per_period: int):
    """(reflected efficiencies, transmitted efficiencies, complex t0) of the solved cell."""
    k0 = problem.wavenumber
    locator = hpfem.PointLocator2D(mesh)
    points = 8 * cells_per_period
    n_sub = float(np.real(problem.setup.materials.at(TAG_SUBSTRATE).refractive_index))
    reflected = hpfem.fourier_coefficients(
        dofs, solution.unknown, locator, HEIGHT + 0.5 * MARGIN, 0.0, PERIOD, 0.0, 1, points
    )
    transmitted = hpfem.fourier_coefficients(
        lambda x: problem.total_field(solution, locator, x),
        -0.5 * MARGIN,
        0.0,
        PERIOD,
        0.0,
        1,
        points,
    )
    r = hpfem.diffraction_efficiencies(reflected, k0, 1.0, PERIOD, 0.0, k0, 1.0)
    t = hpfem.diffraction_efficiencies(transmitted, k0, n_sub, PERIOD, 0.0, k0, 1.0)
    return r, t, complex(transmitted[1][1])  # y component of the zeroth order


def sweep(widths_nm, cells_per_period: int = 12, order: int = 3, verbose: bool = True):
    """Elements for the given ridge widths; the phase is relative to the bare substrate."""
    _, _, reference, _ = _solve_orders(0.0, cells_per_period, order)
    elements = []
    for width_nm in widths_nm:
        r, t, t0, num_dofs = _solve_orders(width_nm * units.nm, cells_per_period, order)
        phase = float(np.angle(t0 / reference))
        elements.append(
            Element(
                width_nm,
                sum(o.efficiency for o in t),
                sum(o.efficiency for o in r),
                phase,
                num_dofs,
            )
        )
        if verbose:
            e = elements[-1]
            print(
                f"w = {e.width_nm:6.1f} nm  T = {e.transmission:.4f}  R = {e.reflection:.4f}  "
                f"R + T = {e.transmission + e.reflection:.4f}  phase = {e.phase / np.pi:+.3f} pi"
            )
    return elements


def _solve_orders(width: float, cells_per_period: int, order: int):
    """(reflected, transmitted, t0, num_dofs) of one width."""
    problem, solution, mesh, dofs = unit_cell(width, cells_per_period, order)
    return (*orders(problem, solution, mesh, dofs, cells_per_period), dofs.num_dofs)


def main(argv=None) -> int:
    hpfem.set_log_level("warn")
    quick = "--quick" in (argv or sys.argv[1:])
    cells = 8 if quick else 12
    widths = (
        np.arange(1, 8) * PERIOD / 8 / units.nm
        if quick
        else np.arange(1, 12) * PERIOD / 12 / units.nm
    )
    print(
        f"TiO2 ridges on SiO2, a = {PERIOD / units.nm:.0f} nm, h = {HEIGHT / units.nm:.0f} nm, "
        f"lambda = {WAVELENGTH / units.nm:.0f} nm, {cells} cells per period, p = 3"
    )
    elements = sweep(widths, cells_per_period=cells)
    with open("metasurface_unitcell.json", "w", encoding="utf-8") as out:
        json.dump([asdict(e) for e in elements], out, indent=2)
    unwrapped = np.unwrap([e.phase for e in elements])
    print(f"phase coverage {(unwrapped.max() - unwrapped.min()) / np.pi:.2f} pi over the widths")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        w = [e.width_nm for e in elements]
        ax.plot(w, unwrapped / np.pi, "o-", label="phase / pi")
        ax.plot(w, [e.transmission for e in elements], "s-", label="transmission")
        ax.set_xlabel("ridge width [nm]")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.savefig("metasurface_unitcell.png", dpi=120)
    except ImportError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

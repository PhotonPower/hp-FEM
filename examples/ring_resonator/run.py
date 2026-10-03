"""Ring resonator: coupling and resonance in the 2D effective-index model.

A bus waveguide runs past a ring of the same waveguide; at the ring's resonances
(circumference = integer number of guided wavelengths) light couples into the ring and
the transmission of the bus shows sharp dips. The effective-index model replaces the
silicon-on-insulator slab by its TE effective index (n_eff = 2.4 in silica, n = 1.44),
and the in-plane electric field (H_z polarisation) is solved on the chip plane.

Two views of the same device:
* `hpfem.Resonance2D` finds the quasi-normal modes of the ring coupled to the bus near the
  target wavelength: resonance wavelengths and quality factors (the bus and the radiation
  loss of the staircase ring set Q).
* a Gaussian line current in the bus excites the guided mode; the transmitted power at the
  bus end, normalised by the same source without the ring, gives the transmission
  spectrum, whose dips coincide with the resonances.

Run `python examples/ring_resonator/run.py [--quick]`; results go to `ring_resonator.json`,
the spectrum and the resonant field to `ring_resonator.png`.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass

import numpy as np

import hpfem
from hpfem import units

LAMBDA = 1550 * units.nm
N_CORE, N_CLAD = 2.4, 1.44  # TE effective index of a 220 nm SOI slab, silica cladding
WIDTH = 450 * units.nm  # waveguide width
RADIUS = 2.5 * units.um  # ring radius (to the centre line)
GAP = 150 * units.nm  # bus-ring gap
TAG_CORE = 2
BOX_SIDES = [hpfem.box_tag.X_MIN, hpfem.box_tag.X_MAX, hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX]


@dataclass
class Resonances:
    wavelength_nm: list[float]
    quality: list[float]
    num_dofs: int


@dataclass
class Spectrum:
    wavelength_nm: list[float]
    transmission: list[float]


def device_mesh(cell, with_ring=True, margin=None, pml=None):
    """Structured mesh of the chip plane: bus along x at y = 0, ring above it; returns
    (mesh, half width, half height, pml, ring centre)."""
    margin = margin or 1.0 * units.um
    pml = pml or 1.0 * units.um
    centre_y = WIDTH / 2 + GAP + RADIUS + WIDTH / 2
    half_x = RADIUS + WIDTH + margin + pml
    y_lo = -(WIDTH / 2 + margin + pml)
    y_hi = centre_y + RADIUS + WIDTH / 2 + margin + pml
    nx = round(2 * half_x / cell)
    ny = round((y_hi - y_lo) / cell)
    mesh = hpfem.rectangle(nx, ny, [-half_x, y_lo], [half_x, y_hi])
    centroids = mesh.cell_centroids
    r = np.hypot(centroids[:, 0], centroids[:, 1] - centre_y)
    in_bus = np.abs(centroids[:, 1]) < WIDTH / 2
    in_ring = np.abs(r - RADIUS) < WIDTH / 2
    for c in np.flatnonzero(in_bus | (in_ring if with_ring else False)):
        mesh.set_cell_tag(int(c), TAG_CORE)
    return mesh, half_x, (y_lo, y_hi), pml, centre_y


def materials():
    m = hpfem.MaterialMap(hpfem.Material.dielectric(N_CLAD))
    m.set(TAG_CORE, hpfem.Material.dielectric(N_CORE))
    return m


def pml_box(mesh_data, k0):
    _, half_x, (y_lo, y_hi), pml, _ = mesh_data
    lower, upper = [-half_x + pml, y_lo + pml], [half_x - pml, y_hi - pml]
    return hpfem.PmlBox2D.uniform(lower, upper, pml, k0, N_CLAD, hpfem.PmlProfile(2, 1e-8))


def resonances(order=3, cell=50 * units.nm, num_modes=4, target=LAMBDA):
    """Quasi-normal modes of the ring coupled to the bus around the target wavelength."""
    data = device_mesh(cell)
    mesh = data[0]
    dofs = hpfem.NedelecDofMap2D(mesh, order)
    k0 = 2 * np.pi / target
    setup = hpfem.ResonanceSetup2D()
    setup.target_omega = k0 * hpfem.constants.c0
    setup.materials = materials()
    setup.pec_tags = BOX_SIDES
    setup.pml = pml_box(data, k0)
    setup.num_modes = num_modes
    setup.krylov_dimension = 4 * num_modes + 20
    modes = hpfem.Resonance2D(dofs, setup).solve()
    modes.sort(key=lambda m: m.wavelength)
    return (
        Resonances(
            [m.wavelength / units.nm for m in modes], [m.quality for m in modes], dofs.num_dofs
        ),
        modes,
        mesh,
        dofs,
    )


def transmission(wavelengths, order=3, cell=50 * units.nm):
    """Transmission of the bus (power at the output cross-section with the ring over the
    power without it) for the given vacuum wavelengths."""
    results = []
    for with_ring in (True, False):
        data = device_mesh(cell, with_ring=with_ring)
        mesh, half_x, (y_lo, y_hi), pml, _ = data
        dofs = hpfem.NedelecDofMap2D(mesh, order)
        source = np.array([-(half_x - pml) + 0.5 * units.um, 0.0])
        xs = np.unique(mesh.vertices[:, 0])
        x_out = xs[np.argmin(np.abs(xs - (half_x - pml - 0.5 * units.um)))]  # a mesh line
        # output cross-section of the bus: facets on the line x = x_out across the core
        facets = []
        for f in range(mesh.num_facets):
            v = mesh.vertices[list(mesh.facet_vertices(f))]
            if np.all(np.abs(v[:, 0] - x_out) < 1e-12) and np.all(np.abs(v[:, 1]) < 3 * WIDTH):
                cells = [c for c in mesh.facet_cells(f) if c != hpfem.INVALID_INDEX]
                left = min(cells, key=lambda c: mesh.cell_centroid(c)[0])
                facets.append(hpfem.Surface2D.Facet(f, left))
        surface = hpfem.Surface2D()
        surface.facets = facets
        powers = []
        for lam in wavelengths:
            k0 = 2 * np.pi / lam
            omega = k0 * hpfem.constants.c0
            setup = hpfem.ScatteringSetup2D()
            setup.omega = omega
            setup.materials = materials()
            setup.current = hpfem.gaussian_current(source, np.array([0.0, 1.0]), 0.5 * cell, omega)
            setup.formulation = hpfem.Formulation.TOTAL_FIELD
            setup.pml = pml_box(data, k0)
            setup.pec_tags = BOX_SIDES
            solution = hpfem.Scattering2D(dofs, setup).solve()
            powers.append(
                hpfem.poynting_flux(dofs, solution.unknown, omega, setup.materials, surface)
            )
        results.append(powers)
    return Spectrum(
        [lam / units.nm for lam in wavelengths],
        [a / b for a, b in zip(results[0], results[1], strict=True)],
    )


def main(argv=None) -> int:
    hpfem.set_log_level("warn")
    quick = "--quick" in (argv or sys.argv[1:])
    order, cell = (2, 80 * units.nm) if quick else (3, 80 * units.nm)
    res, modes, mesh, dofs = resonances(order, cell, num_modes=2 if quick else 4)
    print(
        f"ring R = {RADIUS / units.um:.1f} um, w = {WIDTH / units.nm:.0f} nm, "
        f"gap {GAP / units.nm:.0f} nm, n_eff = {N_CORE}, {res.num_dofs} DoF"
    )
    print(f"{'lambda_res [nm]':>16} {'Q':>10}")
    for lam, q in zip(res.wavelength_nm, res.quality, strict=True):
        print(f"{lam:>16.3f} {q:>10.1f}")
    # transmission around the resonance closest to 1550 nm
    nearest = min(
        range(len(res.wavelength_nm)), key=lambda i: abs(res.wavelength_nm[i] - LAMBDA / units.nm)
    )
    lam_res, q_res = res.wavelength_nm[nearest], res.quality[nearest]
    width = lam_res / q_res
    points = 7 if quick else 21
    wavelengths = (lam_res + np.linspace(-3, 3, points) * width) * units.nm
    spec = transmission(wavelengths, order, cell)
    print(f"{'lambda [nm]':>12} {'T':>8}")
    for lam, t in zip(spec.wavelength_nm, spec.transmission, strict=True):
        print(f"{lam:>12.3f} {t:>8.4f}")
    dip = spec.wavelength_nm[int(np.argmin(spec.transmission))]
    print(
        f"resonance {lam_res:.3f} nm (Q = {q_res:.0f}), transmission dip at {dip:.3f} nm, "
        f"minimum T = {min(spec.transmission):.3f}"
    )
    with open("ring_resonator.json", "w", encoding="utf-8") as out:
        json.dump({"resonances": asdict(res), "spectrum": asdict(spec)}, out, indent=2)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from hpfem import interop

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(spec.wavelength_nm, spec.transmission, "o-")
        axes[0].axvline(lam_res, color="0.6", linestyle="--", label="resonance (eigenmode)")
        axes[0].set_xlabel("wavelength [nm]")
        axes[0].set_ylabel("transmission")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        interop.plot_field(
            dofs, modes[nearest].field, ax=axes[1], component="abs", subdivisions=2, colorbar=False
        )
        axes[1].set_title(f"|E| of the mode at {lam_res:.2f} nm, Q = {q_res:.0f}")
        fig.tight_layout()
        fig.savefig("ring_resonator.png", dpi=120)
    except ImportError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

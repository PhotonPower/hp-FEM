"""Textured silicon solar cell: light trapping, carrier generation and heating.

A thin crystalline-silicon absorber with a periodic surface texture (a ridge grating under
a silica anti-reflection layer, 2D effective model) is lit at normal incidence. The
texture scatters light into oblique orders that are trapped by total internal reflection
and lengthen the path in the absorber, so the absorbed fraction of the incident power rises
compared with the flat cell. The absorbed photons are the carrier generation of a device
solver: `hpfem.pv` turns the field solutions at several wavelengths into the generation
rate per cell weighted by a (sample) solar spectrum, the laterally averaged depth profile
G(z) and a meshio export. The absorbed power also heats the cell (`Thermal2D`, substrate
at the ambient temperature, adiabatic front and periodic sides).

Run `python examples/solar_cell_texture/run.py [--quick]`; results go to
`solar_cell_texture.json`, `solar_cell_texture_generation.vtu` (cell data) and
`solar_cell_texture_profile.csv` (depth profile), plots to `solar_cell_texture.png`.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass

import numpy as np

import hpfem
from hpfem import materials, pv, units

PERIOD = 600 * units.nm
THICKNESS = 2.0 * units.um  # silicon absorber
TEXTURE = 250 * units.nm  # ridge height
ARC = 80 * units.nm  # silica anti-reflection layer
TAG_SI, TAG_SIO2 = 2, 3
# a few bands of the solar spectrum (irradiance in W/(m^2 m), AM1.5G order of magnitude)
SPECTRUM = {
    500 * units.nm: 1.55e9,
    600 * units.nm: 1.50e9,
    700 * units.nm: 1.40e9,
    800 * units.nm: 1.15e9,
    900 * units.nm: 0.90e9,
    1000 * units.nm: 0.75e9,
}
SIDES = [hpfem.box_tag.X_MIN, hpfem.box_tag.X_MAX, hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX]


@dataclass
class Result:
    wavelength_nm: list[float]
    absorbed_textured: list[float]  # fraction of the incident power
    absorbed_flat: list[float]
    generation_total: float  # photons per second per metre of width [1/(m s)]
    temperature_rise: float  # K at the front
    num_dofs: int


def cell_mesh(textured: bool, cell: float):
    """Periodic unit cell: substrate PML and margin below the absorber (y < 0), the silicon
    (0 < y < THICKNESS, with ridges up to THICKNESS + TEXTURE on half the period), the ARC,
    vacuum margin and PML above. Cells are tagged by centroid."""
    margin, pml = 0.6 * units.um, 0.8 * units.um
    y_lo = -(margin + pml)
    y_hi = THICKNESS + TEXTURE + ARC + margin + pml
    nx = max(2, round(PERIOD / cell))
    ny = round((y_hi - y_lo) / cell)
    mesh = hpfem.rectangle(nx, ny, [0.0, y_lo], [PERIOD, y_hi])
    for c, (x, y) in enumerate(mesh.cell_centroids):
        top = THICKNESS + (TEXTURE if textured and x < PERIOD / 2 else 0.0)
        if y < 0:
            mesh.set_cell_tag(c, TAG_SI)  # substrate continues the silicon (PML absorbs)
        elif y < top:
            mesh.set_cell_tag(c, TAG_SI)
        elif y < top + ARC:
            mesh.set_cell_tag(c, TAG_SIO2)
    return mesh, (y_lo, y_hi, margin, pml)


def solve(mesh, bounds, wavelength, order):
    y_lo, y_hi, margin, pml = bounds
    omega = units.angular_frequency(wavelength=wavelength)
    k0 = units.vacuum_wavenumber(omega)
    dofs = hpfem.NedelecDofMap2D(mesh, order)
    setup = hpfem.ScatteringSetup2D()
    setup.omega = omega
    setup.materials.set(TAG_SI, materials.get("Si").at(omega))
    setup.materials.set(TAG_SIO2, materials.get("SiO2").at(omega))
    k = np.array([0.0, -k0])  # downwards
    setup.incident = hpfem.plane_wave([1.0, 0.0], k)
    setup.formulation = hpfem.Formulation.SCATTERED_FIELD
    setup.pml = hpfem.PmlBox2D(
        [0.0, y_lo + pml],
        [PERIOD, y_hi - pml],
        [0.0, 0.0, pml, pml],
        k0,
        1.0,
        hpfem.PmlProfile(2, 1e-8),
    )
    setup.pec_tags = [hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX]
    shift = np.array([PERIOD, 0.0])
    setup.periodic = [
        hpfem.PeriodicPair2D(
            hpfem.box_tag.X_MIN, hpfem.box_tag.X_MAX, shift, hpfem.bloch_phase(k, shift)
        )
    ]
    problem = hpfem.Scattering2D(dofs, setup)
    solution = problem.solve()
    total = solution.unknown + hpfem.interpolate(dofs, setup.incident.value)
    return dofs, setup, total


def absorber_power(mesh, dofs, setup, total):
    """Absorbed power in the absorber above the substrate margin, per unit width."""
    per_cell = np.asarray(hpfem.absorbed_power_per_cell(dofs, total, setup.omega, setup.materials))
    inside = mesh.cell_centroids[:, 1] > 0
    return float(per_cell[inside].sum()), per_cell


def run(quick: bool) -> Result:
    cell = 60 * units.nm if quick else 40 * units.nm
    order = 2 if quick else 3
    wavelengths = sorted(SPECTRUM)
    if quick:
        wavelengths = wavelengths[1::2]
    irradiance = np.array([SPECTRUM[w] for w in wavelengths])
    textured, bounds = cell_mesh(True, cell)
    flat, _ = cell_mesh(False, cell)
    incident_intensity = hpfem.plane_wave_intensity(1.0, hpfem.Material.vacuum()) * PERIOD
    absorbed_t, absorbed_f, generation = [], [], []
    for wavelength in wavelengths:
        dofs, setup, total = solve(textured, bounds, wavelength, order)
        power, per_cell = absorber_power(textured, dofs, setup, total)
        absorbed_t.append(power / incident_intensity)
        generation.append(pv.generation_per_cell(dofs, total, setup.omega, setup.materials))
        dofs_f, setup_f, total_f = solve(flat, bounds, wavelength, order)
        absorbed_f.append(absorber_power(flat, dofs_f, setup_f, total_f)[0] / incident_intensity)
    weights = pv.spectral_weights(wavelengths, irradiance)
    g_total = pv.spectral_sum(generation, weights)  # 1/(m^2 s) per unit width in 2D
    g_total[textured.cell_centroids[:, 1] < 0] = 0.0  # the substrate margin is not the device
    centres, profile = pv.depth_profile(
        textured, g_total, axis=1, bins=40, extent=(0.0, THICKNESS + TEXTURE)
    )
    pv.write_generation("solar_cell_texture_generation.vtu", textured, g_total)
    pv.write_profile("solar_cell_texture_profile.csv", centres, profile)
    # heating by the spectrum: loads of the monochromatic solutions weighted alike
    h1 = hpfem.DofMap2D(textured, order)
    load = np.zeros(h1.num_dofs, dtype=complex)
    for wavelength, weight in zip(wavelengths, weights, strict=True):
        dofs, setup, total = solve(textured, bounds, wavelength, order)
        load += weight * np.asarray(
            hpfem.absorbed_power_load(dofs, total, setup.omega, setup.materials, h1)
        )
    thermal_setup = hpfem.ThermalSetup()
    thermal_setup.background_conductivity = 0.026  # air above
    thermal_setup.conductivity = {TAG_SI: 150.0, TAG_SIO2: 1.4}
    thermal_setup.fixed_temperature = [(hpfem.box_tag.Y_MIN, 300.0)]
    temperature = hpfem.Thermal2D(h1, thermal_setup).solve_load(load)
    front = hpfem.evaluate_h1(
        h1, temperature, hpfem.PointLocator2D(textured), [PERIOD / 4, THICKNESS]
    ).real
    return Result(
        [w / units.nm for w in wavelengths], absorbed_t, absorbed_f,
        float((g_total * textured.cell_volumes).sum()), front - 300.0, dofs.num_dofs,
    )  # fmt: skip


def main(argv=None) -> int:
    hpfem.set_log_level("warn")
    quick = "--quick" in (argv or sys.argv[1:])
    result = run(quick)
    print(f"{'lambda [nm]':>12} {'A textured':>11} {'A flat':>8}")
    for lam, a, f in zip(
        result.wavelength_nm, result.absorbed_textured, result.absorbed_flat, strict=True
    ):
        print(f"{lam:>12.0f} {a:>11.3f} {f:>8.3f}")
    print(
        f"generation {result.generation_total:.3e} 1/(m s) per period, front temperature rise "
        f"{result.temperature_rise:.3e} K ({result.num_dofs} DoF)"
    )
    with open("solar_cell_texture.json", "w", encoding="utf-8") as out:
        json.dump(asdict(result), out, indent=2)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = np.loadtxt("solar_cell_texture_profile.csv", delimiter=",")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].plot(result.wavelength_nm, result.absorbed_textured, "o-", label="textured")
        axes[0].plot(result.wavelength_nm, result.absorbed_flat, "s-", label="flat")
        axes[0].set_xlabel("wavelength [nm]")
        axes[0].set_ylabel("absorbed fraction")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[1].semilogy(rows[:, 0], rows[:, 1])
        axes[1].set_xlabel("depth from the rear [um]")
        axes[1].set_ylabel("generation rate [1/(m^3 s)]")
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig("solar_cell_texture.png", dpi=120)
    except ImportError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

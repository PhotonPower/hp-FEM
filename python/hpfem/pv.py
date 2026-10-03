"""Carrier-generation profiles for photovoltaic device solvers.

Every absorbed photon of energy ħω generates one electron-hole pair, so the generation
rate is the absorbed power density over the photon energy, G = q / (ħω) [1/(m³ s)]. The
functions here turn monochromatic field solutions into the quantities a device solver
needs: the generation rate of every cell (exact cell integrals), a continuous field for
plots, the sum over a spectrum weighted by the spectral irradiance (the field solutions
carry a unit incident amplitude; the weight is the incident power of the band over the
intensity of that unit wave), depth profiles G(z) averaged laterally (for 1D device
solvers) and the export as meshio cell data or CSV. SI units throughout; the incident
plane wave of a solution must have the amplitude given to ``spectral_weights``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from hpfem import _hpfem as core
from hpfem import units

HBAR = core.constants.h_planck / (2 * np.pi)


def generation_per_cell(dofs, coefficients, omega, materials, extra_order: int = 2):
    """Generation rate of every cell [1/(m^dim s)] of a monochromatic field: the absorbed
    power of the cell over the photon energy and the cell measure."""
    power = np.asarray(
        core.absorbed_power_per_cell(dofs, coefficients, omega, materials, extra_order)
    )
    return power / (HBAR * omega) / dofs.mesh.cell_volumes


def generation_field(dofs, coefficients, omega, materials, h1):
    """Generation rate as the H1 interpolant on the same mesh (for plots and VTK; it smears
    the jumps at material interfaces over a cell)."""
    density = np.asarray(core.absorbed_power_density(dofs, coefficients, omega, materials, h1))
    return density.real / (HBAR * omega)


def spectral_weights(wavelengths, irradiance, amplitude: float = 1.0, medium=None):
    """Weights that turn the monochromatic solutions at `wavelengths` (incident plane wave of
    `amplitude` [V/m] in `medium`, vacuum by default) into their share of a spectrum: the
    incident power of each band (spectral irradiance [W/(m² m)] times the band width from
    the midpoints) over the intensity of the unit wave."""
    wavelengths = np.asarray(wavelengths, dtype=float)
    irradiance = np.asarray(irradiance, dtype=float)
    if wavelengths.shape != irradiance.shape or wavelengths.ndim != 1 or len(wavelengths) < 1:
        raise ValueError("spectral_weights: wavelengths and irradiance must be 1-D of equal length")
    if len(wavelengths) == 1:
        widths = np.ones(1)  # a single band: the caller scales
    else:
        edges = np.concatenate(
            [[wavelengths[0]], 0.5 * (wavelengths[1:] + wavelengths[:-1]), [wavelengths[-1]]]
        )
        widths = np.diff(edges)
        widths[0] *= 2  # half-open end bands get the full spacing
        widths[-1] *= 2
    intensity = core.plane_wave_intensity(amplitude, medium or core.Material.vacuum())
    return irradiance * widths / intensity


def spectral_sum(per_wavelength: Sequence[np.ndarray], weights) -> np.ndarray:
    """Weighted sum of per-wavelength generation arrays (cells or field coefficients)."""
    weights = np.asarray(weights, dtype=float)
    if len(per_wavelength) != len(weights):
        raise ValueError("spectral_sum: one weight per wavelength")
    return sum(w * np.asarray(g) for w, g in zip(weights, per_wavelength, strict=True))


def depth_profile(mesh, per_cell, axis: int = 1, bins: int = 50, extent=None):
    """Lateral average of a per-cell quantity over bins along `axis` (volume-weighted by
    the cells whose centroid falls in a bin): ``(centres, profile)`` — the G(z) a 1D device
    solver takes. `extent` limits the binned range (default: the mesh)."""
    centroids = mesh.cell_centroids[:, axis]
    volumes = mesh.cell_volumes
    per_cell = np.asarray(per_cell, dtype=float)
    vertices = mesh.vertices[:, axis]
    lo, hi = extent if extent is not None else (vertices.min(), vertices.max())
    edges = np.linspace(lo, hi, bins + 1)
    index = np.clip(np.searchsorted(edges, centroids, side="right") - 1, 0, bins - 1)
    inside = (centroids >= lo) & (centroids <= hi)
    weighted = np.bincount(index[inside], weights=(per_cell * volumes)[inside], minlength=bins)
    volume = np.bincount(index[inside], weights=volumes[inside], minlength=bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        profile = np.where(volume > 0, weighted / volume, 0.0)
    return 0.5 * (edges[1:] + edges[:-1]), profile


def write_generation(path, mesh, per_cell, name: str = "generation", **cell_data):
    """Writes the per-cell generation rate (and further per-cell arrays) as cell data of the
    mesh in any format meshio writes (`.vtu`, `.xdmf`, `.msh`, …)."""
    from hpfem import interop

    data = {name: np.asarray(per_cell, dtype=float)}
    data.update({k: np.asarray(v, dtype=float) for k, v in cell_data.items()})
    interop.to_meshio(mesh, cell_data=data).write(str(path))


def write_profile(path, centres, profile, length_unit: float = units.um, header: str = "depth"):
    """CSV of a depth profile: depth in `length_unit`, generation rate in 1/(m^3 s)."""
    rows = np.column_stack([np.asarray(centres) / length_unit, np.asarray(profile)])
    np.savetxt(
        str(path),
        rows,
        delimiter=",",
        header=f"{header} [{length_unit:g} m], G [1/(m^3 s)]",
        comments="# ",
    )


__all__ = [
    "generation_per_cell", "generation_field", "spectral_weights", "spectral_sum",
    "depth_profile", "write_generation", "write_profile",
]  # fmt: skip

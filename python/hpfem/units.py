"""Unit conversion at the Python boundary.

The C++ core is strictly SI (metres, seconds, rad/s, V/m; CLAUDE.md §6). This module holds
the multipliers of the units nano-optics is usually written in and converts between the
spectral variables wavelength, angular frequency, frequency, photon energy and vacuum
wavenumber. Values are plain floats or NumPy arrays; multiply to go *into* SI
(``500 * nm``), divide to come *out* (``wavelength(omega) / nm``).

>>> omega = angular_frequency(wavelength=500 * nm)
>>> photon_energy(omega) / eV
2.4796...
"""

from __future__ import annotations

import numpy as np

from hpfem._hpfem import constants

# --- length -------------------------------------------------------------------------------
m = 1.0
cm = 1e-2
mm = 1e-3
um = 1e-6
nm = 1e-9
pm = 1e-12
angstrom = 1e-10

# --- time, frequency ----------------------------------------------------------------------
s = 1.0
ns = 1e-9
ps = 1e-12
fs = 1e-15
Hz = 1.0
kHz = 1e3
MHz = 1e6
GHz = 1e9
THz = 1e12
PHz = 1e15

# --- energy -------------------------------------------------------------------------------
J = 1.0
eV = constants.e_charge
meV = 1e-3 * eV

# --- misc ---------------------------------------------------------------------------------
rad = 1.0
deg = np.pi / 180.0
V_per_m = 1.0
W = 1.0
mW = 1e-3


def angular_frequency(
    *,
    wavelength: float | np.ndarray | None = None,
    frequency: float | np.ndarray | None = None,
    energy: float | np.ndarray | None = None,
    wavenumber: float | np.ndarray | None = None,
) -> float | np.ndarray:
    """Angular frequency omega [rad/s] from exactly one of the vacuum wavelength [m], the
    frequency [Hz], the photon energy [J] or the vacuum wavenumber k0 [1/m].

    >>> angular_frequency(wavelength=1550 * nm) / THz
    1215.6...
    """
    given = {
        "wavelength": wavelength,
        "frequency": frequency,
        "energy": energy,
        "wavenumber": wavenumber,
    }
    names = [k for k, v in given.items() if v is not None]
    if len(names) != 1:
        raise ValueError(f"angular_frequency: give exactly one spectral variable, got {names}")
    if wavelength is not None:
        return 2 * np.pi * constants.c0 / np.asarray(wavelength, dtype=float)
    if frequency is not None:
        return 2 * np.pi * np.asarray(frequency, dtype=float)
    if energy is not None:
        return np.asarray(energy, dtype=float) / (constants.h_planck / (2 * np.pi))
    return constants.c0 * np.asarray(wavenumber, dtype=float)


def wavelength(omega: float | np.ndarray) -> float | np.ndarray:
    """Vacuum wavelength [m] of the angular frequency omega [rad/s]."""
    return 2 * np.pi * constants.c0 / np.asarray(omega, dtype=float)


def frequency(omega: float | np.ndarray) -> float | np.ndarray:
    """Frequency [Hz] of the angular frequency omega [rad/s]."""
    return np.asarray(omega, dtype=float) / (2 * np.pi)


def photon_energy(omega: float | np.ndarray) -> float | np.ndarray:
    """Photon energy [J] (divide by ``eV`` for electron-volts) of omega [rad/s]."""
    return constants.h_planck / (2 * np.pi) * np.asarray(omega, dtype=float)


def vacuum_wavenumber(omega: float | np.ndarray) -> float | np.ndarray:
    """k0 = omega / c0 [1/m]."""
    return np.asarray(omega, dtype=float) / constants.c0


__all__ = [
    "m", "cm", "mm", "um", "nm", "pm", "angstrom",
    "s", "ns", "ps", "fs", "Hz", "kHz", "MHz", "GHz", "THz", "PHz",
    "J", "eV", "meV", "rad", "deg", "V_per_m", "W", "mW",
    "angular_frequency", "wavelength", "frequency", "photon_energy", "vacuum_wavenumber",
]  # fmt: skip

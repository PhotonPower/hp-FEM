"""Dispersive materials and a small library of optical constants.

A :class:`Dispersive` material gives the complex relative permittivity ``eps_r(omega)`` of
a medium at an angular frequency and turns it into the frequency-independent
:class:`Material` the C++ core uses at one frequency (``at(omega)``), so a frequency
sweep is ``for omega in omegas: setup.materials.set(tag, library["Au"].at(omega))``.
Convention ``exp(-i omega t)``: ``eps_r = (n + i k)^2`` with ``k >= 0``, lossy media have
``Im eps_r > 0`` (CLAUDE.md §6). Wavelengths are vacuum wavelengths in metres.

Models: :class:`Tabulated` n, k with linear interpolation in wavelength (the library data
in ``hpfem/data/*.csv``, each file naming its source), :class:`Sellmeier` (lossless
dielectrics), :class:`Drude` and :class:`DrudeLorentz` oscillator models, :class:`Constant`.

Library (``library[name]`` / ``get(name)``), all from the refractiveindex.info database
(CC0 1.0) with the original references in the data files:

======== ================================================ =================
name     source                                           range [µm]
======== ================================================ =================
Si       Green 2008 (intrinsic c-Si, 300 K)               0.25 – 1.45
SiO2     Malitson 1965 Sellmeier (fused silica)           0.21 – 6.7
Au       Johnson & Christy 1972                           0.188 – 1.94
Ag       Johnson & Christy 1972                           0.188 – 1.94
Al       Rakić 1995 (Kramers–Kronig consistent)           0.000124 – 200
TiO2     Devore 1951 Sellmeier (rutile, ordinary ray)     0.43 – 1.53
GaAs     Aspnes et al. 1986                               0.207 – 0.827
MAPbI3   Phillips et al. 2015 (CH3NH3PbI3 perovskite)     0.30 – 1.50
======== ================================================ =================
"""

from __future__ import annotations

import importlib.resources
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from hpfem import units
from hpfem._hpfem import Material, MaterialMap, constants


class Dispersive:
    """Base class: ``eps_r(omega)`` is complex, ``mu_r`` is a constant (1 unless given)."""

    name: str = "material"
    source: str = ""
    mu_r: complex = 1.0

    def eps_r(self, omega):
        raise NotImplementedError

    def refractive_index(self, omega):
        """n + i k = sqrt(eps_r mu_r), principal branch (Im >= 0 for passive media)."""
        return np.sqrt(np.asarray(self.eps_r(omega)) * self.mu_r + 0j)

    def at(self, omega: float) -> Material:
        """The frequency-independent core material at angular frequency omega [rad/s]."""
        return Material(complex(self.eps_r(float(omega))), complex(self.mu_r))

    def at_wavelength(self, wavelength: float) -> Material:
        """The same for a vacuum wavelength [m]."""
        return self.at(units.angular_frequency(wavelength=wavelength))

    def __repr__(self):
        return f"<hpfem.materials.{type(self).__name__} {self.name}: {self.source}>"


@dataclass(repr=False)
class Constant(Dispersive):
    """Frequency-independent eps_r (and mu_r)."""

    value: complex = 1.0
    mu_r: complex = 1.0
    name: str = "constant"
    source: str = "user"

    def eps_r(self, omega):
        return np.full_like(np.asarray(omega, dtype=float), self.value, dtype=complex)


@dataclass(repr=False)
class Tabulated(Dispersive):
    """n, k tabulated over the vacuum wavelength [m], interpolated linearly in n and k.
    Outside the tabulated range the policy ``out_of_range`` decides: ``"error"`` (default)
    raises ``ValueError``, ``"clamp"`` uses the nearest tabulated value and warns once
    (:func:`with_policy` makes a copy with another policy)."""

    wavelengths: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n: np.ndarray = field(default_factory=lambda: np.zeros(0))
    k: np.ndarray = field(default_factory=lambda: np.zeros(0))
    name: str = "tabulated"
    source: str = "user"
    mu_r: complex = 1.0
    out_of_range: str = "error"

    def __post_init__(self):
        self.wavelengths = np.asarray(self.wavelengths, dtype=float)
        self.n = np.asarray(self.n, dtype=float)
        self.k = np.asarray(self.k, dtype=float)
        if not (self.wavelengths.shape == self.n.shape == self.k.shape) or self.n.ndim != 1:
            raise ValueError("Tabulated: wavelengths, n and k must be 1-D arrays of one length")
        if len(self.wavelengths) < 2 or np.any(np.diff(self.wavelengths) <= 0):
            raise ValueError("Tabulated: wavelengths must be strictly increasing")

    @property
    def range(self) -> tuple[float, float]:
        """(shortest, longest) tabulated wavelength [m]."""
        return float(self.wavelengths[0]), float(self.wavelengths[-1])

    @classmethod
    def from_csv(cls, path, name: str | None = None, wavelength_unit: float = units.um):
        """Reads ``wavelength, n, k`` rows (comment lines start with ``#``; the first comment
        line is taken as the name, the ``# source:`` line as the source)."""
        text = open(path, encoding="utf-8").read().splitlines()
        comments = [line[1:].strip() for line in text if line.startswith("#")]
        source = next(
            (c[len("source:") :].strip() for c in comments if c.startswith("source:")), ""
        )
        rows = np.array(
            [[float(v) for v in line.split(",")] for line in text if line and line[0] != "#"]
        )
        return cls(
            rows[:, 0] * wavelength_unit, rows[:, 1], rows[:, 2],
            name=name or (comments[0].split(":")[0] if comments else "tabulated"),
            source=source,
        )  # fmt: skip

    def refractive_index(self, omega):
        lam = units.wavelength(omega)
        lo, hi = self.range
        if np.any(lam < lo * (1 - 1e-12)) or np.any(lam > hi * (1 + 1e-12)):
            given = f"{np.min(lam) / units.nm:.1f}-{np.max(lam) / units.nm:.1f} nm"
            message = (
                f"{self.name}: wavelength {given} outside the tabulated range "
                f"{lo / units.nm:.1f}-{hi / units.nm:.1f} nm"
            )
            if self.out_of_range == "clamp":
                warnings.warn(message + " (clamped to the nearest tabulated value)", stacklevel=2)
                lam = np.clip(lam, lo, hi)
            elif self.out_of_range == "error":
                raise ValueError(message)
            else:
                raise ValueError(
                    f"{self.name}: out_of_range = {self.out_of_range!r} (use 'error' or 'clamp')"
                )
        return np.interp(lam, self.wavelengths, self.n) + 1j * np.interp(
            lam, self.wavelengths, self.k
        )

    def eps_r(self, omega):
        return self.refractive_index(omega) ** 2 / self.mu_r


@dataclass(repr=False)
class Sellmeier(Dispersive):
    """Lossless dielectric, ``n^2 = 1 + A + sum_i B_i lam^2 / (lam^2 - C_i)`` with the vacuum
    wavelength in µm (refractiveindex.info formula 1 uses ``C_i = c_i^2``; pass ``C_i`` here).
    ``n^2 = A' + sum B_i / (lam^2 - C_i)`` forms (formula 4 with exponent 0) are written with
    ``lam2_numerator=False``."""

    A: float = 0.0
    B: Sequence[float] = ()
    C: Sequence[float] = ()
    lam2_numerator: bool = True
    range_um: tuple[float, float] = (0.0, np.inf)
    name: str = "sellmeier"
    source: str = "user"
    mu_r: complex = 1.0

    def eps_r(self, omega):
        lam = np.asarray(units.wavelength(omega), dtype=float) / units.um
        lo, hi = self.range_um
        if np.any(lam < lo * (1 - 1e-12)) or np.any(lam > hi * (1 + 1e-12)):
            raise ValueError(
                f"{self.name}: wavelength {np.min(lam):.4g}-{np.max(lam):.4g} um outside the "
                f"range of validity {lo}-{hi} um"
            )
        lam2 = lam**2
        n2 = 1.0 + self.A if self.lam2_numerator else self.A
        for b, c in zip(self.B, self.C, strict=True):
            n2 = n2 + (b * lam2 if self.lam2_numerator else b) / (lam2 - c)
        return (n2 + 0j) / self.mu_r


@dataclass(repr=False)
class DrudeLorentz(Dispersive):
    """``eps_r = eps_inf - omega_p^2 / (omega^2 + i gamma omega) + sum_j f_j omega_p^2 /
    (omega_j^2 - omega^2 - i gamma_j omega)`` (all rates in rad/s; the signs give
    ``Im eps_r > 0`` with ``exp(-i omega t)``). A pure Drude metal has no oscillators."""

    eps_inf: float = 1.0
    omega_p: float = 0.0
    gamma: float = 0.0
    oscillators: Sequence[tuple[float, float, float]] = ()  # (f_j, omega_j, gamma_j)
    name: str = "drude-lorentz"
    source: str = "user"
    mu_r: complex = 1.0

    def eps_r(self, omega):
        w = np.asarray(omega, dtype=float)
        eps = self.eps_inf - self.omega_p**2 / (w**2 + 1j * self.gamma * w)
        for f, w_j, g_j in self.oscillators:
            eps = eps + f * self.omega_p**2 / (w_j**2 - w**2 - 1j * g_j * w)
        return eps


def Drude(eps_inf: float, omega_p: float, gamma: float, name: str = "drude") -> DrudeLorentz:
    """Drude metal ``eps_r = eps_inf - omega_p^2 / (omega^2 + i gamma omega)``."""
    return DrudeLorentz(eps_inf, omega_p, gamma, (), name=name)


def with_policy(material: Dispersive, out_of_range: str) -> Dispersive:
    """A copy of a ``Tabulated`` material with another out-of-range policy (``"error"`` or
    ``"clamp"``); other materials are returned unchanged."""
    if isinstance(material, Tabulated):
        return replace(material, out_of_range=out_of_range)
    return material


class DispersiveMap:
    """Materials by cell tag that may depend on the frequency (M15 F13): ``set(tag, m)`` takes
    a :class:`Dispersive` model, a core ``Material`` or a library name, ``at(omega)`` returns
    the frozen ``MaterialMap`` for one frequency and ``apply(setup, omega)`` sets
    ``setup.omega`` and ``setup.materials`` in one call, so a sweep is ``for omega in ...:
    dmap.apply(setup, omega); problem = ...``. ``out_of_range`` (``"error"`` / ``"clamp"``)
    is applied to every tabulated material."""

    def __init__(self, background=None, out_of_range: str = "error"):
        self.background = self._coerce(background) if background is not None else Constant(1.0)
        self.out_of_range = out_of_range
        self._materials: dict[int, Dispersive] = {}

    @staticmethod
    def _coerce(material) -> Dispersive:
        if isinstance(material, Dispersive):
            return material
        if isinstance(material, str):
            return get(material)
        if isinstance(material, Material):
            return Constant(complex(material.eps_r), complex(material.mu_r), name="material")
        if isinstance(material, (int, float, complex)):
            return Constant(complex(material))
        raise TypeError(f"DispersiveMap: cannot use {material!r} as a material")

    def set(self, tag: int, material) -> DispersiveMap:
        self._materials[int(tag)] = self._coerce(material)
        return self

    def has(self, tag: int) -> bool:
        return int(tag) in self._materials

    @property
    def tags(self) -> list[int]:
        return sorted(self._materials)

    def of_tag(self, tag: int) -> Dispersive:
        return self._materials.get(int(tag), self.background)

    @property
    def range(self) -> tuple[float, float]:
        """Common wavelength range [m] of all tabulated / Sellmeier materials (0, inf without)."""
        lo, hi = 0.0, np.inf
        for material in [self.background, *self._materials.values()]:
            rng = getattr(material, "range", None)
            if rng is None and isinstance(material, Sellmeier):
                rng = (material.range_um[0] * units.um, material.range_um[1] * units.um)
            if rng is not None:
                lo, hi = max(lo, rng[0]), min(hi, rng[1])
        return lo, hi

    def at(self, omega: float) -> MaterialMap:
        """The core materials at ``omega`` as a ``MaterialMap`` (policy applied)."""
        out = MaterialMap(with_policy(self.background, self.out_of_range).at(omega))
        for tag, material in self._materials.items():
            out.set(tag, with_policy(material, self.out_of_range).at(omega))
        return out

    def at_wavelength(self, wavelength: float) -> MaterialMap:
        return self.at(units.angular_frequency(wavelength=wavelength))

    def apply(self, setup, omega: float):
        """Sets ``setup.omega`` and ``setup.materials`` for one frequency; returns the setup."""
        setup.omega = float(omega)
        setup.materials = self.at(omega)
        return setup

    def items(self):
        return self._materials.items()

    def __repr__(self):
        return f"<hpfem.materials.DispersiveMap {self.tags} background={self.background.name}>"


def fit_drude_lorentz(
    material: Dispersive,
    wavelengths,
    oscillators: int = 1,
    *,
    eps_inf: float | None = None,
    max_iterations: int = 2000,
) -> tuple[DrudeLorentz, float]:
    """Fits a :class:`DrudeLorentz` model (Drude term plus ``oscillators`` Lorentz poles) to the
    permittivity of ``material`` sampled at ``wavelengths`` [m] (an array; ``(start, stop,
    count)`` is expanded to a geometric grid). All rates are kept positive, so the fit is
    passive (``Im eps_r > 0``) and smooth in ω: for adaptive runs, eigenproblems and sweeps
    beyond the tabulated samples; ``eps_inf`` is kept at or above 1. Returns ``(model, max
    relative error of eps_r on the samples)``. ``eps_inf`` fixes the high-frequency limit if
    given."""
    from scipy.optimize import least_squares

    if isinstance(wavelengths, tuple) and len(wavelengths) == 3:
        wavelengths = np.geomspace(wavelengths[0], wavelengths[1], int(wavelengths[2]))
    lam = np.asarray(wavelengths, dtype=float)
    omega = np.asarray(units.angular_frequency(wavelength=lam), dtype=float)
    target = np.asarray(material.eps_r(omega), dtype=complex)
    scale = float(np.max(np.abs(target)))
    w_ref = float(np.sqrt(omega.min() * omega.max()))

    def unpack(x):
        # log-parameters keep every rate and strength positive; frequencies in units of w_ref
        e_inf = 1.0 + float(np.exp(x[0])) if eps_inf is None else float(eps_inf)
        offset = 0 if eps_inf is None else -1
        w_p = w_ref * np.exp(x[1 + offset])
        gamma = w_ref * np.exp(x[2 + offset])
        osc = []
        for j in range(oscillators):
            base = 3 + offset + 3 * j
            osc.append((np.exp(x[base]), w_ref * np.exp(x[base + 1]), w_ref * np.exp(x[base + 2])))
        return DrudeLorentz(e_inf, w_p, gamma, tuple(osc), name=f"{material.name} fit")

    def residual(x):
        eps = unpack(x).eps_r(omega)
        d = (eps - target) / scale
        return np.concatenate([d.real, d.imag])

    # start: Drude parameters from the longest wavelength, oscillators spread over the range
    eps_long = target[np.argmax(lam)]
    e_inf0 = max(
        1.0, float(np.real(target[np.argmin(lam)])) if np.real(target[np.argmin(lam)]) > 0 else 1.0
    )
    w_long = float(omega[np.argmax(lam)])
    wp0 = max(np.sqrt(max((e_inf0 - eps_long.real) * w_long**2, 1e-6 * w_ref**2)), 1e-3 * w_ref)
    gamma0 = 0.05 * w_ref
    x0 = ([np.log(max(e_inf0 - 1.0, 1e-3))] if eps_inf is None else []) + [
        np.log(wp0 / w_ref),
        np.log(gamma0 / w_ref),
    ]
    for j in range(oscillators):
        w_j = omega.min() * (omega.max() / omega.min()) ** ((j + 1) / (oscillators + 1))
        x0 += [np.log(0.1), np.log(w_j / w_ref), np.log(0.1)]
    best = None
    for attempt in range(3):
        start = np.array(x0, dtype=float)
        if attempt > 0:
            start = start + np.random.default_rng(attempt).normal(0, 0.3, size=start.shape)
        fit = least_squares(residual, start, max_nfev=max_iterations, method="trf")
        if best is None or fit.cost < best.cost:
            best = fit
    model = unpack(best.x)
    error = float(np.max(np.abs(model.eps_r(omega) - target) / np.abs(target)))
    return model, error


def _data_file(name: str):
    return importlib.resources.files("hpfem").joinpath("data", f"{name}.csv")


def _tabulated(name: str) -> Tabulated:
    with importlib.resources.as_file(_data_file(name)) as path:
        return Tabulated.from_csv(path, name=name)


def _build_library() -> dict[str, Dispersive]:
    lib: dict[str, Dispersive] = {}
    for name in ("Si", "Au", "Ag", "Al", "GaAs", "MAPbI3"):
        lib[name] = _tabulated(name)
    lib["SiO2"] = Sellmeier(
        0.0,
        (0.6961663, 0.4079426, 0.8974794),
        (0.0684043**2, 0.1162414**2, 9.896161**2),
        range_um=(0.21, 6.7),
        name="SiO2",
        source="I. H. Malitson, J. Opt. Soc. Am. 55, 1205 (1965), doi:10.1364/JOSA.55.001205 "
        "(fused silica, 20 C; validity to 6.7 um after C. Z. Tan 1998)",
    )
    lib["TiO2"] = Sellmeier(
        5.913,
        (0.2441,),
        (0.0803,),
        lam2_numerator=False,
        range_um=(0.43, 1.53),
        name="TiO2",
        source="J. R. Devore, J. Opt. Soc. Am. 41, 416 (1951), doi:10.1364/JOSA.41.000416 "
        "(rutile, ordinary ray, room temperature)",
    )
    lib["vacuum"] = Constant(1.0, name="vacuum", source="definition")
    lib["air"] = Constant(1.000293**2, name="air", source="n = 1.000293 (dry air, visible)")
    lib["water"] = Sellmeier(
        0.0,
        (5.684027565e-1, 1.726177391e-1, 2.086189578e-2, 1.130748688e-1),
        (5.101829712e-3, 1.821153936e-2, 2.620722293e-2, 1.069792721e1),
        range_um=(0.2, 2.0),
        name="water",
        source="M. Daimon and A. Masumura, Appl. Opt. 46, 3811 (2007), doi:10.1364/AO.46.003811 "
        "(distilled water, 20 C)",
    )
    return lib


library: dict[str, Dispersive] = _build_library()
"""Materials by name, see the module docstring for the sources."""


def get(name: str) -> Dispersive:
    """``library[name]`` with a helpful error."""
    try:
        return library[name]
    except KeyError:
        raise KeyError(f"unknown material '{name}'; available: {sorted(library)}") from None


def plasma_frequency(carrier_density: float, effective_mass: float = 1.0) -> float:
    """Drude plasma frequency ``sqrt(N e^2 / (eps0 m* m_e))`` [rad/s] of a carrier density
    [1/m^3] and effective mass in units of the electron mass."""
    m_e = 9.1093837015e-31
    return float(
        np.sqrt(carrier_density * constants.e_charge**2 / (constants.eps0 * effective_mass * m_e))
    )


__all__ = [
    "Dispersive", "Constant", "Tabulated", "Sellmeier", "DrudeLorentz", "Drude",
    "library", "get", "plasma_frequency",
]  # fmt: skip

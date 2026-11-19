"""Data ports of the fusion: emitter response tables and field maps for BindOptiX (issue #214).

BindOptiX (optical biosensors: transport, binding kinetics, camera) computes no light; it reads
maps that the light solvers write in the result envelope ``raytatouille-result`` (Raytatouille
ADR 0023) with the types of ``bindoptix.ports``:

- :class:`EmitterResponseTable` (port M): per emitter position the real symmetric 3×3 tensors
  A_tot (total emitted power), A_rad (power into the far field of both half-spaces) and A_det,d
  (power reaching detector region d), normalised to the free dipole in the host medium; the power
  of a unit orientation μ̂ is μ̂ᵀ A μ̂.
- :class:`FieldMap` (port F): the complex field E on a regular grid per 1 mW incident power.

Positions are in the sensor frame: µm, the surface (the top interface of the stack) at z = 0, the
solution (the cover, the emitter's host) at z > 0, the substrate below. Time factor e^(−iωt).

:func:`planar_emitter_response` computes the table of a planar stack of isotropic layers
semi-analytically (no mesh; a grid along z only, x and y invariant): A_tot from the Sommerfeld
integral of the reflected field at the emitter (Chance–Prock–Silbey, on a path below the real
axis past the guided-mode poles of lossless layers), the complex far field of the unit dipoles
along x, y, z in both half-spaces from the Fresnel coefficients of the stack, and A_rad and A_det
from that far field on a quadrature grid of directions. Conventions in docs/python.md ("Data ports
for BindOptiX").
"""

from __future__ import annotations

import cmath
import hashlib
import json
import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import roots_legendre

import hpfem

FORMAT = "raytatouille-result"
SCHEMA_VERSION = "0.1.0"
SYMMETRY_TOL = 1e-12
PSD_TOL = 1e-9

__all__ = [
    "AxisymmetricEmitterResponse",
    "Detector",
    "EmitterResponseTable",
    "FarField",
    "FieldMap",
    "PlanarEmitterResponse",
    "RegularGrid",
    "decode_array",
    "encode_array",
    "envelope",
    "axisymmetric_emitter_response",
    "axisymmetric_field_map",
    "grating_field_map",
    "meridian_mesh",
    "planar_emitter_response",
    "planar_field_map",
]


# --- the envelope (the codec of bindoptix.results) ------------------------------------------------


def _encode_float(x: float) -> float | str:
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    return float(x)


def encode_array(a) -> dict[str, Any]:
    """Array → ``{"dtype", "shape", "values"}`` in C order; complex values as ``[re, im]``,
    non-finite floats as ``"NaN"`` / ``"Infinity"`` / ``"-Infinity"``."""
    arr = np.ascontiguousarray(a)
    flat = arr.ravel()
    if np.iscomplexobj(arr):
        values: list[Any] = [[_encode_float(v.real), _encode_float(v.imag)] for v in flat]
    elif np.issubdtype(arr.dtype, np.floating):
        values = [_encode_float(float(v)) for v in flat]
    elif np.issubdtype(arr.dtype, np.integer) or arr.dtype == np.bool_:
        values = flat.tolist()
    else:
        raise TypeError(f"unsupported dtype {arr.dtype}")
    return {"dtype": str(arr.dtype), "shape": list(arr.shape), "values": values}


def decode_array(d: dict[str, Any]) -> np.ndarray:
    """Inverse of :func:`encode_array`."""
    special = {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}
    dtype = np.dtype(d["dtype"])
    shape = tuple(int(n) for n in d["shape"])

    def num(v):
        return special[v] if isinstance(v, str) else float(v)

    if np.issubdtype(dtype, np.complexfloating):
        out = np.array([complex(num(v[0]), num(v[1])) for v in d["values"]], dtype=dtype)
    elif np.issubdtype(dtype, np.floating):
        out = np.array([num(v) for v in d["values"]], dtype=dtype)
    else:
        out = np.array(d["values"], dtype=dtype)
    if out.size != math.prod(shape):
        raise ValueError("number of values does not match the shape")
    return out.reshape(shape)


def envelope(type_name: str, data: dict[str, Any]) -> dict[str, Any]:
    """The ``raytatouille-result`` envelope of ``data``."""
    return {"format": FORMAT, "schema_version": SCHEMA_VERSION, "type": type_name, "data": data}


def _to_json(type_name: str, data: dict[str, Any]) -> str:
    return json.dumps(envelope(type_name, data), allow_nan=False, indent=1)


# --- the types of bindoptix.ports -----------------------------------------------------------------


@dataclass(frozen=True)
class RegularGrid:
    """Axis-aligned grid in the sensor frame [µm]: three strictly increasing axes (x, y, z); an
    axis of length 1 means invariance along it."""

    axes_um: tuple

    def __post_init__(self) -> None:
        axes = tuple(np.asarray(a, dtype=np.float64).ravel() for a in self.axes_um)
        if len(axes) != 3:
            raise ValueError("three axes required")
        for a in axes:
            if a.size < 1 or not np.isfinite(a).all():
                raise ValueError("axes must be non-empty and finite")
            if a.size > 1 and not np.all(np.diff(a) > 0.0):
                raise ValueError("axes must be strictly increasing")
        object.__setattr__(self, "axes_um", axes)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(a.size) for a in self.axes_um)

    def to_data(self) -> dict[str, Any]:
        return {"axes_um": [encode_array(a) for a in self.axes_um]}


def _check_psd(name: str, a: np.ndarray, reference: np.ndarray | None = None) -> None:
    """The acceptance of ``bindoptix.ports`` (after BindOptiX #20): finite, symmetric to 1e-12 of
    the largest entry of ``reference`` (default ``a``), eigenvalues of the symmetric part
    ≥ −1e-9 |trace of reference|. The differences A_tot − A_rad and A_rad − ΣA_det are measured
    against A_tot and A_rad: where they vanish physically they are rounding noise."""
    if a.size == 0:
        return
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains non-finite values")
    ref = a if reference is None else reference
    scale = max(float(np.max(np.abs(ref))), 1e-300)
    if np.max(np.abs(a - np.swapaxes(a, -1, -2))) > SYMMETRY_TOL * scale:
        raise ValueError(f"{name} is not symmetric")
    ev = np.linalg.eigvalsh(0.5 * (a + np.swapaxes(a, -1, -2)))
    tr = np.abs(np.trace(ref, axis1=-2, axis2=-1))[..., None]
    worst = float(np.min(ev / np.maximum(tr, 1e-300)))
    if worst < -PSD_TOL:
        raise ValueError(
            f"{name} is not positive semidefinite (smallest eigenvalue {worst:.2e} of the trace)"
        )


@dataclass(frozen=True)
class EmitterResponseTable:
    """Emitter response tensors on a grid (``bindoptix.ports.EmitterResponseTable``):
    ``a_tot``, ``a_rad`` of shape (nx, ny, nz, 3, 3) and ``a_det`` (nx, ny, nz, D, 3, 3), real,
    normalised to the free dipole in the host of real index ``n_host``; ``wavelength_um`` the
    vacuum wavelength; ``source`` the provenance (strings). The constructor applies the checks of
    BindOptiX: symmetric, positive semidefinite, A_det,d summed ≤ A_rad ≤ A_tot."""

    grid: RegularGrid
    wavelength_um: float
    n_host: float
    detectors: tuple
    a_tot: np.ndarray
    a_rad: np.ndarray
    a_det: np.ndarray
    source: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (math.isfinite(self.wavelength_um) and self.wavelength_um > 0.0):
            raise ValueError("wavelength_um must be finite and positive")
        if not (math.isfinite(self.n_host) and self.n_host > 0.0):
            raise ValueError("n_host must be finite and positive")
        g = self.grid.shape
        tot = np.asarray(self.a_tot, dtype=np.float64)
        rad = np.asarray(self.a_rad, dtype=np.float64)
        det = np.asarray(self.a_det, dtype=np.float64)
        if tot.shape != (*g, 3, 3) or rad.shape != (*g, 3, 3):
            raise ValueError("a_tot and a_rad must have shape (nx, ny, nz, 3, 3)")
        if det.shape != (*g, len(self.detectors), 3, 3):
            raise ValueError("a_det must have shape (nx, ny, nz, D, 3, 3)")
        _check_psd("a_tot", tot)
        _check_psd("a_rad", rad)
        _check_psd("a_det", det)
        _check_psd("a_tot - a_rad (absorbed power)", tot - rad, reference=tot)
        _check_psd(
            "a_rad - sum(a_det) (power missing the detectors)", rad - det.sum(axis=3), reference=rad
        )
        object.__setattr__(self, "detectors", tuple(str(d) for d in self.detectors))
        object.__setattr__(self, "a_tot", tot)
        object.__setattr__(self, "a_rad", rad)
        object.__setattr__(self, "a_det", det)
        object.__setattr__(self, "source", {str(k): str(v) for k, v in self.source.items()})

    def to_json(self) -> str:
        """The ``raytatouille-result`` envelope of type ``EmitterResponseTable``."""
        return _to_json(
            "EmitterResponseTable",
            {
                "grid": self.grid.to_data(),
                "wavelength_um": float(self.wavelength_um),
                "n_host": float(self.n_host),
                "detectors": list(self.detectors),
                "a_tot": encode_array(self.a_tot),
                "a_rad": encode_array(self.a_rad),
                "a_det": encode_array(self.a_det),
                "source": dict(self.source),
            },
        )


@dataclass(frozen=True)
class FieldMap:
    """Excitation field on a grid (``bindoptix.ports.FieldMap``, port F): ``e`` complex
    (nx, ny, nz, 3) [V/m] of one incident plane wave, time factor e^(−iωt); ``nonlinear``
    (computed with temperature feedback, then valid only at ``power_mw``); optional absorbed power
    density ``q_w_m3`` and temperature rise ``delta_t_k``, shape (nx, ny, nz).

    Normalisation (BindOptiX #23): the map belongs to a plane wave of irradiance
    ``irradiance_w_m2`` = I₀ = n |E₀|² / (2 Z₀), measured normal to the beam in the incidence
    medium; BindOptiX converts it to "per mW at the input port" with the irradiance at the spot per
    mW from the illumination side. ``k_hat`` (3,) is the unit wavevector of the incident wave in
    the incidence medium, ``polarisation`` (3,) complex its unit field vector (⟂ ``k_hat``), both
    in the sensor frame; ``medium`` (nx, ny, nz) int the medium of each grid point (an index into
    the names ``media``); the phase refers to the incident wave with E = ``polarisation`` at the
    origin (``phase_reference`` in ``source``). ``to_json`` writes them as keys of their own, the
    format of BindOptiX #24. A map with ``irradiance_w_m2`` is linear (``nonlinear`` refused)."""

    grid: RegularGrid
    wavelength_um: float
    e: np.ndarray
    nonlinear: bool = False
    power_mw: float | None = None
    q_w_m3: np.ndarray | None = None
    delta_t_k: np.ndarray | None = None
    source: dict = field(default_factory=dict)
    irradiance_w_m2: float | None = None
    k_hat: np.ndarray | None = None
    polarisation: np.ndarray | None = None
    medium: np.ndarray | None = None
    media: tuple | None = None

    def __post_init__(self) -> None:
        if not (math.isfinite(self.wavelength_um) and self.wavelength_um > 0.0):
            raise ValueError("wavelength_um must be finite and positive")
        e = np.asarray(self.e, dtype=np.complex128)
        if e.shape != (*self.grid.shape, 3) or not np.isfinite(e).all():
            raise ValueError("e must be finite with shape (nx, ny, nz, 3)")
        object.__setattr__(self, "e", e)
        if self.nonlinear and not (self.power_mw is not None and self.power_mw > 0.0):
            raise ValueError("a nonlinear map needs the power it was computed for")
        for name in ("q_w_m3", "delta_t_k"):
            v = getattr(self, name)
            if v is not None:
                arr = np.asarray(v, dtype=np.float64)
                if arr.shape != self.grid.shape or not np.isfinite(arr).all():
                    raise ValueError(f"{name} must be finite with the grid shape")
                object.__setattr__(self, name, arr)
        if self.irradiance_w_m2 is not None and not self.irradiance_w_m2 > 0:
            raise ValueError("irradiance_w_m2 must be positive")
        if self.irradiance_w_m2 is not None and self.nonlinear:
            raise ValueError("a map normalised to an irradiance is linear (nonlinear refused)")
        if self.k_hat is not None:
            k = np.asarray(self.k_hat, dtype=np.float64).reshape(3)
            if abs(np.linalg.norm(k) - 1.0) > 1e-9:
                raise ValueError("k_hat must be a unit vector")
            object.__setattr__(self, "k_hat", k)
        if self.polarisation is not None:
            pol = np.asarray(self.polarisation, dtype=np.complex128).reshape(3)
            if abs(np.linalg.norm(pol) - 1.0) > 1e-9:
                raise ValueError("polarisation must be a unit vector")
            if self.k_hat is not None and abs(np.dot(self.k_hat, pol)) > 1e-9:
                raise ValueError("polarisation must be perpendicular to k_hat")
            object.__setattr__(self, "polarisation", pol)
        if self.medium is not None:
            med = np.asarray(self.medium)
            if med.shape != self.grid.shape or not np.issubdtype(med.dtype, np.integer):
                raise ValueError("medium must be an integer array with the grid shape")
            if self.media is None or med.min() < 0 or med.max() >= len(self.media):
                raise ValueError("medium indexes the names in media")
            object.__setattr__(self, "medium", med.astype(np.int64))
            object.__setattr__(self, "media", tuple(str(m) for m in self.media))
        object.__setattr__(self, "source", {str(k): str(v) for k, v in self.source.items()})

    def to_json(self) -> str:
        """The ``raytatouille-result`` envelope of type ``FieldMap``; ``irradiance_w_m2``,
        ``k_hat``, ``polarisation``, ``medium`` and ``media`` as keys of their own (the format of
        BindOptiX #24; ``null`` / absent when not set), ``phase_reference`` in ``source``."""
        return _to_json(
            "FieldMap",
            {
                "grid": self.grid.to_data(),
                "wavelength_um": float(self.wavelength_um),
                "e": encode_array(self.e),
                "nonlinear": bool(self.nonlinear),
                "power_mw": None if self.power_mw is None else float(self.power_mw),
                "q_w_m3": None if self.q_w_m3 is None else encode_array(self.q_w_m3),
                "delta_t_k": None if self.delta_t_k is None else encode_array(self.delta_t_k),
                "irradiance_w_m2": None
                if self.irradiance_w_m2 is None
                else float(self.irradiance_w_m2),
                "k_hat": None if self.k_hat is None else encode_array(self.k_hat),
                "polarisation": None
                if self.polarisation is None
                else encode_array(self.polarisation),
                "medium": None if self.medium is None else encode_array(self.medium),
                "media": None if self.media is None else list(self.media),
                "source": dict(self.source),
            },
        )


# --- planar stacks: Fresnel coefficients -------------------------------------------------------


def _kz(n, kt):
    """Normal wavenumber sqrt(n² − kt²) (units of k0) on the decaying / outgoing branch, continued
    along the path below the real axis as in the M17 reference; scalars or arrays."""
    z = np.sqrt(np.asarray(n * n - kt * kt, dtype=complex))
    return np.where(z.real + z.imag < 0, -z, z)


def _fresnel(s, n1: float, layers, n3: complex, k0: float):
    """r_s, r_p, t_s, t_p of the stack seen from the cover n1 at the in-plane wavenumbers s·n1·k0
    (scalar or array): the amplitudes of the E field in the bases ŝ = ẑ × k̂_∥, p̂ = ŝ × k̂ of each
    wave (r_p is then the reflection of H, +1 for a perfect conductor). t refers the transmitted
    amplitude at the bottom interface to the incident one at the top interface. Recursion from the
    substrate up (stable for thick metal layers)."""
    kt = np.asarray(s, dtype=complex) * n1
    media = [complex(n1)] + [complex(n) for n, _d in layers] + [complex(n3)]
    out = []
    for pol in ("s", "p"):
        r = t = None
        for i in range(len(media) - 2, -1, -1):
            na, nb = media[i], media[i + 1]
            ka, kb = _kz(na, kt), _kz(nb, kt)
            if pol == "s":
                rab = (ka - kb) / (ka + kb)
                tab = 2 * ka / (ka + kb)
            else:  # the H field
                rab = (nb * nb * ka - na * na * kb) / (nb * nb * ka + na * na * kb)
                tab = 2 * nb * nb * ka / (nb * nb * ka + na * na * kb)
            if r is None:
                r, t = rab, tab
            else:
                half = np.exp(1j * k0 * kb * layers[i][1])
                den = 1 + rab * r * half * half
                r, t = (rab + r * half * half) / den, tab * t * half / den
        out.append((r, t))
    (rs, ts), (rp, tp_h) = out
    return rs, rp, ts, tp_h * media[0] / media[-1]  # H = (n/Z0) E_p: t_E = t_H n1 / n3


def _purcell(n1, layers, n3, k0, heights, depth, points) -> np.ndarray:
    """(F_⊥, F_∥) per height (nz, 2) of a point dipole in n1 above the stack
    (Chance–Prock–Silbey): F_⊥ = 1 + (3/2) Re ∫ s³/s_z r_p e^{2ik1h s_z} ds, F_∥ = 1 + (3/4) Re
    ∫ s/s_z (r_s − s_z² r_p) e^{2ik1h s_z} ds; s on t − i·depth·sin(πt/S) up to S beyond the
    largest index (the guided poles of lossless layers on the causal side), then on the real axis
    to where e^{2ik1h s_z} < e^{−40}."""
    k1 = n1 * k0
    # beyond the largest real index: the guided poles of lossless layers and the plasmon poles of
    # metals (Re n small, |n| large) lie below it; a path ending at |n| of a metal would stretch
    # far and pass the branch point s = 1 too closely
    s_top = max([complex(n3).real, n1] + [complex(n).real for n, _d in layers]) / n1 + 0.3
    t, w = roots_legendre(points)
    t = 0.5 * s_top * (t + 1.0)
    w = 0.5 * s_top * w
    s = t - 1j * depth * np.sin(np.pi * t / s_top)
    ds = 1.0 - 1j * depth * (np.pi / s_top) * np.cos(np.pi * t / s_top)
    sz = _kz(1.0, s)
    rs, rp, _ts, _tp = _fresnel(s, n1, layers, n3, k0)
    perp_kernel = w * ds * s**3 / sz * rp
    par_kernel = w * ds * s / sz * (rs - sz * sz * rp)
    # the tail s = S + x / (2 k1 h), x in [0, 50]: e^{2ik1h s_z} ~ e^{−x}; composite Gauss on panels
    # graded towards x = 0, where the integrand varies on the scale 2 k1 h of small heights
    g, gw = roots_legendre(16)
    edges = np.concatenate([[0.0], np.geomspace(1e-3, 50.0, 48)])
    panels = list(zip(edges[:-1], edges[1:], strict=True))
    x = np.concatenate([0.5 * (b - a) * (g + 1.0) + a for a, b in panels])
    xw = np.concatenate([0.5 * (b - a) * gw for a, b in panels])
    out = np.zeros((len(heights), 2))
    for iz, h in enumerate(heights):
        e = np.exp(2j * k1 * h * sz)
        sv = s_top + x / (2 * k1 * h)
        szt = 1j * np.sqrt(sv * sv - 1)
        rst, rpt, _a, _b = _fresnel(sv, n1, layers, n3, k0)
        e_t = np.exp(2j * k1 * h * szt) * xw / (2 * k1 * h)
        tail_perp = np.sum(sv**3 / szt * rpt * e_t).real
        tail_par = np.sum(sv / szt * (rst - szt * szt * rpt) * e_t).real
        out[iz, 0] = 1 + 1.5 * (np.sum(perp_kernel * e).real + tail_perp)
        out[iz, 1] = 1 + 0.75 * (np.sum(par_kernel * e).real + tail_par)
    return out


# --- planar stacks: the far field ---------------------------------------------------------------


@dataclass
class Detector:
    """A detector region of the far field: the directions of one half-space (``side``
    ``"cover"``, the solution, or ``"substrate"``) with the polar angle in [``theta_min``,
    ``theta_max``] [rad] from the outward normal in that medium (+z in the cover, −z in the
    substrate). ``jones(theta, phi)`` (optional) returns the 2×2 Jones matrix from the far-field
    components (E·ŝ, E·p̂) to the detector (e.g. an analyser, a transmission, the K5 chain's J_d);
    without it the region collects all power. Regions must not overlap."""

    name: str
    side: str
    theta_min: float = 0.0
    theta_max: float = 0.5 * math.pi
    jones: Callable | None = None


@dataclass
class FarField:
    """The far field of the unit dipoles in one half-space on the quadrature grid of directions:
    ``theta`` (from the outward normal of the medium), ``phi`` (azimuth from x), ``weights`` the
    solid-angle weights of the (theta, phi) nodes, ``field`` (nz, ntheta, nphi, 3, 3) complex: the
    vector amplitude E∞ (x, y, z) of the dipole along x, y, z (fourth index) at every height,
    with E(r) → E∞ e^{ikr}/r, in units of k1² p / (4π ε0 ε1) (the free dipole has E∞ = (I − k̂k̂)p);
    the phase refers to the point of the surface (z = 0) below the emitter, in the substrate to the
    plane waves continued from the bottom interface to z = 0. ``basis`` (ntheta, nphi, 3, 2): ŝ, p̂
    of each direction. The power through the half-space is ``factor`` · Σ weights |E∞|² in units
    of the free dipole's (``factor`` = 3/(8π) · n/n_host)."""

    side: str
    index: float
    theta: np.ndarray
    phi: np.ndarray
    weights: np.ndarray
    field: np.ndarray
    basis: np.ndarray
    factor: float


@dataclass
class PlanarEmitterResponse:
    """Result of :func:`planar_emitter_response`: the ``table`` (:class:`EmitterResponseTable`),
    the far fields per half-space (``far_field["cover"]``, ``far_field["substrate"]`` when the
    substrate is lossless) and the convergence data also stored in ``table.source``."""

    table: EmitterResponseTable
    far_field: dict
    convergence: dict


def _media(stack):
    """(n1, layers [(n, d)], n3, bottom − top) of an isotropic LayerStack2D / LayerStack3D."""
    if isinstance(stack, hpfem.AnisotropicStack):
        raise ValueError("planar_emitter_response takes a LayerStack2D of isotropic media")
    materials = [stack.material(i) for i in range(stack.num_layers + 2)]
    if not all(m.is_isotropic for m in materials):
        raise ValueError("planar_emitter_response: crystal media are not supported")
    n = [complex(m.refractive_index) for m in materials]
    if abs(n[0].imag) > 1e-12 or n[0].real <= 0:
        raise ValueError("the cover (the emitter's host) must be lossless")
    layers = [
        (n[i], float(stack.interface(i - 1) - stack.interface(i)))
        for i in range(1, stack.num_layers + 1)
    ]
    return n[0].real, layers, n[-1], float(stack.bottom - stack.top)


def _theta_grid(breaks: Sequence[float], points: int):
    """Cosine-mapped Gauss–Legendre nodes and weights in theta on each interval between the sorted
    breaks, θ = a + (b − a)(1 − cos πu)/2: the nodes cluster at both ends, so the square-root
    behaviour at the critical angle (s_z → 0) does not spoil the convergence."""
    g, w = roots_legendre(points)
    u = 0.5 * (g + 1.0)
    nodes, weights = [], []
    for a, b in zip(breaks[:-1], breaks[1:], strict=True):
        if b - a <= 1e-15:
            continue
        nodes.append(a + (b - a) * 0.5 * (1.0 - np.cos(np.pi * u)))
        weights.append((b - a) * 0.25 * np.pi * np.sin(np.pi * u) * w)
    return np.concatenate(nodes), np.concatenate(weights)


def _side_grid(side: str, base: list[float], detectors, theta_points: int, phi_points: int):
    breaks = set(base)
    for d in detectors:
        if d.side == side:
            breaks.update((float(d.theta_min), float(d.theta_max)))
    breaks = sorted(b for b in breaks if 0.0 <= b <= 0.5 * math.pi)
    theta, w_theta = _theta_grid(breaks, theta_points)
    phi = 2 * math.pi * np.arange(phi_points) / phi_points
    weights = np.outer(w_theta * np.sin(theta), np.full(phi_points, 2 * math.pi / phi_points))
    return theta, phi, weights


def _bases(theta, phi, down: bool):
    """k̂ and the basis ŝ = ẑ × k̂_∥ / |·|, p̂ = ŝ × k̂ per direction, (ntheta, nphi, 3)."""
    st, ct = np.sin(theta)[:, None], np.cos(theta)[:, None]
    cp, sp = np.cos(phi)[None, :], np.sin(phi)[None, :]
    one = np.ones_like(st * cp)
    k = np.stack([st * cp, st * sp, (-ct if down else ct) * one], axis=-1)
    s = np.stack([-sp * one, cp * one, 0.0 * one], axis=-1)
    return k, s, np.cross(s, k)


def _field_at(side: str, t: float, phi, n1, layers, n3, z_bottom, k0, heights) -> np.ndarray:
    """E∞ (nz, nphi, 3 dipoles, 3 components) at the polar angle t of one half-space.

    Cover (direct and reflected wave, φ = k1 h cos t): E∞ = ŝ(ŝ·p)(e^{−iφ} + r_s e^{iφ})
    + p̂_u((p̂_u·p) e^{−iφ} + r_p (p̂_d·p) e^{iφ}).
    Substrate (t from −z, s = n3 sin t / n1, s_z = sqrt(1 − s²), imaginary beyond the critical
    angle: the supercritical light): E∞ = (n3 cos t / (n1 s_z)) [ŝ(ŝ·p) t_s + p̂3 (p̂_d·p) t_p]
    e^{i k1 h s_z} e^{i k3 cos t (bottom − top)} (stationary phase of the transmitted spectrum)."""
    heights = np.asarray(heights, dtype=float)
    k, s_vec, p_out = _bases(np.array([t]), phi, down=side == "substrate")
    s_vec, p_out = s_vec[0], p_out[0]
    if side == "cover":
        s = math.sin(t)
        sz = complex(math.cos(t))
        rs, rp, _ts, _tp = _fresnel(s, n1, layers, n3, k0)
    else:
        n3r = complex(n3).real
        s = n3r * math.sin(t) / n1
        sz = _kz(1.0, s)
        _rs, _rp, ts, tp = _fresnel(s, n1, layers, n3r, k0)
    # the downward wave of the cover at this in-plane vector (complex s_z beyond the light cone)
    kd = np.stack([s * np.cos(phi), s * np.sin(phi), -sz * np.ones_like(phi)], axis=-1)
    p_d = np.cross(s_vec, kd)  # (nphi, 3)
    down = np.exp(1j * k0 * n1 * heights * sz)  # the downward wave at z = 0
    out = np.zeros((heights.size, len(phi), 3, 3), dtype=complex)
    for j in range(3):
        s_j = s_vec[:, j][None, :, None]
        pd_j = p_d[:, j][None, :, None]
        if side == "cover":
            up = np.conj(down)[:, None, None]  # e^{−iφ}, φ real
            dn = down[:, None, None]
            pu_j = p_out[:, j][None, :, None]
            out[:, :, j, :] = s_vec[None] * s_j * (up + rs * dn) + p_out[None] * (
                pu_j * up + rp * pd_j * dn
            )
        else:
            amp = (n3r * math.cos(t) / (n1 * sz)) * cmath.exp(
                1j * k0 * n3r * math.cos(t) * z_bottom
            )
            dn = (amp * down)[:, None, None]
            out[:, :, j, :] = dn * (s_vec[None] * s_j * ts + p_out[None] * pd_j * tp)
    return out


def _far_fields(n1, layers, n3, z_bottom, k0, heights, detectors, theta_points, phi_points):
    """The FarField of the cover and, for a lossless substrate, of the substrate."""
    sides = [("cover", n1, [0.0, 0.5 * math.pi])]
    n3c = complex(n3)
    if abs(n3c.imag) <= 1e-12 and n3c.real > 0:
        base = [0.0, 0.5 * math.pi]
        if n3c.real > n1:
            base.append(math.asin(n1 / n3c.real))  # the critical angle
        sides.append(("substrate", n3c.real, base))
    out = {}
    for side, index, base in sides:
        theta, phi, weights = _side_grid(side, base, detectors, theta_points, phi_points)
        field_ = np.stack(
            [_field_at(side, t, phi, n1, layers, n3, z_bottom, k0, heights) for t in theta],
            axis=1,
        )
        _k, s_vec, p_vec = _bases(theta, phi, down=side == "substrate")
        out[side] = FarField(side, index, theta, phi, weights, field_,
                             np.stack([s_vec, p_vec], axis=-1),
                             3 / (8 * math.pi) * index / n1)  # fmt: skip
    return out


def _adaptive_diagonal(ff: FarField, base, n1, layers, n3, z_bottom, k0, heights) -> np.ndarray:
    """The diagonal of the half-space's A_rad (nz, 3) with θ integrated adaptively (quad_vec, the
    azimuth on the same nodes): the reference for the Gauss grid."""
    w_phi = 2 * math.pi / len(ff.phi)

    def integrand(t):
        e = _field_at(ff.side, t, ff.phi, n1, layers, n3, z_bottom, k0, heights)
        return np.sum(np.abs(e) ** 2, axis=(1, 3)) * w_phi * math.sin(t)

    return ff.factor * sum(
        quad_vec(integrand, a, b, epsabs=0.0, epsrel=1e-11, limit=400)[0]
        for a, b in zip(base[:-1], base[1:], strict=True)
    )


def _gram(ff: FarField, mask=None, jones=None) -> np.ndarray:
    """Re Σ w conj(E_i)·E_j of the far field (nz, 3, 3), optionally restricted by a node mask and
    passed through Jones matrices acting on the (ŝ, p̂) components."""
    e = ff.field  # (nz, nt, np, j, c)
    w = ff.weights if mask is None else ff.weights * mask
    if jones is not None:
        comp = np.einsum("tpcb,ztpjc->ztpjb", ff.basis, e)  # (E·ŝ, E·p̂)
        e = np.einsum("tpab,ztpjb->ztpja", jones, comp)
    return ff.factor * np.einsum("tp,ztpic,ztpjc->zij", w, e.conj(), e).real


def _symmetrise(a: np.ndarray) -> tuple[np.ndarray, float]:
    """The symmetric part and the relative asymmetry max|A − Aᵀ| / max|A| (convergence)."""
    asym = float(np.max(np.abs(a - np.swapaxes(a, -1, -2)))) / max(float(np.max(np.abs(a))), 1e-300)
    return 0.5 * (a + np.swapaxes(a, -1, -2)), asym


def _stack_provenance(n1, layers, n3, wavelength) -> dict[str, str]:
    """``stack``: the stack in plain text as JSON (sorted keys) — ``{"cover": [n1, 0.0],
    "layers": [[Re n, Im n, thickness_m], …] (top to bottom), "substrate": [Re n, Im n],
    "wavelength_m": λ}`` — so that the table can be recomputed (BindOptiX T4); ``content_hash``:
    the SHA-256 of exactly that string (UTF-8)."""
    data = {
        "cover": [float(n1), 0.0],
        "layers": [[complex(n).real, complex(n).imag, float(d)] for n, d in layers],
        "substrate": [complex(n3).real, complex(n3).imag],
        "wavelength_m": float(wavelength),
    }
    text = json.dumps(data, sort_keys=True)
    return {"stack": text, "content_hash": hashlib.sha256(text.encode()).hexdigest()}


def planar_emitter_response(
    stack,
    wavelength: float,
    heights,
    *,
    detectors: Sequence[Detector] | None = None,
    theta_points: int = 96,
    phi_points: int = 16,
    depth: float = 0.02,
    points: int = 4000,
    source: dict | None = None,
) -> PlanarEmitterResponse:
    """The emitter response table of a planar stack of isotropic media (issue #214).

    ``stack``: ``hpfem.LayerStack2D`` (or 3D) with the solution as the incidence medium (the
    emitter's host, lossless), the layers and the substrate below; the top interface is the
    surface z = 0 of the sensor frame. ``wavelength``: vacuum wavelength [m]. ``heights``: the
    emitter heights above the surface [m] (> 0, strictly increasing); the table's grid is
    (0, 0, heights in µm), x and y invariant. ``detectors``: :class:`Detector` regions of the far
    field (A_det,d, in the order given; default: the whole cover ``"solution"`` and the whole
    substrate ``"substrate"``; at least one, BindOptiX reads no empty detector axis).

    A_tot = diag(F_∥, F_∥, F_⊥) from the Sommerfeld integrals (Chance–Prock–Silbey; ``points``
    Gauss nodes on a path ``depth`` below the real axis past the guided-mode poles, then a graded
    Gauss rule on the evanescent tail). The far field of the unit dipoles x, y, z
    (:class:`FarField`) in the cover and, for a lossless substrate, in the substrate (including
    the supercritical directions beyond the critical angle, the coupled surface-plasmon emission
    of a metal film) on cosine-mapped Gauss nodes in θ (``theta_points`` per interval between 0,
    the critical angle, the detector edges and π/2) times ``phi_points`` azimuths; A_rad and A_det
    are its quadratic forms on that grid, so A_det ≤ A_rad holds by construction. A lossy
    substrate takes no far field (its power counts as absorbed).

    A_tot − A_rad < 0 beyond 1e-9 tr A_tot (the import check of BindOptiX) raises ``ValueError``
    with the hint to refine the quadratures.

    Convergence (in ``convergence`` and ``table.source["convergence"]``): the asymmetry of A_rad,
    ``radiated_check`` = the largest deviation of the diagonal of A_rad from an adaptive
    integration of the same far field, ``energy`` = min eig(A_tot − A_rad) / tr A_tot (≈ 0 for a
    lossless stack without guided modes, > 0 with absorption or guided modes). ``source``: extra
    provenance entries (e.g. ``"commit"``), merged over the defaults (solver, version, method,
    ``stack``: the stack and wavelength in plain text as JSON, ``content_hash``: its SHA-256)."""
    n1, layers, n3, z_bottom = _media(stack)
    heights = np.asarray(heights, dtype=float).ravel()
    if heights.size < 1 or np.any(heights <= 0) or np.any(np.diff(heights) <= 0):
        raise ValueError("heights must be positive and strictly increasing")
    if detectors is None:
        detectors = [Detector("solution", "cover"), Detector("substrate", "substrate")]
    detectors = list(detectors)
    if not detectors:
        raise ValueError("at least one detector region is needed (BindOptiX reads no empty axis)")
    for d in detectors:
        if d.side not in ("cover", "substrate"):
            raise ValueError(f"detector {d.name!r}: side must be 'cover' or 'substrate'")
        if not 0.0 <= d.theta_min < d.theta_max <= 0.5 * math.pi + 1e-15:
            raise ValueError(f"detector {d.name!r}: need 0 <= theta_min < theta_max <= pi/2")
    for side in ("cover", "substrate"):
        spans = sorted((d.theta_min, d.theta_max, d.name) for d in detectors if d.side == side)
        for (_a0, b0, n0), (a1, _b1, n1_) in zip(spans[:-1], spans[1:], strict=True):
            if a1 < b0:
                raise ValueError(f"the detector regions {n0!r} and {n1_!r} overlap")
    k0 = 2 * math.pi / float(wavelength)

    purcell = _purcell(n1, layers, n3, k0, heights, depth, points)
    a_tot = np.zeros((heights.size, 3, 3))
    for iz, (f_perp, f_par) in enumerate(purcell):
        a_tot[iz] = np.diag([f_par, f_par, f_perp])

    far = _far_fields(n1, layers, n3, z_bottom, k0, heights, detectors, theta_points, phi_points)
    a_rad, asymmetry = _symmetrise(sum(_gram(ff) for ff in far.values()))
    a_det = np.zeros((heights.size, len(detectors), 3, 3))
    for k, d in enumerate(detectors):
        ff = far.get(d.side)
        if ff is None:
            continue  # a lossy substrate radiates nothing into it
        inside = (ff.theta >= d.theta_min) & (ff.theta <= d.theta_max)
        mask = np.repeat(inside[:, None].astype(float), len(ff.phi), axis=1)
        jones = None
        if d.jones is not None:
            jones = np.array([[np.asarray(d.jones(t, p), dtype=complex) for p in ff.phi]
                              for t in ff.theta])  # fmt: skip
        a_det[:, k], _ = _symmetrise(_gram(ff, mask, jones))
    # convergence: the diagonal of A_rad against an adaptive integration of the same far field
    check = 0.0
    for ff in far.values():
        base = [0.0, 0.5 * math.pi]
        if ff.side == "substrate" and ff.index > n1:
            base = sorted(base + [math.asin(n1 / ff.index)])
        grid = np.einsum("tp,ztpjc->zj", ff.weights, np.abs(ff.field) ** 2) * ff.factor
        exact = _adaptive_diagonal(ff, base, n1, layers, n3, z_bottom, k0, heights)
        check = max(check, float(np.max(np.abs(grid - exact))))
    balance = np.linalg.eigvalsh(a_tot - a_rad) / np.trace(a_tot, axis1=1, axis2=2)[:, None]

    if float(np.min(balance)) < -PSD_TOL:
        raise ValueError(
            f"A_tot < A_rad by {-float(np.min(balance)):.2e} of the trace, beyond the import "
            f"tolerance of BindOptiX (quadrature error {check:.2e}): raise theta_points (the "
            "far-field grid) or points (the Sommerfeld path)"
        )
    convergence = {
        "asymmetry_a_rad": asymmetry,
        "radiated_check": check,
        "energy": float(np.min(balance)),
        "theta_points": int(theta_points),
        "phi_points": int(phi_points),
        "path_depth": float(depth),
        "path_points": int(points),
    }
    provenance = {
        "solver": "hp-FEM",
        "version": hpfem.__version__,
        "method": "hpfem.dataports.planar_emitter_response (Sommerfeld / Fresnel, semi-analytic)",
        **_stack_provenance(n1, layers, n3, float(wavelength)),
        "convergence": json.dumps(convergence, sort_keys=True),
    }
    provenance.update(source or {})
    table = EmitterResponseTable(
        grid=RegularGrid(([0.0], [0.0], heights * 1e6)),
        wavelength_um=float(wavelength) * 1e6,
        n_host=n1,
        detectors=tuple(d.name for d in detectors),
        a_tot=a_tot[None, None],
        a_rad=a_rad[None, None],
        a_det=a_det[None, None],
        source=provenance,
    )
    return PlanarEmitterResponse(table=table, far_field=far, convergence=convergence)


# --- field maps (port F) -------------------------------------------------------------------------

_PHASE_REFERENCE = "incident wave: E = polarisation at (0, 0, 0) um, time factor exp(-i omega t)"
_MIRROR_Z = np.array([1.0, 1.0, -1.0])


def _check_surface_plane(grid: RegularGrid) -> None:
    """Warns when the z axis spans the surface without a plane exactly at z = 0: BindOptiX
    evaluates F at z ≥ 0 only and refuses points whose interpolation cell spans two media, so
    the surface needs a grid plane of its own (e.g. ``np.linspace`` from −a to b rarely hits 0
    exactly; build the axis from both sides of 0 instead)."""
    z = np.asarray(grid.axes_um[2])
    if z.size > 1 and z.min() < 0.0 < z.max() and not np.any(z == 0.0):
        warnings.warn(
            f"the z axis spans the surface but has no plane at exactly z = 0 (closest "
            f"{z[np.argmin(np.abs(z))]:.3g} um): BindOptiX refuses points near the surface "
            "whose interpolation cell reaches into the layer below",
            stacklevel=3,
        )


def _grid_points(grid: RegularGrid) -> np.ndarray:
    """The grid points (nx, ny, nz, 3) [m]."""
    x, y, z = (np.asarray(a) * 1e-6 for a in grid.axes_um)
    return np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1)


def _unit_incidence(k_hat, polarisation):
    k = np.asarray(k_hat, dtype=float).reshape(3)
    if not np.linalg.norm(k) > 0:
        raise ValueError("k_hat must not vanish")
    k = k / np.linalg.norm(k)
    pol = np.asarray(polarisation, dtype=complex).reshape(3)
    if not np.linalg.norm(pol) > 0:
        raise ValueError("polarisation must not vanish")
    pol = pol / np.linalg.norm(pol)
    if abs(np.dot(k, pol)) > 1e-9:
        raise ValueError("polarisation must be perpendicular to k_hat (a transverse plane wave)")
    return k, pol


def _absorbed_density(e: np.ndarray, eps: np.ndarray, omega: float) -> np.ndarray:
    """q = (ω ε0 / 2) E^H Im_h(ε) E with Im_h(ε) = (ε − ε^H)/(2i) per point; ``eps`` (…, 3, 3)."""
    im = (eps - np.conj(np.swapaxes(eps, -1, -2))) / 2j
    return 0.5 * omega * hpfem.constants.eps0 * np.einsum("...i,...ij,...j->...", e.conj(), im,
                                                          e).real  # fmt: skip


@dataclass
class _PlaneWaves:
    """The s and p stack waves of one incidence (sensor frame: ``lab`` the stack with its top
    interface at z = 0; for incidence from the substrate ``frame`` is the stack mirrored at
    z = 0 and ``mirror`` = diag(1, 1, −1)); ``basis`` (3, 2): their field vectors at the origin."""

    lab: object
    frame: object
    mirror: np.ndarray
    waves: list
    basis: np.ndarray
    n_inc: float
    from_substrate: bool
    materials: list
    angle: float


def _stack_plane_waves(stack, wavelength: float, k: np.ndarray) -> _PlaneWaves:
    _media(stack)  # isotropic media, lossless cover
    materials = [stack.material(i) for i in range(stack.num_layers + 2)]
    thickness = [float(stack.interface(i - 1) - stack.interface(i))
                 for i in range(1, stack.num_layers + 1)]  # fmt: skip
    if abs(k[2]) < 1e-12:
        raise ValueError("k_hat must have a normal component (grazing incidence)")
    from_substrate = bool(k[2] > 0)
    lab = hpfem.LayerStack3D(materials[0], [hpfem.Layer(m, d) for m, d in
                                            zip(materials[1:-1], thickness, strict=True)],
                             materials[-1], 0.0)  # fmt: skip
    if from_substrate:
        sub = complex(materials[-1].refractive_index)
        if abs(sub.imag) > 1e-12:
            raise ValueError("incidence from the substrate needs a lossless substrate")
        # the stack mirrored at z = 0: the substrate on top, its top interface at the depth
        frame = hpfem.LayerStack3D(materials[-1], [hpfem.Layer(m, d) for m, d in
                                                   zip(materials[-2:0:-1], thickness[::-1],
                                                       strict=True)],
                                   materials[0], float(sum(thickness)))  # fmt: skip
        mirror = _MIRROR_Z
        n_inc = sub.real
    else:
        frame, mirror, n_inc = lab, np.ones(3), complex(materials[0].refractive_index).real
    k_frame = k * mirror  # pointing down in the frame of the incidence
    angle = math.acos(min(1.0, -k_frame[2]))
    azimuth = math.atan2(k_frame[1], k_frame[0]) if math.hypot(k_frame[0], k_frame[1]) > 0 else 0.0
    k0 = 2 * math.pi / float(wavelength)
    waves = [frame.plane_wave(k0, angle, p, 1.0, azimuth)
             for p in (hpfem.Polarisation.S, hpfem.Polarisation.P)]  # fmt: skip
    basis = np.column_stack([mirror * np.asarray(w.incident_wave.value([0.0, 0.0, 0.0]),
                                                 dtype=complex) for w in waves])  # fmt: skip
    return _PlaneWaves(lab, frame, mirror, waves, basis, n_inc, from_substrate, materials, angle)


def planar_field_map(
    stack,
    wavelength: float,
    grid: RegularGrid,
    k_hat,
    polarisation,
    *,
    media: Sequence[str] | None = None,
    absorbed: bool = True,
    source: dict | None = None,
) -> FieldMap:
    """Port F of a planar stack of isotropic media for one incident plane wave (BindOptiX #23):
    the exact stack field on ``grid`` (sensor frame, µm: the top interface at z = 0, the solution
    above it) for the wave of unit amplitude with the unit wavevector ``k_hat`` (sensor frame) and
    the field vector ``polarisation`` (complex, ⟂ ``k_hat``; normalised). ``k_hat`` pointing down
    (k_z < 0) comes from the solution, pointing up from the substrate (lossless; the Kretschmann
    configuration through the glass).

    The map carries ``irradiance_w_m2`` = n |E₀|²/(2 Z₀) of the incidence medium (|E₀| = 1 V/m),
    ``k_hat``, ``polarisation``, ``medium`` (0 the solution, 1..N the layers, N + 1 the substrate;
    names ``media``, default ``solution``, ``layer 1``, …, ``substrate``) and, with ``absorbed``,
    ``q_w_m3`` = (ω ε0/2) Im ε |E|² at that irradiance. On an interface the value is the limit
    from above (the solution side at z = 0, as BindOptiX evaluates F at z ≥ 0 only). The phase
    refers to the incident wave with E = polarisation at the origin (continued analytically into
    the solution or the substrate)."""
    _check_surface_plane(grid)
    k, pol = _unit_incidence(k_hat, polarisation)
    pw = _stack_plane_waves(stack, wavelength, k)
    lab, mirror, waves, n_inc = pw.lab, pw.mirror, pw.waves, pw.n_inc
    from_substrate = pw.from_substrate
    k0 = 2 * math.pi / float(wavelength)
    materials = pw.materials
    coefficients, *_ = np.linalg.lstsq(pw.basis, pol, rcond=None)
    if np.linalg.norm(pw.basis @ coefficients - pol) > 1e-9:
        raise ValueError("the polarisation is not transverse to the incident wave")
    points = _grid_points(grid)
    flat = points.reshape(-1, 3)
    interfaces = [float(lab.interface(i)) for i in range(lab.num_layers + 1)]
    delta = 1e-12 * float(wavelength)
    e = np.zeros((flat.shape[0], 3), dtype=complex)
    medium = np.zeros(flat.shape[0], dtype=np.int64)
    eps = np.zeros((flat.shape[0], 3, 3), dtype=complex)
    tensors = [np.asarray(m.eps_tensor, dtype=complex) for m in materials]
    for i, r in enumerate(flat):
        z = float(r[2])
        if any(z == zi for zi in interfaces):
            z += delta  # the limit from above (the solution side)
        q = np.array([r[0], r[1], z]) * mirror
        value = sum(c * np.asarray(w.field.value(q), dtype=complex)
                    for c, w in zip(coefficients, waves, strict=True))  # fmt: skip
        e[i] = mirror * value
        medium[i] = int(lab.region(float(r[2])))
        eps[i] = tensors[medium[i]]
    irradiance = n_inc / (2 * hpfem.constants.Z0)
    names = (
        media
        if media is not None
        else (["solution"] + [f"layer {i}" for i in range(1, lab.num_layers + 1)] + ["substrate"])
    )
    if len(names) != lab.num_layers + 2:
        raise ValueError(f"media needs {lab.num_layers + 2} names (solution, layers, substrate)")
    provenance = {
        "solver": "hp-FEM",
        "version": hpfem.__version__,
        "method": "hpfem.dataports.planar_field_map (stack plane wave, exact)",
        **_stack_provenance(*_media(stack)[:3], float(wavelength)),
        "phase_reference": _PHASE_REFERENCE,
        "incidence": "substrate" if from_substrate else "solution",
    }
    provenance.update(source or {})
    shape = grid.shape
    return FieldMap(
        grid=grid,
        wavelength_um=float(wavelength) * 1e6,
        e=e.reshape(*shape, 3),
        q_w_m3=_absorbed_density(e, eps, k0 * hpfem.constants.c0).reshape(shape)
        if absorbed
        else None,
        source=provenance,
        irradiance_w_m2=irradiance,
        k_hat=k,
        polarisation=pol,
        medium=medium.reshape(shape),
        media=tuple(names),
    )


def grating_field_map(
    result,
    grid: RegularGrid,
    *,
    media: dict | None = None,
    absorbed: bool = True,
    source: dict | None = None,
) -> FieldMap:
    """Port F of a periodic structure from a solved grating cell (``hpfem.grating.solve``, any
    polarisation and incidence from the cover, BindOptiX #23).

    Frames: the solver frame of the cell (x across the lines, y the normal, z along them) is the
    sensor frame (x across, y along the lines, z the normal) with y and z exchanged, the top
    interface of the stack at the sensor z = 0, the cover the solution. The total field at the
    sensor point (x, y, z) is P E_cell(x', z) e^{i kx (x − x')} e^{i β y} with x' in the cell
    (the Bloch periodicity across the lines, the invariance along them) and P the exchange of the
    y and z components. Grid points must lie in the cell's height range outside the PML; on a
    horizontal mesh line the value is the limit from above (the solution side at z = 0).

    The map carries ``irradiance_w_m2`` = n_c / (2 Z₀) of the unit incident wave, ``k_hat`` and
    ``polarisation`` of the incident wave (sensor frame, E = polarisation at the origin: the
    phase reference of the stack wave, x = 0 on the top interface), ``medium`` by cell tag (names
    from ``media`` {tag: name}, default ``"tag <n>"``) and, with ``absorbed``, ``q_w_m3`` =
    (ω ε0/2) E^H Im ε E of each cell's material (W/m³ at that irradiance)."""
    wave = result.wave
    inputs = result.inputs or {}
    omega = float(inputs.get("omega", 0.0))
    if not omega > 0:
        raise ValueError("the grating result carries no frequency (solve it with grating.solve)")
    if inputs.get("side") == "substrate":
        raise ValueError("grating_field_map takes a result of incidence from the cover")
    _check_surface_plane(grid)
    problem, mesh = result.problem, result.mesh
    top = float(problem.setup.background.top)
    k0 = omega / hpfem.constants.c0
    n_c = complex(problem.setup.background.incidence_medium.refractive_index).real
    period, x_min = float(result.period), float(result.x_min)
    box = result.pml
    y_lo, y_hi = float(box.lower[1]), float(box.upper[1])
    points = _grid_points(grid)
    flat = points.reshape(-1, 3)
    y_solver = flat[:, 2] + top
    if np.any(y_solver < y_lo - 1e-12 * period) or np.any(y_solver > y_hi + 1e-12 * period):
        raise ValueError("grid points outside the cell's physical region (in the PML or beyond)")
    # horizontal mesh lines: the limit from above
    lines = {round(float(mesh.vertex(v)[1]) / period, 12) for v in range(mesh.num_vertices)}
    on_line = np.array([round(y / period, 12) in lines for y in y_solver])
    y_eval = np.where(on_line, y_solver + 1e-9 * period, y_solver)
    shift = np.floor((flat[:, 0] - x_min) / period)
    x_cell = flat[:, 0] - shift * period
    cell_points = np.column_stack([x_cell, y_eval])
    e_cell = np.asarray(result.field(cell_points, "E"), dtype=complex)  # solver frame, physical
    phase = np.exp(1j * wave.kx * shift * period) * np.exp(1j * wave.beta * flat[:, 1])
    e_cell = e_cell * phase[:, None]
    # materials per point (the cell the solver used)
    locator = hpfem.PointLocator2D(mesh)
    materials = problem.setup.materials
    tags = np.zeros(flat.shape[0], dtype=np.int64)
    eps = np.zeros((flat.shape[0], 3, 3), dtype=complex)
    for i, p in enumerate(cell_points):
        found = locator.locate(np.asarray(p))
        if found is None:
            raise ValueError(f"grid point {flat[i] * 1e6} um is outside the cell")
        tags[i] = int(mesh.cell_tag(found.cell))
        eps[i] = np.asarray(materials.of_cell(mesh, found.cell).eps_tensor, dtype=complex)
    unique = sorted(set(tags.tolist()))
    names = media or {}
    medium = np.array([unique.index(t) for t in tags], dtype=np.int64)
    q = _absorbed_density(e_cell, eps, omega) if absorbed else None
    e_sensor = e_cell[:, [0, 2, 1]]
    incident = np.asarray(wave.incident(np.array([0.0, top])), dtype=complex)
    pol_solver = np.array([incident[0], incident[1], 1j * incident[2]])
    pol = pol_solver[[0, 2, 1]]
    k_hat = np.array([wave.kx, wave.beta, -wave.ky]) / (k0 * n_c)
    provenance = {
        "solver": "hp-FEM",
        "version": hpfem.__version__,
        "method": "hpfem.dataports.grating_field_map (conical FEM, grating.solve)",
        "phase_reference": _PHASE_REFERENCE,
        "incidence": "solution",
        "dofs": str(int(result.dofs)),
        "power_balance_residual": repr(float(result.power_balance_residual)),
    }
    provenance.update(source or {})
    shape = grid.shape
    return FieldMap(
        grid=grid,
        wavelength_um=2 * math.pi / k0 * 1e6,
        e=e_sensor.reshape(*shape, 3),
        q_w_m3=None if q is None else q.reshape(shape),
        source=provenance,
        irradiance_w_m2=n_c / (2 * hpfem.constants.Z0),
        k_hat=k_hat / np.linalg.norm(k_hat),
        polarisation=pol / np.linalg.norm(pol),
        medium=medium.reshape(shape),
        media=tuple(names.get(t, f"tag {t}") for t in unique),
    )


# --- structured surfaces: bodies of revolution (stage 3) ------------------------------------------


def meridian_mesh(r_lines, z_lines, tag_at: Callable, axis_tag: int = 77):
    """A structured triangle mesh of the meridian half-plane r ≥ 0 for the axisymmetric solver:
    the tensor grid of the increasing coordinates ``r_lines`` (from 0) and ``z_lines`` [m], two
    triangles per rectangle, each cell tagged by ``tag_at(r, z)`` of its centroid. The facets on
    the axis carry ``axis_tag``, the outer ones ``box_tag.X_MAX`` / ``Y_MIN`` / ``Y_MAX``. Put
    the stack interfaces, the walls of holes and cavities and the PML boundaries on grid lines,
    and grade the lines towards the emitters."""
    r = np.asarray(r_lines, dtype=float)
    z = np.asarray(z_lines, dtype=float)
    if r[0] != 0.0 or np.any(np.diff(r) <= 0) or np.any(np.diff(z) <= 0):
        raise ValueError("r_lines must start at 0 and both line sets increase strictly")
    nr = r.size
    vertices = [(float(ri), float(zj)) for zj in z for ri in r]
    cells, tags = [], []
    for j in range(z.size - 1):
        for i in range(nr - 1):
            a = j * nr + i
            b, c, d = a + 1, a + nr + 1, a + nr
            for tri in ((a, b, c), (a, c, d)):
                cells.append(tri)
                cr = sum(vertices[v][0] for v in tri) / 3.0
                cz = sum(vertices[v][1] for v in tri) / 3.0
                tags.append(int(tag_at(cr, cz)))
    mesh = hpfem.Mesh2D(vertices, cells, tags)
    tol = 1e-9 * float(r[-1])
    for f in mesh.boundary_facets:
        v = mesh.facet_vertices(int(f))
        p0, p1 = mesh.vertex(int(v[0])), mesh.vertex(int(v[1]))
        mr, mz = 0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1])
        if abs(p0[0]) < tol and abs(p1[0]) < tol:
            mesh.set_facet_tag(int(f), int(axis_tag))
        elif abs(mr - r[-1]) < tol:
            mesh.set_facet_tag(int(f), hpfem.box_tag.X_MAX)
        elif abs(mz - z[0]) < tol:
            mesh.set_facet_tag(int(f), hpfem.box_tag.Y_MIN)
        elif abs(mz - z[-1]) < tol:
            mesh.set_facet_tag(int(f), hpfem.box_tag.Y_MAX)
    return mesh


def _surface_around(mesh, cells: set, axis_tag: int):
    """The closed surface (normal outwards) around a set of cells, the facets on the axis left
    out (no flux through r = 0)."""
    facets = []
    for c in cells:
        for f in mesh.cell_facets(int(c)):
            f = int(f)
            if mesh.facet_tag(f) == axis_tag:
                continue
            neighbours = [int(k) for k in mesh.facet_cells(f)]
            if all(k in cells for k in neighbours):
                continue
            facets.append(hpfem.Surface2D.Facet(f, int(c)))
    surface = hpfem.Surface2D()
    surface.facets = facets
    return surface


@dataclass
class AxisymmetricEmitterResponse:
    """Result of :func:`axisymmetric_emitter_response`: the ``table``, the far fields per height
    (``far_field[i]["axial" | "transverse"]["cover" | "substrate"]``: the
    ``AxisymmetricFarField`` of the order m = 0 / m = 1 on the Gauss nodes) and ``convergence``
    (also in ``table.source``)."""

    table: EmitterResponseTable
    far_field: list
    convergence: dict


def _structure_hash(mesh, materials, stack, wavelength, order, sigma) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(mesh.vertices, dtype=float).tobytes())
    h.update(np.asarray(mesh.cells, dtype=np.int64).tobytes())
    h.update(np.asarray(mesh.cell_tags, dtype=np.int64).tobytes())
    for tag in sorted(materials):
        eps = np.asarray(materials[tag].eps_tensor, dtype=complex)
        h.update(str(int(tag)).encode() + eps.tobytes())
    n1, layers, n3, _z = _media(stack)
    h.update(_stack_provenance(n1, layers, n3, wavelength)["stack"].encode())
    h.update(f"{int(order)} {float(sigma)!r}".encode())
    return h.hexdigest()


def _project_balance(a_tot: np.ndarray, a_rad: np.ndarray) -> float:
    """Raise A_tot (n, 3, 3) in place where A_tot − A_rad has a negative eigenvalue (the
    discretisation of a lossless structure): A_tot += |λ| v vᵀ. Returns the largest |λ| relative
    to the trace of A_tot."""
    worst = 0.0
    for i in range(a_tot.shape[0]):
        lam, vec = np.linalg.eigh(a_tot[i] - a_rad[i])
        for k in np.nonzero(lam < 0)[0]:
            worst = max(worst, -lam[k] / np.trace(a_tot[i]))
            a_tot[i] -= lam[k] * np.outer(vec[:, k], vec[:, k])
    return float(worst)


def _offaxis_emitter_response(
    nd, h1, mesh, materials, stack, wavelength, order, sigma, detectors, theta_points,
    balance_tolerance, source, radii, heights, order_tolerance, max_azimuthal_order, omega,
    material_map, lab, centroids, radius, host_cells, axis_tag, far_surface, base_setup, bulk,
    n_host, radiating_substrate, up, down,
):  # fmt: skip
    """The off-axis branch of :func:`axisymmetric_emitter_response` (see there)."""
    theta_up, w_up = up
    theta_down, w_down = down
    positions = [(float(rho), float(z0)) for rho in radii for z0 in heights]
    surfaces, supports = [], []
    for rho, z0 in positions:
        dist = np.hypot(centroids[:, 0] - rho, centroids[:, 1] - z0)
        ball = set(np.nonzero(dist < 6 * sigma)[0].tolist())
        if not ball or not ball <= host_cells:
            raise ValueError(
                f"the dipole at (r, z) = ({rho:.3g}, {z0:.3g}) m (6 sigma = {6 * sigma:.3g} m) "
                "reaches another medium or has no cells around it: lower sigma or move it"
            )
        surfaces.append(_surface_around(mesh, ball, axis_tag))
        supports.append(np.nonzero(dist < 10 * sigma + radius)[0].tolist())
    moments = [np.eye(3, dtype=complex)[i] for i in range(3)]  # x, y, z
    n_pos, n_e = len(positions), nd.num_dofs
    sides = {"cover": (w_up, theta_up)}
    if radiating_substrate:
        sides["substrate"] = (w_down, math.pi - theta_down)
    p_tot = np.zeros((n_pos, 3, 3))
    density = {side: np.zeros((n_pos, theta.size, 3, 3)) for side, (_w, theta) in sides.items()}
    far_orders = {side: [[[] for _ in range(3)] for _ in range(n_pos)] for side in sides}
    cap = (int(max_azimuthal_order) if max_azimuthal_order is not None
           else int(math.ceil(8 * float(radii.max()) / sigma)) + 8)  # fmt: skip
    quiet_tot = quiet_far = 0
    far_done = False
    m = 0
    m_far = 0
    while quiet_tot < 2 or not far_done:
        if m > cap:
            raise ValueError(
                f"the azimuthal orders did not converge to order_tolerance {order_tolerance:.1e} "
                f"by m = {cap}: raise max_azimuthal_order or sigma"
            )
        weight = 1.0 if m == 0 else 2.0  # the order -m is the mirror image of +m
        problems = []
        for rho, z0 in positions:
            for moment in moments:
                setup = base_setup(m)
                setup.current = hpfem.axisymmetric_offaxis_gaussian_dipole(
                    rho, z0, moment, float(sigma), omega, m
                )
                setup.keep_factorisation = not problems
                problems.append(hpfem.AxisymmetricScattering(nd, h1, setup))
        first = problems[0].solve()
        loads = np.column_stack([problems[k].load(supports[k // 3])
                                 for k in range(1, len(problems))])  # fmt: skip
        blocks = np.column_stack([np.concatenate([first.meridian, first.azimuthal]),
                                  first.factorisation.solve_many(loads)])  # fmt: skip
        del first
        change_tot = change_far = 0.0
        for ip in range(n_pos):
            b = blocks[:, 3 * ip : 3 * ip + 3]

            def flux(block, ip=ip, m=m):
                return hpfem.axisymmetric_poynting_flux(
                    nd, h1, block[:n_e], block[n_e:], m, omega, material_map, surfaces[ip]
                )

            d = np.zeros((3, 3))
            for i in range(3):
                d[i, i] = flux(b[:, i])
            d[0, 2] = d[2, 0] = 0.5 * (flux(b[:, 0] + b[:, 2]) - d[0, 0] - d[2, 2])
            p_tot[ip] += weight * d
            change_tot = max(change_tot, weight * np.trace(d) / max(np.trace(p_tot[ip]), 1e-300))
            if far_done:
                continue
            amplitudes = {side: [] for side in sides}
            impedance = {}
            for i in range(3):
                far = hpfem.axisymmetric_layered_far_field(
                    nd, h1, b[:n_e, i], b[n_e:, i], m, omega, material_map, far_surface, lab,
                    theta_up.tolist(), theta_down.tolist())  # fmt: skip
                for side in sides:
                    ff = far.up if side == "cover" else far.down
                    f = np.column_stack([np.asarray(ff.f_theta), np.asarray(ff.f_phi)])
                    amplitudes[side].append(f)
                    far_orders[side][ip][i].append(f)
                    impedance[side] = ff.impedance
            before = sum(np.trace(density[side][ip].sum(axis=0)) for side in sides)
            for side, (w, theta) in sides.items():
                f = np.stack(amplitudes[side], axis=-1)  # (θ, component, orientation)
                gram = np.einsum("tci,tcj->tij", f.conj(), f).real
                # the y dipole decouples: its cross terms cancel between the orders ±m
                gram[:, 0, 1] = gram[:, 1, 0] = gram[:, 1, 2] = gram[:, 2, 1] = 0.0
                density[side][ip] += (weight * math.pi / impedance[side]
                                      * (np.sin(theta) * w)[:, None, None] * gram)  # fmt: skip
            after = sum(np.trace(density[side][ip].sum(axis=0)) for side in sides)
            change_far = max(change_far, (after - before) / max(after, 1e-300))
        quiet_tot = quiet_tot + 1 if change_tot < order_tolerance else 0
        if not far_done:
            m_far = m
            quiet_far = quiet_far + 1 if change_far < order_tolerance else 0
            far_done = quiet_far >= 2
        m += 1
    m_tot = m - 1

    a_tot = p_tot / bulk
    a_rad = sum(density[side].sum(axis=1) for side in sides) / bulk
    a_det = np.zeros((n_pos, len(detectors), 3, 3))
    for k, d in enumerate(detectors):
        if d.side not in sides:
            continue
        theta = sides[d.side][1]
        inside = (theta >= d.theta_min) & (theta <= d.theta_max)
        a_det[:, k] = density[d.side][:, inside].sum(axis=1) / bulk
    for side in ("cover", "substrate"):
        spans = sorted((d.theta_min, d.theta_max, d.name) for d in detectors if d.side == side)
        for (_a0, b0, n0), (a1, _b1, n1_) in zip(spans[:-1], spans[1:], strict=True):
            if a1 < b0:
                raise ValueError(f"the detector regions {n0!r} and {n1_!r} overlap")
    projection = _project_balance(a_tot, a_rad)
    if projection > balance_tolerance:
        raise ValueError(
            f"A_rad exceeds A_tot by {projection:.2e} (relative), beyond balance_tolerance "
            f"{balance_tolerance:.1e}: refine the mesh near the emitter or raise the order"
        )
    gap = np.linalg.eigvalsh(a_tot - a_rad).min(axis=1) / np.trace(a_tot, axis1=1, axis2=2)
    convergence = {
        "order": int(order),
        "dofs": int(nd.num_dofs + h1.num_dofs),
        "cells": int(mesh.num_cells),
        "sigma_m": float(sigma),
        "energy": float(gap.min()),
        "energy_projection": float(projection),
        "theta_points": int(theta_points),
        "azimuthal_orders": int(m_tot),
        "far_field_orders": int(m_far),
        "factorisations": int(m_tot + 1),
        "order_tolerance": float(order_tolerance),
    }
    provenance = {
        "solver": "hp-FEM",
        "version": hpfem.__version__,
        "method": "hpfem.dataports.axisymmetric_emitter_response (axisymmetric FEM, off-axis "
        "Gaussian dipoles summed over the azimuthal orders, far field by reciprocity)",
        "content_hash": _structure_hash(mesh, materials, stack, float(wavelength), order, sigma),
        "convergence": json.dumps(convergence, sort_keys=True),
    }
    provenance.update(source or {})
    shape = (radii.size, 1, heights.size)
    table = EmitterResponseTable(
        grid=RegularGrid((radii * 1e6, [0.0], heights * 1e6)),
        wavelength_um=float(wavelength) * 1e6,
        n_host=n_host,
        detectors=tuple(d.name for d in detectors),
        a_tot=a_tot.reshape(shape + (3, 3)),
        a_rad=a_rad.reshape(shape + (3, 3)),
        a_det=a_det.reshape(shape + (len(detectors), 3, 3)),
        source=provenance,
    )
    far_fields = [
        {name: {side: np.asarray(far_orders[side][ip][i]) for side in sides}
         for i, name in enumerate(("x", "y", "z"))}
        for ip in range(n_pos)
    ]  # fmt: skip
    return AxisymmetricEmitterResponse(table=table, far_field=far_fields, convergence=convergence)


def axisymmetric_emitter_response(
    mesh,
    materials: dict,
    stack,
    wavelength: float,
    heights,
    *,
    pml: dict,
    order: int = 3,
    sigma: float | None = None,
    axis_tag: int = 77,
    detectors: Sequence[Detector] | None = None,
    theta_points: int = 64,
    balance_tolerance: float = 1e-2,
    source: dict | None = None,
    radii=None,
    order_tolerance: float = 1e-8,
    max_azimuthal_order: int | None = None,
) -> AxisymmetricEmitterResponse:
    """The emitter response table of emitters on the axis of a body of revolution (a hole, a
    cavity, a pillar) in a planar stack, by the axisymmetric FEM (issue #214, stage 3).

    ``mesh``: the meridian half-plane (r = x ≥ 0, z = y; e.g. :func:`meridian_mesh`) with the
    facets on the axis tagged ``axis_tag``, cells tagged by material (``materials`` {tag:
    Material}, every tag used); the stack (``LayerStack2D`` / ``3D`` of isotropic media, its
    top interface the surface z = 0 of the sensor frame) is the background of the far field.
    ``pml``: {"r": t, "top": t, "bottom": t} thicknesses [m] of the PML inside the mesh.
    ``heights``: the emitter positions z [m] on the axis (strictly increasing, in one lossless
    host medium, the solution above the surface or the solution filling a hole below it); the
    table's grid is (0, 0, heights in µm).

    Per height two solves (total field of a Gaussian dipole of width ``sigma``, default
    λ/(40 n_host); it must stay ≥ 5σ away from other media): the axial dipole (order m = 0)
    gives A_zz, the transverse one (m = ±1, equal by symmetry) A_xx = A_yy; on the axis the
    tensors are diagonal. The two orders are factorised once for all heights (the operator does
    not depend on the source; a further height is one solve with the kept factorisation and a
    load on the support of its Gaussian). A_tot is the Poynting flux through a closed surface of
    the host cells within 6σ of the emitter divided by the power of the same smeared dipole in
    the bulk host
    (the smearing cancels, as for the array scanning of ADR-0013); A_rad and A_det come from the
    far field in both half-spaces by reciprocity through the boundary of the region inside the
    PML (``axisymmetric_layered_far_field``), integrated on cosine-mapped Gauss nodes in θ
    (``theta_points`` per interval between 0, the critical angle, the detector edges and π/2).
    Detectors as in :func:`planar_emitter_response`, without Jones matrices (on the axis the
    tensors stay diagonal only for rotation-invariant detectors).

    In a lossless structure without guided modes A_tot − A_rad vanishes up to the discretisation
    error and may come out slightly negative, which BindOptiX refuses: A_tot is then raised to
    A_rad on that diagonal entry and the change recorded (``energy_projection``, relative to A_tot);
    above ``balance_tolerance`` the call raises ``ValueError`` (refine the mesh or raise
    ``order``).

    **Off the axis** (``radii``: the distances ρ0 ≥ 0 [m] from the axis, strictly increasing; the
    emitters sit at (x, y, z) = (ρ0, 0, z0), the table's grid is (radii, 0, heights) in µm and the
    tensors are Cartesian there; other azimuths follow by rotating about z): the Gaussian dipole
    is expanded in all azimuthal orders (``axisymmetric_offaxis_gaussian_dipole``). The mirror
    φ → −φ makes the orders ±m of the x, y and z dipoles mirror images, so m ≥ 0 suffice (weight
    2 for m > 0), and A_xy = A_yz = 0; A_xz is the cross power of the x and z dipoles. Every
    order is factorised once for all positions and the three orientations (the loads on the
    support of the Gaussians, one multi-right-hand-side solve); the power and the far field of
    the orders add up. The sum runs until two consecutive orders change A_tot (and, separately,
    A_rad) by less than ``order_tolerance`` relative to its trace; the order content of the
    source decays as exp(−m²σ²/2ρ0²); 20 orders at ρ0/σ = 15 and the default tolerance
    (``max_azimuthal_order``, default ⌈8 ρ_max/σ⌉ + 8, raises ``ValueError`` when reached). A
    negative eigenvalue of A_tot − A_rad from the discretisation is projected away as above.
    ``radii`` = [0] gives the on-axis table (the test)."""
    n1, layers, n3, z_bottom = _media(stack)
    heights = np.asarray(heights, dtype=float).ravel()
    if heights.size < 1 or np.any(np.diff(heights) <= 0):
        raise ValueError("heights must be strictly increasing")
    if detectors is None:
        detectors = [Detector("solution", "cover"), Detector("substrate", "substrate")]
    detectors = list(detectors)
    if not detectors:
        raise ValueError("at least one detector region is needed (BindOptiX reads no empty axis)")
    for d in detectors:
        if d.jones is not None:
            raise ValueError("axisymmetric_emitter_response: detectors without Jones matrices")
        if d.side not in ("cover", "substrate"):
            raise ValueError(f"detector {d.name!r}: side must be 'cover' or 'substrate'")
    k0 = 2 * math.pi / float(wavelength)
    omega = k0 * hpfem.constants.c0
    lab = hpfem.LayerStack3D(stack.material(0), [hpfem.Layer(stack.material(i), d) for i, (_n, d)
                                                 in enumerate(layers, start=1)],
                             stack.material(stack.num_layers + 1), 0.0)  # fmt: skip
    material_map = hpfem.MaterialMap()
    for tag, mat in materials.items():
        material_map.set(int(tag), mat)
    tags = np.asarray(mesh.cell_tags, dtype=int)
    missing = set(tags.tolist()) - set(int(t) for t in materials)
    if missing:
        raise ValueError(f"cell tags without a material: {sorted(missing)}")
    vertices = np.asarray(mesh.vertices, dtype=float)
    cells = np.asarray(mesh.cells, dtype=int)
    centroids = vertices[cells].mean(axis=1)
    r_max, z_min, z_max = vertices[:, 0].max(), vertices[:, 1].min(), vertices[:, 1].max()
    t_r, t_top, t_bottom = float(pml["r"]), float(pml["top"]), float(pml["bottom"])
    inner = (
        (centroids[:, 0] < r_max - t_r)
        & (centroids[:, 1] > z_min + t_bottom)
        & (centroids[:, 1] < z_max - t_top)
    )
    # the host medium at the emitters
    locator = hpfem.PointLocator2D(mesh)
    if radii is not None:
        radii = np.asarray(radii, dtype=float).ravel()
        if radii.size < 1 or np.any(np.diff(radii) <= 0) or radii[0] < 0:
            raise ValueError("radii must be non-negative and strictly increasing")
    host_tags = set()
    for rho in [0.0] if radii is None else radii:
        for z0 in heights:
            found = locator.locate(np.array([float(rho), float(z0)]))
            if found is None or not inner[found.cell]:
                raise ValueError(
                    f"emitter at (r, z) = ({rho:.3g}, {z0:.3g}) m outside the region inside the PML"
                )
            host_tags.add(int(tags[found.cell]))
    host = {complex(materials[t].refractive_index) for t in host_tags}
    if len(host) != 1:
        raise ValueError("all emitters must lie in one host medium")
    n_host_c = host.pop()
    if abs(n_host_c.imag) > 1e-12:
        raise ValueError("the host medium of the emitters must be lossless")
    n_host = n_host_c.real
    if sigma is None:
        sigma = float(wavelength) / (40 * n_host)
    host_cells = {c for c in range(mesh.num_cells)
                  if complex(materials[int(tags[c])].refractive_index) == n_host_c}  # fmt: skip
    n_ref = min(n1, complex(n3).real) if complex(n3).real > 0 else n1
    pml_box = hpfem.PmlBox2D(
        [0.0, z_min + t_bottom],
        [r_max - t_r, z_max - t_top],
        [0.0, t_r, t_bottom, t_top],
        k0,
        n_ref,
    )
    far_surface = _surface_around(mesh, {c for c in range(mesh.num_cells) if inner[c]}, axis_tag)
    nd = hpfem.NedelecDofMap2D(mesh, int(order))
    h1 = hpfem.DofMap2D(mesh, int(order))

    # direction grids
    up_breaks = [0.0, 0.5 * math.pi] + [b for d in detectors if d.side == "cover"
                                        for b in (d.theta_min, d.theta_max)]  # fmt: skip
    theta_up, w_up = _theta_grid(sorted(set(up_breaks)), theta_points)
    n3c = complex(n3)
    radiating_substrate = abs(n3c.imag) <= 1e-12 and n3c.real > 0
    theta_down = w_down = np.zeros(0)
    if radiating_substrate:
        sub_breaks = [0.0, 0.5 * math.pi] + [b for d in detectors if d.side == "substrate"
                                             for b in (d.theta_min, d.theta_max)]  # fmt: skip
        if n3c.real > n1:
            sub_breaks.append(math.asin(n1 / n3c.real))
        theta_s, w_down = _theta_grid(sorted(set(sub_breaks)), theta_points)
        theta_down = math.pi - theta_s  # polar angle from +z
    bulk = n_host * hpfem.dipole_vacuum_power(1.0, omega) * math.exp(-((n_host * k0 * sigma) ** 2))
    radius = np.linalg.norm(vertices[cells] - centroids[:, None, :], axis=2).max(axis=1)

    def base_setup(m):
        setup = hpfem.AxisymmetricScatteringSetup()
        setup.omega = omega
        setup.materials = material_map
        setup.axis_tag = int(axis_tag)
        setup.azimuthal_order = m
        setup.pml = pml_box
        setup.background = lab
        setup.extra_quadrature_order = 6
        return setup

    if radii is not None:
        return _offaxis_emitter_response(
            nd, h1, mesh, materials, stack, wavelength, order, sigma, detectors, theta_points,
            balance_tolerance, source, radii, heights, order_tolerance, max_azimuthal_order,
            omega, material_map, lab, centroids, radius, host_cells, axis_tag, far_surface,
            base_setup, bulk, n_host, radiating_substrate, (theta_up, w_up), (theta_down, w_down),
        )  # fmt: skip

    a_tot = np.zeros((heights.size, 3, 3))
    a_rad = np.zeros((heights.size, 3, 3))
    a_det = np.zeros((heights.size, len(detectors), 3, 3))
    far_fields = []
    projection = 0.0
    # one factorisation per order for all heights: the operator does not depend on the source,
    # the load of a further height is assembled on the support of its Gaussian (10 σ, e^-50)
    kept = {}
    for iz, z0 in enumerate(heights):
        # the Gaussian must stay in the host: 6σ ball of host cells around the emitter
        dist = np.hypot(centroids[:, 0], centroids[:, 1] - z0)
        ball = {c for c in np.nonzero(dist < 6 * sigma)[0].tolist()}
        if not ball or not ball <= host_cells:
            raise ValueError(
                f"the dipole at z = {z0:.3g} m (6 sigma = {6 * sigma:.3g} m) reaches another "
                "medium or has no cells around it: lower sigma or move it"
            )
        src_surface = _surface_around(mesh, ball, axis_tag)
        entry = {}
        for name, m, orientation, factor, comps in (
            ("axial", 0, hpfem.AxisDipole.AXIAL, 1.0, (2,)),
            ("transverse", 1, hpfem.AxisDipole.TRANSVERSE, 2.0, (0, 1)),
        ):
            setup = base_setup(m)
            setup.current = hpfem.axisymmetric_gaussian_dipole(float(z0), 1.0, orientation,
                                                               float(sigma), omega, m)  # fmt: skip
            setup.keep_factorisation = m not in kept
            problem = hpfem.AxisymmetricScattering(nd, h1, setup)
            if m in kept:
                support = np.nonzero(dist < 10 * sigma + radius)[0].tolist()
                field = problem.solution_from(kept[m].solve(problem.load(support)))
            else:
                field = problem.solve()
                kept[m] = field.factorisation
            emitted = factor * hpfem.axisymmetric_poynting_flux(
                nd, h1, field.meridian, field.azimuthal, m, omega, material_map, src_surface
            )
            far = hpfem.axisymmetric_layered_far_field(
                nd, h1, field.meridian, field.azimuthal, m, omega, material_map, far_surface,
                lab, theta_up.tolist(), theta_down.tolist())  # fmt: skip
            sides = {"cover": (far.up, w_up, theta_up)}
            if radiating_substrate:
                sides["substrate"] = (far.down, w_down, math.pi - theta_down)
            powers = {}
            for side, (ff, w, theta_side) in sides.items():
                density = np.abs(np.asarray(ff.f_theta)) ** 2 + np.abs(np.asarray(ff.f_phi)) ** 2
                powers[side] = (factor * math.pi / ff.impedance * density * np.sin(theta_side) * w,
                                theta_side)  # fmt: skip
            radiated = sum(float(p.sum()) for p, _t in powers.values())
            for comp in comps:
                a_tot[iz, comp, comp] = emitted / bulk
                a_rad[iz, comp, comp] = radiated / bulk
                for k, d in enumerate(detectors):
                    if d.side not in powers:
                        continue
                    p, theta_side = powers[d.side]
                    inside = (theta_side >= d.theta_min) & (theta_side <= d.theta_max)
                    a_det[iz, k, comp, comp] = float(p[inside].sum()) / bulk
            entry[name] = {"cover": far.up, "substrate": far.down if radiating_substrate else None}
        far_fields.append(entry)
        # discretisation: A_tot may fall below A_rad in a lossless structure
        for comp in range(3):
            gap = a_rad[iz, comp, comp] - a_tot[iz, comp, comp]
            if gap > 0:
                projection = max(projection, gap / a_tot[iz, comp, comp])
                a_tot[iz, comp, comp] = a_rad[iz, comp, comp]
    if projection > balance_tolerance:
        raise ValueError(
            f"A_rad exceeds A_tot by {projection:.2e} (relative), beyond balance_tolerance "
            f"{balance_tolerance:.1e}: refine the mesh near the emitter or raise the order"
        )
    for side in ("cover", "substrate"):
        spans = sorted((d.theta_min, d.theta_max, d.name) for d in detectors if d.side == side)
        for (_a0, b0, n0), (a1, _b1, n1_) in zip(spans[:-1], spans[1:], strict=True):
            if a1 < b0:
                raise ValueError(f"the detector regions {n0!r} and {n1_!r} overlap")
    balance = np.min((np.diagonal(a_tot, axis1=1, axis2=2) - np.diagonal(a_rad, axis1=1,
                                                                            axis2=2))
                     / np.diagonal(a_tot, axis1=1, axis2=2))  # fmt: skip
    convergence = {
        "order": int(order),
        "dofs": int(nd.num_dofs + h1.num_dofs),
        "cells": int(mesh.num_cells),
        "sigma_m": float(sigma),
        "energy": float(balance),
        "energy_projection": float(projection),
        "theta_points": int(theta_points),
    }
    provenance = {
        "solver": "hp-FEM",
        "version": hpfem.__version__,
        "method": "hpfem.dataports.axisymmetric_emitter_response (axisymmetric FEM, on-axis "
        "Gaussian dipoles, far field by reciprocity)",
        "content_hash": _structure_hash(mesh, materials, stack, float(wavelength), order, sigma),
        "convergence": json.dumps(convergence, sort_keys=True),
    }
    provenance.update(source or {})
    table = EmitterResponseTable(
        grid=RegularGrid(([0.0], [0.0], heights * 1e6)),
        wavelength_um=float(wavelength) * 1e6,
        n_host=n_host,
        detectors=tuple(d.name for d in detectors),
        a_tot=a_tot[None, None],
        a_rad=a_rad[None, None],
        a_det=a_det[None, None],
        source=provenance,
    )
    return AxisymmetricEmitterResponse(table=table, far_field=far_fields, convergence=convergence)


def _rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def axisymmetric_field_map(
    mesh,
    materials: dict,
    stack,
    wavelength: float,
    grid: RegularGrid,
    k_hat,
    polarisation,
    *,
    pml: dict,
    order: int = 3,
    axis_tag: int = 77,
    max_order: int | None = None,
    media: dict | None = None,
    absorbed: bool = True,
    source: dict | None = None,
) -> FieldMap:
    """Port F of a body of revolution (a hole, a cavity, a pillar) in a planar stack for one
    incident plane wave, by the axisymmetric FEM (issue #214, stage 3; conventions of
    :func:`planar_field_map`).

    ``mesh``, ``materials``, ``stack``, ``pml``, ``order`` and ``axis_tag`` as in
    :func:`axisymmetric_emitter_response` (the axis of the body is the z axis of the sensor
    frame). The stack wave of ``k_hat`` (from the solution, k_z < 0, or through the lossless
    substrate) is expanded in the azimuthal orders m (`layered_axisymmetric_wave` in the frame
    turned by the azimuth of ``k_hat``); every order is one scattered-field solve per
    polarisation component (s, p) that ``polarisation`` contains, and the total field
    Σ_m (E_scat,m + E_stack,m) e^{imφ} is evaluated on ``grid`` (``axisymmetric_fields_at``).
    One factorisation per |m| serves both polarisations and both signs of m (the operators of ±m
    are related by A_{-m} = D A_m D, ``AxisymmetricScattering.mirrored``); the provenance records
    ``solves`` and ``factorisations``.
    Normal incidence needs m = ±1 only; oblique incidence |m| ≤ ``max_order`` (default
    ⌈k_ρ r_max⌉ + 6 with r_max the largest radius of the grid and the in-plane wavenumber k_ρ of
    the incidence). Grid points must lie inside the PML; on horizontal mesh lines the value is
    the limit from above. Without a body the map equals :func:`planar_field_map` (the test)."""
    _check_surface_plane(grid)
    k, pol = _unit_incidence(k_hat, polarisation)
    k0 = 2 * math.pi / float(wavelength)
    omega = k0 * hpfem.constants.c0
    phi_k = math.atan2(k[1], k[0]) if math.hypot(k[0], k[1]) > 1e-15 else 0.0
    turn = _rotation_z(phi_k)  # frame (in-plane k along +x) -> sensor
    k_frame = turn.T @ k
    pw = _stack_plane_waves(stack, wavelength, k_frame)
    pol_frame = turn.T @ pol
    coefficients, *_ = np.linalg.lstsq(pw.basis, pol_frame, rcond=None)
    if np.linalg.norm(pw.basis @ coefficients - pol_frame) > 1e-9:
        raise ValueError("the polarisation is not transverse to the incident wave")
    side = "bottom" if pw.from_substrate else "top"
    theta = pw.angle
    k_rho = k0 * pw.n_inc * math.sin(theta)
    points = _grid_points(grid)
    flat = points.reshape(-1, 3)
    radius = np.hypot(flat[:, 0], flat[:, 1])
    if theta < 1e-12:
        orders = [-1, 1]
    else:
        m_max = max_order if max_order is not None else math.ceil(k_rho * radius.max()) + 6
        orders = list(range(-int(m_max), int(m_max) + 1))
    # the solver setup (as axisymmetric_emitter_response)
    material_map = hpfem.MaterialMap()
    for tag, mat in materials.items():
        material_map.set(int(tag), mat)
    tags = np.asarray(mesh.cell_tags, dtype=int)
    missing = set(tags.tolist()) - set(int(t) for t in materials)
    if missing:
        raise ValueError(f"cell tags without a material: {sorted(missing)}")
    vertices = np.asarray(mesh.vertices, dtype=float)
    r_max, z_min, z_max = vertices[:, 0].max(), vertices[:, 1].min(), vertices[:, 1].max()
    t_r, t_top, t_bottom = float(pml["r"]), float(pml["top"]), float(pml["bottom"])
    if (np.any(radius > r_max - t_r) or np.any(flat[:, 2] < z_min + t_bottom)
            or np.any(flat[:, 2] > z_max - t_top)):  # fmt: skip
        raise ValueError("grid points outside the region inside the PML")
    n1, _layers, n3, _z = _media(stack)
    n_ref = min(n1, complex(n3).real) if complex(n3).real > 0 else n1
    pml_box = hpfem.PmlBox2D(
        [0.0, z_min + t_bottom],
        [r_max - t_r, z_max - t_top],
        [0.0, t_r, t_bottom, t_top],
        k0,
        n_ref,
    )
    nd = hpfem.NedelecDofMap2D(mesh, int(order))
    h1 = hpfem.DofMap2D(mesh, int(order))
    # the evaluation points in the frame: (r, z) with z on a horizontal mesh line taken from above
    lines = {round(float(v), 12) for v in vertices[:, 1] / float(wavelength)}
    nudge = np.array([round(z / float(wavelength), 12) in lines for z in flat[:, 2]])
    z_eval = np.where(nudge, flat[:, 2] + 1e-9 * float(wavelength), flat[:, 2])
    r_eval = np.maximum(radius, 1e-9 * float(wavelength))  # the axis: the limit along +x'
    xy_frame = flat[:, :2] @ turn[:2, :2]  # rows (x', y') = turn^T (x, y)
    phi = np.where(radius > 0, np.arctan2(xy_frame[:, 1], xy_frame[:, 0]), 0.0)
    rz = np.column_stack([r_eval, z_eval])
    on_axis = radius == 0
    c, sn = np.cos(phi), np.sin(phi)  # phi = 0 on the axis
    e_frame = np.zeros((flat.shape[0], 3), dtype=complex)
    solves = 0
    # one factorisation per |m| for both polarisations and both signs: A_{-m} = D A_m D
    kept = {}
    for p_index, pol_name in enumerate(("s", "p")):
        a = complex(coefficients[p_index])
        if abs(a) < 1e-14:
            continue
        # the scale of layered_axisymmetric_wave against the planar wave of unit amplitude at the
        # origin: compared at a probe point off the axis in the incidence medium
        probe_r = 0.05 * float(wavelength) / pw.n_inc
        probe_z = (0.1 if side == "top" else -0.1) * float(wavelength) + (
            0.0 if side == "top" else float(stack.bottom - stack.top)
        )
        analytic = np.zeros(3, dtype=complex)
        for m in orders:
            wave = hpfem.layered_axisymmetric_wave(pw.lab, k0, theta, pol_name, m, side)
            e = np.asarray(wave.value([probe_r, probe_z]), dtype=complex)
            analytic += np.array([e[0], 1j * e[1] / probe_r, e[2]])
        q = pw.mirror * np.array([probe_r, 0.0, probe_z])
        planar = pw.mirror * np.asarray(pw.waves[p_index].field.value(q), dtype=complex)
        alpha = np.vdot(planar, analytic) / np.vdot(planar, planar)
        if np.linalg.norm(analytic - alpha * planar) > 1e-6 * np.linalg.norm(planar):
            raise ValueError("the azimuthal orders do not reproduce the stack wave at the probe "
                             "point: raise max_order")  # fmt: skip
        weight = a / alpha
        for m in orders:
            wave = hpfem.layered_axisymmetric_wave(pw.lab, k0, theta, pol_name, m, side)
            setup = hpfem.AxisymmetricScatteringSetup()
            setup.omega = omega
            setup.materials = material_map
            setup.axis_tag = int(axis_tag)
            setup.azimuthal_order = m
            setup.pml = pml_box
            setup.background = pw.lab
            setup.incident = wave.value
            setup.keep_factorisation = abs(m) not in kept
            problem = hpfem.AxisymmetricScattering(nd, h1, setup)
            if abs(m) not in kept:
                field = problem.solve()
                kept[abs(m)] = (m, field.factorisation)
            else:
                m_kept, factorisation = kept[abs(m)]
                load = problem.load()
                if m == m_kept:
                    block = factorisation.solve(load)
                else:
                    block = problem.mirrored(factorisation.solve(problem.mirrored(load)))
                field = problem.solution_from(block)
            solves += 1
            scattered = np.asarray(hpfem.axisymmetric_fields_at(
                nd, h1, field.meridian, field.azimuthal, rz), dtype=complex)  # fmt: skip
            incident = np.array([np.asarray(wave.value(list(p)), dtype=complex) for p in rz])
            incident[:, 1] = 1j * incident[:, 1] / r_eval  # v -> E_phi
            total = weight * (scattered + incident)  # (E_r, E_phi, E_z) of order m
            if not np.all(np.isfinite(total)):
                raise ValueError("grid points outside the mesh")
            rot = np.exp(1j * m * phi)
            e_frame[:, 0] += (total[:, 0] * c - total[:, 1] * sn) * rot
            e_frame[:, 1] += (total[:, 0] * sn + total[:, 1] * c) * rot
            e_frame[:, 2] += total[:, 2] * rot
            # on the axis the regular limit, independent of the direction of approach (the
            # discrete field satisfies E_phi = ±i E_r there only to the discretisation error)
            if np.any(on_axis):
                a = total[on_axis]
                if abs(m) == 1:
                    amp = 0.5 * (a[:, 0] - m * 1j * a[:, 1])
                    e_frame[on_axis] += np.column_stack([amp, m * 1j * amp, 0 * amp]) - (
                        np.column_stack([a[:, 0], a[:, 1], a[:, 2]])
                    )
                elif m == 0:
                    e_frame[on_axis] += np.column_stack([-a[:, 0], -a[:, 1], 0 * a[:, 2]])
                else:
                    e_frame[on_axis] -= a
    e = e_frame @ turn.T
    # media and absorption per point
    locator = hpfem.PointLocator2D(mesh)
    point_tags = np.zeros(flat.shape[0], dtype=np.int64)
    eps = np.zeros((flat.shape[0], 3, 3), dtype=complex)
    for i, p in enumerate(rz):
        found = locator.locate(np.asarray(p))
        point_tags[i] = int(tags[found.cell])
        eps[i] = np.asarray(materials[int(point_tags[i])].eps_tensor, dtype=complex)
    unique = sorted(set(point_tags.tolist()))
    names = media or {}
    medium = np.array([unique.index(t) for t in point_tags], dtype=np.int64)
    provenance = {
        "solver": "hp-FEM",
        "version": hpfem.__version__,
        "method": "hpfem.dataports.axisymmetric_field_map (axisymmetric FEM, layered background)",
        "content_hash": _structure_hash(mesh, materials, stack, float(wavelength), order, 0.0),
        "phase_reference": _PHASE_REFERENCE,
        "incidence": "substrate" if pw.from_substrate else "solution",
        "azimuthal_orders": json.dumps(orders),
        "solves": str(solves),
        "factorisations": str(len(kept)),
    }
    provenance.update(source or {})
    shape = grid.shape
    return FieldMap(
        grid=grid,
        wavelength_um=float(wavelength) * 1e6,
        e=e.reshape(*shape, 3),
        q_w_m3=_absorbed_density(e, eps, omega).reshape(shape) if absorbed else None,
        source=provenance,
        irradiance_w_m2=pw.n_inc / (2 * hpfem.constants.Z0),
        k_hat=k,
        polarisation=pol,
        medium=medium.reshape(shape),
        media=tuple(names.get(t, f"tag {t}") for t in unique),
    )

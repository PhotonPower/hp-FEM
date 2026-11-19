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
    "Detector",
    "EmitterResponseTable",
    "FarField",
    "FieldMap",
    "PlanarEmitterResponse",
    "RegularGrid",
    "decode_array",
    "encode_array",
    "envelope",
    "planar_emitter_response",
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
    """Excitation field on a grid (``bindoptix.ports.FieldMap``): ``e`` complex (nx, ny, nz, 3)
    [V/m] per 1 mW incident power, time factor e^(−iωt); ``nonlinear`` (computed with temperature
    feedback, then valid only at ``power_mw``); optional absorbed power density ``q_w_m3`` and
    temperature rise ``delta_t_k`` per mW, shape (nx, ny, nz)."""

    grid: RegularGrid
    wavelength_um: float
    e: np.ndarray
    nonlinear: bool = False
    power_mw: float | None = None
    q_w_m3: np.ndarray | None = None
    delta_t_k: np.ndarray | None = None
    source: dict = field(default_factory=dict)

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
        object.__setattr__(self, "source", {str(k): str(v) for k, v in self.source.items()})

    def to_json(self) -> str:
        """The ``raytatouille-result`` envelope of type ``FieldMap``."""
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


def _content_hash(n1, layers, n3, wavelength) -> str:
    data = {
        "cover": [n1, 0.0],
        "layers": [[complex(n).real, complex(n).imag, d] for n, d in layers],
        "substrate": [complex(n3).real, complex(n3).imag],
        "wavelength_m": wavelength,
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


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
    content hash of the stack and wavelength)."""
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
        "content_hash": _content_hash(n1, layers, n3, float(wavelength)),
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

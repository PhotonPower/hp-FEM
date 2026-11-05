"""A point dipole above a planar stack of possibly anisotropic layers (ADR-0015 §8, M19 S7).

The reference for dipole emitters above crystal stacks: the power delivered by a current moment
``m`` (J = m δ(r − r0), A·m) in the isotropic, lossless cover of an
:class:`hpfem.AnisotropicStack` at the height ``h`` above its top interface, from the plane-wave
(Weyl) spectrum of the dyadic Green's function and the 2×2 Jones reflection matrix of the stack
at every in-plane wavevector (``AnisotropicStack.reflection_at``, propagating and evanescent).
With the in-plane wavevector k0 (a, b), q = sqrt(n² − a² − b²) (Im q ≥ 0) and the unit vectors
ŝ = ẑ × k̂_∥, p̂ = ŝ × k̂ / n of the downward (d) and upward (u) waves,

    F = P / P_bulk = 1 + 3 / (4π n |m|²) Re ∬ da db (1/q) e^{2i k0 q h} m^H [ŝ, p̂_u] r [ŝ, p̂_d]^T m,

P_bulk = Z0 k0² n |m|² / (12π) (``grating.dipole_bulk_power`` without smearing). For the
Gaussian dipole of ADR-0013 the smearing multiplies every spectral component by
exp(−σ² k·k) = exp(−(n k0 σ)²) (k·k = n² k0² also for evanescent waves) and so cancels in F: the
reference holds for the FEM emitter as it is. Convention exp(−iωt);
docs/theory/anisotropic-stack.md.

The radial integral runs on the real axis (s = n sin t inside the light cone, s = n cosh u
beyond), the azimuth by the trapezoidal rule (periodic and smooth). Guided modes of a lossless
stack are poles on the real axis and are not handled: use absorbing layers (or a stack without
guided modes, e.g. a half-space) for this reference.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad

import hpfem

__all__ = ["dipole_above_stack", "array_sample_above_stack"]


def _vectors(a: float, b: float, n: float):
    """ŝ, p̂_d, p̂_u and q for the normalised in-plane wavevector (a, b) in the medium n."""
    par = math.hypot(a, b)
    s = np.array([-b / par, a / par, 0.0]) if par > 0 else np.array([0.0, 1.0, 0.0])
    q = complex(np.sqrt(complex(n * n - a * a - b * b)))
    if q.imag < 0:
        q = -q
    k_d = np.array([a, b, -q])
    k_u = np.array([a, b, q])
    return s.astype(complex), np.cross(s, k_d) / n, np.cross(s, k_u) / n, q


def dipole_above_stack(stack, k0: float, height: float, moment, *, azimuths: int = 64,
                       tolerance: float = 1e-10) -> float:  # fmt: skip
    """Purcell factor P / P_bulk of the current moment ``moment`` (3 components, laboratory
    frame: z the stack normal) at ``height`` [m] above the top interface of the
    ``hpfem.AnisotropicStack`` ``stack`` (its incidence medium is the dipole's host) at the
    vacuum wavenumber ``k0`` [1/m]. ``azimuths``: trapezoidal points over the azimuth of the
    in-plane wavevector (a tilted crystal needs more than a c-cut one, whose integrand is a
    trigonometric polynomial of low degree); ``tolerance``: relative tolerance of the radial
    integrals. Lossless stacks with guided modes are not covered (poles on the real axis)."""
    m = np.asarray(moment, dtype=complex).reshape(3)
    if not height > 0:
        raise ValueError("dipole_above_stack: the height must be positive")
    n = complex(stack.material(0).refractive_index).real
    # kinks of the integrand beyond the light cone: the principal indices of the other media
    kinks = []
    for region in range(1, stack.num_layers + 2):
        eps = np.asarray(stack.material(region).eps_tensor)
        for value in np.linalg.eigvalsh(eps.real):
            if value > n * n:
                kinks.append(math.acosh(math.sqrt(value) / n))
    kinks = sorted({round(u, 12) for u in kinks if u > 0})
    u_max = math.asinh(40.0 / (2.0 * k0 * n * height))
    u_max = max(u_max, 1.2 * max(kinks, default=0.0))

    def term(a: float, b: float, jacobian: complex) -> float:
        s, p_d, p_u, q = _vectors(a, b, n)
        try:
            r = np.asarray(stack.reflection_at(k0, a, b), dtype=complex)
        except hpfem.InvalidArgument:  # exactly grazing in a medium: a measure-zero point
            r = np.asarray(stack.reflection_at(k0, a * (1 + 1e-12), b * (1 + 1e-12)), dtype=complex)
        dyad = np.column_stack([s, p_u]) @ r @ np.column_stack([s, p_d]).T
        value = (m.conj() @ dyad @ m) * np.exp(2j * k0 * q * height) * jacobian
        return value.real

    total = 0.0
    for j in range(azimuths):
        psi = 2.0 * math.pi * j / azimuths
        c, sn = math.cos(psi), math.sin(psi)

        def inside(t, c=c, sn=sn):
            s = n * math.sin(t)  # s ds / q = n sin t dt
            return term(s * c, s * sn, n * math.sin(t))

        def outside(u, c=c, sn=sn):
            s = n * math.cosh(u)  # s ds / q = −i n cosh u du
            return term(s * c, s * sn, -1j * n * math.cosh(u))

        radial = quad(inside, 0.0, 0.5 * math.pi, epsabs=0.0, epsrel=tolerance, limit=400)[0]
        radial += quad(outside, 0.0, u_max, points=kinks or None, epsabs=0.0, epsrel=tolerance,
                       limit=400)[0]  # fmt: skip
        total += radial
    total *= 2.0 * math.pi / azimuths
    return 1.0 + 3.0 / (4.0 * math.pi * n * float(np.vdot(m, m).real)) * total


def array_sample_above_stack(stack, k0: float, kx: float, beta: float, period: float,
                             height: float, moment, sigma: float = 0.0,
                             orders: int | None = None) -> float:  # fmt: skip
    """Power per period [W/m, per unit β] delivered by the Bloch-periodic array of Gaussian
    dipoles of ``grating.emit`` (one cell problem of the array scanning, ADR-0013) at the height
    ``height`` above the top of the ``hpfem.AnisotropicStack`` ``stack``: the homogeneous part
    and the reflected part of the plane-wave spectrum on the orders k_m = kx + 2πm/P at the
    longitudinal β,

        P_cell = ω μ0 / (4P) Σ_m Re[(1/k_z) (|m_⊥|² (propagating) + e^{2ik_zh} m^H M_m m)]
                 · e^{−σ² n² k0²},

    with M_m the reflection dyad of :func:`dipole_above_stack`. ``moment`` is the current moment
    in the solver frame of the conical solver (x across the lines, y the normal, z along them);
    ``orders``: orders −N..N (default: until e^{−2|k_z|h} < 1e-17)."""
    from hpfem import constants

    m_solver = np.asarray(moment, dtype=complex).reshape(3)
    m = np.array([m_solver[0], m_solver[2], m_solver[1]])  # laboratory frame: y <-> z
    n = complex(stack.material(0).refractive_index).real
    k = n * k0
    omega = k0 * constants.c0
    mu0 = constants.Z0 / constants.c0
    if orders is None:
        reach = math.sqrt(k * k + (37.0 / (2.0 * height)) ** 2) + abs(kx)
        orders = int(math.ceil(reach * period / (2 * math.pi))) + 1
    total = 0.0
    for j in range(-orders, orders + 1):
        km = kx + 2 * math.pi * j / period
        a, b = km / k0, beta / k0
        s, p_d, p_u, q = _vectors(a, b, n)
        kz = k0 * q
        if abs(q.imag) < 1e-14 and q.real > 0:
            for k_hat in (np.array([a, b, q]) / n, np.array([a, b, -q]) / n):
                perp = m - k_hat * (k_hat @ m)
                total += 0.5 * float(np.vdot(perp, perp).real) / kz.real
        try:
            r = np.asarray(stack.reflection_at(k0, a, b), dtype=complex)
        except hpfem.InvalidArgument:  # grazing in a medium
            continue
        dyad = np.column_stack([s, p_u]) @ r @ np.column_stack([s, p_d]).T
        total += ((m.conj() @ dyad @ m) * np.exp(2j * kz * height) / kz).real
    return omega * mu0 / (4 * period) * total * math.exp(-((sigma * k) ** 2))

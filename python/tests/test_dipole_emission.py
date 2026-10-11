"""Single dipole by array scanning (M17 S3, ADR-0013 §3, §3a): the quadrature of the scan on the
analytic closed form of a homogeneous cell, the kx integral of the FEM cell problems on the
complex contour around the light lines against the closed form on the real axis (independent of
the contour depth), and the interface of ``grating.dipole_emission`` on a small problem. The
full scans (homogeneous cell, a dipole above glass against the Sommerfeld integral, period
independence) are long local runs: ``benchmarks/m17_dipole_scan.py``."""

import cmath
import math

import numpy as np
import pytest

import hpfem
from hpfem import grating, units

UM = 1e-6
N = 1.5
LAMBDA = 1.0 * UM
OMEGA = units.angular_frequency(wavelength=LAMBDA)
K0 = 2 * math.pi / LAMBDA
K = N * K0
PERIOD = 1.0 * UM
SIGMA = 0.06 * UM
MU0 = 4e-7 * math.pi * 1.00000000055  # CODATA 2018 value used by the library


def closed_form(kx, beta, p):
    """pᴴAp of a phased array in the homogeneous medium, continued analytically in kx: per order
    and side ωμ0 (1 − (k̂·p)²) e^{−σ²k²}/(8P k_y), k_y on the branch Re + Im > 0 (for a real kx
    the evanescent orders are imaginary and drop out of the real part)."""
    p = np.asarray(p, dtype=float)
    total = 0.0
    for m in range(-6, 7):
        km = kx + 2 * math.pi * m / PERIOD
        ky = cmath.sqrt(K**2 - km**2 - beta**2)
        if ky.real + ky.imag < 0:
            ky = -ky
        for s in (1.0, -1.0):
            khat_p = (km * p[0] + s * ky * p[1] + beta * p[2]) / K
            total += (
                OMEGA
                * MU0
                * (p @ p - khat_p**2)
                * math.exp(-((SIGMA * K) ** 2))
                / (8 * PERIOD * ky)
            )
    return total


@pytest.mark.parametrize("depth", [0.0, 0.5])
@pytest.mark.parametrize("symmetric", [False, True])
def test_scan_rule_integrates_the_closed_form(depth, symmetric):
    bulk = grating.dipole_bulk_power((1.0, 0.0, 0.0), OMEGA, N, SIGMA)
    errors = {}
    for nodes in (4, 6):
        rule = grating.array_scan_rule(PERIOD, K0, [N, N], beta_max=K, nodes=nodes,
                                       symmetric=symmetric, depth=depth)  # fmt: skip
        assert rule.depth == depth
        assert (np.abs(rule.kx.imag) > 0).any() == (depth > 0)
        for p in np.eye(3):
            value = sum(w * closed_form(k, b, p) for k, b, w in
                        zip(rule.kx, rule.beta, rule.weight, strict=True))  # fmt: skip
            errors[(nodes, tuple(p))] = abs(value.real / bulk - 1.0)
    # the β quadrature dominates (the same on the real axis and on the contour): 3.0e-3 at 4
    # nodes per panel (z dipole), 1.0e-5 at 6
    assert max(e for (n, _p), e in errors.items() if n == 4) < 4e-3
    assert max(e for (n, _p), e in errors.items() if n == 6) < 2e-5


def test_contour_sides_follow_causality():
    # below the crossings −2πm/P + q (orders running along +x), above −2πm/P − q
    beta = 0.3 * K
    q = math.sqrt(K**2 - beta**2)
    crossings = grating._light_line_crossings(PERIOD, K0, [N], beta)
    lo, hi = -math.pi / PERIOD, math.pi / PERIOD
    kx, w, _panels = grating._kx_contour(lo, hi, crossings, 6, 0.5)
    g = 2 * math.pi / PERIOD
    for c, s in crossings:
        assert abs((s * q - c) / g - round((s * q - c) / g)) < 1e-12
        near = np.abs(kx.real - c) < 0.02 / PERIOD
        assert near.any()
        assert np.all(np.sign(kx[near].imag) == -s)
    # the weights integrate dkx: the contour is closed up to the period
    assert abs(w.sum() - (hi - lo)) < 1e-12 * (hi - lo)
    # a medium evanescent at this β puts branch points at 2πm/P ± iκ; the bumps stay below
    # 0.05 of their distance to them (the PML turns their cuts into strings of poles)
    kappa = 0.1 * K
    branch = [complex(0.0, kappa), complex(0.0, -kappa)]
    kx_c, w_c, _panels = grating._kx_contour(lo, hi, crossings, 6, 0.5, branch)
    for c, _s in crossings:
        near = np.abs(kx_c.real - c) < 0.5 / PERIOD
        limit = 0.05 * min(abs(c - b) for b in branch)
        assert np.abs(kx_c[near].imag).max() <= limit * (1 + 1e-12)
    assert abs(w_c.sum() - (hi - lo)) < 1e-12 * (hi - lo)


def homogeneous_cell(h=0.125 * UM):
    nx, ny = round(PERIOD / h), round(4.0 * UM / h)
    mesh = hpfem.rectangle(nx, ny, [-PERIOD / 2, -2.0 * UM], [PERIOD / 2, 2.0 * UM])
    medium = hpfem.Material.dielectric(N)
    return mesh, hpfem.LayerStack2D(medium, [], medium, 0.0)


def test_cell_power_on_the_complex_contour_is_independent_of_the_depth():
    # one β slice of a y dipole: the kx integral of the FEM power matrix on the contour matches
    # the closed form on the real axis, for two depths (on the real axis the grazing orders at
    # the light lines are not absorbed by the PML and the slice is off by percent)
    mesh, stack = homogeneous_cell()
    position, p = np.array([0.0, 0.1 * UM]), np.array([0.0, 1.0, 0.0])
    cell = grating._dipole_cell(mesh, {}, stack, position, SIGMA, OMEGA, 3,
                                pml={"top": UM, "bottom": UM}, bottom="pml",
                                snap_tolerance=1e-9, pml_target=1e-6, pml_wavelengths=0.5,
                                extra_quadrature_order=4, solver=None)  # fmt: skip
    beta = 0.245 * K
    crossings = grating._light_line_crossings(PERIOD, K0, [N], beta)
    lo, hi = 0.0, math.pi / PERIOD
    inside = [(c, s) for c, s in crossings if lo < c < hi]
    kx_ref, w_ref, _ = grating._kx_contour(lo, hi, inside, 40, 0.0)
    reference = sum(w * closed_form(k, beta, p) for k, w in zip(kx_ref, w_ref, strict=True)).real
    results = {}
    for depth in (0.5, 1.0):
        kx, w, _ = grating._kx_contour(lo, hi, inside, 4, depth)
        fem = sum(wi * (p @ grating._dipole_power_matrix(cell, complex(k), beta) @ p)
                  for k, wi in zip(kx, w, strict=True))  # fmt: skip
        results[depth] = fem.real / reference - 1.0
    assert abs(results[0.5]) < 5e-3, results
    assert abs(results[0.5] - results[1.0]) < 1e-3, results


def test_dipole_emission_interface():
    # a coarse, truncated scan: the keys, the own moment as pᴴAp of the unit moments, the
    # isotropic mean, channels per moment and the argument checks (accuracy: the tests above
    # and the long runs)
    mesh, stack = homogeneous_cell(h=0.25 * UM)
    dipole = {"position": (0.0, 0.1 * UM), "sigma": SIGMA, "moment": (0.0, 2.0, 0.0)}
    calls = []
    r = grating.dipole_emission(mesh, {}, stack, dipole, OMEGA, nodes=2, kx_nodes=2, order=2,
                                beta_max=0.3 * K, symmetric=True, angle_nodes=(2, 4),
                                pml={"top": UM, "bottom": UM},
                                progress=lambda i, n: calls.append((i, n)))  # fmt: skip
    assert set(r.P_em) == {"x", "y", "z", "isotropic", "moment"}
    assert r.P_em["moment"] == pytest.approx(4 * r.P_em["y"], rel=1e-12)
    assert r.purcell["moment"] == pytest.approx(r.purcell["y"], rel=1e-12)
    assert r.P_em["isotropic"] == pytest.approx(np.mean([r.P_em[k] for k in "xyz"]), rel=1e-12)
    assert r.up["moment"] == pytest.approx(4 * r.up["y"], rel=1e-12)
    for key in r.P_em:
        assert r.nonradiated[key] == pytest.approx(r.P_em[key] - r.up[key] - r.down[key])
    assert not r.lossy
    assert r.n_host == pytest.approx(N)
    assert r.samples == len(r.rule.weight) == r.contributions.shape[0]
    assert r.directions == 2 * 2 * 2 * 2  # 2 θ × 2 φ (mirror symmetries of 4) × 2 sides × s, p
    assert calls[-1][1] == r.samples + r.directions and len(calls) == calls[-1][1]
    off = grating.dipole_emission(mesh, {}, stack, dipole, OMEGA, nodes=2, kx_nodes=2, order=2,
                                  beta_max=0.3 * K, symmetric=True, channels=False,
                                  pml={"top": UM, "bottom": UM})  # fmt: skip
    assert off.up is None and off.nonradiated is None
    assert off.P_em["y"] == pytest.approx(r.P_em["y"], rel=1e-12)
    with pytest.raises(grating.GratingError, match="6 sigma"):
        grating.dipole_emission(mesh, {}, stack, {"position": (0.0, 1.1 * UM), "sigma": SIGMA},
                                OMEGA, nodes=2, pml={"top": UM, "bottom": UM})  # fmt: skip
    with pytest.raises(grating.GratingError, match="multiple of 4"):
        grating.dipole_emission(mesh, {}, stack, dipole, OMEGA, nodes=2, kx_nodes=2, order=2,
                                beta_max=0.3 * K, angle_nodes=(2, 6),
                                pml={"top": UM, "bottom": UM})  # fmt: skip
    with pytest.raises(hpfem.Cancelled):
        grating.dipole_emission(mesh, {}, stack, dipole, OMEGA, nodes=2, kx_nodes=2, order=2,
                                pml={"top": UM, "bottom": UM}, cancel=lambda: True)  # fmt: skip

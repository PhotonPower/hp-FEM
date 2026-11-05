"""A point dipole above a planar crystal stack (hpfem.planar, M19 S7a): the 2D plane-wave
spectrum with the 4×4 stack reflection against independent 1D Sommerfeld integrals with closed
form reflection coefficients, and the rotational covariance of a tilted crystal."""

import cmath
import math

import numpy as np
import pytest
from scipy.integrate import quad

import hpfem
from hpfem.planar import dipole_above_stack

K0 = 2 * np.pi / 1e-6
H = 0.1e-6


def sommerfeld(reflection, n, height):
    """(F_perp, F_par) of a point dipole in the cover n from r_s(s), r_p(s) at the in-plane
    wavenumber s k0 (Chance–Prock–Silbey, r_p = +1 for a perfect conductor):
    F_⊥ = 1 + 3/(2n³) Re ∫ s³/q r_p e^{2ik0qh} ds, F_∥ = 1 + 3/(4n) Re ∫ s/q (r_s − q²/n² r_p)
    e^{2ik0qh} ds, q = sqrt(n² − s²)."""

    def integrand(s, q, ds_over_q, which):
        rs, rp = reflection(s)
        e = cmath.exp(2j * K0 * q * height)
        if which == 0:
            return (s**3 * rp * e * ds_over_q).real
        return (s * (rs - q * q / (n * n) * rp) * e * ds_over_q).real

    def integral(which):
        inside = quad(lambda t: integrand(n * math.sin(t), n * math.cos(t), n, which),
                      0, math.pi / 2, epsabs=0, epsrel=1e-11, limit=400)[0]  # fmt: skip
        outside = quad(
            lambda u: integrand(n * math.cosh(u), 1j * n * math.sinh(u), -1j * n, which),
            0, math.asinh(40 / (2 * K0 * n * height)) + 2.0, epsabs=0, epsrel=1e-11, limit=400,
        )[0]  # fmt: skip
        return inside + outside

    return 1 + 1.5 / n**3 * integral(0), 1 + 0.75 / n * integral(1)


def kz(eps, s):
    q = cmath.sqrt(eps - s * s)
    return q if q.imag >= 0 else -q


def test_isotropic_half_space_and_lossy_layer():
    """Glass half-space and an absorbing layer on glass: the 2D integral equals the 1D
    Sommerfeld integral with the Fresnel / Airy coefficients."""
    glass = hpfem.Material.dielectric(1.5)
    layer_eps, d = 4.0 + 0.6j, 0.12e-6

    def airy(s):
        out = []
        for pol in ("s", "p"):
            media = [1.0, layer_eps, 2.25]
            q = [kz(e, s) for e in media]
            r = [
                (q[i] - q[i + 1]) / (q[i] + q[i + 1]) if pol == "s"
                else (media[i + 1] * q[i] - media[i] * q[i + 1])
                / (media[i + 1] * q[i] + media[i] * q[i + 1])
                for i in range(2)
            ]  # fmt: skip
            ph = cmath.exp(2j * K0 * q[1] * d)
            out.append((r[0] + r[1] * ph) / (1 + r[0] * r[1] * ph))
        return out

    def airy_half(s):
        q1, q2 = kz(1.0, s), kz(2.25, s)
        return (q1 - q2) / (q1 + q2), (2.25 * q1 - q2) / (2.25 * q1 + q2)

    cases = (([], airy_half), ([hpfem.Layer(hpfem.Material(layer_eps, 1.0), d)], airy))

    for layers, reflection in cases:
        stack = hpfem.AnisotropicStack(hpfem.Material.dielectric(1.0), layers, glass)
        f_perp, f_par = sommerfeld(reflection, 1.0, H)
        assert dipole_above_stack(stack, K0, H, [0, 0, 1], azimuths=8) == pytest.approx(
            f_perp, abs=1e-9
        )
        assert dipole_above_stack(stack, K0, H, [1, 0, 0], azimuths=8) == pytest.approx(
            f_par, abs=1e-9
        )
    assert abs(f_perp - 1) > 0.1


def test_c_cut_crystal_half_space():
    """A uniaxial half-space with the optic axis along the normal: r_s with n_o and the closed
    form r_p = (ε_o q1 − q2e)/(ε_o q1 + q2e), q2e = sqrt(ε_o (1 − s²/ε_e)) (cover vacuum)."""
    eo, ee = 2.25, 3.4
    crystal = hpfem.Material.anisotropic(np.diag([eo, eo, ee]).astype(complex))
    stack = hpfem.AnisotropicStack(hpfem.Material.dielectric(1.0), [], crystal)

    def reflection(s):
        q1, qo = kz(1.0, s), kz(eo, s)
        qe = cmath.sqrt(eo * (1 - s * s / ee))
        qe = qe if qe.imag >= 0 else -qe
        return (q1 - qo) / (q1 + qo), (eo * q1 - qe) / (eo * q1 + qe)

    f_perp, f_par = sommerfeld(reflection, 1.0, H)
    assert dipole_above_stack(stack, K0, H, [0, 0, 1], azimuths=8) == pytest.approx(
        f_perp, abs=1e-9
    )
    assert dipole_above_stack(stack, K0, H, [0, 1, 0], azimuths=8) == pytest.approx(f_par, abs=1e-9)


def test_tilted_crystal_rotational_covariance():
    """An absorbing crystal layer with a tilted axis on glass: rotating the crystal and the
    dipole together about the normal leaves F unchanged; the azimuth rule has converged; the
    tilt makes F depend on the in-plane orientation of the dipole."""

    def stack_at(alpha):
        axis = np.array([math.cos(alpha) * 0.8, math.sin(alpha) * 0.8, 0.6])
        eps = 2.25 * np.eye(3) + 1.2 * np.outer(axis, axis) + 0.15j * np.eye(3)
        return hpfem.AnisotropicStack(
            hpfem.Material.dielectric(1.0),
            [hpfem.Layer(hpfem.Material.anisotropic(eps), 0.15e-6)],
            hpfem.Material.dielectric(1.5),
        )

    m = np.array([0.6, 0.3, 0.5])
    f = dipole_above_stack(stack_at(0.0), K0, H, m, azimuths=48)
    assert dipole_above_stack(stack_at(0.0), K0, H, m, azimuths=96) == pytest.approx(f, abs=1e-9)
    alpha = 0.7
    rot = np.array([[math.cos(alpha), -math.sin(alpha), 0], [math.sin(alpha), math.cos(alpha), 0],
                    [0, 0, 1]])  # fmt: skip
    assert dipole_above_stack(stack_at(alpha), K0, H, rot @ m, azimuths=48) == pytest.approx(
        f, abs=1e-9
    )
    fx = dipole_above_stack(stack_at(0.0), K0, H, [1, 0, 0], azimuths=48)
    fy = dipole_above_stack(stack_at(0.0), K0, H, [0, 1, 0], azimuths=48)
    assert abs(fx - fy) > 1e-3


UM = 1e-6
OMEGA = 2 * math.pi * hpfem.constants.c0 / UM  # vacuum wavelength 1 um, K0 above
PERIOD, SIGMA = 1.0 * UM, 0.05 * UM


def emitter_cell(d, bottom=-2 * UM, top=2 * UM, h=0.1 * UM):
    ny = round((top - bottom) / h)
    mesh = hpfem.rectangle(10, ny, [-PERIOD / 2, bottom], [PERIOD / 2, top])
    for c in range(mesh.num_cells):
        y = np.mean([mesh.vertex(int(v))[1] for v in mesh.cell_vertices(c)])
        if -d < y < 0:
            mesh.set_cell_tag(c, 4)
        elif y < -d:
            mesh.set_cell_tag(c, 5)
    return mesh


@pytest.mark.parametrize(
    "kx, beta, moment, tol",
    [
        (0.3, 0.2, [1, 0, 0], 1e-6),
        (0.3, 0.2, [0, 1, 0], 1e-6),
        (0.1, 1.3, [0, 1, 0], 1e-4),  # beyond the light cone of the cover: evanescent coupling
        (0.1, 1.3, [0, 0, 1], 1e-4),
    ],
)
def test_emitter_sample_above_a_crystal_layer(kx, beta, moment, tol):
    """M19 S7b: one Bloch sample of grating.emit above a tilted absorbing crystal layer on glass
    (AnisotropicStack) against the plane-wave spectrum on the orders."""
    from hpfem import grating
    from hpfem.planar import array_sample_above_stack

    d, height = 0.2 * UM, 0.32 * UM
    axis = np.array([0.8, 0.0, 0.6])
    crystal = hpfem.Material.anisotropic(
        2.25 * np.eye(3) + 1.2 * np.outer(axis, axis) + 0.15j * np.eye(3)
    )
    glass = hpfem.Material.dielectric(1.5)
    stack = hpfem.AnisotropicStack(hpfem.Material.dielectric(1.0), [hpfem.Layer(crystal, d)], glass)
    dipole = {"position": (0.05 * UM, height), "moment": moment, "sigma": SIGMA}
    result = grating.emit(emitter_cell(d), {4: crystal, 5: glass}, stack, dipole, OMEGA,
                          kx * K0, beta * K0, order=4, pml={"top": UM, "bottom": UM})  # fmt: skip
    reference = array_sample_above_stack(stack, K0, kx * K0, beta * K0, PERIOD, height, moment,
                                         SIGMA)  # fmt: skip
    assert result.P_cell / reference - 1 == pytest.approx(0.0, abs=tol)
    assert not result.warnings


def test_emitter_above_a_tilted_crystal_substrate_with_the_modal_bottom():
    """A lossless crystal half-space with a tilted axis, truncated by the modal boundary: the
    delivered power against the spectrum, the transmitted power resolved into the ordinary and
    extraordinary channels, and the balance P_cell = up + down (nothing absorbed or guided)."""
    from hpfem import grating
    from hpfem.planar import array_sample_above_stack

    axis = np.array([np.sin(0.7), 0.3, np.cos(0.7)])
    crystal = hpfem.Material.anisotropic(2.4 * np.eye(3) + 1.0 * np.outer(axis, axis) / 1.09)
    stack = hpfem.AnisotropicStack(hpfem.Material.dielectric(1.0), [], crystal)
    height = 0.32 * UM
    mesh = emitter_cell(0.0, bottom=-1.0 * UM)
    dipole = {"position": (0.05 * UM, height), "moment": [0.6, 0.5, 0.4], "sigma": SIGMA}
    kx, beta = 0.3 * K0, 0.4 * K0
    result = grating.emit(mesh, {5: crystal}, stack, dipole, OMEGA, kx, beta, order=4,
                          pml={"top": UM}, bottom="modal")  # fmt: skip
    reference = array_sample_above_stack(stack, K0, kx, beta, PERIOD, height, dipole["moment"],
                                         SIGMA)  # fmt: skip
    assert result.P_cell / reference - 1 == pytest.approx(0.0, abs=1e-5)
    zeroth = next(o for o in result.orders_down if o.m == 0)
    assert set(zeroth.channel_power) == {"ordinary", "extraordinary"}
    assert sum(zeroth.channel_power.values()) == pytest.approx(zeroth.power, rel=1e-9)
    assert result.up + result.down == pytest.approx(result.P_cell, rel=1e-4)
    # the Gaussian reaching into the substrate is reported
    close = dict(dipole, position=(0.05 * UM, 0.15 * UM))
    near = grating.emit(mesh, {5: crystal}, stack, close, OMEGA, kx, beta, order=2,
                        pml={"top": UM}, bottom="modal")  # fmt: skip
    assert any("reaches across" in w for w in near.warnings)

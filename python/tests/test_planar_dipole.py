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


def test_crystal_light_lines():
    """M19 S7c: the light line (k_y = 0) of a crystal from its Fresnel quartic, against the
    closed forms of a diagonal tensor (solver frame): E ∥ y on kx² + β² = ε_yy k0², E in the
    xz plane on kx²/ε_zz + β²/ε_xx = k0²; the roots travel along the sign of kx; the turning
    points of β are sqrt(ε_xx) and sqrt(ε_yy) (times k0)."""
    from hpfem import grating

    eps = np.diag([2.0, 3.0, 2.5])
    beta = 0.4 * K0
    real, branch = grating._crystal_kx_roots(eps, K0, beta)
    expected = sorted([math.sqrt(3 * K0**2 - beta**2), math.sqrt(2.5 * (K0**2 - beta**2 / 2))])
    roots = sorted(r for r, _s in real)
    assert np.allclose(roots, [-expected[1], -expected[0], *expected], rtol=1e-10)
    assert all(s == math.copysign(1.0, r) for r, s in real)
    assert branch == []
    # beyond both sheets: four complex roots, the branch points of the evanescent sheets
    real, branch = grating._crystal_kx_roots(eps, K0, 1.8 * K0)
    assert real == [] and len(branch) == 4
    extrema = sorted(grating._crystal_beta_extrema(eps, K0))
    assert np.allclose(extrema, [math.sqrt(2.0) * K0, math.sqrt(3.0) * K0], rtol=1e-10)


def test_scan_rule_with_a_crystal_half_space():
    """The array-scan rule with the crystal light lines (β over (−β_max, β_max), the full kx zone)
    integrates the exact sample spectrum of a dipole above a tilted lossless crystal half-space
    to the 2D-spectrum reference of S7a (1e-4 at 6 nodes per panel; 2e-6 at 10)."""
    from hpfem import grating
    from hpfem.planar import array_sample_above_stack

    axis = np.array([np.sin(0.7), 0.3, np.cos(0.7)])
    eps_lab = 2.4 * np.eye(3) + np.outer(axis, axis) / 1.09
    stack = hpfem.AnisotropicStack(
        hpfem.Material.dielectric(1.0), [], hpfem.Material.anisotropic(eps_lab)
    )
    swap = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0.0]])  # laboratory <-> solver frame
    height, moment = 0.32 * UM, np.array([0.6, 0.5, 0.4])
    reference = dipole_above_stack(stack, K0, height, swap @ moment, azimuths=96)
    rule = grating.array_scan_rule(PERIOD, K0, [1.0, swap @ eps_lab @ swap], beta_max=1.9 * K0,
                                   nodes=6, depth=0.0)  # fmt: skip
    assert rule.beta.min() < 0 < rule.beta.max()
    value = sum(w * array_sample_above_stack(stack, K0, kx.real, b, PERIOD, height, moment)
                for kx, b, w in zip(rule.kx, rule.beta, rule.weight, strict=True))  # fmt: skip
    bulk = hpfem.constants.Z0 * K0**2 * float(moment @ moment) / (12 * math.pi)
    assert value.real / bulk == pytest.approx(reference, rel=3e-4)
    with pytest.raises(grating.GratingError, match="mirror"):
        grating.array_scan_rule(PERIOD, K0, [1.0, swap @ eps_lab @ swap], beta_max=K0,
                                symmetric=True)  # fmt: skip


def test_dipole_emission_refusals_with_crystals():
    """What the crystal scan of S7c does not do yet is refused up front."""
    from hpfem import grating

    crystal = hpfem.Material.anisotropic(np.diag([2.0, 2.0, 2.6]).astype(complex))
    stack = hpfem.AnisotropicStack(hpfem.Material.dielectric(1.0), [], crystal)
    mesh = emitter_cell(0.0)
    dipole = {"position": (0.05 * UM, 0.32 * UM), "sigma": SIGMA}
    common = dict(order=2, pml={"top": UM, "bottom": UM})
    with pytest.raises(grating.GratingError, match="S7d"):
        grating.dipole_emission(mesh, {5: crystal}, stack, dipole, OMEGA, **common)
    with pytest.raises(grating.GratingError, match="mirror"):
        grating.dipole_emission(mesh, {5: crystal}, stack, dipole, OMEGA, symmetric=True,
                                channels=False, **common)  # fmt: skip
    with pytest.raises(grating.GratingError, match="real Bloch"):
        grating.dipole_emission(mesh, {5: crystal}, stack, dipole, OMEGA, bottom="modal",
                                channels=False, order=2, pml={"top": UM})  # fmt: skip


def test_emission_pattern_above_a_crystal_stack():
    """M19 S7d: the angle-resolved emission by reciprocity above a tilted crystal layer on an
    isotropic substrate equals |p · E_pw(x0)|² of the analytic stack field of the incident plane
    wave (``AnisotropicStack.plane_wave``), up into the cover and down into the substrate (the
    cell, the tensors and the stack mirrored), per polarisation and at azimuths without symmetry.
    The Gaussian factor is that of the host medium, e^{−(n_host k0 σ)²}, on both sides."""
    from hpfem import grating

    period, d, height = 0.5 * UM, 0.2 * UM, 0.32 * UM
    axis = np.array([0.8, 0.0, 0.6])
    eps = 2.25 * np.eye(3) + 1.1 * np.outer(axis, axis)
    crystal = hpfem.Material.anisotropic(eps)
    substrate, cover = hpfem.Material.dielectric(2.0), hpfem.Material.dielectric(1.0)
    stack = hpfem.AnisotropicStack(cover, [hpfem.Layer(crystal, d)], substrate)
    mirror = np.diag([1.0, 1.0, -1.0])  # z -> -z, laboratory frame
    mirrored = hpfem.AnisotropicStack(
        substrate, [hpfem.Layer(hpfem.Material.anisotropic(mirror @ eps @ mirror), d)], cover, d
    )
    mesh = hpfem.rectangle(5, 40, [-period / 2, -2 * UM], [period / 2, 2 * UM])
    for c in range(mesh.num_cells):
        y = np.mean([mesh.vertex(int(v))[1] for v in mesh.cell_vertices(c)])
        if -d < y < 0:
            mesh.set_cell_tag(c, 4)
        elif y < -d:
            mesh.set_cell_tag(c, 5)
    moment = np.array([0.6, 0.5, 0.4])  # solver frame
    swap = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0.0]])
    directions = [(0.3, 0.4, "up"), (0.6, 2.0, "up"), (0.2, -1.0, "down"), (0.5, 2.5, "down")]
    dipole = {"position": (0.05 * UM, height), "moment": moment, "sigma": SIGMA}
    materials = {4: crystal, 5: substrate}
    pml = {"top": UM, "bottom": UM}
    pattern = grating.emission_pattern(mesh, materials, stack, dipole, OMEGA, directions,
                                       order=4, pml=pml)  # fmt: skip
    gauss = math.exp(-((K0 * SIGMA) ** 2))  # host: vacuum
    for i, (theta, phi, side) in enumerate(directions):
        for j, pol in enumerate((hpfem.Polarisation.S, hpfem.Polarisation.P)):
            if side == "up":
                wave, x, p, n = stack, [0.05 * UM, 0.0, height], swap @ moment, 1.0
            else:
                wave, x, p, n = mirrored, [0.05 * UM, 0.0, -height], mirror @ swap @ moment, 2.0
            e = np.asarray(wave.plane_wave(K0, theta, pol, 1.0, phi + math.pi).field.value(x))
            scale = n * K0**2 * hpfem.constants.Z0 * gauss / (32 * math.pi**2)
            reference = scale * abs(p @ e) ** 2
            assert pattern.dP_dOmega[i, j] == pytest.approx(reference, rel=2e-4)
    # a crystal substrate has no 'down' directions yet
    crystal_sub = hpfem.AnisotropicStack(cover, [], crystal)
    with pytest.raises(grating.GratingError, match="crystal substrate"):
        grating.emission_pattern(mesh, {4: crystal, 5: crystal}, crystal_sub, dipole, OMEGA,
                                 [(0.2, 0.0, "down")], order=2, pml=pml)  # fmt: skip

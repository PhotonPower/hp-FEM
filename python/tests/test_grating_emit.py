"""Emission of a Bloch-periodic array of Gaussian dipoles (M17 S1, ADR-0013): in a homogeneous
cell the delivered power, the power of every Floquet order up and down and the Poynting fluxes
match the closed form of a phased array of smeared dipoles (a sum of current sheets, one per
order), for three orientations and several (kx, beta), with orders evanescent and beyond the
light cone; on a lossy ridge grating the delivered power balances the fluxes and the
absorption; argument checks."""

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
DIPOLE_AT = (0.05 * UM, 0.1 * UM)
MU0 = 4e-7 * math.pi * 1.00000000055  # CODATA 2018 value used by the library
# 1 um of PML: at 67 degrees (kx, beta = 0.6 K, 0.7 K) 0.5 um reflect 1.7 % back onto the
# source, 1 um 0.35 % (mesh size and order change nothing there)
PML = 1.0 * UM


def homogeneous_cell(h=0.1 * UM):
    nx, ny = round(PERIOD / h), round(4.0 * UM / h)
    mesh = hpfem.rectangle(nx, ny, [-PERIOD / 2, -2.0 * UM], [PERIOD / 2, 2.0 * UM])
    medium = hpfem.Material.dielectric(N)
    stack = hpfem.LayerStack2D(medium, [], medium, 0.0)
    return mesh, stack


def closed_form(kx, beta, p):
    """Per order and side: omega mu0 |p_perp|^2 exp(-sigma^2 k^2) / (8 P k_y)."""
    out = {}
    for m in range(-4, 5):
        km = kx + 2 * math.pi * m / PERIOD
        ky2 = K**2 - km**2 - beta**2
        if ky2 <= 0:
            continue
        ky = math.sqrt(ky2)
        for side, s in (("up", 1.0), ("down", -1.0)):
            khat = np.array([km, s * ky, beta]) / K
            perp = p - khat * (khat @ p)
            out[(m, side)] = (
                OMEGA * MU0 * float(np.vdot(perp, perp).real) * math.exp(-((SIGMA * K) ** 2))
                / (8 * PERIOD * ky)
            )  # fmt: skip
    return out


@pytest.mark.parametrize(
    "kx, beta, tol",
    [
        (0.0, 0.0, 5e-3),
        (0.3 * K, 0.4 * K, 5e-3),
        (0.5 * K, 0.5 * K, 5e-3),
        # 67 degrees: the PML (1 um) still reflects up to 0.8 % back onto the source
        (0.6 * K, 0.7 * K, 1.5e-2),
        (0.2 * K, 1.2 * K, 5e-3),  # beyond the light cone
    ],
)
def test_homogeneous_cell_matches_the_phased_array_closed_form(kx, beta, tol):
    mesh, stack = homogeneous_cell()
    for moment in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.3, 1j, -0.5]):
        p = np.array(moment, dtype=complex)
        dipole = {"position": DIPOLE_AT, "moment": p, "sigma": SIGMA}
        res = grating.emit(mesh, {}, stack, dipole, OMEGA, kx, beta, order=4,
                           pml={"top": PML, "bottom": PML}, orders_max=2)  # fmt: skip
        exact = closed_form(kx, beta, p)
        total = sum(exact.values())
        if total == 0.0:  # beyond the light cone: nothing radiates, nothing is delivered
            reference = sum(closed_form(0.0, 0.0, p).values())
            assert abs(res.P_cell) < 1e-6 * reference
            assert abs(res.flux_up) + abs(res.flux_down) < 1e-6 * reference
            continue
        assert res.P_cell == pytest.approx(total, rel=tol)
        assert res.flux_up + res.flux_down == pytest.approx(total, rel=tol)
        for o in res.orders_up + res.orders_down:
            expected = exact.get((o.m, o.side), 0.0)
            assert abs(o.power - expected) < tol * total, (o.m, o.side, o.power, expected)
        assert abs(res.guided) < tol * total and res.absorbed == 0.0


def ridge_cell():
    """The structured ridge cell of test_grating_solve: period 400 nm, ridge 200 x 148 nm on
    glass, with a lossy ridge."""
    nm = units.nm
    period, height, ridge = 400 * nm, 148 * nm, 200 * nm
    row = height / 6
    y0, y1 = -(16 + 17) * row, (6 + 15 + 17) * row
    mesh = hpfem.rectangle(16, 16 + 17 + 6 + 15 + 17, [-period / 2, y0], [period / 2, y1])
    for c in range(mesh.num_cells):
        x = np.mean([mesh.vertex(int(v)) for v in mesh.cell_vertices(c)], axis=0)
        if x[1] < 0:
            mesh.set_cell_tag(c, 2)
        elif x[1] < height and abs(x[0]) < ridge / 2:
            mesh.set_cell_tag(c, 3)
    lossy = hpfem.Material()
    lossy.eps_r = 2.25 + 0.3j
    glass = hpfem.Material.dielectric(1.5)
    stack = hpfem.LayerStack2D(hpfem.Material.dielectric(1.0), [], glass, 0.0)
    return mesh, {2: glass, 3: lossy}, stack, 17 * row, height


def test_lossy_ridge_balances_delivered_power_fluxes_and_absorption():
    mesh, materials, stack, pml, height = ridge_cell()
    omega = units.angular_frequency(wavelength=405 * units.nm)
    k0 = omega / hpfem.constants.c0
    dipole = {"position": (0.0, height + 60 * units.nm), "moment": (1.0, 0.5, 0.2j),
              "sigma": 20 * units.nm}  # fmt: skip
    res = grating.emit(mesh, materials, stack, dipole, omega, 0.2 * k0, 0.3 * k0, order=4,
                       pml={"top": pml, "bottom": pml}, orders_max=2)  # fmt: skip
    assert res.P_cell > 0 and res.absorbed > 0 and set(res.A_by_tag) == {3}
    # no guided modes inside the cover light cone: delivered = out through the PML + absorbed
    assert res.flux_up + res.flux_down + res.absorbed == pytest.approx(res.P_cell, rel=1e-2)
    assert abs(res.guided) < 1e-2 * res.P_cell
    assert res.up == pytest.approx(res.flux_up, rel=2e-2)
    assert res.down == pytest.approx(res.flux_down, rel=2e-2)
    assert res.field([[0.0, height + 100 * units.nm]]).shape == (1, 3)


def test_emit_argument_checks():
    mesh, materials, stack, pml, height = ridge_cell()
    omega = units.angular_frequency(wavelength=405 * units.nm)
    common = dict(pml={"top": pml, "bottom": pml}, order=2)
    nm = units.nm

    def run(position, sigma=20 * nm, moment=(1.0, 0.0, 0.0)):
        dipole = {"position": position, "moment": moment, "sigma": sigma}
        return grating.emit(mesh, materials, stack, dipole, omega, **common)

    with pytest.raises(grating.GratingError, match="along x"):
        run((190 * nm, height + 60 * nm))  # 6 sigma would cross the Bloch face
    with pytest.raises(grating.GratingError, match="PML"):
        run((0.0, (6 + 15 + 16) * height / 6))  # inside the top PML
    with pytest.raises(grating.GratingError, match="lossless"):
        run((0.0, height / 2))  # inside the lossy ridge
    with pytest.raises(grating.GratingError):
        run((0.0, height + 60 * nm), sigma=0.0)
    with pytest.raises(grating.GratingError):
        run((0.0, height + 60 * nm), moment=(1.0, 0.0))


def flat_glass_cell(pml_rows):
    """Air over glass, period 400 nm at 405 nm: the orders m = +-1 are barely evanescent in
    air (decay length ~ 280 nm at beta = 0.2 k0), the situation of a Rayleigh anomaly."""
    nm = units.nm
    row = 148 * nm / 6
    y0, y1 = -(16 + pml_rows) * row, (6 + 15 + pml_rows) * row
    mesh = hpfem.rectangle(16, 16 + pml_rows + 6 + 15 + pml_rows, [-200 * nm, y0], [200 * nm, y1])
    for c in range(mesh.num_cells):
        x = np.mean([mesh.vertex(int(v)) for v in mesh.cell_vertices(c)], axis=0)
        if x[1] < 0:
            mesh.set_cell_tag(c, 2)
    glass = hpfem.Material.dielectric(1.5)
    stack = hpfem.LayerStack2D(hpfem.Material.dielectric(1.0), [], glass, 0.0)
    return mesh, {2: glass}, stack, pml_rows * row


def test_measurement_lines_clear_the_dipole_and_the_pml_leak_is_reported():
    nm = units.nm
    omega = units.angular_frequency(wavelength=405 * nm)
    k0 = omega / hpfem.constants.c0
    common = dict(order=3, orders_max=2)
    # the default cover line (midway to the top PML, 259 nm) would cut a dipole at 250 nm
    mesh, materials, stack, pml = flat_glass_cell(17)
    dipole = {"position": (0.0, 250 * nm), "moment": (1.0, 0.0, 0.0), "sigma": 8 * nm}
    res = grating.emit(mesh, materials, stack, dipole, omega, 0.0, 0.2 * k0,
                       pml={"top": pml, "bottom": pml}, **common)  # fmt: skip
    assert res.cover_line >= 250 * nm + 6 * 8 * nm
    with pytest.raises(grating.GratingError, match="cover_line"):
        grating.emit(mesh, materials, stack, dipole, omega, 0.0, 0.2 * k0, cover_line=280 * nm,
                     pml={"top": pml, "bottom": pml}, **common)  # fmt: skip
    # evanescent orders reaching a thin PML exchange power with it: flux != order power (with the
    # PML designed for the largest order angle, #156, 0.24 % at 296 nm of PML; it converges to the
    # order power as the PML moves away)
    dipole = {"position": (0.0, 160 * nm), "moment": (1.0, 0.3, 0.2j), "sigma": 8 * nm}
    leaks = {}
    for rows in (12, 30):
        mesh, materials, stack, pml = flat_glass_cell(rows)
        r = grating.emit(mesh, materials, stack, dipole, omega, 0.0, 0.2 * k0,
                         pml={"top": pml, "bottom": pml}, **common)  # fmt: skip
        assert abs(r.pml_leak - (r.flux_up - r.up + r.flux_down - r.down)) < 1e-12 * r.P_cell
        assert r.guided == pytest.approx(r.P_cell - r.up - r.down - r.absorbed, rel=1e-12)
        leaks[rows] = (r.pml_leak / r.P_cell, r.warnings)
    assert abs(leaks[12][0]) > 1.5e-3 and leaks[12][1]  # 296 nm of PML: 0.24 %, warned
    assert abs(leaks[30][0]) < 1e-3 and not leaks[30][1]  # 740 nm: below 1e-3

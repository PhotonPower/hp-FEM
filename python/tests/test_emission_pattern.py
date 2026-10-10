"""Emission pattern by reciprocity (M17 S2, hpfem.grating.emission_pattern): the Gaussian dipole
in a homogeneous cell against its closed-form pattern n k0^2 Z0 |p_perp|^2 e^{-(n k0 sigma)^2}
/ (32 pi^2) per direction and polarisation, integrated over the sphere (a spherical 3-design)
to P_bulk; and a dipole above a glass substrate against the plane-wave Fresnel far field
(direct + reflected wave in the cover, transmitted wave in the substrate, also beyond the
critical angle), computed here with NumPy independently of the solver."""

import numpy as np
import pytest

import hpfem
from hpfem import grating, units

NM = units.nm
C0 = hpfem.constants.c0
Z0 = hpfem.constants.Z0
WAVELENGTH = 600 * NM
OMEGA = units.angular_frequency(wavelength=WAVELENGTH)
K0 = 2 * np.pi / WAVELENGTH
PERIOD = 200 * NM
HALF = 600 * NM
PML = 300 * NM
COVER, SUBSTRATE = 1, 2


def cell(n_cover, n_substrate):
    """Structured cell |y| < 600 nm, the interface y = 0 on a mesh line, PML 300 nm."""
    mesh = hpfem.rectangle(6, 36, [-PERIOD / 2, -HALF], [PERIOD / 2, HALF])
    for c in range(mesh.num_cells):
        mesh.set_cell_tag(c, SUBSTRATE if mesh.cell_centroid(c)[1] < 0 else COVER)
    materials = {COVER: hpfem.Material.dielectric(n_cover),
                 SUBSTRATE: hpfem.Material.dielectric(n_substrate)}  # fmt: skip
    stack = hpfem.LayerStack2D(materials[COVER], [], materials[SUBSTRATE], 0.0)
    return mesh, materials, stack


def direction(theta, phi, side):
    sign = 1.0 if side == "up" else -1.0
    return np.array(
        [np.sin(theta) * np.cos(phi), sign * np.cos(theta), np.sin(theta) * np.sin(phi)]
    )


COMMON = dict(order=2, pml={"top": PML, "bottom": PML}, check=False)


def test_homogeneous_pattern_matches_the_dipole_and_integrates_to_p_bulk():
    n = 1.5
    mesh, materials, stack = cell(n, n)
    moment = np.array([1.0, 0.5j, 0.3])
    sigma = 8 * NM
    dipole = {"position": (10 * NM, 40 * NM), "moment": moment, "sigma": sigma}
    # the eight cube corners: a spherical 3-design, exact for the quadratic dipole pattern
    theta = np.arccos(1 / np.sqrt(3))
    phis = np.radians([45, 135, 225, 315])
    directions = [(theta, phi, side) for side in ("up", "down") for phi in phis]
    result = grating.emission_pattern(mesh, materials, stack, dipole, OMEGA, directions,
                                      **COMMON)  # fmt: skip
    p_bulk = (
        hpfem.dipole_vacuum_power(np.linalg.norm(moment), OMEGA)
        * n
        * np.exp(-((n * K0 * sigma) ** 2))
    )
    assert result.P_bulk == pytest.approx(p_bulk, rel=1e-12)
    assert result.dP_dOmega.shape == (8, 2) and np.all(result.n == n)
    scale = n * K0**2 * Z0 / (32 * np.pi**2) * np.exp(-((n * K0 * sigma) ** 2))
    for i, (t, f, side) in enumerate(directions):
        r = direction(t, f, side)
        p_perp2 = np.vdot(moment, moment).real - abs(r @ moment) ** 2
        assert result.total[i] == pytest.approx(scale * p_perp2, rel=1e-6), (t, f, side)
    integral = 4 * np.pi / len(directions) * result.total.sum()
    assert integral == pytest.approx(p_bulk, rel=1e-6)
    normalized = grating.emission_pattern(mesh, materials, stack, dipole, OMEGA, directions[:1],
                                          normalized=True, **COMMON)  # fmt: skip
    assert normalized.dP_dOmega[0] == pytest.approx(result.dP_dOmega[0] / p_bulk, rel=1e-9)
    # a dipole along the lines seen in the x-y plane: all of it is s polarised
    along_z = {"position": (0.0, 0.0), "moment": (0.0, 0.0, 1.0), "sigma": sigma}
    s_only = grating.emission_pattern(mesh, materials, stack, along_z, OMEGA,
                                      [(np.radians(30), 0.0, "up")], **COMMON)  # fmt: skip
    assert s_only.dP_dOmega[0, 1] < 1e-9 * s_only.dP_dOmega[0, 0]


def fresnel_pattern(moment, y0, n1, n2, sigma, theta, phi, side):
    """dP/dOmega (s, p) of a dipole at height y0 > 0 in medium n1 above medium n2 (interface
    y = 0): the reciprocity amplitude from the plane-wave Fresnel fields, NumPy only."""
    x0 = np.array([0.0, y0, 0.0])
    yhat = np.array([0.0, 1.0, 0.0])
    r = direction(theta, phi, side)
    n_in = n1 if side == "up" else n2
    k_in = -n_in * K0 * r  # the plane wave arrives from the direction r
    kx, kz = k_in[0], k_in[2]
    k1y = np.sqrt(complex((n1 * K0) ** 2 - kx**2 - kz**2))
    k2y = np.sqrt(complex((n2 * K0) ** 2 - kx**2 - kz**2))
    es = np.cross(k_in, yhat)
    es = np.array([0.0, 0.0, 1.0]) if np.linalg.norm(es) < 1e-12 * K0 else es / np.linalg.norm(es)

    def wave(k):
        return np.exp(1j * (k @ x0))

    if side == "up":
        k_r = np.array([kx, k1y, kz])
        e_s = es * (wave(k_in) + (k1y - k2y) / (k1y + k2y) * wave(k_r))
        r_h = (n2**2 * k1y - n1**2 * k2y) / (n2**2 * k1y + n1**2 * k2y)
        e_p = (np.cross(es, k_in / (n1 * K0)) * wave(k_in)
               + r_h * np.cross(es, k_r / (n1 * K0)) * wave(k_r))  # fmt: skip
    else:
        k_t = np.array([kx, k1y, kz])  # upward into the cover, evanescent beyond the critical angle
        e_s = es * 2 * k2y / (k2y + k1y) * wave(k_t)
        t_h = 2 * n1**2 * k2y / (n1**2 * k2y + n2**2 * k1y)
        e_p = (n2 / n1) * t_h * np.cross(es, k_t / (n1 * K0)) * wave(k_t)
    smear = np.exp(-((n1 * K0 * sigma) ** 2))  # every wave at the dipole has k.k = (n1 k0)^2
    scale = n_in * K0**2 * Z0 / (32 * np.pi**2) * smear
    return np.array([scale * abs(moment @ e_s) ** 2, scale * abs(moment @ e_p) ** 2])


def test_dipole_above_glass_matches_the_fresnel_far_field():
    n1, n2 = 1.0, 1.5
    mesh, materials, stack = cell(n1, n2)
    moment = np.array([0.4, 1.0, 0.2j])
    y0, sigma = 80 * NM, 8 * NM
    dipole = {"position": (0.0, y0), "moment": moment, "sigma": sigma}
    directions = [(0.0, 0.0, "up"), (np.radians(30), 0.0, "up"),
                  (np.radians(60), np.radians(40), "up"), (np.radians(20), 0.0, "down"),
                  (np.radians(50), np.radians(30), "down")]  # fmt: skip
    # the last direction lies beyond the critical angle of glass (41.8 deg): forbidden light
    result = grating.emission_pattern(mesh, materials, stack, dipole, OMEGA, directions,
                                      **COMMON)  # fmt: skip
    assert list(result.n) == [n1, n1, n1, n2, n2]
    for i, d in enumerate(directions):
        expected = fresnel_pattern(moment, y0, n1, n2, sigma, *d)
        if d[0] == 0.0:  # normal incidence: s and p are a convention, their sum is not
            assert result.total[i] == pytest.approx(expected.sum(), rel=1e-6), d
        else:
            assert result.dP_dOmega[i] == pytest.approx(expected, rel=1e-6), d
    assert result.dP_dOmega[4].sum() > 0.05 * result.dP_dOmega[3].sum()  # forbidden light
    with pytest.raises(grating.GratingError, match="side"):
        grating.emission_pattern(mesh, materials, stack, dipole, OMEGA, [(0.1, 0.0, "left")])
    with pytest.raises(grating.GratingError, match="theta"):
        grating.emission_pattern(mesh, materials, stack, dipole, OMEGA, [(2.0, 0.0, "up")])


def test_grating_pattern_agrees_with_the_phased_array_of_stage_a():
    # a glass ridge on glass, the dipole in the air above it: Stage C (plane waves) against the
    # m = 0 order of the phased array of Stage A (dipole source) at the same (kx, beta), by
    # the array scanning dP/dOmega = P / (4 pi^2) (n k0)^2 cos(theta) P_0(kx, beta)
    period = 400 * NM
    mesh = hpfem.rectangle(16, 48, [-period / 2, -HALF], [period / 2, HALF])
    for c in range(mesh.num_cells):
        x, y = mesh.cell_centroid(c)
        if y < 0:
            mesh.set_cell_tag(c, SUBSTRATE)
        elif y < 100 * NM and abs(x) < 100 * NM:
            mesh.set_cell_tag(c, 3)
        else:
            mesh.set_cell_tag(c, COVER)
    glass = hpfem.Material.dielectric(1.5)
    materials = {COVER: hpfem.Material.vacuum(), SUBSTRATE: glass, 3: glass}
    stack = hpfem.LayerStack2D(hpfem.Material.vacuum(), [], glass, 0.0)
    dipole = {"position": (40 * NM, 160 * NM), "moment": (0.3, 1.0, 0.5j), "sigma": 8 * NM}
    kx, beta = 0.3 * K0, 0.2 * K0
    common = dict(order=3, pml={"top": PML, "bottom": PML})
    # the cover line above the Gaussian (the default, midway to the PML, would cut through it)
    array = grating.emit(mesh, materials, stack, dipole, OMEGA, kx, beta, orders_max=2,
                         cover_line=260 * NM, **common)  # fmt: skip
    p0 = next(o.power for o in array.orders_up if o.m == 0)
    sin_t = np.hypot(kx, beta) / K0
    theta, phi = np.arcsin(sin_t), np.arctan2(beta, kx)
    directions = [(theta, phi, "up"), (theta, phi + np.pi, "up")]
    pattern = grating.emission_pattern(mesh, materials, stack, dipole, OMEGA, directions,
                                       check=False, **common)  # fmt: skip
    expected = period / (4 * np.pi**2) * K0**2 * np.cos(theta) * p0
    assert pattern.total[0] == pytest.approx(expected, rel=1e-4)  # 2e-5 on this cell
    assert pattern.total[1] > 2 * pattern.total[0]  # the opposite direction differs: no symmetry

"""Derivatives of resonances (M16 S4): the Fabry-Perot slab of test_grating_resonances, whose
complex resonance k = (m pi - i ln((n+1)/(n-1))) / (n d) is known, for the permittivity and the
thickness (exact derivatives and re-solved resonances), beta and the Bloch wavenumber against
re-solved resonances, and a dispersive (Lorentz) slab: the self-consistent resonance of
refine_resonance against the exact one and its derivative with the d eps / d omega term."""

import numpy as np
import pytest

import hpfem
from hpfem import grating, units
from hpfem.materials import DrudeLorentz

NM = units.nm
C0 = hpfem.constants.c0
INDEX = 3.5
THICKNESS = 300 * NM
PERIOD = 100 * NM
MARGIN = 150 * NM
PML = 900 * NM
MODE = 4
SLAB = 2


def slab_cell(thickness=THICKNESS):
    half = THICKNESS / 2 + MARGIN + PML
    ny = round(2 * half / (THICKNESS / 4))
    mesh = hpfem.rectangle(1, ny, [0.0, -half], [PERIOD, half])
    for c in range(mesh.num_cells):
        if abs(mesh.cell_centroid(c)[1]) < THICKNESS / 2:
            mesh.set_cell_tag(c, SLAB)
    if thickness != THICKNESS:  # move the slab faces (and nothing else)
        hpfem.move_nodes(mesh, face_velocity(mesh), (thickness - THICKNESS) / 2)
    return mesh


def face_velocity(mesh):
    """The vertices on the slab faces y = +-d/2 move outward along y (the slab spans the period,
    so region_normal_velocity would also move its nodes on the Bloch faces)."""
    y = np.asarray(mesh.vertices, dtype=float)[:, 1]
    v = np.zeros((len(y), 2))
    on = np.abs(np.abs(y) - THICKNESS / 2) < 1e-6 * THICKNESS
    v[on, 1] = np.sign(y[on])
    return v


def exact_k(n, d=THICKNESS, m=MODE):
    return (m * np.pi - 1j * np.log((n + 1) / (n - 1))) / (n * d)


STACK = hpfem.LayerStack2D(hpfem.Material.vacuum(), [], hpfem.Material.vacuum())


def solve(mesh, materials, kx=0.0, beta=0.0, target=None):
    target = 0.97 * exact_k(INDEX).real * C0 if target is None else target
    pml = {"top": PML, "bottom": PML}
    return grating.resonances(mesh, materials, STACK, target, kx=kx, beta=beta, num_modes=4,
                              order=3, pml=pml, krylov_dimension=40)  # fmt: skip


def closest(result, omega):
    return min(result.modes, key=lambda m: abs(m.omega - omega))


def test_slab_resonance_derivatives_match_the_exact_ones_and_re_solved_modes():
    mesh = slab_cell()
    eps = INDEX**2
    result = solve(mesh, {SLAB: hpfem.Material(eps_r=eps)})
    k = exact_k(INDEX)
    mode = closest(result, k * C0)
    velocity = face_velocity(mesh)  # both faces outward: d grows by 2 per unit
    sens = grating.resonance_sensitivity(result, mode, [("eps", SLAB), ("shape", velocity)])
    assert set(sens) == {f"eps[{SLAB}].re", f"eps[{SLAB}].im", "shape[1]"}
    # exact: dk/dn of the Fabry-Perot formula, d eps = 2 n dn; dk/dd = -k/d
    dlog = 1 / (INDEX + 1) - 1 / (INDEX - 1)
    dk_dn = -1j * dlog / (INDEX * THICKNESS) - k / INDEX
    exact_eps = C0 * dk_dn / (2 * INDEX)
    exact_shape = C0 * (-2 * k / THICKNESS)
    d_eps = sens[f"eps[{SLAB}].re"].domega
    d_shape = sens["shape[1]"].domega
    assert abs(d_eps - exact_eps) < 5e-3 * abs(exact_eps)  # the discretisation of the mode
    assert abs(d_shape - exact_shape) < 5e-3 * abs(exact_shape)
    # against re-solved resonances of the same discretisation
    h = 1e-4
    fd_eps = (closest(solve(mesh, {SLAB: hpfem.Material(eps_r=eps + h)}), mode.omega).omega
              - closest(solve(mesh, {SLAB: hpfem.Material(eps_r=eps - h)}), mode.omega).omega
              ) / (2 * h)  # fmt: skip
    assert abs(d_eps - fd_eps) < 1e-6 * abs(fd_eps)
    fd_im = (closest(solve(mesh, {SLAB: hpfem.Material(eps_r=eps + 1j * h)}), mode.omega).omega
             - closest(solve(mesh, {SLAB: hpfem.Material(eps_r=eps - 1j * h)}), mode.omega).omega
             ) / (2 * h)  # fmt: skip
    assert abs(sens[f"eps[{SLAB}].im"].domega - fd_im) < 1e-6 * abs(fd_im)
    t = 1e-4 * THICKNESS
    fd_shape = (closest(solve(slab_cell(THICKNESS + 2 * t), {SLAB: hpfem.Material(eps_r=eps)}),
                        mode.omega).omega
                - closest(solve(slab_cell(THICKNESS - 2 * t), {SLAB: hpfem.Material(eps_r=eps)}),
                          mode.omega).omega) / (2 * t)  # fmt: skip
    assert abs(d_shape - fd_shape) < 1e-5 * abs(fd_shape)
    # Q and wavelength follow from d omega
    q = mode.Q
    q_plus = closest(solve(mesh, {SLAB: hpfem.Material(eps_r=eps + h)}), mode.omega).Q
    q_minus = closest(solve(mesh, {SLAB: hpfem.Material(eps_r=eps - h)}), mode.omega).Q
    assert sens[f"eps[{SLAB}].re"].dquality == pytest.approx((q_plus - q_minus) / (2 * h), rel=1e-5)
    assert q > 0


def test_beta_and_bloch_wavenumber_derivatives_match_re_solved_modes():
    mesh = slab_cell()
    materials = {SLAB: hpfem.Material(eps_r=INDEX**2)}
    kx, beta = 2.0e6, 1.5e6  # [1/m], conical and off-normal
    result = solve(mesh, materials, kx=kx, beta=beta)
    mode = closest(result, exact_k(INDEX) * C0)
    sens = grating.resonance_sensitivity(result, mode, ["beta", "kx"])
    h = 0.01 * kx  # the O(h^2) error of the reference meets the eigensolver noise near here
    for name, plus, minus in (
        ("beta", solve(mesh, materials, kx=kx, beta=beta + h),
         solve(mesh, materials, kx=kx, beta=beta - h)),
        ("kx", solve(mesh, materials, kx=kx + h, beta=beta),
         solve(mesh, materials, kx=kx - h, beta=beta)),
    ):  # fmt: skip
        fd = (closest(plus, mode.omega).omega - closest(minus, mode.omega).omega) / (2 * h)
        assert abs(sens[name].domega - fd) < 1e-4 * abs(fd), (name, sens[name].domega, fd)
    with pytest.raises(grating.GratingError):
        grating.resonance_sensitivity(result, mode, ["height"])


def lorentz_slab():
    """A dispersive slab: a Drude term and one Lorentz oscillator at twice the resonance
    frequency, eps = 14 - 4 + 2.7 + small loss near the mode (n about 3.6)."""
    w0 = 2.0 * exact_k(INDEX).real * C0
    return DrudeLorentz(eps_inf=14.0, omega_p=w0, gamma=0.0, oscillators=[(2.0, w0, 0.05 * w0)])


def exact_dispersive_omega(model, d=THICKNESS):
    """The self-consistent Fabry-Perot resonance: omega = c0 k(n(omega)), scalar Newton."""
    omega = exact_k(INDEX, d) * C0
    for _ in range(50):
        n = np.sqrt(complex(model.eps_r(omega)))
        g = omega - C0 * exact_k(n, d)
        h = 1e-7 * abs(omega)
        n_h = np.sqrt(complex(model.eps_r(omega + h)))
        dg = 1 - C0 * (exact_k(n_h, d) - exact_k(n, d)) / h
        omega = omega - g / dg
        if abs(g) < 1e-14 * abs(omega):
            break
    return omega


def test_refine_resonance_of_a_dispersive_slab_and_its_derivative():
    model = lorentz_slab()
    exact = exact_dispersive_omega(model)
    mesh = slab_cell()
    result = solve(mesh, {SLAB: model}, target=0.97 * exact.real)
    assert not result.self_consistent
    frozen = closest(result, exact)
    refined = grating.refine_resonance(result, frozen, tolerance=1e-12)
    assert refined.self_consistent and 1 <= refined.iterations <= 8
    mode = refined.modes[0]
    # self-consistent: eps evaluated at the mode's own (complex) omega reproduces the mode
    assert abs(mode.omega - exact) < 3e-3 * abs(exact)  # the discretisation
    assert abs(mode.omega - frozen.omega) > 10 * abs(mode.omega - exact)  # refining mattered
    # the derivative along the thickness with the d eps / d omega term, against refined
    # resonances on moved meshes and the exact implicit derivative
    velocity = face_velocity(mesh)
    d = grating.resonance_sensitivity(refined, mode, [("shape", velocity)])["shape[0]"].domega
    t = 1e-4 * THICKNESS

    def refined_at(thickness):
        moved = slab_cell(thickness)
        first = solve(moved, {SLAB: model}, target=0.97 * exact.real)
        return grating.refine_resonance(first, closest(first, mode.omega)).modes[0].omega

    fd = (refined_at(THICKNESS + 2 * t) - refined_at(THICKNESS - 2 * t)) / (2 * t)
    assert abs(d - fd) < 1e-5 * abs(fd)
    exact_d = (
        exact_dispersive_omega(model, THICKNESS + 1e-4 * THICKNESS)
        - exact_dispersive_omega(model, THICKNESS - 1e-4 * THICKNESS)
    ) / (1e-4 * THICKNESS)
    assert abs(d - exact_d) < 5e-3 * abs(exact_d)  # fmt: skip
    # without the dispersive term the derivative is that of the frozen pencil: clearly off
    frozen_d = grating.resonance_sensitivity(
        grating.ResonanceResult(**{**refined.__dict__, "self_consistent": False}),
        mode,
        [("shape", velocity)],
    )["shape[0]"].domega
    assert abs(frozen_d - fd) > 1e-2 * abs(fd)
    with pytest.raises(grating.GratingError):
        grating.refine_resonance(solve(mesh, {SLAB: hpfem.Material(eps_r=INDEX**2)}))

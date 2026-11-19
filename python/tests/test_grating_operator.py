"""The grating cell as an operator (issue #216): one factorisation per (kx, β, ω) for s and p
from the cover and from the substrate (``grating.operator``)."""

import numpy as np
import pytest

import hpfem
from hpfem import grating, units

LAM = 1e-6
OMEGA = units.angular_frequency(wavelength=LAM)
SUB, RIDGE, AIR = 5, 6, 7
PERIOD, HEIGHT = 0.6 * LAM, 0.2 * LAM
PML = {"top": 0.5 * LAM, "bottom": 0.5 * LAM}
FLIP = np.diag([1.0, -1.0])


def _cell(h=0.1 * LAM, bottom=-1.2 * LAM, top=1.4 * LAM):
    """A dielectric ridge (n = 2, half the period) on glass in air; every cell tagged, so that the
    cell mirrored at y = 0 keeps its materials."""
    ny = int(round((top - bottom) / h))
    mesh = hpfem.rectangle(int(round(PERIOD / h)), ny, [0.0, bottom], [PERIOD, bottom + ny * h])
    for c in range(mesh.num_cells):
        x = np.mean([mesh.vertex(int(v)) for v in mesh.cell_vertices(c)], axis=0)
        if x[1] < 0:
            mesh.set_cell_tag(c, SUB)
        elif x[1] < HEIGHT and x[0] < PERIOD / 2:
            mesh.set_cell_tag(c, RIDGE)
        else:
            mesh.set_cell_tag(c, AIR)
    return mesh


GLASS = hpfem.Material.dielectric(1.5)
MATERIALS = {SUB: GLASS, RIDGE: hpfem.Material.dielectric(2.0), AIR: hpfem.Material.dielectric(1.0)}
STACK = hpfem.LayerStack2D(hpfem.Material.dielectric(1.0), [], GLASS, 0.0)


def _operator(mesh, theta, phi, order=2, **options):
    return grating.operator(mesh, MATERIALS, STACK, theta, phi, OMEGA, order=order, pml=PML,
                            orders_max=2, **options)  # fmt: skip


def _close(a, b, tol):
    return np.abs(a - b).max() <= tol * max(1.0, np.abs(b).max())


def test_cover_blocks_equal_jones_for_random_angles():
    """Acceptance 1: for 20 seeded random (θ, φ) the cover blocks of the operator equal
    grating.jones (two separate solves) to the solver's rounding: 1e-8, since grating.jones
    reproduces itself only to about 2e-9 (the parallel assembly sums in a varying order and the
    PML system amplifies the rounding; with one thread the two agree to 1e-10)."""
    mesh = _cell()
    rng = np.random.default_rng(216)
    for theta, phi in zip(rng.uniform(0.0, 1.0, 20), rng.uniform(0.0, 2 * np.pi, 20),
                          strict=True):  # fmt: skip
        op = _operator(mesh, theta, phi, substrate=False)
        ref = grating.jones(mesh, MATERIALS, STACK, theta, phi, OMEGA, order=2, pml=PML,
                            orders_max=2)  # fmt: skip
        assert op.substrate is None
        for side in ("R", "T"):
            got = op.cover.R_orders if side == "R" else op.cover.T_orders
            want = ref.R_orders if side == "R" else ref.T_orders
            assert [o.m for o in got] == [o.m for o in want]
            for a, b in zip(got, want, strict=True):
                assert _close(a.jones, b.jones, 1e-8), (theta, phi, side, a.m)
        assert op.cover.s.R == pytest.approx(ref.s.R, abs=1e-8)
        assert op.cover.p.T == pytest.approx(ref.p.T, abs=1e-8)


def test_one_factorisation_for_all_four_excitations(monkeypatch):
    """Acceptance 2: s and p from the cover and from the substrate cost one ConicalScattering
    solve (the factorisation); the other three are loads on its kept factors."""
    mesh = _cell()
    calls = []
    original = hpfem.ConicalScattering.solve

    def counting(self):
        calls.append(self)
        return original(self)

    monkeypatch.setattr(hpfem.ConicalScattering, "solve", counting)
    op = _operator(mesh, 0.4, 0.9)
    assert len(calls) == 1
    assert op.factorisations == 1
    assert op.substrate is not None
    assert {"factorisation", "loads", "solves", "postprocess", "total"} <= set(op.timing)


def test_reciprocity_of_the_four_blocks():
    """Acceptance 3: a reciprocal cell. The reversed path (in-plane wavevector −(kx, β), φ + π)
    gives the transposed block in the s / p basis of conical_polarisation, J(φ + π) =
    D J(φ)ᵀ D with D = diag(1, −1): the specular reflection on either side (equal diagonals,
    J_ps(φ) = −J_sp(φ + π)) and the order-0 transmission, cover to substrate at φ against
    substrate to cover at φ + π. The efficiencies are measured on lines, so the discrete problem
    is reciprocal only up to the discretisation error, which falls with p and h (reflection /
    transmission: 2e-3 / 6e-3 at p = 2, 1e-4 / 8e-4 at p = 3, 3e-5 / 8e-5 at p = 3 on the twice
    finer mesh)."""
    mesh = _cell()
    theta, phi = 0.45, 0.8
    a = _operator(mesh, theta, phi, order=3)
    b = _operator(mesh, theta, phi + np.pi, order=3)
    for side in ("cover", "substrate"):
        ja = getattr(a, side).order(0, "R").jones
        jb = getattr(b, side).order(0, "R").jones
        assert abs(ja[1, 0]) > 1e-2  # the ridge converts s into p
        assert np.abs(jb - FLIP @ ja.T @ FLIP).max() < 3e-4, side
    t12 = a.cover.order(0, "T").jones
    t21 = b.substrate.order(0, "T").jones
    assert abs(t12[1, 0]) > 1e-2
    assert np.abs(t21 - FLIP @ t12.T @ FLIP).max() < 2e-3


def test_substrate_incidence_against_the_mirrored_cell():
    """Acceptance 4: the substrate blocks equal grating.jones on the cell mirrored at y = 0 (the
    substrate on top) at the substrate angle: the same discrete problem mirrored, so to the
    solver's rounding (1e-7 with the parallel assembly), with the p sign of the mirrored basis
    (p̂ → −M p̂, J = D J_mirrored D)."""
    mesh = _cell()
    theta, phi = 0.5, 0.6
    op = _operator(mesh, theta, phi, order=3)
    assert op.theta_substrate == pytest.approx(np.arcsin(np.sin(theta) / 1.5))
    mirrored = grating.jones(grating._mirrored_cell(mesh), MATERIALS,
                             grating._mirrored_stack(STACK), op.theta_substrate, phi, OMEGA,
                             order=3, pml=PML, orders_max=2)  # fmt: skip
    for got, want in ((op.substrate.R_orders, mirrored.R_orders),
                      (op.substrate.T_orders, mirrored.T_orders)):  # fmt: skip
        assert [o.m for o in got] == [o.m for o in want]
        assert len(got) >= 1
        for a, b in zip(got, want, strict=True):
            assert _close(a.jones, FLIP @ b.jones @ FLIP, 1e-7), (a.side, a.m)
    assert len(op.substrate.R_orders) > 1  # an order beyond the specular one in the glass
    # a lossless cell conserves the power from below as well, to the discretisation error as
    # from above (4e-4 at p = 3, 6e-6 at p = 4)
    for res in (op.substrate.s, op.substrate.p, op.cover.s, op.cover.p):
        assert abs(res.power_balance_residual) < 1e-3, res.power_balance_residual
        assert res.A == pytest.approx(0.0, abs=1e-12)


def test_substrate_side_availability():
    """``substrate="auto"`` skips a lossy substrate and a crystal stack, ``True`` refuses them;
    the cover blocks of a crystal stack equal grating.jones."""
    mesh = _cell()
    lossy = hpfem.Material(complex(2.25, 0.3), 1.0)
    stack = hpfem.LayerStack2D(hpfem.Material.dielectric(1.0), [], lossy, 0.0)
    materials = {**MATERIALS, SUB: lossy}
    op = grating.operator(mesh, materials, stack, 0.3, 0.4, OMEGA, order=2, pml=PML,
                          orders_max=1)  # fmt: skip
    assert op.substrate is None and op.theta_substrate is None
    with pytest.raises(grating.GratingError, match="substrate"):
        grating.operator(mesh, materials, stack, 0.3, 0.4, OMEGA, order=2, pml=PML,
                         orders_max=1, substrate=True)  # fmt: skip
    with pytest.raises(grating.GratingError, match="substrate="):
        _operator(mesh, 0.3, 0.4, substrate="both")
    crystal = hpfem.Material.anisotropic(np.diag([2.25, 2.4, 2.6]).astype(complex))
    cstack = hpfem.AnisotropicStack(hpfem.Material.dielectric(1.0), [], crystal)
    cmaterials = {**MATERIALS, SUB: crystal}
    op = grating.operator(mesh, cmaterials, cstack, 0.3, 0.4, OMEGA, order=2, pml=PML,
                          orders_max=1)  # fmt: skip
    ref = grating.jones(mesh, cmaterials, cstack, 0.3, 0.4, OMEGA, order=2, pml=PML,
                        orders_max=1)  # fmt: skip
    assert op.substrate is None
    for a, b in zip(op.cover.T_orders, ref.T_orders, strict=True):
        assert a.channels == b.channels
        assert _close(a.jones, b.jones, 1e-8)

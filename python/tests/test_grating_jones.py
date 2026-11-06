"""Crystal stacks, mode-resolved orders, the modal bottom and the Jones / Mueller matrices of
hpfem.grating (ADR-0015 §2, §7; M19 S5c)."""

import numpy as np
import pytest

import hpfem
from hpfem import grating, units

LAM = 1e-6
OMEGA = units.angular_frequency(wavelength=LAM)
K0 = 2 * np.pi / LAM
PLATE, SUB, RIDGE = 4, 5, 6


def uniaxial(eo, ee, axis):
    a = np.asarray(axis, dtype=float)
    a /= np.linalg.norm(a)
    return eo * np.eye(3) + (ee - eo) * np.outer(a, a)


def cell(period, nx, layers, h=0.05 * LAM, bottom=-1.2 * LAM, top=1.4 * LAM):
    """Structured cell; ``layers`` maps a cell (centroid) to a tag or None."""
    ny = int(round((top - bottom) / h))
    mesh = hpfem.rectangle(nx, ny, [0.0, bottom], [period, bottom + ny * h])
    for c in range(mesh.num_cells):
        x = np.mean([mesh.vertex(int(v)) for v in mesh.cell_vertices(c)], axis=0)
        tag = layers(x)
        if tag is not None:
            mesh.set_cell_tag(c, tag)
    return mesh


def test_flat_crystal_stack_reproduces_the_stack_jones_matrix():
    """A crystal plate on a c-cut crystal substrate given as AnisotropicStack: the reflected
    Jones matrix of order 0 is the stack's with the off-diagonal signs of the conical_polarisation
    basis (p̂ = ŝ × k̂ in the solver frame, a reflection of the laboratory frame), the
    transmitted channels are the stack's ordinary / extraordinary channels."""
    d = 0.2 * LAM
    plate = hpfem.Material.anisotropic(uniaxial(2.25, 2.9, [1.0, 0.5, 0.7]))
    sub = hpfem.Material.anisotropic(uniaxial(2.4, 2.9, [0.0, 0.0, 1.0]))
    stack = hpfem.AnisotropicStack(hpfem.Material.dielectric(1.0), [hpfem.Layer(plate, d)], sub)
    mesh = cell(0.3 * LAM, 4, lambda x: PLATE if -d < x[1] < 0 else (SUB if x[1] < -d else None))
    theta, phi = 0.5, 0.7
    result = grating.jones(mesh, {PLATE: plate, SUB: sub}, stack, theta, phi, OMEGA, order=3,
                           pml={"top": 0.5 * LAM, "bottom": 0.5 * LAM}, orders_max=1)  # fmt: skip
    ref = stack.response(K0, theta, phi)
    flip = np.diag([1.0, -1.0])
    assert np.allclose(result.order(0).jones, flip @ ref.reflection @ flip, atol=1e-8)
    assert abs(ref.reflection[1, 0]) > 1e-2
    t = result.order(0, "T")
    assert t.channels == ("ordinary", "extraordinary")
    assert np.allclose(np.abs(t.jones) ** 2, ref.channel_transmittance, atol=1e-8)
    assert result.s.T_orders[1].channels[0].kind == "ordinary"
    # Mueller: the efficiency for unpolarised light, the matrix of the identity
    j = result.order(0).jones
    assert result.order(0).mueller[0, 0] == pytest.approx(0.5 * np.sum(np.abs(j) ** 2))
    assert np.allclose(grating.mueller(np.eye(2)), np.eye(4))
    assert result.s.R == pytest.approx(ref.reflectance[0], abs=1e-8)


def test_reciprocity_of_a_crystal_grating():
    """ADR-0015 §9 item 8 in the conical solver: a tilted crystal ridge on glass; the reversed
    path of the specular order is the incidence at phi + pi, J_ss and J_pp agree and
    J_ps(phi) = -J_sp(phi + pi) (the β ↔ −β pairing of the transposed system)."""
    period, height = 0.6 * LAM, 0.2 * LAM
    ridge = hpfem.Material.anisotropic(uniaxial(2.25, 3.2, [1.0, 0.6, 0.5]))
    glass = hpfem.Material.dielectric(1.5)
    stack = hpfem.LayerStack2D(hpfem.Material.dielectric(1.0), [], glass, 0.0)
    mesh = cell(period, 12, lambda x: SUB if x[1] < 0 else
                (RIDGE if x[1] < height and x[0] < 0.35 * period else None))  # fmt: skip
    common = dict(order=3, pml={"top": 0.6 * LAM, "bottom": 0.6 * LAM}, orders_max=2)
    f = grating.jones(mesh, {SUB: glass, RIDGE: ridge}, stack, 0.4, 0.5, OMEGA, **common)
    g = grating.jones(mesh, {SUB: glass, RIDGE: ridge}, stack, 0.4, 0.5 + np.pi, OMEGA, **common)
    jf, jg = f.order(0).jones, g.order(0).jones
    assert abs(jf[1, 0]) > 0.01  # the crystal converts s into p
    assert abs(jf[0, 0] - jg[0, 0]) < 1e-5
    assert abs(jf[1, 1] - jg[1, 1]) < 1e-5
    assert abs(jf[1, 0] + jg[0, 1]) < 2e-5
    assert abs(jf[0, 1] + jg[1, 0]) < 2e-5


def test_backward_order_needs_the_modal_bottom():
    """A lossless grating on a tilted crystal substrate whose transmitted order -1 is a backward
    wave: bottom='pml' is refused, bottom='modal' closes the energy balance."""
    period, height = 0.4835 * LAM, 0.2 * LAM
    sub = hpfem.Material.anisotropic(uniaxial(2.4, 3.4, [np.sin(0.7), 0.0, np.cos(0.7)]))
    stack = hpfem.AnisotropicStack(hpfem.Material.dielectric(1.0), [], sub)
    ridge = hpfem.Material.dielectric(1.5)
    mesh = cell(period, 4, lambda x: SUB if x[1] < 0 else
                (RIDGE if x[1] < height and x[0] < period / 2 else None),
                bottom=-0.5 * LAM)  # fmt: skip
    args = (mesh, {SUB: sub, RIDGE: ridge}, stack, "p", 0.5, 0.676, OMEGA)
    with pytest.raises(grating.GratingError, match="backward"):
        grating.solve(*args, order=3, pml={"top": 0.6 * LAM, "bottom": 0.3 * LAM})
    result = grating.solve(*args, order=4, pml={"top": 0.6 * LAM}, bottom="modal", orders_max=3)
    assert abs(result.power_balance_residual) < 1e-5, result.power_balance_residual
    minus_one = next(o for o in result.T_orders if o.m == -1)
    assert minus_one.propagating and minus_one.channels[1].kind == "extraordinary"
    assert minus_one.channels[1].efficiency > 1e-3


def test_crystal_stack_frames_and_refusals():
    """A crystal in a LayerStack2D (frame ambiguous) is refused with the hint to
    AnisotropicStack; features without crystal-stack support refuse an AnisotropicStack."""
    crystal = hpfem.Material.anisotropic(uniaxial(2.4, 2.9, [0.0, 0.0, 1.0]))
    mesh = cell(0.3 * LAM, 2, lambda x: SUB if x[1] < 0 else None)
    flat = hpfem.LayerStack2D(hpfem.Material.dielectric(1.0), [], crystal, 0.0)
    with pytest.raises(grating.GratingError, match="AnisotropicStack"):
        grating.solve(mesh, {SUB: crystal}, flat, "s", 0.3, 0.0, OMEGA, order=2)
    stack = hpfem.AnisotropicStack(hpfem.Material.dielectric(1.0), [], crystal)
    found = grating.validate(mesh, {SUB: crystal}, stack, "s", 0.3, 0.0, OMEGA, order=2)
    assert "setup" not in [d.code for d in found]
    with pytest.raises(grating.GratingError, match="AnisotropicStack"):
        grating._prepare(mesh, {SUB: crystal}, stack, "s", 0.3, 0.0, OMEGA, pml=None,
                         bottom="pml", orders_max=1, snap_tolerance=1e-9, pml_target=1e-6,
                         pml_wavelengths=0.5, cover_line=None, substrate_line=None)  # fmt: skip


def _lamellar_jones(eps_ridge, period, height, theta, phi, h=0.025 * LAM):
    """Order-0 reflection Jones matrix of a lamellar grating (fill 1/2, ridge ``eps_ridge`` in the
    laboratory frame, air between) of the given period and height on glass."""
    glass = hpfem.Material.dielectric(1.5)
    ridge = (hpfem.Material.anisotropic(np.asarray(eps_ridge, dtype=complex))
             if np.ndim(eps_ridge) == 2 else hpfem.Material(complex(eps_ridge), 1.0))  # fmt: skip
    stack = hpfem.LayerStack2D(hpfem.Material.dielectric(1.0), [], glass, 0.0)
    nx = max(2, round(period / h))
    nx += nx % 2
    mesh = hpfem.rectangle(nx, round(2.6 * LAM / h), [0.0, -1.2 * LAM],
                           [period, -1.2 * LAM + round(2.6 * LAM / h) * h])  # fmt: skip
    for c in range(mesh.num_cells):
        x = np.mean([mesh.vertex(int(v)) for v in mesh.cell_vertices(c)], axis=0)
        if x[1] < 0:
            mesh.set_cell_tag(c, SUB)
        elif x[1] < height and x[0] < period / 2:
            mesh.set_cell_tag(c, RIDGE)
    result = grating.jones(mesh, {SUB: glass, RIDGE: ridge}, stack, theta, phi, OMEGA, order=4,
                           pml={"top": 0.5 * LAM, "bottom": 0.5 * LAM}, orders_max=1)  # fmt: skip
    return result.order(0).jones


def test_form_birefringence_against_the_effective_medium_plate():
    """ADR-0015 §9 item 5 (M19 S3c): a sub-wavelength lamellar grating of isotropic materials
    reflects like the uniaxial plate of effective-medium theory, ε_∥ = f ε1 + (1 − f) ε2 along
    the lines and the normal, ε_⊥ = (f/ε1 + (1 − f)/ε2)⁻¹ across them (optic axis along x). The
    difference falls like P/λ, not (P/λ)²: the evanescent orders at the top and the bottom of the
    grating form boundary layers of thickness ~P that the plate lacks (the FEM error, 4e-5 by
    mesh refinement, is far below); the Richardson extrapolation 2 J(P/2) − J(P) removes that
    term and falls like (P/λ)². The same with crystal lamellae whose axes are those of the
    lamellae, component by component."""
    height, theta, phi = 0.3 * LAM, 0.5, 0.7
    flip = np.diag([1.0, -1.0])  # conical_polarisation basis against the stack's (J = D r D)
    for eps_ridge in (4.0, np.diag([4.0, 3.0, 2.5])):
        diag = np.diag(np.asarray(eps_ridge, dtype=float) * np.ones(3)) if np.ndim(eps_ridge) == 0 \
            else np.asarray(eps_ridge, dtype=float)  # fmt: skip
        par = 0.5 * np.diag(diag) + 0.5  # arithmetic, per component
        perp = 1.0 / (0.5 / np.diag(diag) + 0.5)  # harmonic
        emt = np.diag([perp[0], par[1], par[2]])  # x across the lines
        plate = hpfem.AnisotropicStack(hpfem.Material.dielectric(1.0),
                                       [hpfem.Layer(hpfem.Material.anisotropic(emt), height)],
                                       hpfem.Material.dielectric(1.5))  # fmt: skip
        # the stack refers its reflection to the top of the plate, the grating to y = 0 below it
        shift = np.exp(-2j * K0 * np.cos(theta) * height)
        reference = shift * flip @ plate.response(K0, theta, phi).reflection @ flip
        jones = [_lamellar_jones(eps_ridge, period, height, theta, phi)
                 for period in (0.2 * LAM, 0.1 * LAM, 0.05 * LAM)]  # fmt: skip
        errors = [np.abs(j - reference).max() for j in jones]
        extrapolated = [np.abs(2 * fine - coarse - reference).max()
                        for coarse, fine in zip(jones[:-1], jones[1:], strict=True)]  # fmt: skip
        assert abs(reference[1, 0]) > 1e-2  # the form birefringence converts s into p
        assert errors[0] / errors[1] > 1.4 and errors[1] / errors[2] > 1.6, errors  # ~ P
        assert extrapolated[0] / extrapolated[1] > 3.5, extrapolated  # ~ P²
        assert extrapolated[1] < 1.5e-3, extrapolated

"""Crystals in the 2D / 3D volume solver through the bindings (ADR-0015 §2, M19 S2)."""

import numpy as np
from conftest import mie_problem

import hpfem
from hpfem import diagnostics as dg


def codes(diagnostics):
    return [d.code for d in diagnostics]


def uniaxial(no, ne, axis):
    a = np.asarray(axis, dtype=float)
    a /= np.linalg.norm(a)
    return no**2 * np.eye(3) + (ne**2 - no**2) * np.outer(a, a)


def with_scatterer(material):
    mesh, dofs, problem = mie_problem(n=4, p=2)
    setup = problem.setup
    setup.materials.set(2, material)
    return mesh, dofs, setup


def test_eps_identity_through_the_tensor_path_matches_the_scalar_path():
    mesh, dofs, setup = with_scatterer(hpfem.Material.dielectric(1.5))
    scalar = hpfem.Scattering2D(dofs, setup).solve().unknown
    setup.materials.set(2, hpfem.Material.anisotropic(2.25 * np.eye(3)))
    tensor = hpfem.Scattering2D(dofs, setup).solve().unknown
    assert np.linalg.norm(tensor - scalar) < 1e-12 * np.linalg.norm(scalar)
    # eps_xx enters the in-plane problem, eps_zz does not
    setup.materials.set(2, hpfem.Material.anisotropic(np.diag([2.25, 2.25, 7.0])))
    same = hpfem.Scattering2D(dofs, setup).solve().unknown
    assert np.linalg.norm(same - scalar) < 1e-12 * np.linalg.norm(scalar)
    setup.materials.set(2, hpfem.Material.anisotropic(np.diag([4.0, 2.25, 2.25])))
    other = hpfem.Scattering2D(dofs, setup).solve().unknown
    assert np.linalg.norm(other - scalar) > 1e-3 * np.linalg.norm(scalar)


def test_volume_diagnostics_for_crystals():
    # a crystal filling the domain, PML included: aligned is fine, tilted in the plane is not
    mesh, dofs, setup = with_scatterer(hpfem.Material.dielectric(1.5))
    setup.materials = hpfem.MaterialMap(hpfem.Material.anisotropic(uniaxial(1.5, 1.7, [1, 0, 0])))
    assert "pml_backward_wave" not in codes(dg.validate_scattering(mesh, setup, 2))
    tilted = uniaxial(1.5, 1.7, [np.cos(0.5), np.sin(0.5), 0.0])
    setup.materials = hpfem.MaterialMap(hpfem.Material.anisotropic(tilted))
    found = [d for d in dg.validate_scattering(mesh, setup, 2) if d.code == "pml_backward_wave"]
    assert len(found) == 2 and all(d.severity == "warning" for d in found)  # x and y layers
    # a tensor coupling the plane to z: refused in 2D
    coupled = uniaxial(1.5, 1.7, [1, 0, 1])
    setup.materials = hpfem.MaterialMap(hpfem.Material.dielectric(1.0)).set(
        2, hpfem.Material.anisotropic(coupled)
    )
    found = dg.validate_scattering(mesh, setup, 2)
    assert codes(found) == ["anisotropic_unsupported"] and "tag 2" in found[0].text
    # the conical solver takes the coupling crystal (M19 S3), its scalar E_z path does not
    conical = hpfem.ConicalScatteringSetup()
    conical.omega = setup.omega
    conical.materials = setup.materials
    assert "anisotropic_unsupported" not in codes(dg.validate_scattering(mesh, conical, 2))
    conical.scalar_ez = True
    assert codes(dg.validate_scattering(mesh, conical, 2)) == ["anisotropic_unsupported"]
    # a tilted crystal in its PML: the 3D condition on the stretched axes x and y
    conical.scalar_ez = False
    conical.pml = setup.pml
    conical.materials = hpfem.MaterialMap(hpfem.Material.anisotropic(coupled))
    found = [d for d in dg.validate_scattering(mesh, conical, 2) if d.code == "pml_backward_wave"]
    assert len(found) >= 1


def test_pml_tensor_stretch_and_stability_condition():
    box = hpfem.PmlBox2D.uniform([-1.0, -1.0], [1.0, 1.0], 0.5, 6.0)
    x = [1.2, -1.3]
    eps = np.array([[2.0, 0.3], [0.3, 3.0]], dtype=complex)
    s = np.asarray(box.stretch(x))
    expected = eps * np.prod(s) / np.outer(s, s)
    assert np.allclose(box.permittivity(eps, x), expected, rtol=1e-14, atol=0)
    assert np.allclose(box.permittivity(2.25 + 0j, x), box.permittivity(2.25 * np.eye(2), x))
    assert hpfem.pml_backward_wave_2d(eps, 0) is not None
    assert hpfem.pml_backward_wave_2d(np.diag([2.0, 3.0]).astype(complex), 0) is None
    # aligned biaxial: fails only along the intermediate principal axis
    biaxial = np.diag([2.0, 3.0, 4.5]).astype(complex)
    assert [hpfem.pml_backward_wave_3d(biaxial, d) is None for d in range(3)] == [
        True,
        False,
        True,
    ]
    direction, worst = hpfem.pml_backward_wave_3d(biaxial, 1)
    assert -0.02 < worst < 0 and abs(np.linalg.norm(direction) - 1) < 1e-12
    crystal = hpfem.Material.anisotropic(np.diag([2.25, 2.25, 4.0]))
    assert abs(crystal.max_real_index - 2.0) < 1e-12
    assert abs(hpfem.Material.dielectric(1.5).max_real_index - 1.5) < 1e-12


def test_grating_with_a_crystal_ridge():
    """M19 S3: a lossy crystal ridge with a tilted axis (eps_xz, eps_yz) on glass at conical
    incidence: orders, absorption and the flux balance close; the automatic scalar path steps
    aside for the coupling crystal; the reciprocity paths still refuse crystals."""
    from test_grating_solve import OMEGA, RIDGE_TAG, SUB, unit_cell

    from hpfem import grating, materials, units

    mesh, pml = unit_cell()
    glass = hpfem.Material.dielectric(1.5)
    stack = hpfem.LayerStack2D(hpfem.Material.dielectric(1.0), [], glass, 0.0)
    eps = uniaxial(1.6, 1.9, [1.0, 0.5, 0.8]) + 0.05j * np.eye(3)
    crystal = hpfem.Material.anisotropic(eps)
    common = dict(order=3, pml={"top": pml, "bottom": pml}, orders_max=2)
    for pol, theta, phi in (("p", 40.0, 30.0), ("s", 30.0, 0.0)):
        result = grating.solve(mesh, {SUB: glass, RIDGE_TAG: crystal}, stack, pol,
                               theta * units.deg, phi * units.deg, OMEGA, **common)  # fmt: skip
        assert not result.scalar
        assert result.A > 1e-3
        assert abs(result.power_balance_residual) < 3e-3, result.power_balance_residual
        fb = result.flux_balance
        assert fb is not None and abs(fb["relative_residual"]) < 5e-3
    # the same crystal as a dispersive model
    model = materials.Anisotropic([2.5 + 0.05j, 3.6 + 0.05j, 3.0 + 0.05j],
                                  materials.axis_rotation(theta=0.6, phi=0.4))  # fmt: skip
    result = grating.solve(mesh, {SUB: glass, RIDGE_TAG: model}, stack, "p", 40 * units.deg,
                           30 * units.deg, OMEGA, **common)  # fmt: skip
    assert abs(result.power_balance_residual) < 3e-3
    assert "anisotropic_unsupported" not in codes(
        grating.validate(
            mesh,
            {SUB: glass, RIDGE_TAG: crystal},
            stack,
            "p",
            40 * units.deg,
            30 * units.deg,
            OMEGA,
            **common,
        )  # fmt: skip
    )


def test_grating_takes_crystals_in_the_laboratory_frame():
    """hpfem.materials gives z as the layer normal; the conical solver has y as the normal and z
    along the lines: a c-cut crystal (axis along the normal) has its axis along y there, an
    axis along the lines (lab y) lies along z."""
    from hpfem import grating, materials

    c_cut = grating._to_conical_frame(materials.Uniaxial(2.0, 3.0, theta=0.0).at(1e15))
    assert np.allclose(np.asarray(c_cut.eps_tensor), np.diag([2.0, 3.0, 2.0]))
    along = materials.Uniaxial(2.0, 3.0, theta=np.pi / 2, phi=np.pi / 2).at(1e15)
    lines = grating._to_conical_frame(along)
    assert np.allclose(np.asarray(lines.eps_tensor), np.diag([2.0, 2.0, 3.0]), atol=1e-12)
    stack = hpfem.LayerStack2D(hpfem.Material.dielectric(1.0), [], hpfem.Material.dielectric(1.5))
    tensor_map = hpfem.MaterialMap(hpfem.Material.dielectric(1.0)).set(3, along)
    import pytest

    with pytest.raises(grating.GratingError, match="laboratory frame"):
        grating._material_map(tensor_map, stack, 1e15)


def test_anisotropic_stack_binding():
    """M19 S4: the bound AnisotropicStack reproduces LayerStack3D for isotropic layers and
    conserves energy for lossless crystals."""
    k0 = 2 * np.pi / 1e-6
    glass = hpfem.Material.dielectric(1.5)
    layers = [hpfem.Layer(hpfem.Material(4.0 + 0.3j, 1.0), 120e-9)]
    stack = hpfem.AnisotropicStack(hpfem.Material.dielectric(1.0), layers, glass)
    r = stack.response(k0, 0.5, 0.3)
    ref = hpfem.LayerStack3D(hpfem.Material.dielectric(1.0), layers, glass, 0.0)
    s = ref.plane_wave(k0, 0.5, hpfem.Polarisation.S, 1.0, 0.3)
    p = ref.plane_wave(k0, 0.5, hpfem.Polarisation.P, 1.0, 0.3)
    assert abs(r.reflection[0, 0] - s.reflection) < 1e-12
    assert abs(r.reflection[1, 1] - p.reflection) < 1e-12
    crystal = hpfem.Material.anisotropic(uniaxial(1.5, 1.8, [1.0, 1.0, 1.0]))
    lossless = hpfem.AnisotropicStack(
        hpfem.Material.dielectric(1.0), [hpfem.Layer(crystal, 300e-9)], glass
    ).response(k0, 0.6, 1.0)
    assert np.allclose(lossless.reflectance + lossless.transmittance, 1.0, atol=1e-12)
    assert abs(lossless.reflection[0, 1]) > 1e-3
    assert hpfem.AnisotropicStack.berreman_matrix(np.eye(3), 0.0, 0.0).shape == (4, 4)


def test_anisotropic_stack_as_background():
    """M19 S4b: the analytic field of a crystal stack, its LayerStack3D geometry, the substrate
    mode kinds and the conical wave are bound (the solver checks are in the C++ suite)."""
    k0 = 2 * np.pi
    crystal = hpfem.Material.anisotropic(uniaxial(1.5, 1.7, [0.0, 0.0, 1.0]))
    stack = hpfem.AnisotropicStack(hpfem.Material.dielectric(1.0), [], crystal, 0.0)
    wave = stack.plane_wave(k0, 0.4, hpfem.Polarisation.S, 1.0, 0.3)
    assert wave.response.substrate_mode_kinds[0] == hpfem.SubstrateMode.ORDINARY
    assert np.asarray(wave.field.value([0.1, 0.2, 0.3])).shape == (3,)
    geometry = stack.layer_stack()
    assert geometry.num_layers == 0
    cw = hpfem.anisotropic_conical_wave(stack, k0, 0.4, 0.3, hpfem.Polarisation.P)
    assert abs(cw.wave.beta - k0 * np.sin(0.4) * np.sin(0.3)) < 1e-12
    assert not cw.background.substrate.is_isotropic


def test_mode_resolved_orders_binding():
    """M19 S5a: half-space modes and the orders resolved into them are bound; an ordinary wave
    below the line comes back as the ordinary channel of order 0."""
    k0 = 2 * np.pi
    eps = uniaxial(1.5, 1.7, [0.3, -0.7, 0.6])  # solver frame
    up = hpfem.half_space_modes(np.eye(3) * 2.25, 0.3, 0.2, True)
    assert up.kinds[0] == hpfem.SubstrateMode.S
    assert abs(up.kz[0] - np.sqrt(2.25 - 0.13)) < 1e-13
    # local frame of the line with normal -y: (x, z, -y)
    r = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])
    kt0, beta, period = 0.3 * k0, 0.25 * k0, 0.8
    modes = hpfem.half_space_modes(r @ eps @ r.T, kt0 / k0, beta / k0, True)
    assert modes.kinds[0] == hpfem.SubstrateMode.ORDINARY
    e0 = r.T @ np.asarray(modes.fields[0])

    def field(x):
        return 0.7 * e0 * np.exp(1j * (kt0 * x[0] - k0 * modes.kz[0] * x[1]))

    line = hpfem.OrderLine(
        np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, -1.0]), period
    )
    orders = hpfem.conical_mode_orders(field, line, k0, eps, kt0, beta, 0.9 * k0, max_order=1)
    zeroth = orders[1]
    assert zeroth.order == 0
    assert abs(zeroth.channels[0].amplitude - 0.7) < 1e-10
    assert abs(zeroth.channels[1].amplitude) < 1e-10
    assert abs(zeroth.efficiency - 0.49 * modes.flux[0] / 0.9) < 1e-10

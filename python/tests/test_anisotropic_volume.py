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

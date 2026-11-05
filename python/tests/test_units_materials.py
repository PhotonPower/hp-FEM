"""hpfem.units conversions and the hpfem.materials library against reference values."""

import numpy as np
import pytest

import hpfem
from hpfem import materials, units


def test_spectral_conversions_round_trip():
    omega = units.angular_frequency(wavelength=500 * units.nm)
    assert np.isclose(units.wavelength(omega) / units.nm, 500)
    assert np.isclose(units.photon_energy(omega) / units.eV, 2.4797, rtol=1e-4)
    assert np.isclose(units.frequency(omega) / units.THz, 599.58, rtol=1e-4)
    assert np.isclose(units.vacuum_wavenumber(omega) * 500 * units.nm, 2 * np.pi)
    assert np.isclose(units.angular_frequency(energy=2.4797 * units.eV), omega, rtol=1e-4)
    assert np.isclose(units.angular_frequency(frequency=599.58 * units.THz), omega, rtol=1e-4)
    assert np.isclose(units.angular_frequency(wavenumber=omega / hpfem.constants.c0), omega)
    lams = np.array([400, 800]) * units.nm
    assert units.angular_frequency(wavelength=lams).shape == (2,)
    with pytest.raises(ValueError):
        units.angular_frequency(wavelength=1e-6, energy=1.0)
    with pytest.raises(ValueError):
        units.angular_frequency()
    assert np.isclose(90 * units.deg, np.pi / 2)


def test_library_values_against_the_sources():
    lam = 589.3 * units.nm  # sodium D
    omega = units.angular_frequency(wavelength=lam)
    assert np.isclose(materials.get("SiO2").refractive_index(omega).real, 1.4584, atol=2e-4)
    assert np.isclose(materials.get("TiO2").refractive_index(omega).real, 2.613, atol=5e-3)
    assert np.isclose(materials.get("water").refractive_index(omega).real, 1.333, atol=1e-3)
    # Johnson & Christy gold around 1.96 eV (632.8 nm): n = 0.18, k = 3.43 (tabulated rows)
    au = materials.get("Au").refractive_index(units.angular_frequency(wavelength=632.8 * units.nm))
    assert abs(au.real - 0.18) < 0.03 and abs(au.imag - 3.43) < 0.1
    si = materials.get("Si").refractive_index(units.angular_frequency(wavelength=500 * units.nm))
    assert abs(si.real - 4.30) < 0.05 and 0.03 < si.imag < 0.06  # alpha ~ 1.1e4 /cm
    for name in ("Au", "Ag", "Al", "Si", "GaAs", "MAPbI3"):
        eps = materials.get(name).eps_r(omega)
        assert eps.imag > 0, name  # lossy: Im eps_r > 0 with exp(-i omega t)
    assert materials.get("Ag").eps_r(omega).real < -10  # noble metal in the visible
    assert (
        materials.get("MAPbI3").eps_r(units.angular_frequency(wavelength=900 * units.nm)).imag < 0.2
    )
    with pytest.raises(ValueError):
        materials.get("Si").eps_r(units.angular_frequency(wavelength=2 * units.um))
    with pytest.raises(ValueError):
        materials.get("TiO2").eps_r(units.angular_frequency(wavelength=300 * units.nm))
    with pytest.raises(KeyError):
        materials.get("unobtainium")
    assert "Johnson" in materials.get("Au").source and materials.get("Au").range[0] < 200 * units.nm


def test_models_and_core_material():
    omega = units.angular_frequency(wavelength=800 * units.nm)
    drude = materials.Drude(1.0, 1.37e16, 1.0e14)
    eps = drude.eps_r(omega)
    assert eps.real < 0 and eps.imag > 0
    dl = materials.DrudeLorentz(1.0, 1.37e16, 1.0e14, [(0.1, 3.0e15, 5.0e14)])
    assert dl.eps_r(omega) != eps
    table = materials.Tabulated([400e-9, 800e-9], [1.5, 1.6], [0.0, 0.1], name="t")
    assert np.isclose(
        table.refractive_index(units.angular_frequency(wavelength=600e-9)), 1.55 + 0.05j
    )
    mat = materials.get("Si").at(omega)
    assert isinstance(mat, hpfem.Material) and mat.eps_r.imag >= 0
    assert np.isclose(materials.get("Si").at_wavelength(800 * units.nm).eps_r, mat.eps_r)
    setup = hpfem.ScatteringSetup2D()
    setup.materials.set(2, materials.get("Au").at(omega))
    assert setup.materials.at(2).eps_r.real < 0
    assert isinstance(materials.plasma_frequency(5.9e28), float)
    with pytest.raises(ValueError):
        materials.Tabulated([800e-9, 400e-9], [1, 1], [0, 0])


def test_pml_profile_for_angle_and_resolution():
    # R0 = (target / (2 r))^(2 / cos theta): the values of the M14-A specification
    profile = hpfem.PmlProfile.for_angle(50.0, 1e-4)
    assert profile.order == 2
    assert abs(profile.reflection / 4.1e-14 - 1) < 0.03
    assert abs(hpfem.PmlProfile.for_angle(70.0, 1e-4, 0.31).reflection / 6.7e-23 - 1) < 0.03
    assert hpfem.PmlProfile.for_angle(70.0, 1e-4).reflection < profile.reflection
    with pytest.raises(ValueError):
        hpfem.PmlProfile.for_angle(90.0, 1e-4)
    # |k s| h at the far end of the layer, and the thickness that keeps it below the limit
    k0 = 2 * np.pi / 405e-9
    h = 37e-9
    box = hpfem.PmlBox2D([0.0, 0.0], [400e-9, 400e-9], [0.0, 1215e-9, 0.0, 0.0], k0, 1.0,
                         hpfem.PmlProfile(2, 1e-16))  # fmt: skip
    assert box.k0 == k0
    sigma = -3 * np.log(1e-16) / (2 * k0 * 1215e-9)
    assert abs(box.max_resolution(h, 1.0) - k0 * np.sqrt(1 + sigma**2) * h) < 1e-12
    assert box.max_resolution(h, 1.0) < hpfem.PmlBox2D.resolution_limit(5)
    assert box.max_resolution(h, 5.44) > hpfem.PmlBox2D.resolution_limit(5)
    thick = hpfem.PmlBox2D.recommended_thickness(k0, 1.0, h, hpfem.PmlProfile(2, 1e-16), 5)
    assert thick >= hpfem.PmlBox2D.recommended_thickness(k0, 1.0, h)
    resolved = hpfem.PmlBox2D([0.0, 0.0], [1e-6, 1e-6], [0.0, thick, 0.0, 0.0], k0, 1.0,
                              hpfem.PmlProfile(2, 1e-16))  # fmt: skip
    assert resolved.max_resolution(h, 1.0) <= hpfem.PmlBox2D.resolution_limit(5) * (1 + 1e-12)


def test_anisotropic_tensor_orientation_and_scalar_refusal():
    # ADR-0015 §6: R diag(eps) R^T with the optic axis at (theta, phi)
    omega = units.angular_frequency(wavelength=600 * units.nm)
    crystal = materials.Uniaxial(2.0, 3.0, theta=np.pi / 2, phi=np.pi / 4)
    axis = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    expected = 2.0 * np.eye(3) + (3.0 - 2.0) * np.outer(axis, axis)
    assert np.allclose(crystal.eps_tensor(omega), expected)
    tilted = materials.Uniaxial(2.0, 3.0, theta=0.4, phi=1.1)
    eps = tilted.eps_tensor(omega)
    assert np.allclose(eps, eps.T)  # symmetric
    direction = materials.axis_rotation(0.4, 1.1) @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(eps @ direction, 3.0 * direction)  # the axis is the e eigenvector
    assert np.allclose(
        direction, [np.sin(0.4) * np.cos(1.1), np.sin(0.4) * np.sin(1.1), np.cos(0.4)]
    )
    # Euler z-x-z: columns are the principal axes; beta = 0 is a turn about z only
    r = materials.euler_rotation(0.3, 0.0, 0.2)
    assert np.allclose(r, materials.axis_rotation(0.0, 0.5))
    biaxial = materials.Anisotropic((2.0, 2.5, 3.0), materials.euler_rotation(0.3, 0.7, -0.2))
    values = np.linalg.eigvals(biaxial.eps_tensor(omega))
    assert np.allclose(np.sort(values.real), [2.0, 2.5, 3.0])
    # the core material is a tensor; the scalar accessors raise
    material = tilted.at(omega)
    assert not material.is_isotropic
    assert np.allclose(material.eps_tensor, eps)
    with pytest.raises(hpfem.AnisotropicUnsupported):
        _ = material.eps_r
    with pytest.raises(NotImplementedError):
        _ = material.refractive_index
    with pytest.raises(hpfem.AnisotropicUnsupported):
        tilted.eps_r(omega)
    assert materials.is_anisotropic(tilted) and materials.is_anisotropic(material)
    assert not materials.is_anisotropic(materials.get("SiO2"))
    # setting the scalar makes it isotropic again; gyrotropic tensors are refused
    material.eps_r = 2.25
    assert material.is_isotropic and material.eps_r == 2.25
    with pytest.raises(ValueError):
        hpfem.Material.anisotropic(np.array([[2, 0.1j, 0], [-0.1j, 2, 0], [0, 0, 2]]))
    with pytest.raises(ValueError):
        materials.Anisotropic((2.0, 2.0, 3.0), np.diag([1.0, 2.0, 1.0]))


def test_sapphire_ordinary_and_extraordinary():
    # Malitson & Dodge 1972 at the sodium D line: n_o = 1.7681, n_e = 1.7600
    sapphire = materials.get("Al2O3")
    omega = units.angular_frequency(wavelength=589.3 * units.nm)
    n = np.sqrt(sapphire.principal_values(omega))
    assert np.allclose(n.real, [1.7681, 1.7681, 1.7600], atol=1e-4)
    assert sapphire.range == pytest.approx((0.2 * units.um, 5.0 * units.um))
    a_cut = sapphire.rotated(materials.axis_rotation(np.pi / 2, 0.0))
    eps = a_cut.eps_tensor(omega)
    assert np.isclose(eps[0, 0], n[2] ** 2) and np.isclose(eps[2, 2], n[0] ** 2)
    # DispersiveMap takes models and tensor materials and freezes them per frequency
    dmap = materials.DispersiveMap("SiO2").set(1, sapphire).set(2, a_cut.at(omega))
    frozen = dmap.at(omega)
    assert not frozen.is_isotropic
    assert np.allclose(frozen.at(2).eps_tensor, eps)
    assert materials.is_anisotropic(dmap)

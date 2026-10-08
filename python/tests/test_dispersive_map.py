"""hpfem.materials additions of M15 F13: the out-of-range policy of tabulated data, the
DispersiveMap with apply(setup, omega), and the Drude-Lorentz fit."""

import warnings

import numpy as np
import pytest

import hpfem
from hpfem import materials, units


def test_out_of_range_policy_error_and_clamp():
    silver = materials.get("Ag")
    lo, hi = silver.range
    omega_out = units.angular_frequency(wavelength=hi * 1.5)
    with pytest.raises(ValueError, match="outside"):
        silver.at(omega_out)
    clamped = materials.with_policy(silver, "clamp")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = clamped.at(omega_out)
    assert caught and "clamped" in str(caught[0].message)
    assert m.eps_r == silver.at(units.angular_frequency(wavelength=hi)).eps_r
    assert silver.out_of_range == "error"  # the original is untouched
    with pytest.raises(ValueError, match="out_of_range"):
        materials.with_policy(silver, "bogus").at(omega_out)
    assert materials.with_policy(materials.Constant(2.0), "clamp").at(omega_out).eps_r == 2.0


def test_dispersive_map_freezes_a_frequency_into_the_setup():
    dmap = (
        materials.DispersiveMap("air")
        .set(2, "Ag")
        .set(3, hpfem.Material.dielectric(1.5))
        .set(4, 2.25)
    )
    assert dmap.tags == [2, 3, 4] and dmap.has(2) and not dmap.has(9)
    omega = units.angular_frequency(wavelength=600 * units.nm)
    frozen = dmap.at(omega)
    assert isinstance(frozen, hpfem.MaterialMap)
    assert frozen.has(2) and frozen.background.eps_r == materials.get("air").at(omega).eps_r
    mesh = hpfem.rectangle(2, 2)
    mesh.set_cell_tag(0, 2)
    assert frozen.of_cell(mesh, 0).eps_r == materials.get("Ag").at(omega).eps_r
    setup = hpfem.ConicalScatteringSetup()
    dmap.apply(setup, omega)
    assert setup.omega == omega and setup.materials.has(4)
    lo, hi = dmap.range
    assert lo == materials.get("Ag").range[0] and hi == materials.get("Ag").range[1]
    assert (
        dmap.at_wavelength(600 * units.nm).of_cell(mesh, 0).eps_r == frozen.of_cell(mesh, 0).eps_r
    )
    with pytest.raises(TypeError):
        dmap.set(5, object())
    assert "DispersiveMap" in repr(dmap)


def test_drude_lorentz_fit_recovers_a_synthetic_model_and_fits_silver():
    rng = (300 * units.nm, 900 * units.nm, 40)
    truth = materials.DrudeLorentz(3.0, 1.2e16, 2.0e14, ((0.5, 4.0e15, 5.0e14),), name="truth")
    model, error = materials.fit_drude_lorentz(truth, rng, oscillators=1)
    assert error < 2e-2
    lam = np.geomspace(*rng)
    omega = units.angular_frequency(wavelength=lam)
    assert np.allclose(model.eps_r(omega), truth.eps_r(omega), rtol=2e-2, atol=1e-2)
    assert (
        model.omega_p > 0
        and model.gamma > 0
        and all(f > 0 and w > 0 and g > 0 for f, w, g in model.oscillators)
    )
    # silver in the visible: Drude + two Lorentz poles reproduce n + ik to a few per cent
    silver = materials.get("Ag")
    fit, error = materials.fit_drude_lorentz(
        silver, (350 * units.nm, 800 * units.nm, 60), oscillators=2
    )
    assert error < 0.15
    omega = units.angular_frequency(wavelength=np.linspace(400, 700, 7) * units.nm)
    n_fit = fit.refractive_index(omega)
    n_tab = silver.refractive_index(omega)
    assert np.max(np.abs(n_fit - n_tab) / np.abs(n_tab)) < 0.15
    assert np.all(fit.eps_r(omega).imag > 0)  # passive
    assert fit.name.startswith("Ag")
    # a fit is a Dispersive: usable in a DispersiveMap
    frozen = materials.DispersiveMap().set(2, fit).at(omega[0])
    assert frozen.has(2)

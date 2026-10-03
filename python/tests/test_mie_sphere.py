"""Mie series of the sphere from Python: efficiencies, fields and the optional cross-check
against the independent `miepython` package (skipped when it is not installed)."""

import numpy as np
import pytest

import hpfem


def test_mie_sphere_series_and_fields():
    k = 2 * np.pi / 500e-9
    a = 100e-9
    s = hpfem.mie_sphere(k, a, -10 + 1j)
    assert s.size_parameter == pytest.approx(k * a)
    assert s.max_order == len(s.a) == len(s.b) == len(s.c) == len(s.d)
    assert s.absorption_efficiency() > 0
    assert s.extinction_efficiency() == pytest.approx(
        s.scattering_efficiency() + s.absorption_efficiency()
    )
    assert s.scattering_cross_section() == pytest.approx(s.scattering_efficiency() * np.pi * a * a)
    # fields: incident wave, interface continuity of the tangential field
    x = np.array([30e-9, -70e-9, 120e-9])
    assert np.allclose(s.incident_field(x), [np.exp(1j * k * x[2]), 0, 0])
    n = np.array([0.3, 0.5, 0.81])
    n /= np.linalg.norm(n)
    e_out = s.incident_field(a * n * (1 + 1e-9)) + s.scattered_field(a * n * (1 + 1e-9))
    e_in = s.internal_field(a * n * (1 - 1e-9))
    tangential = lambda e: e - np.dot(n, e) * n  # noqa: E731
    assert np.linalg.norm(tangential(e_out) - tangential(e_in)) < 1e-6 * np.linalg.norm(e_out)
    assert np.allclose(s.total_field(0.5 * a * n), s.internal_field(0.5 * a * n))
    # lossless sphere in glass
    g = hpfem.mie_sphere(1.5 * k, a, 1.5**2 * 4.0, background_index=1.5)
    assert abs(g.absorption_efficiency()) < 1e-12
    with pytest.raises(ValueError):
        hpfem.mie_sphere(0.0, a, 4.0)


def test_mie_sphere_against_miepython():
    miepython = pytest.importorskip("miepython")
    k = 2 * np.pi / 600e-9
    a = 150e-9
    for eps in [4.0, -10 + 1j, 2.25 + 0.5j]:
        s = hpfem.mie_sphere(k, a, eps)
        # miepython uses the exp(+iwt) convention: the refractive index has a negative
        # imaginary part there
        m = np.conj(np.sqrt(complex(eps)))
        qext, qsca, _qback, _g = miepython.efficiencies_mx(m, k * a)  # miepython >= 3
        assert s.scattering_efficiency() == pytest.approx(qsca, rel=1e-8)
        assert s.extinction_efficiency() == pytest.approx(qext, rel=1e-8)
        # near field: with the conjugated index miepython's Cartesian field agrees directly
        # (verified against hpfem at these points to 1e-6)
        for pt in [(0.3, 0.3, 0.3), (1.5, 0.1, 0.0), (0.0, 0.0, 1.5)]:
            x = np.array(pt) * a
            e_ref = np.asarray(
                miepython.e_near_cartesian(2 * np.pi / k, 2 * a, m, 1.0, *x), dtype=complex
            ).ravel()
            assert np.allclose(s.total_field(x), e_ref, rtol=1e-6, atol=1e-8)

"""Data ports for BindOptiX (issue #214): emitter response tables of planar stacks and the
``raytatouille-result`` writer (``hpfem.dataports``)."""

import json
import math

import numpy as np
import pytest
from scipy.integrate import quad

import hpfem
from hpfem import dataports as dp

LAM = 0.67e-6
K0 = 2 * math.pi / LAM
N_W, N_G = 1.33, 1.52
WATER = hpfem.Material.dielectric(N_W)
GLASS = hpfem.Material.dielectric(N_G)
EPS_AU = complex(-14.0, 1.1)  # gold near 670 nm


def _stack(layers=(), substrate=GLASS):
    return hpfem.LayerStack2D(WATER, [hpfem.Layer(m, d) for m, d in layers], substrate, 0.0)


def _decode(text, type_name):
    """The reader of bindoptix.results: envelope, version 0.1.x, type."""
    obj = json.loads(text)
    assert obj["format"] == "raytatouille-result"
    assert str(obj["schema_version"]).startswith("0.1.")
    assert obj["type"] == type_name
    return obj["data"]


def test_no_interface_gives_the_free_dipole():
    """Issue #214: without a layer A_tot = A_rad = I; each half-space takes half the power."""
    r = dp.planar_emitter_response(_stack(substrate=WATER), LAM, [5e-9, 0.2e-6])
    t = r.table
    assert t.a_tot.shape == (1, 1, 2, 3, 3)
    assert np.allclose(t.a_tot, np.eye(3), atol=1e-9)
    assert np.allclose(t.a_rad, np.eye(3), atol=1e-9)
    assert t.detectors == ("solution", "substrate")
    assert np.allclose(t.a_det[0, 0, :, 0], 0.5 * np.eye(3), atol=1e-9)
    assert np.allclose(t.a_det[0, 0, :, 1], 0.5 * np.eye(3), atol=1e-9)
    assert r.convergence["radiated_check"] < 1e-8


def test_perfect_mirror_against_the_image_dipole():
    """Issue #214: a dipole in front of a perfect mirror, x = 2 k d: F_⊥ = 1 + 3(sin x/x³ −
    cos x/x²), F_∥ = 1 − (3/2)(sin x/x + cos x/x² − sin x/x³), all power radiated up. A metal of
    ε = −1e8 approximates the mirror to O(1/|n|) = 1e-4 (the deviation falls like 1/|n|:
    6e-2 at ε = −1e4, 6e-3 at −1e6)."""
    mirror = hpfem.Material(complex(-1e8, 1.0), 1.0)
    heights = [5e-9, 50e-9, 150e-9, 400e-9]
    r = dp.planar_emitter_response(_stack(substrate=mirror), LAM, heights)
    assert "substrate" not in r.far_field  # a lossy substrate radiates nothing into it
    for i, h in enumerate(heights):
        x = 2 * K0 * N_W * h
        perp = 1 + 3 * (math.sin(x) / x**3 - math.cos(x) / x**2)
        par = 1 - 1.5 * (math.sin(x) / x + math.cos(x) / x**2 - math.sin(x) / x**3)
        expected = np.diag([par, par, perp])
        assert np.abs(r.table.a_tot[0, 0, i] - expected).max() < 2e-3, h
        assert np.abs(r.table.a_rad[0, 0, i] - expected).max() < 2e-3, h
    assert r.table.a_tot[0, 0, 0, 2, 2] == pytest.approx(2.0, abs=0.01)  # limits 2 and 0
    assert r.table.a_tot[0, 0, 0, 0, 0] < 0.01


def test_water_on_glass_conserves_energy_and_matches_cps():
    """A lossless interface without guided modes: A_tot = A_rad, the power into the water equals
    the Chance–Prock–Silbey upward fraction (independent closed integral), the rest goes into the
    glass, part of it beyond the critical angle (the supercritical light)."""
    heights = [2e-9, 20e-9, 200e-9]
    critical = math.asin(N_W / N_G)
    detectors = [
        dp.Detector("water", "cover"),
        dp.Detector("glass_sub", "substrate", 0.0, critical),
        dp.Detector("glass_super", "substrate", critical, 0.5 * math.pi),
    ]
    r = dp.planar_emitter_response(_stack(), LAM, heights, detectors=detectors)
    t = r.table
    assert np.allclose(t.a_rad, t.a_tot, rtol=1e-9, atol=0)
    assert abs(r.convergence["energy"]) < 1e-9

    def fresnel(s):
        kz1 = math.sqrt(1 - s * s)
        kz2 = math.sqrt((N_G / N_W) ** 2 - s * s)
        e1, e2 = N_W**2, N_G**2
        return (kz1 - kz2) / (kz1 + kz2), (e2 * kz1 - e1 * kz2) / (e2 * kz1 + e1 * kz2)

    for i, h in enumerate(heights):
        k1h = K0 * N_W * h

        def up(tt, which, k1h=k1h):
            s, sz = math.sin(tt), math.cos(tt)
            rs, rp = fresnel(s)
            e = complex(math.cos(2 * k1h * sz), math.sin(2 * k1h * sz))
            if which == 0:
                return 0.75 * s**3 * abs(1 + rp * e) ** 2
            return 0.375 * s * (abs(1 + rs * e) ** 2 + sz**2 * abs(1 - rp * e) ** 2)

        water = t.a_det[0, 0, i, 0]
        assert water[2, 2] == pytest.approx(quad(up, 0, math.pi / 2, args=(0,))[0], rel=1e-9)
        assert water[0, 0] == pytest.approx(quad(up, 0, math.pi / 2, args=(1,))[0], rel=1e-9)
        assert np.allclose(t.a_det[0, 0, i].sum(axis=0), t.a_rad[0, 0, i], rtol=1e-10, atol=1e-15)
    # close to the interface most of the glass light is supercritical; far away none
    super_near = t.a_det[0, 0, 0, 2, 2, 2] / t.a_rad[0, 0, 0, 2, 2]
    super_far = t.a_det[0, 0, 2, 2, 0, 0] / t.a_rad[0, 0, 2, 0, 0]
    assert super_near > 0.3
    assert super_far < super_near


def test_gold_film_quenching_and_plasmon_coupled_emission():
    """Glass / 50 nm Au / water: absorption (A_tot − A_rad positive definite, growing towards the
    film), and the normal dipole radiates into the glass at the surface-plasmon resonance angle,
    n_g sin θ ≈ Re sqrt(ε_Au ε_w / (ε_Au + ε_w)) (the leaky plasmon of the water side)."""
    gold = hpfem.Material(EPS_AU, 1.0)
    heights = [2e-9, 10e-9, 50e-9]
    r = dp.planar_emitter_response(_stack([(gold, 50e-9)]), LAM, heights)
    t = r.table
    absorbed = t.a_tot - t.a_rad
    assert np.all(np.linalg.eigvalsh(absorbed[0, 0]) > 0)
    yield_z = t.a_rad[0, 0, :, 2, 2] / t.a_tot[0, 0, :, 2, 2]
    assert yield_z[0] < 0.01 and yield_z[0] < yield_z[1] < yield_z[2]
    ff = r.far_field["substrate"]
    power = np.sum(np.abs(ff.field[2, :, :, 2, :]) ** 2, axis=(1, 2))  # z dipole, 50 nm
    theta_peak = ff.theta[np.argmax(power)]
    eps_w = N_W**2
    spr = math.asin((np.sqrt(EPS_AU * eps_w / (EPS_AU + eps_w))).real / N_G)
    assert abs(theta_peak - spr) < math.radians(1.0), (math.degrees(theta_peak), math.degrees(spr))
    assert theta_peak > math.asin(N_W / N_G)  # supercritical


def test_jones_detectors_split_the_power():
    """Detectors with Jones matrices: an s and a p analyser on the same directions add up to the
    region without analyser."""
    region = (0.6, 1.2)
    calls = {}
    for name, jones in (("all", None), ("s", np.diag([1.0, 0.0])), ("p", np.diag([0.0, 1.0]))):
        det = dp.Detector(name, "substrate", *region,
                          None if jones is None else (lambda t, p, j=jones: j))  # fmt: skip
        r = dp.planar_emitter_response(_stack(), LAM, [30e-9], detectors=[det])
        calls[name] = r.table.a_det[0, 0, 0, 0]
    assert np.allclose(calls["s"] + calls["p"], calls["all"], rtol=1e-12, atol=1e-15)
    assert calls["s"][0, 0] > 0 and calls["p"][2, 2] > 0 and calls["s"][2, 2] == pytest.approx(0)


def test_refusals():
    with pytest.raises(ValueError, match="overlap"):
        dp.planar_emitter_response(
            _stack(),
            LAM,
            [30e-9],
            detectors=[
                dp.Detector("a", "substrate", 0.0, 1.0),
                dp.Detector("b", "substrate", 0.5, 1.2),
            ],
        )
    with pytest.raises(ValueError, match="at least one"):
        dp.planar_emitter_response(_stack(), LAM, [30e-9], detectors=[])
    with pytest.raises(ValueError, match="positive"):
        dp.planar_emitter_response(_stack(), LAM, [0.0, 30e-9])


def test_emitter_response_table_envelope_round_trip():
    """The writer: the envelope of bindoptix.results, the keys of bindoptix.ports, arrays in C
    order; the table passes the checks BindOptiX applies on import."""
    r = dp.planar_emitter_response(_stack([(hpfem.Material(EPS_AU, 1.0), 50e-9)]), LAM,
                                   [5e-9, 20e-9], source={"commit": "abc123"})  # fmt: skip
    data = _decode(r.table.to_json(), "EmitterResponseTable")
    assert set(data) == {"grid", "wavelength_um", "n_host", "detectors", "a_tot", "a_rad",
                         "a_det", "source"}  # fmt: skip
    assert data["wavelength_um"] == pytest.approx(0.67)
    assert data["n_host"] == pytest.approx(N_W)
    axes = [dp.decode_array(a) for a in data["grid"]["axes_um"]]
    assert np.allclose(axes[2], [0.005, 0.02]) and axes[0].shape == (1,)
    a_tot = dp.decode_array(data["a_tot"])
    assert a_tot.shape == (1, 1, 2, 3, 3) and np.array_equal(a_tot, r.table.a_tot)
    assert dp.decode_array(data["a_det"]).shape == (1, 1, 2, 2, 3, 3)
    src = data["source"]
    assert src["commit"] == "abc123" and src["solver"] == "hp-FEM"
    assert len(src["content_hash"]) == 64
    assert json.loads(src["convergence"])["psd_projection"] < 1e-9
    # the import checks of BindOptiX, here in the constructor
    rebuilt = dp.EmitterResponseTable(r.table.grid, 0.67, N_W, r.table.detectors, a_tot,
                                      dp.decode_array(data["a_rad"]),
                                      dp.decode_array(data["a_det"]))  # fmt: skip
    assert rebuilt.detectors == r.table.detectors
    with pytest.raises(ValueError, match="positive semidefinite"):
        dp.EmitterResponseTable(r.table.grid, 0.67, N_W, r.table.detectors, 0.5 * r.table.a_rad,
                                r.table.a_rad, r.table.a_det)  # fmt: skip


def test_field_map_envelope():
    grid = dp.RegularGrid(([0.0, 0.1], [0.0], [0.01, 0.02, 0.03]))
    e = np.arange(18, dtype=float).reshape(2, 1, 3, 3) * (1 + 1j)
    fm = dp.FieldMap(grid, 0.633, e, q_w_m3=np.ones((2, 1, 3)))
    data = _decode(fm.to_json(), "FieldMap")
    assert np.array_equal(dp.decode_array(data["e"]), e)
    assert data["delta_t_k"] is None and data["nonlinear"] is False
    assert dp.encode_array(np.array([np.nan, np.inf]))["values"] == ["NaN", "Infinity"]
    with pytest.raises(ValueError, match="nonlinear"):
        dp.FieldMap(grid, 0.633, e, nonlinear=True)

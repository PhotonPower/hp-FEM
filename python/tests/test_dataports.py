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
    assert np.allclose(t.a_rad, t.a_tot, rtol=1e-9, atol=1e-15)
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
        assert np.allclose(t.a_det[0, 0, i].sum(axis=0), t.a_rad[0, 0, i], rtol=1e-12, atol=1e-15)
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
    assert json.loads(src["convergence"])["energy"] > 0  # absorption in the gold
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


# --- port F: field maps -------------------------------------------------------------------------


def _source_json(fm, key):
    return json.loads(_decode(fm.to_json(), "FieldMap")["source"][key])


def test_planar_field_map_at_normal_incidence_from_both_sides():
    """Water on glass at normal incidence: from the solution E_x = e^{−ik1z} + r e^{ik1z}, r =
    (n_w − n_g)/(n_w + n_g); from the glass E_x = t e^{ik1z}, t = 2 n_g/(n_g + n_w) (the incident
    wave has E = polarisation at the origin, continued into the glass)."""
    grid = dp.RegularGrid(([0.0], [0.0], np.linspace(0.0, 0.5, 11)))
    z = grid.axes_um[2] * 1e-6
    k1 = K0 * N_W
    above = dp.planar_field_map(_stack(), LAM, grid, [0, 0, -1], [1, 0, 0])
    r = (N_W - N_G) / (N_W + N_G)
    assert (
        np.abs(above.e[0, 0, :, 0] - (np.exp(-1j * k1 * z) + r * np.exp(1j * k1 * z))).max() < 1e-10
    )
    assert np.abs(above.e[..., 1:]).max() < 1e-12
    below = dp.planar_field_map(_stack(), LAM, grid, [0, 0, 1], [0, 1, 0])
    t = 2 * N_G / (N_G + N_W)
    assert np.abs(below.e[0, 0, :, 1] - t * np.exp(1j * k1 * z)).max() < 1e-10
    assert above.irradiance_w_m2 == pytest.approx(N_W / (2 * hpfem.constants.Z0))
    assert below.irradiance_w_m2 == pytest.approx(N_G / (2 * hpfem.constants.Z0))
    assert above.media == ("solution", "substrate") and np.all(above.medium == 0)
    assert _source_json(below, "k_hat") == [0.0, 0.0, 1.0]
    assert "phase_reference" in _decode(below.to_json(), "FieldMap")["source"]


def test_kretschmann_surface_field_dp1():
    """DP1 (BindOptiX): p light through the glass onto 50 nm gold under water. The field at the
    gold surface (z = 0, solution side) equals the transmitted wave of the Airy recursion
    (|E|² = |t_p|² (|k_x|² + |k_z|²) / n_w² of the evanescent p wave) and peaks near the
    surface-plasmon resonance angle with a strong enhancement."""
    gold = hpfem.Material(EPS_AU, 1.0)
    stack = _stack([(gold, 50e-9)])
    grid = dp.RegularGrid(([0.0], [0.0], [0.0, 0.02]))
    angles = np.radians(np.linspace(60.0, 80.0, 41))
    surface = []
    for t in angles:
        k = np.array([math.sin(t), 0.0, math.cos(t)])
        fm = dp.planar_field_map(stack, LAM, grid, k, [-math.cos(t), 0.0, math.sin(t)])
        assert np.all(fm.medium == 0)
        s = N_G * math.sin(t)  # in-plane index
        _rs, _rp, _ts, tp = dp._fresnel(s / N_G, N_G, [(np.sqrt(EPS_AU), 50e-9)], N_W, K0)
        kz = np.sqrt(complex(N_W**2 - s * s))
        expected = abs(tp) ** 2 * (s * s + abs(kz) ** 2) / N_W**2
        assert np.sum(np.abs(fm.e[0, 0, 0]) ** 2) == pytest.approx(expected, rel=1e-9)
        surface.append(np.sum(np.abs(fm.e[0, 0, 0]) ** 2))
    peak = angles[int(np.argmax(surface))]
    spr = math.asin((np.sqrt(EPS_AU * N_W**2 / (EPS_AU + N_W**2))).real / N_G)
    assert abs(peak - spr) < math.radians(1.5), (math.degrees(peak), math.degrees(spr))
    assert max(surface) > 10.0


def test_planar_absorption_density_integrates_to_the_absorptance_dp2():
    """DP2: ∫ q dz over an absorbing layer equals the absorptance of the stack times the incident
    flux I₀ cos θ (the stack's own power balance, LayerStack3D.plane_wave)."""
    layer = hpfem.Material(complex(4.0, 0.6), 1.0)
    d = 80e-9
    stack = _stack([(layer, d)])
    theta = 0.4
    z = np.linspace(-d, 0.0, 801)
    grid = dp.RegularGrid(([0.0], [0.0], z * 1e6))
    k = [math.sin(theta), 0.0, -math.cos(theta)]
    fm = dp.planar_field_map(stack, LAM, grid, k, [0.0, 1.0, 0.0])
    q = fm.q_w_m3[0, 0]
    q[-1] = dp.planar_field_map(
        stack, LAM, dp.RegularGrid(([0.0], [0.0], [-1e-9])), k, [0.0, 1.0, 0.0]
    ).q_w_m3.ravel()[0]  # the layer side of z = 0
    absorbed = float(np.sum(0.5 * (q[1:] + q[:-1]) * np.diff(z)))
    reference = hpfem.LayerStack3D(WATER, [hpfem.Layer(layer, d)], GLASS, 0.0).plane_wave(
        K0, theta, hpfem.Polarisation.S).absorptance  # fmt: skip
    assert absorbed / (fm.irradiance_w_m2 * math.cos(theta)) == pytest.approx(reference, rel=1e-5)
    assert np.all(fm.medium[0, 0, 1:-1] == 1) and fm.medium[0, 0, -1] == 0


def _grating_result():
    from hpfem import grating, units

    sub, ridge, air = 5, 6, 7
    period, height, h = 0.4e-6, 0.1e-6, 0.05e-6
    ny = int(round(2.4e-6 / h))
    mesh = hpfem.rectangle(int(round(period / h)), ny, [0.0, -1.2e-6],
                           [period, -1.2e-6 + ny * h])  # fmt: skip
    for c in range(mesh.num_cells):
        x = np.mean([mesh.vertex(int(v)) for v in mesh.cell_vertices(c)], axis=0)
        tag = sub if x[1] < 0 else (ridge if x[1] < height and x[0] < period / 2 else air)
        mesh.set_cell_tag(c, tag)
    materials = {sub: GLASS, ridge: hpfem.Material(complex(4.0, 0.8), 1.0), air: WATER}
    result = grating.solve(mesh, materials, _stack(), "p", 0.3, 0.5,
                           units.angular_frequency(wavelength=LAM), order=3,
                           pml={"top": 0.4e-6, "bottom": 0.4e-6})  # fmt: skip
    return result, period, height, ridge


def test_grating_field_map_frames_periodicity_and_absorption():
    """The map of a solved cell: k_hat and polarisation of the incident wave in the sensor frame,
    the Bloch phase across the lines, the phase e^{iβy} along them, the medium by tag, and DP2:
    the midpoint sum of q over the lossy ridge equals the solver's absorbed fraction times the
    incident flux per period."""
    result, period, height, ridge = _grating_result()
    n = 30
    xs = (np.arange(n) + 0.5) * (period / 2) / n * 1e6
    zs = (np.arange(n) + 0.5) * height / n * 1e6
    fm = dp.grating_field_map(result, dp.RegularGrid((xs, [0.0], zs)), media={ridge: "ridge"})
    assert fm.media == ("ridge",)
    assert np.allclose(fm.k_hat, [math.sin(0.3) * math.cos(0.5), math.sin(0.3) * math.sin(0.5),
                                  -math.cos(0.3)])  # fmt: skip
    assert abs(np.dot(fm.k_hat, fm.polarisation)) < 1e-12
    absorbed = fm.q_w_m3.sum() * (period / 2 / n) * (height / n)
    incident = fm.irradiance_w_m2 * math.cos(0.3) * period
    assert absorbed / incident == pytest.approx(result.A_by_tag[ridge], rel=2e-3)
    grid = dp.RegularGrid(([0.05, 0.05 + period * 1e6], [0.0, 0.3], [0.0, 0.2]))
    f2 = dp.grating_field_map(result, grid)
    bloch = np.exp(1j * result.wave.kx * period)
    assert np.abs(f2.e[1, 0] - f2.e[0, 0] * bloch).max() < 1e-12 * np.abs(f2.e).max()
    along = np.exp(1j * result.wave.beta * 0.3e-6)
    assert np.abs(f2.e[0, 1] - f2.e[0, 0] * along).max() < 1e-12 * np.abs(f2.e).max()
    with pytest.raises(ValueError, match="PML"):
        dp.grating_field_map(result, dp.RegularGrid(([0.1], [0.0], [1.0])))
    data = _decode(f2.to_json(), "FieldMap")
    assert json.loads(data["source"]["media"]) == list(f2.media)
    assert dp.decode_array(json.loads(data["source"]["medium"])).shape == grid.shape

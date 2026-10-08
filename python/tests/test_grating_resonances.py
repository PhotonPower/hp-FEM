"""Resonances and bands of the unit cell (M15 F14): the Fabry-Perot slab through
hpfem.grating.resonances against the exact complex wavenumber, the mode fields, the band sweep,
and the job runner's eigen tasks."""

import numpy as np
import pytest

import hpfem
from hpfem import grating, units
from hpfem import run as runner

NM = units.nm
INDEX = 3.5
THICKNESS = 300 * NM
PERIOD = 100 * NM
MARGIN = 150 * NM
PML = 900 * NM
MODE = 4
HALF = THICKNESS / 2 + MARGIN + PML


def exact_wavenumber():
    return (MODE * np.pi - 1j * np.log((INDEX + 1) / (INDEX - 1))) / (INDEX * THICKNESS)


def slab_cell():
    """One period of a free-standing slab |y| < d/2, PML above and below, 4 cells per 300 nm."""
    half = THICKNESS / 2 + MARGIN + PML
    cell = THICKNESS / 4
    ny = round(2 * half / cell)
    mesh = hpfem.rectangle(1, ny, [0.0, -half], [PERIOD, half])
    for c in range(mesh.num_cells):
        if abs(mesh.cell_centroid(c)[1]) < THICKNESS / 2:
            mesh.set_cell_tag(c, 2)
    return mesh


def test_slab_resonance_matches_the_exact_complex_wavenumber():
    k = exact_wavenumber()
    mesh = slab_cell()
    stack = hpfem.LayerStack2D(hpfem.Material.vacuum(), [], hpfem.Material.vacuum())
    events = []
    result = grating.resonances(
        mesh,
        {2: hpfem.Material.dielectric(INDEX)},
        stack,
        0.97 * k.real * hpfem.constants.c0,
        num_modes=4,
        order=3,
        pml={"top": PML, "bottom": PML},
        krylov_dimension=40,
        progress=lambda e: events.append(e.phase),
    )
    assert events[:2] == ["assembly", "constraints"] and events[-1] == "done"
    assert result.period == pytest.approx(PERIOD) and result.dofs > 0
    assert set(result.timing) >= {"assembly", "eigensolve", "total"}
    ks = result.omegas / hpfem.constants.c0
    best = min(ks, key=lambda z: abs(z - k))
    assert abs(best - k) / abs(k) < 2e-3
    mode = min(result.modes, key=lambda m: abs(m.omega / hpfem.constants.c0 - k))
    assert mode.Q == pytest.approx(k.real / (-2 * k.imag), rel=0.05)
    assert mode.omega.imag < 0 and mode.residual < 1e-8
    assert mode.wavelength == pytest.approx(2 * np.pi * hpfem.constants.c0 / mode.omega.real)
    # the field: finite inside, NaN outside the mesh, Bloch-wrapped along x
    pts = np.array([[PERIOD / 2, 0.0], [PERIOD / 2 + PERIOD, 50 * NM], [PERIOD / 2, 10.0]])
    e = mode.field(pts)
    assert e.shape == (3, 3) and np.isfinite(e[0]).all() and np.isfinite(e[1]).all()
    assert np.isnan(e[2]).all()
    assert np.allclose(e[1], mode.field(np.array([[PERIOD / 2, 50 * NM]]))[0])  # kx = 0
    h = mode.field(pts[:1], quantity="H")
    assert np.isfinite(h).all() and np.linalg.norm(h) > 0
    assert mode.raw.beta == 0.0
    # cancellation
    with pytest.raises(hpfem.Cancelled):
        grating.resonances(
            mesh, {2: hpfem.Material.dielectric(INDEX)}, stack, 0.97 * k.real * hpfem.constants.c0,
            order=2, pml={"top": PML, "bottom": PML}, cancel=lambda: True,
        )  # fmt: skip


def test_bands_sweep_and_the_job_runner_tasks(tmp_path):
    k = exact_wavenumber()
    mesh = slab_cell()
    stack = hpfem.LayerStack2D(hpfem.Material.vacuum(), [], hpfem.Material.vacuum())
    g = 2 * np.pi / PERIOD
    results = grating.bands(
        mesh, {2: hpfem.Material.dielectric(INDEX)}, stack, 0.97 * k.real * hpfem.constants.c0,
        [0.0, 0.1 * g], num_modes=2, order=2, pml={"top": PML, "bottom": PML},
    )  # fmt: skip
    assert len(results) == 2 and results[1].kx == pytest.approx(0.1 * g)
    assert all(len(r.modes) == 2 for r in results)
    # the job runner: resonances task with a map per mode, then a bands task
    job = {
        "version": 1,
        "name": "slab",
        "task": "resonances",
        "unit": 1e-9,
        "model": {
            "period": PERIOD / NM,
            "y_bottom": -HALF / NM,
            "y_top": HALF / NM,
            "background_tag": 1,
            "slabs": [{"tag": 2, "y0": -THICKNESS / 2 / NM, "y1": THICKNESS / 2 / NM}],
        },
        "mesh": {
            "structured": {
                "nx": 1,
                "rows": [
                    [-HALF / NM, -THICKNESS / 2 / NM, 14],
                    [-THICKNESS / 2 / NM, THICKNESS / 2 / NM, 4],
                    [THICKNESS / 2 / NM, HALF / NM, 14],
                ],
            }
        },
        "materials": {"2": {"n": [INDEX, 0.0]}},
        "stack": {"incidence": {"eps": 1.0}, "substrate": {"eps": 1.0}},
        "resonance": {
            "wavelength": 2 * np.pi / (0.97 * k.real) / NM,
            "num_modes": 2,
            "krylov_dimension": 40,
        },
        "solver": {"order": 2, "pml": {"top": PML / NM, "bottom": PML / NM}},
        "maps": [{"quantity": "E", "x": [0, PERIOD / NM, 3], "y": [-200, 200, 5], "modes": [0]}],
    }
    events = []
    out = runner.run_job(job, tmp_path, events.append)
    kinds = [e["event"] for e in events]
    assert out["task"] == "resonances" and len(out["points"]) == 1
    assert kinds.count("mode") == 2 and "progress" in kinds and kinds[-1] == "done"
    assert len(out["maps"]) == 1 and out["maps"][0]["mode"] == 0
    data = np.load(out["maps"][0]["file"])
    assert data["values"].shape == (5, 3, 3) and data["omega"].shape == (1,)
    modes = out["points"][0]["modes"]
    assert all(m["omega"][1] < 0 for m in modes)
    job_bands = dict(job, task="bands", sweep={"kx_over_g": [0.0, 0.25]}, maps=[])
    events = []
    out = runner.run_job(job_bands, None, events.append)
    assert out["task"] == "bands" and [p["kx_over_g"] for p in out["points"]] == [0.0, 0.25]
    assert [e["event"] for e in events].count("point") == 2
    with pytest.raises(runner.JobError, match="task"):
        runner.run_job(dict(job, task="modes"))

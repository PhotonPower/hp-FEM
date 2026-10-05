"""Project files and the command line: the example projects reproduce the C++ examples."""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from test_waveguide import slab_te_even

import hpfem
from hpfem import cli, project, units

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def test_mie_project_reproduces_the_series(tmp_path):
    spec = project.load(EXAMPLES / "mie_cylinder" / "project.json")
    spec["outputs"]["vtk"]["file"] = str(tmp_path / "mie.vtu")
    project.validate(spec)
    results = project.run(spec)
    assert results["problem"] == "scattering" and len(results["results"]) == 1
    entry = results["results"][0]
    exact = hpfem.mie_cylinder_scattering_width(6.0, 0.25, 1.5)
    assert abs(entry["cross_sections"]["scattering"] - exact) / exact < 1e-3
    assert abs(entry["far_field"]["scattering_cross_section"] - exact) / exact < 3e-2
    assert len(entry["far_field"]["pattern_abs"]) == 36
    assert len(entry["points"]) == 2 and len(entry["points"][0]) == 2
    assert (tmp_path / "mie.vtu").exists()
    json.dumps(results)  # serialisable


def test_grating_project_conserves_energy():
    spec = project.load(EXAMPLES / "lamellar_grating" / "project.json")
    results = project.run(spec)
    diffraction = results["results"][0]["diffraction"]
    total = sum(o["efficiency"] for o in diffraction["above"]) + sum(
        o["efficiency"] for o in diffraction["below"]
    )
    assert abs(total - 1.0) < 2e-2
    assert [o["order"] for o in diffraction["above"]] == [-1, 0, 1]


def test_grating_project_orders_on_a_line_and_power_balance():
    # the same grating: the orders on a line of explicit orientation with the incident wave
    # subtracted agree with the scattered-field orders above, and the flux balance closes
    spec = project.load(EXAMPLES / "lamellar_grating" / "project.json")
    spec["outputs"]["diffraction"]["line"] = {
        "origin": [1.0, 0.0],
        "tangent": [0.0, 1.0],
        "normal": [1.0, 0.0],
    }
    spec["outputs"]["diffraction"]["balance"] = {"axis": 0, "reflection": 1.0, "transmission": -0.5}
    diffraction = project.run(spec)["results"][0]["diffraction"]
    line = diffraction["line"]
    assert [o["order"] for o in line] == [-1, 0, 1]
    for on_line, above in zip(line, diffraction["above"], strict=True):
        assert abs(on_line["efficiency"] - above["efficiency"]) < 2e-3
        assert len(on_line["amplitude"]) == 2
    balance = diffraction["balance"]
    assert balance["absorbed"] == 0.0
    assert abs(balance["relative_residual"]) < 2e-2
    reflected = sum(o["efficiency"] for o in line)
    assert abs(balance["reflected"] / balance["incident"] - reflected) < 2e-3


def test_waveguide_and_cavity_projects():
    modes = project.run(project.load(EXAMPLES / "slab_waveguide" / "project.json"))
    n_eff = modes["results"][0]["effective_index"]
    assert abs(n_eff[0] - slab_te_even()) < 1e-6  # k0 d = 2, n = 1.5 / 1.0 as in test_waveguide
    cavity = project.run(project.load(EXAMPLES / "cavity_modes" / "project.json"))
    k2 = np.array(cavity["k0_squared"]) * units.um**2  # mesh of 1 um: pi^2 (m^2 + n^2) / um^2
    assert np.allclose(k2, np.pi**2 * np.array([1, 1, 2, 4, 4]), rtol=2e-3)


def test_sweep_dispersive_material_and_errors(tmp_path):
    spec = {
        "problem": "scattering",
        "dim": 2,
        "length_unit": "nm",
        "mesh": {"type": "square_with_disc", "n": 2, "radius": 50, "half_width": 200, "outer": 400},
        "order": 2,
        "wavelength": {"value": [500, 600], "unit": "nm"},
        "materials": {"background": "vacuum", "2": "Au"},
        "source": {"type": "plane_wave", "angle": 90, "polarisation": "TE"},
        "pml": {"lower": [-200, -200], "upper": [200, 200], "thickness": 200},
        "boundaries": {"pec": ["x_min", "x_max", "y_min", "y_max"]},
        "outputs": {"cross_sections": {"around_tag": 2}, "estimate": True},
    }
    results = project.run(spec)
    assert len(results["results"]) == 2
    for entry in results["results"]:
        assert entry["cross_sections"]["absorption"] > 0  # gold absorbs
        assert entry["estimate"]["total"] > 0
    with pytest.raises(project.ProjectError):
        project.validate({**spec, "mesh": {"type": "cube"}})
    with pytest.raises(project.ProjectError):
        project.validate({**spec, "materials": {"2": "unobtainium"}})
    with pytest.raises(project.ProjectError):
        project.validate({**spec, "boundaries": {"pec": ["north"]}})
    with pytest.raises(project.ProjectError):
        project.validate({**spec, "frequency": 1.0})


def test_cli(tmp_path, capsys):
    assert cli.main(["info"]) == 0
    assert "backends" in capsys.readouterr().out
    assert cli.main(["materials"]) == 0
    assert "Johnson" in capsys.readouterr().out
    path = EXAMPLES / "cavity_modes" / "project.json"
    assert cli.main(["validate", str(path)]) == 0
    out = tmp_path / "results.json"
    assert cli.main(["run", str(path), "-o", str(out), "-q"]) == 0
    assert json.loads(out.read_text())["problem"] == "cavity"
    bad = tmp_path / "bad.json"
    bad.write_text('{"problem": "nonsense"}')
    assert cli.main(["validate", str(bad)]) == 1
    assert "error" in capsys.readouterr().err


def test_pml_profile_from_the_incidence_angle(tmp_path):
    spec = {
        "problem": "scattering",
        "dim": 2,
        "length_unit": "nm",
        "mesh": {"type": "square_with_disc", "n": 2, "radius": 50, "half_width": 200, "outer": 400},
        "order": 2,
        "wavelength": {"value": 500, "unit": "nm"},
        "materials": {"background": "vacuum", "2": "Au"},
        "source": {"type": "plane_wave", "angle": 60, "polarisation": "TE"},
        "pml": {"lower": [-200, -200], "upper": [200, 200], "thickness": [0, 0, 200, 200],
                "profile": {"theta_max": 60, "target": 1e-4}},
        "boundaries": {"pec": ["x_min", "x_max", "y_min", "y_max"]},
    }  # fmt: skip
    omega = hpfem.units.angular_frequency(wavelength=500e-9)
    setup = project.scattering_setup(spec, omega)
    expected = hpfem.PmlProfile.for_angle(60.0, 1e-4).reflection
    assert abs(setup.pml.profile.reflection / expected - 1) < 1e-12
    assert setup.pml.profile.order == 2
    # the plain reflection key with layers on the y sides and a wave at 60 deg from the x
    # axis (20 deg against the layer normals): no warning; at 10 deg (80 deg against the
    # normals) a warning; layers on every side never warn
    steep = {
        **spec,
        "source": {"type": "plane_wave", "angle": 70, "polarisation": "TE"},
        "pml": {**spec["pml"], "profile": {"order": 2, "reflection": 1e-8}},
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        setup = project.scattering_setup(steep, omega)
    assert setup.pml.profile.reflection == 1e-8
    grazing = {**steep, "source": {"type": "plane_wave", "angle": 10}}
    with pytest.warns(UserWarning, match="theta_max"):
        project.scattering_setup(grazing, omega)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        project.scattering_setup({**grazing, "pml": {**steep["pml"], "thickness": 200}}, omega)

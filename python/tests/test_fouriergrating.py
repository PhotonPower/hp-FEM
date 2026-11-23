"""Import of FourierGrating crossed-grating models (M21 S3): the frame and geometry mapping, the
scan, refusals with JSON pointers, the CLI on a stack without shapes (exact against the transfer
matrix) and the conventions against FourierGrating's RCWA (azimuth, s / p signs, mixed
polarisation, order labels; reference values embedded, docs/validation.md N)."""

import json
import math

import numpy as np
import pytest

import hpfem
from hpfem import cli
from hpfem import fouriergrating as fgi

um = 1e-6


def model(**changes):
    spec = {
        "dimension": "2D",
        "solver": "rcwa",
        "truncation": {"Nx": 8, "Nz": 8},
        "period_um": {"x": 0.5, "z": 0.4},
        "incidence": {"lambda_um": 0.6, "phi_deg": 20, "psi_deg": 30, "delta_deg": 0},
        "materials": {
            "air": {"n": 1.0},
            "glass": {"n": 1.5},
            "si": {"n": [3.9, 0.02]},
            "e": {"eps": [2.25, 0.0]},
        },
        "superstrate": "air",
        "substrate": "glass",
        "layers": [
            {"type": "homogeneous", "thickness_um": 0.05, "material": "e"},
            {
                "type": "contour2d",
                "thickness_um": 0.1,
                "background": "glass",
                "regions": [
                    {"material": "air", "circle": {"center_um": [0.25, 0.1], "radius_um": 0.08}},
                    {
                        "material": "si",
                        "ellipse": {
                            "center_um": [0.1, 0.3],
                            "radii_um": [0.06, 0.03],
                            "angle_deg": 30,
                        },
                    },
                ],
            },
            {
                "type": "polygon2d",
                "thickness_um": 0.07,
                "background": "air",
                "inclusions": [
                    {"material": "glass", "vertices_um": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.2]]},
                    {"material": "si", "rectangle_um": {"center": [0.4, 0.3], "size": [0.1, 0.05]}},
                ],
            },
        ],
    }
    spec.update(changes)
    return spec


def test_frame_geometry_and_materials():
    m = fgi.load(model())
    g = m.grating
    assert g.period == pytest.approx((0.5 * um, 0.4 * um))
    assert m.thickness == pytest.approx(0.22 * um)
    # FourierGrating lists the layers from the bottom, hpfem from the top
    assert [layer.thickness for layer in g.layers] == pytest.approx(
        [0.07 * um, 0.1 * um, 0.05 * um]
    )
    top, middle, bottom = g.layers
    assert complex(bottom.material.eps_r) == pytest.approx(2.25)
    # (x, z) -> (x, -z), wrapped into [0, period)
    circle, ellipse = middle.shapes
    assert circle.center == pytest.approx((0.25 * um, 0.3 * um))
    assert ellipse.center == pytest.approx((0.1 * um, 0.1 * um))
    assert ellipse.angle == pytest.approx(-math.radians(30))
    assert complex(ellipse.material.refractive_index) == pytest.approx(complex(3.9, 0.02))
    triangle, rect = top.shapes
    pts = np.asarray(triangle.points)
    assert pts[:, 0] == pytest.approx([0.3 * um, 0.3 * um, 0.1 * um])  # reversed: counter-clockwise
    assert pts[:, 1] == pytest.approx([0.2 * um, 0.3 * um, 0.3 * um])
    assert rect.center == pytest.approx((0.4 * um, 0.1 * um))
    assert (rect.width, rect.height) == pytest.approx((0.1 * um, 0.05 * um))
    # points and fields between the frames
    p = m.fg_to_hpfem([[0.1, 0.22, 0.05]])
    assert p[0] == pytest.approx([0.1 * um, -0.05 * um, 0.0])
    assert m.hpfem_to_fg_field([[1, 2, 3]])[0] == pytest.approx([1, 3, -2])


def test_scan_and_polarisation_convention():
    m = fgi.load(
        model(
            incidence={
                "lambda_um": [0.5, 0.6],
                "phi_deg": 10,
                "psi_deg": {"from": 0, "to": 90, "steps": 3},
                "delta_deg": [0, 90],
            }
        )
    )
    assert len(m.incidences) == 2 * 3 * 2
    first = m.incidences[0]
    assert first.wavelength == pytest.approx(0.5 * um) and first.theta == pytest.approx(
        math.radians(10)
    )
    psi45 = [i for i in m.incidences if i.fg["psi_deg"] == 45 and i.fg["delta_deg"] == 90][0]
    assert psi45.phi == pytest.approx(-math.radians(45))
    assert psi45.polarisation == pytest.approx((0.0, -1.0), abs=1e-15)  # FG's p is -p here


@pytest.mark.parametrize(
    ("change", "pointer"),
    [
        ({"dimension": "1D"}, "/dimension"),
        ({"sweep": {}}, "/sweep"),
        ({"materials": {"air": {"n": 1.0}, "glass": {"n": {"cauchy": {"A": 1.5}}}}}, "/materials"),
        ({"layers": [{"type": "profile2d", "thickness_um": 0.1}]}, "/layers/0"),
        (
            {
                "layers": [
                    {
                        "type": "contour2d",
                        "thickness_um": 0.1,
                        "background": "air",
                        "regions": [{"material": "glass", "nurbs": {}}],
                    }
                ]
            },
            "/layers/0/regions/0",
        ),
        ({"incidence": {"lambda_um": 0.5, "type": "gaussian_beam"}}, "/incidence/type"),
    ],
)
def test_unsupported_features_name_the_pointer(change, pointer):
    with pytest.raises(fgi.FourierGratingError, match=pointer):
        fgi.load(model(**change))


def test_cli_on_layers_without_shapes(tmp_path):
    """Homogeneous layers only: R and T of the (0, 0) order are the stack's, for both
    polarisations of the scan."""
    pytest.importorskip("gmsh")
    spec = model(
        layers=[
            {"type": "homogeneous", "thickness_um": 0.05, "material": "e"},
            {"type": "homogeneous", "thickness_um": 0.1, "material": "si"},
        ],
        incidence={"lambda_um": 0.6, "phi_deg": 20, "psi_deg": 30, "delta_deg": [0, 90]},
    )
    path = tmp_path / "stack.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    assert (
        cli.main(["fg", str(path), "-p", "1", "-e", "1.5", "-o", str(tmp_path / "out"), "-q"]) == 0
    )
    rows = fgi.read_efficiencies(tmp_path / "out" / "efficiencies.csv")
    k0 = 2 * math.pi / (0.6 * um)
    stack = hpfem.LayerStack3D(
        hpfem.Material.vacuum(),
        [
            hpfem.Layer(hpfem.Material(complex(3.9, 0.02) ** 2, 1.0), 0.1 * um),
            hpfem.Layer(hpfem.Material(2.25, 1.0), 0.05 * um),
        ],
        hpfem.Material.dielectric(1.5),
    )
    for delta, pol in ((0.0, hpfem.Polarisation.S), (90.0, hpfem.Polarisation.P)):
        exact = stack.plane_wave(k0, math.radians(20), pol, 1.0, -math.radians(30))
        assert rows[(0.6, 20.0, 30.0, delta, "R", 0, 0)] == pytest.approx(
            exact.reflectance, abs=1e-9
        )
        assert rows[(0.6, 20.0, 30.0, delta, "T", 0, 0)] == pytest.approx(
            exact.transmittance, abs=1e-9
        )


# FourierGrating 2caa01a (fgsolve, RCWA Nx = Nz = 8): an off-centre tilted elliptic hole in a
# glass film on glass, phi = 25°, psi = 30°, delta = ±45° (docs/validation.md N); the orders most
# sensitive to the conventions: with a wrong azimuth or p sign they swap or change by ~0.025
FG_OBLIQUE = {
    (45.0, "T", 1, -1): 0.004921,
    (-45.0, "T", 1, -1): 0.030569,
    (45.0, "T", 0, 1): 0.02915,
    (-45.0, "T", 0, 1): 0.007366,
    (45.0, "T", 0, -1): 0.014589,
    (-45.0, "T", 0, -1): 0.009099,
    (45.0, "T", 1, 0): 0.010443,
    (-45.0, "T", 1, 0): 0.023356,
    (45.0, "R", 0, 0): 0.03276,
    (-45.0, "R", 0, 0): 0.030975,
}


def test_conventions_against_fouriergrating():
    pytest.importorskip("gmsh")
    spec = {
        "dimension": "2D",
        "period_um": {"x": 0.5, "z": 0.5},
        "incidence": {"lambda_um": 0.405, "phi_deg": 25, "psi_deg": 30, "delta_deg": [45, -45]},
        "materials": {"air": {"n": 1.0}, "glass": {"n": 1.5}},
        "superstrate": "air",
        "substrate": "glass",
        "layers": [
            {
                "type": "contour2d",
                "thickness_um": 0.15,
                "background": "glass",
                "regions": [
                    {
                        "material": "air",
                        "ellipse": {
                            "center_um": [0.2, 0.3],
                            "radii_um": [0.14, 0.08],
                            "angle_deg": 25,
                        },
                    }
                ],
            }
        ],
    }
    m = fgi.load(spec)
    hpfem.set_log_level("warn")
    rows = {
        (r["delta_deg"], r["side"], r["n"], r["m"]): r["efficiency"]
        for r in fgi.efficiency_rows(fgi.solve(m, order=2, elements_per_wavelength=3.0))
    }
    for key, reference in FG_OBLIQUE.items():
        assert rows[key] == pytest.approx(reference, abs=3e-3), key

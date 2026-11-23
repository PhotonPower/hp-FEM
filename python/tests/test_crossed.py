"""Crossed (doubly periodic) gratings in 3D (M21 S2): the model classes, the Gmsh unit cell with
periodic side faces, and hpfem.crossed.solve on a bare stack (exact against the transfer matrix)
and on a lossless and an absorbing hole array (energy balance, C4 symmetry, mixed
polarisations)."""

import math

import numpy as np
import pytest

import hpfem
from hpfem import crossed as cg

um = 1e-6


def test_model_validation_and_shapes():
    with pytest.raises(cg.CrossedError):
        cg.Shape("hexagon", 1.0)
    with pytest.raises(cg.CrossedError):
        cg.Shape("circle", 1.0, radius=-1.0)
    with pytest.raises(cg.CrossedError):
        cg.CrossedGrating((0.5 * um,), 1.0, 1.5)
    with pytest.raises(cg.CrossedError):
        cg.CrossedGrating((0.5 * um, 0.5 * um), 1.0, 1.5, [cg.Layer(0.0, 1.5)])
    ellipse = cg.Shape("ellipse", 1.0, center=(1.0, 2.0), rx=0.3, ry=0.1, angle=math.pi / 2)
    x0, x1, y0, y1 = ellipse.bounds()
    assert (x0, x1) == pytest.approx((0.9, 1.1)) and (y0, y1) == pytest.approx((1.7, 2.3))
    rect = cg.Shape("rectangle", 1.0, center=(0.0, 0.0), width=2.0, height=1.0)
    assert rect.bounds() == pytest.approx((-1.0, 1.0, -0.5, 0.5))
    grating = cg.CrossedGrating(
        (0.5 * um, 0.4 * um), 1.0, 1.5, [cg.Layer(0.1 * um, 1.5), cg.Layer(0.2 * um, 2.0)]
    )
    assert grating.depth == pytest.approx(0.3 * um)
    assert grating.interfaces() == pytest.approx([0.0, -0.1 * um, -0.3 * um])
    assert cg._polarisation("s") == (1, 0)
    a_s, a_p = cg._polarisation((1.0, 1.0j))
    assert abs(a_s) == pytest.approx(1 / math.sqrt(2)) and a_p / a_s == pytest.approx(1j)
    material = cg.as_material(complex(1.5, 0.1), 1e15)
    assert complex(material.refractive_index) == pytest.approx(complex(1.5, 0.1))


def test_unit_cell_mesh_is_periodic_and_tagged():
    pytest.importorskip("gmsh")
    grating = cg.CrossedGrating(
        (0.5 * um, 0.5 * um),
        1.0,
        1.5,
        # a pillar crossing the corner of the cell: its four periodic pieces must match
        [cg.Layer(0.15 * um, 1.0, [cg.Shape("circle", 1.5, center=(0.0, 0.0), radius=0.15 * um)])],
    )
    omega = 2 * math.pi * hpfem.constants.c0 / (0.6 * um)
    sizes = cg.element_sizes(grating, omega, 2, 2.0)
    layout = cg.layout_for(grating, 0.6 * um, sizes)
    mesh, links = cg.unit_cell_mesh(grating, layout, sizes)
    assert sorted((int(m), int(s)) for m, s, _ in links) == [(1, 2), (3, 4)]
    tags = {mesh.cell_tag(c) for c in range(mesh.num_cells)}
    assert tags == {cg.SUPERSTRATE_TAG, cg.SUBSTRATE_TAG, cg.layer_tag(0), cg.shape_tag(0, 0)}
    # cells of the pillar lie in one of its periodic copies around the corners, the layer's
    # background cells outside (up to the chord of the curved boundary)
    r = 0.15 * um
    for c in range(mesh.num_cells):
        tag = mesh.cell_tag(c)
        if tag not in (cg.layer_tag(0), cg.shape_tag(0, 0)):
            continue
        x = np.mean([np.asarray(mesh.vertex(int(v))) for v in mesh.cell_vertices(c)], axis=0)
        d = min(math.hypot(x[0] - a, x[1] - b) for a in (0.0, 0.5 * um) for b in (0.0, 0.5 * um))
        assert -0.15 * um - 1e-12 < x[2] < 1e-12
        if tag == cg.shape_tag(0, 0):
            assert d < r
        else:
            assert d > 0.8 * r


def test_bare_stack_against_the_transfer_matrix():
    """Layers without shapes are the background itself: the orders are the stack's R and T."""
    pytest.importorskip("gmsh")
    grating = cg.CrossedGrating(
        (0.3 * um, 0.25 * um), 1.0, 1.5, [cg.Layer(0.1 * um, 2.0), cg.Layer(0.05 * um, 1.3)]
    )
    wavelength, theta, phi = 0.6 * um, 0.4, 0.8
    result = cg.solve(grating, wavelength, theta, phi, "s", order=1, elements_per_wavelength=1.5)
    k0 = 2 * math.pi / wavelength
    stack = hpfem.LayerStack3D(
        hpfem.Material.vacuum(),
        [
            hpfem.Layer(hpfem.Material.dielectric(2.0), 0.1 * um),
            hpfem.Layer(hpfem.Material.dielectric(1.3), 0.05 * um),
        ],
        hpfem.Material.dielectric(1.5),
    )
    for pol, key in ((hpfem.Polarisation.S, "s"), (hpfem.Polarisation.P, "p")):
        exact = stack.plane_wave(k0, theta, pol, 1.0, phi)
        r = result.polarised(key)
        assert r.order(0, 0, "R").efficiency == pytest.approx(exact.reflectance, abs=1e-9)
        assert r.order(0, 0, "T").efficiency == pytest.approx(exact.transmittance, abs=1e-9)
        assert abs(r.balance) < 1e-9
        assert r.absorption == 0.0
    direction = result.order(0, 0, "R").direction
    assert direction[2] > 0 and direction[:2] / np.linalg.norm(direction[:2]) == pytest.approx(
        [math.cos(phi), math.sin(phi)]
    )


def test_hole_array_balance_symmetry_and_mixed_polarisation():
    """Lossless glass film with air holes on glass (period 0.8 λ: first orders propagate in the
    substrate): ΣR + ΣT = 1, x and y polarisation equal at normal incidence (C4); with an
    absorbing film the mixed polarisation's absorption closes the balance as well."""
    pytest.importorskip("gmsh")
    wavelength = 1.0 * um
    hole = cg.Shape("circle", 1.0, center=(0.4 * um, 0.4 * um), radius=0.2 * um)
    grating = cg.CrossedGrating((0.8 * um, 0.8 * um), 1.0, 1.5, [cg.Layer(0.3 * um, 1.5, [hole])])
    result = cg.solve(grating, wavelength, 0.0, 0.0, "s", order=2)
    p_pol = result.polarised("p")
    assert result.order(1, 0, "T").efficiency > 1e-3  # the grating diffracts
    assert abs(result.balance) < 2e-3
    assert abs(p_pol.balance) < 2e-3
    # at phi = 0, s has E along -y and p along +x: equal by the C4 symmetry of the cell
    assert result.R == pytest.approx(p_pol.R, abs=1e-3)
    assert result.T == pytest.approx(p_pol.T, abs=1e-3)
    assert result.order(1, 0, "T").efficiency == pytest.approx(
        p_pol.order(0, 1, "T").efficiency, abs=1e-4
    )
    lossy = cg.CrossedGrating(
        (0.8 * um, 0.8 * um), 1.0, 1.5, [cg.Layer(0.3 * um, complex(1.5, 0.05), [hole])]
    )
    absorbing = cg.solve(lossy, wavelength, 0.3, 0.5, "s", order=2)
    p_abs = absorbing.polarised("p")
    left, right = absorbing.polarised((1.0, 1.0j)), absorbing.polarised((1.0, -1.0j))
    for r in (absorbing, p_abs, left, right):
        assert r.absorption > 1e-2
        assert abs(r.balance) < 2e-3
    # the cross terms of the two circular polarisations cancel: their mean absorption is that of
    # s and p (checks the mixed path with the combined background field)
    assert 0.5 * (left.absorption + right.absorption) == pytest.approx(
        0.5 * (absorbing.absorption + p_abs.absorption), rel=1e-9
    )

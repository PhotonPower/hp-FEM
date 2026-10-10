"""Far field by reciprocity on a layered background from Python (ADR-0014, M18 S3): the
homogeneous limit against the near-to-far transform, the half-space split of the power and the
collection over a cone."""

import numpy as np
import pytest

import hpfem

AXIS = 77


def half_disc(cells_per_radius, sphere_tag=2):
    full = hpfem.square_with_disc(cells_per_radius, 1.0, 2.0, 6.0, sphere_tag)
    mesh = hpfem.extract(full, lambda centroid: centroid[0] > 0)
    for f in mesh.boundary_facets:
        v = mesh.facet_vertices(int(f))
        if abs(mesh.vertex(int(v[0]))[0]) < 1e-12 and abs(mesh.vertex(int(v[1]))[0]) < 1e-12:
            mesh.set_facet_tag(int(f), AXIS)
    return mesh


def test_homogeneous_limit_and_cones():
    x = 1.5
    mesh = half_disc(4)
    nd = hpfem.NedelecDofMap2D(mesh, 2)
    h1 = hpfem.DofMap2D(mesh, 2)
    setup = hpfem.AxisymmetricScatteringSetup()
    setup.omega = x * hpfem.constants.c0
    setup.materials.set(2, hpfem.Material.dielectric(2.0))
    setup.axis_tag = AXIS
    setup.azimuthal_order = 1
    setup.pml = hpfem.PmlBox2D([0.0, -3.0], [3.0, 3.0], [0.0, 3.0, 3.0, 3.0], x)
    setup.incident = hpfem.axial_plane_wave(1.0, x, 1)
    field = hpfem.AxisymmetricScattering(nd, h1, setup).solve()
    surface = hpfem.Surface2D.around_cells(mesh, 2)
    vacuum = hpfem.LayerStack3D(hpfem.Material.vacuum(), [], hpfem.Material.vacuum(), 0.0)
    up = np.linspace(0.0, np.pi / 2 - 1e-6, 121)
    down = np.linspace(np.pi / 2 + 1e-6, np.pi, 121)
    far = hpfem.axisymmetric_layered_far_field(
        nd, h1, field.meridian, field.azimuthal, 1, setup.omega, setup.materials, surface,
        vacuum, up, down,
    )  # fmt: skip
    reference = hpfem.axisymmetric_far_field(
        nd, h1, field.meridian, field.azimuthal, 1, setup.omega, setup.materials, surface,
        np.concatenate([up, down]),
    )  # fmt: skip
    f_theta = np.concatenate([far.up.f_theta, far.down.f_theta])
    f_phi = np.concatenate([far.up.f_phi, far.down.f_phi])
    scale = np.max(np.abs(reference.f_theta)) + np.max(np.abs(reference.f_phi))
    assert np.max(np.abs(f_theta - np.array(reference.f_theta))) < 1e-9 * scale
    assert np.max(np.abs(f_phi - np.array(reference.f_phi))) < 1e-9 * scale
    # the two half-spaces carry the scattered power
    scattered = hpfem.axisymmetric_poynting_flux(
        nd, h1, field.meridian, field.azimuthal, 1, setup.omega, setup.materials, surface
    )
    total = far.up.radiated_power() + far.down.radiated_power()
    assert total == pytest.approx(scattered, rel=1e-2)
    # a cone of NA 0.5 above collects a part of the upper half
    cone = np.arcsin(0.5)
    assert 0 < far.up.power_between(0.0, cone) < far.up.radiated_power()

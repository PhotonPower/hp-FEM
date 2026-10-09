"""Floquet-Bloch band structures from Python: the empty lattice has the bands |k + G|, the
lowest band vanishes at Gamma without spurious modes, and a path walks Gamma-X-M."""

import numpy as np
import pytest

import hpfem


def square_lattice(bands=4):
    setup = hpfem.BandStructureSetup2D()
    setup.lattice = [
        hpfem.PeriodicPair2D(hpfem.box_tag.X_MIN, hpfem.box_tag.X_MAX, [1.0, 0.0]),
        hpfem.PeriodicPair2D(hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX, [0.0, 1.0]),
    ]
    setup.num_bands = bands
    setup.krylov_dimension = 40
    return setup


def empty_lattice(k, count):
    g = 2 * np.pi * np.array([[m, n] for m in range(-3, 4) for n in range(-3, 4)])
    return np.sort(np.linalg.norm(g + np.asarray(k), axis=1))[:count]


def test_empty_lattice_bands():
    mesh = hpfem.rectangle(6, 6)
    nd = hpfem.NedelecDofMap2D(mesh, 3)
    h1 = hpfem.DofMap2D(mesh, 3)
    problem = hpfem.BandStructure2D(nd, h1, square_lattice())
    assert problem.lattice_constant == 1.0
    k = np.array([0.7, 1.1])
    bands = problem.bands(k)
    assert np.allclose(bands.wave_vector, k)
    assert np.allclose(bands.wavenumber, empty_lattice(k, 4), rtol=2e-3)
    assert max(bands.residual) < 1e-8
    assert np.allclose(bands.normalised(1.0), np.array(bands.wavenumber) / (2 * np.pi))
    gamma = problem.bands([0.0, 0.0])
    assert abs(gamma.wavenumber[0]) < 1e-5 and abs(gamma.wavenumber[1]) < 1e-5
    assert np.isclose(gamma.wavenumber[2], 2 * np.pi, rtol=2e-3)


def test_path_and_dielectric_rods():
    mesh = hpfem.rectangle(8, 8, [-0.5, -0.5], [0.5, 0.5])
    for c, x in enumerate(mesh.cell_centroids):
        if np.linalg.norm(x) < 0.2:
            mesh.set_cell_tag(c, 2)
    nd = hpfem.NedelecDofMap2D(mesh, 2)
    h1 = hpfem.DofMap2D(mesh, 2)
    setup = square_lattice(3)
    setup.materials.set(2, hpfem.Material.dielectric(np.sqrt(8.9)))
    crystal = hpfem.BandStructure2D(nd, h1, setup)
    vacuum = hpfem.BandStructure2D(nd, h1, square_lattice(3))
    x_point = [np.pi, 0.0]
    assert crystal.bands(x_point).wavenumber[0] < vacuum.bands(x_point).wavenumber[0]
    path = crystal.path([[0.0, 0.0], x_point, [np.pi, np.pi], [0.0, 0.0]], 2)
    assert len(path) == 7
    assert np.allclose(path[-1].wave_vector, 0.0)
    lowest = [b.normalised(1.0)[0] for b in path]
    assert lowest[0] < 1e-5 and max(lowest) == max(lowest[2:5])


def rod_crystal(eps=8.9, keep_modes=True):
    mesh = hpfem.rectangle(8, 8, [-0.5, -0.5], [0.5, 0.5])
    for c, x in enumerate(mesh.cell_centroids):
        if np.linalg.norm(x) < 0.2:
            mesh.set_cell_tag(c, 2)
    nd = hpfem.NedelecDofMap2D(mesh, 2)
    h1 = hpfem.DofMap2D(mesh, 2)
    setup = square_lattice(3)
    setup.materials.set(2, hpfem.Material(eps, 1.0))
    setup.keep_modes = keep_modes
    return mesh, nd, h1, hpfem.BandStructure2D(nd, h1, setup)


def test_band_derivatives_and_group_velocity():
    mesh, nd, h1, crystal = rod_crystal()
    k = np.array([1.3, 0.6])
    bands = crystal.bands(k)
    assert bands.modes.shape == (nd.num_dofs, 3)
    # permittivity of the rods against re-solved bands
    d = hpfem.band_permittivity_derivative(crystal, bands, 2)
    h = 1e-3
    plus = rod_crystal(8.9 + h, False)[3].bands(k).wavenumber
    minus = rod_crystal(8.9 - h, False)[3].bands(k).wavenumber
    fd = (np.array(plus) - np.array(minus)) / (2 * h)
    assert np.allclose(d.wavenumber, fd, rtol=1e-6)
    assert list(d.multiplicity) == [1, 1, 1]
    assert np.allclose(d.angular_frequency, hpfem.constants.c0 * np.array(d.wavenumber))
    # rod radius by the mesh velocity of the normal growth: larger rods lower the bands
    velocity = hpfem.region_normal_velocity(mesh, 2)
    shape = hpfem.band_shape_derivative(crystal, bands, velocity)
    assert all(np.array(shape.wavenumber) < 0)
    # group velocity: components against the directional derivative and differences in k
    v = hpfem.group_velocity(crystal, bands)
    assert v.shape == (3, 2)
    direction = np.array([0.6, -0.8])
    along = hpfem.band_wave_vector_derivative(crystal, bands, direction)
    assert np.allclose(v @ direction, along.angular_frequency, rtol=1e-8)
    t = 1e-4
    fd_k = (
        np.array(crystal.bands(k + t * direction).wavenumber)
        - np.array(crystal.bands(k - t * direction).wavenumber)
    ) / (2 * t)
    assert np.allclose(along.wavenumber, fd_k, rtol=1e-6)
    # without kept modes the derivatives refuse
    plain = rod_crystal(keep_modes=False)[3]
    with pytest.raises(ValueError, match="keep_modes"):
        hpfem.group_velocity(plain, plain.bands(k))

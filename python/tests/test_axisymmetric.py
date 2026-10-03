"""Axisymmetric (2.5D) problems from Python: the PEC cylinder modes of orders 0 and 1 hit the
Bessel zeros, and the scattering cross-section of a dielectric sphere under the axial plane
wave matches the Mie series (SciPy spherical Bessel functions)."""

import numpy as np
import scipy.special

import hpfem

c0 = hpfem.constants.c0
AXIS = 77


def test_cylinder_modes():
    a, h = 1.0, 1.5
    mesh = hpfem.rectangle(6, 9, [0.0, 0.0], [a, h])
    nd = hpfem.NedelecDofMap2D(mesh, 3)
    h1 = hpfem.DofMap2D(mesh, 3)
    setup = hpfem.AxisymmetricCavitySetup()
    setup.axis_tag = hpfem.box_tag.X_MIN
    setup.pec_tags = [hpfem.box_tag.X_MAX, hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX]
    setup.num_modes = 3
    setup.krylov_dimension = 30
    modes0 = hpfem.AxisymmetricCavity(nd, h1, setup).solve()
    assert np.isclose(modes0[0].wavenumber, scipy.special.jn_zeros(0, 1)[0] / a, rtol=1e-4)
    assert modes0[0].meridian.shape == (nd.num_dofs,)
    setup.azimuthal_order = 1
    problem = hpfem.AxisymmetricCavity(nd, h1, setup)
    modes1 = problem.solve()
    te111 = np.sqrt((scipy.special.jnp_zeros(1, 1)[0] / a) ** 2 + (np.pi / h) ** 2)
    assert np.isclose(modes1[0].wavenumber, te111, rtol=1e-4)
    assert problem.num_free_dofs < nd.num_dofs + h1.num_dofs


def mie_scattering_efficiency(x, n, lmax=20):
    """Q_sca of a lossless sphere (Bohren & Huffman 4.53, 4.61)."""
    ls = np.arange(1, lmax + 1)
    jn, yn = scipy.special.spherical_jn, scipy.special.spherical_yn

    def psi(z):
        return z * jn(ls, z)

    def dpsi(z):
        return jn(ls, z) + z * jn(ls, z, derivative=True)

    def xi(z):
        return z * (jn(ls, z) + 1j * yn(ls, z))

    def dxi(z):
        return (jn(ls, z) + 1j * yn(ls, z)) + z * (
            jn(ls, z, derivative=True) + 1j * yn(ls, z, derivative=True)
        )

    a = (n * psi(n * x) * dpsi(x) - psi(x) * dpsi(n * x)) / (
        n * psi(n * x) * dxi(x) - xi(x) * dpsi(n * x)
    )
    b = (psi(n * x) * dpsi(x) - n * psi(x) * dpsi(n * x)) / (
        psi(n * x) * dxi(x) - n * xi(x) * dpsi(n * x)
    )
    return 2 / x**2 * np.sum((2 * ls + 1) * (abs(a) ** 2 + abs(b) ** 2))


def half_disc(cells_per_radius, sphere_tag=2):
    full = hpfem.square_with_disc(cells_per_radius, 1.0, 2.0, 6.0, sphere_tag)
    mesh = hpfem.extract(full, lambda centroid: centroid[0] > 0)
    for f in mesh.boundary_facets:
        v = mesh.facet_vertices(int(f))
        if abs(mesh.vertex(int(v[0]))[0]) < 1e-12 and abs(mesh.vertex(int(v[1]))[0]) < 1e-12:
            mesh.set_facet_tag(int(f), AXIS)
    return mesh


def test_sphere_scattering_matches_mie():
    n, x = 2.0, 1.5
    mesh = half_disc(4)
    nd = hpfem.NedelecDofMap2D(mesh, 2)
    h1 = hpfem.DofMap2D(mesh, 2)
    setup = hpfem.AxisymmetricScatteringSetup()
    setup.omega = x * c0
    setup.materials.set(2, hpfem.Material.dielectric(n))
    setup.axis_tag = AXIS
    setup.azimuthal_order = 1
    setup.pml = hpfem.PmlBox2D([0.0, -3.0], [3.0, 3.0], [0.0, 3.0, 3.0, 3.0], x)
    setup.incident = hpfem.axial_plane_wave(1.0, x, 1)
    problem = hpfem.AxisymmetricScattering(nd, h1, setup)
    assert np.isclose(problem.wavenumber, x)
    field = problem.solve()
    assert field.azimuthal_order == 1
    surface = hpfem.Surface2D.around_cells(mesh, 2)
    power = hpfem.axisymmetric_poynting_flux(
        nd, h1, field.meridian, field.azimuthal, 1, setup.omega, setup.materials, surface
    )
    sigma = 2 * power / (1.0 / (2 * hpfem.constants.Z0))
    mie = mie_scattering_efficiency(x, n) * np.pi
    assert np.isclose(sigma, mie, rtol=1e-2)
    # the incident component itself: (E_r, v, E_z) = (1/2, r/2, 0) e^{ikz}
    value = setup.incident(np.array([0.5, 0.25]))
    assert np.isclose(value[0], 0.5 * np.exp(1j * x * 0.25))
    assert np.isclose(value[1], 0.25 * np.exp(1j * x * 0.25))

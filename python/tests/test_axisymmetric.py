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


def test_dipole_in_vacuum_radiates_the_larmor_power():
    x, sigma = 1.5, 0.08
    mesh = half_disc(4)
    nd = hpfem.NedelecDofMap2D(mesh, 2)
    h1 = hpfem.DofMap2D(mesh, 2)
    setup = hpfem.AxisymmetricScatteringSetup()
    setup.omega = x * c0
    setup.axis_tag = AXIS
    setup.azimuthal_order = 0
    setup.pml = hpfem.PmlBox2D([0.0, -3.0], [3.0, 3.0], [0.0, 3.0, 3.0, 3.0], x)
    setup.current = hpfem.axisymmetric_gaussian_dipole(
        0.0, 1.0, hpfem.AxisDipole.AXIAL, sigma, setup.omega, 0
    )
    setup.extra_quadrature_order = 6
    field = hpfem.AxisymmetricScattering(nd, h1, setup).solve()
    surface = hpfem.Surface2D.around_cells(mesh, 2)
    power = hpfem.axisymmetric_poynting_flux(
        nd, h1, field.meridian, field.azimuthal, 0, setup.omega, setup.materials, surface
    )
    reference = hpfem.dipole_vacuum_power(1.0, setup.omega) * np.exp(-(x**2) * sigma**2)
    assert np.isclose(power, reference, rtol=3e-2)
    # far field: Larmor pattern sin(theta), no azimuthal component, same power
    theta = np.linspace(0.0, np.pi, 181)
    far = hpfem.axisymmetric_far_field(
        nd, h1, field.meridian, field.azimuthal, 0, setup.omega, setup.materials, surface, theta
    )
    f_theta = np.abs(np.array(far.f_theta))
    assert np.max(np.abs(np.array(far.f_phi))) < 1e-6 * f_theta.max()
    assert np.allclose(f_theta[30:151] / f_theta[90], np.sin(theta[30:151]), rtol=1e-2)
    assert np.isclose(far.radiated_power(), power, rtol=1e-2)


def test_oblique_incidence_sums_the_orders_to_mie():
    # a sphere scatters the same cross-section at every angle; at 50 degrees the plane wave
    # spreads over the orders m = 0, +-1, +-2, ... which are solved one by one and summed
    n, x = 2.0, 1.5
    theta_i = np.radians(50.0)
    mesh = half_disc(4)
    nd = hpfem.NedelecDofMap2D(mesh, 2)
    h1 = hpfem.DofMap2D(mesh, 2)
    setup = hpfem.AxisymmetricScatteringSetup()
    setup.omega = x * c0
    setup.materials.set(2, hpfem.Material.dielectric(n))
    setup.axis_tag = AXIS
    setup.pml = hpfem.PmlBox2D([0.0, -3.0], [3.0, 3.0], [0.0, 3.0, 3.0, 3.0], x)
    surface = hpfem.Surface2D.around_cells(mesh, 2)
    result = hpfem.scatter_orders(
        nd,
        h1,
        setup,
        lambda m: hpfem.oblique_plane_wave(1.0, x, theta_i, hpfem.PlanePolarisation.S, m),
        8,
        surface,
        1e-5,
    )
    assert result.orders[:3] == [0, 1, -1]
    assert len(result.orders) == len(result.fields) == len(result.power)
    power = np.asarray(result.power)
    assert np.allclose(power[1::2][: (len(power) - 1) // 2], power[2::2], rtol=1e-6)
    sigma = result.total_power() / (1.0 / (2 * hpfem.constants.Z0))
    mie = mie_scattering_efficiency(x, n) * np.pi
    assert np.isclose(sigma, mie, rtol=5e-2)
    # the summed orders restore the plane wave at a point: E_z = -sin(theta) e^{ik.x} for p
    theta = np.linspace(0.0, np.pi, 19)
    patterns = [
        hpfem.axisymmetric_far_field(
            nd, h1, f.meridian, f.azimuthal, m, setup.omega, setup.materials, surface, theta
        )
        for m, f in zip(result.orders, result.fields)
    ]
    total = hpfem.superpose_far_field(patterns, result.orders, 0.3)
    assert len(total.f_theta) == len(theta)
    assert total.radiated_power() > 0.0
    point = np.array([0.6, 0.3])
    orders = range(-20, 21)
    e_z = sum(
        np.exp(1j * m * 0.7)
        * hpfem.oblique_plane_wave(1.0, x, theta_i, hpfem.PlanePolarisation.P, m)(point)[2]
        for m in orders
    )
    phase = np.exp(1j * x * (np.sin(theta_i) * 0.6 * np.cos(0.7) + np.cos(theta_i) * 0.3))
    assert np.isclose(e_z, -np.sin(theta_i) * phase)


def test_estimator_and_error_on_a_locally_refined_meridian_mesh():
    # manufactured gradient mode at a re-entrant PEC edge (axisymmetric_corner.hpp of the
    # C++ tests):
    # local refinement of the edge cells (hanging nodes) lowers the error, the estimator follows
    full = hpfem.rectangle(4, 4, [0.0, -1.0], [2.0, 1.0])
    root = hpfem.extract(full, lambda c: not (c[0] > 1.0 and c[1] < 0.0))
    for f in root.boundary_facets:
        v = [root.vertex(int(i)) for i in root.facet_vertices(int(f))]
        root.set_facet_tag(int(f), AXIS if max(abs(p[0]) for p in v) < 1e-12 else 9)
    nu = 2.0 / 3.0

    def potential(x):
        r, z = x
        rho = np.hypot(r - 1.0, z)
        theta = np.arctan2(z, r - 1.0) % (2 * np.pi)
        return r * (2.0 - r) * (1.0 - z * z) * rho**nu * np.sin(nu * theta)

    def exact(x):
        h = 1e-6
        d_r = (potential(x + [h, 0.0]) - potential(x - [h, 0.0])) / (2 * h)
        d_z = (potential(x + [0.0, h]) - potential(x - [0.0, h])) / (2 * h)
        return np.array([d_r, potential(x), d_z], dtype=complex)

    adaptive = hpfem.AdaptiveMesh2D(root)
    errors = []
    for _ in range(2):
        mesh = adaptive.mesh
        nd = hpfem.NedelecDofMap2D(mesh, 2)
        h1 = hpfem.DofMap2D(mesh, 2)
        setup = hpfem.AxisymmetricScatteringSetup()
        setup.omega = c0
        setup.pec_tags = [9]
        setup.axis_tag = AXIS
        setup.azimuthal_order = 1
        setup.current = lambda x: -exact(x)
        problem = hpfem.AxisymmetricScattering(nd, h1, setup)
        field = problem.solve()
        err = problem.error(field, exact)
        est = problem.estimate(field)
        assert len(est.indicators) == mesh.num_cells
        assert 0.1 < est.total() / np.hypot(err.l2, err.curl) < 50.0
        assert np.isclose(
            err.l2,
            hpfem.axisymmetric_error(nd, h1, field.meridian, field.azimuthal, 1, exact).l2,
        )
        errors.append(np.hypot(err.l2, err.curl))
        corner = [
            c
            for c in range(mesh.num_cells)
            if any(
                np.linalg.norm(mesh.vertex(int(v)) - [1.0, 0.0]) < 1e-12
                for v in mesh.cell_vertices(c)
            )
        ]
        adaptive.refine(corner)
    assert not adaptive.mesh.is_conforming
    assert errors[1] < 0.9 * errors[0]

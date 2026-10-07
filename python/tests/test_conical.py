"""Conical incidence and the E_z polarisation: a flat interface under conical incidence with
the layered background has a vanishing scattered field and the orders give the Fresnel R
and T; the E_z Mie cylinder matches the series through the flux in the vacuum."""

import numpy as np
import scipy.special

import hpfem

c0 = hpfem.constants.c0


def test_flat_interface_conical_orders_match_the_stack():
    k0, n2, period = 2 * np.pi, 1.6, 0.7
    angle, azimuth = np.radians(35.0), np.radians(50.0)
    stack = hpfem.LayerStack2D(
        hpfem.Material.dielectric(1.0), [], hpfem.Material.dielectric(n2), 0.0
    )
    mesh = hpfem.rectangle(4, 24, [0.0, -2.0], [period, 2.0])
    for c in range(mesh.num_cells):
        if mesh.cell_centroid(c)[1] < 0:
            mesh.set_cell_tag(c, 2)
    nd = hpfem.NedelecDofMap2D(mesh, 3)
    h1 = hpfem.DofMap2D(mesh, 3)
    for pol in (hpfem.Polarisation.S, hpfem.Polarisation.P):
        wave = hpfem.layered_conical_wave(stack, k0, angle, azimuth, pol)
        assert np.isclose(wave.beta, k0 * np.sin(angle) * np.sin(azimuth))
        setup = hpfem.ConicalScatteringSetup()
        setup.omega = k0 * c0
        setup.beta = wave.beta
        setup.materials.set(2, hpfem.Material.dielectric(n2))
        setup.background = stack
        setup.incident = wave.field
        setup.pml = hpfem.PmlBox2D([0.0, -1.0], [period, 1.0], [0.0, 0.0, 1.0, 1.0], k0)
        setup.pec_tags = [hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX]
        setup.periodic = [
            hpfem.PeriodicPair2D(
                hpfem.box_tag.X_MIN,
                hpfem.box_tag.X_MAX,
                [period, 0.0],
                hpfem.bloch_phase([wave.kx, 0.0], [period, 0.0]),
            )
        ]
        problem = hpfem.ConicalScattering(nd, h1, setup)
        solution = problem.solve()
        assert np.linalg.norm(solution.transverse) < 1e-8
        assert np.linalg.norm(solution.longitudinal) < 1e-8
        locator = hpfem.PointLocator2D(mesh)

        def reflected_field(x):
            total = problem.total_field(solution, locator, x)
            i = wave.incident(x)
            return total - np.array([i[0], i[1], 1j * i[2]])

        reflected = hpfem.conical_fourier_coefficients(
            reflected_field, [0.0, 0.5], [1.0, 0.0], period, wave.kx, 1, 32
        )
        orders = hpfem.conical_diffraction_efficiencies(
            reflected, k0, 1.0, period, wave.kx, wave.beta, wave.ky, 1.0
        )
        assert np.isclose(orders[1].efficiency, wave.reflectance, rtol=1e-5)
        assert orders[1].propagating and orders[1].order == 0
        transmitted = hpfem.conical_fourier_coefficients(
            lambda x: problem.total_field(solution, locator, x),
            [0.0, -0.5],
            [1.0, 0.0],
            period,
            wave.kx,
            1,
            32,
        )
        t = hpfem.conical_diffraction_efficiencies(
            transmitted, k0, n2, period, wave.kx, wave.beta, wave.ky, 1.0
        )
        assert np.isclose(t[1].efficiency, wave.transmittance, rtol=1e-5)


def test_ez_mie_cylinder_matches_the_series():
    radius, index, k = 0.25, 1.5, 6.0
    x, m = k * radius, index
    jv, jvp = scipy.special.jv, scipy.special.jvp
    h1v, h1vp = scipy.special.hankel1, scipy.special.h1vp
    s = 0.0
    for n in range(0, 40):
        b = (jvp(n, x) * jv(n, m * x) - m * jvp(n, m * x) * jv(n, x)) / (
            m * jvp(n, m * x) * h1v(n, x) - h1vp(n, x) * jv(n, m * x)
        )
        s += (1 if n == 0 else 2) * abs(b) ** 2
    exact = 2 * radius * (2 / x) * s
    mesh = hpfem.square_with_disc(4, radius, 1.0, 2.0, 2)
    for c in range(mesh.num_cells):
        if mesh.cell_tag(c) != 2 and np.linalg.norm(mesh.cell_centroid(c)) < 0.45:
            mesh.set_cell_tag(c, 7)
    nd = hpfem.NedelecDofMap2D(mesh, 3)
    h1 = hpfem.DofMap2D(mesh, 3)
    setup = hpfem.ConicalScatteringSetup()
    setup.omega = k * c0
    setup.beta = 0.0
    setup.materials.set(2, hpfem.Material.dielectric(index))
    amplitude = np.array([0.0, 0.0, 1.0], dtype=complex)
    setup.incident = hpfem.conical_plane_wave(amplitude, [k, 0.0, 0.0])
    setup.pml = hpfem.PmlBox2D.uniform(
        [-1.0, -1.0], [1.0, 1.0], 1.0, k, 1.0, hpfem.PmlProfile(2, 1e-10)
    )
    setup.pec_tags = [
        hpfem.box_tag.X_MIN,
        hpfem.box_tag.X_MAX,
        hpfem.box_tag.Y_MIN,
        hpfem.box_tag.Y_MAX,
    ]
    problem = hpfem.ConicalScattering(nd, h1, setup)
    solution = problem.solve()
    assert np.linalg.norm(solution.transverse) < 1e-10 * np.linalg.norm(solution.longitudinal)
    # the measurement surface: the outer boundary of the tagged vacuum region
    inner = hpfem.Surface2D.around_cells(mesh, 7)
    outer = hpfem.Surface2D()
    outer.facets = [
        f
        for f in inner.facets
        if all(mesh.cell_tag(int(c)) != 2 for c in mesh.facet_cells(f.facet) if c >= 0)
    ]
    power = hpfem.conical_poynting_flux(
        nd, h1, solution.transverse, solution.longitudinal, 0.0, setup.omega, setup.materials, outer
    )
    width = power / (1.0 / (2 * hpfem.constants.Z0))
    assert np.isclose(width, exact, rtol=1e-3)


def test_hp_loop_with_the_conical_estimator():
    """SOLVE - ESTIMATE - MARK - DECIDE - REFINE on the L-shaped domain with the curl-free mode
    E = grad(psi exp(i beta z)), psi = (1 - x^2)(1 - y^2) r^(2/3) sin(2 theta / 3): the
    estimator of the coupled system drives h-refinement to the corner and the error against
    the exact field falls."""
    beta = 1.3
    boundary = 9
    nu = 2.0 / 3.0

    def l_shape(n):
        square = hpfem.rectangle(2 * n, 2 * n, [-1.0, -1.0], [1.0, 1.0])
        mesh = hpfem.extract(square, lambda x: not (x[0] > 0 and x[1] < 0))
        for f in mesh.boundary_facets:
            mesh.set_facet_tag(f, boundary)
        return mesh

    def potential(x):
        r = np.hypot(x[0], x[1])
        if r == 0:
            return 0.0, 0.0, 0.0
        theta = np.arctan2(x[1], x[0])
        theta = theta + 2 * np.pi if theta < 0 else theta
        dr = nu * r ** (nu - 1) * np.sin(nu * theta)
        dt = nu * r ** (nu - 1) * np.cos(nu * theta)
        s, s_x, s_y = (
            r**nu * np.sin(nu * theta),
            np.cos(theta) * dr - np.sin(theta) * dt,
            np.sin(theta) * dr + np.cos(theta) * dt,
        )
        # the bump (1 - x^2)(1 - y^2) makes the tangential field vanish on the PEC walls
        a = (1 - x[0] ** 2) * (1 - x[1] ** 2)
        a_x = -2 * x[0] * (1 - x[1] ** 2)
        a_y = -2 * x[1] * (1 - x[0] ** 2)
        return a * s, a_x * s + a * s_x, a_y * s + a * s_y

    def exact(x):  # physical (E_x, E_y, E_z = i beta psi)
        psi, dx, dy = potential(x)
        return np.array([dx, dy, 1j * beta * psi], dtype=complex)

    def current(x):  # scaled source f = -k^2 E: (f_x, f_y, f_v = -i f_z)
        psi, dx, dy = potential(x)
        return -np.array([dx, dy, beta * psi], dtype=complex)

    adaptive = hpfem.AdaptiveMesh2D(l_shape(2))
    orders = np.ones(adaptive.mesh.num_cells, dtype=int)
    predicted = np.zeros(0)
    errors, dofs_count, effectivity = [], [], []
    for _ in range(5):
        mesh = adaptive.mesh
        nd = hpfem.NedelecDofMap2D(mesh, orders.tolist())
        h1 = hpfem.DofMap2D(mesh, orders.tolist())
        setup = hpfem.ConicalScatteringSetup()
        setup.omega = hpfem.constants.c0  # k0 = 1
        setup.beta = beta
        setup.pec_tags = [boundary]
        setup.current = current
        problem = hpfem.ConicalScattering(nd, h1, setup)
        solution = problem.solve()
        estimate = problem.estimate(solution)
        assert estimate.indicators.shape == (mesh.num_cells,)
        error = problem.error(solution, exact)
        errors.append(np.hypot(error.l2, error.curl))
        effectivity.append(estimate.total() / errors[-1])
        dofs_count.append(len(problem.free_dofs))
        marked = hpfem.dorfler_marking(estimate.indicators, 0.5)
        decision = hpfem.hp_decide_by_prediction(estimate.indicators, predicted, marked)
        hp = hpfem.hp_refine(adaptive, orders.tolist(), decision.h_marked, decision.p_marked)
        predicted = hpfem.predict_indicators(estimate.indicators, orders.tolist(), hp)
        orders = hp.orders
    assert dofs_count[-1] > dofs_count[0]
    assert errors[-1] < 0.5 * errors[0]
    assert all(1.0 < e < 10.0 for e in effectivity)
    assert adaptive.max_level >= 2  # the corner is h-refined
    assert orders.max() >= 2  # smooth cells are p-refined


def test_adaptive_solve_streams_steps_and_stops_on_tolerance():
    """hpfem.adaptive_solve on the conical corner problem of the hp-loop test with a point
    goal (conical_dwr_estimate): the steps stream DoFs, eta, observables and the goal error,
    the goal error falls, and the loop stops when the observable settles."""
    beta = 1.3
    boundary = 9
    nu = 2.0 / 3.0

    def l_shape(n):
        square = hpfem.rectangle(2 * n, 2 * n, [-1.0, -1.0], [1.0, 1.0])
        mesh = hpfem.extract(square, lambda x: not (x[0] > 0 and x[1] < 0))
        for f in mesh.boundary_facets:
            mesh.set_facet_tag(f, boundary)
        return mesh

    def potential(x):
        r = np.hypot(x[0], x[1])
        if r == 0:
            return 0.0, 0.0, 0.0
        theta = np.arctan2(x[1], x[0])
        theta = theta + 2 * np.pi if theta < 0 else theta
        dr = nu * r ** (nu - 1) * np.sin(nu * theta)
        dt = nu * r ** (nu - 1) * np.cos(nu * theta)
        s, s_x, s_y = (
            r**nu * np.sin(nu * theta),
            np.cos(theta) * dr - np.sin(theta) * dt,
            np.sin(theta) * dr + np.cos(theta) * dt,
        )
        a = (1 - x[0] ** 2) * (1 - x[1] ** 2)
        a_x = -2 * x[0] * (1 - x[1] ** 2)
        a_y = -2 * x[1] * (1 - x[0] ** 2)
        return a * s, a_x * s + a * s_x, a_y * s + a * s_y

    def current(x):
        psi, dx, dy = potential(x)
        return -np.array([dx, dy, beta * psi], dtype=complex)

    point = np.array([-0.55, 0.45])
    weight = np.array([1.0, 0.5, -0.7j])
    psi, dx, dy = potential(point)
    exact_goal = np.array([dx, dy, 1j * beta * psi]) @ weight
    functional = hpfem.conical_point_functional(point, weight)

    def factory(mesh, orders):
        nd = hpfem.NedelecDofMap2D(mesh, orders.tolist())
        h1 = hpfem.DofMap2D(mesh, orders.tolist())
        setup = hpfem.ConicalScatteringSetup()
        setup.omega = hpfem.constants.c0
        setup.beta = beta
        setup.pec_tags = [boundary]
        setup.current = current
        problem = hpfem.ConicalScattering(nd, h1, setup)
        return problem, problem.solve()

    def goal(problem, solution):
        return hpfem.conical_dwr_estimate(problem, solution, functional)

    adaptive = hpfem.AdaptiveMesh2D(l_shape(2))
    hpfem.refine_at_points(adaptive, [np.zeros(2)], 2)  # corner pre-refinement
    assert adaptive.max_level == 2
    steps = list(
        hpfem.adaptive_solve(
            adaptive,
            factory,
            observe=lambda p, s: {"q_re": float(goal(p, s).value.real)},
            goal=goal,
            tolerance=2e-3,
            max_dofs=4000,
            max_steps=25,
        )
    )
    assert len(steps) >= 3
    assert steps[-1].dofs > steps[0].dofs
    assert all(isinstance(s, hpfem.AdaptiveStep) for s in steps)
    assert steps[1].change["q_re"] >= 0.0
    goal_errors = [abs(s.goal_value - exact_goal) for s in steps]
    assert goal_errors[-1] < 0.2 * goal_errors[0]
    assert steps[-1].converged or steps[-1].dofs > 4000
    if steps[-1].converged:
        assert steps[-1].goal_error < 2e-3
        assert all(c < 2e-3 for c in steps[-1].change.values())

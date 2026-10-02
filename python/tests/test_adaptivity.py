"""The adaptive loop from Python: estimate, mark, decide, hp-refine, prolongate on the
L-shaped domain with the singular reference field (convergence test #7 at small size),
plus the goal-oriented estimator."""

import numpy as np
from conftest import mie_problem

import hpfem

K = 1.0
BOUNDARY = 9


def l_shape(n):
    square = hpfem.rectangle(2 * n, 2 * n, [-1.0, -1.0], [1.0, 1.0])
    mesh = hpfem.extract(square, lambda x: not (x[0] > 0 and x[1] < 0))
    for f in mesh.boundary_facets:
        mesh.set_facet_tag(f, BOUNDARY)
    return mesh


def singular_field():
    """E = grad(r^(2/3) sin(2 theta / 3)) curl-free with k^2 E + ... : the standard corner
    singularity as a total field, with the matching curl (zero) and a volume current."""
    nu = 2.0 / 3.0

    def angle(x):
        theta = np.arctan2(x[1], x[0])
        return theta + 2 * np.pi if theta < 0 else theta

    def value(x):
        r = np.hypot(x[0], x[1])
        if r == 0:
            return np.zeros(2, dtype=complex)
        theta = angle(x)
        dr = nu * r ** (nu - 1) * np.sin(nu * theta)
        dt = r ** (nu - 1) * nu * np.cos(nu * theta)
        return np.array(
            [np.cos(theta) * dr - np.sin(theta) * dt, np.sin(theta) * dr + np.cos(theta) * dt],
            dtype=complex,
        )

    return hpfem.IncidentField2D(value, lambda x: np.zeros(1, dtype=complex)), value


def setup_for(value):
    setup = hpfem.ScatteringSetup2D()
    setup.omega = K * hpfem.constants.c0
    field, _ = singular_field()
    setup.incident = field
    setup.incident_tags = [BOUNDARY]
    # curl curl E - k^2 E = f with curl E = 0: f = -k^2 E
    setup.current = lambda x: -(K**2) * value(x)
    return setup


def test_hp_loop_reduces_the_error():
    field, value = singular_field()
    adaptive = hpfem.AdaptiveMesh2D(l_shape(2))
    orders = np.ones(adaptive.mesh.num_cells, dtype=int)
    predicted = np.zeros(0)
    errors, dofs_count = [], []
    previous = None
    for _ in range(5):
        mesh = adaptive.mesh
        dofs = hpfem.NedelecDofMap2D(mesh, orders.tolist())
        problem = hpfem.Scattering2D(dofs, setup_for(value))
        solution = problem.solve()
        if previous is not None:
            old_dofs, old_solution, step = previous
            transferred = hpfem.prolongate(old_dofs, old_solution, dofs, step)
            assert transferred.shape == (dofs.num_dofs,)
        estimate = problem.estimate(solution)
        error = problem.error(solution, field)
        errors.append(np.hypot(error.l2, error.curl))
        dofs_count.append(dofs.num_dofs)
        marked = hpfem.dorfler_marking(estimate.indicators, 0.5)
        decision = hpfem.hp_decide_by_prediction(estimate.indicators, predicted, marked)
        assert len(decision.h_marked) + len(decision.p_marked) == len(marked)
        hp = hpfem.hp_refine(adaptive, orders.tolist(), decision.h_marked, decision.p_marked)
        predicted = hpfem.predict_indicators(estimate.indicators, orders.tolist(), hp)
        previous = (dofs, solution.unknown, hp.step)
        orders = hp.orders
    assert dofs_count[-1] > dofs_count[0]
    assert errors[-1] < 0.5 * errors[0]
    assert adaptive.max_level >= 2  # the corner is h-refined
    assert orders.max() >= 2  # smooth cells are p-refined


def test_marking_and_p_refinement_helpers():
    eta = np.array([4.0, 3.0, 2.0, 1.0])
    assert hpfem.dorfler_marking(eta, 0.5).tolist() == [0]
    assert hpfem.maximum_marking(eta, 0.6).tolist() == [0, 1]
    assert hpfem.p_refine([1, 1, 2, 2], [1, 2], max_order=2).tolist() == [1, 2, 2, 2]
    step = hpfem.identity_step(4)
    assert step.num_cells == 4 and not step.refined(0)


def test_coefficient_decay_is_large_for_a_polynomial():
    mesh = hpfem.rectangle(2, 2)
    dofs = hpfem.DofMap2D(mesh, 3)
    u = hpfem.interpolate(dofs, lambda x: x[0] + 2 * x[1])
    decay = hpfem.coefficient_decay(dofs, u, [0, 1])
    assert np.all(np.isinf(decay))  # degree 1 is represented exactly: no energy above
    decision = hpfem.hp_decide(dofs, u, [0, 1])
    assert len(decision.p_marked) == 2


def test_goal_oriented_estimate_for_a_point_value():
    mesh, dofs, problem = mie_problem(n=2, p=2)
    solution = problem.solve()
    x = [0.5, 0.1]
    weight = [1.0, 0.0]
    goal = hpfem.dwr_estimate(problem, solution, hpfem.point_value_functional(x, weight))
    locator = hpfem.PointLocator2D(mesh)
    assert np.isclose(goal.value, problem.scattered_field(solution, locator, x)[0])
    assert goal.indicators.shape == (mesh.num_cells,)
    assert goal.total() >= abs(goal.error)

    # the same goal as a Python functional returning q with Q(E_h) = q^T e_h
    def functional(map_):
        loc = hpfem.PointLocator2D(map_.mesh)
        return hpfem.point_functional(map_, loc, [x], [weight])

    again = hpfem.dwr_estimate(problem, solution, functional)
    assert np.isclose(again.error, goal.error)

"""Dual formulation and hypercircle bound from Python: on the coercive problem
curl curl E + E = f with PEC walls the estimate bounds the energy error from above, and the
dual space of order p + 1 makes the bound sharp."""

import numpy as np

import hpfem


def coercive_problem(n, p):
    mesh = hpfem.rectangle(n, n)
    nd = hpfem.NedelecDofMap2D(mesh, p)
    exact = lambda x: np.array([np.sin(np.pi * x[1]), np.sin(np.pi * x[0])], dtype=complex)  # noqa: E731
    form = hpfem.MaxwellForm2D(
        inverse_permeability=lambda x: np.eye(1, dtype=complex),
        permittivity=lambda x: np.eye(2, dtype=complex),
        source=lambda x: (np.pi**2 + 1) * exact(x),
    )
    k2 = -1.0
    system = hpfem.assemble_maxwell_operator(nd, lambda c: form, k2)
    pec = hpfem.homogeneous_dirichlet(nd, [int(f) for f in mesh.boundary_facets])
    matrix, rhs = hpfem.apply_dirichlet(system.matrix, system.rhs, pec)
    e_h = hpfem.solve_direct(matrix, rhs)
    errors = hpfem.hcurl_error(
        nd,
        e_h,
        exact,
        lambda x: np.array([np.pi * (np.cos(np.pi * x[0]) - np.cos(np.pi * x[1]))], dtype=complex),
    )
    energy = np.sqrt(errors.curl**2 + errors.l2**2)
    return mesh, nd, form, k2, e_h, energy


def test_hypercircle_bound_and_sharpness():
    mesh, nd, form, k2, e_h, energy = coercive_problem(4, 2)
    assert 0.01 < energy < 0.2
    etas = {}
    for dual_order in (2, 3):
        h1 = hpfem.DofMap2D(mesh, dual_order)
        sigma = hpfem.dual_solution(h1, lambda c: form, k2)
        assert sigma.shape == (h1.num_dofs,)
        estimate = hpfem.hypercircle_estimate(nd, e_h, h1, sigma, lambda c: form, k2)
        assert estimate.indicators.shape == (mesh.num_cells,)
        assert np.isclose(estimate.total(), np.sqrt(np.sum(estimate.indicators**2)))
        assert all(p.constitutive >= 0 and p.equilibrium >= 0 for p in estimate.parts)
        assert 0 <= estimate.argmax() < mesh.num_cells
        etas[dual_order] = estimate.total()
    assert etas[2] >= energy and etas[3] >= energy  # guaranteed upper bound
    assert etas[2] < 6 * energy
    assert etas[3] < 1.2 * energy  # sharp with the dual order p + 1


def quadratic_form():
    # E = (y(1-y), x(1-x)): curl E = 2(y - x), f = curl curl E + E
    return hpfem.MaxwellForm2D(
        source=lambda x: np.array([2 + x[1] * (1 - x[1]), 2 + x[0] * (1 - x[0])], dtype=complex)
    )


def test_dual_solution_with_hanging_nodes_is_exact():
    adaptive = hpfem.AdaptiveMesh2D(hpfem.rectangle(3, 3))
    adaptive.refine([0, 4])
    adaptive.refine([1])
    mesh = adaptive.mesh
    h1 = hpfem.DofMap2D(mesh, 2)
    assert hpfem.hanging_constraints(h1).num_constrained > 0
    form = quadratic_form()
    sigma = hpfem.dual_solution(h1, lambda c: form, -1.0)
    errors = hpfem.h1_error(h1, sigma, lambda x: 2 * (x[1] - x[0]), lambda x: np.array([-2.0, 2.0]))
    assert errors.l2 < 1e-10 and errors.h1_semi < 1e-9


def test_dual_solution_carries_the_bloch_phase():
    mesh = hpfem.rectangle(4, 4)
    h1 = hpfem.DofMap2D(mesh, 2)
    shift = np.array([1.0, 0.0])
    phase = np.exp(0.7j)
    pairs = [hpfem.PeriodicPair2D(hpfem.box_tag.X_MIN, hpfem.box_tag.X_MAX, shift, phase)]
    form = quadratic_form()
    sigma = hpfem.dual_solution(h1, lambda c: form, -1.0, periodic=pairs)
    assert sigma.shape == (h1.num_dofs,)
    # the slaves follow their masters: sigma = P sigma_free
    c = hpfem.bloch_constraints(h1, pairs)
    p = c.prolongation()
    free = [i for i in range(h1.num_dofs) if not c.is_constrained(i)]
    assert np.abs(p @ sigma[free] - sigma).max() < 1e-12 * np.abs(sigma).max()

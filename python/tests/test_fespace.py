import numpy as np
import pytest

import hpfem


def test_dof_counts_uniform_and_variable_order():
    m = hpfem.rectangle(2, 2)
    h1 = hpfem.DofMap2D(m, 2)
    assert h1.num_dofs == m.num_vertices + m.num_edges  # p = 2: one bubble per edge
    nd = hpfem.NedelecDofMap2D(m, 1)
    assert nd.num_dofs == m.num_edges
    orders = [1] * m.num_cells
    orders[0] = 3
    mixed = hpfem.NedelecDofMap2D(m, orders)
    assert mixed.max_order == 3 and mixed.cell_order(0) == 3
    assert mixed.cell_orders.tolist() == orders
    # minimum rule: an edge of cell 0 shared with an order-1 cell keeps order 1
    shared = [e for e in m.cell_edges(0) if not m.is_boundary_facet(e)]
    assert all(mixed.edge_order(e) == 1 for e in shared)
    assert len(mixed.interior_dofs(0)) == 6  # p(p - 1) interior functions in 2D
    with pytest.raises(ValueError):
        hpfem.DofMap2D(m, [1, 2])


def test_facet_dofs_and_tags():
    m = hpfem.rectangle(3, 3)
    nd = hpfem.NedelecDofMap2D(m, 2)
    left = nd.dofs_on_tag(hpfem.box_tag.X_MIN)
    assert len(left) == 2 * 3  # p DoFs per boundary edge
    f = m.facets_with_tag(hpfem.box_tag.X_MIN)[0]
    assert set(nd.facet_dofs(f)) <= set(left)


def test_hanging_constraints_reduce_the_space():
    adaptive = hpfem.AdaptiveMesh2D(hpfem.rectangle(2, 2))
    adaptive.refine([0])
    nd = hpfem.NedelecDofMap2D(adaptive.mesh, 2)
    c = hpfem.hanging_constraints(nd)
    assert c.num_dofs == nd.num_dofs and 0 < c.num_constrained < nd.num_dofs
    slave = next(d for d in range(nd.num_dofs) if c.is_constrained(d))
    assert all(not c.is_constrained(t.master) for t in c.terms(slave))
    x = np.ones(c.num_free, dtype=complex)
    full = c.expand(x)
    assert full.shape == (nd.num_dofs,)
    p = c.prolongation()
    assert p.shape == (nd.num_dofs, c.num_free)
    assert np.allclose(p @ x, full)


def test_bloch_constraints_and_phase():
    m = hpfem.rectangle(2, 3)
    nd = hpfem.NedelecDofMap2D(m, 2)
    k = np.array([0.0, 0.7])
    phase = hpfem.bloch_phase(k, [0.0, 1.0])
    assert np.isclose(phase, np.exp(1j * 0.7))
    pair = hpfem.PeriodicPair2D(hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX, [0.0, 1.0], phase)
    c = hpfem.bloch_constraints(nd, [pair])
    assert c.num_constrained == len(nd.dofs_on_tag(hpfem.box_tag.Y_MAX))
    s = hpfem.assemble_maxwell(nd, hpfem.MaxwellForm2D()).stiffness
    reduced, _ = c.reduce(s, np.zeros(nd.num_dofs, dtype=complex))
    assert reduced.shape == (c.num_free, c.num_free)

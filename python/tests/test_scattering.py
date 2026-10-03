"""Scattering pipeline from Python: Mie cylinder cross-section, field evaluation, far field,
angle sweeps with one factorisation, total-field formulation with prescribed traces."""

import numpy as np
import pytest
from conftest import BOX_SIDES, mie_problem

import hpfem


def test_mie_cross_section_matches_series():
    mesh, dofs, problem = mie_problem()
    solution = problem.solve()
    surface = hpfem.Surface2D.around_cells(mesh, 2)
    cs = hpfem.cross_sections(problem, solution, surface, 1.0)
    exact = hpfem.mie_cylinder_scattering_width(6.0, 0.25, 1.5)
    assert abs(cs.scattering - exact) / exact < 1e-3
    assert abs(cs.absorption) < 1e-2 * exact
    assert np.isclose(cs.extinction, cs.scattering + cs.absorption)
    far = hpfem.FarField2D(
        mesh,
        surface,
        hpfem.discrete_field(dofs, solution.unknown),
        problem.setup.omega,
        hpfem.Material.vacuum(),
        8,
    )
    assert abs(far.scattering_cross_section(1.0) - exact) / exact < 3e-2
    pattern = far.pattern([1.0, 0.0])
    assert pattern.shape == (2,) and np.linalg.norm(pattern) > 0


def test_field_evaluation_and_estimate():
    mesh, dofs, problem = mie_problem(n=2, p=2)
    solution = problem.solve()
    locator = hpfem.PointLocator2D(mesh)
    x = [0.5, 0.1]
    total = problem.total_field(solution, locator, x)
    scattered = problem.scattered_field(solution, locator, x)
    incident = problem.setup.incident.value(x)
    assert np.allclose(total, scattered + incident)
    assert problem.total_field(solution, locator, [5.0, 0.0]) is None
    hit = locator.locate(x)
    assert np.allclose(problem.total_field(solution, hit.cell, hit.xi), total)
    estimate = problem.estimate(solution)
    assert estimate.indicators.shape == (mesh.num_cells,)
    assert estimate.total() > 0 and 0 <= estimate.argmax() < mesh.num_cells
    assert len(problem.interior_cells()) < mesh.num_cells
    marked = hpfem.dorfler_marking(estimate.indicators, 0.3)
    assert 0 < len(marked) < mesh.num_cells


def test_plane_wave_sweep_reuses_the_factorisation():
    _, dofs, problem = mie_problem(n=2, p=2)
    k = 6.0
    reference = problem.solve()
    operator = hpfem.ScatteringOperator2D(problem)
    again = operator.solve()
    assert np.allclose(again.unknown, reference.unknown, atol=1e-10)
    solutions = hpfem.plane_wave_sweep(
        problem, [[k, 0.0], [0.0, k]], lambda kk: np.array([-kk[1], kk[0]]) / k
    )
    assert len(solutions) == 2
    assert np.allclose(solutions[0].unknown, reference.unknown, atol=1e-10)
    rotated = hpfem.Scattering2D(dofs, _rotated_setup(problem.setup, k)).solve()
    assert np.allclose(solutions[1].unknown, rotated.unknown, atol=1e-10)
    # the batched solve of the operator (one solve_many) gives the same solutions
    incidents = [problem.setup.incident, _rotated_setup(problem.setup, k).incident]
    batched = operator.solve_many(incidents)
    assert len(batched) == 2
    assert np.allclose(batched[0].unknown, reference.unknown, atol=1e-10)
    assert np.allclose(batched[1].unknown, rotated.unknown, atol=1e-10)
    assert operator.solve_many([]) == []


def _rotated_setup(setup, k):
    rotated = hpfem.ScatteringSetup2D()
    rotated.omega = setup.omega
    rotated.materials = setup.materials
    rotated.formulation = setup.formulation
    rotated.pml = setup.pml
    rotated.pec_tags = setup.pec_tags
    rotated.incident = hpfem.plane_wave([-1.0, 0.0], [0.0, k])
    return rotated


def test_total_field_with_prescribed_trace_reproduces_a_plane_wave():
    mesh = hpfem.rectangle(4, 4)
    dofs = hpfem.NedelecDofMap2D(mesh, 3)
    setup = hpfem.ScatteringSetup2D()
    k = 2.0
    setup.omega = k * hpfem.constants.c0
    setup.incident = hpfem.plane_wave([0.6, -0.8], k * np.array([0.8, 0.6]))
    setup.incident_tags = BOX_SIDES
    problem = hpfem.Scattering2D(dofs, setup)
    solution = problem.solve()
    error = problem.error(solution, setup.incident)
    assert error.l2 / error.l2_norm < 1e-3
    with pytest.raises(ValueError):
        bad = hpfem.ScatteringSetup2D()
        bad.omega = 0.0
        hpfem.Scattering2D(dofs, bad)


def test_current_source_callback_assembles():
    mesh = hpfem.rectangle(3, 3)
    dofs = hpfem.NedelecDofMap2D(mesh, 2)
    setup = hpfem.ScatteringSetup2D()
    setup.omega = 1.0 * hpfem.constants.c0
    setup.pec_tags = BOX_SIDES
    setup.current = lambda x: np.array([x[1], 1j])
    problem = hpfem.Scattering2D(dofs, setup)
    system = problem.assemble()
    assert np.linalg.norm(system.rhs) > 0
    assert problem.solve().unknown.shape == (dofs.num_dofs,)

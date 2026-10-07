"""Progress callbacks, cancellation, timing and the memory estimate (M15 F9)."""

import numpy as np
import pytest

import hpfem


def _conical_problem():
    mesh = hpfem.rectangle(3, 3)
    nd = hpfem.NedelecDofMap2D(mesh, 2)
    h1 = hpfem.DofMap2D(mesh, 2)
    setup = hpfem.ConicalScatteringSetup()
    setup.omega = 2.0 * hpfem.constants.c0
    setup.beta = 0.7
    setup.pec_tags = [hpfem.box_tag.X_MIN, hpfem.box_tag.X_MAX, hpfem.box_tag.Y_MIN,
                      hpfem.box_tag.Y_MAX]  # fmt: skip
    k = np.array([1.0, 1.0, 0.7])
    setup.incident = hpfem.conical_plane_wave(
        hpfem.conical_polarisation(k, np.array([1.0, 0.0, 0.0]), hpfem.Polarisation.P), k
    )
    return mesh, nd, h1, setup


def test_progress_events_and_timing_of_the_conical_solve():
    mesh, nd, h1, setup = _conical_problem()
    events = []

    def progress(event):
        events.append((event.phase, event.step, event.num_steps, event.seconds))
        return True

    setup.progress = progress
    solution = hpfem.ConicalScattering(nd, h1, setup).solve()
    assert [e[0] for e in events] == [
        "assembly", "constraints", "factorisation", "solve", "post", "done",
    ]  # fmt: skip
    assert events[-1][1] == events[-1][2] == 5
    assert all(events[i][3] <= events[i + 1][3] for i in range(len(events) - 1))
    timing = solution.timing
    assert isinstance(timing, dict)
    assert set(timing) == {"assembly", "constraints", "factorisation", "solve", "post", "total"}
    assert timing["total"] >= timing["factorisation"]


def test_cancellation_raises_cancelled():
    mesh, nd, h1, setup = _conical_problem()
    setup.progress = lambda event: event.phase != "solve"
    with pytest.raises(hpfem.Cancelled, match="solve"):
        hpfem.ConicalScattering(nd, h1, setup).solve()
    # Cancelled is a RuntimeError, so generic handlers catch it
    with pytest.raises(RuntimeError):
        hpfem.ConicalScattering(nd, h1, setup).solve()


def test_in_plane_progress_and_timing():
    mesh = hpfem.square_with_disc(3, 0.25, 1.0, 2.0, 2)
    dofs = hpfem.NedelecDofMap2D(mesh, 2)
    setup = hpfem.ScatteringSetup2D()
    setup.omega = 3.0 * hpfem.constants.c0
    setup.materials.set(2, hpfem.Material.vacuum())
    setup.incident = hpfem.plane_wave([0.0, 1.0], [3.0, 0.0])
    setup.formulation = hpfem.Formulation.SCATTERED_FIELD
    setup.incident_tags = [hpfem.box_tag.X_MIN, hpfem.box_tag.X_MAX, hpfem.box_tag.Y_MIN,
                           hpfem.box_tag.Y_MAX]  # fmt: skip
    phases = []
    setup.progress = lambda event: phases.append(event.phase) or True
    solution = hpfem.Scattering2D(dofs, setup).solve()
    assert phases[0] == "assembly" and phases[-1] == "done"
    assert solution.timing["total"] > 0


def test_estimate_memory_matches_the_maps_and_the_factorisation():
    mesh = hpfem.rectangle(16, 16)
    estimate = hpfem.estimate_memory(mesh, 2, hpfem.DirectSolverBackend.SPARSE_LU)
    nd = hpfem.NedelecDofMap2D(mesh, 2)
    assert 0 < estimate.dofs < nd.num_dofs  # condensed
    full = hpfem.estimate_memory(nd, hpfem.DirectSolverBackend.SPARSE_LU, False)
    assert full.dofs == nd.num_dofs
    assert full.total_bytes == full.matrix_bytes + full.factor_bytes > 0
    assert "DoFs" in full.describe() and "SparseLU" in repr(full)
    conical = hpfem.estimate_memory(mesh, 2, hpfem.DirectSolverBackend.SPARSE_LU, True)
    assert conical.dofs == nd.num_dofs + hpfem.DofMap2D(mesh, 2).num_dofs
    # against the real factorisation of the Maxwell operator
    form = hpfem.MaxwellForm2D()
    system = hpfem.assemble_maxwell_operator(nd, lambda c: form, 40.0, 0)
    assert abs(full.matrix_nonzeros / system.matrix.nnz - 1) < 0.05
    solver = hpfem.make_direct_solver(hpfem.DirectSolverBackend.SPARSE_LU)
    assert solver.factor_entries == -1
    solver.factorize(system.matrix)
    assert 0.6 < full.factor_entries / solver.factor_entries < 1.6
    assert hpfem.format_bytes(2_500_000) == "2 MB"
    with pytest.raises(ValueError):
        hpfem.estimate_memory(mesh, 0)

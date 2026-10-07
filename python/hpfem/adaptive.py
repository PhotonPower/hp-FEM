"""The adaptive loop as a generator (M15 F1): ``adaptive_solve`` runs SOLVE – ESTIMATE –
MARK – DECIDE – REFINE on an :class:`hpfem.AdaptiveMesh2D` and yields one
:class:`AdaptiveStep` per iteration, so a GUI can stream the convergence of its observables
and stop on a tolerance. The problem is supplied as a factory that builds the problem on
the current mesh and orders; the estimate is the energy-norm estimate of the problem or a
goal (DWR) estimate; the stop criterion is the change of the observables between steps
and, with a goal, the estimated goal error.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import hpfem


@dataclass
class AdaptiveStep:
    """One step of :func:`adaptive_solve`."""

    step: int
    dofs: int
    max_order: int
    eta: float
    """energy-norm estimate (``problem.estimate(solution).total()``)."""
    observables: dict[str, Any]
    """the values returned by the factory's ``observe`` callable."""
    goal_value: complex | None = None
    goal_error: float | None = None
    """|estimated error of the goal| (``GoalEstimate.error``) when a goal is given."""
    change: dict[str, float] = field(default_factory=dict)
    """|observable − previous observable| per real-valued observable."""
    converged: bool = False
    problem: Any = None
    solution: Any = None
    mesh: Any = None
    orders: np.ndarray | None = None


@dataclass
class _State:
    orders: np.ndarray
    predicted: np.ndarray


def _equalise_periodic_orders(adaptive, orders, periodic_pairs, tolerance=1e-6):
    """Paired face cells of the Bloch faces get the larger of the two orders (the Bloch
    constraints pair the facet DoFs one to one until the non-matching coupling exists)."""
    mesh = adaptive.mesh
    for master, slave, shift in periodic_pairs:
        shift = np.asarray(shift, dtype=float)
        cells_by_mid = {}
        for f in mesh.facets_with_tag(master):
            v = mesh.facet_vertices(f)
            mid = 0.5 * (np.asarray(mesh.vertex(int(v[0]))) + np.asarray(mesh.vertex(int(v[1]))))
            key = tuple(np.round((mid + shift) / (tolerance * np.linalg.norm(shift))).astype(int))
            cells_by_mid[key] = int(mesh.facet_cells(f)[0])
        for f in mesh.facets_with_tag(slave):
            v = mesh.facet_vertices(f)
            mid = 0.5 * (np.asarray(mesh.vertex(int(v[0]))) + np.asarray(mesh.vertex(int(v[1]))))
            key = tuple(np.round(mid / (tolerance * np.linalg.norm(shift))).astype(int))
            a = cells_by_mid.get(key)
            if a is None:
                continue
            b = int(mesh.facet_cells(f)[0])
            p = max(orders[a], orders[b])
            orders[a] = p
            orders[b] = p
    return orders


def adaptive_solve(
    adaptive,
    factory: Callable[[Any, np.ndarray], Any],
    *,
    observe: Callable[[Any, Any], dict[str, Any]] | None = None,
    goal: Callable[[Any, Any], Any] | None = None,
    tolerance: float | None = None,
    max_dofs: int = 200_000,
    max_steps: int = 50,
    initial_order: int = 1,
    theta: float = 0.5,
    periodic_pairs=(),
    keep: bool = False,
) -> Iterator[AdaptiveStep]:
    """Generator of the hp-adaptive loop on ``adaptive`` (an ``AdaptiveMesh2D``).

    ``factory(mesh, orders)`` returns a solved problem as ``(problem, solution)`` (any class
    with ``estimate(solution)`` and ``free_dofs``, e.g. ``ConicalScattering`` or
    ``Scattering2D``). ``observe(problem, solution)`` returns the observables of a step as a
    dict (reflectances, fluxes, ...). ``goal(problem, solution)`` returns a ``GoalEstimate``
    (``conical_dwr_estimate`` / ``dwr_estimate``); with it the goal indicators drive the
    marking and ``goal_error`` is reported. The loop stops (``converged = True``) when every
    real-valued observable changed by less than ``tolerance`` since the previous step and, with
    a goal, the estimated goal error is below ``tolerance``; it always stops at ``max_dofs`` or
    ``max_steps``. ``periodic_pairs`` are ``(master_tag, slave_tag, shift)`` of Bloch faces
    whose paired cells keep equal orders (``AdaptiveMesh2D.set_periodic`` mirrors their
    refinement). With ``keep`` the step carries the problem, solution, mesh and orders.
    """
    orders = np.full(adaptive.mesh.num_cells, initial_order, dtype=int)
    predicted = np.zeros(0)
    previous: dict[str, Any] = {}
    for step in range(max_steps):
        mesh = adaptive.mesh
        problem, solution = factory(mesh, orders)
        estimate = problem.estimate(solution)
        goal_estimate = goal(problem, solution) if goal is not None else None
        observables = observe(problem, solution) if observe is not None else {}
        change = {}
        for name, value in observables.items():
            if name in previous and np.isscalar(value) and np.isrealobj(value):
                change[name] = float(abs(value - previous[name]))
        dofs = int(len(problem.free_dofs))
        goal_error = float(abs(goal_estimate.error)) if goal_estimate is not None else None
        converged = (
            tolerance is not None
            and step > 0
            and all(c < tolerance for c in change.values())
            and (goal_error is None or goal_error < tolerance)
        )
        out = AdaptiveStep(
            step=step,
            dofs=dofs,
            max_order=int(orders.max()),
            eta=float(estimate.total()),
            observables=observables,
            goal_value=goal_estimate.value if goal_estimate is not None else None,
            goal_error=goal_error,
            change=change,
            converged=converged,
        )
        if keep:
            out.problem, out.solution, out.mesh, out.orders = problem, solution, mesh, orders.copy()
        yield out
        if converged or dofs > max_dofs:
            return
        previous = dict(observables)
        indicators = goal_estimate.indicators if goal_estimate is not None else estimate.indicators
        marked = hpfem.dorfler_marking(indicators, theta)
        decision = hpfem.hp_decide_by_prediction(indicators, predicted, marked)
        hp = hpfem.hp_refine(adaptive, orders.tolist(), decision.h_marked, decision.p_marked)
        predicted = hpfem.predict_indicators(indicators, orders.tolist(), hp)
        orders = np.asarray(hp.orders, dtype=int)
        if periodic_pairs:
            orders = _equalise_periodic_orders(adaptive, orders, periodic_pairs)

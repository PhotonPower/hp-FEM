"""Parameter sweeps spread over processes (M15 F8).

A direct solver scales poorly with threads beyond a few cores, while the points of a sweep
are independent: ``solve_sweep`` runs ``task(value)`` for every value in a pool of worker
processes, each pinned to a single hpfem thread, and returns the results in order. The
task must be a module-level function (the workers import it by name) and must return
something picklable (efficiencies, fields as NumPy arrays), not a solver object. Inside
the task, build the problem from the value — or from module-level state the workers set up
on first use — and keep the heavy objects out of the return value.

Example::

    def point(wavelength):              # at module level
        problem = build(wavelength)      # mesh, maps, setup ...
        solution = problem.solve()
        return orders(problem, solution)

    results = hpfem.sweep.solve_sweep(point, wavelengths, processes=4)

For a single process the threads of hpfem stay as they are and the loop runs in place; a
``ConicalSweep`` (affine operator, one analysis) is the better tool when one process should
run the whole sweep.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable, Sequence
from typing import Any

import hpfem


def _initialise_worker(threads: int) -> None:
    hpfem.set_num_threads(threads)


def solve_sweep(
    task: Callable[[Any], Any],
    values: Sequence[Any],
    processes: int | None = None,
    threads_per_process: int = 1,
    chunksize: int = 1,
) -> list[Any]:
    """Runs ``task`` for every value in ``processes`` worker processes.

    ``processes`` defaults to the number of CPUs; ``processes=1`` runs the loop in the
    calling process without a pool (useful for debugging and for tasks that are not
    picklable). ``threads_per_process`` sets ``hpfem.set_num_threads`` in every worker.
    """
    values = list(values)
    if processes is None:
        processes = os.cpu_count() or 1
    if processes < 1 or threads_per_process < 1 or chunksize < 1:
        raise ValueError("solve_sweep: processes, threads_per_process and chunksize must be >= 1")
    if processes == 1 or len(values) <= 1:
        return [task(value) for value in values]
    processes = min(processes, len(values))
    context = multiprocessing.get_context("spawn")
    with context.Pool(
        processes, initializer=_initialise_worker, initargs=(threads_per_process,)
    ) as pool:
        return pool.map(task, values, chunksize=chunksize)

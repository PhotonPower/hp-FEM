"""The evaluator contract of optimisation studies (M16 S2, ADR-0012 §2).

An *evaluator* maps a point of the design space (a dict of parameter values in SI units,
keyed by parameter name) to an :class:`Evaluation`: a vector of ``m`` **real** observables,
optionally their Jacobian ``(m, n)`` with columns in parameter order, an estimate of the
discretisation error per observable (a magnitude, not a standard deviation), the cost in
seconds, the fidelity it was computed at and the reference mesh it used. Complex quantities
(amplitudes, eigenvalues) enter as two observables, real and imaginary part
(:func:`split_complex`). A failed evaluation carries ``status="failed"`` and NaN values
instead of raising, so that a study survives a bad point.

:class:`FunctionEvaluator` turns a plain Python function into an evaluator and normalises
what it returns (:func:`as_evaluation`): a number, an array, a ``(values, jacobian)`` or
``(values, jacobian, error)`` tuple, a dict or an :class:`Evaluation`.
"""

from __future__ import annotations

import inspect
import math
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

import hpfem

STATUSES = ("ok", "failed", "cancelled")
"""Possible values of :attr:`Evaluation.status`."""


def _real_array(value, what: str, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise TypeError(
            f"Evaluation: complex {what}; the contract is real-valued, split complex quantities "
            "into real and imaginary part (hpfem.opt.split_complex)"
        )
    array = np.array(array, dtype=float)
    if ndim == 1:
        array = np.atleast_1d(array)
    if array.ndim != ndim:
        raise ValueError(f"Evaluation: {what} must have {ndim} dimension(s), got {array.shape}")
    return array


@dataclass
class Evaluation:
    """One evaluation of an evaluator at one point (ADR-0012 §2).

    ``values`` has shape ``(m,)`` (real observables); ``jacobian`` is ``None`` or
    ``(m, n)`` with ``jacobian[i, j] = d values[i] / d params[j]`` in the evaluator's parameter
    order and SI units (a gradient ``(n,)`` is accepted for ``m = 1``; columns of integer or
    categorical parameters are meaningless and may be NaN); ``error`` is ``None`` or ``(m,)``,
    the (DWR) estimate of ``|error|`` per observable; ``cost`` is the wall time in seconds
    (NaN until measured); ``fidelity`` describes the discretisation (e.g. ``{"order": 4}``),
    ``mesh_id`` the reference mesh (it changes on a remesh); ``status`` is ``"ok"``,
    ``"failed"`` (NaN values, the reason in ``meta["error"]``) or ``"cancelled"``; ``meta``
    holds anything else that is JSON-serialisable (DoFs, backend, timings). Complex values are
    rejected; see :func:`split_complex`."""

    params: dict[str, Any]
    values: np.ndarray
    jacobian: np.ndarray | None = None
    error: np.ndarray | None = None
    cost: float = math.nan
    fidelity: dict = field(default_factory=dict)
    mesh_id: int = 0
    status: str = "ok"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.params = dict(self.params)
        self.values = _real_array(self.values, "values", 1)
        m = len(self.values)
        if self.jacobian is not None:
            jac = _real_array(self.jacobian, "jacobian", np.ndim(self.jacobian))
            if jac.ndim == 1 and m == 1:
                jac = jac[None, :]
            if jac.ndim != 2 or jac.shape[0] != m:
                raise ValueError(
                    f"Evaluation: jacobian of shape {jac.shape} for {m} values, expected (m, n) "
                    "(a gradient (n,) only for a single value)"
                )
            self.jacobian = jac
        if self.error is not None:
            err = _real_array(self.error, "error", 1)
            if err.shape != (m,):
                raise ValueError(f"Evaluation: error of shape {err.shape} for {m} values")
            self.error = err
        self.cost = float(self.cost)
        self.fidelity = dict(self.fidelity or {})
        self.mesh_id = int(self.mesh_id)
        if self.status not in STATUSES:
            raise ValueError(f"Evaluation: status {self.status!r}, use one of {STATUSES}")
        self.meta = dict(self.meta or {})

    @property
    def ok(self) -> bool:
        """``status == "ok"``."""
        return self.status == "ok"

    @property
    def gradient(self) -> np.ndarray | None:
        """The gradient ``(n,)`` of a single observable (``None`` without a Jacobian)."""
        if self.jacobian is None:
            return None
        if self.jacobian.shape[0] != 1:
            raise ValueError("Evaluation.gradient: more than one observable, use jacobian")
        return self.jacobian[0]

    @classmethod
    def failure(cls, params: Mapping[str, Any], error: BaseException | str, num_values: int = 0,
                status: str = "failed", **kwargs) -> Evaluation:  # fmt: skip
        """A failed (or cancelled) evaluation with ``num_values`` NaN values; the message of
        ``error`` goes to ``meta["error"]``, the traceback of an exception to
        ``meta["traceback"]``."""
        meta = dict(kwargs.pop("meta", {}) or {})
        if isinstance(error, BaseException):
            meta["error"] = f"{type(error).__name__}: {error}"
            meta["traceback"] = "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )
        else:
            meta["error"] = str(error)
        return cls(params, np.full(num_values, np.nan), status=status, meta=meta, **kwargs)


@runtime_checkable
class Evaluator(Protocol):
    """What a study evaluates (ADR-0012 §2): ``parameters`` (objects with a ``name``, in the
    order of the Jacobian columns), ``observables`` (names, length m) and a call
    ``evaluator(params, *, jacobian=False, fidelity=None, cancel=None) -> Evaluation`` with
    ``params`` a dict of SI values. ``cancel()`` may be polled during a long evaluation; a
    true value should end it with ``status="cancelled"`` (or raise ``hpfem.Cancelled``).
    Optional attributes used by :class:`hpfem.opt.Study`: ``name`` (str) and ``settings``
    (a JSON-serialisable dict of everything that changes the results — order, mesh, solver
    options; its hash is part of the cache key)."""

    parameters: list
    observables: list[str]

    def __call__(
        self,
        params: Mapping[str, Any],
        *,
        jacobian: bool = False,
        fidelity: dict | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> Evaluation: ...


def complex_names(names: Sequence[str]) -> list[str]:
    """Observable names of complex quantities after :func:`split_complex`:
    ``["a"] -> ["a.re", "a.im"]``."""
    return [f"{n}.{part}" for n in names for part in ("re", "im")]


def split_complex(values, jacobian=None) -> tuple[np.ndarray, np.ndarray | None]:
    """Complex observables ``(k,)`` as ``2k`` real ones, ``[Re v0, Im v0, Re v1, Im v1, ...]``,
    and their Jacobian ``(k, n)`` with respect to real parameters as ``(2k, n)`` rows
    ``Re dv/dp``, ``Im dv/dp`` in the same order. Real input passes through unchanged."""
    v = np.atleast_1d(np.asarray(values))
    if not np.iscomplexobj(v):
        return v, (None if jacobian is None else np.asarray(jacobian))
    out_v = np.column_stack([v.real, v.imag]).ravel()
    if jacobian is None:
        return out_v, None
    j = np.asarray(jacobian)
    if j.ndim == 1:
        j = j[None, :]
    if j.ndim != 2 or j.shape[0] != len(v):
        raise ValueError(f"split_complex: jacobian {j.shape} for {len(v)} values")
    j = np.stack([j.real, j.imag], axis=1).reshape(2 * len(v), j.shape[1])
    return out_v, j


_DICT_KEYS = {"values", "value", "jacobian", "gradient", "error", "cost", "fidelity", "mesh_id",
              "status", "meta"}  # fmt: skip


def as_evaluation(result, params: Mapping[str, Any], num_params: int | None = None,
                  num_values: int | None = None) -> Evaluation:  # fmt: skip
    """Normalises what an evaluator function returns into an :class:`Evaluation` at ``params``:

    - an :class:`Evaluation` (its ``params`` are filled in when empty);
    - a dict with ``values`` (or ``value``) and optionally ``jacobian`` (or ``gradient``),
      ``error``, ``cost``, ``fidelity``, ``mesh_id``, ``status``, ``meta``;
    - a tuple ``(values, jacobian)`` or ``(values, jacobian, error)``;
    - anything else is the values (a number or an array).

    Complex values (and their Jacobian) are split into real and imaginary part
    (:func:`split_complex`). ``num_params`` / ``num_values``, when given, are checked against
    the Jacobian columns and the number of values (``ValueError``)."""
    if isinstance(result, Evaluation):
        evaluation = result
        if not evaluation.params:
            evaluation.params = dict(params)
    else:
        if isinstance(result, Mapping):
            unknown = set(result) - _DICT_KEYS
            if unknown:
                raise ValueError(f"evaluator result: unknown keys {sorted(unknown)}")
            if "values" in result and "value" in result:
                raise ValueError("evaluator result: give 'values' or 'value', not both")
            kwargs = {k: result[k] for k in ("cost", "fidelity", "mesh_id", "status", "meta")
                      if k in result}  # fmt: skip
            values = result.get("values", result.get("value"))
            if values is None:
                raise ValueError("evaluator result: a dict needs 'values' (or 'value')")
            jacobian = result.get("jacobian", result.get("gradient"))
            error = result.get("error")
        elif isinstance(result, tuple):
            if len(result) not in (2, 3):
                raise ValueError(
                    f"evaluator result: a tuple is (values, jacobian[, error]), got {len(result)}"
                )
            values, jacobian = result[0], result[1]
            error = result[2] if len(result) == 3 else None
            kwargs = {}
        else:
            values, jacobian, error, kwargs = result, None, None, {}
        values, jacobian = split_complex(values, jacobian)
        evaluation = Evaluation(dict(params), values, jacobian, error, **kwargs)
    if evaluation.ok:
        if num_values is not None and len(evaluation.values) != num_values:
            raise ValueError(
                f"evaluator returned {len(evaluation.values)} values for {num_values} observables"
            )
        if (
            num_params is not None
            and evaluation.jacobian is not None
            and evaluation.jacobian.shape[1] != num_params
        ):
            raise ValueError(
                f"evaluator returned a jacobian with {evaluation.jacobian.shape[1]} columns "
                f"for {num_params} parameters"
            )
    return evaluation


def is_cancellation(error: BaseException) -> bool:
    """``hpfem.Cancelled`` (raised by the solvers) or another exception named ``Cancelled``
    (the job runner's)."""
    return isinstance(error, hpfem.Cancelled) or type(error).__name__ == "Cancelled"


class FunctionEvaluator:
    """A Python function as an :class:`Evaluator`.

    ``function(params, **kwargs)`` receives the parameter dict (SI values) — or, with
    ``as_array=True``, a float array in parameter order — and the keywords ``jacobian``,
    ``fidelity`` and ``cancel`` it declares (or all of them with ``**kwargs``). It may return
    anything :func:`as_evaluation` accepts. ``parameters`` is a :class:`hpfem.opt.DesignSpace`
    (kept as ``space``, with its constraints) or a list of parameters (anything with a
    ``name``); ``observables`` the names of the
    values or their number (default: one value). An exception becomes a ``"failed"``
    evaluation (``hpfem.Cancelled`` a ``"cancelled"`` one) with NaN values and the message in
    ``meta["error"]``; the cost is the measured wall time unless the function reports one.
    ``name`` and ``settings`` describe the evaluator in the study header; put into
    ``settings`` everything that changes the results, so that a changed setting never reuses
    cached values."""

    def __init__(self, function: Callable, parameters, observables: Sequence[str] | int = 1, *,
                 name: str | None = None, settings: Mapping | None = None,
                 as_array: bool = False):  # fmt: skip
        self.function = function
        self.space = parameters if hasattr(parameters, "constraints") else None
        """the design space when one was given (a study takes its constraints from it)"""
        self.parameters = list(getattr(parameters, "parameters", parameters))
        self.names = [p.name for p in self.parameters]
        if isinstance(observables, int):
            observables = ["value"] if observables == 1 else [f"y{i}" for i in range(observables)]
        self.observables = [str(o) for o in observables]
        self.name = name or getattr(function, "__qualname__", type(function).__name__)
        self.settings = dict(settings or {})
        self.as_array = bool(as_array)
        try:
            signature = inspect.signature(function)
            accepts_all = any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
            )
            self._keywords = {
                k for k in ("jacobian", "fidelity", "cancel")
                if accepts_all or k in signature.parameters
            }  # fmt: skip
        except (TypeError, ValueError):  # builtins without a signature
            self._keywords = set()

    def __call__(self, params: Mapping[str, Any], *, jacobian: bool = False,
                 fidelity: dict | None = None,
                 cancel: Callable[[], bool] | None = None) -> Evaluation:  # fmt: skip
        """Evaluates the function at ``params`` and returns the normalised evaluation."""
        t0 = time.perf_counter()
        given = {"jacobian": jacobian, "fidelity": fidelity, "cancel": cancel}
        kwargs = {k: v for k, v in given.items() if k in self._keywords}
        try:
            argument = (
                np.array([params[n] for n in self.names], dtype=float)
                if self.as_array
                else dict(params)
            )
            evaluation = as_evaluation(
                self.function(argument, **kwargs), params, len(self.names), len(self.observables)
            )
        except Exception as error:
            status = "cancelled" if is_cancellation(error) else "failed"
            evaluation = Evaluation.failure(params, error, len(self.observables), status)
        if math.isnan(evaluation.cost):
            evaluation.cost = time.perf_counter() - t0
        if fidelity and not evaluation.fidelity:
            evaluation.fidelity = dict(fidelity)
        return evaluation


__all__ = [
    "STATUSES",
    "Evaluation",
    "Evaluator",
    "FunctionEvaluator",
    "as_evaluation",
    "complex_names",
    "is_cancellation",
    "split_complex",
]

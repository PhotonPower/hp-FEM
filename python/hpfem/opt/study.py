"""Design spaces and studies with a JSON-lines store (M16 S2, ADR-0012 §5 and §7).

A :class:`DesignSpace` is an ordered list of parameters — :class:`Continuous` (bounds,
optionally a logarithmic scale), :class:`Integer` and :class:`Categorical` — with
:class:`LinearConstraint` and :class:`NonlinearConstraint` on the parameter values. The S1
parameters (:class:`hpfem.opt.MaterialParameter`, :class:`hpfem.opt.GeometryParameter`) are
accepted as continuous dimensions (name, bounds, scale). Values are SI throughout. The space
validates points, encodes them to the unit hypercube and back (for optimisers and surrogates)
and draws initial designs.

A :class:`Study` evaluates points of the space with an evaluator (:mod:`hpfem.opt.evaluator`)
**sequentially**, caches every evaluation by a canonical key of the point (each value divided
by its scale and rounded to 12 significant digits, values below 1e-12 of the scale to zero,
plus the fidelity and the hash of the evaluator settings), appends every evaluation to a
JSON-lines file (``*.study.jsonl``) and resumes from that file without evaluating again.
Optimisers (S3, S5) drive it by ``propose → evaluate → record``: :meth:`Study.run` (or
:meth:`Study.propose` and :meth:`Study.evaluate_open`), :meth:`Study.evaluate` for one point,
:meth:`Study.checkpoint` for their resumable state.

Store format (one JSON object per line, UTF-8, appended and flushed line by line):

- the header ``{"type": "study", "schema": 1, "hpfem": ..., "space": {...},
  "evaluator": {"name", "settings", "hash", "observables"}, "created": ISO 8601, "meta": {...}}``;
- ``"evaluation"``: the fields of :class:`hpfem.opt.Evaluation` (``params``, ``values``,
  ``jacobian``, ``error``, ``cost``, ``fidelity``, ``mesh_id``, ``status``, ``meta``) plus the
  ``evaluator`` hash and the ``time``;
- ``"proposal"`` (``points``, ``fidelity``, ``jacobian``, ``source``), ``"state"``
  (``method``, ``iteration``, ``evaluations``, ``state``), ``"remesh"`` (``mesh_id``,
  ``params``, ``reason``) and ``"note"`` (``text`` and free fields).

Arrays are nested lists, NaN is ``null``, ±∞ are the strings ``"inf"`` / ``"-inf"``, floats
are written with ``repr`` (they round-trip exactly), complex numbers and complex arrays are
``{"__complex__": [re, im]}``. A truncated last line (a crash during a write) is dropped on
resume.
"""

from __future__ import annotations

import datetime
import hashlib
import inspect
import json
import math
import numbers
import time
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

import hpfem
from hpfem.opt.evaluator import (
    STATUSES,
    Evaluation,
    Evaluator,
    FunctionEvaluator,
    as_evaluation,
    is_cancellation,
)

SCHEMA_VERSION = 1
"""Version of the study store format."""

_TOL = 1e-12  # relative slack of bounds and constraints


class StudyError(ValueError):
    """A study store that cannot be read or does not match the study."""


class EvaluationFailed(RuntimeError):
    """Raised by a study with ``raise_errors=True`` after recording a failed evaluation."""


# --- JSON ------------------------------------------------------------------------------------


def to_jsonable(obj):
    """``obj`` as JSON values (ADR-0012 §5): arrays as nested lists, NaN as ``None``, ±∞ as
    ``"inf"`` / ``"-inf"``, complex scalars and arrays as ``{"__complex__": [re, im]}``,
    tuples as lists, dict keys as strings, NumPy scalars as Python numbers; other objects by
    ``str``."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, numbers.Integral):
        return int(obj)
    if isinstance(obj, numbers.Real):
        x = float(obj)
        if math.isnan(x):
            return None
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return x
    if isinstance(obj, numbers.Complex):
        z = complex(obj)
        return {"__complex__": [to_jsonable(z.real), to_jsonable(z.imag)]}
    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            return {"__complex__": [to_jsonable(obj.real), to_jsonable(obj.imag)]}
        return to_jsonable(obj.tolist())
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _float(x) -> float:
    return math.nan if x is None else float(x)


def _floats(x) -> np.ndarray:
    return np.array(x, dtype=float)


def from_jsonable(obj):
    """Inverse of :func:`to_jsonable` for the tagged complex values (scalars and arrays);
    everything else is returned as read (numeric fields of the store are converted by
    :class:`Study`, so ``None`` / ``"inf"`` become NaN / ∞ there)."""
    if isinstance(obj, dict):
        if set(obj) == {"__complex__"}:
            re, im = obj["__complex__"]
            if isinstance(re, list):
                return _floats(re) + 1j * _floats(im)
            return complex(_float(re), _float(im))
        return {k: from_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [from_jsonable(v) for v in obj]
    return obj


def _canonical(obj) -> str:
    return json.dumps(to_jsonable(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)


def settings_hash(settings: Mapping) -> str:
    """SHA-256 (hex) of the canonical JSON of evaluator settings (sorted keys, no spaces)."""
    return hashlib.sha256(_canonical(settings).encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


# --- parameters ------------------------------------------------------------------------------


def _number(name: str, value) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"parameter {name}: {value!r} is not a real number")
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"parameter {name}: {x!r} is not finite")
    return x


@dataclass(frozen=True)
class Continuous:
    """A real parameter in ``[lower, upper]`` (SI units, ``unit`` is a label such as ``"m"``
    or ``"rad"``). With ``log=True`` (needs ``lower > 0``) the unit-cube coordinate is
    logarithmic. ``scale`` is the typical magnitude used to normalise the cache key; default
    ``upper - lower``."""

    name: str
    lower: float
    upper: float
    log: bool = False
    unit: str = ""
    scale: float | None = None
    kind: ClassVar[str] = "continuous"

    def __post_init__(self):
        lower, upper = float(self.lower), float(self.upper)
        if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
            raise ValueError(
                f"parameter {self.name}: bounds [{lower!r}, {upper!r}] must be finite with "
                "lower < upper"
            )
        if self.log and lower <= 0:
            raise ValueError(f"parameter {self.name}: a logarithmic scale needs lower > 0")
        if self.scale is not None and not float(self.scale) > 0:
            raise ValueError(f"parameter {self.name}: scale must be positive")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "log", bool(self.log))
        if self.scale is not None:
            object.__setattr__(self, "scale", float(self.scale))

    @classmethod
    def from_parameter(cls, parameter, lower: float | None = None, upper: float | None = None,
                       log: bool = False, unit: str = "") -> Continuous:  # fmt: skip
        """A continuous dimension from an object with ``name``, ``lower``, ``upper`` and
        optionally ``scale`` (the S1 :class:`~hpfem.opt.MaterialParameter` /
        :class:`~hpfem.opt.GeometryParameter`); ``lower`` / ``upper`` override its bounds
        (needed when they are infinite)."""
        return cls(
            parameter.name,
            parameter.lower if lower is None else lower,
            parameter.upper if upper is None else upper,
            log,
            unit,
            getattr(parameter, "scale", None),
        )

    @property
    def typical(self) -> float:
        """The normalisation of the cache key: ``scale`` or ``upper - lower``."""
        return self.scale if self.scale is not None else self.upper - self.lower

    def check(self, value) -> float:
        """``value`` as a float inside the bounds (a relative slack of 1e-12 of the range is
        clipped); raises ``ValueError``."""
        x = _number(self.name, value)
        slack = _TOL * (self.upper - self.lower)
        if x < self.lower - slack or x > self.upper + slack:
            raise ValueError(
                f"parameter {self.name}: {x!r} outside [{self.lower!r}, {self.upper!r}]"
            )
        return min(max(x, self.lower), self.upper)

    def encode(self, value) -> float:
        """Unit-cube coordinate of ``value`` (linear or logarithmic)."""
        x = float(value)
        if self.log:
            return math.log(x / self.lower) / math.log(self.upper / self.lower)
        return (x - self.lower) / (self.upper - self.lower)

    def decode(self, u: float) -> float:
        """The value at unit-cube coordinate ``u`` (clipped to [0, 1])."""
        u = min(max(float(u), 0.0), 1.0)
        if self.log:
            x = self.lower * math.exp(u * math.log(self.upper / self.lower))
        else:
            x = self.lower + u * (self.upper - self.lower)
        return min(max(x, self.lower), self.upper)

    def to_number(self, value) -> float:
        return float(value)

    def from_number(self, x: float) -> float:
        return float(x)

    def key(self, value) -> float:
        """``value / typical`` rounded to 12 significant digits (ADR-0012 §5), counted from
        the scale for values below it: ``|value| < 1e-12 · typical`` gives 0, so that round-off
        around zero does not split the cache."""
        x = float(value) / self.typical
        exponent = max(math.floor(math.log10(abs(x))), 0) if x != 0.0 else 0
        return round(x, 11 - exponent) + 0.0  # + 0.0: no -0.0

    def to_dict(self) -> dict:
        return {"kind": self.kind, "name": self.name, "lower": self.lower, "upper": self.upper,
                "log": self.log, "unit": self.unit, "scale": self.scale}  # fmt: skip


@dataclass(frozen=True)
class Integer:
    """An integer parameter in ``[lower, upper]`` (both included). Its unit-cube coordinate
    splits [0, 1] into ``upper - lower + 1`` equal cells, value ``k`` at the cell centre."""

    name: str
    lower: int
    upper: int
    unit: str = ""
    kind: ClassVar[str] = "integer"

    def __post_init__(self):
        lower, upper = self._integral(self.lower), self._integral(self.upper)
        if lower > upper:
            raise ValueError(f"parameter {self.name}: lower {lower} > upper {upper}")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def _integral(self, value) -> int:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"parameter {self.name}: {value!r} is not an integer")
        if isinstance(value, numbers.Integral):
            return int(value)
        x = _number(self.name, value)
        if not x.is_integer():
            raise ValueError(f"parameter {self.name}: {value!r} is not an integer")
        return int(x)

    @property
    def size(self) -> int:
        return self.upper - self.lower + 1

    @property
    def typical(self) -> float:
        return float(max(self.upper - self.lower, 1))

    def check(self, value) -> int:
        k = self._integral(value)
        if not self.lower <= k <= self.upper:
            raise ValueError(f"parameter {self.name}: {k} outside [{self.lower}, {self.upper}]")
        return k

    def encode(self, value) -> float:
        return (int(value) - self.lower + 0.5) / self.size

    def decode(self, u: float) -> int:
        u = min(max(float(u), 0.0), 1.0)
        return self.lower + min(int(math.floor(u * self.size)), self.size - 1)

    def to_number(self, value) -> float:
        return float(value)

    def from_number(self, x: float) -> int:
        return int(round(float(x)))

    def key(self, value) -> int:
        return int(value)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "name": self.name, "lower": self.lower, "upper": self.upper,
                "unit": self.unit}  # fmt: skip


@dataclass(frozen=True)
class Categorical:
    """A parameter taking one of ``choices`` (JSON scalars: strings, numbers, booleans;
    unordered). Its unit-cube coordinate splits [0, 1] into ``len(choices)`` equal cells (an
    ordinal encoding); its number (:meth:`DesignSpace.to_vector`) is the choice index."""

    name: str
    choices: tuple
    kind: ClassVar[str] = "categorical"

    def __post_init__(self):
        choices = tuple(c.item() if isinstance(c, np.generic) else c for c in self.choices)
        if not choices:
            raise ValueError(f"parameter {self.name}: no choices")
        for c in choices:
            ok = isinstance(c, (str, bool, int)) or (isinstance(c, float) and math.isfinite(c))
            if not ok:
                raise ValueError(f"parameter {self.name}: choice {c!r} is not a JSON scalar")
        for i, c in enumerate(choices):
            if any(self._same(c, d) for d in choices[:i]):
                raise ValueError(f"parameter {self.name}: duplicate choice {c!r}")
        object.__setattr__(self, "choices", choices)

    @staticmethod
    def _same(a, b) -> bool:
        return (
            a == b
            and isinstance(a, bool) == isinstance(b, bool)
            and (isinstance(a, str) == isinstance(b, str))
        )

    def index(self, value) -> int:
        """Position of ``value`` in ``choices``; raises ``ValueError``."""
        if isinstance(value, np.generic):
            value = value.item()
        for i, c in enumerate(self.choices):
            if self._same(c, value):
                return i
        raise ValueError(f"parameter {self.name}: {value!r} is not one of {list(self.choices)}")

    def check(self, value):
        return self.choices[self.index(value)]

    def encode(self, value) -> float:
        return (self.index(value) + 0.5) / len(self.choices)

    def decode(self, u: float):
        u = min(max(float(u), 0.0), 1.0)
        return self.choices[min(int(math.floor(u * len(self.choices))), len(self.choices) - 1)]

    def to_number(self, value) -> float:
        return float(self.index(value))

    def from_number(self, x: float):
        i = int(round(float(x)))
        if not 0 <= i < len(self.choices):
            raise ValueError(f"parameter {self.name}: index {i} outside the choices")
        return self.choices[i]

    def key(self, value):
        return self.check(value)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "name": self.name, "choices": list(self.choices)}


Dimension = Continuous | Integer | Categorical


def _dimension(p) -> Dimension:
    if isinstance(p, (Continuous, Integer, Categorical)):
        return p
    if all(hasattr(p, a) for a in ("name", "lower", "upper")):
        return Continuous.from_parameter(p)
    raise TypeError(f"design space: {p!r} is not a parameter (Continuous, Integer, Categorical)")


# --- constraints -----------------------------------------------------------------------------


def _excess(value, lower, upper) -> float:
    v = np.atleast_1d(np.asarray(value, dtype=float))
    lo = np.broadcast_to(np.asarray(lower, dtype=float), v.shape)
    hi = np.broadcast_to(np.asarray(upper, dtype=float), v.shape)
    size = np.abs(v)
    size = np.maximum(size, np.where(np.isfinite(lo), np.abs(lo), 0.0))
    size = np.maximum(size, np.where(np.isfinite(hi), np.abs(hi), 0.0))
    slack = _TOL * size
    if np.any(np.isnan(v)):
        return math.inf
    over = np.maximum(np.maximum(lo - v - slack, v - hi - slack), 0.0)
    return float(over.max()) if over.size else 0.0


def _bound(b):
    return float(b) if np.ndim(b) == 0 else [float(x) for x in np.ravel(b)]


@dataclass
class LinearConstraint:
    """``lower <= Σ coefficients[name] · params[name] <= upper`` on numeric parameters (SI
    values; integers count with their value)."""

    coefficients: Mapping[str, float]
    lower: float = -math.inf
    upper: float = math.inf
    name: str = ""

    def __post_init__(self):
        self.coefficients = {str(k): float(v) for k, v in dict(self.coefficients).items()}
        self.lower, self.upper = float(self.lower), float(self.upper)
        if not self.coefficients:
            raise ValueError(f"LinearConstraint {self.name}: no coefficients")
        if self.lower > self.upper:
            raise ValueError(f"LinearConstraint {self.name}: lower > upper")

    def value(self, params: Mapping[str, Any]) -> float:
        return float(sum(a * float(params[n]) for n, a in self.coefficients.items()))

    def violation(self, params: Mapping[str, Any]) -> float:
        """Amount by which the constraint is violated (0 when satisfied)."""
        return _excess(self.value(params), self.lower, self.upper)

    def to_dict(self) -> dict:
        return {"kind": "linear", "name": self.name, "coefficients": dict(self.coefficients),
                "lower": self.lower, "upper": self.upper}  # fmt: skip


@dataclass
class NonlinearConstraint:
    """``lower <= function(params) <= upper`` with ``function`` taking the parameter dict (SI
    values) and returning a number or an array (bounds broadcast). Meant for cheap
    (geometric) conditions. The function is not stored in the study file, only ``name`` and
    the bounds; a space read back from a file cannot evaluate it."""

    function: Callable[[dict], Any] | None
    lower: Any = -math.inf
    upper: Any = math.inf
    name: str = ""

    def __post_init__(self):
        self.lower, self.upper = _bound(self.lower), _bound(self.upper)

    def value(self, params: Mapping[str, Any]) -> np.ndarray:
        if self.function is None:
            raise StudyError(f"NonlinearConstraint {self.name}: no function (read from a file)")
        return np.atleast_1d(np.asarray(self.function(dict(params)), dtype=float))

    def violation(self, params: Mapping[str, Any]) -> float:
        return _excess(self.value(params), self.lower, self.upper)

    def to_dict(self) -> dict:
        return {"kind": "nonlinear", "name": self.name, "lower": self.lower, "upper": self.upper}


Constraint = LinearConstraint | NonlinearConstraint


# --- design space ----------------------------------------------------------------------------


def _qmc_engine(kind: str, d: int, rng: np.random.Generator):
    from scipy.stats import qmc

    cls = qmc.LatinHypercube if kind == "lhs" else qmc.Sobol
    keyword = "rng" if "rng" in inspect.signature(cls).parameters else "seed"
    return cls(d, **{keyword: rng})


class DesignSpace:
    """An ordered list of parameters with constraints.

    ``parameters`` are :class:`Continuous`, :class:`Integer`, :class:`Categorical` or S1
    parameters (anything with ``name``, finite ``lower`` / ``upper`` and optionally
    ``scale``, taken as continuous); ``constraints`` are :class:`LinearConstraint` /
    :class:`NonlinearConstraint`. A point is a dict ``{name: value}`` in SI units; the
    vector forms (:meth:`to_vector`, :meth:`encode`) follow the parameter order."""

    def __init__(self, parameters: Iterable, constraints: Iterable[Constraint] = ()):
        self.parameters: list[Dimension] = [_dimension(p) for p in parameters]
        if not self.parameters:
            raise ValueError("DesignSpace: no parameters")
        names = [p.name for p in self.parameters]
        if len(set(names)) != len(names):
            raise ValueError(f"DesignSpace: duplicate parameter names {names}")
        self._by_name = {p.name: p for p in self.parameters}
        self.constraints: list[Constraint] = list(constraints)
        for c in self.constraints:
            if isinstance(c, LinearConstraint):
                for n in c.coefficients:
                    if n not in self._by_name:
                        raise ValueError(f"LinearConstraint {c.name}: unknown parameter {n!r}")
                    if isinstance(self._by_name[n], Categorical):
                        raise ValueError(f"LinearConstraint {c.name}: {n!r} is categorical")
            elif not isinstance(c, NonlinearConstraint):
                raise TypeError(f"DesignSpace: {c!r} is not a constraint")

    @property
    def names(self) -> list[str]:
        """Parameter names in order."""
        return [p.name for p in self.parameters]

    def __len__(self) -> int:
        return len(self.parameters)

    def __iter__(self):
        return iter(self.parameters)

    def __getitem__(self, key: str | int) -> Dimension:
        return self.parameters[key] if isinstance(key, int) else self._by_name[key]

    def __repr__(self) -> str:
        return f"DesignSpace({self.names}, {len(self.constraints)} constraints)"

    # validation and constraints

    def validate(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """The point with every value checked against its parameter (type, bounds, choices)
        and in parameter order; raises ``ValueError`` for missing, unknown or invalid values.
        Constraints are not checked (:meth:`check`)."""
        if not isinstance(params, Mapping):
            raise ValueError(f"design space: a point is a dict {{name: value}}, got {params!r}")
        unknown = set(params) - set(self._by_name)
        missing = [n for n in self.names if n not in params]
        if unknown or missing:
            raise ValueError(
                f"design space: unknown parameters {sorted(unknown)}, missing {missing}"
            )
        return {p.name: p.check(params[p.name]) for p in self.parameters}

    def violation(self, params: Mapping[str, Any]) -> float:
        """Largest constraint violation of a valid point (0 when feasible)."""
        return max((c.violation(params) for c in self.constraints), default=0.0)

    def feasible(self, params: Mapping[str, Any]) -> bool:
        """Whether the point is valid and satisfies every constraint."""
        try:
            params = self.validate(params)
        except ValueError:
            return False
        return self.violation(params) == 0.0

    def check(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """:meth:`validate` plus the constraints; raises ``ValueError`` naming the first
        violated constraint."""
        params = self.validate(params)
        for i, c in enumerate(self.constraints):
            v = c.violation(params)
            if v > 0:
                label = c.name or f"#{i}"
                raise ValueError(f"design space: constraint {label} violated by {v:.6g}")
        return params

    # vector forms

    def to_vector(self, params: Mapping[str, Any]) -> np.ndarray:
        """Numeric coordinates in parameter order (SI values; a categorical parameter as its
        choice index)."""
        return np.array([p.to_number(params[p.name]) for p in self.parameters], dtype=float)

    def from_vector(self, x) -> dict[str, Any]:
        """Inverse of :meth:`to_vector` (integers and choice indices are rounded); not
        validated."""
        x = np.asarray(x, dtype=float).ravel()
        if len(x) != len(self):
            raise ValueError(f"design space: vector of length {len(x)} for {len(self)} parameters")
        return {p.name: p.from_number(v) for p, v in zip(self.parameters, x, strict=True)}

    def encode(self, params: Mapping[str, Any]) -> np.ndarray:
        """Coordinates in the unit hypercube ``[0, 1]^d`` (logarithmic for ``log=True``,
        cell centres for integers and choices)."""
        params = self.validate(params)
        return np.array([p.encode(params[p.name]) for p in self.parameters])

    def decode(self, u) -> dict[str, Any]:
        """The point at unit-cube coordinates ``u`` (clipped to [0, 1]); always valid, not
        necessarily feasible."""
        u = np.asarray(u, dtype=float).ravel()
        if len(u) != len(self):
            raise ValueError(f"design space: vector of length {len(u)} for {len(self)} parameters")
        return {p.name: p.decode(v) for p, v in zip(self.parameters, u, strict=True)}

    def bounds(self) -> np.ndarray:
        """``(d, 2)`` bounds of :meth:`to_vector` (a categorical parameter: ``[0, n - 1]``)."""
        out = []
        for p in self.parameters:
            if isinstance(p, Categorical):
                out.append((0.0, float(len(p.choices) - 1)))
            else:
                out.append((float(p.lower), float(p.upper)))
        return np.array(out)

    def linear_constraints(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The linear constraints as ``(A, lower, upper)`` on :meth:`to_vector`, ``A`` of
        shape ``(k, d)`` (e.g. for ``scipy.optimize.LinearConstraint``)."""
        rows = [c for c in self.constraints if isinstance(c, LinearConstraint)]
        a = np.zeros((len(rows), len(self)))
        index = {n: i for i, n in enumerate(self.names)}
        for r, c in enumerate(rows):
            for n, coefficient in c.coefficients.items():
                a[r, index[n]] = coefficient
        return a, np.array([c.lower for c in rows]), np.array([c.upper for c in rows])

    def key(self, params: Mapping[str, Any]) -> list:
        """The canonical cache key of a valid point (ADR-0012 §5): each continuous value
        divided by its scale and rounded to 12 significant digits (:meth:`Continuous.key`),
        integers and choices as they are."""
        return [p.key(params[p.name]) for p in self.parameters]

    def sample(self, n: int, method: str = "lhs", seed=None) -> list[dict[str, Any]]:
        """``n`` feasible points: ``method`` ``"lhs"`` (Latin hypercube), ``"sobol"``
        (scrambled Sobol') or ``"random"`` in the unit cube, decoded, infeasible points
        rejected and replaced. ``seed`` is an int or a ``numpy.random.Generator``."""
        if method not in ("lhs", "sobol", "random"):
            raise ValueError(f"sample: method {method!r}, use 'lhs', 'sobol' or 'random'")
        rng = np.random.default_rng(seed)
        out: list[dict] = []
        drawn = 0
        while len(out) < n:
            missing = n - len(out)
            batch = missing if not self.constraints else max(2 * missing, 16)
            if method == "random":
                u = rng.random((batch, len(self)))
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # Sobol' balance for n != 2^k
                    u = _qmc_engine(method, len(self), rng).random(batch)
            for row in u:
                point = self.decode(row)
                if self.violation(point) == 0.0:
                    out.append(point)
                    if len(out) == n:
                        break
            drawn += batch
            if len(out) < n and drawn > 1000 * n + 1000:
                raise ValueError(
                    f"sample: only {len(out)} of {n} feasible points after {drawn} draws"
                )
        return out

    # serialisation

    def to_dict(self) -> dict:
        """The space as a JSON-ready dict (nonlinear constraints without their function)."""
        return {
            "parameters": [p.to_dict() for p in self.parameters],
            "constraints": [c.to_dict() for c in self.constraints],
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> DesignSpace:
        """Inverse of :meth:`to_dict`; nonlinear constraints come back without a function."""
        parameters = []
        for p in data["parameters"]:
            kind = p["kind"]
            if kind == "continuous":
                scale = p.get("scale")
                parameters.append(Continuous(p["name"], float(p["lower"]), float(p["upper"]),
                                             bool(p.get("log", False)), p.get("unit", ""),
                                             None if scale is None else float(scale)))  # fmt: skip
            elif kind == "integer":
                parameters.append(Integer(p["name"], p["lower"], p["upper"], p.get("unit", "")))
            elif kind == "categorical":
                parameters.append(Categorical(p["name"], tuple(p["choices"])))
            else:
                raise StudyError(f"design space: unknown parameter kind {kind!r}")
        constraints = []
        for c in data.get("constraints", []):
            if c["kind"] == "linear":
                bounds = float(c["lower"]), float(c["upper"])
                constraints.append(LinearConstraint(c["coefficients"], *bounds, c.get("name", "")))
            elif c["kind"] == "nonlinear":
                bounds = c["lower"], c["upper"]  # NonlinearConstraint converts "inf" and lists
                constraints.append(NonlinearConstraint(None, *bounds, c.get("name", "")))
            else:
                raise StudyError(f"design space: unknown constraint kind {c['kind']!r}")
        return cls(parameters, constraints)


# --- the store -------------------------------------------------------------------------------


def _read_store(path: Path, repair: bool) -> tuple[dict, list[dict], bool]:
    """Header, later records and whether a truncated last line was dropped (and, with
    ``repair``, cut from the file)."""
    data = path.read_bytes()
    lines = data.split(b"\n")
    records: list[dict] = []
    offset = 0
    dropped = False
    for i, raw in enumerate(lines):
        last = i == len(lines) - 1
        if last and raw == b"":
            break
        try:
            obj = json.loads(raw.decode("utf-8")) if raw.strip() else None
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            if last:  # crash during the last write
                dropped = True
                if repair:
                    with open(path, "r+b") as f:
                        f.truncate(offset)
                break
            raise StudyError(f"{path}:{i + 1}: not a JSON line ({error})") from error
        offset += len(raw) + 1
        if obj is None:
            continue
        if not isinstance(obj, dict):
            raise StudyError(f"{path}:{i + 1}: not a JSON object")
        if last and repair:  # complete record without its newline
            with open(path, "ab") as f:
                f.write(b"\n")
        records.append(obj)
    if not records or records[0].get("type") != "study":
        raise StudyError(f"{path}: the first line is not a study header")
    header = records[0]
    if header.get("schema") != SCHEMA_VERSION:
        raise StudyError(
            f"{path}: study schema {header.get('schema')!r}, this version reads {SCHEMA_VERSION}"
        )
    return header, records[1:], dropped


@dataclass
class History:
    """Evaluations of a study as arrays (rows in study order): ``params`` (dicts), ``x``
    ``(k, d)`` (:meth:`DesignSpace.to_vector`), ``values`` ``(k, m)``, ``jacobian``
    ``(k, m, d)`` or ``None`` when no row has one, ``error`` ``(k, m)`` or ``None``, ``cost``
    ``(k,)`` seconds, ``status``, ``fidelity``, ``mesh_id`` and ``index`` (position in
    :attr:`Study.evaluations`). Missing entries are NaN."""

    names: list[str]
    observables: list[str]
    params: list[dict]
    x: np.ndarray
    values: np.ndarray
    jacobian: np.ndarray | None
    error: np.ndarray | None
    cost: np.ndarray
    status: list[str]
    fidelity: list[dict]
    mesh_id: np.ndarray
    index: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    def __len__(self) -> int:
        return len(self.params)


class Study:
    """Sequential evaluation of a design space with a cache and a JSON-lines store
    (ADR-0012 §5, §7).

    ``evaluator`` is an :class:`~hpfem.opt.Evaluator` or a plain function (wrapped by
    :class:`~hpfem.opt.FunctionEvaluator`, then ``space`` is required). ``space`` defaults to
    the evaluator's parameters; its names must match them in order. ``path`` is the store
    (``*.study.jsonl``); ``None`` keeps the study in memory. An existing non-empty store is
    resumed (``resume=True``): the cache, proposals, remeshes, notes and the last optimiser
    ``state`` are rebuilt and nothing is evaluated again; its design space must equal
    ``space`` (``StudyError`` otherwise). Evaluator settings that changed since are recorded
    in a note; their hash is part of the cache key, so old values are not reused for them.

    ``emit(event)`` receives a JSON-ready dict per event (``event`` = ``"study"`` when
    opened, ``"proposal"``, ``"evaluation"`` with the stored record plus ``index``,
    ``cached`` and, inside :meth:`run`, ``i`` / ``n``; ``"remesh"``, ``"state"``, ``"note"``,
    ``"cancelled"``). ``cancel()`` is polled before every evaluation and passed to the
    evaluator; when it returns true, or an evaluation comes back ``"cancelled"``, the study
    raises ``hpfem.Cancelled`` with the store consistent (the open proposals are evaluated
    after a resume, :meth:`evaluate_open`).

    A failed evaluation (``status="failed"`` or an exception of the evaluator) is recorded
    with NaN values and its message in ``meta["error"]``, cached, and returned;
    ``raise_errors=True`` raises :class:`EvaluationFailed` after recording it,
    ``retry_failed=True`` evaluates cached failures again. Cancelled evaluations are recorded
    but never cached. An evaluator that remeshed reports it in ``meta["remesh"] =
    {"mesh_id": ..., "reason": ...}``; the study writes a ``"remesh"`` record for it.

    :meth:`load` opens a store read-only for inspection (no evaluator needed)."""

    def __init__(self, evaluator=None, path: str | Path | None = None,
                 space: DesignSpace | Sequence | None = None, *, resume: bool = True,
                 emit: Callable[[dict], None] | None = None,
                 cancel: Callable[[], bool] | None = None, raise_errors: bool = False,
                 retry_failed: bool = False, meta: Mapping | None = None):  # fmt: skip
        self.path = Path(path) if path is not None else None
        self.emit = emit
        self.cancel = cancel
        self.raise_errors = bool(raise_errors)
        self.retry_failed = bool(retry_failed)
        self.cache_hits = 0
        """number of evaluations answered from the cache"""
        self.remeshes: list[dict] = []
        self.notes: list[dict] = []
        self.state: dict | None = None
        """the last ``"state"`` record (optimiser checkpoint) or ``None``"""
        self._evaluations: list[Evaluation] = []
        self._hashes: list[str] = []
        self._cache: dict[str, int] = {}
        self._proposals: list[dict] = []
        existing = self.path is not None and self.path.exists() and self.path.stat().st_size > 0
        if evaluator is None:
            if not existing:
                raise StudyError("a study without an evaluator is read-only: give a store file")
            self.evaluator = None
            header, records, _ = _read_store(self.path, repair=False)
            self.header = header
            self.space = DesignSpace.from_dict(header["space"]) if space is None else space
            info = dict(header["evaluator"])
            self.observables = list(info.get("observables") or [])
            self.evaluator_info = info
            self._load(records)
            return
        if not isinstance(evaluator, Evaluator):
            if not callable(evaluator):
                raise TypeError(f"Study: {evaluator!r} is not an evaluator or a function")
            if space is None:
                raise ValueError("Study: a plain function needs the design space (space=...)")
            evaluator = FunctionEvaluator(evaluator, space)
        self.evaluator = evaluator
        if space is None:
            space = getattr(evaluator, "space", None) or DesignSpace(evaluator.parameters)
        elif not isinstance(space, DesignSpace):
            space = DesignSpace(space)
        names = [p.name for p in evaluator.parameters]
        if names != space.names:
            raise ValueError(f"Study: evaluator parameters {names} != design space {space.names}")
        self.space = space
        self.observables = [str(o) for o in evaluator.observables]
        settings = to_jsonable(dict(getattr(evaluator, "settings", None) or {}))
        self.evaluator_info = {
            "name": str(getattr(evaluator, "name", type(evaluator).__name__)),
            "settings": settings,
            "hash": settings_hash(settings),
            "observables": self.observables,
        }
        if existing:
            if not resume:
                raise FileExistsError(f"Study: {self.path} exists (resume=False)")
            header, records, dropped = _read_store(self.path, repair=True)
            if _canonical(header["space"]) != _canonical(space.to_dict()):
                raise StudyError(f"{self.path}: the stored design space differs from this study")
            self.header = header
            self._last_info = dict(header["evaluator"])
            self._load(records)
            if dropped:
                self.note("dropped a truncated last line on resume")
            if self._last_info.get("hash") != self.hash:
                self.note("evaluator settings changed", evaluator=self.evaluator_info)
        else:
            self.header = {
                "type": "study",
                "schema": SCHEMA_VERSION,
                "hpfem": hpfem.__version__,
                "space": to_jsonable(space.to_dict()),
                "evaluator": self.evaluator_info,
                "created": _now(),
                "meta": to_jsonable(dict(meta or {})),
            }
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._write(self.header)
        self._emit({"event": "study", "path": None if self.path is None else str(self.path),
                    "resumed": existing, "evaluations": len(self),
                    "open": len(self.open_proposals())})  # fmt: skip

    @classmethod
    def load(cls, path: str | Path) -> Study:
        """A study store opened read-only (history, best point; no evaluations)."""
        return cls(None, path)

    @property
    def hash(self) -> str:
        """Hash of the current evaluator settings (part of the cache key)."""
        return self.evaluator_info["hash"]

    # --- loading -----------------------------------------------------------------------------

    def _load(self, records: list[dict]) -> None:
        default_hash = self.header["evaluator"].get("hash", "")
        for record in records:
            kind = record.get("type")
            if kind == "evaluation":
                self._add(self._decode_evaluation(record), record.get("evaluator", default_hash))
            elif kind == "proposal":
                self._proposals.append(self._decode_proposal(record))
            elif kind == "state":
                self.state = from_jsonable(record)
            elif kind == "remesh":
                self.remeshes.append(from_jsonable(record))
            elif kind == "note":
                self.notes.append(from_jsonable(record))
                if isinstance(record.get("evaluator"), dict):
                    self._last_info = dict(record["evaluator"])
            # other record types: written by a newer version, ignored

    def _params(self, raw: Mapping) -> dict:
        try:
            return self.space.validate(raw)
        except ValueError as error:
            raise StudyError(f"{self.path}: stored point {raw!r} is not in the space") from error

    def _decode_evaluation(self, r: Mapping) -> Evaluation:
        jacobian = r.get("jacobian")
        error = r.get("error")
        return Evaluation(
            self._params(r.get("params", {})),
            _floats(r.get("values", [])),
            None if jacobian is None else _floats(jacobian),
            None if error is None else _floats(error),
            cost=_float(r.get("cost")),
            fidelity=from_jsonable(r.get("fidelity") or {}),
            mesh_id=int(r.get("mesh_id", 0)),
            status=str(r.get("status", "ok")),
            meta=from_jsonable(r.get("meta") or {}),
        )

    def _decode_proposal(self, r: Mapping) -> dict:
        return {
            "points": [self._params(p) for p in r.get("points", [])],
            "fidelity": from_jsonable(r.get("fidelity") or {}),
            "jacobian": bool(r.get("jacobian", False)),
            "source": r.get("source"),
        }

    # --- records -----------------------------------------------------------------------------

    def _emit(self, event: dict) -> None:
        if self.emit is not None:
            self.emit(event)

    def _write(self, record: Mapping) -> None:
        if self.path is None:
            return
        line = json.dumps(to_jsonable(record), allow_nan=False, ensure_ascii=False)
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")
            f.flush()

    def _require_writable(self) -> None:
        if self.evaluator is None:
            raise StudyError("this study was opened read-only (Study.load)")

    def _key(self, params: Mapping, fidelity: Mapping, settings_hash_: str) -> str:
        return _canonical([self.space.key(params), dict(fidelity), settings_hash_])

    def _add(self, evaluation: Evaluation, hash_: str) -> int:
        index = len(self._evaluations)
        self._evaluations.append(evaluation)
        self._hashes.append(hash_)
        if evaluation.status in ("ok", "failed"):
            key = self._key(evaluation.params, evaluation.fidelity, hash_)
            old = self._cache.get(key)
            if old is None or evaluation.ok or not self._evaluations[old].ok:
                self._cache[key] = index  # a failure never replaces a success
        return index

    @staticmethod
    def _record(evaluation: Evaluation, hash_: str, stamp: bool = True) -> dict:
        record = {
            "type": "evaluation",
            "params": evaluation.params,
            "values": evaluation.values,
            "jacobian": evaluation.jacobian,
            "error": evaluation.error,
            "cost": evaluation.cost,
            "fidelity": evaluation.fidelity,
            "mesh_id": evaluation.mesh_id,
            "status": evaluation.status,
            "meta": evaluation.meta,
            "evaluator": hash_,
        }
        if stamp:
            record["time"] = _now()
        return to_jsonable(record)

    def note(self, text: str, **data) -> None:
        """Appends a ``"note"`` record (``text`` and JSON-serialisable fields)."""
        self._require_writable()
        record = to_jsonable({"type": "note", "text": str(text), **data, "time": _now()})
        self._write(record)
        self.notes.append(from_jsonable(record))
        if isinstance(record.get("evaluator"), dict):
            self._last_info = dict(record["evaluator"])
        self._emit({"event": "note", **record})

    def checkpoint(self, method: str, iteration: int, state: Mapping) -> None:
        """Appends an optimiser checkpoint (``"state"`` record): the ``method``, its
        ``iteration`` and the JSON-serialisable ``state`` it needs to resume (arrays and
        complex values allowed). After a resume it is :attr:`state`."""
        self._require_writable()
        record = to_jsonable({"type": "state", "method": str(method), "iteration": int(iteration),
                              "evaluations": len(self), "state": dict(state),
                              "time": _now()})  # fmt: skip
        self._write(record)
        self.state = from_jsonable(record)
        self._emit({"event": "state", **record})

    def record_remesh(self, mesh_id: int, params: Mapping[str, Any], reason: str = "") -> None:
        """Appends a ``"remesh"`` record: the study switched to reference mesh ``mesh_id``
        built at ``params`` (ADR-0012 §3)."""
        self._require_writable()
        record = to_jsonable({"type": "remesh", "mesh_id": int(mesh_id), "params": dict(params),
                              "reason": str(reason), "time": _now()})  # fmt: skip
        self._write(record)
        self.remeshes.append(from_jsonable(record))
        self._emit({"event": "remesh", **record})

    # --- evaluation --------------------------------------------------------------------------

    def _cached(self, params: Mapping, jacobian: bool, fidelity: Mapping) -> int | None:
        index = self._cache.get(self._key(params, fidelity, self.hash))
        if index is None:
            return None
        hit = self._evaluations[index]
        if hit.ok:
            return index if (not jacobian or hit.jacobian is not None) else None
        return None if self.retry_failed else index

    def _check_cancel(self, context: Mapping) -> None:
        if self.cancel is not None and self.cancel():
            self._emit({"event": "cancelled", "evaluations": len(self), **context})
            raise hpfem.Cancelled(f"study cancelled after {len(self)} evaluations")

    def _call(self, params: dict, jacobian: bool, fidelity: dict) -> Evaluation:
        t0 = time.perf_counter()
        m = len(self.observables)
        try:
            raw = self.evaluator(params, jacobian=jacobian, fidelity=dict(fidelity) or None,
                                 cancel=self.cancel)  # fmt: skip
            evaluation = as_evaluation(raw, params, len(self.space), m or None)
        except Exception as error:
            status = "cancelled" if is_cancellation(error) else "failed"
            evaluation = Evaluation.failure(params, error, m, status)
        if math.isnan(evaluation.cost):
            evaluation.cost = time.perf_counter() - t0
        evaluation.params = dict(params)
        if evaluation.fidelity != fidelity:
            if evaluation.fidelity:
                evaluation.meta.setdefault("fidelity_reported", evaluation.fidelity)
            evaluation.fidelity = dict(fidelity)
        if not evaluation.ok and len(evaluation.values) != m:
            evaluation.values = np.full(m, np.nan)
            evaluation.jacobian = evaluation.error = None
        return evaluation

    def _evaluate(self, params: dict, jacobian: bool, fidelity: dict, context: Mapping):
        index = self._cached(params, jacobian, fidelity)
        if index is not None:
            self.cache_hits += 1
            record = self._record(self._evaluations[index], self._hashes[index], stamp=False)
            self._emit({"event": "evaluation", **record, "index": index, "cached": True,
                        **context})  # fmt: skip
            return self._evaluations[index]
        self._check_cancel(context)
        evaluation = self._call(params, jacobian, fidelity)
        remesh = evaluation.meta.pop("remesh", None)
        if isinstance(remesh, Mapping):
            self.record_remesh(int(remesh.get("mesh_id", evaluation.mesh_id)), params,
                               str(remesh.get("reason", "")))  # fmt: skip
        record = self._record(evaluation, self.hash)
        self._write(record)
        index = self._add(evaluation, self.hash)
        self._emit({"event": "evaluation", **record, "index": index, "cached": False, **context})
        if evaluation.status == "cancelled":
            self._emit({"event": "cancelled", "evaluations": len(self), **context})
            raise hpfem.Cancelled(f"study cancelled during evaluation {index}")
        if evaluation.status == "failed" and self.raise_errors:
            raise EvaluationFailed(
                f"evaluation {index} at {params} failed: {evaluation.meta.get('error', '')}"
            )
        return evaluation

    def evaluate(self, params: Mapping[str, Any], *, jacobian: bool = False,
                 fidelity: Mapping | None = None) -> Evaluation:  # fmt: skip
        """The evaluation at ``params`` (checked against the space and its constraints,
        ``ValueError`` otherwise): from the cache when the same point (same fidelity, same
        evaluator settings; with a Jacobian if ``jacobian``) was evaluated before, else by
        the evaluator, recorded in the store."""
        self._require_writable()
        return self._evaluate(self.space.check(params), bool(jacobian), dict(fidelity or {}), {})

    def propose(self, points: Iterable[Mapping[str, Any]], *, jacobian: bool = False,
                fidelity: Mapping | None = None,
                source: str | None = None) -> list[dict]:  # fmt: skip
        """Records ``points`` (checked) as a ``"proposal"`` (the ones not in the cache) and
        returns them checked; they stay open until evaluated (:meth:`evaluate_open`)."""
        self._require_writable()
        checked = [self.space.check(p) for p in points]
        fidelity = dict(fidelity or {})
        new, keys = [], set()
        for p in checked:
            key = self._key(p, fidelity, self.hash)
            if key not in keys and self._cached(p, jacobian, fidelity) is None:
                keys.add(key)
                new.append(p)
        if new:
            proposal = {"points": new, "fidelity": fidelity, "jacobian": bool(jacobian),
                        "source": source}  # fmt: skip
            record = to_jsonable({"type": "proposal", **proposal, "time": _now()})
            self._write(record)
            self._proposals.append(proposal)
            self._emit({"event": "proposal", **record})
        return checked

    def open_proposals(self) -> list[dict]:
        """Proposed points not evaluated yet (for the current evaluator settings), as dicts
        ``{"params", "fidelity", "jacobian"}``."""
        out, keys = [], set()
        for proposal in self._proposals:
            for p in proposal["points"]:
                fidelity, jacobian = proposal["fidelity"], proposal["jacobian"]
                key = self._key(p, fidelity, self.hash)
                if key in keys or self._cached(p, jacobian, fidelity) is not None:
                    continue
                keys.add(key)
                out.append({"params": p, "fidelity": dict(proposal["fidelity"]),
                            "jacobian": proposal["jacobian"]})  # fmt: skip
        return out

    def evaluate_open(self) -> list[Evaluation]:
        """Evaluates the open proposals (after a resume), sequentially."""
        self._require_writable()
        todo = self.open_proposals()
        return [
            self._evaluate(t["params"], t["jacobian"], t["fidelity"], {"i": i, "n": len(todo)})
            for i, t in enumerate(todo)
        ]

    def run(self, points: Iterable[Mapping[str, Any]], *, jacobian: bool = False,
            fidelity: Mapping | None = None,
            source: str | None = None) -> list[Evaluation]:  # fmt: skip
        """Proposes ``points`` and evaluates them sequentially (cache hits included); returns
        the evaluations in order. Raises ``hpfem.Cancelled`` when cancelled — the evaluated
        points are stored, the rest stays open."""
        checked = self.propose(points, jacobian=jacobian, fidelity=fidelity, source=source)
        fidelity = dict(fidelity or {})
        n = len(checked)
        return [
            self._evaluate(p, bool(jacobian), fidelity, {"i": i, "n": n})
            for i, p in enumerate(checked)
        ]

    # --- inspection --------------------------------------------------------------------------

    @property
    def evaluations(self) -> list[Evaluation]:
        """Every recorded evaluation in order (including failed and cancelled ones)."""
        return list(self._evaluations)

    @property
    def failed(self) -> list[Evaluation]:
        """The failed evaluations."""
        return [e for e in self._evaluations if e.status == "failed"]

    def __len__(self) -> int:
        return len(self._evaluations)

    def __repr__(self) -> str:
        return (
            f"Study({self.path}, {len(self)} evaluations, {len(self.failed)} failed, "
            f"{len(self.space)} parameters)"
        )

    def history(self, status: str | Iterable[str] | None = "ok",
                fidelity: Mapping | None = None) -> History:  # fmt: skip
        """The evaluations with ``status`` (a status, several, or ``None`` for all) and, if
        given, exactly this ``fidelity``, as arrays (:class:`History`)."""
        if isinstance(status, str):
            status = {status}
        elif status is not None:
            status = set(status)
        if status is not None and not status <= set(STATUSES):
            raise ValueError(f"history: status {sorted(status)}, use {STATUSES}")
        rows = [
            i
            for i, e in enumerate(self._evaluations)
            if (status is None or e.status in status)
            and (fidelity is None or e.fidelity == dict(fidelity))
        ]
        evals = [self._evaluations[i] for i in rows]
        k, d = len(evals), len(self.space)
        m = max([len(e.values) for e in evals] + [len(self.observables)])
        values = np.full((k, m), np.nan)
        x = np.full((k, d), np.nan)
        has_jac = any(e.jacobian is not None for e in evals)
        has_err = any(e.error is not None for e in evals)
        jac = np.full((k, m, d), np.nan) if has_jac else None
        err = np.full((k, m), np.nan) if has_err else None
        for r, e in enumerate(evals):
            x[r] = self.space.to_vector(e.params)
            values[r, : len(e.values)] = e.values
            if has_jac and e.jacobian is not None and e.jacobian.shape[1] == d:
                jac[r, : e.jacobian.shape[0]] = e.jacobian
            if has_err and e.error is not None:
                err[r, : len(e.error)] = e.error
        return History(
            names=self.space.names,
            observables=list(self.observables),
            params=[dict(e.params) for e in evals],
            x=x,
            values=values,
            jacobian=jac,
            error=err,
            cost=np.array([e.cost for e in evals], dtype=float),
            status=[e.status for e in evals],
            fidelity=[dict(e.fidelity) for e in evals],
            mesh_id=np.array([e.mesh_id for e in evals], dtype=int),
            index=np.array(rows, dtype=int),
        )

    def best(self, objective: int | str | Callable[[np.ndarray], float] = 0, *,
             minimize: bool = True,
             fidelity: Mapping | None = None) -> Evaluation | None:  # fmt: skip
        """The successful evaluation with the smallest (``minimize``) or largest objective:
        an observable index, an observable name, or a function of the values vector. NaN
        objectives are skipped; ``None`` if there is no candidate."""
        if isinstance(objective, str):
            if objective not in self.observables:
                raise ValueError(f"best: unknown observable {objective!r} ({self.observables})")
            objective = self.observables.index(objective)
        if isinstance(objective, int):
            column = objective

            def objective(values):
                return values[column]

        best, best_value = None, None
        for e in self._evaluations:
            if not e.ok or (fidelity is not None and e.fidelity != dict(fidelity)):
                continue
            value = float(objective(e.values))
            if math.isnan(value):
                continue
            better = best_value is None or (value < best_value if minimize else value > best_value)
            if better:
                best, best_value = e, value
        return best


__all__ = [
    "SCHEMA_VERSION",
    "Categorical",
    "Continuous",
    "DesignSpace",
    "EvaluationFailed",
    "History",
    "Integer",
    "LinearConstraint",
    "NonlinearConstraint",
    "Study",
    "StudyError",
    "from_jsonable",
    "settings_hash",
    "to_jsonable",
]

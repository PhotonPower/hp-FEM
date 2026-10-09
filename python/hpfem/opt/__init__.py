"""Optimisation, calibration and uncertainty quantification on the discrete sensitivities
(milestone M16, ADR-0012).

Imported explicitly (``import hpfem.opt``), not by ``import hpfem``. So far:
:mod:`hpfem.opt.parameters` — material and geometry parameters of a grating unit cell, the
mesh velocities of the geometry parameters and the morphing of a reference mesh with its
quality guard; :mod:`hpfem.opt.evaluator` — the evaluator contract (:class:`Evaluation`,
:class:`Evaluator`, :class:`FunctionEvaluator`); :mod:`hpfem.opt.study` — design spaces with
constraints and studies with an evaluation cache and a resumable JSON-lines store;
:mod:`hpfem.opt.scatterometry` — the efficiencies of a grating under several measurement
configurations as an evaluator of its geometry and material parameters.
"""

from hpfem.opt.evaluator import (
    Evaluation,
    Evaluator,
    FunctionEvaluator,
    as_evaluation,
    complex_names,
    split_complex,
)
from hpfem.opt.parameters import (
    GeometryParameter,
    MaterialParameter,
    MeshQualityError,
    Morph,
    cell_quality,
    shape_velocity,
    trapezoid_parameters,
)
from hpfem.opt.scatterometry import Configuration, GratingEvaluator
from hpfem.opt.study import (
    Categorical,
    Continuous,
    DesignSpace,
    EvaluationFailed,
    History,
    Integer,
    LinearConstraint,
    NonlinearConstraint,
    Study,
    StudyError,
)

__all__ = [
    "Categorical",
    "Configuration",
    "Continuous",
    "DesignSpace",
    "Evaluation",
    "EvaluationFailed",
    "Evaluator",
    "FunctionEvaluator",
    "GeometryParameter",
    "GratingEvaluator",
    "History",
    "Integer",
    "LinearConstraint",
    "MaterialParameter",
    "MeshQualityError",
    "Morph",
    "NonlinearConstraint",
    "Study",
    "StudyError",
    "as_evaluation",
    "cell_quality",
    "complex_names",
    "shape_velocity",
    "split_complex",
    "trapezoid_parameters",
]

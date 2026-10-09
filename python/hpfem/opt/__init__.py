"""Optimisation, calibration and uncertainty quantification on the discrete sensitivities
(milestone M16, ADR-0012).

Imported explicitly (``import hpfem.opt``), not by ``import hpfem``. So far:
:mod:`hpfem.opt.parameters` — material and geometry parameters of a grating unit cell, the
mesh velocities of the geometry parameters and the morphing of a reference mesh with its
quality guard; :mod:`hpfem.opt.evaluator` — the evaluator contract (:class:`Evaluation`,
:class:`Evaluator`, :class:`FunctionEvaluator`); :mod:`hpfem.opt.study` — design spaces with
constraints and studies with an evaluation cache and a resumable JSON-lines store;
:mod:`hpfem.opt.optimize` — L-BFGS-B, Nelder–Mead and differential evolution driving a study
(:func:`minimize`); :mod:`hpfem.opt.lsq` — Gauss–Newton / Levenberg–Marquardt reconstruction
(:func:`fit`) and the Laplace approximation of the parameter uncertainties (:func:`laplace`);
:mod:`hpfem.opt.scatterometry` — the efficiencies of a grating under several measurement
configurations as an evaluator of its geometry and material parameters; :mod:`hpfem.opt.gp` —
a light Gaussian process (Matérn / squared-exponential ARD kernels, noise, derivative
observations); :mod:`hpfem.opt.bo` — Bayesian optimisation driving a study
(:func:`bayesian_optimize`: expected improvement or lower confidence bound, known and outcome
constraints, gradient-enhanced surrogate), the Pareto front of a study and, with the optional
extra ``opt-bo`` (BoTorch), multi-objective and multi-fidelity Bayesian optimisation.
"""

from hpfem.opt.bo import (
    BOResult,
    OutcomeConstraint,
    ParetoFront,
    ParetoResult,
    bayesian_optimize,
    expected_improvement,
    log_expected_improvement,
    log_probability_of_feasibility,
    lower_confidence_bound,
    multi_fidelity_optimize,
    non_dominated,
    pareto_front,
    pareto_optimize,
)
from hpfem.opt.evaluator import (
    Evaluation,
    Evaluator,
    FunctionEvaluator,
    as_evaluation,
    complex_names,
    split_complex,
)
from hpfem.opt.gp import GaussianProcess, MultiOutputGP
from hpfem.opt.lsq import FitResult, IdentifiabilityWarning, Laplace, fit, laplace
from hpfem.opt.optimize import OptimizeResult, minimize
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
    "BOResult",
    "Categorical",
    "Configuration",
    "Continuous",
    "DesignSpace",
    "Evaluation",
    "EvaluationFailed",
    "Evaluator",
    "FitResult",
    "FunctionEvaluator",
    "GaussianProcess",
    "GeometryParameter",
    "GratingEvaluator",
    "History",
    "IdentifiabilityWarning",
    "Integer",
    "Laplace",
    "LinearConstraint",
    "MaterialParameter",
    "MeshQualityError",
    "Morph",
    "MultiOutputGP",
    "NonlinearConstraint",
    "OptimizeResult",
    "OutcomeConstraint",
    "ParetoFront",
    "ParetoResult",
    "Study",
    "StudyError",
    "as_evaluation",
    "bayesian_optimize",
    "cell_quality",
    "complex_names",
    "expected_improvement",
    "fit",
    "laplace",
    "log_expected_improvement",
    "log_probability_of_feasibility",
    "lower_confidence_bound",
    "minimize",
    "multi_fidelity_optimize",
    "non_dominated",
    "pareto_front",
    "pareto_optimize",
    "shape_velocity",
    "split_complex",
    "trapezoid_parameters",
]

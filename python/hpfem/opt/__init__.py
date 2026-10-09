"""Optimisation, calibration and uncertainty quantification on the discrete sensitivities
(milestone M16, ADR-0012).

Imported explicitly (``import hpfem.opt``), not by ``import hpfem``. So far:
:mod:`hpfem.opt.parameters` — material and geometry parameters of a grating unit cell, the
mesh velocities of the geometry parameters and the morphing of a reference mesh with its
quality guard.
"""

from hpfem.opt.parameters import (
    GeometryParameter,
    MaterialParameter,
    MeshQualityError,
    Morph,
    cell_quality,
    shape_velocity,
    trapezoid_parameters,
)

__all__ = [
    "GeometryParameter",
    "MaterialParameter",
    "MeshQualityError",
    "Morph",
    "cell_quality",
    "shape_velocity",
    "trapezoid_parameters",
]

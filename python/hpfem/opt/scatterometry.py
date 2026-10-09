"""Scatterometry evaluator (M16 S3, ADR-0012 §2, §3): the diffraction efficiencies of a grating
unit cell under several measurement configurations as a function of geometry and material
parameters, with their Jacobian.

The geometry parameters come from a :class:`~hpfem.opt.Morph` (e.g. the CD / height /
side-wall triple of :func:`~hpfem.opt.trapezoid_parameters`): every evaluation moves the
reference mesh (same topology, tags and DoF numbering), so the observables are smooth
functions of the parameters and the Jacobian along the morph velocities is their exact
derivative. Each configuration (wavelength, angles, polarisation) is one
:func:`hpfem.grating.solve` with the kept factorisation and one :func:`hpfem.grating.jacobian`
(shape columns along the velocities, permittivity columns for the material parameters). The
PML boxes and the measurement lines are designed once on the reference mesh and kept, so that
nothing in the discretisation jumps with the parameters. When the quality guard of the morph
trips, an optional ``mesher`` builds a new reference mesh at the current geometry (a new
``mesh_id``, recorded by the study as a ``"remesh"``); without one the evaluation fails.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

import hpfem
from hpfem import grating, units

from .evaluator import Evaluation
from .parameters import MaterialParameter, MeshQualityError, Morph


@dataclass(frozen=True)
class Configuration:
    """One measurement configuration: vacuum ``wavelength`` [m], polar angle ``theta`` and
    azimuth ``phi`` [rad] (as :func:`hpfem.grating.solve`), ``polarisation`` ``"s"`` / ``"p"``
    and the observed ``orders``, pairs ``(side, order)`` with ``side`` ``"R"`` or ``"T"`` (they
    must propagate in the whole parameter range)."""

    wavelength: float
    theta: float
    phi: float = 0.0
    polarisation: str = "s"
    orders: tuple = (("R", 0),)

    def __post_init__(self):
        orders = tuple((str(side), int(order)) for side, order in self.orders)
        for side, _ in orders:
            if side not in ("R", "T"):
                raise ValueError(f"Configuration: side {side!r}, use 'R' or 'T'")
        if not orders:
            raise ValueError("Configuration: no orders observed")
        object.__setattr__(self, "orders", orders)

    @property
    def omega(self) -> float:
        """Angular frequency [rad/s]."""
        return float(units.angular_frequency(wavelength=self.wavelength))

    def label(self, side: str, order: int) -> str:
        """Observable name, e.g. ``"s 405nm 40deg: R0"``."""
        azimuth = f"/{math.degrees(self.phi):g}deg" if self.phi else ""
        return (f"{self.polarisation} {self.wavelength / units.nm:g}nm "
                f"{math.degrees(self.theta):g}deg{azimuth}: {side}{order}")  # fmt: skip


def _describe_material(material) -> object:
    """A deterministic, JSON-friendly description (for the settings hash)."""
    if isinstance(material, hpfem.Material):
        eps, mu = complex(material.eps_r), complex(material.mu_r)
        return {"eps_r": [eps.real, eps.imag], "mu_r": [mu.real, mu.imag]}
    return repr(material)  # dispersive models are dataclasses with a stable repr


def _describe_stack(stack) -> dict:
    regions = [
        _describe_material(stack.material(r)) for r in range(stack.num_layers + 2)
    ]  # the incidence medium, the layers, the substrate
    interfaces = [float(stack.interface(i)) for i in range(stack.num_layers + 1)]
    return {"regions": regions, "interfaces": interfaces}


class GratingEvaluator:
    """Efficiencies of a grating unit cell under ``configurations`` as an evaluator (ADR-0012
    §2) of the geometry parameters of ``morph`` followed by ``material_parameters``.

    ``materials`` and ``stack`` are those of :func:`hpfem.grating.solve` (dispersive models are
    evaluated at every wavelength; a :class:`~hpfem.opt.MaterialParameter` needs a plain
    ``hpfem.Material`` for its tag). ``order`` is the polynomial order (a fidelity
    ``{"order": p}`` overrides it per evaluation), ``pml`` the PML thicknesses
    ``{"top": t, "bottom": t}`` or ``None`` for the design of ``solve`` — either way the boxes
    and the measurement lines of each configuration are fixed on the reference mesh at
    construction. ``mesher(cell) -> mesh`` (e.g. ``meshing.structured_unit_cell`` or the Gmsh
    mesher) remeshes when the morph's quality guard trips. ``solve_options`` go to every
    ``solve`` (``orders_max``, ``solver``, ``extra_quadrature_order``, ``bottom``, ...).

    ``observables`` are the configuration labels; ``values`` the efficiencies, ``jacobian``
    their derivatives (direct or adjoint mode per configuration, whichever needs fewer
    solves), in SI units of the parameters (metres, radians). ``meta`` reports the DoFs, the
    smallest cell-quality ratio of the morphed mesh, the order and the solve time."""

    def __init__(self, morph: Morph, materials: Mapping[int, object], stack,
                 configurations: Sequence[Configuration], *,
                 material_parameters: Sequence[MaterialParameter] = (), order: int = 3,
                 pml: Mapping | None = None, mesher: Callable | None = None,
                 solve_options: Mapping | None = None, name: str = "grating"):  # fmt: skip
        self.morph = morph
        self.materials = dict(materials)
        self.stack = stack
        self.configurations = [
            c if isinstance(c, Configuration) else Configuration(**dict(c)) for c in configurations
        ]
        if not self.configurations:
            raise ValueError("GratingEvaluator: no configurations")
        self.material_parameters = list(material_parameters)
        for p in self.material_parameters:
            if not isinstance(self.materials.get(p.tag), hpfem.Material):
                raise ValueError(
                    f"GratingEvaluator: material parameter {p.name} needs a plain "
                    f"hpfem.Material for tag {p.tag}"
                )
        self.parameters = list(morph.parameters) + self.material_parameters
        self.names = [p.name for p in self.parameters]
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"GratingEvaluator: duplicate parameter names {self.names}")
        self.observables = [c.label(s, o) for c in self.configurations for s, o in c.orders]
        self.order = int(order)
        self.mesher = mesher
        self.options = {"orders_max": 3, **dict(solve_options or {})}
        self.mesh_id = 0
        self.name = name
        self._frames = [self._frame(c, pml) for c in self.configurations]
        self._check_lines()
        self.settings = {
            "order": self.order,
            "configurations": [asdict(c) for c in self.configurations],
            "materials": {str(t): _describe_material(m) for t, m in sorted(self.materials.items())},
            "stack": _describe_stack(stack),
            "mesh": {"cells": int(morph.mesh.num_cells), "vertices": int(morph.mesh.num_vertices)},
            "reference": dict(morph.reference),
            "pml": [list(f["pml"].thickness) for f in self._frames],
            "lines": [[f["cover_line"], f["substrate_line"]] for f in self._frames],
            "options": {k: repr(v) for k, v in sorted(self.options.items())},
        }
        """everything that changes the results (the study hashes it)"""

    def _frame(self, config: Configuration, pml) -> dict:
        """PML box and measurement lines of a configuration on the reference mesh."""
        options = {k: v for k, v in self.options.items() if k in ("orders_max", "bottom")}
        prepared = grating._prepare(
            self.morph.mesh.copy(), self.materials, self.stack, config.polarisation,
            config.theta, config.phi, config.omega, pml=pml,
            bottom=options.get("bottom", "pml"), orders_max=options["orders_max"],
            snap_tolerance=1e-9, pml_target=1e-6, pml_wavelengths=0.5, cover_line=None,
            substrate_line=None,
        )  # fmt: skip
        return {
            "pml": prepared.box,
            "cover_line": float(prepared.cover_line),
            "substrate_line": (
                None if prepared.substrate_line is None else float(prepared.substrate_line)
            ),
        }

    def _check_lines(self) -> None:
        """The morph must not deform the cells at a measurement line that lies on mesh facets
        (see :func:`hpfem.grating.shape_sensitivity`)."""
        tol = 1e-9 * self.morph.cell.period
        for frame in self._frames:
            for key in ("cover_line", "substrate_line"):
                line = frame[key]
                if line is None:
                    continue
                for p in self.morph.parameters:
                    count = grating.deformed_line_cells(
                        self.morph.mesh, self.morph.velocity(p.name), line, tol
                    )
                    if count:
                        raise ValueError(
                            f"GratingEvaluator: parameter {p.name} deforms {count} cells at the "
                            f"{key} (y = {line:.6g} m), which lies on mesh facets; the order "
                            "amplitudes are not differentiable along it. Build the Morph with a "
                            "band that stays at least one cell row inside the measurement lines "
                            "(Morph(..., band=(y_low, y_high)))"
                        )

    def _mesh(self, geometry: Mapping[str, float], meta: dict):
        try:
            mesh = self.morph.mesh_at(geometry)
        except MeshQualityError as error:
            if self.mesher is None:
                raise
            cell = self.morph.cell_at(geometry)
            mesh = self.mesher(cell)
            self.morph = Morph(cell, mesh, self.morph.parameters, self.morph.quality_threshold,
                               self.morph.step, self.morph.band)  # fmt: skip
            self.mesh_id += 1
            meta["remesh"] = {"mesh_id": self.mesh_id, "reason": str(error)}
            self._check_lines()
        meta["quality"] = float(self.morph.check(mesh))
        return mesh

    def __call__(self, params: Mapping[str, float], *, jacobian: bool = False,
                 fidelity: Mapping | None = None,
                 cancel: Callable[[], bool] | None = None) -> Evaluation:  # fmt: skip
        t0 = time.perf_counter()
        values = {n: float(params[n]) for n in self.names}
        meta: dict = {}
        mesh = self._mesh({p.name: values[p.name] for p in self.morph.parameters}, meta)
        materials = dict(self.materials)
        for p in self.material_parameters:
            materials = p.apply(materials, values[p.name])
        order = int((fidelity or {}).get("order", self.order))
        # Jacobian columns: one shape column per geometry parameter, the permittivity pair of
        # every material tag once
        tags = list(dict.fromkeys(p.tag for p in self.material_parameters))
        spec = [("shape", self.morph.velocity(p.name)) for p in self.morph.parameters]
        spec += [("eps", t) for t in tags]
        n_geometry = len(self.morph.parameters)
        columns = list(range(n_geometry)) + [
            n_geometry + 2 * tags.index(p.tag) + (0 if p.part == "re" else 1)
            for p in self.material_parameters
        ]
        rows, jac_rows, dofs = [], [], 0
        for config, frame in zip(self.configurations, self._frames, strict=True):
            if cancel is not None and cancel():
                raise hpfem.Cancelled("GratingEvaluator: cancelled between configurations")
            result = grating.solve(
                mesh, materials, self.stack, config.polarisation, config.theta, config.phi,
                config.omega, order, pml=frame["pml"], cover_line=frame["cover_line"],
                substrate_line=frame["substrate_line"], keep_factorisation=jacobian,
                check=False, **self.options,
            )  # fmt: skip
            dofs = max(dofs, int(result.dofs))
            for side, m in config.orders:
                orders = result.R_orders if side == "R" else result.T_orders
                match = [o for o in orders if o.m == m]
                if not match or not match[0].propagating:
                    raise grating.GratingError(
                        f"{config.label(side, m)}: the order does not propagate"
                    )
                rows.append(match[0].efficiency)
            if jacobian:
                jac, _, _ = grating.jacobian(result, spec, observables=config.orders)
                jac_rows.append(jac[:, columns])
        meta.update({"dofs": dofs, "order": order, "solve_seconds": time.perf_counter() - t0})
        return Evaluation(
            params=values,
            values=np.asarray(rows, dtype=float),
            jacobian=np.vstack(jac_rows) if jacobian else None,
            cost=time.perf_counter() - t0,
            fidelity=dict(fidelity or {}),
            mesh_id=self.mesh_id,
            meta=meta,
        )

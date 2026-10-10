"""One-call periodic scattering (M15 F2): :func:`solve` runs the conical solver on a
grating unit cell and returns the diffraction efficiencies, the vector amplitudes, the
absorbed power and the power balance; :func:`validate` runs the diagnostics (M15 F7) alone.

Frame (as :class:`hpfem.ConicalScattering`): the period along ``x``, the stack normal along
``y`` (layers above / below the structure), the structure invariant along ``z``. The mesh is
the unit cell including the PML regions at its top and bottom, with the Bloch faces tagged
``box_tag.X_MIN`` / ``box_tag.X_MAX`` and the material cells tagged; the layered background
is a :class:`hpfem.LayerStack2D`. The incident plane wave of unit amplitude comes from the
incidence medium at the polar angle ``theta`` against the normal and the azimuth ``phi`` about
it (``phi = 0``: the plane of incidence is the x–y plane, ``s`` is the ``E_z`` polarisation).
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

import hpfem
from hpfem import units


class GratingError(ValueError):
    """A problem with the unit cell, the stack or the options, with a hint."""


@dataclass
class Order:
    """One diffraction order of the reflected or transmitted field."""

    m: int
    efficiency: float
    """power fraction of the incident wave (0 for evanescent orders)"""
    amplitude: np.ndarray
    """complex vector amplitude (E_x, E_y, E_z) of the order on the measurement line"""
    kt: float
    """tangential wavenumber k_x + 2 pi m / period [1/m]"""
    kn: complex
    """normal wavenumber in the medium of the line [1/m], imaginary for evanescent orders"""
    propagating: bool


@dataclass
class GratingResult:
    R_orders: list[Order]
    T_orders: list[Order]
    """empty with a lossy substrate or ``bottom="pec"``"""
    R: float
    T: float
    A: float
    """absorbed power per period, as a fraction of the incident power"""
    A_by_tag: dict[int, float]
    power_balance_residual: float
    """R + T + A - 1"""
    wave: object
    """the ``LayeredConicalWave`` of the bare stack (beta, kx, ky, reflectance, ...)"""
    problem: object
    solution: object
    mesh: object
    pml: object
    cover_line: float
    substrate_line: float | None
    dofs: int
    timing: dict[str, float] = field(default_factory=dict)
    diagnostics: list = field(default_factory=list)
    """the warnings and infos of :func:`validate` (errors stop ``solve``)"""
    flux_balance: dict | None = None
    scalar: bool = False
    """solved on the scalar E_z path (H1 block only; s polarisation at phi = 0)"""
    period: float = 0.0
    x_min: float = 0.0
    fourier_points: int = 256
    """flux-based balance (``conical_power_balance``) through the PML boundaries: incident,
    reflected, transmitted, absorbed [W/m] and ``relative_residual``; ``None`` when the PML
    boundaries are no mesh lines (unstructured meshes)"""
    inputs: dict | None = None
    """the arguments of :func:`solve` that define the problem besides the mesh (``materials``,
    ``stack``, ``polarisation``, ``theta``, ``phi``, ``omega``, ``orders_max``,
    ``extra_quadrature_order``, ``solver``): :func:`jacobian` rebuilds the problem at
    neighbouring frequencies and angles from them"""
    _locator: object = None

    def field(self, points, quantity: str = "E", scattered: bool = False) -> np.ndarray:
        """Total (or scattered) physical field at ``points`` of shape (n, 2): (n, 3) complex;
        ``quantity`` ``"E"``, ``"H"`` or ``"S"`` as ``ConicalScattering.sample``."""
        if self._locator is None:
            self._locator = hpfem.PointLocator2D(self.mesh)
        values, _cells = self.problem.sample(
            self.solution,
            self._locator,
            np.asarray(points, dtype=float),
            scattered,
            True,
            0,
            quantity,
        )
        return values


def _polarisation(pol):
    if isinstance(pol, hpfem.Polarisation):
        return pol
    key = str(pol).strip().lower()
    if key in ("s", "te", "ez", "e_z"):
        return hpfem.Polarisation.S
    if key in ("p", "tm", "hz", "h_z"):
        return hpfem.Polarisation.P
    raise GratingError(f"polarisation {pol!r}: use 's' (E_z at phi = 0) or 'p' (H_z at phi = 0)")


def _material_map(materials, stack, omega: float) -> hpfem.MaterialMap:
    """``MaterialMap`` as given, or a dict of ``Material`` / dispersive models (evaluated at
    ``omega`` through their ``at``) with the incidence medium as background."""
    if isinstance(materials, hpfem.MaterialMap):
        return materials
    out = hpfem.MaterialMap(stack.incidence_medium)
    for tag, material in dict(materials).items():
        if hasattr(material, "at") and not isinstance(material, hpfem.Material):
            material = material.at(omega)
        out.set(int(tag), material)
    return out


def _snap_interfaces(mesh, stack, tolerance: float) -> int:
    """Moves vertices within ``tolerance`` of a stack interface onto it; returns the count."""
    interfaces = [stack.interface(i) for i in range(stack.num_layers + 1)]
    moved = 0
    for v in range(mesh.num_vertices):
        x = np.array(mesh.vertex(v), dtype=float)
        for y in interfaces:
            if 0 < abs(x[1] - y) <= tolerance:
                mesh.set_vertex(v, [x[0], y])
                moved += 1
                break
    return moved


def _check_interfaces(mesh, stack, period: float, tolerance: float):
    """Every cell lies within one region of the stack (the solver checks too; this gives the
    position of the first offending cell and a hint)."""
    interfaces = [stack.interface(i) for i in range(stack.num_layers + 1)]
    for c in range(mesh.num_cells):
        ys = [float(mesh.vertex(int(v))[1]) for v in mesh.cell_vertices(c)]
        lo, hi = min(ys), max(ys)
        for y in interfaces:
            if lo < y - tolerance and hi > y + tolerance:
                raise GratingError(
                    f"cell {c} (y from {lo:.6g} to {hi:.6g} m) straddles the stack interface at "
                    f"y = {y:.6g} m: put the interfaces on mesh lines (snap_tolerance = "
                    f"{tolerance / period:.1e} of the period moves vertices closer than that)"
                )


def _structure_extent(mesh, material_map, stack):
    """Lowest and highest vertex of the cells whose material differs from the stack's at their
    centroid (the scatterers), None if there are none."""
    lo, hi = np.inf, -np.inf
    for c in range(mesh.num_cells):
        centroid = np.mean([mesh.vertex(int(v)) for v in mesh.cell_vertices(c)], axis=0)
        material = material_map.of_cell(mesh, c)
        background = stack.material_at([float(centroid[0]), float(centroid[1])])
        if material.eps_r != background.eps_r or material.mu_r != background.mu_r:
            ys = [float(mesh.vertex(int(v))[1]) for v in mesh.cell_vertices(c)]
            lo, hi = min(lo, min(ys)), max(hi, max(ys))
    if lo > hi:
        return None
    return lo, hi


def _max_order_angle(kx: float, k0: float, n: float, period: float, orders_max: int) -> float:
    """Largest propagation angle of an order in a medium of index n (rad), 0 if none."""
    worst = 0.0
    for m in range(-orders_max, orders_max + 1):
        kt = kx + 2 * np.pi * m / period
        s = abs(kt) / (k0 * n)
        if s < 1.0:
            worst = max(worst, float(np.arcsin(s)))
    return worst


def _orders(coefficients, k0, n, period, kx, beta, ky_incident, orders_max) -> list[Order]:
    raw = hpfem.conical_diffraction_efficiencies(
        coefficients, k0, n, period, kx, beta, ky_incident, 1.0
    )
    out = []
    for o in raw:
        out.append(
            Order(
                m=int(o.order),
                efficiency=float(o.efficiency),
                amplitude=np.asarray(o.amplitude, dtype=complex),
                kt=float(o.ky),
                kn=complex(o.kx),
                propagating=bool(o.propagating),
            )
        )
    out.sort(key=lambda o: o.m)
    return out


def _conical_setup(omega, wave, material_map, stack, box, period, extra_quadrature_order, scalar):
    """The ``ConicalScatteringSetup`` of the unit cell (without solver choice and callbacks)."""
    setup = hpfem.ConicalScatteringSetup()
    setup.omega = float(omega)
    setup.beta = wave.beta
    setup.materials = material_map
    setup.background = stack
    setup.incident = wave.field
    setup.pml = box
    setup.pec_tags = [hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX]
    setup.periodic = [
        hpfem.PeriodicPair2D(
            hpfem.box_tag.X_MIN,
            hpfem.box_tag.X_MAX,
            [period, 0.0],
            hpfem.bloch_phase([wave.kx, 0.0], [period, 0.0]),
        )
    ]
    setup.extra_quadrature_order = int(extra_quadrature_order)
    setup.scalar_ez = bool(scalar)
    return setup


def _measure_orders(problem, solution, locator, wave, k0, n_cover, n_sub, lines, orders_max,
                    fourier_points):  # fmt: skip
    """Reflected and transmitted orders of ``solution`` on the measurement lines
    ``lines = (x_min, period, cover_line, substrate_line or None)``."""
    x_min, period, cover_line, substrate_line = lines

    def incident_physical(x):
        i = wave.incident(x)  # the scaled (E_x, E_y, -i E_z) of the downward wave
        return np.array([i[0], i[1], 1j * i[2]])

    def reflected_field(x):
        return np.asarray(problem.total_field(solution, locator, x)) - incident_physical(x)

    coefficients = hpfem.conical_fourier_coefficients(
        reflected_field, [x_min, cover_line], [1.0, 0.0], period, wave.kx, orders_max,
        fourier_points,
    )  # fmt: skip
    r_orders = _orders(
        coefficients, k0, n_cover.real, period, wave.kx, wave.beta, wave.ky, orders_max
    )
    t_orders: list[Order] = []
    if substrate_line is not None:
        coefficients = hpfem.conical_fourier_coefficients(
            lambda x: np.asarray(problem.total_field(solution, locator, x)),
            [x_min, substrate_line], [1.0, 0.0], period, wave.kx, orders_max, fourier_points,
        )  # fmt: skip
        t_orders = _orders(
            coefficients, k0, n_sub.real, period, wave.kx, wave.beta, wave.ky, orders_max
        )
    return r_orders, t_orders, incident_physical


@dataclass
class _Prepared:
    """Everything :func:`solve` needs besides the mesh, as computed by :func:`_prepare`."""

    period: float
    x_min: float
    y_min: float
    y_max: float
    k0: float
    n_cover: complex
    n_sub: complex
    material_map: object
    wave: object
    box: object
    t_top: float
    t_bottom: float
    cover_line: float
    substrate_line: float | None
    transmitted: bool
    structure_bottom: float
    moved: int


def _prepare(
    mesh,
    materials,
    stack,
    polarisation,
    theta,
    phi,
    omega,
    *,
    pml,
    bottom,
    orders_max,
    snap_tolerance,
    pml_target,
    pml_wavelengths,
    cover_line,
    substrate_line,
) -> _Prepared:
    pol = _polarisation(polarisation)
    if bottom not in ("pml", "pec"):
        raise GratingError(f"bottom={bottom!r}: use 'pml' or 'pec'")
    k0 = float(units.vacuum_wavenumber(omega))
    vertices = np.array([mesh.vertex(v) for v in range(mesh.num_vertices)], dtype=float)
    x_min, x_max = float(vertices[:, 0].min()), float(vertices[:, 0].max())
    y_min, y_max = float(vertices[:, 1].min()), float(vertices[:, 1].max())
    period = x_max - x_min
    if not period > 0:
        raise GratingError("the mesh has no extent along x (the period)")
    n_cover = complex(stack.incidence_medium.refractive_index)
    n_sub = complex(stack.substrate.refractive_index)
    if abs(n_cover.imag) > 1e-12:
        raise GratingError("the incidence medium of the stack must be lossless")
    material_map = _material_map(materials, stack, float(omega))

    # --- geometry: interfaces on mesh lines, PML layers, measurement lines ---------------------
    moved = _snap_interfaces(mesh, stack, snap_tolerance * period)
    _check_interfaces(mesh, stack, period, snap_tolerance * period)
    wave = hpfem.layered_conical_wave(stack, k0, float(theta), float(phi), pol)
    n_ref = min(n_cover.real, n_sub.real)
    theta_max = max(
        _max_order_angle(wave.kx, k0, n_cover.real, period, orders_max),
        _max_order_angle(wave.kx, k0, n_sub.real, period, orders_max)
        if abs(n_sub.imag) < 1e-12
        else 0.0,
        float(theta),
    )
    theta_max = min(theta_max, 80 * units.deg)
    # the binding takes degrees
    profile = hpfem.PmlProfile.for_angle(theta_max / units.deg, pml_target, 1.0, 2)
    if isinstance(pml, hpfem.PmlBox2D):
        box = pml
        t_top = y_max - box.upper[1]
        t_bottom = box.lower[1] - y_min
    else:
        if isinstance(pml, Mapping):
            t_top = float(pml.get("top", 0.0))
            t_bottom = float(pml.get("bottom", 0.0)) if bottom == "pml" else 0.0
        else:
            heights = []
            for c in range(mesh.num_cells):
                ys = [float(mesh.vertex(int(v))[1]) for v in mesh.cell_vertices(c)]
                heights.append(max(ys) - min(ys))
            cell = float(np.median(heights))
            t_top = hpfem.PmlBox2D.recommended_thickness(k0, n_cover.real, cell, pml_wavelengths)
            t_bottom = (
                hpfem.PmlBox2D.recommended_thickness(
                    k0, max(n_sub.real, 1.0), cell, pml_wavelengths
                )
                if bottom == "pml"
                else 0.0
            )
        if t_top <= 0 or (bottom == "pml" and t_bottom <= 0):
            raise GratingError("PML thicknesses must be positive (bottom='pec' needs none below)")
        if t_top + t_bottom >= y_max - y_min:
            raise GratingError("the PML layers do not fit into the mesh")
        box = hpfem.PmlBox2D(
            [x_min, y_min + t_bottom],
            [x_max, y_max - t_top],
            [0.0, 0.0, t_bottom, t_top],
            k0,
            n_ref,
            profile,
        )
    extent = _structure_extent(mesh, material_map, stack)
    structure_top = extent[1] if extent else stack.top
    structure_bottom = extent[0] if extent else stack.bottom
    pml_top_start = y_max - t_top
    pml_bottom_start = y_min + t_bottom
    if cover_line is None:
        cover_line = 0.5 * (max(structure_top, stack.top) + pml_top_start)
    if cover_line <= max(structure_top, stack.top) or cover_line >= pml_top_start:
        raise GratingError(
            "cover_line must lie above the structure and the stack and below the top PML"
        )
    transmitted = bottom == "pml" and abs(n_sub.imag) < 1e-12
    if substrate_line is None and transmitted:
        substrate_line = 0.5 * (min(structure_bottom, stack.bottom) + pml_bottom_start)
    if transmitted and (
        substrate_line >= min(structure_bottom, stack.bottom) or substrate_line <= pml_bottom_start
    ):
        raise GratingError(
            "substrate_line must lie below the structure and the stack and above the bottom PML"
        )
    if not transmitted:
        substrate_line = None
    return _Prepared(
        period=period,
        x_min=x_min,
        y_min=y_min,
        y_max=y_max,
        k0=k0,
        n_cover=n_cover,
        n_sub=n_sub,
        material_map=material_map,
        wave=wave,
        box=box,
        t_top=t_top,
        t_bottom=t_bottom,
        cover_line=cover_line,
        substrate_line=substrate_line,
        transmitted=transmitted,
        structure_bottom=min(structure_bottom, stack.bottom),
        moved=moved,
    )


def _diagnostics(mesh, pr: _Prepared, stack, materials, order, bottom, orders_max, omega):
    from hpfem import diagnostics as dg

    given = pr.material_map if isinstance(materials, hpfem.MaterialMap) else materials
    out = dg.validate_mesh(mesh, given)
    out += dg.validate_stack(stack)
    pair = (hpfem.box_tag.X_MIN, hpfem.box_tag.X_MAX, [pr.period, 0.0])
    out += dg.validate_periodic(mesh, [pair])
    p = int(order) if np.isscalar(order) else int(np.median(order))
    out += dg.validate_pml(mesh, pr.box, p, min(pr.n_cover.real, max(pr.n_sub.real, 1.0)))
    out += dg.validate_resolution(mesh, pr.material_map, omega, order)
    out += dg.validate_materials(materials, omega)
    out += dg.validate_orders(
        pr.k0,
        pr.wave.kx,
        pr.n_cover.real,
        pr.n_sub.real if abs(pr.n_sub.imag) < 1e-12 else None,
        pr.period,
        orders_max,
    )
    if bottom == "pec":
        out += dg.validate_bottom_wall(pr.structure_bottom - pr.y_min, pr.n_sub, pr.k0)
    return out


@dataclass
class ResonantMode:
    """One resonance of the unit cell: complex ``omega`` (Im < 0 for a decaying mode),
    ``wavelength`` of the real part, quality factor ``Q``, Arnoldi ``residual``, ``beta``."""

    index: int
    omega: complex
    wavelength: float
    Q: float
    residual: float
    beta: float
    _result: object = field(default=None, repr=False)

    @property
    def raw(self):
        """the ``hpfem.ConicalResonantMode`` (block coefficients)"""
        return self._result.raw.modes[self.index]

    def field(self, points, quantity: str = "E") -> np.ndarray:
        """Physical mode field at ``points`` of shape (n, 2): (n, 3) complex; ``quantity``
        ``"E"``, ``"H"`` (with the complex omega) or ``"S"``; Bloch-wrapped along the period,
        NaN outside the mesh. The modes are normalised to unit 2-norm of the coefficients."""
        values, _cells = self._result.problem.sample(
            self.raw, self._result._locator_or_build(), np.asarray(points, dtype=float), True, 0,
            quantity,
        )  # fmt: skip
        return values


@dataclass
class ResonanceResult:
    """Result of :func:`resonances`: the modes closest to the target, the problem and the
    timing. ``omegas`` is the array of complex angular frequencies."""

    mesh: object
    problem: object
    raw: object
    modes: list[ResonantMode]
    kx: float
    beta: float
    target_omega: float
    period: float
    dofs: int
    timing: dict[str, float] = field(default_factory=dict)
    inputs: dict | None = None
    """``materials`` and ``stack`` as given to :func:`resonances` (for
    :func:`refine_resonance`)"""
    self_consistent: bool = False
    """the dispersive materials are evaluated at the mode's own frequency
    (:func:`refine_resonance`), not at the target"""
    iterations: int = 0
    """Newton steps of :func:`refine_resonance`"""
    _locator: object = None

    @property
    def omegas(self) -> np.ndarray:
        return np.array([m.omega for m in self.modes], dtype=complex)

    def _locator_or_build(self):
        if self._locator is None:
            self._locator = hpfem.PointLocator2D(self.mesh)
        return self._locator


def resonances(
    mesh,
    materials,
    stack,
    omega_target: float,
    *,
    kx: float = 0.0,
    beta: float = 0.0,
    num_modes: int = 4,
    order=4,
    pml=None,
    bottom: str = "pml",
    snap_tolerance: float = 1e-9,
    pml_target: float = 1e-6,
    pml_wavelengths: float = 0.5,
    solver=None,
    krylov_dimension: int = 0,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
    extra_quadrature_order: int = 2,
    progress=None,
    cancel=None,
) -> ResonanceResult:
    """Resonances (quasi-normal modes) of the grating unit cell closest to ``omega_target``
    [rad/s]: the conical eigenproblem of :class:`hpfem.ConicalResonance` with the Bloch
    wavenumber ``kx`` [1/m] along the period, the longitudinal wavenumber ``beta`` [1/m] (0:
    in-plane, both polarisations in one call), the PML of :func:`solve` designed at the
    target frequency and the angle of ``kx``, PEC at the top and bottom of the mesh. ``mesh``,
    ``materials``, ``stack``, ``order``, ``pml``, ``bottom`` and the snapping as in
    :func:`solve`; ``num_modes``, ``krylov_dimension``, ``tolerance``, ``max_iterations`` as
    in ``ConicalResonanceSetup``; ``progress`` / ``cancel`` as in :func:`solve` (phases
    assembly, constraints, eigensolve, post). Returns a :class:`ResonanceResult` whose modes
    are ordered by the distance of omega to the target and sample their fields with
    :meth:`ResonantMode.field`."""
    t0 = time.perf_counter()
    k0 = float(units.vacuum_wavenumber(omega_target))
    n_cover = complex(stack.incidence_medium.refractive_index)
    # the PML design angle of the Bloch wavenumber, capped at 80 deg (evanescent kx)
    sin_theta = min(np.sin(80 * units.deg), abs(float(kx)) / (k0 * max(n_cover.real, 1e-300)))
    pr = _prepare(
        mesh,
        materials,
        stack,
        "p",
        float(np.arcsin(sin_theta)),
        0.0,
        float(omega_target),
        pml=pml,
        bottom=bottom,
        orders_max=0,
        snap_tolerance=snap_tolerance,
        pml_target=pml_target,
        pml_wavelengths=pml_wavelengths,
        cover_line=None,
        substrate_line=None,
    )
    period = pr.period
    orders = [int(order)] * mesh.num_cells if np.isscalar(order) else [int(p) for p in order]
    nd = hpfem.NedelecDofMap2D(mesh, orders)
    h1 = hpfem.DofMap2D(mesh, orders)
    setup = hpfem.ConicalResonanceSetup()
    setup.target_omega = float(omega_target)
    setup.beta = float(beta)
    setup.materials = pr.material_map
    setup.pml = pr.box
    setup.pec_tags = [hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX]
    setup.periodic = [
        hpfem.PeriodicPair2D(
            hpfem.box_tag.X_MIN,
            hpfem.box_tag.X_MAX,
            [period, 0.0],
            hpfem.bloch_phase([float(kx), 0.0], [period, 0.0]),
        )
    ]
    setup.num_modes = int(num_modes)
    setup.krylov_dimension = int(krylov_dimension)
    setup.tolerance = float(tolerance)
    setup.max_iterations = int(max_iterations)
    setup.extra_quadrature_order = int(extra_quadrature_order)
    if solver is not None:
        setup.solver = solver
    if progress is not None or cancel is not None:

        def report(event):
            if progress is not None:
                progress(event)
            return not (cancel is not None and cancel())

        setup.progress = report
    try:
        problem = hpfem.ConicalResonance(nd, h1, setup)
    except ValueError as error:
        raise GratingError(f"{error} (snapped {pr.moved} vertices)") from error
    raw = problem.solve()
    timing = dict(raw.timing)
    timing["total"] = time.perf_counter() - t0
    result = ResonanceResult(
        mesh=mesh,
        problem=problem,
        raw=raw,
        modes=[],
        kx=float(kx),
        beta=float(beta),
        target_omega=float(omega_target),
        period=period,
        dofs=int(len(problem.free_dofs)),
        timing=timing,
        inputs={"materials": materials, "stack": stack},
    )
    result.modes = [
        ResonantMode(
            index=i,
            omega=complex(m.omega),
            wavelength=float(m.wavelength),
            Q=float(m.quality),
            residual=float(m.residual),
            beta=float(m.beta),
            _result=result,
        )
        for i, m in enumerate(raw.modes)
    ]
    return result


def _is_model(material) -> bool:
    """A dispersive model (``eps_r(omega)`` callable) rather than a fixed ``Material``."""
    return callable(getattr(material, "eps_r", None))


def _eps_at(model, omega: complex) -> complex:
    """eps_r of a dispersive model at the complex frequency of a quasi-normal mode: the
    analytic continuation for ``analytic`` models (Drude–Lorentz, constant), the real axis
    (``Re omega``) for tabulated and Sellmeier data."""
    if getattr(model, "analytic", False):
        return complex(np.asarray(model.eps_r(complex(omega))))
    return complex(np.asarray(model.eps_r(float(np.real(omega)))))


def _deps_domega(model, omega: complex, relative_step: float = 1e-6) -> complex:
    """d eps_r / d omega of a dispersive model at ``omega`` (central differences along the real
    axis: the complex derivative for analytic models, the real-axis slope otherwise)."""
    h = relative_step * abs(omega)
    return (_eps_at(model, omega + h) - _eps_at(model, omega - h)) / (2 * h)


def _material_map_at(materials, stack, omega: complex):
    """The ``MaterialMap`` with the dispersive models of ``materials`` at ``omega``."""
    if isinstance(materials, hpfem.MaterialMap):
        return materials
    out = hpfem.MaterialMap(stack.incidence_medium)
    for tag, material in dict(materials).items():
        if _is_model(material) and not isinstance(material, hpfem.Material):
            material = hpfem.Material(_eps_at(material, omega), complex(material.mu_r))
        out.set(int(tag), material)
    return out


_SETUP_FIELDS = (
    "target_omega", "beta", "materials", "pml", "pec_tags", "periodic", "num_modes",
    "krylov_dimension", "tolerance", "max_iterations", "remove_gradients", "solver",
    "extra_quadrature_order", "pml_extra_quadrature_order",
)  # fmt: skip


def _resonance_problem_like(result: ResonanceResult, **changes):
    """The conical resonance problem of ``result`` on the same maps with some setup fields
    changed (``materials``, ``target_omega``, ``periodic``, ``beta``)."""
    old = result.problem.setup
    setup = hpfem.ConicalResonanceSetup()
    for name in _SETUP_FIELDS:
        setattr(setup, name, changes.get(name, getattr(old, name)))
    problem = result.problem
    return hpfem.ConicalResonance(problem.transverse_dofs, problem.longitudinal_dofs, setup)


def _bloch_pair(result: ResonanceResult, kx: float):
    return hpfem.PeriodicPair2D(
        hpfem.box_tag.X_MIN, hpfem.box_tag.X_MAX, [result.period, 0.0],
        hpfem.bloch_phase([float(kx), 0.0], [result.period, 0.0]),
    )  # fmt: skip


def _closest_mode(raw_modes, omega: complex) -> int:
    lam = (omega / hpfem.constants.c0) ** 2
    distances = [abs((m.omega / hpfem.constants.c0) ** 2 - lam) for m in raw_modes]
    return int(np.argmin(distances))


def _dispersive_tags(result: ResonanceResult) -> dict:
    materials = (result.inputs or {}).get("materials", {})
    if isinstance(materials, hpfem.MaterialMap):
        return {}
    return {int(t): m for t, m in dict(materials).items() if _is_model(m)}


def refine_resonance(
    result: ResonanceResult,
    mode: ResonantMode | int = 0,
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 20,
) -> ResonanceResult:
    """The self-consistent resonance of dispersive materials: :func:`resonances` evaluates the
    dispersive models of ``materials`` at the target frequency, so its modes solve the pencil
    frozen there. The resonance proper solves :math:`\\hat\\lambda(\\omega) = (\\omega/c_0)^2`,
    where :math:`\\hat\\lambda(\\omega)` is the eigenvalue of the pencil with
    :math:`\\varepsilon(\\omega)` at the mode's own complex ω (analytic continuation for
    Drude–Lorentz and constant models, the real part of ω for tabulated data). Newton on
    :math:`f(\\omega) = \\hat\\lambda(\\omega) - (\\omega/c_0)^2` with
    :math:`f' = \\sum_t (\\partial\\hat\\lambda/\\partial\\varepsilon_t)\\,\\varepsilon_t'(\\omega)
    - 2\\omega/c_0^2` (the material derivatives of the dispersive tags on the left eigenvector,
    ``conical_resonance_material_derivative``): one eigensolve and one adjoint per step,
    quadratic convergence; the PML stays the one of ``result``. Returns a new
    :class:`ResonanceResult` (``self_consistent=True``, ``iterations``) whose modes are those of
    the last pencil, the refined one first. Raises ``GratingError`` without dispersive
    materials or if Newton does not converge."""
    index = mode.index if isinstance(mode, ResonantMode) else int(mode)
    models = _dispersive_tags(result)
    if not models:
        raise GratingError("refine_resonance: no dispersive materials in the result")
    c0 = hpfem.constants.c0
    omega = complex(result.raw.modes[index].omega)
    stack = result.inputs["stack"]
    materials = result.inputs["materials"]
    steps = 0
    for steps in range(1, int(max_iterations) + 1):  # noqa: B007 (the count is reported)
        problem = _resonance_problem_like(
            result, materials=_material_map_at(materials, stack, omega),
            target_omega=float(abs(omega)),
        )  # fmt: skip
        raw = problem.solve()
        i = _closest_mode(raw.modes, omega)
        frozen = raw.modes[i]
        lam_hat = (complex(frozen.omega) / c0) ** 2
        f = lam_hat - (omega / c0) ** 2
        if abs(f) <= tolerance * abs(lam_hat):
            break
        adjoint = hpfem.conical_resonance_adjoint(problem, frozen)
        dlam_domega = sum(
            hpfem.conical_resonance_material_derivative(problem, frozen, adjoint, t).dlambda
            * _deps_domega(model, omega)
            for t, model in models.items()
        )
        omega = omega - f / (dlam_domega - 2.0 * omega / c0**2)
    else:
        raise GratingError(
            f"refine_resonance: no self-consistent resonance after {max_iterations} Newton steps "
            f"(|f| / |lambda| = {abs(f) / abs(lam_hat):.2e})"
        )
    out = ResonanceResult(
        mesh=result.mesh, problem=problem, raw=raw, modes=[], kx=result.kx, beta=result.beta,
        target_omega=float(abs(omega)), period=result.period, dofs=result.dofs,
        timing=dict(raw.timing), inputs=result.inputs, self_consistent=True,
        iterations=steps, _locator=result._locator,
    )  # fmt: skip
    order = [i] + [j for j in range(len(raw.modes)) if j != i]
    out.modes = [
        ResonantMode(index=j, omega=complex(raw.modes[j].omega),
                     wavelength=float(raw.modes[j].wavelength), Q=float(raw.modes[j].quality),
                     residual=float(raw.modes[j].residual), beta=float(raw.modes[j].beta),
                     _result=out)
        for j in order
    ]  # fmt: skip
    return out


def resonance_sensitivity(result: ResonanceResult, mode: ResonantMode | int = 0,
                          parameters=("beta",), *, step: float = 1e-6) -> dict:  # fmt: skip
    """Derivatives of a resonance of :func:`resonances` (or :func:`refine_resonance`) with
    respect to ``parameters`` (M16 S4): a sequence of ``("eps", tag)`` (the relative
    permittivity of the cells with ``tag``: entries ``"eps[tag].re"`` and ``"eps[tag].im"``),
    ``("shape", velocity)`` (a mesh velocity, entry ``"shape[j]"``), ``"beta"`` [per 1/m] and
    ``"kx"`` (the Bloch wavenumber [per 1/m]: the complex dispersion of a leaky mode, from the
    problems at ``kx ∓ h``); the difference steps of beta and kx are ``step`` relative to
    ``max(|beta|, k0)`` and ``max(|kx|, 2 pi / period)``. Returns
    ``{label: hpfem.ResonanceDerivative}`` with ``domega`` (complex [rad/s per unit]: the real
    part moves the resonance, the imaginary part its width), ``dquality``, ``dwavelength``
    [m per unit] and ``dlambda``.

    The derivatives use the left eigenvector of the Bloch-reduced pencil
    (``conical_resonance_adjoint``) and hold for a simple eigenvalue. For a self-consistent
    resonance of dispersive materials (``result.self_consistent``) the term
    :math:`\\partial_\\omega\\hat\\lambda = \\sum_t (\\partial\\hat\\lambda/\\partial
    \\varepsilon_t)\\,\\varepsilon_t'(\\omega)` enters the denominator,
    :math:`d\\omega/dp = \\partial_p\\hat\\lambda / (2\\omega/c_0^2 - \\partial_\\omega
    \\hat\\lambda)`; for a result of :func:`resonances` with dispersive models it is the
    derivative of the pencil frozen at the target (refine first for the resonance proper)."""
    index = mode.index if isinstance(mode, ResonantMode) else int(mode)
    problem = result.problem
    raw = result.raw.modes[index]
    omega = complex(raw.omega)
    adjoint = hpfem.conical_resonance_adjoint(problem, raw)
    dlam_domega = 0j
    if result.self_consistent:
        dlam_domega = sum(
            hpfem.conical_resonance_material_derivative(problem, raw, adjoint, t).dlambda
            * _deps_domega(model, omega)
            for t, model in _dispersive_tags(result).items()
        )
    out = {}

    def add(label, dlambda):
        out[label] = hpfem.resonance_derivative_from(omega, complex(dlambda), complex(dlam_domega))

    for j, parameter in enumerate(parameters):
        kind, value = (parameter, None) if isinstance(parameter, str) else parameter
        if kind == "eps":
            d = hpfem.conical_resonance_material_derivative(problem, raw, adjoint, int(value))
            add(f"eps[{int(value)}].re", d.dlambda)
            add(f"eps[{int(value)}].im", 1j * d.dlambda)
        elif kind == "shape":
            velocity = np.asarray(value, dtype=float)
            d = hpfem.conical_resonance_shape_derivative(problem, raw, adjoint, velocity)
            add(f"shape[{j}]", d.dlambda)
        elif kind == "beta":
            h = step * max(abs(result.beta), abs(omega) / hpfem.constants.c0)  # relative step
            d = hpfem.conical_resonance_beta_derivative(problem, raw, adjoint, h)
            add("beta", d.dlambda)
        elif kind == "kx":
            h = step * max(abs(result.kx), 2 * np.pi / result.period)
            minus = _resonance_problem_like(result, periodic=[_bloch_pair(result, result.kx - h)])
            plus = _resonance_problem_like(result, periodic=[_bloch_pair(result, result.kx + h)])
            d = hpfem.conical_resonance_bloch_derivative(problem, raw, adjoint, minus, plus, h)
            add("kx", d.dlambda)
        else:
            raise GratingError(f"parameter kind {kind!r}: use 'eps', 'shape', 'beta' or 'kx'")
    return out


def bands(
    mesh, materials, stack, omega_target: float, kx_values, **kwargs
) -> list[ResonanceResult]:
    """The resonances of :func:`resonances` for every Bloch wavenumber in ``kx_values`` [1/m]:
    the complex band structure (dispersion of the leaky and guided modes) of the open unit
    cell near ``omega_target``. Keyword arguments as :func:`resonances`; ``cancel`` is polled
    per wavenumber as well. Closed photonic crystals are ``hpfem.BandStructure2D``."""
    cancel = kwargs.get("cancel")
    out = []
    for kx in kx_values:
        if cancel is not None and cancel():
            raise hpfem.Cancelled("bands cancelled")
        out.append(resonances(mesh, materials, stack, omega_target, kx=float(kx), **kwargs))
    return out


def sensitivity(result: GratingResult, tag: int, order: int = 0, side: str = "R"):
    """Derivative of the efficiency of the reflected (``side="R"``) or transmitted (``"T"``)
    order ``order`` with respect to the relative permittivity of the cells tagged ``tag``, by
    one adjoint solve on the problem of ``result`` (a transposed solve on the kept
    factorisation with ``solve(..., keep_factorisation=True)``): returns
    ``(dR/dRe eps, dR/dIm eps)``.
    The linearised goal is :math:`Q = A_m \\cdot \\bar A_m / |A_m|` (``conical_order_functional``)
    so that :math:`dR_m = 2 R_m\\,\\mathrm{Re}(dQ)/|A_m|`; the holomorphic derivative
    ``dQ/d eps`` gives both directions. Raises ``GratingError`` for an order that is not in
    the result or evanescent, ``ValueError`` for a tag inside the PML."""
    functional, scale = _order_functional(result, order, side)
    if functional is None:
        return 0.0, 0.0
    problem = result.problem
    q_e, q_v = functional(problem.transverse_dofs, problem.longitudinal_dofs)
    z_e, z_v = hpfem.conical_adjoint_solution(problem, result.solution, q_e, q_v)
    dq = hpfem.conical_material_sensitivity(problem, result.solution, z_e, z_v, int(tag))
    return scale * float(np.real(dq)), scale * float(np.real(1j * dq))


def _order_functional(result: GratingResult, order: int, side: str):
    """The linearised goal Q = A_m . conj(A_m) / |A_m| of an order of the result, its
    efficiency scale 2 R_m / |A_m| and the C++ functional (``None`` for a zero amplitude)."""
    if side not in ("R", "T"):
        raise GratingError(f"side={side!r}: use 'R' or 'T'")
    orders = result.R_orders if side == "R" else result.T_orders
    match = [o for o in orders if o.m == int(order)]
    if not match or not match[0].propagating:
        raise GratingError(f"order {order} is not a propagating {side} order of the result")
    o = match[0]
    amplitude = np.asarray(o.amplitude, dtype=complex)
    norm = float(np.linalg.norm(amplitude))
    if norm == 0.0:
        return None, 0.0
    line = result.cover_line if side == "R" else result.substrate_line
    functional = hpfem.conical_order_functional(
        [result.x_min, line], [1.0, 0.0], result.period, result.wave.kx, int(order),
        result.fourier_points, np.conj(amplitude) / norm,
    )  # fmt: skip
    return functional, 2.0 * o.efficiency / norm


def shape_sensitivity(result: GratingResult, velocity, order: int = 0, side: str = "R"):
    """Derivative of the efficiency of the reflected (``"R"``) or transmitted (``"T"``) order
    ``order`` with respect to a geometry parameter given by its mesh velocity ``velocity``
    (array (num_geometry_nodes, 2): displacement of every vertex, then of every edge node of
    a second-order mesh, per unit of the parameter; ``hpfem.region_normal_velocity`` builds
    the uniform normal growth of a tagged region, e.g. a ridge; see
    docs/theory/maxwell.md "Shape derivatives"), by one adjoint solve on the problem of
    ``result``: :math:`dR_m/dp = 2 R_m\\,\\mathrm{Re}(dQ/dp)/|A_m|`. The velocity must vanish on
    the Bloch faces and the top and bottom of the mesh; the measurement lines may cross
    deforming cells transversally (the derivative of the line functional is included), but a
    line that lies on mesh facets must not touch deforming cells: the order amplitude sampled
    on facets is not differentiable along such a velocity (the normal Nédélec component jumps
    across facets, the located cell changes; ``GratingError``). Keep those cells fixed, e.g.
    ``hpfem.opt.Morph(..., band=...)`` one cell row inside the lines, or move the lines off
    the mesh lines."""
    functional, scale = _order_functional(result, order, side)
    if functional is None:
        return 0.0
    velocity = np.asarray(velocity, dtype=float)
    _check_lines_fixed(result, velocity)
    dq = hpfem.conical_shape_derivative(result.problem, result.solution, functional, velocity)
    return scale * float(np.real(dq))


def deformed_line_cells(mesh, velocity, line: float, tolerance: float) -> int:
    """Number of cells with a vertex on the horizontal line ``y = line`` (a measurement line
    on mesh facets) that the mesh ``velocity`` deforms (any of their vertices moves)."""
    y = np.asarray(mesh.vertices, dtype=float)[:, 1]
    on = np.abs(y - line) <= tolerance
    if not on.any():
        return 0
    moving = np.linalg.norm(np.asarray(velocity, dtype=float)[: len(y)], axis=1) > 0.0
    count = 0
    for c in range(mesh.num_cells):
        vertices = [int(v) for v in mesh.cell_vertices(c)][:3]
        if on[vertices].any() and moving[vertices].any():
            count += 1
    return count


def _check_lines_fixed(result: GratingResult, velocity: np.ndarray) -> None:
    """Raises if the velocity deforms a cell on a measurement line that lies on mesh facets."""
    tol = 1e-9 * result.period
    for name, line in (
        ("cover_line", result.cover_line),
        ("substrate_line", result.substrate_line),
    ):
        if line is None:
            continue
        count = deformed_line_cells(result.mesh, velocity, line, tol)
        if count:
            raise GratingError(
                f"the velocity deforms {count} cells at the {name} (y = {line:.6g} m), which "
                "lies on mesh facets: the order amplitude sampled on the facets is not "
                "differentiable along such a velocity (the normal Nédélec component jumps "
                "across facets, the located cell changes); keep the cells at the measurement "
                "lines fixed (hpfem.opt.Morph(..., band=(y_low, y_high)) at least one cell row "
                "inside the lines) or put the lines off the mesh lines"
            )


_SETUP_PARAMETERS = {"theta": 1e-6, "phi": 1e-6, "omega": 1e-7, "wavelength": 1e-7}
"""parameters of the whole setup and their default difference steps (rad for the angles,
relative to omega for the frequency)"""


def _neighbour(result: GratingResult, theta: float, phi: float, omega: float):
    """The problem of ``result`` at other angles and frequency: the same DoF maps, PML box,
    measurement lines and path (scalar or vector); dispersive materials evaluated at
    ``omega``. Returns ``(problem, wave, k0)``."""
    inputs = result.inputs
    stack = inputs["stack"]
    k0 = float(units.vacuum_wavenumber(omega))
    pol = _polarisation(inputs["polarisation"])
    wave = hpfem.layered_conical_wave(stack, k0, float(theta), float(phi), pol)
    material_map = _material_map(inputs["materials"], stack, float(omega))
    setup = _conical_setup(
        omega, wave, material_map, stack, result.pml, result.period,
        inputs["extra_quadrature_order"], result.scalar,
    )  # fmt: skip
    if inputs["solver"] is not None:
        setup.solver = inputs["solver"]
    problem = hpfem.ConicalScattering(
        result.problem.transverse_dofs, result.problem.longitudinal_dofs, setup
    )
    return problem, wave, k0


def _efficiencies_at(result: GratingResult, problem, wave, k0, rows) -> np.ndarray:
    """Efficiencies of the ``(side, order)`` rows of the fixed coefficients of ``result``
    post-processed with another problem (the explicit dependence on the parameters)."""
    stack = result.inputs["stack"]
    n_cover = complex(stack.incidence_medium.refractive_index)
    n_sub = complex(stack.substrate.refractive_index)
    lines = (result.x_min, result.period, result.cover_line, result.substrate_line)
    if result._locator is None:
        result._locator = hpfem.PointLocator2D(result.mesh)
    r_orders, t_orders, _ = _measure_orders(
        problem, result.solution, result._locator, wave, k0, n_cover, n_sub, lines,
        result.inputs["orders_max"], result.fourier_points,
    )  # fmt: skip
    out = np.zeros(len(rows))
    for i, (side, order) in enumerate(rows):
        match = [o for o in (r_orders if side == "R" else t_orders) if o.m == order]
        out[i] = match[0].efficiency if match else 0.0
    return out


def _setup_column(result: GratingResult, name: str, step, rows, q, scales) -> np.ndarray:
    """dR_i/d(name) for a parameter of the whole setup: the tangent with the derivative of
    the Bloch constraints (``conical_parameter_tangent``) plus the explicit dependence of the
    post-processing, both by central differences of step ``step``."""
    if result.inputs is None:
        raise GratingError("the result has no solve inputs: use hpfem.grating.solve")
    if name == "phi" and result.scalar:
        raise GratingError(
            "the phi derivative leaves the scalar E_z path: solve(..., scalar=False)"
        )
    theta, phi, omega = (result.inputs[k] for k in ("theta", "phi", "omega"))
    step = _SETUP_PARAMETERS[name] if step is None else float(step)
    if not step > 0:
        raise GratingError(f"the step of {name} must be positive")
    base = np.array([theta, phi, omega])
    if name in ("omega", "wavelength"):
        h = step * omega
        shift = np.array([0.0, 0.0, h])
    else:
        h = step
        shift = np.array([h, 0.0, 0.0]) if name == "theta" else np.array([0.0, h, 0.0])
    minus, plus = _neighbour(result, *(base - shift)), _neighbour(result, *(base + shift))
    de_e, de_v = hpfem.conical_parameter_tangent(
        result.problem, result.solution, minus[0], plus[0], h
    )
    implicit = np.real(q.T @ np.concatenate([de_e, de_v])) * np.asarray(scales)
    explicit = (_efficiencies_at(result, *plus, rows) - _efficiencies_at(result, *minus, rows)) / (
        2.0 * h
    )
    column = implicit + explicit
    if name == "wavelength":  # d/d lambda = -(omega / lambda) d/d omega
        column *= -(omega**2) / (2.0 * np.pi * hpfem.constants.c0)
    return column


def jacobian(result: GratingResult, parameters, observables=None, mode: str = "auto"):
    """Jacobian of diffraction efficiencies with respect to several parameters on the
    factorisation kept by :func:`solve` (``keep_factorisation=True``, ADR-0012).

    ``parameters`` is a sequence of

    - ``("eps", tag)``: the relative permittivity of the cells with ``tag`` (two columns,
      d/dRe eps and d/dIm eps);
    - ``("shape", velocity)``: a geometry parameter by its mesh velocity, as
      :func:`shape_sensitivity` (one column; the same condition on the measurement lines);
    - ``"theta"``, ``"phi"`` [1/rad], ``"omega"`` [1/(rad/s)] or ``"wavelength"`` (vacuum,
      [1/m]), also as ``(name, step)``: a parameter of the whole setup (one column). It moves
      the incident wave, the Bloch phases, beta, k0 and the dispersive materials (the
      ``materials`` given to :func:`solve` as models are evaluated at the new frequency; a
      ``MaterialMap`` and the stack's indices stay fixed), at a fixed PML box and fixed
      measurement lines. The derivative is the tangent on the kept factorisation with the
      derivative of the Bloch constraints (``conical_parameter_tangent``) plus the explicit
      dependence of the order post-processing, both by central differences of the step
      (default 1e-6 rad for the angles, 1e-7 relative for the frequency; no solve is
      repeated: two assemblies and two post-processings per column). Always the direct mode.
      ``"phi"`` needs the vector path (``solve(..., scalar=False)``).

    ``observables`` is a sequence of ``(side, order)`` with ``side`` ``"R"`` or ``"T"``
    (default: every propagating order of the result). ``mode`` ``"direct"`` solves once per
    parameter (:math:`de/dp = A^{-1} r_p`), ``"adjoint"`` once per observable
    (:math:`z = A^{-T} q`), ``"auto"`` (default) takes the one with fewer solves for the
    material and shape columns; both are one multi-right-hand-side solve on the kept factors
    and give the same matrix up to round-off. Returns ``(J, rows, columns)``:
    ``J[i, j] = dR_i/dp_j`` (real, shape (observables, columns)), ``rows`` the
    ``(side, order)`` pairs, ``columns`` labels such as ``"eps[3].re"``, ``"eps[3].im"``,
    ``"shape[0]"``, ``"theta"``. Raises ``GratingError`` without a kept factorisation, for an
    unknown parameter kind or mode, or an order that is not propagating."""
    kept = result.solution.factorisation
    if kept is None:
        raise GratingError(
            "jacobian needs the kept factorisation: solve(..., keep_factorisation=True)"
        )
    if mode not in ("auto", "direct", "adjoint"):
        raise GratingError(f"mode={mode!r}: use 'auto', 'direct' or 'adjoint'")
    problem, solution = result.problem, result.solution
    nd, h1 = problem.transverse_dofs, problem.longitudinal_dofs
    if observables is None:
        observables = [("R", o.m) for o in result.R_orders if o.propagating]
        observables += [("T", o.m) for o in result.T_orders if o.propagating]
    rows = [(str(side), int(order)) for side, order in observables]
    functionals, scales, q = [], [], np.zeros((kept.num_dofs, len(rows)), dtype=complex)
    for i, (side, order) in enumerate(rows):
        functional, scale = _order_functional(result, order, side)
        functionals.append(functional)
        scales.append(scale)
        if functional is not None:
            q_e, q_v = functional(nd, h1)
            q[:, i] = np.concatenate([q_e, q_v])
    # columns: ("eps" | "shape", index of the residual, velocity) or ("setup", name, step)
    residuals, columns, kinds = [], [], []
    for j, parameter in enumerate(parameters):
        kind, value = (parameter, None) if isinstance(parameter, str) else parameter
        if kind == "eps":
            kinds.append(("eps", len(residuals), None))
            residuals.append(
                hpfem.conical_material_residual_derivative(problem, solution, int(value))
            )
            columns += [f"eps[{int(value)}].re", f"eps[{int(value)}].im"]
        elif kind == "shape":
            velocity = np.asarray(value, dtype=float)
            _check_lines_fixed(result, velocity)
            kinds.append(("shape", len(residuals), velocity))
            residuals.append(hpfem.conical_shape_residual_derivative(problem, solution, velocity))
            columns.append(f"shape[{j}]")
        elif kind in _SETUP_PARAMETERS:
            kinds.append(("setup", kind, value))
            columns.append(kind)
        else:
            raise GratingError(
                f"parameter kind {kind!r}: use 'eps', 'shape', 'theta', 'phi', 'omega' or "
                "'wavelength'"
            )
    r = np.column_stack(residuals) if residuals else np.zeros((kept.num_dofs, 0), dtype=complex)
    if mode == "auto":
        mode = "direct" if r.shape[1] <= len(rows) else "adjoint"
    if r.shape[1] == 0:
        dq = np.zeros((len(rows), 0), dtype=complex)
    elif mode == "direct":
        dq = q.T @ kept.solve_many(r)  # (observables, parameters), complex
    else:
        dq = kept.solve_adjoint_many(q).T @ r
    e = np.concatenate([solution.transverse, solution.longitudinal])
    jac = np.zeros((len(rows), len(columns)))
    col = 0
    for kind, index, data in kinds:
        if kind == "setup":
            jac[:, col] = _setup_column(result, index, data, rows, q, scales)
            col += 1
            continue
        for i in range(len(rows)):
            if functionals[i] is None:
                continue
            if kind == "eps":
                jac[i, col] = scales[i] * np.real(dq[i, index])
                jac[i, col + 1] = scales[i] * np.real(1j * dq[i, index])
            else:
                term = hpfem.conical_functional_shape_derivative(nd, h1, functionals[i], data)
                jac[i, col] = scales[i] * np.real(dq[i, index] + term @ e)
        col += 2 if kind == "eps" else 1
    return jac, rows, columns


def estimate_memory(mesh, order=4, solver=None):
    """Predicted sizes of the factorisation of :func:`solve` on ``mesh`` with the polynomial
    ``order`` (``hpfem.MemoryEstimate``: DoFs, nonzeros, factor entries, bytes, backend);
    ``solver`` is the ``DirectSolverBackend`` (default: the automatic choice)."""
    backend = hpfem.DirectSolverBackend.AUTO if solver is None else solver
    if np.isscalar(order):
        return hpfem.estimate_memory(mesh, int(order), backend, True)
    orders = [int(p) for p in order]
    nd = hpfem.NedelecDofMap2D(mesh, orders)
    h1 = hpfem.DofMap2D(mesh, orders)
    return hpfem.estimate_memory(nd, backend, False, h1)


def validate(
    mesh,
    materials,
    stack,
    polarisation,
    theta: float,
    phi: float,
    omega: float,
    order=4,
    *,
    pml=None,
    bottom: str = "pml",
    orders_max: int = 3,
    snap_tolerance: float = 1e-9,
    pml_target: float = 1e-6,
    pml_wavelengths: float = 0.5,
    cover_line: float | None = None,
    substrate_line: float | None = None,
) -> list:
    """The diagnostics of :func:`solve` without solving (:mod:`hpfem.diagnostics`): a list of
    ``Diagnostic`` with ``code``, ``severity``, ``text`` and ``hint``; a failing set-up
    (straddling cells, bad options) is reported as the single error ``setup``."""
    from hpfem import diagnostics as dg

    try:
        prepared = _prepare(
            mesh,
            materials,
            stack,
            polarisation,
            theta,
            phi,
            omega,
            pml=pml,
            bottom=bottom,
            orders_max=orders_max,
            snap_tolerance=snap_tolerance,
            pml_target=pml_target,
            pml_wavelengths=pml_wavelengths,
            cover_line=cover_line,
            substrate_line=substrate_line,
        )
    except (GratingError, ValueError, RuntimeError) as error:
        return [dg.Diagnostic("setup", "error", str(error), "")]
    return _diagnostics(mesh, prepared, stack, materials, order, bottom, orders_max, omega)


def solve(
    mesh,
    materials,
    stack,
    polarisation,
    theta: float,
    phi: float,
    omega: float,
    order=4,
    *,
    pml=None,
    bottom: str = "pml",
    orders_max: int = 3,
    snap_tolerance: float = 1e-9,
    pml_target: float = 1e-6,
    pml_wavelengths: float = 0.5,
    cover_line: float | None = None,
    substrate_line: float | None = None,
    fourier_points: int = 256,
    extra_quadrature_order: int = 4,
    solver=None,
    check: bool = True,
    progress=None,
    cancel=None,
    scalar="auto",
    keep_factorisation: bool = False,
) -> GratingResult:
    """Solves the grating unit cell and returns a :class:`GratingResult`.

    ``mesh``: the unit cell (``hpfem.Mesh2D``) including the PML regions, Bloch faces tagged
    ``box_tag.X_MIN`` / ``X_MAX``, scatterer cells tagged. ``materials``: ``MaterialMap`` or a
    ``{tag: Material}`` dict (dispersive models are evaluated at ``omega``; untagged cells take
    the stack's material at their position; the map's background is the incidence medium).
    ``stack``: ``LayerStack2D`` with the normal along ``y``. ``polarisation``: ``"s"`` / ``"p"``
    (``"Ez"`` / ``"Hz"``) or ``hpfem.Polarisation``. ``theta``, ``phi`` in radians
    (``hpfem.units.deg``), ``omega`` in rad/s. ``order``: the polynomial order (int) or per-cell
    orders.

    ``pml``: ``None`` designs the layers (thickness ``pml_wavelengths`` local wavelengths rounded
    up to whole cells of the mesh's median cell height, profile ``PmlProfile.for_angle`` for the
    largest propagating-order angle in cover and substrate, reflectance target ``pml_target``,
    reference index ``min(n_cover, n_substrate)``), a ``{"top": t, "bottom": t}`` dict gives the
    thicknesses [m] with that profile, and a ``PmlBox2D`` is used as given. ``bottom="pec"``
    puts no PML at the bottom (a thick lossy substrate ending on the PEC wall; no transmitted
    orders). Stack interfaces are snapped onto mesh vertices closer than
    ``snap_tolerance`` × period; cells straddling an interface raise :class:`GratingError`.

    With ``check`` (default) the diagnostics of :func:`validate` run first: an error-level
    diagnostic raises :class:`GratingError`, warnings and infos are kept in
    ``result.diagnostics``.

    The reflected orders are measured on ``cover_line`` (default: midway between the highest
    scatterer vertex and the top PML), the transmitted ones on ``substrate_line`` (default:
    midway between the lowest scatterer vertex and the bottom PML; skipped for a lossy
    substrate or ``bottom="pec"``). ``A`` is the absorbed power of the lossy cells per period
    divided by the incident power, ``power_balance_residual = R + T + A - 1``.

    ``progress(event)`` is called with an ``hpfem.ProgressEvent`` at the start of every phase
    of the solve (assembly, constraints, factorisation, solve, post) and when it is done;
    ``cancel()`` is polled at the same moments and raises ``hpfem.Cancelled`` when it returns
    true. ``result.timing`` holds the seconds of the setup, the solve (with the phases as
    ``solver.<phase>``) and the post-processing. :func:`estimate_memory` predicts the sizes
    of the factorisation before a run.

    ``scalar``: ``"auto"`` (default) takes the scalar E_z path of ``ConicalScattering``
    (``scalar_ez``: only the H1 block is factorised, about a third of the DoFs, identical
    results) for the s polarisation at ``phi = 0``; ``True`` / ``False`` force it on or off
    (``True`` raises for other polarisations or azimuths). ``result.scalar`` says which.

    ``keep_factorisation=True`` keeps the factorised system in ``result.solution``
    (``solution.factorisation``, ADR-0012): :func:`sensitivity` and :func:`shape_sensitivity`
    then cost one transposed solve each instead of an assembly and a factorisation, at the
    price of holding the factors while the result lives.
    """
    from hpfem import diagnostics as dg

    t0 = time.perf_counter()
    timing: dict[str, float] = {}
    pr = _prepare(
        mesh,
        materials,
        stack,
        polarisation,
        theta,
        phi,
        omega,
        pml=pml,
        bottom=bottom,
        orders_max=orders_max,
        snap_tolerance=snap_tolerance,
        pml_target=pml_target,
        pml_wavelengths=pml_wavelengths,
        cover_line=cover_line,
        substrate_line=substrate_line,
    )
    found = []
    if check:
        found = _diagnostics(mesh, pr, stack, materials, order, bottom, orders_max, omega)
        if dg.errors(found):
            raise GratingError("; ".join(str(d) for d in dg.errors(found)))
    period, x_min, k0, wave, box = pr.period, pr.x_min, pr.k0, pr.wave, pr.box
    material_map, n_cover, n_sub = pr.material_map, pr.n_cover, pr.n_sub
    cover_line, substrate_line, transmitted = pr.cover_line, pr.substrate_line, pr.transmitted
    timing["setup"] = time.perf_counter() - t0

    # --- solve -------------------------------------------------------------------------------
    t1 = time.perf_counter()
    orders = [int(order)] * mesh.num_cells if np.isscalar(order) else [int(p) for p in order]
    nd = hpfem.NedelecDofMap2D(mesh, orders)
    h1 = hpfem.DofMap2D(mesh, orders)
    if scalar == "auto":
        scalar = _polarisation(polarisation) == hpfem.Polarisation.S and float(phi) == 0.0
    elif scalar not in (True, False):
        raise GratingError(f"scalar={scalar!r}: use 'auto', True or False")
    setup = _conical_setup(
        omega, wave, material_map, stack, box, period, extra_quadrature_order, scalar
    )
    setup.keep_factorisation = bool(keep_factorisation)
    if solver is not None:
        setup.solver = solver
    if progress is not None or cancel is not None:

        def report(event):
            if progress is not None:
                progress(event)
            return not (cancel is not None and cancel())

        setup.progress = report
    try:
        problem = hpfem.ConicalScattering(nd, h1, setup)
    except Exception as error:  # the solver's own diagnostics, with the hint
        raise GratingError(f"{error} (snapped {pr.moved} vertices)") from error
    solution = problem.solve()
    timing["solve"] = time.perf_counter() - t1
    for phase, seconds in solution.timing.items():
        if phase != "total":
            timing[f"solver.{phase}"] = seconds

    # --- orders, absorption, balance ------------------------------------------------------------
    t2 = time.perf_counter()
    locator = hpfem.PointLocator2D(mesh)
    lines = (x_min, period, cover_line, substrate_line if transmitted else None)
    r_orders, t_orders, incident_physical = _measure_orders(
        problem, solution, locator, wave, k0, n_cover, n_sub, lines, orders_max, fourier_points
    )
    absorbed = hpfem.absorbed_power_by_tag(problem, solution)
    # incident power per period and unit length: |E0| = 1 V/m, S.n = n cos(theta) / (2 Z0)
    incident_power = 0.5 * period * wave.ky / (k0 * hpfem.constants.Z0)
    # the flux-based balance through the PML boundaries (mesh lines on structured cells)
    flux_balance = None
    try:
        reflection = hpfem.Surface2D.plane(mesh, 1, pr.y_max - pr.t_top, 1)
        transmission = (
            hpfem.Surface2D.plane(mesh, 1, pr.y_min + pr.t_bottom, -1) if transmitted else None
        )
        balance = hpfem.conical_power_balance(
            problem, solution, reflection, period, wave.ky, incident_physical, 1.0,
            transmission, extra_quadrature_order,
        )  # fmt: skip
        flux_balance = {
            "incident": balance.incident,
            "reflected": balance.reflected,
            "transmitted": balance.transmitted,
            "absorbed": balance.absorbed,
            "relative_residual": balance.relative_residual(),
        }
    except (hpfem.InvalidArgument, ValueError, RuntimeError):
        flux_balance = None
    a_total = float(absorbed.total) / incident_power
    a_by_tag = {int(t): float(p) / incident_power for t, p in absorbed.by_tag.items()}
    r_total = sum(o.efficiency for o in r_orders)
    t_total = sum(o.efficiency for o in t_orders)
    timing["postprocess"] = time.perf_counter() - t2
    timing["total"] = time.perf_counter() - t0
    return GratingResult(
        R_orders=r_orders,
        T_orders=t_orders,
        R=r_total,
        T=t_total,
        A=a_total,
        A_by_tag=a_by_tag,
        power_balance_residual=r_total + t_total + a_total - 1.0,
        wave=wave,
        problem=problem,
        solution=solution,
        mesh=mesh,
        pml=box,
        cover_line=cover_line,
        substrate_line=substrate_line,
        dofs=int(len(problem.free_dofs)),
        timing=timing,
        diagnostics=found,
        flux_balance=flux_balance,
        scalar=bool(scalar),
        period=period,
        x_min=x_min,
        fourier_points=int(fourier_points),
        inputs={
            "materials": materials,
            "stack": stack,
            "polarisation": polarisation,
            "theta": float(theta),
            "phi": float(phi),
            "omega": float(omega),
            "orders_max": int(orders_max),
            "extra_quadrature_order": int(extra_quadrature_order),
            "solver": solver,
        },
        _locator=locator,
    )


# --- M17: dipole emitters (Stage A, ADR-0013) ----------------------------------------------------


@dataclass
class EmissionOrder:
    """One Floquet order of the field radiated by a phased emitter array: ``m``, ``side``
    (``"up"`` into the cover, ``"down"`` into the substrate), the radiated ``power`` per period
    [W/m, per unit β], the vector ``amplitude`` (E_x, E_y, E_z) on the measurement line, the
    tangential wavenumber ``kt`` = kx + 2πm/P and the normal wavenumber ``kn``."""

    m: int
    side: str
    power: float
    amplitude: np.ndarray
    kt: float
    kn: complex
    propagating: bool


@dataclass
class EmissionResult:
    """Result of :func:`emit` (one cell problem of the array scanning, ADR-0013): the power
    ``P_cell`` delivered by the source per period [W/m per unit β], the radiated orders
    ``orders_up`` / ``orders_down`` with their powers and sums ``up`` / ``down``, the Poynting
    fluxes ``flux_up`` / ``flux_down`` through the PML boundaries (``None`` when they are not
    on mesh lines), the ``absorbed`` power and ``A_by_tag``, the ``guided`` remainder
    ``P_cell - flux_up - flux_down - absorbed`` (the order sums where no flux is available),
    ``kx``, ``beta``, ``omega`` and the problem, solution, mesh and measurement lines."""

    P_cell: float
    orders_up: list[EmissionOrder]
    orders_down: list[EmissionOrder]
    up: float
    down: float
    flux_up: float | None
    flux_down: float | None
    absorbed: float
    A_by_tag: dict[int, float]
    guided: float
    kx: float
    beta: float
    omega: float
    problem: object
    solution: object
    mesh: object
    pml: object
    cover_line: float
    substrate_line: float | None
    dofs: int
    timing: dict[str, float] = field(default_factory=dict)
    _locator: object = None

    def field(self, points, quantity: str = "E"):
        """The field of the cell problem at ``points`` (k, 2) (``ConicalScattering.sample``)."""
        if self._locator is None:
            self._locator = hpfem.PointLocator2D(self.mesh)
        points = np.asarray(points, dtype=float)
        values, _ = self.problem.sample(self.solution, self._locator, points, quantity=quantity)
        return values


def _emission_orders(problem, solution, locator, k0, n_line, period, x_min, line, kx, beta,
                     orders_max, fourier_points, side):  # fmt: skip
    coefficients = hpfem.conical_fourier_coefficients(
        lambda x: np.asarray(problem.total_field(solution, locator, x)),
        [x_min, line], [1.0, 0.0], period, kx, orders_max, fourier_points,
    )  # fmt: skip
    z0 = hpfem.constants.Z0
    out = []
    for i, amplitude in enumerate(coefficients):
        m = i - orders_max
        kt = kx + 2 * np.pi * m / period
        kn = np.sqrt(complex((k0 * n_line) ** 2 - kt**2 - beta**2))
        amplitude = np.asarray(amplitude, dtype=complex)
        propagating = abs(kn.imag) < 1e-12 * k0 and kn.real > 0
        power = period * kn.real * float(np.vdot(amplitude, amplitude).real) / (2 * k0 * z0)
        out.append(EmissionOrder(m, side, power if propagating else 0.0, amplitude, float(kt),
                                 complex(kn), bool(propagating)))  # fmt: skip
    return out


def emit(
    mesh,
    materials,
    stack,
    dipole,
    omega: float,
    kx: float = 0.0,
    beta: float = 0.0,
    order=4,
    *,
    pml=None,
    bottom: str = "pml",
    orders_max: int = 3,
    snap_tolerance: float = 1e-9,
    pml_target: float = 1e-6,
    pml_wavelengths: float = 0.5,
    cover_line: float | None = None,
    substrate_line: float | None = None,
    fourier_points: int = 256,
    extra_quadrature_order: int = 4,
    solver=None,
    progress=None,
    cancel=None,
    keep_factorisation: bool = False,
) -> EmissionResult:
    """Emission of a Bloch-periodic array of Gaussian dipoles (M17 Stage A, ADR-0013): one
    dipole per period with the phase e^{i kx P} from cell to cell and the dependence e^{iβz}
    along the lines — the cell problem of the array scanning of a single dipole.

    ``dipole`` is a mapping with ``position`` (x0, y0) [m], ``moment`` (p_x, p_y, p_z) [A m],
    the physical current moment in the solver frame (x along the period, y the stack normal, z
    along the lines), and ``sigma`` [m], the Gaussian smearing (the source of
    ``hpfem.conical_gaussian_dipole``, its z-smearing as the factor exp(-σ²β²/2)). The
    Gaussian must lie 6σ inside the cell, above the bottom and below the top PML, in a lossless
    medium. The cell, the materials, the stack, the PML and the measurement lines are those of
    :func:`solve` (same arguments); cells whose tag is not in ``materials`` take the stack
    material at their centroid (a given ``hpfem.MaterialMap`` is used as it is). The PML is
    designed for the direction of (kx, β) in the cover, up to 80 degrees. Returns an
    :class:`EmissionResult`; powers in W/m per unit β (the integrand of the array scanning).
    Raises ``GratingError`` for an invalid dipole or geometry."""
    t0 = time.perf_counter()
    timing: dict[str, float] = {}
    k0 = float(units.vacuum_wavenumber(omega))
    n_cover = complex(stack.incidence_medium.refractive_index)
    position = np.asarray(dipole["position"], dtype=float)
    moment = np.asarray(dipole["moment"], dtype=complex)
    sigma = float(dipole["sigma"])
    if position.shape != (2,) or moment.shape != (3,) or not sigma > 0:
        raise GratingError("dipole: position (x0, y0), moment (px, py, pz) and sigma > 0 needed")
    # PML and measurement geometry of solve() for the equivalent direction of (kx, beta)
    s = min(math.hypot(kx, beta) / (k0 * n_cover.real), math.sin(80 * units.deg))
    pr = _prepare(
        mesh, materials, stack, "s", math.asin(s), math.atan2(beta, kx), omega, pml=pml,
        bottom=bottom, orders_max=orders_max, snap_tolerance=snap_tolerance,
        pml_target=pml_target, pml_wavelengths=pml_wavelengths, cover_line=cover_line,
        substrate_line=substrate_line,
    )  # fmt: skip
    material_map = pr.material_map
    if not isinstance(materials, hpfem.MaterialMap):
        for c in range(mesh.num_cells):
            if not material_map.has(int(mesh.cell_tag(c))):
                centroid = np.mean([mesh.vertex(int(v)) for v in mesh.cell_vertices(c)], axis=0)
                material_map.set_cell(
                    c, stack.material_at([float(centroid[0]), float(centroid[1])])
                )
    x0, y0 = float(position[0]), float(position[1])
    if not (pr.x_min + 6 * sigma <= x0 <= pr.x_min + pr.period - 6 * sigma):
        raise GratingError("dipole: the Gaussian must lie 6 sigma inside the cell along x")
    if not (pr.y_min + pr.t_bottom + 6 * sigma <= y0 <= pr.y_max - pr.t_top - 6 * sigma):
        raise GratingError("dipole: the Gaussian must lie 6 sigma outside the PML layers")
    locator = hpfem.PointLocator2D(mesh)
    located = locator.locate([x0, y0])
    if located is None:
        raise GratingError("dipole: the position lies outside the mesh")
    tag = int(mesh.cell_tag(int(located.cell)))
    host = material_map.at(tag) if material_map.has(tag) else stack.material_at([x0, y0])
    if abs(complex(host.eps_r).imag) > 1e-12:
        raise GratingError("dipole: the emitter must lie in a lossless medium (ADR-0013)")
    orders = [int(order)] * mesh.num_cells if np.isscalar(order) else [int(p) for p in order]
    nd = hpfem.NedelecDofMap2D(mesh, orders)
    h1 = hpfem.DofMap2D(mesh, orders)
    setup = hpfem.ConicalScatteringSetup()
    setup.omega = float(omega)
    setup.beta = float(beta)
    setup.materials = material_map
    setup.current = hpfem.conical_gaussian_dipole(
        [x0, y0], moment, sigma, float(omega), float(beta)
    )
    setup.pml = pr.box
    setup.pec_tags = [hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX]
    setup.periodic = [
        hpfem.PeriodicPair2D(hpfem.box_tag.X_MIN, hpfem.box_tag.X_MAX, [pr.period, 0.0],
                             hpfem.bloch_phase([float(kx), 0.0], [pr.period, 0.0]))
    ]  # fmt: skip
    setup.extra_quadrature_order = int(extra_quadrature_order)
    setup.keep_factorisation = bool(keep_factorisation)
    if solver is not None:
        setup.solver = solver
    if progress is not None or cancel is not None:

        def report(event):
            if progress is not None:
                progress(event)
            return not (cancel is not None and cancel())

        setup.progress = report
    t1 = time.perf_counter()
    problem = hpfem.ConicalScattering(nd, h1, setup)
    solution = problem.solve()
    timing["solve"] = time.perf_counter() - t1
    t2 = time.perf_counter()
    p_cell = float(hpfem.conical_source_power(problem, solution))
    n_sub = pr.n_sub
    orders_up = _emission_orders(problem, solution, locator, k0, n_cover.real, pr.period,
                                 pr.x_min, pr.cover_line, kx, beta, orders_max, fourier_points,
                                 "up")  # fmt: skip
    orders_down = []
    if pr.transmitted:
        orders_down = _emission_orders(problem, solution, locator, k0, n_sub.real, pr.period,
                                       pr.x_min, pr.substrate_line, kx, beta, orders_max,
                                       fourier_points, "down")  # fmt: skip
    up = sum(o.power for o in orders_up)
    down = sum(o.power for o in orders_down)

    def flux(surface) -> float:
        return float(hpfem.conical_poynting_flux(
            nd, h1, solution.transverse, solution.longitudinal, float(beta), float(omega),
            material_map, surface,
        ))  # fmt: skip

    flux_up = flux_down = None
    try:
        flux_up = flux(hpfem.Surface2D.plane(mesh, 1, pr.y_max - pr.t_top, 1))
        flux_down = 0.0
        if bottom == "pml":
            flux_down = flux(hpfem.Surface2D.plane(mesh, 1, pr.y_min + pr.t_bottom, -1))
    except (hpfem.InvalidArgument, ValueError, RuntimeError):
        flux_up = flux_down = None
    absorbed = hpfem.absorbed_power_by_tag(problem, solution)
    leaving = (flux_up + flux_down) if flux_up is not None else (up + down)
    guided = p_cell - leaving - float(absorbed.total)
    timing["postprocess"] = time.perf_counter() - t2
    timing["total"] = time.perf_counter() - t0
    return EmissionResult(
        P_cell=p_cell, orders_up=orders_up, orders_down=orders_down, up=up, down=down,
        flux_up=flux_up, flux_down=flux_down, absorbed=float(absorbed.total),
        A_by_tag={int(t): float(p) for t, p in absorbed.by_tag.items()}, guided=guided,
        kx=float(kx), beta=float(beta), omega=float(omega), problem=problem, solution=solution,
        mesh=mesh, pml=pr.box, cover_line=pr.cover_line,
        substrate_line=pr.substrate_line if pr.transmitted else None,
        dofs=int(nd.num_dofs + h1.num_dofs), timing=timing, _locator=locator,
    )  # fmt: skip


# --- M17 Stage C: emission pattern by reciprocity (ADR-0013 §6) ----------------------------------


@dataclass
class EmissionPattern:
    """Result of :func:`emission_pattern`: the emitted power per unit solid angle
    ``dP_dOmega`` (num_directions, num_pol) [W/sr], divided by ``P_bulk`` when ``normalized``,
    for the ``directions`` ``(theta, phi, side)`` and polarisations ``pol``; ``P_bulk`` the
    power of the same Gaussian dipole in the homogeneous host medium [W]; ``amplitude``
    (num_directions, num_pol) the reciprocity amplitude p · ⟨E_pw⟩ (complex) and ``n`` the index
    of the medium of every direction."""

    directions: list[tuple[float, float, str]]
    pol: tuple[str, ...]
    dP_dOmega: np.ndarray
    P_bulk: float
    normalized: bool
    amplitude: np.ndarray
    n: np.ndarray
    timing: dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> np.ndarray:
        """dP/dΩ summed over the polarisations, per direction."""
        return self.dP_dOmega.sum(axis=1)


def _mirrored_cell(mesh):
    """The mesh mirrored at y = 0 (y → −y), cells reoriented, sides tagged by position."""
    if mesh.geometry_order != 1:
        raise GratingError("emission_pattern: directions into the substrate need a straight mesh")
    vertices = np.asarray(mesh.vertices, dtype=float) * np.array([1.0, -1.0])
    cells = np.asarray(mesh.cells, dtype=int)[:, [0, 2, 1]]  # the reflection flips the order
    out = hpfem.Mesh2D([tuple(v) for v in vertices], [tuple(c) for c in cells],
                       [int(t) for t in mesh.cell_tags])  # fmt: skip
    x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
    y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
    tol = 1e-9 * (x_max - x_min)
    for f in out.boundary_facets:
        a, b = (out.vertex(int(v)) for v in out.facet_vertices(f))
        mx, my = 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])
        if abs(mx - x_min) < tol:
            out.set_facet_tag(f, hpfem.box_tag.X_MIN)
        elif abs(mx - x_max) < tol:
            out.set_facet_tag(f, hpfem.box_tag.X_MAX)
        elif abs(my - y_min) < tol:
            out.set_facet_tag(f, hpfem.box_tag.Y_MIN)
        elif abs(my - y_max) < tol:
            out.set_facet_tag(f, hpfem.box_tag.Y_MAX)
    return out


def _mirrored_stack(stack):
    """The layer stack mirrored at y = 0: the substrate becomes the incidence medium."""
    n_layers = int(stack.num_layers)
    layers = [
        hpfem.Layer(stack.material(n_layers - i),
                    float(stack.interface(n_layers - i - 1) - stack.interface(n_layers - i)))
        for i in range(n_layers)
    ]  # fmt: skip
    return hpfem.LayerStack2D(stack.substrate, layers, stack.incidence_medium, -float(stack.bottom))


def _gauss_hermite_points(position, sigma: float, points: int):
    """Tensor Gauss–Hermite nodes and weights of the in-plane Gaussian g2 around ``position``."""
    xi, w = np.polynomial.hermite.hermgauss(int(points))
    gx, gy = np.meshgrid(xi, xi, indexing="ij")
    wx, wy = np.meshgrid(w, w, indexing="ij")
    nodes = np.column_stack([gx.ravel(), gy.ravel()]) * (np.sqrt(2.0) * sigma) + position
    return nodes, (wx * wy).ravel() / np.pi


def emission_pattern(
    mesh,
    materials,
    stack,
    dipole,
    omega: float,
    directions,
    pol=("s", "p"),
    *,
    normalized: bool = False,
    quadrature_points: int = 6,
    order=4,
    pml=None,
    **solve_kwargs,
) -> EmissionPattern:
    """Angle-resolved far-field emission of a single Gaussian dipole by reciprocity (M17
    Stage C, ADR-0013 §6): the power per unit solid angle radiated into the direction
    ``(theta, phi, side)`` with polarisation ``pol``.

    ``side`` ``"up"``: the direction r̂ = (sinθ cosφ, cosθ, sinθ sinφ) into the cover (x along
    the period, y the stack normal, z along the lines; the direction of the specular order of
    a wave incident at (θ, φ)); ``"down"``: r̂ = (sinθ cosφ, −cosθ, sinθ sinφ) into a lossless
    substrate. Lorentz reciprocity with a distant dipole in that direction gives

    .. math:: \\frac{dP}{d\\Omega} = \\frac{n k_0^2 Z_0}{32\\pi^2}
              \\left|p\\cdot\\langle E_{pw}\\rangle\\right|^2 ,

    with the index n of the medium of the direction and the total field E_pw of the unit
    plane wave incident from r̂ (travelling along −r̂, polarisation s or p of that plane of
    incidence: :func:`solve` at (θ, φ + π); directions into the substrate on the problem
    mirrored at y = 0), averaged over the dipole's Gaussian: the in-plane g2 by a
    ``quadrature_points``² Gauss–Hermite rule on the solved field, the z-smearing as the factor
    exp(−σ²β²/2) of the direction's β. For a homogeneous medium the sum over s and p is
    n k0² Z0 |p⊥|² e^{−(n k0 σ)²}/(32π²), whose integral over the sphere is
    ``P_bulk = dipole_vacuum_power(|p|) · n · e^{−(n k0 σ)²}``. One solve per direction and
    polarisation, no singularities; it gives the radiated part only (not the guided or the
    absorbed power, not the total Purcell factor).

    ``dipole``: ``position`` (x0, y0), ``moment`` (p_x, p_y, p_z) [A m], ``sigma`` [m], as
    :func:`emit`. ``mesh``, ``materials``, ``stack``, ``order``, ``pml`` (``None`` or a
    ``{"top", "bottom"}`` dict for directions into the substrate) and ``solve_kwargs`` as
    :func:`solve`. ``normalized`` divides by ``P_bulk``. Returns an :class:`EmissionPattern`.
    Raises ``GratingError`` for a lossy medium of a direction, an invalid dipole or a curved
    mesh with directions into the substrate."""
    t0 = time.perf_counter()
    k0 = float(units.vacuum_wavenumber(omega))
    position = np.asarray(dipole["position"], dtype=float)
    moment = np.asarray(dipole["moment"], dtype=complex)
    sigma = float(dipole["sigma"])
    if position.shape != (2,) or moment.shape != (3,) or not sigma > 0:
        raise GratingError("dipole: position (x0, y0), moment (px, py, pz) and sigma > 0 needed")
    pol = tuple(str(p) for p in pol)
    directions = [(float(t), float(f), str(s)) for t, f, s in directions]
    for theta, _phi, side in directions:
        if side not in ("up", "down"):
            raise GratingError(f"emission_pattern: side {side!r}, use 'up' or 'down'")
        if not 0.0 <= theta < 0.5 * np.pi:
            raise GratingError("emission_pattern: theta must lie in [0, pi/2)")
    # the host medium of the dipole (for P_bulk)
    material_map = _material_map(materials, stack, float(omega))
    locator = hpfem.PointLocator2D(mesh)
    located = locator.locate([float(position[0]), float(position[1])])
    if located is None:
        raise GratingError("dipole: the position lies outside the mesh")
    tag = int(mesh.cell_tag(int(located.cell)))
    host = material_map.at(tag) if material_map.has(tag) else stack.material_at(list(position))
    n_host = complex(host.refractive_index)
    if abs(n_host.imag) > 1e-12:
        raise GratingError("dipole: the emitter must lie in a lossless medium (ADR-0013)")
    p_norm = float(np.linalg.norm(moment))
    p_bulk = (
        float(hpfem.dipole_vacuum_power(p_norm, float(omega)))
        * n_host.real
        * math.exp(-((n_host.real * k0 * sigma) ** 2))
    )
    # the problems: as given for "up", mirrored at y = 0 for "down"
    frames = {"up": (mesh, stack, pml, position, moment)}
    if any(side == "down" for _t, _f, side in directions):
        if solve_kwargs.get("bottom", "pml") != "pml":
            raise GratingError("emission_pattern: directions into the substrate need bottom='pml'")
        if pml is not None and not isinstance(pml, Mapping):
            raise GratingError("emission_pattern: give the PML as None or {'top', 'bottom'}")
        mirrored_pml = None if pml is None else {"top": pml.get("bottom", 0.0),
                                                 "bottom": pml.get("top", 0.0)}  # fmt: skip
        frames["down"] = (_mirrored_cell(mesh), _mirrored_stack(stack), mirrored_pml,
                          position * np.array([1.0, -1.0]),
                          moment * np.array([1.0, -1.0, 1.0]))  # fmt: skip
    z0 = hpfem.constants.Z0
    amplitude = np.zeros((len(directions), len(pol)), dtype=complex)
    n_dir = np.zeros(len(directions))
    nodes_cache = {}
    for i, (theta, phi, side) in enumerate(directions):
        cell_mesh, cell_stack, cell_pml, x0, p = frames[side]
        n_medium = complex(cell_stack.incidence_medium.refractive_index)
        if abs(n_medium.imag) > 1e-12:
            raise GratingError(f"emission_pattern: the medium of the {side} directions is lossy")
        n_dir[i] = n_medium.real
        if side not in nodes_cache:
            nodes_cache[side] = _gauss_hermite_points(x0, sigma, quadrature_points)
        nodes, weights = nodes_cache[side]
        for j, polarisation in enumerate(pol):
            result = solve(cell_mesh, materials, cell_stack, polarisation, theta, phi + np.pi,
                           omega, order, pml=cell_pml, **solve_kwargs)  # fmt: skip
            values = np.asarray(result.field(nodes, quantity="E"), dtype=complex)
            if not np.all(np.isfinite(values)):
                raise GratingError("emission_pattern: the dipole's Gaussian leaves the mesh")
            mean = weights @ values  # the in-plane Gaussian average of (E_x, E_y, E_z)
            smear = math.exp(-0.5 * (sigma * float(result.wave.beta)) ** 2)
            amplitude[i, j] = smear * complex(p @ mean)
    scale = n_dir[:, None] * k0**2 * z0 / (32 * np.pi**2)
    d_p = scale * np.abs(amplitude) ** 2
    if normalized:
        d_p = d_p / p_bulk
    return EmissionPattern(directions, pol, d_p, p_bulk, bool(normalized), amplitude, n_dir,
                           {"total": time.perf_counter() - t0})  # fmt: skip


__all__ = [
    "EmissionOrder",
    "EmissionPattern",
    "EmissionResult",
    "GratingError",
    "GratingResult",
    "Order",
    "emission_pattern",
    "emit",
    "solve",
    "validate",
]

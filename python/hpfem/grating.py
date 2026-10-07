"""One-call periodic scattering (M15 F2): :func:`solve` runs the conical solver on a
grating unit cell and returns the diffraction efficiencies, the vector amplitudes, the
absorbed power and the power balance.

Frame (as :class:`hpfem.ConicalScattering`): the period along ``x``, the stack normal along
``y`` (layers above / below the structure), the structure invariant along ``z``. The mesh is
the unit cell including the PML regions at its top and bottom, with the Bloch faces tagged
``box_tag.X_MIN`` / ``box_tag.X_MAX`` and the material cells tagged; the layered background
is a :class:`hpfem.LayerStack2D`. The incident plane wave of unit amplitude comes from the
incidence medium at the polar angle ``theta`` against the normal and the azimuth ``phi`` about
it (``phi = 0``: the plane of incidence is the x–y plane, ``s`` is the ``E_z`` polarisation).
"""

from __future__ import annotations

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


def _material_map(materials, stack) -> hpfem.MaterialMap:
    if isinstance(materials, hpfem.MaterialMap):
        return materials
    out = hpfem.MaterialMap(stack.incidence_medium)
    for tag, material in dict(materials).items():
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
) -> GratingResult:
    """Solves the grating unit cell and returns a :class:`GratingResult`.

    ``mesh``: the unit cell (``hpfem.Mesh2D``) including the PML regions, Bloch faces tagged
    ``box_tag.X_MIN`` / ``X_MAX``, scatterer cells tagged. ``materials``: ``MaterialMap`` or a
    ``{tag: Material}`` dict (untagged cells take the stack's material at their position; the
    map's background is the incidence medium). ``stack``: ``LayerStack2D`` with the normal along
    ``y``. ``polarisation``: ``"s"`` / ``"p"`` (``"Ez"`` / ``"Hz"``) or ``hpfem.Polarisation``.
    ``theta``, ``phi`` in radians (``hpfem.units.deg``), ``omega`` in rad/s. ``order``: the
    polynomial order (int) or per-cell orders.

    ``pml``: ``None`` designs the layers (thickness ``pml_wavelengths`` local wavelengths rounded
    up to whole cells of the mesh's median cell height, profile ``PmlProfile.for_angle`` for the
    largest propagating-order angle in cover and substrate, reflectance target ``pml_target``,
    reference index ``min(n_cover, n_substrate)``), a ``{"top": t, "bottom": t}`` dict gives the
    thicknesses [m] with that profile, and a ``PmlBox2D`` is used as given. ``bottom="pec"``
    puts no PML at the bottom (a thick lossy substrate ending on the PEC wall; no transmitted
    orders). Stack interfaces are snapped onto mesh vertices closer than
    ``snap_tolerance`` × period; cells straddling an interface raise :class:`GratingError`.

    The reflected orders are measured on ``cover_line`` (default: midway between the highest
    scatterer vertex and the top PML), the transmitted ones on ``substrate_line`` (default:
    midway between the lowest scatterer vertex and the bottom PML; skipped for a lossy
    substrate or ``bottom="pec"``). ``A`` is the absorbed power of the lossy cells per period
    divided by the incident power, ``power_balance_residual = R + T + A - 1``.
    """
    t0 = time.perf_counter()
    timing: dict[str, float] = {}
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
    material_map = _material_map(materials, stack)

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
    profile = hpfem.PmlProfile.for_angle(theta_max, pml_target, 1.0, 2)
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
    timing["setup"] = time.perf_counter() - t0

    # --- solve -------------------------------------------------------------------------------
    t1 = time.perf_counter()
    orders = [int(order)] * mesh.num_cells if np.isscalar(order) else [int(p) for p in order]
    nd = hpfem.NedelecDofMap2D(mesh, orders)
    h1 = hpfem.DofMap2D(mesh, orders)
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
    if solver is not None:
        setup.solver = solver
    try:
        problem = hpfem.ConicalScattering(nd, h1, setup)
    except Exception as error:  # the solver's own diagnostics, with the hint
        raise GratingError(f"{error} (snapped {moved} vertices)") from error
    solution = problem.solve()
    timing["solve"] = time.perf_counter() - t1

    # --- orders, absorption, balance ------------------------------------------------------------
    t2 = time.perf_counter()
    locator = hpfem.PointLocator2D(mesh)
    origin_r = [x_min, cover_line]

    def reflected_field(x):
        total = problem.total_field(solution, locator, x)
        i = wave.incident(x)
        return np.asarray(total) - np.array([i[0], i[1], 1j * i[2]])

    coefficients = hpfem.conical_fourier_coefficients(
        reflected_field, origin_r, [1.0, 0.0], period, wave.kx, orders_max, fourier_points
    )
    r_orders = _orders(
        coefficients, k0, n_cover.real, period, wave.kx, wave.beta, wave.ky, orders_max
    )
    t_orders: list[Order] = []
    if transmitted:
        coefficients = hpfem.conical_fourier_coefficients(
            lambda x: np.asarray(problem.total_field(solution, locator, x)),
            [x_min, substrate_line],
            [1.0, 0.0],
            period,
            wave.kx,
            orders_max,
            fourier_points,
        )
        t_orders = _orders(
            coefficients, k0, n_sub.real, period, wave.kx, wave.beta, wave.ky, orders_max
        )
    absorbed = hpfem.absorbed_power_by_tag(problem, solution)
    # incident power per period and unit length: |E0| = 1 V/m, S.n = n cos(theta) / (2 Z0)
    incident_power = 0.5 * period * wave.ky / (k0 * hpfem.constants.Z0)
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
        _locator=locator,
    )


__all__ = ["GratingError", "GratingResult", "Order", "solve"]

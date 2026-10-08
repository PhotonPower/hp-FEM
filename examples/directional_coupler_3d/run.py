"""3D directional coupler of two silicon-on-insulator waveguides between modal ports (M12):
the power transferred from the fed waveguide to its neighbour after the length L against
coupled-mode theory, P_cross = sin^2(kappa L) with kappa = (beta_even - beta_odd) / 2 from the
supermodes of the two-core cross-section (the same 2D mode solver that builds the ports).

Run `python examples/directional_coupler_3d/run.py [--length um] [--gap nm] [--order p]
[--cell nm] [--backend auto|mumps|cudss] [--ports half|full] [--out file.json] [--quick]`;
the results go to `directional_coupler_3d.json`. Geometry (x across, y up, z along the
guides): two cores width x height in silica, PEC walls at the margins (the guided modes decay
to below 1e-2 within the margins). `--ports half` splits the end faces into four modal ports
(1: core A at z = 0, 2: core B at z = 0, 3: A at z = L, 4: B at z = L) and launches the mode of
the half section of core A; `--ports full` uses one two-mode port per face (the even and the
odd supermode) and reports the S-matrix in the supermode basis. The CPU run with --quick
(L = 2 um, p = 1, 100 nm cells) takes seconds; the production runs (L = 10-30 um, p = 2-3,
millions of DoFs) are the GPU agent's part (`benchmarks/results/`, `docs/validation.md`).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np

import hpfem
from hpfem import units

NM = units.nm
UM = units.um
N_SI = 3.48
N_SIO2 = 1.44
WAVELENGTH = 1550 * NM
TAG_CLAD, TAG_A, TAG_B = 1, 2, 3
PORT_TAGS = (11, 12, 13, 14)
BACKENDS = {
    None: None,
    "auto": None,
    "mumps": hpfem.DirectSolverBackend.MUMPS,
    "cudss": hpfem.DirectSolverBackend.CUDSS,
    "sparselu": hpfem.DirectSolverBackend.SPARSE_LU,
}


@dataclass
class Geometry:
    width: float = 450 * NM  # core width (x)
    height: float = 220 * NM  # core height (y)
    gap: float = 150 * NM
    margin_x: float = 750 * NM  # cladding beside the outer core edges
    margin_y: float = 605 * NM  # cladding below and above the cores
    length: float = 2 * UM
    cell: float = 75 * NM  # target cell size; the grid lines hit the core edges

    def x_edges(self):
        a0 = -self.gap / 2 - self.width
        b1 = self.gap / 2 + self.width
        return [a0 - self.margin_x, a0, -self.gap / 2, self.gap / 2, b1, b1 + self.margin_x]

    def y_edges(self):
        return [-self.margin_y, 0.0, self.height, self.height + self.margin_y]


def _grid(edges, cell):
    """Grid lines through all edges with about `cell` spacing in between."""
    lines = [edges[0]]
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        n = max(1, round((b - a) / cell))
        lines.extend(a + (b - a) * (i + 1) / n for i in range(n))
    return np.array(lines)


def _tag_cores(mesh, geometry: Geometry, centroids):
    e = geometry.x_edges()
    for c, x in enumerate(centroids):
        inside_y = 0.0 < x[1] < geometry.height
        if inside_y and e[1] < x[0] < e[2]:
            mesh.set_cell_tag(c, TAG_A)
        elif inside_y and e[3] < x[0] < e[4]:
            mesh.set_cell_tag(c, TAG_B)
        else:
            mesh.set_cell_tag(c, TAG_CLAD)


def section_mesh(geometry: Geometry):
    """2D cross-section (x, y) with the two cores tagged, for the coupled-mode reference."""
    xs, ys = _grid(geometry.x_edges(), geometry.cell), _grid(geometry.y_edges(), geometry.cell)
    mesh = hpfem.rectangle(len(xs) - 1, len(ys) - 1, [xs[0], ys[0]], [xs[-1], ys[-1]])
    _tag_cores(mesh, geometry, np.asarray(mesh.cell_centroids))
    return mesh


def coupler_mesh(geometry: Geometry, ports: str = "half"):
    """3D box mesh with tagged cores; ``ports="half"`` splits the end faces into the four port
    half faces, ``"full"`` keeps the box tags Z_MIN / Z_MAX for one two-mode port per face."""
    xs, ys = _grid(geometry.x_edges(), geometry.cell), _grid(geometry.y_edges(), geometry.cell)
    nz = max(1, round(geometry.length / geometry.cell))
    lower, upper = [xs[0], ys[0], 0.0], [xs[-1], ys[-1], geometry.length]
    mesh = hpfem.box(len(xs) - 1, len(ys) - 1, nz, lower, upper)
    _tag_cores(mesh, geometry, np.asarray(mesh.cell_centroids))
    if ports == "half":
        faces = ((hpfem.box_tag.Z_MIN, PORT_TAGS[:2]), (hpfem.box_tag.Z_MAX, PORT_TAGS[2:]))
        for face, (left, right) in faces:
            for f in mesh.facets_with_tag(face):
                x = np.mean([mesh.vertex(int(v)) for v in mesh.facet_vertices(int(f))], axis=0)
                mesh.set_facet_tag(int(f), left if x[0] < 0 else right)
    return mesh


def materials():
    m = hpfem.MaterialMap(hpfem.Material.dielectric(N_SIO2))
    m.set(TAG_A, hpfem.Material.dielectric(N_SI))
    m.set(TAG_B, hpfem.Material.dielectric(N_SI))
    return m


def supermodes(geometry: Geometry, order: int, solver=None):
    """beta of the even and the odd supermode of the two-core section."""
    omega = units.angular_frequency(wavelength=WAVELENGTH)
    mesh = section_mesh(geometry)
    nd, h1 = hpfem.NedelecDofMap2D(mesh, order), hpfem.DofMap2D(mesh, order)
    setup = hpfem.WaveguideSetup()
    setup.omega = omega
    setup.materials = materials()
    setup.pec_tags = [
        hpfem.box_tag.X_MIN,
        hpfem.box_tag.X_MAX,
        hpfem.box_tag.Y_MIN,
        hpfem.box_tag.Y_MAX,
    ]
    setup.num_modes = 2
    if solver is not None:
        setup.solver = solver
    modes = hpfem.PropagatingMode(nd, h1, setup).solve()
    betas = sorted((m.beta for m in modes), reverse=True)
    if len(betas) < 2:
        raise RuntimeError("the two-core section guides fewer than two modes")
    return betas[0], betas[1], omega


@dataclass
class Result:
    length: float
    gap: float
    order: int
    cell: float
    backend: str
    ports: str
    dofs: int
    beta_even: float
    beta_odd: float
    kappa: float  # 1/m
    coupling_length: float  # pi / (2 kappa): full transfer
    port_beta: list
    bar_cmt: float
    cross_cmt: float
    bar_fem: float
    cross_fem: float
    reflection: float  # |b_1|^2 P_1 / P_in
    back_coupling: float  # |b_2|^2 P_2 / P_in
    power_sum: float  # of the four outgoing channels (1 for a lossless coupler)
    seconds: dict
    # ports = "full": transmissions of the even and the odd supermode (3D S-parameters in the
    # supermode basis), their phase errors against beta L of the section solver, the leakage
    # between the supermodes; bar_fem / cross_fem are then reconstructed from the 3D phases,
    # cross = sin^2((arg t_e - arg t_o) / 2), the single-core launch of the two-mode picture
    t_even: list | None = None
    t_odd: list | None = None
    phase_error_even: float | None = None
    phase_error_odd: float | None = None
    mode_leakage: float | None = None


def run(
    *,
    length: float = 2 * UM,
    gap: float = 150 * NM,
    order: int = 1,
    cell: float = 75 * NM,
    backend=None,
    ports: str = "half",
    quick: bool = False,
    out: str | None = "directional_coupler_3d.json",
) -> Result:
    """Solves the coupler and returns the :class:`Result`; ``quick`` is the CPU test case
    (L = 2 um, gap 200 nm, p = 1, 100 nm cells). ``backend``: None / "auto", "mumps",
    "cudss" or "sparselu". ``ports``: ``"half"`` launches the mode of the half section that
    contains core A and measures the four half-face ports (the physical single-core launch;
    the port plane in the gap biases the small cross power of a short coupler by about 1 %
    absolute), ``"full"`` uses one two-mode port per end face (the even and the odd
    supermode): the S-matrix in the supermode basis, its phases against beta L of the
    section solver, and the cross power reconstructed from the 3D phases."""
    if quick:
        length, gap, order, cell = 2 * UM, 200 * NM, 1, 100 * NM
    if ports not in ("half", "full"):
        raise ValueError(f"ports={ports!r}: use 'half' or 'full'")
    geometry = Geometry(length=length, gap=gap, cell=cell)
    solver = BACKENDS[backend if backend is None else str(backend).lower()]
    seconds = {}
    t0 = time.perf_counter()
    beta_e, beta_o, omega = supermodes(geometry, order, solver)
    kappa = 0.5 * (beta_e - beta_o)
    seconds["supermodes"] = time.perf_counter() - t0
    cross_cmt = float(np.sin(kappa * length) ** 2)
    bar_cmt = 1.0 - cross_cmt

    t1 = time.perf_counter()
    mesh = coupler_mesh(geometry, ports)
    dofs = hpfem.NedelecDofMap3D(mesh, order)
    setup = hpfem.ScatteringSetup3D()
    setup.omega = omega
    setup.materials = materials()
    setup.pec_tags = [
        hpfem.box_tag.X_MIN,
        hpfem.box_tag.X_MAX,
        hpfem.box_tag.Y_MIN,
        hpfem.box_tag.Y_MAX,
    ]
    if solver is not None:
        setup.solver = solver
    extra = {}
    if ports == "full":
        setup.ports = [
            hpfem.WaveguidePort(hpfem.box_tag.Z_MIN, 2),
            hpfem.WaveguidePort(hpfem.box_tag.Z_MAX, 2),
        ]
        seconds["setup"] = time.perf_counter() - t1
        t2 = time.perf_counter()
        sp = hpfem.s_parameters(dofs, setup)  # four solves: both supermodes from both ends
        seconds["solve"] = time.perf_counter() - t2
        channel = {(c.port, c.mode): i for i, c in enumerate(sp.channels)}
        if len(channel) != 4:
            raise RuntimeError(f"expected two propagating modes per port, got {len(channel)}")
        t_e = sp.s[channel[(1, 0)], channel[(0, 0)]]
        t_o = sp.s[channel[(1, 1)], channel[(0, 1)]]
        leakage = max(
            abs(sp.s[channel[(1, 1)], channel[(0, 0)]]), abs(sp.s[channel[(1, 0)], channel[(0, 1)]])
        )
        reflection = abs(sp.s[channel[(0, 0)], channel[(0, 0)]]) ** 2
        back = abs(sp.s[channel[(0, 1)], channel[(0, 0)]]) ** 2
        port_beta = [sp.channels[channel[(0, m)]].beta.real for m in (0, 1)]
        cross_3d = float(np.sin(np.angle(t_e * np.conj(t_o)) / 2) ** 2)
        extra = dict(
            t_even=[t_e.real, t_e.imag],
            t_odd=[t_o.real, t_o.imag],
            phase_error_even=float(abs(np.angle(t_e * np.exp(-1j * beta_e * length)))),
            phase_error_odd=float(abs(np.angle(t_o * np.exp(-1j * beta_o * length)))),
            mode_leakage=float(leakage),
        )
        outgoing = [reflection, back, 1.0 - cross_3d, cross_3d]
    else:
        setup.ports = [hpfem.WaveguidePort(PORT_TAGS[0], 1, [1.0])] + [
            hpfem.WaveguidePort(tag, 1) for tag in PORT_TAGS[1:]
        ]
        problem = hpfem.Scattering3D(dofs, setup)
        seconds["setup"] = time.perf_counter() - t1
        t2 = time.perf_counter()
        solution = problem.solve()
        seconds["solve"] = time.perf_counter() - t2
        amplitudes = problem.port_coefficients(solution)
        powers = [problem.port_modes(i).modes[0].power for i in range(4)]
        port_beta = [problem.port_modes(i).modes[0].beta.real for i in range(4)]
        p_in = abs(amplitudes[0].incoming[0]) ** 2 * powers[0]
        outgoing = [
            abs(a.outgoing[0]) ** 2 * p / p_in for a, p in zip(amplitudes, powers, strict=True)
        ]
    seconds["total"] = time.perf_counter() - t0
    result = Result(
        length=length,
        gap=gap,
        order=order,
        cell=cell,
        backend=hpfem.backend_name(setup.solver) if solver is not None else "auto",
        ports=ports,
        dofs=dofs.num_dofs,
        beta_even=beta_e,
        beta_odd=beta_o,
        kappa=kappa,
        coupling_length=float(np.pi / (2 * kappa)),
        port_beta=port_beta,
        bar_cmt=bar_cmt,
        cross_cmt=cross_cmt,
        bar_fem=outgoing[2],
        cross_fem=outgoing[3],
        reflection=outgoing[0],
        back_coupling=outgoing[1],
        power_sum=float(sum(outgoing)),
        seconds=seconds,
        **extra,
    )
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--length", type=float, default=2.0, help="coupling length [um]")
    parser.add_argument("--gap", type=float, default=150.0, help="gap between the cores [nm]")
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--cell", type=float, default=75.0, help="target cell size [nm]")
    parser.add_argument("--backend", default=None, help="auto, mumps, cudss or sparselu")
    parser.add_argument("--ports", default="half", help="half (single-core launch) or full")
    parser.add_argument("--out", default="directional_coupler_3d.json")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    r = run(
        length=args.length * UM,
        gap=args.gap * NM,
        order=args.order,
        cell=args.cell * NM,
        backend=args.backend,
        ports=args.ports,
        quick=args.quick,
        out=args.out,
    )
    k0 = 2 * np.pi / WAVELENGTH
    print(
        f"L = {r.length / UM:.2f} um, gap {r.gap / NM:.0f} nm, p = {r.order}, {r.dofs} DoFs, "
        f"{r.backend}, ports {r.ports}, {r.seconds['total']:.1f} s"
    )
    print(
        f"supermodes: n_eff even {r.beta_even / k0:.5f}, odd {r.beta_odd / k0:.5f}, "
        f"kappa {r.kappa * UM:.4f} /um, L_c = {r.coupling_length / UM:.2f} um"
    )
    print(f"bar   : FEM {r.bar_fem:.5f}  CMT {r.bar_cmt:.5f}")
    print(f"cross : FEM {r.cross_fem:.5f}  CMT {r.cross_cmt:.5f}")
    print(
        f"reflection {r.reflection:.2e}, back coupling {r.back_coupling:.2e}, "
        f"power sum {r.power_sum:.5f}"
    )
    if r.ports == "full":
        print(
            f"supermode ports: |t_e| {abs(complex(*r.t_even)):.5f}, "
            f"|t_o| {abs(complex(*r.t_odd)):.5f}, phase errors {r.phase_error_even:.2e} / "
            f"{r.phase_error_odd:.2e} rad, leakage {r.mode_leakage:.2e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

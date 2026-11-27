"""3D add-drop ring resonator between modal ports (M12 stage B): the through and drop spectra of
a silicon ring between two bus waveguides, the resonance (complex pole) fitted from them, and
the resonance wavelength of the isolated ring from the axisymmetric cavity solver as the
reference.

Run `python examples/ring_resonator_3d/run.py [--mesh ring.msh] [--order-core p]
[--order-far p] [--wavelengths nm ...] [--backend auto|mumps|cudss|sparselu]
[--record file.json] [--reference-only] [--quick] [--half] [--quarter]`. Without `--mesh`
the Gmsh mesh is built by `ring_mesh.py` (needs the `gmsh` package); `--record` writes the
JSON after every wavelength and resumes from it. Geometry and physics: README.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field

import numpy as np

import hpfem
from hpfem import units

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ring_mesh import (  # noqa: E402
    TAG_CLAD,
    TAG_LEFT,
    TAG_MIRROR_X,
    TAG_MIRROR_Y,
    TAG_RIGHT,
    TAG_SI,
    TAG_SYMMETRY,
    TAG_WALL,
    Geometry,
)
from ring_mesh import build as build_mesh  # noqa: E402

NM = units.nm
N_SI = 3.48
N_SIO2 = 1.44
# ports: 1 bus A (y < 0) at x = -X (input), 2 bus A at +X (through), 3 bus B (y > 0) at -X
# (drop: the ring turns the wave around), 4 bus B at +X (add)
PORT_TAGS = (TAG_LEFT, TAG_RIGHT, 13, 14)
BACKENDS = {
    "auto": None,
    "mumps": hpfem.DirectSolverBackend.MUMPS,
    "cudss": hpfem.DirectSolverBackend.CUDSS,
    "sparselu": hpfem.DirectSolverBackend.SPARSE_LU,
}


def materials():
    m = hpfem.MaterialMap(hpfem.Material.dielectric(N_SIO2))
    m.set(TAG_CLAD, hpfem.Material.dielectric(N_SIO2))
    m.set(TAG_SI, hpfem.Material.dielectric(N_SI))
    return m


def core_distance(geometry: Geometry, points_nm):
    """Distance [nm] of points from the nearest core (ring or bus), from the exact shapes."""
    g, x = geometry, np.asarray(points_nm)
    d_bus = np.maximum(np.abs(np.abs(x[:, 1]) - g.bus_y) - g.width / 2, 0.0)
    d_ring = np.maximum(np.abs(np.hypot(x[:, 0], x[:, 1]) - g.radius) - g.width / 2, 0.0)
    d_z = np.maximum(np.maximum(-x[:, 2], x[:, 2] - g.height), 0.0)
    return np.hypot(np.minimum(d_bus, d_ring), d_z)


def load_mesh(geometry: Geometry, path: str):
    """Reads the Gmsh mesh and splits the two end faces at y = 0 into the four ports (the
    quarter model has only the face of port 1)."""
    return split_ports(hpfem.read_gmsh(path, NM, 3))


def split_ports(mesh):
    """Tags the facets of the end faces x = -/+X by the bus they belong to (y < 0: bus A)."""
    for face, (below, above) in ((TAG_LEFT, PORT_TAGS[::2]), (TAG_RIGHT, PORT_TAGS[1::2])):
        for f in mesh.facets_with_tag(face):
            y = np.mean([mesh.vertex(int(v))[1] for v in mesh.facet_vertices(int(f))])
            mesh.set_facet_tag(int(f), below if y < 0 else above)
    return mesh


def mirror_quarter(quarter):
    """The half model (four ports) as the images of the quarter mesh under the mirrors about
    x = 0 and y = 0: a mesh with exactly the symmetry the quarter model assumes, for checking
    the parity decomposition against the direct solve on the same discretisation. Curved edge
    nodes are mirrored with the cells; the planes x = 0 and y = 0 become interior."""
    vertices = np.asarray(quarter.vertices)
    cells = np.asarray(quarter.cells)
    tags = np.asarray(quarter.cell_tags)
    images = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
    scale = np.max(np.abs(vertices))
    index, points, maps = {}, [], []
    for sx, sy in images:  # merge the copies of the vertices on the planes
        image = vertices * np.array([sx, sy, 1.0])
        local = np.empty(len(image), dtype=np.int64)
        for i, x in enumerate(image):
            key = tuple(np.round(x / scale, 10))
            if key not in index:
                index[key] = len(points)
                points.append(x)
            local[i] = index[key]
        maps.append(local)
    full = hpfem.Mesh3D(
        np.array(points), np.concatenate([m[cells] for m in maps]), list(np.tile(tags, 4))
    )
    facets, facet_tags = [], []
    for f in quarter.boundary_facets:
        tag = quarter.facet_tag(int(f))
        if tag in (TAG_MIRROR_X, TAG_MIRROR_Y) or tag < 0:
            continue
        fv = list(quarter.facet_vertices(int(f)))
        for (sx, _), m in zip(images, maps, strict=True):
            facets.append(sorted(int(m[v]) for v in fv))
            facet_tags.append(TAG_RIGHT if tag == TAG_LEFT and sx < 0 else tag)
    full.set_facet_tags(facets, facet_tags)
    if quarter.geometry_order == 2:
        nodes = np.asarray(quarter.edge_nodes)
        out = np.zeros((full.num_edges, 3))
        pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        for (sx, sy), m in zip(images, maps, strict=True):
            for c in cells:
                for i, j in pairs:
                    node = nodes[quarter.edge_id(int(c[i]), int(c[j]))]
                    out[full.edge_id(int(m[c[i]]), int(m[c[j]]))] = node * np.array([sx, sy, 1.0])
        full.set_edge_nodes(out)
    return full


def cell_orders(geometry: Geometry, mesh, order_core: int, order_far: int, reach: float):
    """order_core in the cells whose centroid lies within `reach` [nm] of a core, else
    order_far (the guided fields decay over about 150 nm in the silica)."""
    d = core_distance(geometry, np.asarray(mesh.cell_centroids) / NM)
    return np.where(d < reach, order_core, order_far).astype(int)


# --- reference: the isolated ring as an axisymmetric cavity ----------------------------------


def _graded(edges, cell):
    lines = [edges[0]]
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        n = max(1, round((b - a) / cell))
        lines.extend(a + (b - a) * (i + 1) / n for i in range(n))
    return np.array(lines)


def ring_reference(geometry: Geometry, orders, *, p: int = 4, cell: float = 50.0):
    """Resonance wavelengths [nm] of the fundamental TE-like mode of the isolated ring for the
    azimuthal orders `orders`: the meridian (r, z) section in µm units from the axis to the
    PEC wall at the bus distance, the same cladding margins in z; p and the cell [nm] are
    converged to about 0.01 nm (README). The lowest mode of each order is the TE-like one."""
    g = geometry
    um = 1e-3  # nm -> µm: the gauged eigensolver works on k0^2 in mesh units
    r0, r1 = g.radius - g.width / 2, g.radius + g.width / 2
    r_wall = g.bus_y + g.width / 2 + g.margin_y
    xs = np.concatenate(
        [_graded([0.0, r0 - 1000.0], 4 * cell)[:-1], _graded([r0 - 1000.0, r0, r1, r_wall], cell)]
    )
    ys = _graded([-g.margin_z, 0.0, g.height, g.height + g.margin_z], cell)
    mesh = hpfem.rectangle(
        len(xs) - 1, len(ys) - 1, [xs[0] * um, ys[0] * um], [xs[-1] * um, ys[-1] * um]
    )
    dx, dy = (xs[-1] - xs[0]) / (len(xs) - 1), (ys[-1] - ys[0]) / (len(ys) - 1)
    for v in range(mesh.num_vertices):  # the uniform index grid onto the graded lines
        x = np.asarray(mesh.vertex(v)) / um
        i, j = round((x[0] - xs[0]) / dx), round((x[1] - ys[0]) / dy)
        mesh.set_vertex(v, [xs[i] * um, ys[j] * um])
    for c, x in enumerate(np.asarray(mesh.cell_centroids) / um):
        mesh.set_cell_tag(c, TAG_SI if r0 < x[0] < r1 and 0 < x[1] < g.height else TAG_CLAD)
    nd, h1 = hpfem.NedelecDofMap2D(mesh, p), hpfem.DofMap2D(mesh, p)
    setup = hpfem.AxisymmetricCavitySetup()
    setup.materials = materials()
    setup.axis_tag = hpfem.box_tag.X_MIN
    setup.pec_tags = [hpfem.box_tag.X_MAX, hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX]
    setup.num_modes = 1
    setup.krylov_dimension = 30
    out = {}
    for m in orders:
        setup.azimuthal_order = int(m)
        mode = hpfem.AxisymmetricCavity(nd, h1, setup).solve()[0]
        out[int(m)] = 2 * np.pi / mode.wavenumber / um
    return out


# --- the resonance from the spectra ----------------------------------------------------------


def fit_pole(wavelengths, responses):
    """Common complex pole of several responses near one resonance,
    S_k(λ) = A_k / (λ − λ_p) + B_k + C_k (λ − λ_c): multiplied by (λ − λ_p) the model is linear
    in (A_k − B_k q, B_k − C_k q, C_k, q) with q = λ_p − λ_c, solved by least squares (exact for
    data of the model). Returns λ_p [unit of λ] and the relative residual; needs at least
    four wavelengths."""
    lam = np.asarray(wavelengths, dtype=float)
    lc = lam.mean()
    t = lam - lc
    S = np.asarray(responses, dtype=complex).reshape(-1, len(lam))
    k, n = S.shape
    rows = np.zeros((k * n, 3 * k + 1), dtype=complex)
    rhs = np.zeros(k * n, dtype=complex)
    for j in range(k):  # S t = a + b t + c t^2 + q S
        rows[j * n : (j + 1) * n, 3 * j : 3 * j + 3] = np.stack([np.ones(n), t, t**2], axis=1)
        rows[j * n : (j + 1) * n, -1] = S[j]
        rhs[j * n : (j + 1) * n] = S[j] * t
    x, *_ = np.linalg.lstsq(rows, rhs, rcond=None)
    residual = float(np.linalg.norm(rows @ x - rhs) / np.linalg.norm(rhs))
    return complex(x[-1] + lc), residual


# --- the 3D solve ----------------------------------------------------------------------------


@dataclass
class Record:
    geometry: dict
    order_core: int
    order_far: int
    reach: float
    cells: int
    dofs: int
    backend: str
    reference: dict = field(default_factory=dict)  # azimuthal order -> λ [nm] of the isolated ring
    points: list = field(default_factory=list)  # one dict per wavelength
    fit: dict | None = None


PARITIES = ((1, 1), (1, -1), (-1, 1), (-1, -1))  # (x, y): +1 magnetic wall, -1 PEC


def solve_wavelength(
    dofs, wavelength_nm: float, solver=None, half: bool = False, quarter: bool = False
):
    """S-parameters of the excitation at port 1: the complex outgoing amplitudes of the four
    ports, power-normalised (|s_k|^2 = power fraction), with the port n_eff. ``half``: the
    mesh is the half model above the mid-plane of the cores, a magnetic wall (natural in the
    volume, `pmc_tags` on the ports). ``quarter``: the mesh is the quarter x <= 0, y <= 0
    (`solve_quarter`)."""
    if quarter:
        return solve_quarter(dofs, wavelength_nm, solver, half)
    setup = hpfem.ScatteringSetup3D()
    setup.omega = units.angular_frequency(wavelength=wavelength_nm * NM)
    setup.materials = materials()
    setup.pec_tags = [TAG_WALL]
    if solver is not None:
        setup.solver = solver
    pmc = [TAG_SYMMETRY] if half else []
    setup.ports = [hpfem.WaveguidePort(PORT_TAGS[0], 1, [1.0], pmc)] + [
        hpfem.WaveguidePort(tag, 1, [], pmc) for tag in PORT_TAGS[1:]
    ]
    t0 = time.perf_counter()
    problem = hpfem.Scattering3D(dofs, setup)
    solution = problem.solve()
    amplitudes = problem.port_coefficients(solution)
    modes = [problem.port_modes(i).modes[0] for i in range(4)]
    a_in = amplitudes[0].incoming[0] * np.sqrt(modes[0].power)
    s = [a.outgoing[0] * np.sqrt(m.power) / a_in for a, m in zip(amplitudes, modes, strict=True)]
    return {
        "wavelength": wavelength_nm,
        "s": [[v.real, v.imag] for v in s],
        "power": [abs(v) ** 2 for v in s],
        "power_sum": float(sum(abs(v) ** 2 for v in s)),
        "n_eff": modes[0].beta.real / setup.omega * hpfem.constants.c0,
        "seconds": time.perf_counter() - t0,
    }


def solve_quarter(dofs, wavelength_nm: float, solver=None, half: bool = False):
    """The same on the quarter model by the mirror symmetries about x = 0 and y = 0 (ports
    1 <-> 2 and 3 <-> 4 under the first, 1 <-> 3 and 2 <-> 4 under the second): the launch at
    port 1 is the sum of the four parity excitations (sx, sy) / 4 with the incoming amplitudes
    (1, sx, sy, sx sy) at the ports (1, 2, 3, 4) in the mirror images of the port-1 mode. Each
    parity is a one-port problem on the quarter with a magnetic wall (natural; +1) or a PEC
    wall (-1) on each plane and gives the reflection r(sx, sy) at port 1; then
    s_1 = sum r / 4, s_2 = sum sx r / 4, s_3 = sum sy r / 4, s_4 = sum sx sy r / 4. The
    amplitudes of ports 2-4 refer to the mirrored port-1 mode, a fixed sign against the
    full model's own mode; powers and the fitted pole do not depend on it."""
    t0 = time.perf_counter()
    reflections = []
    beta = 0.0
    omega = units.angular_frequency(wavelength=wavelength_nm * NM)
    for sx, sy in PARITIES:
        setup = hpfem.ScatteringSetup3D()
        setup.omega = omega
        setup.materials = materials()
        setup.pec_tags = (
            [TAG_WALL] + ([TAG_MIRROR_X] if sx < 0 else []) + ([TAG_MIRROR_Y] if sy < 0 else [])
        )
        if solver is not None:
            setup.solver = solver
        # the port rim on y = 0 stays PEC for both parities: the full model splits each end
        # face at y = 0 into two ports whose modes see a PEC rim there, and the parity
        # problems must use the same modes (the bus mode hardly reaches y = 0 anyway)
        pmc = [TAG_SYMMETRY] if half else []
        setup.ports = [hpfem.WaveguidePort(PORT_TAGS[0], 1, [1.0], pmc)]
        problem = hpfem.Scattering3D(dofs, setup)
        amplitude = problem.port_coefficients(problem.solve())[0]
        reflections.append(amplitude.outgoing[0] / amplitude.incoming[0])
        beta = problem.port_modes(0).modes[0].beta.real
    r = np.array(reflections)
    signs = np.array(PARITIES, dtype=float)
    s = [
        r.mean(),
        (signs[:, 0] * r).mean(),
        (signs[:, 1] * r).mean(),
        (signs[:, 0] * signs[:, 1] * r).mean(),
    ]
    return {
        "wavelength": wavelength_nm,
        "s": [[v.real, v.imag] for v in s],
        "power": [abs(v) ** 2 for v in s],
        "power_sum": float(sum(abs(v) ** 2 for v in s)),
        "n_eff": beta / omega * hpfem.constants.c0,
        "parities": [[v.real, v.imag] for v in reflections],
        "seconds": time.perf_counter() - t0,
    }


def update_fit(record: Record, window: float | None = None):
    """Fits the pole to the through and drop amplitudes; with `window`, to the points within
    `window` FWHM of the previous fit (the linear background holds near the resonance).
    ``inside`` is False when the fitted resonance lies outside the sampled wavelengths: the pole
    is then an extrapolation, not a resolved resonance (an under-resolved spectrum)."""
    pts = sorted(record.points, key=lambda p: p["wavelength"])
    if window is not None and record.fit:
        c, w = record.fit["wavelength"], window * record.fit["fwhm"]
        near = [p for p in pts if abs(p["wavelength"] - c) <= w]
        pts = near if len(near) >= 6 else pts
    if len(pts) < 4:
        record.fit = None
        return None
    lam = [p["wavelength"] for p in pts]
    through = [complex(*p["s"][1]) for p in pts]
    drop = [complex(*p["s"][2]) for p in pts]
    pole, residual = fit_pole(lam, [through, drop])
    record.fit = {
        "wavelength": pole.real,
        "fwhm": 2 * abs(pole.imag),
        "q": pole.real / (2 * abs(pole.imag)),
        "residual": residual,
        "points": len(pts),
        "inside": bool(lam[0] <= pole.real <= lam[-1]),
        "max_drop": max(p["power"][2] for p in pts),
        "min_through": min(p["power"][1] for p in pts),
    }
    return record.fit


def run(
    *,
    geometry: Geometry | None = None,
    mesh_path: str | None = None,
    order_core: int = 3,
    order_far: int = 2,
    reach: float = 250.0,
    wavelengths=None,
    azimuthal_orders=(19, 20),
    backend: str = "auto",
    record_path: str | None = None,
    reference_only: bool = False,
    search: float = 4.0,
    quick: bool = False,
) -> Record:
    """Builds or reads the mesh, computes the isolated-ring reference and the spectra at the
    given wavelengths [nm] (default: nine points within ±`search` nm of the reference
    resonance of the first azimuthal order, then nine within ±2 FWHM of the fitted pole).
    ``quick``: the CI case (R = 1.5 µm, 150 nm cells, p = 1, the reference at p = 2 on
    100 nm cells, two wavelengths)."""
    if quick:
        half = bool(geometry and geometry.half)
        quarter = bool(geometry and geometry.quarter)
        geometry = Geometry(
            radius=1500.0, cell_core=150.0, cell_far=400.0, half=half, quarter=quarter
        )
        order_core, order_far, azimuthal_orders = 1, 1, (14,)
    geometry = geometry or Geometry()
    solver = BACKENDS[str(backend).lower()]
    record = None
    if record_path and os.path.exists(record_path):
        with open(record_path, encoding="utf-8") as f:
            data = json.load(f)
        record = Record(**data)
        if (record.geometry, record.order_core, record.order_far) != (
            asdict(geometry),
            order_core,
            order_far,
        ):
            raise ValueError(f"{record_path} holds another configuration; use a new record")

    if record is None or not record.reference:
        ref = (
            ring_reference(geometry, azimuthal_orders, p=2, cell=100.0)
            if quick
            else ring_reference(geometry, azimuthal_orders)
        )
        reference = {str(m): lam for m, lam in ref.items()}
    else:
        reference = record.reference
    if reference_only:
        return Record(asdict(geometry), order_core, order_far, reach, 0, 0, backend, reference)

    if mesh_path is None:
        fd, mesh_path = tempfile.mkstemp(suffix=".msh")
        os.close(fd)
        build_mesh(geometry, mesh_path)
    mesh = load_mesh(geometry, mesh_path)
    orders = cell_orders(geometry, mesh, order_core, order_far, reach)
    dofs = hpfem.NedelecDofMap3D(mesh, list(orders))
    if record is None:
        record = Record(
            asdict(geometry), order_core, order_far, reach, mesh.num_cells, dofs.num_dofs, backend
        )
    record.reference = reference
    done = {round(p["wavelength"], 6) for p in record.points}

    def solve_all(lams):
        for lam in lams:
            if round(float(lam), 6) in done:
                continue
            point = solve_wavelength(dofs, float(lam), solver, geometry.half, geometry.quarter)
            record.points.append(point)
            done.add(round(float(lam), 6))
            update_fit(record)
            print(
                f"lambda {point['wavelength']:.4f} nm: through {point['power'][1]:.5f}, drop "
                f"{point['power'][2]:.5f}, reflection {point['power'][0]:.1e}, add "
                f"{point['power'][3]:.1e}, sum {point['power_sum']:.6f} "
                f"({point['seconds']:.0f} s)",
                flush=True,
            )
            if record_path:
                with open(record_path, "w", encoding="utf-8") as f:
                    json.dump(asdict(record), f, indent=2)

    if wavelengths is not None:
        solve_all(wavelengths)
    elif quick:
        solve_all(reference[str(azimuthal_orders[0])] + np.array([-0.02, 0.02]))
    else:
        # search: nine points within ±search of the isolated ring, then nine within ±2 FWHM
        # of the fitted pole (the 3D resonance sits off the reference by the coupling and
        # the discretisation)
        centre = reference[str(azimuthal_orders[0])]
        solve_all(centre + np.linspace(-search, search, 9))
        fit = update_fit(record)
        if fit is None or abs(fit["wavelength"] - centre) > search or fit["fwhm"] > search:
            print("no resonance found within the search window; widen --search", flush=True)
        else:
            solve_all(fit["wavelength"] + fit["fwhm"] * np.linspace(-2.0, 2.0, 9))
            update_fit(record, window=3.0)
            if record_path:
                with open(record_path, "w", encoding="utf-8") as f:
                    json.dump(asdict(record), f, indent=2)
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mesh", default=None, help="Gmsh file from ring_mesh.py")
    parser.add_argument("--radius", type=float, default=2.0, help="[um]")
    parser.add_argument("--gap", type=float, default=200.0, help="[nm]")
    parser.add_argument("--cell-core", type=float, default=100.0, help="[nm]")
    parser.add_argument("--cell-far", type=float, default=300.0, help="[nm]")
    parser.add_argument("--half", action="store_true", help="half model, magnetic mid-plane")
    parser.add_argument(
        "--quarter", action="store_true", help="quarter model x, y <= 0, four mirror parities"
    )
    parser.add_argument("--order-core", type=int, default=3)
    parser.add_argument("--order-far", type=int, default=2)
    parser.add_argument("--reach", type=float, default=250.0, help="[nm] of order-core cells")
    parser.add_argument("--wavelengths", type=float, nargs="+", default=None, help="[nm]")
    parser.add_argument("--azimuthal-orders", type=int, nargs="+", default=[19, 20])
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--record", default=None, help="JSON written per wavelength, resumed")
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--search", type=float, default=4.0, help="[nm] around the reference")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    geometry = Geometry(
        radius=1000 * args.radius,
        gap=args.gap,
        cell_core=args.cell_core,
        cell_far=args.cell_far,
        half=args.half,
        quarter=args.quarter,
    )
    r = run(
        geometry=geometry,
        mesh_path=args.mesh,
        order_core=args.order_core,
        order_far=args.order_far,
        reach=args.reach,
        wavelengths=args.wavelengths,
        azimuthal_orders=tuple(args.azimuthal_orders),
        backend=args.backend,
        record_path=args.record,
        reference_only=args.reference_only,
        search=args.search,
        quick=args.quick,
    )
    for m, lam in r.reference.items():
        print(f"isolated ring, m = {m}: lambda = {lam:.4f} nm")
    if r.dofs:
        print(f"{r.cells} cells, {r.dofs} DoFs, p = {r.order_core} / {r.order_far}")
    if r.fit:
        f = r.fit
        print(
            f"resonance {f['wavelength']:.4f} nm, FWHM {f['fwhm']:.4f} nm, Q {f['q']:.0f}, "
            f"max drop {f['max_drop']:.4f}, min through {f['min_through']:.4f}, "
            f"fit residual {f['residual']:.1e}"
        )
        if not f.get("inside", True):
            print(
                "the fitted resonance lies outside the sampled wavelengths: an extrapolation, "
                "the resonance is not resolved (raise the order or widen --search)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

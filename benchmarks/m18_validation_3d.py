"""M18 3D cross-checks of the axisymmetric solver on layer stacks (long local runs, not in CI).

The body-of-revolution results of M18 (ADR-0014) against the full 3D solver `Scattering3D` with
the same `LayerStack3D` background (docs/axisymmetric-layered-features.md, tests 5 and 7):

- ``sphere_on_stack``: a gold sphere (radius 40 nm, 5 nm air gap) on air / SiO2 20 nm / Si3N4
  60 nm / glass, p-polarised at 45 degrees: absorption of the sphere, scattering up / down /
  lateral through a box around it, and |E| of the total field at three points of the plane of
  incidence (in the gap, beside the sphere, above it); 3D on the half domain y >= 0 (the plane of
  incidence is a mirror plane: PMC on y = 0, powers doubled).
- ``nanohole``: a hole of radius 100 nm through a 100 nm gold film on glass at normal incidence:
  T / T_geom (the power the hole adds through the disc r <= 300 nm at z = -200 nm, over the
  power on the hole area) and the absorption change of the film in the cylinder r <= 250 nm;
  3D on the quarter domain x, y >= 0 (x-polarised: PEC on x = 0, PMC on y = 0, powers times 4).

The 2.5D reference of every wavelength is solved first (seconds); the 3D run follows for each
polynomial order. Records go to ``benchmarks/results/<date>-validation-m18-3d.json`` after every
run, keyed ``<case>/<solver>/p<p>/<wavelength>``; a rerun skips the keys already there, so an
interrupted sweep resumes (``--force`` recomputes).

Steps:

    python benchmarks/m18_validation_3d.py mesh     # gmsh meshes (needs a Python with gmsh)
    python benchmarks/m18_validation_3d.py estimate # DoFs, memory and runtime per run
    python benchmarks/m18_validation_3d.py run [--case nanohole] [--orders 2 3]

The meshes are written by benchmarks/m18_meshes_3d.py, which runs in a Python with gmsh
(``--gmsh-python``, default the environment variable HPFEM_GMSH_PYTHON, else ``python``); they
are 3–7 MB and stay outside the repository (``--mesh-dir``). The 3D factorisations need MUMPS or
cuDSS (SparseLU fills in far too much in 3D): build the Python module with
``-DHPFEM_ENABLE_MUMPS=ON`` (e.g. ``cmake --preset mumps -DHPFEM_BUILD_PYTHON=ON``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import hpfem
from hpfem import materials, units

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
MESH_SCRIPT = ROOT / "benchmarks" / "m18_meshes_3d.py"
NM = units.nm
C0, Z0 = hpfem.constants.c0, hpfem.constants.Z0
INTENSITY = 1.0 / (2 * Z0)  # |E0| = 1 in air
GOLD = materials.get("Au")  # Johnson & Christy 1972
WAVELENGTHS = {"sphere_on_stack": [550.0, 650.0], "nanohole": [600.0, 750.0, 900.0]}
NEAR_FIELD = [[2.0, 2.5], [50.0, 45.0], [10.0, 95.0]]  # (x, z) [nm] at y = 0
TAG_INSIDE = 99
# MUMPS on this machine: 23.7 M factor entries in 2.8 s (benchmarks/results/2026-10-08-fill-in,
# 3D, p = 2); the factorisation work grows like entries^1.5 for nested dissection
CALIBRATION = (23.7e6, 2.8)


def default_mesh_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return Path(base) / "hpfem-m18-meshes"


def record_file() -> Path:
    """The existing record file of this study (so a later run resumes it), else a new one."""
    existing = sorted(RESULTS.glob("*-validation-m18-3d.json"))
    return (
        existing[-1]
        if existing
        else RESULTS / f"{time.strftime('%Y-%m-%d')}-validation-m18-3d.json"
    )


def load_example():
    path = ROOT / "examples" / "particle_on_substrate" / "run.py"
    spec = importlib.util.spec_from_file_location("particle_on_substrate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_case(mesh_dir: Path, case: str):
    geometry = json.loads((mesh_dir / f"{case}.json").read_text(encoding="utf-8"))
    mesh = hpfem.read_gmsh(str(mesh_dir / f"{case}.msh"), NM, 3)
    return mesh, geometry


# --- the record store ----------------------------------------------------------------------------


class Records:
    """JSON record file, written after every run (atomically), keyed by run."""

    def __init__(self, path: Path):
        self.path = path
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.data = {
                "study": "M18 3D cross-checks (ADR-0014; docs/axisymmetric-layered-"
                "features.md tests 5 and 7)",
                "runs": {},
                "estimates": {},
            }
        self.data["info"] = {"hpfem": getattr(hpfem, "__version__", "?"),
                             "python": platform.python_version(), "platform": platform.platform(),
                             "backends": [str(b) for b in hpfem.available_backends()]}  # fmt: skip

    def has(self, key: str) -> bool:
        return key in self.data["runs"]

    def put(self, key: str, record: dict, section: str = "runs") -> None:
        self.data[section][key] = record
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        tmp.replace(self.path)


# --- helpers -------------------------------------------------------------------------------------


def centroids(mesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices)
    return vertices[np.asarray(mesh.cells)].mean(axis=1)


def surface_of(mesh, inside, drop=None):
    """Closed surface around the cells whose centroid satisfies inside(x) (on a copy of the
    mesh), without the facets whose vertices all satisfy drop(x) (a symmetry plane)."""
    marked = mesh.copy()
    c = centroids(mesh)
    for cell in range(mesh.num_cells):
        marked.set_cell_tag(cell, TAG_INSIDE if inside(c[cell]) else 1)
    surface = hpfem.Surface3D.around_cells(marked, TAG_INSIDE)
    if drop is None:
        return surface
    vertices = np.asarray(mesh.vertices)
    kept = [
        f
        for f in surface.facets
        if not all(drop(vertices[int(v)]) for v in mesh.facet_vertices(int(f.facet)))
    ]
    out = hpfem.Surface3D()
    out.facets = kept
    return out


def split_by_height(mesh, surface, top, bottom):
    """The facets of a surface above `top`, below `bottom` and between (by the facet centroid)."""
    vertices = np.asarray(mesh.vertices)
    parts = {"up": [], "down": [], "lateral": []}
    for f in surface.facets:
        z = vertices[[int(v) for v in mesh.facet_vertices(int(f.facet))], 2].mean()
        parts["up" if z > top else "down" if z < bottom else "lateral"].append(f)
    out = {}
    for name, facets in parts.items():
        out[name] = hpfem.Surface3D()
        out[name].facets = facets
    return out


def factor_seconds(estimate) -> float:
    entries, seconds = CALIBRATION
    return seconds * (estimate.factor_entries / entries) ** 1.5


# --- sphere on a two-layer stack -----------------------------------------------------------------


def sphere_stack(g, omega):
    layers = [hpfem.Layer(hpfem.Material.dielectric(n), t * NM) for n, t in g["layers"]]
    return hpfem.LayerStack3D(hpfem.Material.vacuum(), layers,
                              hpfem.Material.dielectric(g["substrate_index"]), 0.0)  # fmt: skip


def sphere_materials(g, omega):
    tags = g["tags"]
    (n1, _), (n2, _) = g["layers"]
    return {
        tags["glass"]: hpfem.Material.dielectric(g["substrate_index"]),
        tags["layer1"]: hpfem.Material.dielectric(n1),
        tags["layer2"]: hpfem.Material.dielectric(n2),
        tags["particle"]: GOLD.at(omega),
    }


def sphere_3d(mesh, g, p, wavelength, backend):
    t0 = time.perf_counter()
    omega = 2 * math.pi * C0 / wavelength
    k0 = omega / C0
    theta = math.radians(g["theta_deg"])
    stack = sphere_stack(g, omega)
    wave = stack.plane_wave(k0, theta, hpfem.Polarisation.P, 1.0, 0.0)
    setup = hpfem.ScatteringSetup3D()
    setup.omega = omega
    for tag, material in sphere_materials(g, omega).items():
        setup.materials.set(tag, material)
    setup.incident = wave.field
    setup.background = stack
    setup.formulation = hpfem.Formulation.SCATTERED_FIELD
    x_in, pml = g["x_in"] * NM, g["pml"] * NM
    z_lo, z_hi = (z * NM for z in g["z_in"])
    setup.pml = hpfem.PmlBox3D([-x_in, 0.0, z_lo], [x_in, x_in, z_hi],
                               [pml, pml, 0.0, pml, pml, pml], k0)  # fmt: skip
    setup.solver = backend
    dofs = hpfem.NedelecDofMap3D(mesh, p)
    problem = hpfem.Scattering3D(dofs, setup)
    t_setup = time.perf_counter()
    solution = problem.solve()
    t_solve = time.perf_counter()
    # half domain: every power twice
    absorbed = 2 * hpfem.absorbed_power_by_tag(problem, solution).of_tag(g["tags"]["particle"])
    box = g["box"]
    b_lo, b_hi = (z * NM for z in g["box_z"])
    half = box["half_width"] * NM
    eps = 1e-6 * half

    def inside(x):  # the measurement cylinder of the mesh (the 2.5D surface of revolution)
        return math.hypot(x[0], x[1]) < half and b_lo < x[2] < b_hi

    surface = surface_of(mesh, inside, drop=lambda x: abs(x[1]) < eps)
    t1, t2 = (t * NM for _n, t in g["layers"])
    parts = split_by_height(mesh, surface, 0.0, -(t1 + t2))
    channels = {name: 2 * hpfem.poynting_flux(dofs, solution.unknown, omega, setup.materials, s)
                for name, s in parts.items()}  # fmt: skip
    locator = hpfem.PointLocator3D(mesh)
    near = []
    for x, z in NEAR_FIELD:
        value = problem.total_field(solution, locator, [x * NM, 0.0, z * NM])
        near.append(float(np.linalg.norm(value)) if value is not None else None)
    sigma = {"absorption": absorbed / INTENSITY,
             **{k: v / INTENSITY for k, v in channels.items()}}  # fmt: skip
    sigma["scattering"] = sigma["up"] + sigma["down"] + sigma["lateral"]
    return {"wavelength_nm": wavelength / NM, "p": p, "dofs": dofs.num_dofs,
            **{f"sigma_{k}_nm2": v / NM**2 for k, v in sigma.items()}, "near_field_abs_E": near,
            "seconds": {"setup": t_setup - t0, "solve": t_solve - t_setup,
                        "post": time.perf_counter() - t_solve}}  # fmt: skip


def sphere_reference(example, g, wavelengths, p=4, cells_per_radius=None):
    """2.5D: the example's mesher with the stack of the case, orders m >= 0 (+-m equal)."""
    radius, gap = g["radius"] * NM, g["gap"] * NM
    n = cells_per_radius or round(radius / gap)
    t1, t2 = (t * NM for _n, t in g["layers"])
    interfaces = [0.0, -t1, -(t1 + t2)]
    cfg = {
        "r_in": g["x_in"] * NM,
        "depth": g["depth"] * NM,
        "height": g["height"] * NM,
        "pml": g["pml"] * NM,
        "coarse": 30 * NM,
        "growth": 0.35,
        "layer_cell": 10 * NM,
        # mesh lines on the measurement cylinder: both solvers integrate over the same surface
        "r_breaks": [g["box"]["half_width"] * NM],
        "z_breaks": [z * NM for z in g["box_z"]],
    }
    mesh, geometry = example.particle_mesh(radius, n, interfaces, cfg)
    # the example's tags for three interfaces: SPACER = first layer, FILM = second layer
    tags = {"glass": example.TAG_GLASS, "layer1": example.TAG_SPACER, "layer2": example.TAG_FILM,
            "particle": example.TAG_PARTICLE}  # fmt: skip
    g2 = dict(g, tags=tags)
    box = g["box"]
    b_lo, b_hi = (z * NM for z in g["box_z"])
    half = box["half_width"] * NM

    def inside(r, z):
        return r < half and b_lo < z < b_hi

    rows = example.particle_spectrum(
        mesh, geometry, lambda omega: sphere_stack(g, omega),
        lambda omega: sphere_materials(g2, omega), np.asarray(wavelengths),
        math.radians(g["theta_deg"]), ("p",), p, inside, 0.5,
    )  # fmt: skip
    # |E| at the near-field points: the orders summed at phi = 0 (E_r, E_z of +-m equal, E_phi
    # odd), solved again per order for the field values
    nd = hpfem.NedelecDofMap2D(mesh, p)
    h1 = hpfem.DofMap2D(mesh, p)
    locator = hpfem.PointLocator2D(mesh)
    for row, wavelength in zip(rows, wavelengths, strict=True):
        omega = 2 * math.pi * C0 / wavelength
        k0 = omega / C0
        stack = sphere_stack(g, omega)
        setup = hpfem.AxisymmetricScatteringSetup()
        setup.omega = omega
        for tag, material in sphere_materials(g2, omega).items():
            setup.materials.set(tag, material)
        setup.axis_tag = example.TAG_AXIS
        setup.pml = example.pml_box(geometry, k0)
        setup.background = stack
        totals = np.zeros((len(NEAR_FIELD), 3), dtype=complex)
        for m in range(row["max_order"] + 1):
            wave = hpfem.layered_axisymmetric_wave(stack, k0, math.radians(g["theta_deg"]), "p", m)
            setup.azimuthal_order = m
            setup.incident = wave.value
            field = hpfem.AxisymmetricScattering(nd, h1, setup).solve()
            weight = 1.0 if m == 0 else 2.0
            for i, (x, z) in enumerate(NEAR_FIELD):
                point = [x * NM, z * NM]
                e_rz = hpfem.evaluate_hcurl(nd, field.meridian, locator, point)
                v = hpfem.evaluate_h1(h1, field.azimuthal, locator, point)
                inc = wave.value(np.array(point))
                totals[i] += weight * np.array([e_rz[0] + inc[0], 0.0, e_rz[1] + inc[2]])
                if m == 0:  # E_phi = i v / r of the order 0 (zero for p, kept for safety)
                    totals[i][1] += 1j * (v + inc[1]) / point[0]
        row["near_field_abs_E"] = [float(np.linalg.norm(t)) for t in totals]
        row["p"] = p
    return rows


# --- nanohole ------------------------------------------------------------------------------------


def nanohole_3d(mesh, g, p, wavelength, backend):
    t0 = time.perf_counter()
    omega = 2 * math.pi * C0 / wavelength
    k0 = omega / C0
    film, a = g["film"] * NM, g["hole_radius"] * NM
    gold, glass = GOLD.at(omega), hpfem.Material.dielectric(g["substrate_index"])
    stack = hpfem.LayerStack3D(hpfem.Material.vacuum(), [hpfem.Layer(gold, film)], glass, 0.0)
    wave = stack.plane_wave(k0, 0.0, hpfem.Polarisation.P, 1.0, 0.0)  # E along x
    tags = g["tags"]
    setup = hpfem.ScatteringSetup3D()
    setup.omega = omega
    setup.materials.set(tags["glass"], glass)
    setup.materials.set(tags["film"], gold)
    setup.materials.set(tags["hole"], hpfem.Material.vacuum())
    setup.incident = wave.field
    setup.background = stack
    setup.formulation = hpfem.Formulation.SCATTERED_FIELD
    setup.pec_tags = [tags["pec"]]  # x = 0: the tangential E of the x-polarised fields vanishes
    x_in, pml = g["x_in"] * NM, g["pml"] * NM
    z_lo, z_hi = (z * NM for z in g["z_in"])
    setup.pml = hpfem.PmlBox3D([0.0, 0.0, z_lo], [x_in, x_in, z_hi],
                               [0.0, pml, 0.0, pml, pml, pml], k0)  # fmt: skip
    setup.solver = backend
    dofs = hpfem.NedelecDofMap3D(mesh, p)
    problem = hpfem.Scattering3D(dofs, setup)
    t_setup = time.perf_counter()
    solution = problem.solve()
    t_solve = time.perf_counter()
    # the disc r <= R at z_disc, normal -z (transmitted power positive); quarter domain: times 4
    z_disc, r_disc = g["disc"]["z"] * NM, g["disc"]["radius"] * NM
    plane = hpfem.Surface3D.plane(mesh, 2, z_disc, -1)
    vertices = np.asarray(mesh.vertices)
    disc = hpfem.Surface3D()
    disc.facets = [f for f in plane.facets
                   if all(math.hypot(*vertices[int(v), :2]) <= r_disc * (1 + 1e-6)
                          for v in mesh.facet_vertices(int(f.facet)))]  # fmt: skip
    total = hpfem.combined_field(hpfem.discrete_field(dofs, solution.unknown),
                                 hpfem.analytic_field(wave.field), 1.0)  # fmt: skip
    order = 2 * p + 2
    through = hpfem.poynting_flux(mesh, disc, total, omega, setup.materials, order)
    stack_only = hpfem.poynting_flux(mesh, disc, hpfem.analytic_field(wave.field), omega,
                                     setup.materials, order)  # fmt: skip
    on_hole = INTENSITY * math.pi * a**2
    # absorption change in the cylinder r <= R of the film: total field minus the bare stack
    # (A I pi R^2, the stack field's absorption is uniform across the film)
    r_region = g["region_radius"] * NM
    c = centroids(mesh)
    per_cell = np.asarray(hpfem.absorbed_power_by_tag(problem, solution).per_cell)
    in_region = (np.asarray(mesh.cell_tags) == tags["film"]) & (
        np.hypot(c[:, 0], c[:, 1]) < r_region
    )
    absorbed = 4 * per_cell[in_region].sum()
    bare = wave.absorptance * INTENSITY * math.pi * r_region**2
    return {"wavelength_nm": wavelength / NM, "p": p, "dofs": dofs.num_dofs,
            "T_over_T_geom": 4 * (through - stack_only) / on_hole,
            "disc_stack_over_expected": 4 * stack_only
            / (wave.transmittance * INTENSITY * math.pi * r_disc**2),
            "absorption_change_over_geom": (absorbed - bare) / on_hole,
            "seconds": {"setup": t_setup - t0, "solve": t_solve - t_setup,
                        "post": time.perf_counter() - t_solve}}  # fmt: skip


def nanohole_reference(example, wavelengths):
    rows = example.nanohole(False, wavelengths=np.asarray(wavelengths))["spectrum"]
    for row in rows:
        row["p"] = 3
    return rows


# --- commands ------------------------------------------------------------------------------------


def command_mesh(args) -> int:
    python = args.gmsh_python or os.environ.get("HPFEM_GMSH_PYTHON") or "python"
    command = [python, str(MESH_SCRIPT), "--out", str(args.mesh_dir), "--size", str(args.size)]
    for case in args.case or ():
        command += ["--case", case]
    print(" ".join(command), flush=True)
    return subprocess.call(command)


def command_estimate(args, records: Records) -> int:
    print(f"{'case':16s} {'p':>2s} {'DoFs':>9s} {'factor entries':>15s} {'memory':>9s} "
          f"{'factor time':>12s} {'runs':>4s}")  # fmt: skip
    for case in args.case or ("sphere_on_stack", "nanohole"):
        mesh, _g = load_case(args.mesh_dir, case)
        for p in args.orders:
            e = hpfem.estimate_memory(mesh, p, hpfem.DirectSolverBackend.MUMPS)
            runs = len(args.wavelengths or WAVELENGTHS[case])
            seconds = factor_seconds(e)
            print(f"{case:16s} {p:2d} {e.dofs:9d} {e.factor_entries / 1e6:13.0f} M "
                  f"{e.total_bytes / 1e9:7.1f} GB {seconds / 60:9.1f} min {runs:4d}")  # fmt: skip
            records.put(f"{case}/p{p}", {"cells": mesh.num_cells, "dofs": e.dofs,
                                         "factor_entries": e.factor_entries,
                                         "memory_gb": e.total_bytes / 1e9,
                                         "factor_minutes_estimate": seconds / 60,
                                         "runs": runs, "describe": e.describe()},
                        section="estimates")  # fmt: skip
    return 0


def command_run(args, records: Records) -> int:
    backend = getattr(hpfem.DirectSolverBackend, args.backend)
    if backend == hpfem.DirectSolverBackend.SPARSE_LU or not hpfem.available(backend):
        if not args.allow_sparse_lu:
            print(
                f"backend {args.backend} unavailable or SparseLU; the 3D runs need MUMPS or "
                "cuDSS (build with -DHPFEM_ENABLE_MUMPS=ON), --allow-sparse-lu to insist"
            )
            return 2
    example = load_example()
    for case in args.case or ("sphere_on_stack", "nanohole"):
        wavelengths = [w * NM for w in (args.wavelengths or WAVELENGTHS[case])]
        mesh, g = load_case(args.mesh_dir, case)
        pending = [
            w for w in wavelengths if args.force or not records.has(f"{case}/2.5D/{w / NM:.1f}")
        ]
        if pending:
            print(f"{case}: 2.5D reference at {[round(w / NM) for w in pending]} nm", flush=True)
            rows = (sphere_reference(example, g, pending, args.reference_order)
                    if case == "sphere_on_stack"
                    else nanohole_reference(example, pending))  # fmt: skip
            for w, row in zip(pending, rows, strict=True):
                records.put(f"{case}/2.5D/{w / NM:.1f}", row)
        for p in args.orders:
            for w in wavelengths:
                key = f"{case}/3D/p{p}/{w / NM:.1f}"
                if records.has(key) and not args.force:
                    print(f"{key}: done", flush=True)
                    continue
                print(f"{key}: solving", flush=True)
                run = sphere_3d if case == "sphere_on_stack" else nanohole_3d
                record = run(mesh, g, p, w, backend)
                record["mesh"] = {"cells": mesh.num_cells, "size": g.get("size", 1.0)}
                records.put(key, record)
                print(f"{key}: {json.dumps({k: v for k, v in record.items() if k != 'mesh'})}",
                      flush=True)  # fmt: skip
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("mesh", "estimate", "run"))
    parser.add_argument("--case", choices=("sphere_on_stack", "nanohole"), action="append")
    parser.add_argument("--orders", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--wavelengths", type=float, nargs="+", help="[nm]")
    parser.add_argument("--mesh-dir", type=Path, default=default_mesh_dir())
    parser.add_argument("--size", type=float, default=1.0, help="mesh size factor (mesh)")
    parser.add_argument("--gmsh-python", help="Python with gmsh (mesh)")
    parser.add_argument(
        "--backend", default="MUMPS", choices=("MUMPS", "CUDSS", "AUTO", "SPARSE_LU")
    )
    parser.add_argument("--allow-sparse-lu", action="store_true")
    parser.add_argument("--reference-order", type=int, default=4, help="p of the 2.5D sphere")
    parser.add_argument("--force", action="store_true", help="recompute existing records")
    parser.add_argument("--out", type=Path, help="record file (default: the existing one)")
    args = parser.parse_args(argv)
    hpfem.set_log_level("warn")
    if args.command == "mesh":
        return command_mesh(args)
    records = Records(args.out or record_file())
    if args.command == "estimate":
        return command_estimate(args, records)
    return command_run(args, records)


if __name__ == "__main__":
    sys.exit(main())

"""Gold sphere dimer (M10 benchmark, Hoffmann, Hafner, Leidenberger, Hesselbarth, Burger,
Proc. SPIE 7390, 73900J, 2009): two gold spheres of 80 nm diameter with a 1 nm gap, a plane
wave at 632 nm incident perpendicular to the dimer axis and polarised along it, and the
squared electric field at the centre of the gap, whose MMP reference is
|E|^2 = 5.47624e5 V^2/m^2 for |E_inc| = 1 V/m. The paper does not state the permittivity of
gold it used; this example takes the library's Johnson & Christy data (eps = -11.685 + 1.267i
at 632 nm) as a documented assumption and reports the sensitivity of the result to eps by
finite differences, so the comparison is "agreement within the eps uncertainty", not a
five-digit validation (docs/validation.md, section F).

Method: the dimer is a body of revolution; the incident wave is expanded in azimuthal orders
(`hpfem.oblique_plane_wave`, theta_i = 90 deg, p polarisation = along the axis) and every order
is solved on the meridian mesh with `hpfem.AxisymmetricScattering` (cylindrical PML). On the
axis only m = 0 (E_z) and m = +-1 (E_x, E_y) are non-zero, so three solves give the gap field.
The meridian mesh comes from Gmsh (second-order, graded from 0.1 nm at the gap), needs the
`gmsh` package. Run `python examples/gold_dimer/run.py [--orders 2 3 4] [--size f]
[--sensitivity] [--out file.json] [--quick]`.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field

import numpy as np

import hpfem
from hpfem import materials, units

NM = units.nm
WAVELENGTH = 632 * NM
RADIUS = 40 * NM
GAP = 1 * NM
REFERENCE = 5.47624e5  # V^2/m^2, MMP (MaX-1), Hoffmann et al. 2009
TAG_VACUUM, TAG_GOLD, TAG_AXIS, TAG_WALL = 1, 2, 77, 5
INNER = 300 * NM  # half-width of the inner box (r up to INNER, |z| up to INNER)
PML = 250 * NM


def gold_permittivity():
    """Johnson & Christy 1972 at 632 nm from the material library."""
    omega = units.angular_frequency(wavelength=WAVELENGTH)
    return complex(materials.get("Au").at(omega).eps_r)


def meridian_mesh(size_factor: float = 1.0, path: str | None = None):
    """Gmsh meridian mesh (r, z) in nm: the half plane r >= 0 with the two half discs, graded
    from 0.1 nm at the gap centre; second-order elements on the sphere surfaces. Returns the
    hpfem mesh (coordinates scaled to metres) with the cell tags gold / vacuum and the facet
    tags of the axis and the outer walls."""
    import gmsh

    r_out = (INNER + PML) / NM
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("gold_dimer")
        occ = gmsh.model.occ
        box = occ.addRectangle(0.0, -r_out, 0.0, r_out, 2 * r_out)
        zc = (RADIUS + GAP / 2) / NM
        discs = [occ.addDisk(0.0, zc, 0.0, RADIUS / NM, RADIUS / NM),
                 occ.addDisk(0.0, -zc, 0.0, RADIUS / NM, RADIUS / NM)]  # fmt: skip
        pieces, _ = occ.fragment([(2, box)], [(2, d) for d in discs])
        occ.synchronize()
        # keep the pieces inside the half plane; classify by the centre of mass
        gold, vacuum, remove = [], [], []
        for dim, tag in pieces:
            x, y, _ = occ.getCenterOfMass(dim, tag)
            if x < -1e-9:
                remove.append((dim, tag))
            elif abs(abs(y) - zc) < RADIUS / NM:
                gold.append(tag)
            else:
                vacuum.append(tag)
        if remove:
            occ.remove(remove, recursive=True)
            occ.synchronize()
        gmsh.model.addPhysicalGroup(2, gold, TAG_GOLD, "gold")
        gmsh.model.addPhysicalGroup(2, vacuum, TAG_VACUUM, "vacuum")
        axis, walls = [], []
        tol = 1e-5  # OCC bounding boxes carry a small tolerance padding
        for dim, tag in gmsh.model.getEntities(1):
            x0, y0, _, x1, y1, _ = gmsh.model.getBoundingBox(dim, tag)
            if max(abs(x0), abs(x1)) < tol:
                axis.append(tag)
            elif x0 > r_out - tol or y1 < -r_out + tol or y0 > r_out - tol:
                walls.append(tag)
        if not axis or not walls:
            raise RuntimeError("gold_dimer: axis or wall curves not found in the Gmsh model")
        gmsh.model.addPhysicalGroup(1, axis, TAG_AXIS, "axis")
        gmsh.model.addPhysicalGroup(1, walls, TAG_WALL, "wall")
        # size field: 0.1 nm at the gap, 3 nm on the spheres away from it, 25 nm in the box
        f_gap = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(f_gap, "PointsList", [occ.addPoint(0.0, 0.0, 0.0)])
        occ.synchronize()
        f_thr = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f_thr, "InField", f_gap)
        gmsh.model.mesh.field.setNumber(f_thr, "SizeMin", 0.1 * size_factor)
        gmsh.model.mesh.field.setNumber(f_thr, "SizeMax", 25.0 * size_factor)
        gmsh.model.mesh.field.setNumber(f_thr, "DistMin", 1.5)
        gmsh.model.mesh.field.setNumber(f_thr, "DistMax", 120.0)
        f_sphere = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(f_sphere, "SurfacesList", gold)
        f_thr2 = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f_thr2, "InField", f_sphere)
        gmsh.model.mesh.field.setNumber(f_thr2, "SizeMin", 3.0 * size_factor)
        gmsh.model.mesh.field.setNumber(f_thr2, "SizeMax", 40.0 * size_factor)
        gmsh.model.mesh.field.setNumber(f_thr2, "DistMin", 5.0)
        gmsh.model.mesh.field.setNumber(f_thr2, "DistMax", 200.0)
        f_min = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", [f_thr, f_thr2])
        gmsh.model.mesh.field.setAsBackgroundMesh(f_min)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        if path is None:
            fd, path = tempfile.mkstemp(suffix=".msh")
            os.close(fd)
        gmsh.write(path)
    finally:
        gmsh.finalize()
    return hpfem.read_gmsh(path, NM, 2)


@dataclass
class OrderSolve:
    order: int
    dofs: int
    seconds: float


def gap_field(mesh, p: int, eps_gold: complex, point_r: float = 0.002 * NM):
    """E = (E_x, E_y, E_z) of the total field at (r = point_r, z = 0, phi = 0) from the orders
    m = 0, +1, -1 (the only ones with a non-zero field on the axis), and the solve records."""
    omega = units.angular_frequency(wavelength=WAVELENGTH)
    k0 = units.vacuum_wavenumber(omega)
    nd, h1 = hpfem.NedelecDofMap2D(mesh, p), hpfem.DofMap2D(mesh, p)
    gold = hpfem.Material()
    gold.eps_r = eps_gold
    locator = hpfem.PointLocator2D(mesh)
    x = np.array([point_r, 0.0])
    total = np.zeros(3, dtype=complex)
    solves = []
    for m in (0, 1, -1):
        setup = hpfem.AxisymmetricScatteringSetup()
        setup.omega = omega
        setup.materials.set(TAG_GOLD, gold)
        setup.axis_tag = TAG_AXIS
        setup.pec_tags = [TAG_WALL]
        setup.azimuthal_order = m
        setup.pml = hpfem.PmlBox2D([0.0, -INNER], [INNER, INNER], [0.0, PML, PML, PML], k0)
        # p polarisation at theta_i = 90 deg is -z; amplitude -1 makes E_inc = +z of unit size
        setup.incident = hpfem.oblique_plane_wave(-1.0, k0, np.pi / 2, hpfem.PlanePolarisation.P, m)
        t0 = time.perf_counter()
        problem = hpfem.AxisymmetricScattering(nd, h1, setup)
        sol = problem.solve()
        solves.append(OrderSolve(m, nd.num_dofs + h1.num_dofs, time.perf_counter() - t0))
        e_rz = hpfem.evaluate_hcurl(nd, sol.meridian, locator, x)  # (E_r, E_z) scattered
        v = hpfem.evaluate_h1(h1, sol.azimuthal, locator, x)  # v = -i r E_phi
        inc = setup.incident(x)  # scaled components (E_r, v, E_z) of the incident order
        e_r, e_z = e_rz[0] + inc[0], e_rz[1] + inc[2]
        e_phi = 1j * (v + inc[1]) / point_r
        total += np.array([e_r, e_phi, e_z])  # at phi = 0: E_x = E_r, E_y = E_phi
    return total, solves


@dataclass
class Result:
    eps_gold: list
    orders: list
    intensity: list  # |E|^2 per polynomial order
    components: list  # [E_x, E_y, E_z] (re, im) of the finest
    dofs: list
    cells: int
    seconds: dict
    reference: float = REFERENCE
    sensitivity: dict | None = None  # d|E|^2 / d Re eps, d Im eps, and the +-5 % band
    deviation: float = 0.0  # (finest - reference) / reference
    notes: list = field(default_factory=list)


def run(*, orders=(2, 3, 4), size_factor: float = 1.0, sensitivity: bool = True,
        eps: complex | None = None, quick: bool = False,
        out: str | None = "gold_dimer.json") -> Result:  # fmt: skip
    """Solves the dimer for the polynomial orders on one graded mesh, optionally the
    permittivity sensitivity at the highest order; ``quick`` is the CI case (p = 2, 3 on a
    coarser mesh, no sensitivity)."""
    if quick:
        orders, size_factor, sensitivity = (2, 3), 2.0, False
    t_start = time.perf_counter()
    eps = gold_permittivity() if eps is None else complex(eps)
    mesh = meridian_mesh(size_factor)
    seconds = {"mesh": time.perf_counter() - t_start}
    intensities, dofs, components = [], [], None
    for p in orders:
        t0 = time.perf_counter()
        e, solves = gap_field(mesh, p, eps)
        seconds[f"p{p}"] = time.perf_counter() - t0
        intensities.append(float(np.sum(np.abs(e) ** 2)))
        dofs.append(solves[0].dofs)
        components = [[float(c.real), float(c.imag)] for c in e]
    result = Result(
        eps_gold=[eps.real, eps.imag], orders=list(orders), intensity=intensities,
        components=components, dofs=dofs, cells=mesh.num_cells, seconds=seconds,
        deviation=(intensities[-1] - REFERENCE) / REFERENCE,
    )  # fmt: skip
    if sensitivity:
        p = orders[-1]
        t0 = time.perf_counter()
        d = 0.01
        de_re = [
            np.sum(np.abs(gap_field(mesh, p, eps * (1 + s * d) + 0j)[0]) ** 2) for s in (1, -1)
        ]
        # Im eps scaled separately: eps.real + i eps.imag (1 +- d)
        de_im = [
            np.sum(np.abs(gap_field(mesh, p, complex(eps.real, eps.imag * (1 + s * d)))[0]) ** 2)
            for s in (1, -1)
        ]
        d_re = float((de_re[0] - de_re[1]) / (2 * d * eps.real))  # per unit of Re eps
        d_im = float((de_im[0] - de_im[1]) / (2 * d * eps.imag))  # per unit of Im eps
        band = 0.05
        result.sensitivity = {
            "d_intensity_d_re_eps": d_re,
            "d_intensity_d_im_eps": d_im,
            "relative_change_per_percent_re_eps": d_re * 0.01 * eps.real / intensities[-1],
            "relative_change_per_percent_im_eps": d_im * 0.01 * eps.imag / intensities[-1],
            "five_percent_band": [
                intensities[-1] - band * (abs(d_re * eps.real) + abs(d_im * eps.imag)),
                intensities[-1] + band * (abs(d_re * eps.real) + abs(d_im * eps.imag)),
            ],
        }
        seconds["sensitivity"] = time.perf_counter() - t0
    seconds["total"] = time.perf_counter() - t_start
    result.notes = [
        "permittivity of gold: Johnson & Christy 1972 (library); the source does not state "
        "its value",
        "orders m = 0 and +-1 only (the others vanish on the axis); cylindrical PML, PEC wall",
        "field evaluated at r = 0.002 nm, z = 0, phi = 0",
    ]
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--orders", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--size", type=float, default=1.0, help="mesh size factor")
    parser.add_argument("--no-sensitivity", action="store_true")
    parser.add_argument("--eps", type=float, nargs=2, default=None, metavar=("RE", "IM"),
                        help="permittivity of gold instead of Johnson & Christy")  # fmt: skip
    parser.add_argument("--out", default="gold_dimer.json")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    eps = None if args.eps is None else complex(args.eps[0], args.eps[1])
    r = run(orders=tuple(args.orders), size_factor=args.size, sensitivity=not args.no_sensitivity,
            eps=eps,
            quick=args.quick, out=args.out)  # fmt: skip
    print(
        f"gold eps = {r.eps_gold[0]:.4f} {r.eps_gold[1]:+.4f}i, "
        f"{r.cells} cells, {r.seconds['total']:.1f} s"
    )
    for p, i, n in zip(r.orders, r.intensity, r.dofs, strict=True):
        print(f"p = {p}: {n:>8} DoFs (m = 0), |E|^2 at the gap centre = {i:.6e} V^2/m^2")
    print(
        f"reference (MMP, Hoffmann et al. 2009): {REFERENCE:.6e}; "
        f"deviation {100 * r.deviation:+.2f} %"
    )
    if r.sensitivity:
        s = r.sensitivity
        print(
            f"sensitivity: {100 * s['relative_change_per_percent_re_eps']:+.2f} % per 1 % of "
            f"Re eps, {100 * s['relative_change_per_percent_im_eps']:+.2f} % per 1 % of Im eps; "
            f"+-5 % eps band [{s['five_percent_band'][0]:.3e}, {s['five_percent_band'][1]:.3e}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

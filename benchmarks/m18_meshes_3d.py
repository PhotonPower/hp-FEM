"""Gmsh meshes of the M18 3D cross-checks (docs/axisymmetric-layered-features.md, tests 5 and 7).

Runs in a Python with the ``gmsh`` package and does NOT import hpfem (on the maintainer's machine
the Windows Python 3.11 has gmsh, the MSYS Python that runs hpfem does not). Coordinates in nm,
second-order tetrahedra (the sphere and the cylinders curved), every layer interface, the PML
boundaries and the measurement surfaces embedded as mesh faces. Each case writes
``<out>/<case>.msh`` and ``<out>/<case>.json`` (the geometry the driver reads); meshes over 1 MB
stay out of the repository (default directory: the user's local application data).

Cases (symmetry-reduced, see benchmarks/m18_validation_3d.py):

- ``sphere_on_stack``: gold sphere (radius 40 nm, 5 nm air gap) on air / SiO2 20 nm / Si3N4
  60 nm / glass, p-polarised at 45 degrees in the x-z plane: half domain y >= 0 (the plane of
  incidence is a mirror plane, PMC on y = 0); a measurement cylinder around the sphere crossing
  the layers.
- ``nanohole``: hole of radius 100 nm through a 100 nm gold film on glass at normal incidence,
  x-polarised: quarter domain x, y >= 0 (PEC on x = 0, PMC on y = 0); the disc r <= 300 nm at
  z = -200 nm and the cylinder r <= 250 nm in the film embedded.

    python benchmarks/m18_meshes_3d.py [--case sphere_on_stack|nanohole] [--size 1.0] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import gmsh

TAG_AIR, TAG_GLASS, TAG_LAYER1, TAG_LAYER2, TAG_PARTICLE, TAG_FILM, TAG_HOLE = 1, 2, 3, 4, 5, 6, 7
TAG_PEC = 11  # boundary faces on x = 0 (nanohole)

SPHERE = {
    "case": "sphere_on_stack",
    "radius": 40.0, "gap": 5.0, "layers": [[1.45, 20.0], [2.0, 60.0]], "substrate_index": 1.5,
    "theta_deg": 45.0, "polarisation": "p",
    "x_in": 400.0, "height": 250.0, "depth": 250.0, "pml": 250.0,
    "box": {"half_width": 120.0, "below": 40.0, "above": 60.0},
    "size_particle": 8.0, "size_layers": 20.0, "size_far": 90.0, "fine_radius": 250.0,
    "near_field_points": [[0.0, 0.0, 2.5], [50.0, 0.0, 45.0], [0.0, 0.0, 95.0]],
    "tags": {"air": TAG_AIR, "glass": TAG_GLASS, "layer1": TAG_LAYER1, "layer2": TAG_LAYER2,
             "particle": TAG_PARTICLE},
}  # fmt: skip

NANOHOLE = {
    "case": "nanohole",
    "hole_radius": 100.0, "film": 100.0, "substrate_index": 1.5,
    "x_in": 700.0, "height": 400.0, "depth": 400.0, "pml": 400.0,
    "disc": {"z": -200.0, "radius": 300.0}, "region_radius": 250.0,
    "size_edge": 8.0, "size_film": 20.0, "size_film_far": 45.0, "size_far": 90.0,
    "fine_radius": 400.0,
    "tags": {"air": TAG_AIR, "glass": TAG_GLASS, "film": TAG_FILM, "hole": TAG_HOLE,
             "pec": TAG_PEC},
}  # fmt: skip


def default_out() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return Path(base) / "hpfem-m18-meshes"


def _keep_inside(lower, upper):
    """Removes the volumes whose centre of mass lies outside the box [lower, upper]."""
    for dim, tag in gmsh.model.getEntities(3):
        c = gmsh.model.occ.getCenterOfMass(dim, tag)
        if any(c[i] < lower[i] - 1e-6 or c[i] > upper[i] + 1e-6 for i in range(3)):
            gmsh.model.occ.remove([(dim, tag)], recursive=True)
    gmsh.model.occ.synchronize()


def _finish(path: Path, fields):
    """Background size field as the minimum of `fields`, second order, ASCII 4.1."""
    f_min = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", fields)
    gmsh.model.mesh.field.setAsBackgroundMesh(f_min)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT
    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.write(str(path))


def _threshold(distance_field, size_min, size_max, dist_min, dist_max):
    f = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f, "InField", distance_field)
    gmsh.model.mesh.field.setNumber(f, "SizeMin", size_min)
    gmsh.model.mesh.field.setNumber(f, "SizeMax", size_max)
    gmsh.model.mesh.field.setNumber(f, "DistMin", dist_min)
    gmsh.model.mesh.field.setNumber(f, "DistMax", dist_max)
    return f


def _box_field(size_in, size_out, lower, upper):
    f = gmsh.model.mesh.field.add("Box")
    for key, value in zip(("XMin", "YMin", "ZMin"), lower, strict=True):
        gmsh.model.mesh.field.setNumber(f, key, value)
    for key, value in zip(("XMax", "YMax", "ZMax"), upper, strict=True):
        gmsh.model.mesh.field.setNumber(f, key, value)
    gmsh.model.mesh.field.setNumber(f, "VIn", size_in)
    gmsh.model.mesh.field.setNumber(f, "VOut", size_out)
    return f


def sphere_on_stack(out: Path, size: float = 1.0, pml_size: float = 0.0) -> dict:
    g = dict(SPHERE)
    a, gap = g["radius"], g["gap"]
    t1, t2 = (t for _n, t in g["layers"])
    zc = a + gap
    x_tot = g["x_in"] + g["pml"]
    z_hi_in = zc + a + g["height"]
    z_lo_in = -(t1 + t2) - g["depth"]
    z_max, z_min = z_hi_in + g["pml"], z_lo_in - g["pml"]
    box = g["box"]
    b_lo, b_hi = -(t1 + t2) - box["below"], zc + a + box["above"]
    g.update(zc=zc, z_in=[z_lo_in, z_hi_in], z_domain=[z_min, z_max], x_total=x_tot,
             interfaces=[0.0, -t1, -(t1 + t2)], box_z=[b_lo, b_hi], size=size,
             pml_size=pml_size)  # fmt: skip
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("sphere_on_stack")
        occ = gmsh.model.occ
        width = 2 * x_tot
        domain = occ.addBox(-x_tot, 0, z_min, width, x_tot, z_max - z_min)
        tools = [
            occ.addBox(-x_tot, 0, -t1, width, x_tot, t1),
            occ.addBox(-x_tot, 0, -(t1 + t2), width, x_tot, t2),
            occ.addBox(-g["x_in"], 0, z_lo_in, 2 * g["x_in"], g["x_in"], z_hi_in - z_lo_in),
            # the measurement cylinder r <= half_width (the surface of revolution of the 2.5D
            # solver: the split up / down / lateral depends on where the side walls stand)
            occ.addCylinder(0, 0, b_lo, 0, 0, b_hi - b_lo, box["half_width"]),
            occ.addSphere(0, 0, zc, a),
        ]
        occ.fragment([(3, domain)], [(3, t) for t in tools])
        occ.synchronize()
        _keep_inside([-x_tot, 0, z_min], [x_tot, x_tot, z_max])
        groups = {}
        for _dim, tag in gmsh.model.getEntities(3):
            x, y, z = gmsh.model.occ.getCenterOfMass(3, tag)
            if math.hypot(x, y, z - zc) < a:
                material = TAG_PARTICLE
            elif z > 0:
                material = TAG_AIR
            elif z > -t1:
                material = TAG_LAYER1
            elif z > -(t1 + t2):
                material = TAG_LAYER2
            else:
                material = TAG_GLASS
            groups.setdefault(material, []).append(tag)
        for material, volumes in groups.items():
            gmsh.model.addPhysicalGroup(3, volumes, material)
        particle = [(3, v) for v in groups[TAG_PARTICLE]]
        sphere_faces = [t for _d, t in gmsh.model.getBoundary(particle, oriented=False)]
        d_sphere = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(d_sphere, "SurfacesList", sphere_faces)
        fields = [
            _threshold(d_sphere, g["size_particle"] * size, g["size_far"] * size, 5.0, 300.0),
            # the layers resolved near the sphere only (they conform everywhere)
            _box_field(
                g["size_layers"] * size,
                g["size_far"] * size,
                [-g["fine_radius"], 0, -(t1 + t2)],
                [g["fine_radius"], g["fine_radius"], 0.0],
            ),  # fmt: skip
        ]
        if pml_size > 0:  # the PML resolved: size_far inside, pml_size in the absorbing layers
            fields.append(
                _box_field(
                    g["size_far"] * size,
                    pml_size,
                    [-g["x_in"], 0, z_lo_in],
                    [g["x_in"], g["x_in"], z_hi_in],
                )
            )
        path = out / "sphere_on_stack.msh"
        _finish(path, fields)
    finally:
        gmsh.finalize()
    return g


def nanohole(out: Path, size: float = 1.0, pml_size: float = 0.0) -> dict:
    g = dict(NANOHOLE)
    a, t = g["hole_radius"], g["film"]
    x_tot = g["x_in"] + g["pml"]
    z_hi_in, z_lo_in = g["height"], -t - g["depth"]
    z_max, z_min = z_hi_in + g["pml"], z_lo_in - g["pml"]
    z_disc, r_disc = g["disc"]["z"], g["disc"]["radius"]
    g.update(
        z_in=[z_lo_in, z_hi_in],
        z_domain=[z_min, z_max],
        x_total=x_tot,
        size=size,
        pml_size=pml_size,
    )
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("nanohole")
        occ = gmsh.model.occ
        domain = occ.addBox(0, 0, z_min, x_tot, x_tot, z_max - z_min)
        tools = [
            occ.addBox(0, 0, -t, x_tot, x_tot, t),                           # the film
            occ.addBox(0, 0, z_disc, x_tot, x_tot, -t - z_disc),             # glass above the disc
            occ.addBox(0, 0, z_lo_in, g["x_in"], g["x_in"], z_hi_in - z_lo_in),  # inside the PML
            occ.addCylinder(0, 0, -t, 0, 0, t, a),                           # the hole
            occ.addCylinder(0, 0, -t, 0, 0, t, g["region_radius"]),          # absorption region
            occ.addCylinder(0, 0, z_disc - 40.0, 0, 0, 80.0, r_disc),        # the disc edge
        ]  # fmt: skip
        occ.fragment([(3, domain)], [(3, s) for s in tools])
        occ.synchronize()
        _keep_inside([0, 0, z_min], [x_tot, x_tot, z_max])
        groups = {}
        for _dim, tag in gmsh.model.getEntities(3):
            x, y, z = gmsh.model.occ.getCenterOfMass(3, tag)
            if z > 0:
                material = TAG_AIR
            elif z > -t:
                material = TAG_HOLE if math.hypot(x, y) < a else TAG_FILM
            else:
                material = TAG_GLASS
            groups.setdefault(material, []).append(tag)
        for material, volumes in groups.items():
            gmsh.model.addPhysicalGroup(3, volumes, material)
        pec = []
        for _dim, tag in gmsh.model.getEntities(2):
            x0, _y0, _z0, x1, _y1, _z1 = gmsh.model.getBoundingBox(2, tag)
            if abs(x0) < 1e-6 and abs(x1) < 1e-6:
                pec.append(tag)
        gmsh.model.addPhysicalGroup(2, pec, TAG_PEC)
        # sizes: fine at the rim of the hole, the film resolved, coarse far away
        rim = [tag for _d, tag in gmsh.model.getEntities(1)
               if _is_rim(tag, a, t)]  # fmt: skip
        d_rim = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(d_rim, "CurvesList", rim)
        fields = [
            _threshold(d_rim, g["size_edge"] * size, g["size_far"] * size, 10.0, 400.0),
            _box_field(
                g["size_film"] * size,
                g["size_far"] * size,
                [0, 0, -t],
                [g["fine_radius"], g["fine_radius"], 0.0],
            ),  # fmt: skip
            _box_field(
                g["size_film_far"] * size, g["size_far"] * size, [0, 0, -t], [x_tot, x_tot, 0.0]
            ),  # fmt: skip
        ]
        if pml_size > 0:
            fields.append(
                _box_field(
                    g["size_far"] * size, pml_size, [0, 0, z_lo_in], [g["x_in"], g["x_in"], z_hi_in]
                )
            )
        path = out / "nanohole.msh"
        _finish(path, fields)
    finally:
        gmsh.finalize()
    return g


def _is_rim(curve: int, radius: float, film: float) -> bool:
    """The circular edges of the hole at the top and the bottom of the film."""
    x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(1, curve)
    flat = abs(z1 - z0) < 1e-6 and (abs(z0) < 1e-6 or abs(z0 + film) < 1e-6)
    return flat and abs(max(x1, y1) - radius) < 1e-3 * radius and min(x0, y0) < 1e-6


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", choices=("sphere_on_stack", "nanohole"), action="append")
    parser.add_argument("--size", type=float, default=1.0, help="factor on every mesh size")
    parser.add_argument(
        "--pml-size", type=float, default=0.0, help="mesh size in the PML [nm] (0: the far size)"
    )
    parser.add_argument("--out", type=Path, default=default_out())
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for case in args.case or ("sphere_on_stack", "nanohole"):
        build = sphere_on_stack if case == "sphere_on_stack" else nanohole
        geometry = build(args.out, args.size, args.pml_size)
        (args.out / f"{case}.json").write_text(json.dumps(geometry, indent=1), encoding="utf-8")
        size = (args.out / f"{case}.msh").stat().st_size
        print(f"{case}: {args.out / (case + '.msh')} ({size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

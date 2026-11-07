"""Gmsh mesh of the 3D add-drop ring resonator (needs only `gmsh` and NumPy, not hpfem, so the
mesh can be built in another Python environment than the solver's).

Geometry in nm (x along the bus waveguides, y across, z up): a silicon ring of centre-line
radius R, width w and height h at the origin, two straight buses of the same cross-section at
y = -/+ (R + w + gap), all in silica between the PEC walls of the box; second-order (curved)
tetrahedra, graded from `cell_core` within the cores to `cell_far` at `grade` from them.

Physical groups: volume 1 cladding, 2 silicon; surface 10 the walls, 11 the face x = -X,
12 the face x = +X (the solver splits each face at y = 0 into the ports of the two buses),
15 the symmetry plane z = h/2 of the half model (`half=True`: only z >= h/2 is meshed; the
cladding is the same above and below, so the TE-like fields are even about the mid-plane of
the cores and the plane is a magnetic wall).

Run `python examples/ring_resonator_3d/ring_mesh.py --out ring.msh [--radius um] [--gap nm]
[--cell-core nm] [--cell-far nm] [--half]`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

TAG_CLAD, TAG_SI = 1, 2
TAG_WALL, TAG_LEFT, TAG_RIGHT, TAG_SYMMETRY = 10, 11, 12, 15


@dataclass
class Geometry:
    """Lengths in nm."""

    radius: float = 2000.0  # centre line of the ring
    width: float = 450.0
    height: float = 220.0
    gap: float = 200.0  # between ring and each bus
    margin_x: float = 1000.0  # cladding beyond the ring along the buses (to the port faces)
    margin_y: float = 750.0  # beside the outer bus edges
    margin_z: float = 605.0  # below and above the cores
    cell_core: float = 100.0  # target cell size in and next to the cores
    cell_far: float = 300.0  # in the cladding at `grade` from the cores
    grade: float = 600.0
    half: bool = False  # only z >= height / 2, the mid-plane of the cores a magnetic wall

    @property
    def bus_y(self) -> float:
        """Centre line of the buses, y = -/+ bus_y."""
        return self.radius + self.width + self.gap

    @property
    def half_x(self) -> float:
        return self.radius + self.width / 2 + self.margin_x

    @property
    def half_y(self) -> float:
        return self.bus_y + self.width / 2 + self.margin_y


def build(geometry: Geometry, path: str) -> str:
    """Writes the Gmsh 4.1 ASCII mesh to `path` and returns it."""
    import gmsh

    g = geometry
    X, Y = g.half_x, g.half_y
    z0, z1 = -g.margin_z, g.height + g.margin_z
    zc = g.height / 2 if g.half else 0.0  # bottom of the cores in the model
    if g.half:
        z0 = zc
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("ring_resonator_3d")
        occ = gmsh.model.occ
        box = occ.addBox(-X, -Y, z0, 2 * X, 2 * Y, z1 - z0)
        hc = g.height - zc
        outer = occ.addCylinder(0, 0, zc, 0, 0, hc, g.radius + g.width / 2)
        inner = occ.addCylinder(0, 0, zc, 0, 0, hc, g.radius - g.width / 2)
        ring, _ = occ.cut([(3, outer)], [(3, inner)])
        buses = [occ.addBox(-X, s * g.bus_y - g.width / 2, zc, 2 * X, g.width, hc) for s in (-1, 1)]
        cores = ring + [(3, b) for b in buses]
        _, parts = occ.fragment([(3, box)], cores)
        occ.synchronize()
        silicon = sorted({t for part in parts[1:] for _, t in part})
        cladding = sorted({t for _, t in parts[0]} - set(silicon))
        if len(silicon) != 3 or len(cladding) != 1:
            raise RuntimeError(f"ring mesh: {len(silicon)} cores, {len(cladding)} claddings")
        gmsh.model.addPhysicalGroup(3, cladding, TAG_CLAD, "cladding")
        gmsh.model.addPhysicalGroup(3, silicon, TAG_SI, "silicon")
        tol = 1e-3 * g.cell_core
        left, right, walls, symmetry = [], [], [], []
        for dim, tag in gmsh.model.getBoundary(
            [(3, t) for t in cladding + silicon], oriented=False
        ):
            x0, y0, zz0, x1, y1, zz1 = gmsh.model.getBoundingBox(dim, tag)
            if x1 < -X + tol:
                left.append(tag)
            elif x0 > X - tol:
                right.append(tag)
            elif g.half and zz1 < z0 + tol:
                symmetry.append(tag)
            elif y1 < -Y + tol or y0 > Y - tol or zz1 < z0 + tol or zz0 > z1 - tol:
                walls.append(tag)
        gmsh.model.addPhysicalGroup(2, walls, TAG_WALL, "wall")
        gmsh.model.addPhysicalGroup(2, left, TAG_LEFT, "port_left")
        gmsh.model.addPhysicalGroup(2, right, TAG_RIGHT, "port_right")
        if symmetry:
            gmsh.model.addPhysicalGroup(2, symmetry, TAG_SYMMETRY, "symmetry")

        core_faces = [
            t for _, t in gmsh.model.getBoundary([(3, t) for t in silicon], oriented=False)
        ]
        field = gmsh.model.mesh.field
        dist = field.add("Distance")
        field.setNumbers(dist, "SurfacesList", core_faces)
        field.setNumber(dist, "Sampling", 60)
        thr = field.add("Threshold")
        field.setNumber(thr, "InField", dist)
        field.setNumber(thr, "SizeMin", g.cell_core)
        field.setNumber(thr, "SizeMax", g.cell_far)
        field.setNumber(thr, "DistMin", 0.5 * g.cell_core)
        field.setNumber(thr, "DistMax", g.grade)
        field.setAsBackgroundMesh(thr)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
        gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)
        gmsh.model.mesh.generate(3)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.write(path)
    finally:
        gmsh.finalize()
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--radius", type=float, default=2.0, help="ring centre-line radius [um]")
    parser.add_argument("--gap", type=float, default=200.0, help="ring-bus gap [nm]")
    parser.add_argument("--cell-core", type=float, default=100.0, help="[nm]")
    parser.add_argument("--cell-far", type=float, default=300.0, help="[nm]")
    parser.add_argument("--half", action="store_true", help="half model above z = h/2")
    args = parser.parse_args(argv)
    geometry = Geometry(
        radius=1000 * args.radius,
        gap=args.gap,
        cell_core=args.cell_core,
        cell_far=args.cell_far,
        half=args.half,
    )
    build(geometry, args.out)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

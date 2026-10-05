"""EUV mask: 3D absorber on a Mo/Si multilayer under oblique incidence, near-field export.

An EUV photomask is a Mo/Si Bragg mirror (bilayers of 6.9 nm, reflective around 13.5 nm)
with a patterned absorber (here a tantalum pad) on top, lit at the chief-ray angle of 6
degrees. The unit cell is Bloch-periodic in x and y, has PML in the vacuum above and in the
silicon substrate below, and is solved in the scattered-field formulation with the plane
wave as incident field. Outputs: the reflectivity of the cell from the Poynting flux of
the scattered field through a plane above the absorber (against the transfer-matrix value
of the bare mirror as a check) and the near field exported as VTK for ParaView.

The mirror is shortened to a few bilayers to keep the 3D problem small; the optical
constants at 13.5 nm are n = 1 - delta + i beta from the CXRO tables (Henke, Gullikson &
Davis 1993): Mo delta = 0.0774, beta = 0.00644; Si delta = 0.00100, beta = 0.00183;
Ta delta = 0.0588, beta = 0.0410.

Run `python examples/euv_mask/run.py [--quick]`; results go to `euv_mask.json`, the near
field to `euv_mask.vtu`.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass

import numpy as np

import hpfem
from hpfem import units

WAVELENGTH = 13.5 * units.nm
ANGLE = 6.0 * units.deg  # chief-ray angle from the normal, in the x-z plane
EUV = {  # CXRO, 13.5 nm: n = 1 - delta + i beta (Im eps > 0 with exp(-i omega t))
    "Mo": complex(1 - 0.0774, 0.00644),
    "Si": complex(1 - 0.00100, 0.00183),
    "Ta": complex(1 - 0.0588, 0.0410),
}
MO, SI = 2.76 * units.nm, 4.14 * units.nm  # bilayer 6.9 nm (Mo fraction 0.4)
TAG_MO, TAG_SI, TAG_TA = 2, 3, 4
TAG_PERIODIC = (hpfem.box_tag.X_MIN, hpfem.box_tag.X_MAX, hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX)


@dataclass
class Result:
    reflectivity: float
    reflectivity_bare_fem: float
    reflectivity_bare_tmm: float
    absorber_nm: float
    pitch_nm: float
    bilayers: int
    num_dofs: int


# --- transfer matrix of the bare mirror (s polarisation, oblique incidence) -------------------


def tmm_reflectivity(bilayers: int, angle: float) -> float:
    """Reflectivity of the Mo/Si stack on a silicon substrate at the given angle, electric
    field perpendicular to the plane of incidence (the polarisation of the example)."""
    k0 = 2 * np.pi / WAVELENGTH
    kx = k0 * np.sin(angle)

    def kz(n):
        return np.sqrt((k0 * n) ** 2 - kx**2 + 0j)

    layers = [(EUV["Si"], SI), (EUV["Mo"], MO)] * bilayers  # from the top: Si cap, Mo, ...
    total = np.eye(2, dtype=complex)
    for n, d in layers:
        q = kz(n)
        m = np.array(
            [[np.cos(q * d), -1j * np.sin(q * d) / q], [-1j * q * np.sin(q * d), np.cos(q * d)]]
        )
        total = total @ m
    q0, qs = kz(1.0), kz(EUV["Si"])
    denominator = (total[0, 0] + total[0, 1] * qs) * q0 + (total[1, 0] + total[1, 1] * qs)
    r = ((total[0, 0] + total[0, 1] * qs) * q0 - (total[1, 0] + total[1, 1] * qs)) / denominator
    return float(abs(r) ** 2)


# --- mesh --------------------------------------------------------------------------------------


def kuhn_mesh(x_nodes, y_nodes, z_nodes, tag_of_cube):
    """Structured tetrahedral mesh of the box grid (six Kuhn tetrahedra per cube, all sharing
    the cube's diagonal, conforming across cubes); `tag_of_cube(i, j, k)` gives the cell tag.
    The six box sides get the box_tag facet tags."""
    nx, ny, nz = len(x_nodes), len(y_nodes), len(z_nodes)
    xs, ys, zs = np.meshgrid(x_nodes, y_nodes, z_nodes, indexing="ij")
    vertices = np.column_stack([xs.ravel(order="F"), ys.ravel(order="F"), zs.ravel(order="F")])

    def vid(i, j, k):
        return i + nx * (j + ny * k)

    # Kuhn / Freudenthal: the six permutations of the unit cube's path from 0 to 7
    paths = [(1, 2, 4), (1, 4, 2), (2, 1, 4), (2, 4, 1), (4, 1, 2), (4, 2, 1)]
    cells, tags = [], []
    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                corner = {
                    0: vid(i, j, k),
                    1: vid(i + 1, j, k),
                    2: vid(i, j + 1, k),
                    4: vid(i, j, k + 1),
                    3: vid(i + 1, j + 1, k),
                    5: vid(i + 1, j, k + 1),
                    6: vid(i, j + 1, k + 1),
                    7: vid(i + 1, j + 1, k + 1),
                }
                tag = tag_of_cube(i, j, k)
                for path in paths:
                    a = 0
                    tet = [corner[0]]
                    for step in path:
                        a |= step
                        tet.append(corner[a])
                    cells.append(tet)
                    tags.append(tag)
    mesh = hpfem.Mesh3D(vertices, np.array(cells), tags)
    bounds = {
        hpfem.box_tag.X_MIN: (0, x_nodes[0]),
        hpfem.box_tag.X_MAX: (0, x_nodes[-1]),
        hpfem.box_tag.Y_MIN: (1, y_nodes[0]),
        hpfem.box_tag.Y_MAX: (1, y_nodes[-1]),
        hpfem.box_tag.Z_MIN: (2, z_nodes[0]),
        hpfem.box_tag.Z_MAX: (2, z_nodes[-1]),
    }
    tolerance = 1e-9 * max(
        x_nodes[-1] - x_nodes[0], y_nodes[-1] - y_nodes[0], z_nodes[-1] - z_nodes[0]
    )
    for f in mesh.boundary_facets:
        v = mesh.vertices[list(mesh.facet_vertices(f))]
        for tag, (axis, value) in bounds.items():
            if np.all(
                np.abs(v[:, axis] - value) < tolerance
            ):  # (np.allclose's atol is far too loose for nm)
                mesh.set_facet_tag(f, tag)
                break
    return mesh


def grid(intervals, cell):
    """1D nodes over consecutive (x0, x1, tag) intervals with at least one cell per interval."""
    nodes, tags = [intervals[0][0]], []
    for x0, x1, tag in intervals:
        count = max(1, int(np.ceil((x1 - x0) / cell - 1e-9)))
        if tag == 0:
            count += count % 2  # even: a node in the middle of the vacuum margin (flux plane)
        for m in range(1, count + 1):
            nodes.append(x0 + (x1 - x0) * m / count)
            tags.append(tag)
    return np.array(nodes), tags


def mask_mesh(pitch, absorber, height, bilayers, lateral_cells, margin, pml, vertical_cell):
    """(mesh, z of the plane above the absorber, z extent) of the unit cell: substrate PML
    and margin below, the Mo/Si stack, the absorber pad (0 for the bare mirror), vacuum
    margin and PML above."""
    stack = []
    z = 0.0
    for _ in range(bilayers):
        stack += [(z, z + MO, TAG_MO), (z + MO, z + MO + SI, TAG_SI)]
        z += MO + SI
    top = z
    intervals = [(-(margin + pml), -margin, TAG_SI), (-margin, 0.0, TAG_SI)] + stack
    if absorber > 0:
        intervals.append((top, top + height, TAG_TA))
    else:
        height = 0.0
    z_end = top + height + margin + pml
    intervals += [(top + height, top + height + margin, 0), (top + height + margin, z_end, 0)]
    z_nodes, z_tags = grid(intervals, vertical_cell)
    lateral = pitch / lateral_cells
    x_nodes = np.linspace(-pitch / 2, pitch / 2, lateral_cells + 1)
    y_nodes = np.linspace(-pitch / 2, pitch / 2, lateral_cells + 1)

    def tag_of_cube(i, j, k):
        tag = z_tags[k]
        if tag == TAG_TA:
            inside = (
                abs(x_nodes[i] + lateral / 2) < absorber / 2
                and abs(y_nodes[j] + lateral / 2) < absorber / 2
            )
            return TAG_TA if inside else 0
        return tag

    mesh = kuhn_mesh(x_nodes, y_nodes, z_nodes, tag_of_cube)
    return mesh, top + height + 0.5 * margin, (-(margin + pml), z_end, top + height, margin, pml)


def plane_surface(mesh, z_plane):
    """Facets on the plane z = z_plane with the cell below as inside (normal +z)."""
    vertices = mesh.vertices
    tolerance = 1e-9 * (vertices[:, 2].max() - vertices[:, 2].min())
    facets = []
    for f in range(mesh.num_facets):
        v = vertices[list(mesh.facet_vertices(f))]
        if np.all(np.abs(v[:, 2] - z_plane) < tolerance):
            cells = [c for c in mesh.facet_cells(f) if c != hpfem.INVALID_INDEX]
            below = min(cells, key=lambda c: mesh.cell_centroid(c)[2])
            facets.append(hpfem.Surface3D.Facet(f, below))
    surface = hpfem.Surface3D()
    surface.facets = facets
    return surface


# --- solve --------------------------------------------------------------------------------------


def reflectivity(
    pitch,
    absorber,
    height,
    bilayers,
    lateral_cells=8,
    order=2,
    vertical_cell=None,
    export=None,
    margin=10 * units.nm,
    pml=13.5 * units.nm,
):
    """Reflectivity of the unit cell (scattered-field flux through the plane above the
    absorber over the incident flux) and the number of DoFs; optionally exports the near
    field of the total field to `export`."""
    vertical_cell = vertical_cell or 2 * units.nm
    mesh, z_plane, (z_lo, z_hi, z_top, margin, pml) = mask_mesh(
        pitch, absorber, height, bilayers, lateral_cells, margin, pml, vertical_cell
    )
    dofs = hpfem.NedelecDofMap3D(mesh, order)
    k0 = 2 * np.pi / WAVELENGTH
    k = k0 * np.array([np.sin(ANGLE), 0.0, -np.cos(ANGLE)])  # downwards, tilted in x
    e0 = np.array([0.0, 1.0, 0.0])  # s polarisation: E along y, perpendicular to k
    setup = hpfem.ScatteringSetup3D()
    setup.omega = k0 * hpfem.constants.c0
    setup.materials.set(TAG_MO, hpfem.Material(EUV["Mo"] ** 2))
    setup.materials.set(TAG_SI, hpfem.Material(EUV["Si"] ** 2))
    setup.materials.set(TAG_TA, hpfem.Material(EUV["Ta"] ** 2))
    setup.incident = hpfem.plane_wave(e0, k)
    setup.formulation = hpfem.Formulation.SCATTERED_FIELD
    setup.pml = hpfem.PmlBox3D(
        [-pitch / 2, -pitch / 2, z_lo + pml],
        [pitch / 2, pitch / 2, z_hi - pml],
        [0.0, 0.0, 0.0, 0.0, pml, pml],
        k0,
        1.0,
        hpfem.PmlProfile(2, 1e-8),  # near-normal incidence; oblique: PmlProfile.for_angle
    )
    setup.pec_tags = [hpfem.box_tag.Z_MIN, hpfem.box_tag.Z_MAX]
    shift_x, shift_y = np.array([pitch, 0.0, 0.0]), np.array([0.0, pitch, 0.0])
    setup.periodic = [
        hpfem.PeriodicPair3D(
            hpfem.box_tag.X_MIN, hpfem.box_tag.X_MAX, shift_x, hpfem.bloch_phase(k, shift_x)
        ),
        hpfem.PeriodicPair3D(
            hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX, shift_y, hpfem.bloch_phase(k, shift_y)
        ),
    ]
    problem = hpfem.Scattering3D(dofs, setup)
    solution = problem.solve()
    surface = plane_surface(mesh, z_plane)
    scattered_flux = hpfem.poynting_flux(
        dofs, solution.unknown, setup.omega, setup.materials, surface
    )
    incident = hpfem.plane_wave_intensity(1.0, hpfem.Material.vacuum()) * np.cos(ANGLE) * pitch**2
    if export:  # near field: total field on the (once) subdivided mesh, about 10 MB
        total = solution.unknown + hpfem.interpolate(dofs, setup.incident.value)
        hpfem.FieldExporter3D(mesh, 1).hcurl("E_total", dofs, total).write(export)
    return scattered_flux / incident, dofs.num_dofs


def run(quick: bool) -> Result:
    pitch = 32 * units.nm
    absorber, height = 16 * units.nm, 16 * units.nm
    bilayers = 3 if quick else 6
    lateral = 4 if quick else 8  # cells across the pitch (8 nm / 4 nm)
    order = 2
    r_bare, _ = reflectivity(pitch, 0.0, height, bilayers, lateral, order)
    r_tmm = tmm_reflectivity(bilayers, ANGLE)
    r_mask, num_dofs = reflectivity(
        pitch, absorber, height, bilayers, lateral, order, export="euv_mask.vtu"
    )
    return Result(r_mask, r_bare, r_tmm, absorber / units.nm, pitch / units.nm, bilayers, num_dofs)


def main(argv=None) -> int:
    hpfem.set_log_level("warn")
    quick = "--quick" in (argv or sys.argv[1:])
    result = run(quick)
    print(f"Mo/Si mirror with {result.bilayers} bilayers at {ANGLE / units.deg:.0f} deg, 13.5 nm")
    print(
        f"bare mirror reflectivity: {result.reflectivity_bare_fem:.4f} (FEM), "
        f"{result.reflectivity_bare_tmm:.4f} (TMM)"
    )
    print(
        f"{result.absorber_nm:.0f} nm Ta pad on a {result.pitch_nm:.0f} nm pitch: "
        f"reflectivity {result.reflectivity:.4f} ({result.num_dofs} DoF)"
    )
    with open("euv_mask.json", "w", encoding="utf-8") as out:
        json.dump(asdict(result), out, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

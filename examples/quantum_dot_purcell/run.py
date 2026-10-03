"""Quantum dot in a micropillar cavity: Purcell factor and beta factor.

A point emitter is a classical dipole; the Purcell factor F_P = P / P_0 is the power it
radiates in the structure divided by the power it radiates in the homogeneous host, and
the beta factor is the fraction of the emitted power that leaves through the top mirror
into the collection direction. The effective 2D model is a line dipole (in-plane moment
across the ridge, H_z polarisation) in the one-wavelength GaAs cavity of a GaAs / AlAs DBR
ridge ("pillar") of finite width in air, on the GaAs substrate.

The dipole is a narrow Gaussian line current (`hpfem.gaussian_current`, width well below
the wavelength, resolved by the mesh) in the total-field formulation, so the PML sees the
physical field; the emitted power is the Poynting flux through a small closed surface
around the current, and P_0 is the same current solved on the same mesh with every
material set to the host.
The ratio is insensitive to the width of the current.

Validation inside the script (`check_mirror`): the same current in front of a PEC mirror,
whose exact power ratio follows from the dipole plus its image, both given by
`hpfem.dipole_field`; the finite width leaves a difference below one per cent.

Run `python examples/quantum_dot_purcell/run.py [--quick]`; results go to
`quantum_dot_purcell.json`, the spectrum to `quantum_dot_purcell.png`.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass

import numpy as np

import hpfem
from hpfem import units

LAMBDA_DESIGN = 950 * units.nm  # InAs quantum dots emit around 950 nm
N_HIGH, N_LOW, N_AIR = 3.48, 2.94, 1.0
TAG_HIGH, TAG_LOW, TAG_AIR = 2, 3, 5
BOX_SIDES = [hpfem.box_tag.X_MIN, hpfem.box_tag.X_MAX, hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX]


@dataclass
class Spectrum:
    wavelength_nm: list[float]
    purcell: list[float]
    beta_top: list[float]
    num_dofs: int


# --- sources and surfaces ---------------------------------------------------------------------


def gaussian_current(position, moment, sigma, omega):
    """f = i omega mu0 J of a line dipole of unit moment smeared over a Gaussian of width
    sigma, evaluated in C++ (`hpfem.gaussian_current`; a Python callback would serialise the
    parallel assembly on the GIL)."""
    return hpfem.gaussian_current(
        np.asarray(position, dtype=float), np.asarray(moment, dtype=complex), sigma, omega
    )


def circle_surface(mesh, center, radius):
    """Closed surface of the mesh facets around the cells inside the circle (normal outwards),
    built by tagging those cells temporarily."""
    tag = 99
    saved = mesh.cell_tags.copy()
    for c, x in enumerate(mesh.cell_centroids):
        if np.hypot(x[0] - center[0], x[1] - center[1]) < radius:
            mesh.set_cell_tag(c, tag)
    surface = hpfem.Surface2D.around_cells(mesh, tag)
    for c, t in enumerate(saved):
        mesh.set_cell_tag(c, int(t))
    return surface


def line_surface(mesh, y_line):
    """Open surface of all facets on the mesh line y = y_line, normal in +y (the inside cell
    is the one below)."""
    vertices = mesh.vertices
    facets = []
    for f in range(mesh.num_facets):
        a, b = mesh.facet_vertices(f)
        if abs(vertices[a, 1] - y_line) < 1e-12 and abs(vertices[b, 1] - y_line) < 1e-12:
            cells = [c for c in mesh.facet_cells(f) if c != hpfem.INVALID_INDEX]
            below = min(cells, key=lambda c: mesh.cell_centroid(c)[1])
            facets.append(hpfem.Surface2D.Facet(f, below))
    surface = hpfem.Surface2D()
    surface.facets = facets
    return surface


def emitted_power(dofs, solution, omega, materials, surface, order=2):
    """Poynting flux of the discrete field through the surface [W/m]."""
    return hpfem.poynting_flux(dofs, solution.unknown, omega, materials, surface, order)


# --- validation: current in front of a PEC mirror ----------------------------------------------


def check_mirror(distance=0.3, k=2 * np.pi, cells=48, order=3, sigma=0.04):
    """Power ratio of a line dipole (moment along the mirror) at `distance` from a PEC plane:
    FEM (Gaussian current, total field, divided by the same current in free space) against
    the image solution evaluated with the flux post-processing."""
    half, pml = 2.0, 1.0
    full_mesh = hpfem.rectangle(cells, cells, [-half - pml, -half - pml], [half + pml, half + pml])
    mesh = hpfem.extract(full_mesh, lambda x: x[0] > 0)  # the mirror is the plane x = 0
    for f in mesh.boundary_facets:
        if abs(mesh.vertices[mesh.facet_vertices(f)[0], 0]) < 1e-12:
            mesh.set_facet_tag(f, 7)
    omega = k * hpfem.constants.c0
    moment = np.array([0.0, 1.0])  # parallel to the mirror
    position = np.array([distance, 0.0])
    profile = hpfem.PmlProfile(2, 1e-10)

    def solve(mesh, pml_box, pec_tags):
        dofs = hpfem.NedelecDofMap2D(mesh, order)
        setup = hpfem.ScatteringSetup2D()
        setup.omega = omega
        setup.current = gaussian_current(position, moment, sigma, omega)
        setup.formulation = hpfem.Formulation.TOTAL_FIELD
        setup.pml = pml_box
        setup.pec_tags = pec_tags
        problem = hpfem.Scattering2D(dofs, setup)
        surface = circle_surface(mesh, position, 0.5 * distance)
        return emitted_power(dofs, problem.solve(), omega, setup.materials, surface), surface

    p_mirror, surface = solve(
        mesh,
        hpfem.PmlBox2D([0.0, -half], [half, half], [0.0, pml, pml, pml], k, 1.0, profile),
        [7, hpfem.box_tag.X_MAX, hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX],
    )
    p_free, _ = solve(
        full_mesh,
        hpfem.PmlBox2D.uniform([-half, -half], [half, half], pml, k, 1.0, profile),
        BOX_SIDES,
    )
    # exact point dipole: dipole plus image (anti-parallel moment at -distance)
    dipole = hpfem.dipole_field(position, moment, k)
    image = hpfem.dipole_field(np.array([-distance, 0.0]), -moment, k)
    both = hpfem.combined_field(hpfem.analytic_field(dipole), hpfem.analytic_field(image), 1.0)
    vacuum = hpfem.MaterialMap()
    p_exact = hpfem.poynting_flux(mesh, surface, both, omega, vacuum, 8)
    p0_exact = hpfem.poynting_flux(mesh, surface, hpfem.analytic_field(dipole), omega, vacuum, 8)
    return p_mirror / p_free, p_exact / p0_exact


# --- the micropillar -----------------------------------------------------------------------------


def pillar_mesh(top_pairs, bottom_pairs, width, margin, pml, cell_fraction=0.08):
    """Structured mesh of the ridge (x across, y vertical, light leaves in +y) with nodes on
    every layer interface: returns (mesh, cavity centre y, stack height, half width)."""
    q_high = LAMBDA_DESIGN / (4 * N_HIGH)
    q_low = LAMBDA_DESIGN / (4 * N_LOW)
    layers = []
    for _ in range(bottom_pairs):
        layers += [(N_HIGH, q_high, TAG_HIGH), (N_LOW, q_low, TAG_LOW)]
    layers.append((N_HIGH, LAMBDA_DESIGN / N_HIGH, TAG_HIGH))
    cavity_centre = sum(t for _, t, _ in layers[:-1]) + layers[-1][1] / 2
    for _ in range(top_pairs):
        layers += [(N_LOW, q_low, TAG_LOW), (N_HIGH, q_high, TAG_HIGH)]
    height = sum(t for _, t, _ in layers)
    cell = cell_fraction * LAMBDA_DESIGN

    def grid(intervals):
        nodes, tags = [intervals[0][0]], []
        for x0, x1, tag in intervals:
            count = max(2, int(np.ceil((x1 - x0) / cell)))
            count += count % 2  # even: a node at the middle of every interval (flux lines)
            for j in range(1, count + 1):
                nodes.append(x0 + (x1 - x0) * j / count)
                tags.append(tag)
        return np.array(nodes), tags

    intervals = [(-(margin + pml), -margin, TAG_HIGH), (-margin, 0.0, TAG_HIGH)]
    y = 0.0
    for _, t, tag in layers:
        intervals.append((y, y + t, tag))
        y += t
    intervals += [
        (height, height + margin, TAG_AIR),
        (height + margin, height + margin + pml, TAG_AIR),
    ]
    y_nodes, y_tags = grid(intervals)
    half = width / 2 + margin + pml
    x_intervals = [(-half, -half + pml), (-half + pml, -width / 2), (-width / 2, width / 2),
                   (width / 2, half - pml), (half - pml, half)]  # fmt: skip
    x_nodes, _ = grid([(x0, x1, 0) for x0, x1 in x_intervals])
    nx, ny = len(x_nodes), len(y_nodes)
    vertices = np.array([[x, yy] for yy in y_nodes for x in x_nodes])
    cells, tags = [], []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a, b, c, d = j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i
            cells += [[a, b, c], [a, c, d]]
            inside = abs(0.5 * (x_nodes[i] + x_nodes[i + 1])) < width / 2
            y_mid = 0.5 * (y_nodes[j] + y_nodes[j + 1])
            tag = y_tags[j] if (inside or y_mid < 0) else TAG_AIR
            tags += [tag, tag]
    mesh = hpfem.Mesh2D(vertices, np.array(cells), tags)
    tolerance = 1e-9 * half  # np.allclose's default atol (1e-8) is far too loose for metres
    for f in mesh.boundary_facets:
        v = mesh.vertices[mesh.facet_vertices(f)]
        if np.all(np.abs(v[:, 0] + half) < tolerance):
            mesh.set_facet_tag(f, hpfem.box_tag.X_MIN)
        elif np.all(np.abs(v[:, 0] - half) < tolerance):
            mesh.set_facet_tag(f, hpfem.box_tag.X_MAX)
        elif np.all(np.abs(v[:, 1] - y_nodes[0]) < tolerance):
            mesh.set_facet_tag(f, hpfem.box_tag.Y_MIN)
        else:
            mesh.set_facet_tag(f, hpfem.box_tag.Y_MAX)
    return mesh, cavity_centre, height, half


def spectrum(
    wavelengths, top_pairs=6, bottom_pairs=12, width=1.5 * units.um, order=3, cell_fraction=0.08
):
    margin, pml = 0.5 * LAMBDA_DESIGN, 1.0 * LAMBDA_DESIGN
    mesh, y_dipole, height, half = pillar_mesh(
        top_pairs, bottom_pairs, width, margin, pml, cell_fraction
    )
    dofs = hpfem.NedelecDofMap2D(mesh, order)
    position = np.array([0.0, y_dipole])
    moment = np.array([1.0, 0.0])  # across the ridge: emission along the axis
    sigma = 0.25 * cell_fraction * LAMBDA_DESIGN
    around = circle_surface(mesh, position, 0.1 * LAMBDA_DESIGN)
    top_line = line_surface(mesh, height + 0.5 * margin)  # in the air above the top mirror
    assert len(top_line.facets) > 0
    purcell, beta = [], []
    for lam in wavelengths:
        omega = units.angular_frequency(wavelength=lam)
        k0 = units.vacuum_wavenumber(omega)
        pml_box = hpfem.PmlBox2D(
            [-half + pml, -margin], [half - pml, height + margin], [pml, pml, pml, pml], k0, 1.0,
            hpfem.PmlProfile(2, 1e-10),
        )  # fmt: skip
        powers = []
        p_top = 0.0
        for homogeneous in (False, True):
            setup = hpfem.ScatteringSetup2D()
            setup.omega = omega
            setup.materials = hpfem.MaterialMap(hpfem.Material.dielectric(N_HIGH))
            if not homogeneous:
                setup.materials.set(TAG_LOW, hpfem.Material.dielectric(N_LOW))
                setup.materials.set(TAG_AIR, hpfem.Material.vacuum())
            setup.current = gaussian_current(position, moment, sigma, omega)
            setup.formulation = hpfem.Formulation.TOTAL_FIELD
            setup.pml = pml_box
            setup.pec_tags = BOX_SIDES
            problem = hpfem.Scattering2D(dofs, setup)
            solution = problem.solve()
            powers.append(emitted_power(dofs, solution, omega, setup.materials, around))
            if not homogeneous:
                p_top = emitted_power(dofs, solution, omega, setup.materials, top_line)
        purcell.append(powers[0] / powers[1])
        beta.append(p_top / powers[0])
    return Spectrum([lam / units.nm for lam in wavelengths], purcell, beta, dofs.num_dofs)


def main(argv=None) -> int:
    hpfem.set_log_level("warn")
    quick = "--quick" in (argv or sys.argv[1:])
    fem, exact = check_mirror()
    print(f"dipole in front of a PEC mirror: F = {fem:.4f} (FEM), {exact:.4f} (image dipole)")
    wavelengths = (np.linspace(940, 960, 9) if quick else np.linspace(935, 965, 31)) * units.nm
    if quick:
        spec = spectrum(wavelengths, top_pairs=4, bottom_pairs=8, order=2, cell_fraction=0.1)
    else:
        spec = spectrum(wavelengths, top_pairs=6, bottom_pairs=12)
    print(f"{'lambda [nm]':>12} {'F_P':>8} {'beta_top':>9}")
    for lam, f, b in zip(spec.wavelength_nm, spec.purcell, spec.beta_top, strict=True):
        print(f"{lam:>12.2f} {f:>8.3f} {b:>9.3f}")
    with open("quantum_dot_purcell.json", "w", encoding="utf-8") as out:
        json.dump(asdict(spec) | {"mirror_check": {"fem": fem, "exact": exact}}, out, indent=2)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(spec.wavelength_nm, spec.purcell, "o-", label="Purcell factor")
        ax.plot(spec.wavelength_nm, spec.beta_top, "s-", label="beta (top)")
        ax.set_xlabel("wavelength [nm]")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.savefig("quantum_dot_purcell.png", dpi=120)
    except ImportError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

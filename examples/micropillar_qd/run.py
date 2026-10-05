"""Quantum dot in a micropillar cavity, computed as a body of revolution (2.5D).

A micropillar is a GaAs / AlAs distributed-Bragg-reflector cavity etched into a cylinder of
one to two micrometres diameter; a single InAs quantum dot at the antinode of the
one-wavelength cavity emits into the fundamental HE11-like mode. The structure is a body
of revolution, so the 3D problem reduces to 2D problems on the meridian plane, one per
azimuthal order m (`docs/theory/axisymmetric.md`, ADR-0010):

- the **resonance** (wavelength and Q of the fundamental mode) is the quasi-normal mode of
  order m = 1 found by `hpfem.AxisymmetricResonance` with the cylindrical PML;
- the **Purcell factor** F_P = P / P_bulk is the power the emitter radiates in the pillar
  divided by the power the same dipole radiates in bulk GaAs; the in-plane dipole on the
  axis has the orders m = +-1 (`hpfem.axisymmetric_gaussian_dipole`), equal by symmetry,
  so one solve per wavelength gives the spectrum; the bulk reference is the analytic
  Larmor power of the smeared dipole, n P0 exp(-(nk)^2 sigma^2);
- the **beta factor** is the fraction of the emitted power leaving upwards through a plane
  above the pillar (collection direction), from the axisymmetric Poynting flux.

The Purcell spectrum must peak at the resonance wavelength with a width given by Q; the
script prints both. Run `python examples/micropillar_qd/run.py [--quick]`; results go to
`micropillar_qd.json`, the spectrum to `micropillar_qd.png`.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass

import numpy as np

import hpfem
from hpfem import units

LAMBDA_DESIGN = 940 * units.nm  # InAs quantum dots in GaAs
N_HIGH, N_LOW, N_AIR = 3.53, 2.95, 1.0  # GaAs, AlAs near 940 nm (non-dispersive model)
TAG_HIGH, TAG_LOW, TAG_CAVITY, TAG_SUBSTRATE, TAG_AIR = 2, 3, 4, 5, 6
TAG_AXIS = 77


def pillar_mesh(top_pairs, bottom_pairs, radius, margin, pml, cell_fraction=0.1):
    """Structured meridian mesh (x = r, y = z, light leaves in +z) with nodes on every layer
    interface and on the pillar side wall: returns (mesh, cavity centre z, pillar height,
    outer radius, z of the collection plane)."""
    q_high = LAMBDA_DESIGN / (4 * N_HIGH)
    q_low = LAMBDA_DESIGN / (4 * N_LOW)
    layers = []
    for _ in range(bottom_pairs):
        layers += [(N_HIGH, q_high, TAG_HIGH), (N_LOW, q_low, TAG_LOW)]
    cavity_centre = sum(t for _, t, _ in layers) + LAMBDA_DESIGN / N_HIGH / 2
    layers.append((N_HIGH, LAMBDA_DESIGN / N_HIGH, TAG_CAVITY))
    for _ in range(top_pairs):
        layers += [(N_LOW, q_low, TAG_LOW), (N_HIGH, q_high, TAG_HIGH)]
    height = sum(t for _, t, _ in layers)
    cell = cell_fraction * LAMBDA_DESIGN

    def grid(intervals):
        nodes, tags = [intervals[0][0]], []
        for x0, x1, tag in intervals:
            count = max(2, int(np.ceil((x1 - x0) / cell)))
            count += count % 2  # even: a node at the middle of every interval
            for j in range(1, count + 1):
                nodes.append(x0 + (x1 - x0) * j / count)
                tags.append(tag)
        return np.array(nodes), tags

    z_intervals = [(-(margin + pml), -margin, TAG_SUBSTRATE), (-margin, 0.0, TAG_SUBSTRATE)]
    z = 0.0
    for _, t, tag in layers:
        z_intervals.append((z, z + t, tag))
        z += t
    z_intervals += [
        (height, height + margin, TAG_AIR),
        (height + margin, height + margin + pml, TAG_AIR),
    ]
    z_nodes, z_tags = grid(z_intervals)
    r_outer = radius + margin + pml
    r_nodes, _ = grid(
        [(0.0, radius, 0), (radius, radius + margin, 0), (radius + margin, r_outer, 0)]
    )
    nr, nz = len(r_nodes), len(z_nodes)
    vertices = np.array([[r, zz] for zz in z_nodes for r in r_nodes])
    cells, tags = [], []
    for j in range(nz - 1):
        for i in range(nr - 1):
            a, b, c, d = j * nr + i, j * nr + i + 1, (j + 1) * nr + i + 1, (j + 1) * nr + i
            cells += [[a, b, c], [a, c, d]]
            inside = 0.5 * (r_nodes[i] + r_nodes[i + 1]) < radius
            z_mid = 0.5 * (z_nodes[j] + z_nodes[j + 1])
            tag = z_tags[j] if (inside or z_mid < 0) else TAG_AIR
            tags += [tag, tag]
    mesh = hpfem.Mesh2D(vertices, np.array(cells), tags)
    tolerance = 1e-9 * r_outer
    for f in mesh.boundary_facets:
        v = mesh.vertices[mesh.facet_vertices(int(f))]
        if np.all(np.abs(v[:, 0]) < tolerance):
            mesh.set_facet_tag(int(f), TAG_AXIS)
    collection_plane = height + margin / 2  # a node line (even counts)
    return mesh, cavity_centre, height, r_outer, collection_plane


def materials():
    m = hpfem.MaterialMap()
    for tag, index in (
        (TAG_HIGH, N_HIGH),
        (TAG_CAVITY, N_HIGH),
        (TAG_SUBSTRATE, N_HIGH),
        (TAG_LOW, N_LOW),
    ):
        m.set(tag, hpfem.Material.dielectric(index))
    return m


def pml_box(radius, margin, pml, height, k0):
    return hpfem.PmlBox2D(
        [0.0, -margin], [radius + margin, height + margin], [0.0, pml, pml, pml], k0, 1.0,
        hpfem.PmlProfile(2, 1e-10),
    )  # fmt: skip


def plane_surface(mesh, z_plane, r_max):
    """Horizontal plane z = z_plane for r < r_max as a surface with the normal pointing up
    (inside cell below the plane)."""
    tolerance = 1e-9 * r_max
    facets = []
    for f in range(mesh.num_facets):
        v = mesh.vertices[mesh.facet_vertices(f)]
        if np.all(np.abs(v[:, 1] - z_plane) < tolerance) and np.all(v[:, 0] < r_max + tolerance):
            cells = [c for c in mesh.facet_cells(f) if c >= 0]
            below = min(cells, key=lambda c: mesh.cell_centroids[c][1])
            facets.append(hpfem.Surface2D.Facet(f, below))
    surface = hpfem.Surface2D()
    surface.facets = facets  # the property hands out a copy; assign the whole list
    return surface


def emitted_power(nd, h1, field, omega, materials, surface):
    """Power of the in-plane dipole through the surface: the orders m = +1 and -1 are equal."""
    return 2 * hpfem.axisymmetric_poynting_flux(
        nd, h1, field.meridian, field.azimuthal, 1, omega, materials, surface
    )


@dataclass
class Resonance:
    wavelength_nm: float
    quality: float
    num_dofs: int


@dataclass
class Spectrum:
    wavelength_nm: list[float]
    purcell: list[float]
    beta_top: list[float]


def resonance(top_pairs, bottom_pairs, radius, order=3, margin=None, pml=None) -> Resonance:
    """Fundamental mode (m = 1) nearest the design wavelength."""
    margin = margin or 0.75 * LAMBDA_DESIGN
    pml = pml or 1.5 * LAMBDA_DESIGN
    mesh, _, height, _, _ = pillar_mesh(top_pairs, bottom_pairs, radius, margin, pml)
    nd = hpfem.NedelecDofMap2D(mesh, order)
    h1 = hpfem.DofMap2D(mesh, order)
    k_design = 2 * np.pi / LAMBDA_DESIGN
    setup = hpfem.AxisymmetricResonanceSetup()
    setup.target_omega = k_design * hpfem.constants.c0
    setup.materials = materials()
    setup.axis_tag = TAG_AXIS
    setup.azimuthal_order = 1
    setup.pml = pml_box(radius, margin, pml, height, k_design)
    setup.num_modes = 4
    setup.krylov_dimension = 40
    modes = hpfem.AxisymmetricResonance(nd, h1, setup).solve()
    # the fundamental mode: the highest Q within 3 % of the design wavelength
    window = [m for m in modes if abs(m.wavelength - LAMBDA_DESIGN) < 0.03 * LAMBDA_DESIGN]
    best = max(window or modes, key=lambda m: m.quality)
    return Resonance(best.wavelength / units.nm, best.quality, nd.num_dofs + h1.num_dofs)


def spectrum(wavelengths, top_pairs, bottom_pairs, radius, order=3, sigma=20 * units.nm,
             margin=None, pml=None) -> Spectrum:  # fmt: skip
    """Purcell and beta factors of the in-plane dipole at the cavity antinode."""
    margin = margin or 0.75 * LAMBDA_DESIGN
    pml = pml or 1.5 * LAMBDA_DESIGN
    mesh, centre, height, r_outer, plane = pillar_mesh(top_pairs, bottom_pairs, radius, margin, pml)
    nd = hpfem.NedelecDofMap2D(mesh, order)
    h1 = hpfem.DofMap2D(mesh, order)
    around = hpfem.Surface2D.around_cells(mesh, TAG_CAVITY)
    top = plane_surface(mesh, plane, r_outer - pml)
    purcell, beta = [], []
    for lam in wavelengths:
        omega = units.angular_frequency(wavelength=lam)
        k0 = 2 * np.pi / lam
        setup = hpfem.AxisymmetricScatteringSetup()
        setup.omega = omega
        setup.materials = materials()
        setup.axis_tag = TAG_AXIS
        setup.azimuthal_order = 1
        setup.pml = pml_box(radius, margin, pml, height, k0)
        setup.current = hpfem.axisymmetric_gaussian_dipole(
            centre, 1.0, hpfem.AxisDipole.TRANSVERSE, sigma, omega, 1
        )
        setup.extra_quadrature_order = 6
        field = hpfem.AxisymmetricScattering(nd, h1, setup).solve()
        total = emitted_power(nd, h1, field, omega, setup.materials, around)
        bulk = (
            N_HIGH * hpfem.dipole_vacuum_power(1.0, omega) * np.exp(-((N_HIGH * k0 * sigma) ** 2))
        )
        purcell.append(total / bulk)
        beta.append(emitted_power(nd, h1, field, omega, setup.materials, top) / total)
    return Spectrum([lam / units.nm for lam in wavelengths], purcell, beta)


@dataclass
class ModalSpectrum:
    """Purcell factor from the Riesz projection: the total (modal sum plus background), the
    share of the fundamental mode and the background, at the same wavelengths."""

    wavelength_nm: list[float]
    purcell: list[float]
    purcell_mode: list[float]
    purcell_background: list[float]
    num_poles: int
    convergence: float  # largest half-rule difference over the contours


def modal_spectrum(wavelengths, top_pairs, bottom_pairs, radius, order=3, sigma=20 * units.nm,
                   margin=None, pml=None, points=(16, 48)) -> ModalSpectrum:  # fmt: skip
    """Purcell factor as a sum over the quasi-normal modes: the resonance pencil (PML frozen
    at the design wavelength) is expanded by Riesz projections around the modes found by
    the resonance solver; the emitted power at every wavelength then costs no further solve.
    The modal shares of the power add up to the total exactly (the power is linear in the
    field for a fixed current); the difference to `spectrum` is the per-wavelength PML of the
    sweep."""
    margin = margin or 0.75 * LAMBDA_DESIGN
    pml = pml or 1.5 * LAMBDA_DESIGN
    mesh, centre, height, _, _ = pillar_mesh(top_pairs, bottom_pairs, radius, margin, pml)
    nd = hpfem.NedelecDofMap2D(mesh, order)
    h1 = hpfem.DofMap2D(mesh, order)
    k_design = 2 * np.pi / LAMBDA_DESIGN
    setup = hpfem.AxisymmetricResonanceSetup()
    setup.target_omega = k_design * hpfem.constants.c0
    setup.materials = materials()
    setup.axis_tag = TAG_AXIS
    setup.azimuthal_order = 1
    setup.pml = pml_box(radius, margin, pml, height, k_design)
    setup.num_modes = 8  # every pole inside the background contour must be listed
    setup.krylov_dimension = 60
    problem = hpfem.AxisymmetricResonance(nd, h1, setup)
    modes = problem.solve()
    window = [m for m in modes if abs(m.wavelength - LAMBDA_DESIGN) < 0.03 * LAMBDA_DESIGN]
    fundamental = max(window or modes, key=lambda m: m.quality)
    omegas = [units.angular_frequency(wavelength=lam) for lam in wavelengths]
    riesz_setup = hpfem.RieszSetup()
    riesz_setup.poles = [m.omega for m in modes]
    riesz_setup.omega_min = min(omegas)
    riesz_setup.omega_max = max(omegas)
    riesz_setup.points_per_pole, riesz_setup.background_points = points
    riesz_setup.background_aspect = 0.5  # keeps the heavily damped PML poles outside
    riesz = hpfem.AxisymmetricRieszProjection(problem, riesz_setup)
    # the current density J of the in-plane dipole: the Gaussian source f = i omega mu0 J at
    # omega = 1 divided by i mu0 (the factor i omega mu0 is applied per contour point)
    f = hpfem.axisymmetric_gaussian_dipole(centre, 1.0, hpfem.AxisDipole.TRANSVERSE, sigma, 1.0, 1)
    source = riesz.add_current(lambda x: f(x) / (1j * hpfem.constants.mu0))
    power = riesz.add_emitted_power(source)
    riesz.run()
    contours = riesz.contours
    spectrum = riesz.spectrum(source, power, omegas)  # contours x wavelengths
    mode_row = next(i for i, c in enumerate(contours) if fundamental.omega in c.poles)
    purcell, purcell_mode, purcell_background = [], [], []
    for j, (lam, omega) in enumerate(zip(wavelengths, omegas, strict=True)):
        k0 = 2 * np.pi / lam
        bulk = (
            N_HIGH * hpfem.dipole_vacuum_power(1.0, omega) * np.exp(-((N_HIGH * k0 * sigma) ** 2))
        )
        # both orders m = +-1 radiate equally
        purcell.append(2 * spectrum[:, j].sum().real / bulk)
        purcell_mode.append(2 * spectrum[mode_row, j].real / bulk)
        purcell_background.append(2 * spectrum[-1, j].real / bulk)
    return ModalSpectrum(
        [lam / units.nm for lam in wavelengths], purcell, purcell_mode, purcell_background,
        len(contours[-1].poles), max(c.convergence for c in contours),
    )  # fmt: skip


def run(quick: bool = False) -> dict:
    top, bottom = (6, 10) if quick else (10, 16)
    radius = (0.75 if quick else 1.0) * units.um
    order = 2 if quick else 3
    res = resonance(top, bottom, radius, order=order)
    fwhm = res.wavelength_nm / res.quality
    span = 1.5 * fwhm  # +- 1.5 linewidths around the resonance
    count = 9 if quick else 25
    wavelengths = np.linspace(res.wavelength_nm - span, res.wavelength_nm + span, count) * units.nm
    spec = spectrum(wavelengths, top, bottom, radius, order=order)
    modal = modal_spectrum(wavelengths, top, bottom, radius, order=order,
                           points=(16, 40) if quick else (16, 48))  # fmt: skip
    return {
        "resonance": asdict(res),
        "spectrum": asdict(spec),
        "modal": asdict(modal),
        "top_pairs": top,
        "bottom_pairs": bottom,
        "radius_um": radius / units.um,
        "order": order,
    }


def main(argv=None) -> int:
    hpfem.set_log_level("warn")
    quick = "--quick" in (argv or sys.argv[1:])
    result = run(quick)
    res, spec = result["resonance"], result["spectrum"]
    print(
        f"GaAs / AlAs micropillar, radius {result['radius_um']:.2f} um, "
        f"{result['top_pairs']} top / "
        f"{result['bottom_pairs']} bottom pairs, p = {result['order']}: fundamental mode "
        f"lambda = {res['wavelength_nm']:.3f} nm, Q = {res['quality']:.0f} ({res['num_dofs']} DoF)"
    )
    modal = result["modal"]
    print(
        f"{'lambda [nm]':>12} {'F_P':>8} {'beta_top':>9} {'F_P modal':>10} {'mode':>8} "
        f"{'backgr.':>8}   (Riesz projection: {modal['num_poles']} poles, half-rule "
        f"difference {modal['convergence']:.1e})"
    )
    for lam, f, b, fm, fmode, fbg in zip(
        spec["wavelength_nm"], spec["purcell"], spec["beta_top"], modal["purcell"],
        modal["purcell_mode"], modal["purcell_background"], strict=True,
    ):  # fmt: skip
        print(f"{lam:>12.3f} {f:>8.2f} {b:>9.3f} {fm:>10.2f} {fmode:>8.2f} {fbg:>8.2f}")
    with open("micropillar_qd.json", "w", encoding="utf-8") as out:
        json.dump(result, out, indent=2)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(spec["wavelength_nm"], spec["purcell"], "o-", label="Purcell factor")
        ax.axvline(res["wavelength_nm"], color="0.6", linewidth=0.8, label="resonance (m = 1)")
        ax.set_xlabel("wavelength [nm]")
        ax.set_ylabel("F_P")
        ax2 = ax.twinx()
        ax2.plot(spec["wavelength_nm"], spec["beta_top"], "s--", color="C1", label="beta (top)")
        ax2.set_ylabel("beta")
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig("micropillar_qd.png", dpi=120)
    except ImportError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

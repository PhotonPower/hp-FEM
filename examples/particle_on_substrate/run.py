"""Bodies of revolution on layer stacks (2.5D, milestone M18, ADR-0014).

Three studies of a plane wave on a stack normal to the axis with a deviation of revolution:

- **dark field**: a gold sphere (radius 40 nm) on glass illuminated from the air at 60 degrees;
  absorption, scattering into the air, the glass and along the interface, extinction, and the
  power an objective of NA 0.5 above the sample collects (the specular reflection lies outside
  its cone: dark field);
- **nanoparticle on a mirror (NPoM)**: the same kind of sphere (radius 30 nm) on a 1 nm spacer
  (n = 1.45) over a gold film, 1 nm of air under the sphere, p-polarised at 55 degrees (the
  vertical gap mode needs E_z); its scattering peak lies far to the red of the isolated sphere;
- **nanohole**: a hole of 200 nm diameter through a 100 nm gold film on glass at normal
  incidence; the power the hole adds below the film, normalised by the power falling on the
  hole area (T / T_geom), and the power balance of the absorption change.

The incident field of order m is the stack's plane wave (`hpfem.layered_axisymmetric_wave`), the
scattered-field source lives in the cells deviating from the stack (`setup.background`), the
channels come from `axisymmetric_absorbed_power`, `axisymmetric_flux_channels`,
`axisymmetric_layered_far_field` and `axisymmetric_disc_flux` (docs/theory/axisymmetric.md).
Planes of incidence are mirror planes, so the orders +-m carry the same powers: m >= 0 is
solved and the m > 0 terms are doubled.

Run `python examples/particle_on_substrate/run.py [--quick] [--study darkfield|npom|hole]`;
results go to `particle_on_substrate.json`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import hpfem
from hpfem import materials, units

TAG_AIR, TAG_GLASS, TAG_FILM, TAG_SPACER, TAG_PARTICLE, TAG_HOLE = 1, 2, 3, 4, 5, 6
TAG_AXIS = 77
TAG_INSIDE = 99  # cells inside the measurement surface (on a copy of the mesh)
N_GLASS, N_SPACER = 1.5, 1.45
Z0, C0 = hpfem.constants.Z0, hpfem.constants.c0
INTENSITY = 1.0 / (2 * Z0)  # |E0| = 1 in air
GOLD = materials.get("Au")  # Johnson & Christy 1972


# --- meshes --------------------------------------------------------------------------------------


def march(start, stop, size, breaks=()):
    """Nodes from start to stop (either direction) with the local cell size size(x), every value
    of `breaks` between them a node."""
    sign = 1.0 if stop > start else -1.0
    stops = sorted({b for b in breaks if (b - start) * sign > 0 and (stop - b) * sign > 0} | {stop},
                   key=lambda b: (b - start) * sign)  # fmt: skip
    nodes, x = [start], start
    for target in stops:
        while (target - x) * sign > 1e-12 * abs(stop - start):
            h = size(x)
            nxt = x + sign * h
            if (target - nxt) * sign < 0.3 * h:
                nxt = target
            nodes.append(nxt)
            x = nxt
    return np.array(nodes)


def merged(*parts, tolerance):
    """Sorted union of node arrays, nodes closer than `tolerance` merged."""
    nodes = np.sort(np.concatenate(parts))
    keep = np.concatenate([[True], np.diff(nodes) > tolerance])
    return nodes[keep]


def build_mesh(r_nodes, z_nodes, tag_of, block=None):
    """Meridian mesh (x = r, y = z): the tensor grid of r_nodes x z_nodes, two triangles per
    rectangle, tagged by tag_of(r, z) at the centroid; `block` = (vertices, cells, tags, h,
    z_low, z_high) replaces the grid in [0, h] x [z_low, z_high] (the sphere). Axis facets carry
    TAG_AXIS; edges on a circle given as block[6] = (centre z, radius) are curved."""
    scale = min(np.min(np.diff(r_nodes)), np.min(np.abs(np.diff(z_nodes))))
    index, vertices = {}, []

    def vertex(x, z):
        key = (round(x / (1e-6 * scale)), round(z / (1e-6 * scale)))
        if key not in index:
            index[key] = len(vertices)
            vertices.append((x, z))
        return index[key]

    z_nodes = np.sort(z_nodes)
    cells, tags = [], []
    for i in range(len(r_nodes) - 1):
        for j in range(len(z_nodes) - 1):
            x0, x1, z0, z1 = r_nodes[i], r_nodes[i + 1], z_nodes[j], z_nodes[j + 1]
            if block is not None and x1 <= block[3] + 1e-9 * scale and \
                    z0 >= block[4] - 1e-9 * scale and z1 <= block[5] + 1e-9 * scale:  # fmt: skip
                continue
            a, b, c, d = vertex(x0, z0), vertex(x1, z0), vertex(x1, z1), vertex(x0, z1)
            for tri in ((a, b, c), (a, c, d)):
                cells.append(tri)
                xs = np.mean([vertices[k] for k in tri], axis=0)
                tags.append(tag_of(xs[0], xs[1]))
    if block is not None:
        b_vertices, b_cells, b_tags = block[0], block[1], block[2]
        mapped = [vertex(float(x), float(z)) for x, z in b_vertices]
        for tri, tag in zip(b_cells, b_tags, strict=True):
            cells.append(tuple(mapped[int(k)] for k in tri))
            xs = np.mean([b_vertices[int(k)] for k in tri], axis=0)
            tags.append(TAG_PARTICLE if tag == TAG_PARTICLE else tag_of(xs[0], xs[1]))
    mesh = hpfem.Mesh2D(np.array(vertices), np.array(cells), tags)
    for f in mesh.boundary_facets:
        v0, v1 = (mesh.vertex(int(v)) for v in mesh.facet_vertices(int(f)))
        if abs(v0[0]) < 1e-9 * scale and abs(v1[0]) < 1e-9 * scale:
            mesh.set_facet_tag(int(f), TAG_AXIS)
    if block is not None:
        zc, radius = block[6]
        nodes = np.zeros((mesh.num_edges, 2))
        for e in range(mesh.num_edges):
            p0, p1 = (np.asarray(mesh.vertex(int(v))) for v in mesh.edge_vertices(e))
            mid = 0.5 * (p0 + p1)
            on_circle = [
                abs(math.hypot(p[0], p[1] - zc) - radius) < 1e-6 * radius for p in (p0, p1)
            ]
            if all(on_circle):
                d = mid - np.array([0.0, zc])
                mid = np.array([0.0, zc]) + radius * d / np.linalg.norm(d)
            nodes[e] = mid
        mesh.set_edge_nodes(nodes)
    return mesh


def sphere_block(radius, cells_per_radius, zc):
    """The half disc of the sphere in the square [0, h] x [zc - h, zc + h], h = radius (1 + 1/n)
    (one ring of cells between the circle and the square)."""
    n = int(cells_per_radius)
    h = radius * (n + 1) / n
    full = hpfem.square_with_disc(n, radius, h, h, TAG_PARTICLE)
    half = hpfem.extract(full, lambda c: c[0] > 0)
    vertices = np.asarray(half.vertices) + np.array([0.0, zc])
    return (vertices, np.asarray(half.cells), list(half.cell_tags), h, zc - h, zc + h,
            (zc, radius))  # fmt: skip


def particle_mesh(radius, cells_per_radius, interfaces, cfg):
    """A sphere whose bounding square sits on the top interface z = 0, the layers below, PML
    outside [0, r_in] x [z_lo, z_hi]. Returns (mesh, geometry dict)."""
    n = int(cells_per_radius)
    h = radius * (n + 1) / n
    zc = h
    fine = radius / n
    block = sphere_block(radius, n, zc)
    r_in, depth, height, pml, coarse = (
        cfg["r_in"],
        cfg["depth"],
        cfg["height"],
        cfg["pml"],
        cfg["coarse"],
    )
    growth = cfg["growth"]
    r_nodes = merged(np.arange(n + 2) * fine,
                     march(h, r_in + pml, lambda x: min(coarse, fine + growth * (x - h)), [r_in]),
                     tolerance=1e-6 * fine)  # fmt: skip
    z_bottom = min(interfaces) - depth
    z_inner = np.arange(-(n + 1), n + 2) * fine + zc
    layer_cell = cfg.get("layer_cell", coarse)

    def size_below(z):
        s = min(coarse, fine + growth * (0.0 - z))
        if len(interfaces) > 1 and min(interfaces) < z < max(interfaces):
            s = min(s, layer_cell)
        return max(s, 0.25 * fine)

    below = march(0.0, z_bottom - pml, size_below, list(interfaces) + [z_bottom])
    above = march(
        zc + h,
        zc + h + height + pml,
        lambda z: min(coarse, fine + growth * (z - zc - h)),
        [zc + h + height],
    )
    z_nodes = merged(below, z_inner, above, tolerance=1e-6 * fine)

    def tag_of(r, z):
        return layer_tag(z, interfaces)

    mesh = build_mesh(r_nodes, z_nodes, tag_of, block)
    geometry = {"zc": zc, "h": h, "r_in": r_in, "z_lo": z_bottom, "z_hi": zc + h + height,
                "pml": pml, "radius": radius, "r_nodes": r_nodes, "z_nodes": z_nodes}  # fmt: skip
    return mesh, geometry


def layer_tag(z, interfaces):
    """Tag of the stack region at z: air above the top interface, then the layers of the study."""
    interfaces = sorted(interfaces, reverse=True)
    if z > interfaces[0]:
        return TAG_AIR
    if len(interfaces) == 1:
        return TAG_GLASS
    # NPoM: spacer, gold film, glass
    if z > interfaces[1]:
        return TAG_SPACER
    if z > interfaces[2]:
        return TAG_FILM
    return TAG_GLASS


def surface_inside(mesh, inside):
    """Closed surface of mesh lines around the cells whose centroid satisfies inside(r, z)."""
    marked = mesh.copy()
    for c in range(mesh.num_cells):
        r, z = mesh.cell_centroid(c)
        marked.set_cell_tag(c, TAG_INSIDE if inside(r, z) else 1)
    return hpfem.Surface2D.around_cells(marked, TAG_INSIDE)


# --- the particle studies ------------------------------------------------------------------------


def pml_box(geometry, k0):
    return hpfem.PmlBox2D([0.0, geometry["z_lo"]], [geometry["r_in"], geometry["z_hi"]],
                          [0.0, geometry["pml"], geometry["pml"], geometry["pml"]], k0)  # fmt: skip


def particle_spectrum(mesh, geometry, stack_at, materials_at, wavelengths, theta, pols, p,
                      inside, aperture, max_order=8, tolerance=1e-4):  # fmt: skip
    """Cross-sections per wavelength (averaged over the polarisations): absorption of the
    particle, scattering up / down / lateral through the surface around the cells inside(r, z),
    extinction, the power collected by an objective of numerical aperture `aperture` above, and
    the power balance of the total field in that region (inflow against absorption)."""
    nd = hpfem.NedelecDofMap2D(mesh, p)
    h1 = hpfem.DofMap2D(mesh, p)
    particle = np.array([mesh.cell_tag(c) == TAG_PARTICLE for c in range(mesh.num_cells)])
    region = np.array([inside(*mesh.cell_centroid(c)) for c in range(mesh.num_cells)])
    surface = surface_inside(mesh, inside)
    cone = np.linspace(0.0, math.asin(aperture), 61)
    rows = []
    for wavelength in wavelengths:
        omega = 2 * math.pi * C0 / wavelength
        k0 = omega / C0
        stack = stack_at(omega)
        totals = dict(absorption=0.0, up=0.0, down=0.0, lateral=0.0, collected=0.0)
        inflow = absorbed_region = 0.0
        orders_used = 0
        for pol in pols:
            setup = hpfem.AxisymmetricScatteringSetup()
            setup.omega = omega
            for tag, material in materials_at(omega).items():
                setup.materials.set(tag, material)
            setup.axis_tag = TAG_AXIS
            setup.pml = pml_box(geometry, k0)
            setup.background = stack
            total_sca = 0.0
            for m in range(max_order + 1):
                wave = hpfem.layered_axisymmetric_wave(stack, k0, theta, pol, m)
                setup.azimuthal_order = m
                setup.incident = wave.value
                problem = hpfem.AxisymmetricScattering(nd, h1, setup)
                field = problem.solve()
                weight = (1.0 if m == 0 else 2.0) / len(pols)  # +-m equal by mirror symmetry
                per_cell = np.asarray(problem.absorbed_power(field).per_cell)
                absorbed = per_cell[particle].sum()
                absorbed_region += weight * per_cell[region].sum()
                inflow -= weight * hpfem.axisymmetric_poynting_flux(
                    nd, h1, field.meridian, field.azimuthal, m, omega, setup.materials, surface, 8,
                    wave.value, wave.curl,
                )  # fmt: skip
                channels = hpfem.axisymmetric_flux_channels(
                    nd, h1, field.meridian, field.azimuthal, m, omega, setup.materials, surface,
                    stack,
                )  # fmt: skip
                far = hpfem.axisymmetric_layered_far_field(
                    nd, h1, field.meridian, field.azimuthal, m, omega, setup.materials, surface,
                    stack, cone, [],
                )  # fmt: skip
                totals["absorption"] += weight * absorbed
                totals["up"] += weight * channels.up
                totals["down"] += weight * channels.down
                totals["lateral"] += weight * channels.lateral
                totals["collected"] += weight * far.up.radiated_power()
                pair = channels.total() * (1.0 if m == 0 else 2.0)
                total_sca += pair
                orders_used = max(orders_used, m)
                if m >= 2 and abs(pair) <= tolerance * abs(total_sca):
                    break
        sigma = {k: v / INTENSITY for k, v in totals.items()}
        sigma["scattering"] = sigma["up"] + sigma["down"] + sigma["lateral"]
        sigma["extinction"] = sigma["scattering"] + sigma["absorption"]
        rows.append({"wavelength_nm": wavelength / units.nm, "max_order": orders_used,
                     **{f"sigma_{k}_nm2": v / units.nm**2 for k, v in sigma.items()},
                     "balance": (inflow - absorbed_region) / absorbed_region})  # fmt: skip
        print(f"  {wavelength / units.nm:6.1f} nm: abs {rows[-1]['sigma_absorption_nm2']:9.1f} "
              f"sca {rows[-1]['sigma_scattering_nm2']:9.1f} (up {rows[-1]['sigma_up_nm2']:8.1f}, "
              f"down {rows[-1]['sigma_down_nm2']:8.1f}, lat {rows[-1]['sigma_lateral_nm2']:7.1f}) "
              f"collected {rows[-1]['sigma_collected_nm2']:8.1f} nm^2, |m| <= {orders_used}, "
              f"balance {rows[-1]['balance']:+.1e}",
              flush=True)  # fmt: skip
    return rows


def darkfield(quick):
    """Gold sphere (radius 40 nm) on glass, 2.5 nm of air under it, from the air at 60 deg."""
    radius = 40 * units.nm
    n = 8 if quick else 16
    cfg = {"r_in": 400 * units.nm, "depth": 250 * units.nm, "height": 250 * units.nm,
           "pml": 250 * units.nm, "coarse": (50 if quick else 30) * units.nm,
           "growth": 0.35}  # fmt: skip
    mesh, geometry = particle_mesh(radius, n, [0.0], cfg)

    def inside(r, z):
        return r < 150 * units.nm and -100 * units.nm < z < geometry["zc"] + 110 * units.nm

    glass = hpfem.Material.dielectric(N_GLASS)

    def stack_at(omega):
        return hpfem.LayerStack3D(hpfem.Material.vacuum(), [], glass, 0.0)

    def materials_at(omega):
        return {TAG_GLASS: glass, TAG_PARTICLE: GOLD.at(omega)}

    wavelengths = (
        np.array([480.0, 540.0, 600.0]) if quick else np.arange(450.0, 701.0, 10.0)
    ) * units.nm
    print(f"dark field: {mesh.num_cells} cells, gap {(geometry['h'] - radius) / units.nm:.1f} nm")
    rows = particle_spectrum(mesh, geometry, stack_at, materials_at, wavelengths,
                             math.radians(60.0), ("s", "p"), 2 if quick else 3, inside, 0.5,
                             2 if quick else 8)  # fmt: skip
    return {"radius_nm": 40, "gap_nm": (geometry["h"] - radius) / units.nm, "theta_deg": 60,
            "aperture": 0.5, "cells": mesh.num_cells, "spectrum": rows}  # fmt: skip


def npom(quick):
    """Gold sphere (radius 30 nm) on 1 nm spacer + 1 nm air over a 100 nm gold film on glass,
    p at 55 deg, against the same sphere in air (no stack)."""
    radius = 30 * units.nm
    # one cell between the sphere and its bounding square: 1 nm of air (2 nm in the quick run)
    n = 15 if quick else 30
    spacer, film = 1 * units.nm, 100 * units.nm
    interfaces = [0.0, -spacer, -spacer - film]
    if quick:
        cfg = {"r_in": 250 * units.nm, "depth": 150 * units.nm, "height": 200 * units.nm,
               "pml": 200 * units.nm, "coarse": 60 * units.nm, "growth": 0.5,
               "layer_cell": 20 * units.nm}  # fmt: skip
    else:
        cfg = {"r_in": 350 * units.nm, "depth": 200 * units.nm, "height": 250 * units.nm,
               "pml": 250 * units.nm, "coarse": 30 * units.nm, "growth": 0.35,
               "layer_cell": 10 * units.nm}  # fmt: skip
    mesh, geometry = particle_mesh(radius, n, interfaces, cfg)

    def inside(r, z):
        return r < 120 * units.nm and -60 * units.nm < z < geometry["zc"] + 100 * units.nm

    glass = hpfem.Material.dielectric(N_GLASS)
    spacer_material = hpfem.Material.dielectric(N_SPACER)

    def stack_at(omega):
        layers = [hpfem.Layer(spacer_material, spacer), hpfem.Layer(GOLD.at(omega), film)]
        return hpfem.LayerStack3D(hpfem.Material.vacuum(), layers, glass, 0.0)

    def materials_at(omega):
        return {TAG_GLASS: glass, TAG_SPACER: spacer_material, TAG_FILM: GOLD.at(omega),
                TAG_PARTICLE: GOLD.at(omega)}  # fmt: skip

    def free_stack(omega):
        return hpfem.LayerStack3D(hpfem.Material.vacuum(), [], hpfem.Material.vacuum(), 0.0)

    def free_materials(omega):
        vacuum = hpfem.Material.vacuum()
        return {
            TAG_GLASS: vacuum,
            TAG_SPACER: vacuum,
            TAG_FILM: vacuum,
            TAG_PARTICLE: GOLD.at(omega),
        }

    wavelengths = (
        np.array([520.0, 600.0, 680.0]) if quick else np.arange(480.0, 901.0, 10.0)
    ) * units.nm
    p = 2 if quick else 3
    max_order = 1 if quick else 8  # m = 0 (E_z, the gap mode) and +-1
    print(f"NPoM: {mesh.num_cells} cells, gap {spacer / units.nm:.0f} nm spacer + "
          f"{(geometry['h'] - radius) / units.nm:.0f} nm air")  # fmt: skip
    theta = math.radians(55.0)
    on_mirror = particle_spectrum(mesh, geometry, stack_at, materials_at, wavelengths, theta,
                                  ("p",), p, inside, 0.5, max_order)  # fmt: skip
    print("isolated sphere (air):")
    isolated = particle_spectrum(mesh, geometry, free_stack, free_materials, wavelengths, theta,
                                 ("p",), p, inside, 0.5, max_order)  # fmt: skip

    def peak(rows):  # the dark-field observable: the scattering peak
        values = [r["sigma_scattering_nm2"] for r in rows]
        return rows[int(np.argmax(values))]["wavelength_nm"]

    return {"radius_nm": 30, "spacer_nm": 1, "air_gap_nm": (geometry["h"] - radius) / units.nm,
            "max_order": max_order,
            "film_nm": 100, "theta_deg": 55, "cells": mesh.num_cells, "on_mirror": on_mirror,
            "isolated": isolated, "peak_on_mirror_nm": peak(on_mirror),
            "peak_isolated_nm": peak(isolated)}  # fmt: skip


# --- the nanohole --------------------------------------------------------------------------------


def nanohole(quick, wavelengths=None):
    """A hole of radius 100 nm through a 100 nm gold film on glass, normal incidence: the power
    the hole adds below the film over the power on the hole area, and the power balance;
    `wavelengths` [m] replaces the sweep of the configuration."""
    r_hole, film = 100 * units.nm, 100 * units.nm
    fine = (10 if quick else 5) * units.nm
    coarse = (60 if quick else 40) * units.nm
    r_in, pml, depth, height = 900 * units.nm, 400 * units.nm, 400 * units.nm, 400 * units.nm
    growth = 0.3
    r_disc = 300 * units.nm
    z_disc = -film - 100 * units.nm
    r_nodes = march(0.0, r_in + pml, lambda x: min(coarse, fine + growth * abs(x - r_hole)),
                    [r_hole, r_disc, r_in])  # fmt: skip

    def size_z(z):
        s = min(coarse, fine + growth * min(abs(z), abs(z + film)))
        return min(s, 20 * units.nm) if -film < z < 0 else s

    z_nodes = merged(march(0.0, height + pml, size_z, [height]),
                     march(0.0, -film - depth - pml, size_z, [-film, z_disc, -film - depth]),
                     tolerance=1e-6 * fine)  # fmt: skip

    def tag_of(r, z):
        if z > 0:
            return TAG_AIR
        if z > -film:
            return TAG_HOLE if r < r_hole else TAG_FILM
        return TAG_GLASS

    mesh = build_mesh(r_nodes, z_nodes, tag_of)
    p = 2 if quick else 3
    nd = hpfem.NedelecDofMap2D(mesh, p)
    h1 = hpfem.DofMap2D(mesh, p)
    inside = lambda r, z: r < 250 * units.nm and -film - 60 * units.nm < z < 60 * units.nm  # noqa: E731
    surface = surface_inside(mesh, inside)
    region = np.array([inside(*mesh.cell_centroid(c)) for c in range(mesh.num_cells)])
    glass = hpfem.Material.dielectric(N_GLASS)
    sweep = (
        np.array([600.0, 750.0, 900.0]) if quick else np.arange(600.0, 1001.0, 10.0)
    ) * units.nm
    wavelengths = sweep if wavelengths is None else np.asarray(wavelengths, dtype=float)
    zero_e, zero_v = np.zeros(nd.num_dofs, complex), np.zeros(h1.num_dofs, complex)
    rows = []
    print(f"nanohole: {mesh.num_cells} cells")
    for wavelength in wavelengths:
        omega = 2 * math.pi * C0 / wavelength
        k0 = omega / C0
        gold = GOLD.at(omega)
        stack = hpfem.LayerStack3D(hpfem.Material.vacuum(), [hpfem.Layer(gold, film)], glass, 0.0)
        wave = hpfem.layered_axisymmetric_wave(stack, k0, 0.0, "p", 1)
        setup = hpfem.AxisymmetricScatteringSetup()
        setup.omega = omega
        setup.materials.set(TAG_GLASS, glass)
        setup.materials.set(TAG_FILM, gold)
        setup.materials.set(TAG_HOLE, hpfem.Material.vacuum())
        setup.axis_tag = TAG_AXIS
        setup.azimuthal_order = 1
        setup.pml = hpfem.PmlBox2D([0.0, -film - depth], [r_in, height], [0.0, pml, pml, pml], k0)
        setup.background = stack
        setup.incident = wave.value
        problem = hpfem.AxisymmetricScattering(nd, h1, setup)
        field = problem.solve()
        disc = hpfem.axisymmetric_disc_flux(nd, h1, field.meridian, field.azimuthal, 1, omega,
                                            setup.materials, z_disc, r_disc, -1, wave.value,
                                            wave.curl)  # fmt: skip
        absorbed = np.asarray(problem.absorbed_power(field).per_cell)[region].sum()
        absorbed_stack = np.asarray(problem.incident_absorbed_power().per_cell)[region].sum()
        inflow = -hpfem.axisymmetric_poynting_flux(nd, h1, field.meridian, field.azimuthal, 1,
                                                   omega, setup.materials, surface, 8, wave.value,
                                                   wave.curl)  # fmt: skip
        inflow_stack = -hpfem.axisymmetric_poynting_flux(nd, h1, zero_e, zero_v, 1, omega,
                                                         setup.materials, surface, 8, wave.value,
                                                         wave.curl)  # fmt: skip
        on_hole = INTENSITY * math.pi * r_hole**2
        change = absorbed - absorbed_stack
        # the orders +-1 carry the same powers: factors 2 cancel in the ratios
        rows.append({
            "wavelength_nm": wavelength / units.nm,
            "T_over_T_geom": 2 * disc.change() / on_hole,
            "film_transmittance": wave.transmittance,
            "absorption_change_over_geom": 2 * change / on_hole,
            "balance": (inflow - inflow_stack - change) / abs(change),
        })  # fmt: skip
        print(f"  {wavelength / units.nm:6.1f} nm: T/T_geom {rows[-1]['T_over_T_geom']:.4f}, "
              f"film T {wave.transmittance:.2e}, dA/A_geom "
              f"{rows[-1]['absorption_change_over_geom']:+.4f}, balance {rows[-1]['balance']:+.1e}",
              flush=True)  # fmt: skip
    return {"hole_diameter_nm": 200, "film_nm": 100, "cells": mesh.num_cells, "spectrum": rows}


# --- driver --------------------------------------------------------------------------------------


def run(quick: bool = False, studies=("darkfield", "npom", "hole"), out: str | None = None) -> dict:
    hpfem.set_log_level("warn")
    result = {"quick": bool(quick)}
    if "darkfield" in studies:
        result["darkfield"] = darkfield(quick)
    if "npom" in studies:
        result["npom"] = npom(quick)
    if "hole" in studies:
        result["hole"] = nanohole(quick)
    if out:
        Path(out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--quick", action="store_true", help="coarse meshes, p = 2, few wavelengths"
    )
    parser.add_argument("--study", choices=("darkfield", "npom", "hole"), action="append")
    parser.add_argument("--out", default="particle_on_substrate.json")
    args = parser.parse_args(argv)
    run(args.quick, tuple(args.study or ("darkfield", "npom", "hole")), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

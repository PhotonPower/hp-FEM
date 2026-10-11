"""Long local runs of the M17 S3 array scanning (ADR-0013 §3, §3a): a single Gaussian dipole in
a structure periodic in x, by ``hpfem.grating.dipole_emission``.

Cases (λ = 1 µm, σ = 60 nm, period 1 µm unless stated, PML 1 µm above and below):

- ``homogeneous``: a dipole in glass (n = 1.5). Purcell factor 1 and half of the power up and
  half down for every orientation.
- ``glass``: a dipole in air 400 nm above glass. The Purcell factors and the upward fractions
  against the Sommerfeld integral of a point dipole above a half-space (Chance, Prock and
  Silbey; the Gaussian factor e^{-(nkσ)²} cancels in the normalisation to the bulk power
  because every plane wave of the spectrum carries it).
- ``glass_period``: the same with a period of 0.8 µm. The single dipole does not depend on the
  period of the computational cell.
- ``slab``: a dipole in air 250 nm above a lossless Si₃N₄ slab (n = 2.0, 250 nm) on glass
  (σ = 40 nm), against the layered Sommerfeld integral; guided-mode poles (ADR-0013 §4a).
- ``mirror_150``, ``mirror_300``, ``mirror_600``: a dipole in air 150, 300 and 600 nm above a
  perfect mirror (σ = 40 nm, 20 nm at 150 nm), against the image-dipole closed form. The cell
  problems end on a PEC wall at the mirror; the plane waves of the channels see it as the
  substrate of the stack, a conductor with ε = −10¹⁰ (r_s = −1, r_p = +1 to 10⁻⁵). All power
  goes up.

Every case runs for ``--nodes`` β points per panel (default 6 and 8) with ``--kx-nodes``
(default 4) at ``--order`` (default 3). The record
``benchmarks/results/<date>-validation-m17-scan.json`` is rewritten after every run and keyed
by case/order/nodes, so a rerun skips what is done. Runtime about 15 min (nodes 4) to 40 min
(nodes 6) per case on 16 cores.

    python benchmarks/m17_dipole_scan.py [--cases homogeneous glass] [--nodes 4 6] [--order 3]
"""

import argparse
import cmath
import datetime
import json
import math
import os
import sys
import time

import numpy as np
from scipy.integrate import quad

import hpfem
from hpfem import grating, units

UM = 1e-6
LAMBDA = 1.0 * UM
SIGMA = 0.06 * UM


def fresnel(s, n1, n2):
    """r_s and r_p from medium n1 into n2 at the in-plane wavenumber s·n1·k0 (exp(−iωt);
    r_p = +1, r_s = −1 for a perfect conductor)."""
    kz1 = cmath.sqrt(1 - s * s + 0j)
    kz2 = cmath.sqrt((n2 / n1) ** 2 - s * s + 0j)
    if kz2.imag < 0:
        kz2 = -kz2
    e1, e2 = n1**2, n2**2
    return (kz1 - kz2) / (kz1 + kz2), (e2 * kz1 - e1 * kz2) / (e2 * kz1 + e1 * kz2)


def sommerfeld(n1, n2, k0, h):
    """Purcell factors (perpendicular, parallel) of a point dipole in n1 at height h above a
    half-space n2 and the fractions of the bulk power radiated upwards (Chance–Prock–Silbey):
    F_⊥ = 1 + (3/2) Re ∫ s³/s_z r_p e^{2ik1h s_z} ds, F_∥ = 1 + (3/4) Re ∫ s/s_z (r_s − s_z² r_p)
    e^{2ik1h s_z} ds, s = sin t on [0, 1) and cosh u beyond."""
    k1 = n1 * k0

    def integral(which):
        def inner(t):
            s, sz = math.sin(t), math.cos(t)
            rs, rp = fresnel(s, n1, n2)
            e = cmath.exp(2j * k1 * h * sz)
            v = s**3 * rp * e if which == 0 else s * (rs - sz**2 * rp) * e
            return v.real

        def outer(u):
            s, sz = math.cosh(u), 1j * math.sinh(u)
            rs, rp = fresnel(s, n1, n2)
            e = cmath.exp(2j * k1 * h * sz)
            v = (s**3 * rp * e if which == 0 else s * (rs - sz**2 * rp) * e) / 1j
            return v.real

        u_hi = math.asinh(40 / (2 * k1 * h))
        brk = [math.acosh(n2 / n1)] if n2 > n1 else None
        return (quad(inner, 0, math.pi / 2, limit=400)[0]
                + quad(outer, 0, max(u_hi, 1.5 * brk[0] if brk else 0.0), points=brk,
                       limit=400)[0])  # fmt: skip

    def up(which):
        def f(t):
            s, sz = math.sin(t), math.cos(t)
            rs, rp = fresnel(s, n1, n2)
            e = cmath.exp(2j * k1 * h * sz)
            if which == 0:
                return 0.75 * s**3 * abs(1 + rp * e) ** 2
            return 0.375 * s * (abs(1 + rs * e) ** 2 + sz**2 * abs(1 - rp * e) ** 2)

        return quad(f, 0, math.pi / 2, limit=400)[0]

    return 1 + 1.5 * integral(0), 1 + 0.75 * integral(1), up(0), up(1)


def airy(s, n1, layers, n3):
    """r_s, r_p seen from the cover n1 of the layers [(n, d), ...] on the substrate n3, at the
    in-plane wavenumber s·n1·k0 (s complex on a deformed path; r_p = +1 for a perfect
    conductor). Recursive Airy formula from the substrate up."""
    k0 = 2 * math.pi / LAMBDA
    kt2 = (s * n1) ** 2

    def kz(n):
        z = cmath.sqrt(n * n - kt2)
        return -z if z.real + z.imag < 0 else z

    media = [n1] + [n for n, _d in layers] + [n3]
    out = []
    for pol in ("s", "p"):
        r = 0.0
        for i in range(len(media) - 2, -1, -1):  # interface i between media i and i+1
            na, nb = media[i], media[i + 1]
            ka, kb = kz(na), kz(nb)
            if pol == "s":
                rab = (ka - kb) / (ka + kb)
            else:
                rab = (nb * nb * ka - na * na * kb) / (nb * nb * ka + na * na * kb)
            if i == len(media) - 2:
                r = rab
            else:
                ph = cmath.exp(2j * k0 * kb * layers[i][1])
                r = (rab + r * ph) / (1 + rab * r * ph)
        out.append(r)
    return out


def sommerfeld_layered(n1, layers, n3, k0, h, depth=0.02, points=4000):
    """Purcell factors and upward fractions of a point dipole in n1 at height h above a layer
    stack, as :func:`sommerfeld`, with the guided-mode poles of lossless layers passed on the
    causal side: s runs on t − i·depth·sin(πt/S) up to S beyond the largest index (exp(−iωt):
    a small loss moves the poles above the real axis), then along the real axis."""
    k1 = n1 * k0
    s_top = max(n for n, _d in layers) / n1 + 0.3
    t, w = np.polynomial.legendre.leggauss(points)
    t = 0.5 * s_top * (t + 1.0)
    w = 0.5 * s_top * w
    s = t - 1j * depth * np.sin(np.pi * t / s_top)
    ds = 1.0 - 1j * depth * (np.pi / s_top) * np.cos(np.pi * t / s_top)
    perp = par = 0.0
    for si, dsi, wi in zip(s, ds, w, strict=True):
        sz = cmath.sqrt(1 - si * si)
        if sz.real + sz.imag < 0:
            sz = -sz
        rs, rp = airy(si, n1, layers, n3)
        e = cmath.exp(2j * k1 * h * sz)
        perp += wi * dsi * si**3 / sz * rp * e
        par += wi * dsi * si / sz * (rs - sz * sz * rp) * e

    def tail(which):
        def f(u):
            sv = s_top + u
            sz = 1j * math.sqrt(sv * sv - 1)
            rs, rp = airy(sv, n1, layers, n3)
            e = cmath.exp(2j * k1 * h * sz)
            v = sv**3 / sz * rp * e if which == 0 else sv / sz * (rs - sz * sz * rp) * e
            return v.real

        return quad(f, 0, 40 / (2 * k1 * h), limit=400)[0]

    f_perp = 1 + 1.5 * (perp.real + tail(0))
    f_par = 1 + 0.75 * (par.real + tail(1))

    def up(which):
        def f(tt):
            si, sz = math.sin(tt), math.cos(tt)
            rs, rp = airy(si, n1, layers, n3)
            e = cmath.exp(2j * k1 * h * sz)
            if which == 0:
                return 0.75 * si**3 * abs(1 + rp * e) ** 2
            return 0.375 * si * (abs(1 + rs * e) ** 2 + sz**2 * abs(1 - rp * e) ** 2)

        return quad(f, 0, math.pi / 2, limit=400)[0]

    return f_perp, f_par, up(0), up(1)


SLAB = {"n1": 1.0, "layers": [(2.0, 0.25 * UM)], "n3": 1.45, "height": 0.25 * UM,
        "sigma": 0.04 * UM, "period": 1.0 * UM}  # fmt: skip


def slab_cell(h=0.125 * UM):
    period, d = SLAB["period"], SLAB["layers"][0][1]
    mesh = hpfem.rectangle(round(period / h), 32, [-period / 2, -2 * UM], [period / 2, 2 * UM])
    for c in range(mesh.num_cells):
        x = np.mean([mesh.vertex(int(v)) for v in mesh.cell_vertices(c)], axis=0)
        if -d < x[1] < 0:
            mesh.set_cell_tag(c, 2)
        elif x[1] < -d:
            mesh.set_cell_tag(c, 3)
    core = hpfem.Material.dielectric(SLAB["layers"][0][0])
    substrate = hpfem.Material.dielectric(SLAB["n3"])
    stack = hpfem.LayerStack2D(hpfem.Material.dielectric(SLAB["n1"]), [hpfem.Layer(core, d)],
                               substrate, 0.0)  # fmt: skip
    return mesh, {2: core, 3: substrate}, stack


def cell(period, substrate_index, cover_index, h=0.1 * UM):
    mesh = hpfem.rectangle(round(period / h), 40, [-period / 2, -2 * UM], [period / 2, 2 * UM])
    for c in range(mesh.num_cells):
        x = np.mean([mesh.vertex(int(v)) for v in mesh.cell_vertices(c)], axis=0)
        if x[1] < 0:
            mesh.set_cell_tag(c, 2)
    substrate = hpfem.Material.dielectric(substrate_index)
    stack = hpfem.LayerStack2D(hpfem.Material.dielectric(cover_index), [], substrate, 0.0)
    return mesh, {2: substrate}, stack


def image_dipole(k0, h):
    """Purcell factors (perpendicular, parallel) of a point dipole at height h above a perfect
    mirror (x = 2kh): F_⊥ = 1 + 3(sin x/x³ − cos x/x²), F_∥ = 1 − (3/2)(sin x/x + cos x/x² −
    sin x/x³); all the power goes up."""
    x = 2 * k0 * h
    f_perp = 1 + 3 * (math.sin(x) / x**3 - math.cos(x) / x**2)
    f_par = 1 - 1.5 * (math.sin(x) / x + math.cos(x) / x**2 - math.sin(x) / x**3)
    return f_perp, f_par, f_perp, f_par


MIRROR = {"mirror_150": 0.15 * UM, "mirror_300": 0.3 * UM, "mirror_600": 0.6 * UM}


def mirror_cell(period=1.0 * UM, h=0.1 * UM):
    mesh = hpfem.rectangle(round(period / h), 25, [-period / 2, 0.0], [period / 2, 2.5 * UM])
    stack = hpfem.LayerStack2D(hpfem.Material.dielectric(1.0), [],
                               hpfem.Material(complex(-1e10, 1.0)), 0.0)  # fmt: skip
    return mesh, {}, stack


CASES = {
    # name: (period, cover index, substrate index, dipole height)
    "homogeneous": (1.0 * UM, 1.5, 1.5, 0.1 * UM),
    "glass": (1.0 * UM, 1.0, 1.5, 0.4 * UM),
    "glass_period": (0.8 * UM, 1.0, 1.5, 0.4 * UM),
    "slab": None,  # SLAB: a lossless waveguide, guided modes (ADR-0013 §4a)
    **{name: None for name in MIRROR},  # a perfect mirror, the image-dipole closed form
}


def reference(name):
    k0 = 2 * math.pi / LAMBDA
    if name == "slab":
        f_perp, f_par, up_perp, up_par = sommerfeld_layered(
            SLAB["n1"], SLAB["layers"], SLAB["n3"], k0, SLAB["height"]
        )
        return {"x": (f_par, up_par), "y": (f_perp, up_perp), "z": (f_par, up_par)}
    if name in MIRROR:
        f_perp, f_par, up_perp, up_par = image_dipole(k0, MIRROR[name])
        return {"x": (f_par, up_par), "y": (f_perp, up_perp), "z": (f_par, up_par)}
    _period, n1, n2, height = CASES[name]
    f_perp, f_par, up_perp, up_par = sommerfeld(n1, n2, k0, height)
    return {"x": (f_par, up_par), "y": (f_perp, up_perp), "z": (f_par, up_par)}


def run(name, order, nodes, kx_nodes):
    bottom, pml = "pml", {"top": UM, "bottom": UM}
    if name == "slab":
        mesh, materials, stack = slab_cell()
        height, sigma = SLAB["height"], SLAB["sigma"]
    elif name in MIRROR:
        mesh, materials, stack = mirror_cell()
        height = MIRROR[name]
        sigma = min(0.04 * UM, height / 7)  # the Gaussian 6σ clear of the PEC wall
        bottom, pml = "pec", {"top": UM}
    else:
        period, n1, n2, height = CASES[name]
        mesh, materials, stack = cell(period, n2, n1)
        sigma = SIGMA
    omega = units.angular_frequency(wavelength=LAMBDA)
    t0 = time.perf_counter()
    r = grating.dipole_emission(
        mesh, materials, stack, {"position": (0.0, height), "sigma": sigma}, omega,
        nodes=nodes, kx_nodes=kx_nodes, order=order, symmetric=True, angle_nodes=(8, 16),
        pml=pml, bottom=bottom,
        progress=lambda i, n: print(f"\r  {name} p={order} nodes={nodes}: {i + 1}/{n}", end="",
                                    flush=True),
    )  # fmt: skip
    print()
    ref = reference(name)
    # down_ref is F_ref − up_ref: what is not radiated upwards, i.e. down plus guided on the slab,
    # whose reference does not split them (compare down + nonradiated there)
    out = {"samples": r.samples, "directions": r.directions, "seconds": time.perf_counter() - t0}
    for k in "xyz":
        out[k] = {
            "purcell": r.purcell[k], "purcell_ref": ref[k][0],
            "up": r.up[k] / r.P_bulk, "up_ref": ref[k][1],
            "down": r.down[k] / r.P_bulk, "down_ref": ref[k][0] - ref[k][1],
            "nonradiated": r.nonradiated[k] / r.P_bulk,
        }  # fmt: skip
        if r.guided is not None:
            out[k]["guided"] = r.guided[k] / r.P_bulk
    if r.guided_modes is not None:
        out["guided_poles"] = len(r.guided_modes)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    parser.add_argument("--nodes", nargs="+", type=int, default=[6, 8])
    parser.add_argument("--kx-nodes", type=int, default=4)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--record", default=os.path.join(
        os.path.dirname(__file__), "results",
        f"{datetime.date.today().isoformat()}-validation-m17-scan.json"))  # fmt: skip
    args = parser.parse_args()
    record = {}
    if os.path.exists(args.record):
        with open(args.record, encoding="utf-8") as f:
            record = json.load(f)
    record.setdefault("description", "M17 S3 array scanning, benchmarks/m17_dipole_scan.py")
    record.setdefault("hpfem", hpfem.__version__)
    runs = record.setdefault("runs", {})
    for name in args.cases:
        for nodes in args.nodes:
            key = f"{name}/p{args.order}/nodes{nodes}/kx{args.kx_nodes}"
            if key in runs:
                print(f"{key}: done")
                continue
            runs[key] = run(name, args.order, nodes, args.kx_nodes)
            tmp = args.record + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=1)
            os.replace(tmp, args.record)
            res = runs[key]
            print(f"{key}: {res['samples']} samples, {res['seconds']:.0f} s")
            for k in "xyz":
                v = res[k]
                print(f"  {k}: F {v['purcell']:.5f} (ref {v['purcell_ref']:.5f}), "
                      f"up {v['up']:.5f} (ref {v['up_ref']:.5f}), "
                      f"nonradiated {v['nonradiated']:+.1e}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())

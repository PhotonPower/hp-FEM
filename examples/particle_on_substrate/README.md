# Particles and holes on layer stacks (2.5D)

**Physics.** A body of revolution whose axis is normal to a planar stack (a particle on a
substrate, a particle on a mirror, a hole in a film) under a plane wave is solved order by order
in the azimuthal index m on the meridian plane, with the stack as the background of the
scattered-field formulation (`docs/theory/axisymmetric.md`, *Layered background*, ADR-0014):
the incident order is the stack's own plane wave (`hpfem.layered_axisymmetric_wave`), the source
lives only in the cells that deviate from the stack (`setup.background`), and the layers run
through the radial PML. Cross-sections are defined per channel and normalised by the incident
intensity |E0|² / (2 Z0):

- **absorption** of the body: the Joule heating of the total field (stack + scattered) in the
  particle cells (`AxisymmetricScattering.absorbed_power`); around a hole the *absorption change*
  in a region: total field with the actual materials minus the bare stack
  (`incident_absorbed_power`);
- **scattering** through a closed surface of mesh lines around the body, split into the part
  above the stack (`up`), below it (`down`) and through the layers (`lateral`,
  `axisymmetric_flux_channels`);
- **extinction** = absorption + scattering (the optical theorem of a homogeneous medium does not
  hold on a stack);
- the **far field** in the air above (`axisymmetric_layered_far_field`, by reciprocity with the
  stack's plane waves) and the power an objective of numerical aperture NA collects
  (`power_between(0, asin(NA))`);
- the **aperture transmission** of a hole: the power through a disc below the film minus the
  film's own transmission (`axisymmetric_disc_flux`), over the power falling on the hole area.

The planes of incidence are mirror planes, so the orders ±m carry the same powers: the program
solves m = 0, 1, 2, … and counts m > 0 twice, until a pair carries less than 10⁻⁴ of the total
scattered power. Every study checks Poynting's theorem for the total field in the region inside
the measurement surface (`balance` = (inflow − absorption) / absorption).

**What the program does.** Three studies, each a wavelength sweep:

1. **Dark field** — a gold sphere (radius 40 nm, Johnson & Christy) on glass (n = 1.5) with a
   2.5 nm air gap (5 nm in the quick run), illuminated from the air at 60° (s and p averaged),
   collected by an objective of NA 0.5 above (the specular reflection lies outside the cone).
2. **Nanoparticle on a mirror (NPoM)** — a gold sphere (radius 30 nm) over a 1 nm spacer
   (n = 1.45) on a 100 nm gold film on glass, with 1 nm of air between sphere and spacer (2 nm in
   the quick run), p-polarised at 55° (the vertical gap mode needs E_z, i.e. the order m = 0),
   against the same sphere in air.
3. **Nanohole** — a hole of 200 nm diameter through a 100 nm gold film on glass at normal
   incidence (orders ±1 only): T / T_geom from the disc of radius 300 nm 100 nm below the film,
   and the absorption change of the film around the hole.

The meridian meshes are built in the script: the sphere and its bounding square come from
`hpfem.square_with_disc` (curved interface, one ring of cells to the square, which sets the
air gap to radius / n), set into a graded tensor grid whose lines carry every layer interface
and the PML boundaries; the hole needs no curved geometry. The quick configuration uses p = 2,
three wavelengths per study and the orders |m| ≤ 2 (dark field) and |m| ≤ 1 (NPoM).

**Expected result.** The quick run (`--quick`, about 30 s in total) gives

| study | quantity | 480 / 520 / 600 nm | 540 / 600 / 750 nm | 600 / 680 / 900 nm |
|---|---|---|---|---|
| dark field (480, 540, 600 nm) | σ_abs [nm²] | 6741 | 5563 | 894 |
| | σ_sca up / down [nm²] | 913 / 646 | 1828 / 1171 | 836 / 489 |
| | collected, NA 0.5 [nm²] | 59.6 | 100.9 | 42.3 |
| NPoM (520, 600, 680 nm) | σ_sca on the mirror [nm²] | 944 | 6327 | 1694 |
| | σ_sca of the sphere in air [nm²] | 468 | 169 | 75 |
| nanohole (600, 750, 900 nm) | T / T_geom | 0.635 | 0.639 | 0.273 |
| | absorption change / (I π a²) | +0.179 | +0.092 | +0.073 |

with the power balance of the total field closed to 10⁻³ … 10⁻² (4·10⁻² for the sphere in air at
680 nm, where it barely absorbs). The gold sphere on glass scatters most at 540 nm of the three
wavelengths (the plasmon resonance; the quick grid has 60 nm steps); the objective collects about
5 % of the upward scattering. On the mirror the scattering peak lies at 600 nm instead of 520 nm
for the same sphere in air, and its maximum is 13 times higher: the gap mode, with all scattered
power going up (nothing passes the 100 nm gold film). The hole transmits about 64 % of the light
falling on its area at 600 and 750 nm, per unit area 240 to 1200 times the film's own
transmittance, and 27 % at 900 nm.

*Caveat on the channels in a lossy film:* where the measurement surface crosses an absorbing
layer, the flux of the scattered field alone through that part (`lateral`, slightly negative on
the mirror) is not a power carried away by guided modes; the interference with the stack field
is absorbed in the film (the absorption change). The balance of the total field holds regardless.

The full run (p = 3, the gaps of 2.5 nm and 1 nm + 1 nm, 10 nm wavelength steps over 450–700,
480–900 and 600–1000 nm, all orders until a pair carries less than 10⁻⁴; record
`benchmarks/results/2026-10-11-particle-on-substrate.json`) gives

| study | result |
|---|---|
| dark field (7040 cells, \|m\| ≤ 3) | absorption maximum 8494 nm² at 520 nm, scattering maximum 3252 nm² at 530 nm (up 1965, down 1287), collected by NA 0.5: 107 nm² at 530 nm |
| NPoM (14042 cells, \|m\| ≤ 2) | scattering maximum 10284 nm² at 610 nm (absorption maximum 9430 nm² at 600 nm), against 469 nm² at 520 nm for the sphere in air: red-shifted by 90 nm and 22 times stronger |
| nanohole (4920 cells) | T / T_geom 0.64 at 600 nm, maximum 0.80 at 660 nm, 0.50 at 800 nm, 0.15 at 1000 nm; absorption change 0.18 → 0.065 of the power on the hole area |

with the power balance of the total field closed to 6·10⁻³ (dark field), 2·10⁻³ (NPoM) and
2·10⁻³ (hole). Where they share a wavelength, the quick configuration is within 6 % of the dark
field (with a 5 nm instead of a 2.5 nm gap) and within 2 % of the hole; the quick NPoM (a 3 nm
instead of a 2 nm gap) reproduces only the red shift, its scattering at 600 nm is 30 % below the
full run's. The same geometries against `Scattering<3>` + `LayerStack<3>` (a sphere on a
two-layer stack, the hole) are in `docs/validation.md` J: 0.03–0.15 % for the sphere, 0.2–2.5 %
for the hole at p = 3.

**Runtime.** Quick: about 30 s (MinGW GCC, 6 threads). Full: 60 min with MUMPS and 8 threads
(15 min dark field, 40 min NPoM with the sphere in air, 5 min hole).

```bash
python examples/particle_on_substrate/run.py --quick                 # all three, quick
python examples/particle_on_substrate/run.py --study hole            # one study, full
```

Results go to `particle_on_substrate.json` (cross-sections per wavelength in nm², the order
range, the power balance). The regression test (`python/tests/test_examples.py`) runs the quick
configuration.

# Quantum dot above a dielectric grating (M17)

**Physics.** A single emitter — a quantum dot, modelled as a Gaussian dipole (σ = 30 nm),
isotropic (the mean of the three orientations) — sits in air 50 nm above the middle of a glass
ridge (250 nm wide, 150 nm high) of a glass grating (period 500 nm) on a glass substrate
(n = 1.5). The grating is periodic in x and invariant along its lines; the emitter is not periodic.
Its emission is the array scanning of the cell problem over the Bloch wavenumber kx and the
wavenumber β along the lines (`hpfem.grating.dipole_emission`, ADR-0013,
`docs/theory/maxwell.md` "Stage B"):

- the **Purcell factor** F_P = P_em / P_bulk, the emitted power over that of the same dipole in
  air;
- the **channels**: the fractions of the emitted power radiated up into the air and down into the
  glass, by reciprocity integrated over the two half-spaces, and the remainder ("nonradiated",
  absorbed or guided power);
- the **extraction into a numerical aperture**: the fraction an objective of NA 0.5 above the
  sample collects (`hpfem.grating.emission_cone`) — the figure of merit of LEDs and single-photon
  sources.

The structure has no absorption and no high-index layer, so no guided modes (their poles on the
real kx axis are not treated yet, ADR-0013 §4): all emitted power is radiated, and the remainder
measures the accuracy. The reference is the flat glass surface, the same dipole 200 nm above
plain glass.

**What the program does.** It writes the job documents of the runner task `emitter`
(`hpfem.run.run_job`, schema 2): the cell as a `UnitCell` with a structured mesh, the stack
air / glass, stage `"B"` with the scan options, the aperture 0.5 and calibration on. It first runs
a `dry_run` with the cost estimate (solves, memory, predicted time), then the sweep for the
grating and for the flat surface.

**Expected result** (full run: p = 3, cells of 25 nm, 6 / 4 points per β / kx panel, 8 × 16
angle nodes, 600 / 700 / 800 nm; record `benchmarks/results/2026-10-11-grating-emitter.json`):

| λ [nm] | F_P grating | F_P flat | up grating | up flat | down grating | NA 0.5 grating | NA 0.5 flat | remainder grating / flat |
|---|---|---|---|---|---|---|---|---|
| 600 | 1.130 | 1.039 | 0.265 | 0.493 | 0.700 | 0.043 | 0.083 | +0.035 / −0.0001 |
| 700 | 1.164 | 1.046 | 0.258 | 0.465 | 0.725 | 0.049 | 0.089 | +0.017 / +0.0004 |
| 800 | 1.209 | 1.050 | 0.261 | 0.435 | 0.722 | 0.053 | 0.088 | +0.017 / +0.0006 |

The glass ridge under the dot raises the Purcell factor by 9–15 % and pulls the emission into the
substrate: only about a quarter goes up instead of nearly a half, and an NA 0.5 objective above
collects 4–5 % instead of 8–9 %. Seen from the substrate side, the grating is the better
extractor; for collection from above, flat glass is better at this position.

**Accuracy.** On flat glass up + down equal the emitted power to 6·10⁻⁴. With the grating a
remainder of 1.7–3.5 % stays, which should vanish in a structure without absorption and guided
modes. It is not the angle quadrature (16 × 32 instead of 8 × 16 nodes changes up + down by
0.3 %), not the distance to the PML (1 or 2 µm further away: 0.2 %), not the density of the scan
rule (quick and full rule give the same) and not the dipole's Gaussian reaching into the ridge (6σ
above the ridge it is −1.4 % at 700 nm). It is an open accuracy question of the array scanning on
a diffracting structure (reported for M17 S3b); the Purcell factors and fractions above carry
this uncertainty of a few percent.

The quick configuration (p = 2, cells of 50 nm, 3 / 2 points, 4 × 8 angle nodes, 650 nm only)
gives F_P 1.125 / 1.032, up 0.257 / 0.483, NA 0.5 0.046 / 0.088 (grating / flat) with remainders
+0.036 / −0.009; it is the regression test in `python/tests/test_examples.py`.

**Cost.** The dry run predicts 676 solves per wavelength in the full configuration (356 cell
problems of 44 k DoFs, 133 MB each, and 320 plane-wave solves), 1.5–1.9 s per cell problem and
0.8–1.0 s per plane-wave solve on this machine; the full run took 98 min for both structures and
three wavelengths with 8 threads next to other jobs (estimate 38–44 min per structure, measured
46–52). Quick: about 1 min, 162 solves per wavelength.

```bash
python examples/grating_emitter/run.py --quick
python examples/grating_emitter/run.py
```

Results go to `grating_emitter.json` (cost estimate and, per structure and wavelength, F_P, the
fractions, the NA extraction, the number of samples and the time).

# Metasurface unit cell: phase map over the ridge width

**Physics.** A meta-lens is built from sub-wavelength dielectric pillars whose width sets
the phase of the transmitted light; a map of transmission $|t|^2$ and phase $\arg t$ over
the width is the design table of such a lens. The effective 2D model here is a periodic
array of TiO$_2$ ridges (period $a = 350$ nm, height $h = 612.5$ nm) on fused silica,
lit at normal incidence from the air side at $\lambda = 600$ nm with the electric field in
the plane (H$_z$ polarisation). The period is sub-wavelength in air and in the substrate
($\lambda / n = 411$ nm), so only the zeroth order propagates on both sides.

**What the program does.** For every width it builds the Bloch-periodic unit cell with
PML above and below (the lamellar-grating setup of `tests/convergence/lamellar_grating.cpp`,
materials from `hpfem.materials`), solves the scattered-field formulation at $p = 3$ and
extracts the zeroth-order reflection and transmission from Fourier coefficients on lines
above and below the ridge (`fourier_coefficients`, `diffraction_efficiencies`). The phase
is reported relative to the bare substrate (width 0). Results go to
`metasurface_unitcell.json`, the phase map to `metasurface_unitcell.png`.

**Expected result** (12 cells per period, 31k DoF per width):

| width [nm] | $T$ | $R$ | phase / $\pi$ |
|---|---|---|---|
| 29 | 0.970 | 0.028 | +0.13 |
| 88 | 0.998 | 0.001 | +0.52 |
| 146 | 0.056 | 0.944 | −0.10 |
| 204 | 0.995 | 0.002 | +0.17 |
| 263 | 0.916 | 0.084 | +0.70 |
| 321 | 0.949 | 0.054 | −0.94 |

The energy balance $R + T$ holds to $2\cdot10^{-3}$ for every width (lossless, the PML
absorbs the propagating orders only), the phase covers about $2.9\pi$ over the widths, and
the widths 117–175 nm cross a guided-mode resonance of the ridge array (reflection up to
0.94, a fast phase jump) that a lens design would avoid. The efficiencies are stable to
$5\cdot10^{-3}$ under refinement from 8 to 16 cells per period.

**Runtime.** About 3 s for the eleven widths (release build, OpenMP); `--quick` runs seven
widths at 8 cells per period.

```bash
pip install -e ".[dev]"
python examples/metasurface_unitcell/run.py [--quick]
```

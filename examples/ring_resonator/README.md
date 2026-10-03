# Ring resonator: coupling and resonance (2D effective-index model)

**Physics.** A ring waveguide next to a bus waveguide is the basic filter and sensor
element of integrated photonics: whenever the ring circumference holds an integer number
of guided wavelengths, light couples from the bus into the ring and the bus transmission
drops. The resonances are spaced by the free spectral range $\lambda^2 / (n_g L)$ and
their width is set by the coupling to the bus and the radiation loss of the bend. The
effective-index model replaces the silicon-on-insulator slab by the TE effective index
$n_{\mathrm{eff}} = 2.4$ of its guided mode in the silica cladding ($n = 1.44$); the
in-plane electric field (H$_z$ polarisation) is solved on the chip plane. Ring radius
2.5 µm, waveguide width 450 nm, gap 150 nm, around 1550 nm.

**What the program does.** Two views of the same device on a structured mesh with the
ring resolved by cell centroids (a staircase): `hpfem.Resonance2D` finds the quasi-normal
modes of the ring coupled to the bus near the target wavelength (resonance wavelengths and
quality factors, `docs/theory/maxwell.md#resonances`), and a Gaussian line current
(`hpfem.gaussian_current`) at the bus entrance excites the guided mode, whose power at the
bus exit — normalised by the same source without the ring — is the transmission; sampling
it around the resonance closest to 1550 nm shows the dip at the eigenmode wavelength.
Results go to `ring_resonator.json`; `ring_resonator.png` shows the spectrum and $|E|$ of
the resonant mode.

**Expected result** (80 nm cells, $p = 3$):

| resonance [nm] | $Q$ |
|---|---|
| 1509.6 / 1510.1 | 399 / 360 |
| 1568.7 / 1569.3 | 282 / 272 |

| $\lambda$ [nm] | $T$ |
|---|---|
| 1552.0 | 0.988 |
| 1563.7 | 0.867 |
| 1567.0 | 0.551 |
| 1568.7 | 0.322 |
| 1570.4 | 0.424 |
| 1575.4 | 0.869 |
| 1585.4 | 0.972 |

The resonances come in doublets (the clockwise and counter-clockwise ring modes split by
the bus and by the staircase), the free spectral range of 59 nm matches
$\lambda^2 / (n_{\mathrm{eff}}\, 2\pi R) = 64$ nm up to the dispersion of the bend, and the
transmission dip sits exactly at the eigenmode resonance (1568.7 nm from both views) with a
minimum of 0.32 and a width consistent with $Q \approx 280$. The $Q$ is limited by the
staircase approximation of the ring; a mesh that follows the ring (Gmsh with curved cells,
`read_gmsh`) raises it.

**Runtime.** About eight minutes for the eigenmodes and 42 transmission solves at
350k DoF (release build, OpenMP); `--quick` (80 nm cells, $p = 2$, two modes, seven
wavelengths) runs in about two minutes.

```bash
pip install -e ".[dev]"
python examples/ring_resonator/run.py [--quick]
```

# Quantum dot in a micropillar: Purcell factor and β factor

**Physics.** An emitter in a cavity radiates faster than in bulk by the Purcell factor
$F_P = P / P_0$ — classically the power a dipole radiates in the structure over the power it
radiates in the homogeneous host — and the β factor is the fraction of that power leaving
through the top mirror towards the collection optics. The effective 2D model is a line
dipole (moment across the ridge, H$_z$ polarisation) at the centre of the one-wavelength
GaAs cavity of a GaAs / AlAs DBR ridge ("pillar", 1.5 µm wide, $n = 3.48$ / $2.94$,
designed for 950 nm, 6 top and 12 bottom pairs) in air on the GaAs substrate.

**What the program does.** The dipole is a narrow Gaussian line current
(`hpfem.gaussian_current`, width a quarter of the cell size) in the total-field
formulation with PML on all sides, on a mesh with nodes on every layer interface. The
emitted power is the Poynting flux of the field through a closed surface of mesh facets
around the current; $P_0$ is the same current on the same mesh with every material set to
GaAs, so the discretisation of the source cancels in the ratio. The β factor is the flux
through a line in the air above the top mirror divided by $P$. The script first validates
the post-processing with a current in front of a PEC mirror, whose exact power ratio
follows from the dipole plus its image (`hpfem.dipole_field`), then sweeps the wavelength.
Results go to `quantum_dot_purcell.json`, the spectrum to `quantum_dot_purcell.png`.

**Expected result.** Mirror check: $F = 1.8161$ (FEM) against $1.8173$ (image dipole); the
0.07 % difference is the finite width of the current and shrinks with it. Spectrum
(6 / 12 pairs, $p = 3$, about 80k DoF):

| $\lambda$ [nm] | $F_P$ | β (top) |
|---|---|---|
| 940 | 1.54 | 0.58 |
| 943 | 4.09 | 0.63 |
| 945 | 7.82 | 0.65 |
| 946 | 8.04 | 0.65 |
| 948 | 4.64 | 0.60 |
| 950 | 2.61 | 0.53 |
| 955 | 1.34 | 0.32 |
| 960 | 1.12 | 0.20 |

The Purcell factor peaks at 8.0 near 946 nm — the cavity resonance of the finite ridge sits
below the 950 nm design wavelength of the planar stack because of the lateral confinement —
with a width of about 4 nm ($Q \approx 240$); at the peak 65 % of the emitted power leaves
through the top mirror, off resonance the emission goes increasingly into the substrate
and the sides. Far from the resonance $F_P \to 1$.

**Runtime.** About five minutes for the 31 wavelengths (two solves each, release build,
OpenMP); `--quick` runs nine wavelengths of a 4 / 8-pair ridge at $p = 2$ in about 20 s.

```bash
pip install -e ".[dev]"
python examples/quantum_dot_purcell/run.py [--quick]
```

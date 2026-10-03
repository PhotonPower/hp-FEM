# Textured silicon solar cell: light trapping, generation profile, heating

**Physics.** A thin crystalline-silicon absorber reflects part of the light at its front
and transmits the long wavelengths that silicon absorbs weakly. A surface texture scatters
the light into oblique orders that are trapped by total internal reflection, so the path
in the absorber grows and so does the absorbed fraction. The absorbed photons are the
carrier generation that a device solver needs as input ($G = q / \hbar\omega$), and the
absorbed power heats the cell. The 2D effective model is a 2 µm silicon slab with a
250 nm ridge grating of 600 nm period under an 80 nm silica anti-reflection layer,
tabulated optical constants (`hpfem.materials`: Si after Green 2008, SiO₂ after
Malitson), lit at normal incidence with the electric field in the plane.

**What the program does.** For every band of a sample solar spectrum it solves the
Bloch-periodic unit cell with PML above and in the substrate (scattered-field
formulation) for the textured and the flat cell and reports the absorbed fraction of the
incident power in the absorber (`absorbed_power_per_cell`). `hpfem.pv` turns the textured
solutions into the generation rate per cell, weights the bands by the spectral irradiance,
exports the result as cell data (`solar_cell_texture_generation.vtu`) and as the
laterally averaged depth profile $G(z)$ (`solar_cell_texture_profile.csv`, the input of a
1D device solver). The same spectral weights sum the thermal loads, and `Thermal2D` gives
the temperature rise of the front with the rear at the ambient temperature. Results go to
`solar_cell_texture.json`, plots to `solar_cell_texture.png`.

**Expected result** (40 nm cells, $p = 3$, 41k DoF per wavelength):

| $\lambda$ [nm] | absorbed, textured | absorbed, flat |
|---|---|---|
| 500 | 0.68 | 0.57 |
| 600 | 0.61 | 0.49 |
| 700 | 0.36 | 0.26 |
| 800 | 0.17 | 0.13 |
| 900 | 0.068 | 0.046 |
| 1000 | 0.015 | 0.010 |

The texture raises the absorbed fraction by 20–50 % across the spectrum; the generation
profile decays from the front into the absorber with the interference fringes of the thin
film, and the temperature rise of the front is of the order of $10^{-5}$ K for the sample
spectrum (a few W/m² in six bands) with the rear 2 µm away at the ambient temperature —
scale the irradiance to the full spectrum and move the heat sink for a device-level
estimate.

**Runtime.** About 15 s for the six bands (two cells each plus the thermal loads, release
build, OpenMP); `--quick` (60 nm cells, $p = 2$, three bands) runs in about 3 s.

```bash
pip install -e ".[dev]"
python examples/solar_cell_texture/run.py [--quick]
```

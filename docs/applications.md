# Application areas

| Area | Problem class | Specific features needed | Example (milestone) |
|---|---|---|---|
| Scatterometry / EUV masks | periodic scattering, oblique incidence | Bloch BCs, layered incident field, Fourier coefficients, goal-oriented estimator | `lamellar_grating` (M4), `euv_mask` (M8) |
| Metasurfaces / meta-lenses | unit-cell scattering sweeps | parameter sweeps, phase/transmission extraction | `metasurface_unitcell` (M8) |
| Photovoltaics | absorption, light trapping | absorbed-power density, heat coupling | `solar_cell_texture` (M9) |
| Integrated photonics | propagating modes, ring resonators | quadratic eigenproblem in $k_z$, port BCs | `slab_waveguide` (M4), `ring_resonator` (M8) |
| VCSEL / LED / OLED | resonances, outcoupling | complex eigenvalues with PML, Q-factor, far field | `vcsel_cavity` (M8) |
| Plasmonics / biosensors | field singularities at metal corners | hp-adaptivity, Drude/tabulated metals | `plasmonic_dimer` (M5) |
| Quantum optics | dipole emission in cavities | point dipole source, Purcell/β factor | `quantum_dot_purcell` (M8) |

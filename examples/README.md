# Examples

One directory per application area, each with a `README.md` (physics, expected result,
runtime), a Python driver script and, where needed, a Gmsh `.geo` file.

| Directory              | Area                         | Milestone |
|------------------------|------------------------------|-----------|
| `cavity_modes/`        | PEC cavity eigenmodes (validation) | M3  |
| `mie_cylinder/`        | Scattering off a dielectric cylinder | M4 |
| `slab_waveguide/`      | Integrated photonics: waveguide modes | M4 |
| `lamellar_grating/`    | Scatterometry: periodic grating | M4 |
| `plasmonic_dimer/`     | Plasmonics: hp-adaptivity at metal corners | M5 |
| `metasurface_unitcell/`| Meta-lens phase map           | M8 |
| `vcsel_cavity/`        | Laser resonator modes, Q-factor | M8 |
| `quantum_dot_purcell/` | Purcell factor in a micropillar | M8 |
| `solar_cell_texture/`  | Light trapping + heat (multiphysics) | M9 |

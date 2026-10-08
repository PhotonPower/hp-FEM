# Examples

One directory per application area, each with a `README.md` (physics, expected result,
runtime) and a driver. The M3–M5 examples are C++ programs (`main.cpp`, built as
`example_<name>` with the `HPFEM_BUILD_EXAMPLES` option, on by default) with an equivalent
`project.json` for the `hpfem` command line; the M8 examples are Python drivers
(`run.py`, `python examples/<name>/run.py [--quick]`) over the bindings, exercised by
`python/tests/test_examples.py`. Each writes its results to stdout and result files
(`.json`, `.vtu`, `.png`) into the working directory.

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
| `directional_coupler_3d/` | SOI directional coupler with 3D modal ports vs coupled-mode theory | M12 |

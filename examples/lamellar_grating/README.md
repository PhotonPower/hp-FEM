# Lamellar grating (scatterometry)

**Physics.** A plane wave hits a periodic array of dielectric ridges (period $a$, fill
factor $f$, thickness $t$) on a substrate. The field above and below the grating is a sum
of plane waves — the diffraction orders $m$ with tangential wavenumbers
$k_{y,m} = k_{y,0} + 2\pi m/a$ — and the diffraction efficiencies $\eta_m$ are the fractions of
the incident power carried by the propagating orders.

**What the program does.** Solves one Bloch-periodic unit cell with PML above and below
in the scattered-field formulation ($p = 3$, in-plane electric field, $10°$ incidence),
takes the Fourier coefficients of the scattered field on a line in the superstrate
(reflected orders) and of the total field on a line in the substrate (transmitted orders),
converts them to efficiencies and writes the scattered field to `lamellar_grating.vtu`.

**Expected result.** Three propagating reflected and transmitted orders ($m = -1, 0, 1$),
efficiencies summing to one within $2\cdot10^{-3}$ (lossless grating). At normal incidence
the values match the rigorous coupled-wave reference of
`tests/convergence/lamellar_grating.cpp` ($R_0 = 0.0515$, $T_{\pm1} = 0.4454$, …).

**Runtime.** About two seconds (release build).

```bash
cmake --build --preset release --target example_lamellar_grating
./build/release/examples/example_lamellar_grating
```

The same setup as a project file for the Python command line: `hpfem run examples/lamellar_grating/project.json`
(see `docs/python.md`).

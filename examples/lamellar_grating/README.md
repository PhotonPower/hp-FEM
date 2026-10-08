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

## Job files of the acceptance cases (`jobs/`)

The conical cases of `docs/gui-support-features.md` section 3 (period 400 nm, ridge 200 nm,
height 148 nm, λ = 405 nm, substrate = ridge material, air above) as job files of the job
runner, `python -m hpfem.run examples/lamellar_grating/jobs/<name>.json --out out`:

| job | case | RCWA reference | uniform p of the job file |
|---|---|---|---|
| `si_tm_50deg.json` | Si (ε = 29.6345 + 2.7721i), TM (p), θ = 50°, φ = 0 | R0 0.143381, R−1 0.142382 | p = 3, 34 k DoFs, 0.6 s: R0 0.143345, R−1 0.142758 (4·10⁻⁴); energy balance 6·10⁻³ (absorbed power of the lossy substrate) |
| `si_conical_50_40deg.json` | Si, TM, θ = 50°, φ = 40° (conical, β ≠ 0) | R0 0.142373, R−1 0.179169 | p = 3: R0 0.142011, R−1 0.179675 (5·10⁻⁴) |
| `ag_te_50deg.json` | Ag (ε = −4.6631 + 0.2160i), TE (s), θ = 50°, φ = 0 (acceptance case R1) | R0 0.319215, R−1 0.643575, A 0.03721 | p = 4, 59 k DoFs, 1 s: R0 0.319216, R−1 0.643572, A 0.03721 (3·10⁻⁶, balance 2·10⁻⁶; in the E_z polarisation the metal corners carry no field singularity, unlike the TM case of the hp test `conical_grating_hp`) |

Each job writes `results.json` (orders, absorption, balance, timing) and a field map
`maps_0_0.npz`; `python/tests/test_examples.py` runs the Si and the Ag job against the
references.

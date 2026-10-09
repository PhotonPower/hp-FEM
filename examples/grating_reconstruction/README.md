# Scatterometry: reconstruction of a silicon line grating with uncertainties

**Physics.** Optical scatterometry measures the specular reflectance of a periodic structure
over wavelength, angle and polarisation and infers its geometry by fitting a rigorous
electromagnetic model; it is the standard in-line metrology for the critical dimension (CD),
height and side-wall angle (SWA) of lithographic lines. The example reconstructs a silicon
line grating on silicon (period 300 nm, mid-height width CD = 100 nm, height 120 nm,
SWA = 86°) from the zeroth-order reflectance $R_0$ for s and p polarisation at 65° incidence
over 400–700 nm, with the optical constants of silicon from M. A. Green, Sol. Energy Mater.
Sol. Cells 92, 1305 (2008) (`hpfem.materials.get("Si")`, used for the line, the substrate
and the background stack at every wavelength).

**What the program does.**

1. *Synthetic measurement.* The reflectances are computed one polynomial order higher than
   the model ($p = 4$) on a mesh built at the true geometry and perturbed with Gaussian noise
   of $\sigma = 0.002$, so that the fit sees a discretisation error as with real data and is
   not handed its own model ("inverse crime").
2. *Model.* A structured mesh whose columns follow the slanted walls is built at the start
   geometry (CD 112 nm, height 108 nm, SWA 88.5°) and morphed by the three parameters
   (`hpfem.opt.Morph` with `trapezoid_parameters`, ADR-0012 §3). The morph keeps the cells at
   the measurement line and the PML fixed (`band`), because the order amplitudes sampled on a
   line of mesh facets are not differentiable when those cells deform. `GratingEvaluator`
   evaluates the 14 reflectances and their Jacobian: one factorisation per wavelength and
   polarisation, the derivatives along the morph velocities by tangent solves on the kept
   factors (M16 S1).
3. *Fit.* `hpfem.opt.fit` (Levenberg–Marquardt on $W^{1/2}(R(p) - R_{meas})$, $W = 1/\sigma^2$)
   from the start values, every evaluation stored in `grating_reconstruction.study.jsonl`;
   the Laplace approximation $(J^\top W J)^{-1}$ at the optimum gives the standard errors and
   correlations (ADR-0012 §6).

**Expected result** (full run, $p = 3$, 32.6k DoFs, 7 wavelengths × 2 polarisations):

| | true | start | estimate | std. error | (estimate − true) / std |
|---|---|---|---|---|---|
| CD [nm] | 100.000 | 112.0 | 100.004 | 0.105 | 0.03 |
| height [nm] | 120.000 | 108.0 | 119.956 | 0.295 | −0.15 |
| SWA [°] | 86.000 | 88.5 | 85.920 | 0.133 | −0.60 |

χ²_red = 0.47 (11 degrees of freedom), model error (order 3 against 4 at the true geometry)
2.8·10⁻⁴ in reflectance, well below the noise; 12 Levenberg–Marquardt iterations with 9 new
evaluations. Correlations: CD–height −0.31, CD–SWA 0.15, height–SWA −0.59.

**Validation of the uncertainties.** The deviations above are one noise realisation (seed 7).
Fits of the model's own data at a nearby geometry agree with the linearised estimate
$\delta p = (J^\top W J)^{-1} J^\top W \varepsilon$ to 0.01 standard errors (the curvature
terms are below 1 % of the noise), and over 200 noise realisations the linearised deviations
have mean −0.02 and standard deviation 0.93 standard errors: the Laplace uncertainties
describe the scatter of the estimates. An earlier version of the morph superposed the
parameter velocities linearly; the missing cross term of height and side-wall angle biased
the SWA by 2.5 standard errors in every realisation (ADR-0012 §3, amendment). The exact
morph removed it.

**Runtime.** About 5 minutes for the full run (14 solves per evaluation, the synthetic data at
order 4 included) on the development machine; `--quick` (3 wavelengths, $p = 2$ for the
model and $p = 3$ for the data, 15k DoFs) takes about 40 s and is the regression test in
`python/tests/test_examples.py` (estimates within 4 standard errors of the truth). In the
quick run the model error (1.7·10⁻³) is of the size of the noise, so its estimates are less
accurate than the standard errors suggest.

```bash
pip install -e ".[dev]"
python examples/grating_reconstruction/run.py [--quick]
```

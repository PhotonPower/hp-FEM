# Fabrication tolerances: from geometry errors to the reflectance spectrum

**Physics.** A lithographic line grating is never made exactly to its design. How much do
small, random errors of the critical dimension (CD), the height and the side-wall angle (SWA)
change its optical response, and which of them matters most at which wavelength? The example
takes the silicon line grating on silicon of `examples/grating_reconstruction` (period
300 nm, nominal CD 100 nm, height 120 nm, SWA 86°; silicon from M. A. Green, Sol. Energy
Mater. Sol. Cells 92, 1305 (2008)) with independent normal errors of 2 nm (CD), 3 nm (height)
and 0.5° (SWA), one standard deviation each, and propagates them to the zeroth-order
reflectance $R_0$ for s and p polarisation at 65° incidence over 400–700 nm (14 outputs).

**What the program does** (M16 S7, `hpfem.opt`):

1. *Linearised propagation* (`linear_propagation`): one evaluation with the Jacobian at the
   nominal geometry; $C_R = J \Sigma J^\top$ and the share of each tolerance in each variance.
2. *Global surrogate* (`build_global_surrogate`): 16 Latin-hypercube evaluations with the
   Jacobian over ±4 standard deviations of the three parameters, then 8 points of active
   learning; one gradient-enhanced Gaussian process per reflectance.
3. *Monte Carlo* on the surrogate (`monte_carlo`, 20 000 samples): mean, standard deviation
   and the 2.5–97.5 % band of every reflectance.
4. *Sobol' indices* (`sobol_indices`, 2¹³ base points): first-order and total index of each
   tolerance for each reflectance.

The model is the one of the reconstruction example (structured mesh whose columns follow the
walls, morphed by CD, height and angle; efficiencies and their Jacobian from the kept
factorisation, `GratingEvaluator`), imported from `examples/grating_reconstruction/run.py`.

**Expected result** (full run, $p = 3$, 7 wavelengths × 2 polarisations):

| $R_0$ | nominal | std (linear) | std (Monte Carlo) | 2.5–97.5 % | dominant tolerance ($S_T$) |
|---|---|---|---|---|---|
| s 400 nm | 0.4026 | 0.0092 | 0.0091 | 0.386–0.422 | height 0.74 |
| p 400 nm | 0.0820 | 0.0124 | 0.0122 | 0.060–0.107 | CD 0.93 |
| s 450 nm | 0.5281 | 0.0055 | 0.0054 | 0.516–0.537 | CD 0.94 |
| p 450 nm | 0.1166 | 0.0189 | 0.0175 | 0.081–0.148 | CD 0.91 |
| s 500 nm | 0.5183 | 0.0057 | 0.0057 | 0.507–0.529 | height 0.70 |
| p 500 nm | 0.1780 | 0.0091 | 0.0092 | 0.160–0.195 | height 0.93 |
| s 550 nm | 0.5095 | 0.0183 | 0.0189 | 0.465–0.538 | CD 0.76 |
| p 550 nm | 0.1904 | 0.0070 | 0.0071 | 0.176–0.204 | height 0.55 |
| s 600 nm | 0.3186 | 0.0231 | 0.0223 | 0.286–0.373 | CD 0.98 |
| p 600 nm | 0.4064 | 0.0060 | 0.0060 | 0.393–0.417 | height 0.61 |
| s 650 nm | 0.4071 | 0.0197 | 0.0193 | 0.370–0.445 | CD 0.53 |
| p 650 nm | 0.4099 | 0.0090 | 0.0091 | 0.390–0.426 | height 0.90 |
| s 700 nm | 0.4929 | 0.0108 | 0.0109 | 0.470–0.512 | CD 0.59 |
| p 700 nm | 0.3572 | 0.0112 | 0.0113 | 0.334–0.378 | height 0.91 |

The linearised standard deviations agree with Monte Carlo on the surrogate to 8 % or better
(the largest difference at p 450 nm, where $R_0$ is curved in the CD), the first-order and
total Sobol' indices agree to 0.01 except at s 600 nm (CD 0.94 against 0.98, height 0.01
against 0.05: an interaction from the curvature in the CD), and both agree with the
linearised variance shares to a few hundredths (s 600 nm: 0.99 for the CD): for tolerances of
this size the
response is nearly linear and additive, and the cheap linearised analysis (one evaluation)
already ranks the tolerances correctly. The side-wall angle contributes at most 0.3 of the
variance (s 500 nm) and is negligible elsewhere; the CD dominates the s reflectance below
600 nm and the height the p reflectance above 500 nm. The 95 % bands are a few hundredths in
reflectance, five to twenty times the 0.002 measurement noise of the reconstruction example.
The surrogate's largest relative standard deviation over a candidate pool is 0.021.

**Runtime.** About 25 minutes for the full run (25 evaluations of 14 cell solves with the
Jacobian, the sampling on the surrogate takes seconds); `--quick` (450 and 600 nm, $p = 2$,
8 surrogate points without active learning) about 30 s, the regression test in
`python/tests/test_examples.py` (Monte Carlo within 15 % of the linearised values, the CD
dominant at s 450 nm).

```bash
pip install -e ".[dev]"
python examples/fabrication_tolerance/run.py [--quick]
```

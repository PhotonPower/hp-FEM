# Validation against published benchmarks

Milestone M10 compares hp-FEM with **independent, published reference values**, not with
our own implementations. Every benchmark on this page is a convergence test in
`tests/convergence/` with the ctest labels `convergence` and `validation`; the long local
runs (label `validation-long`, hidden Catch2 tag, excluded from the test presets) produce
the tables below and are stored as JSON lines in `benchmarks/results/`. Only numbers are
taken from the sources (parameters and reference results, with the full citation); the
set-ups are described here in our own words. Parameters are never adjusted to match a
reference.

Conventions: time dependence $e^{-i\omega t}$, lossy media have $\operatorname{Im}\varepsilon > 0$
(ADR-0002). Values quoted from sources that use $e^{+i\omega t}$ are conjugated here.

## A. Rib waveguide (Vassallo 1997)

**Source.** C. Vassallo, "1993–1995 Optical mode solvers", *Optical and Quantum
Electronics* **29**, 95–114 (1997), section 7 and Table I, column MTRM (the author's
reference method, stated to be exact to four digits).

**Set-up.** A classical semiconductor rib: a guiding layer of refractive index 3.44 on a
substrate of index 3.40, air above, vacuum wavelength 1.15 µm. Under the rib the guiding
layer is 1.0 µm thick in total (rib included); beside the 3.0 µm wide rib it is thinned
to the residual thickness $t \in \{0.1, 0.3, 0.5, 0.7, 0.9\}$ µm. All materials are
lossless, so the 2D cross-section problem is solved with `physics::PropagatingMode<2>`
(Nédélec transverse field, H1 longitudinal field). The quantity compared is the
normalised propagation constant

$$
B = \frac{n_\text{eff}^2 - n_s^2}{n_g^2 - n_s^2}, \qquad n_s = 3.40,\; n_g = 3.44,
$$

of the fundamental quasi-TE mode (dominant $E_x$, parallel to the substrate) and the
fundamental quasi-TM mode (dominant $E_y$). $\Delta B = 10^{-4}$ corresponds to
$\Delta n_\text{eff} \approx 4\cdot 10^{-6}$. The quasi-TM mode at $t = 0.9$ µm is leaky
(its effective index lies below the TE slab mode of the lateral layer); the solver's
value is reported but not asserted.

**Discretisation.** The cross-section is a PEC box. The mesh is a conforming
tensor-product triangulation (built in the test, `tests/convergence/rib_waveguide.cpp`)
whose coordinate lines coincide with the material interfaces and are refined
geometrically (ratio 1/4, `levels` steps) towards the four dielectric corners of the rib,
where the transverse field is singular. On that mesh the polynomial order $p$ is raised
uniformly. The mirror symmetry in $x$ is exploited: a PEC wall at $x = 0$ admits the
modes with even $E_x$ (quasi-TE class), the natural (PMC) condition those with even $E_y$
(quasi-TM class). Within a class the modes are identified by their polarisation fraction
$\int |E_x|^2 / \int (|E_x|^2 + |E_y|^2)$ (from the component mass matrices), never by
their order in the spectrum. The long study uses walls 3 µm below the layer, 0.5 µm above
the rib and 6 µm beside the rib centre (the source's values) and varies them.

**Reference values and results.** The tables are produced by the long run
(`convergence_rib_waveguide "[validation-long]"`, results in
`benchmarks/results/2026-10-03-validation-rib-waveguide.json`); times are wall-clock
seconds for mesh, DoF maps, assembly and eigen-solve on the maintainer's machine
(24 cores, MSYS2 GCC, release preset).

*p-convergence on the graded mesh* (uniform spacing 0.25 µm, 3 geometric levels of ratio 1/4
at the rib corners, walls 3 µm below / 0.5 µm above / 6 µm beside; the DoF counts and times
are those of the $t = 0.5$ µm case, the other cases are within 5 %):

| t [µm] | mode | B (MTRM) | p = 2 | p = 3 | p = 4 | p = 5 | p = 6 | (B(p=6) − B_ref)·10⁴ |
|---|---|---|---|---|---|---|---|---|
| 0.1 | TE | 0.3019 | 0.301531 | 0.301882 | 0.301896 | 0.301896 | 0.301896 | −0.04 |
| 0.1 | TM | 0.2674 | 0.267425 | 0.267475 | 0.267475 | 0.267475 | 0.267475 | +0.75 |
| 0.3 | TE | 0.3110 | 0.310683 | 0.311018 | 0.311032 | 0.311032 | 0.311032 | +0.32 |
| 0.3 | TM | 0.2751 | 0.275119 | 0.275163 | 0.275163 | 0.275163 | 0.275163 | +0.63 |
| 0.5 | TE | 0.3270 | 0.326678 | 0.326991 | 0.327003 | 0.327004 | 0.327004 | +0.04 |
| 0.5 | TM | 0.2890 | 0.288955 | 0.289020 | 0.289019 | 0.289019 | 0.289019 | +0.19 |
| 0.7 | TE | 0.3512 | 0.350871 | 0.351153 | 0.351164 | 0.351165 | 0.351165 | −0.35 |
| 0.7 | TM | 0.3107 | 0.310680 | 0.310720 | 0.310718 | 0.310718 | 0.310718 | +0.18 |
| 0.9 | TE | 0.3883 | 0.388327 | 0.388531 | 0.388540 | 0.388540 | 0.388540 | +2.40 (wall, see below) |
| 0.9 | TM (leaky) | 0.3455 | 0.345353 | 0.345382 | 0.345381 | 0.345381 | 0.345381 | −1.19 (wall, see below) |

| p | DoF (t = 0.5 µm, half cross-section) | time [s] |
|---|---|---|
| 2 | 12 841 | 1 |
| 3 | 27 361 | 2 |
| 4 | 47 281 | 6 |
| 5 | 72 601 | 17 |
| 6 | 103 321 | 34 |

From $p = 4$ on the discrete $B$ is converged to about $10^{-7}$; the remaining differences
to the reference are therefore properties of the reference (four printed digits, i.e. a
rounding of up to $0.5\cdot 10^{-4}$) and of the box walls, not of our discretisation.

*Geometric refinement towards the rib corners* ($t = 0.5$ µm, $p = 4$): without grading the
corner singularities limit the accuracy to about $10^{-4}$ in $B$; three levels suffice.

| levels | DoF | B (TE) | B (TM) |
|---|---|---|---|
| 0 | 22 801 | 0.3270778 | 0.2891156 |
| 1 | 30 129 | 0.3270150 | 0.2890303 |
| 2 | 38 289 | 0.3270047 | 0.2890201 |
| 3 | 47 281 | 0.3270034 | 0.2890188 |
| 4 | 57 105 | 0.3270032 | 0.2890186 |
| 5 | 67 761 | 0.3270032 | 0.2890186 |

*Lateral wall position* ($p = 4$). For $t \le 0.7$ µm the wall is irrelevant from 4 µm on.
For $t = 0.9$ µm the fundamental mode lies close to the TE slab mode of the thick lateral
layer and decays slowly sideways: the wall has to be at 10 µm before the fourth digit of
$B$ is stable. The source's statement that the walls do not matter therefore holds for the
thin lateral layers only; at $t = 0.9$ µm the reference evidently used a wide (or open)
lateral domain. The leaky quasi-TM mode at $t = 0.9$ µm has no converged value in a closed
box: its $B$ and its polarisation fraction (last column, $\int|E_x|^2$ share) jump with the
wall position because it hybridises with the box-quantised slab modes.

| wall \|x\| [µm] | t = 0.1 TE | t = 0.1 TM | t = 0.9 TE | t = 0.9 TM (leaky; E_x fraction) |
|---|---|---|---|---|
| 4 | 0.301896 | 0.267475 | 0.390415 | 0.343346 (0.01) |
| 6 | 0.301896 | 0.267475 | 0.388540 | 0.345381 (0.04) |
| 8 | 0.301896 | 0.267475 | 0.388328 | 0.345292 (0.31) |
| 10 | 0.301896 | 0.267475 | 0.388306 | 0.345467 (0.02) |

*Vertical wall positions* (substrate below / air above, in µm, $p = 4$). The source's walls
(3 below, 0.5 above) are converged to $2\cdot 10^{-5}$ in $B$, i.e. they do not change the
fourth digit; 4 / 1.0 is converged to $10^{-6}$.

| below / above [µm] | t = 0.1 TE | t = 0.1 TM | t = 0.5 TE | t = 0.5 TM |
|---|---|---|---|---|
| 2 / 0.5 | 0.301587 | 0.267920 | 0.326702 | 0.289457 |
| 3 / 0.5 | 0.301896 | 0.267475 | 0.327003 | 0.289019 |
| 4 / 1.0 | 0.301907 | 0.267456 | 0.327013 | 0.289001 |
| 5 / 1.5 | 0.301907 | 0.267455 | 0.327014 | 0.289000 |

*Symmetry check* ($t = 0.5$ µm, $p = 4$): the half cross-section with the symmetry wall at
$x = 0$ reproduces the full cross-section (all four walls PEC, modes classified by their
polarisation fraction) to $10^{-7}$.

| mode | half cross-section (symmetry wall) | full cross-section |
|---|---|---|
| TE | 0.3270034 (47 281 DoF) | 0.3270035 (94 321 DoF) |
| TM | 0.2890188 (47 281 DoF) | 0.2890188 (94 321 DoF) |

*CI variant* (spacing 0.5 µm, 3 levels, $p = 4$, walls 4 / 1.0 µm and 5 µm beside, 10 µm
beside for $t = 0.9$ µm; 32 s for the whole test case including the $p$-sequence):

| t [µm] | mode | wall \|x\| [µm] | DoF | n_eff | B | B (MTRM) | (B − B_ref)·10⁴ | time [s] |
|---|---|---|---|---|---|---|---|---|
| 0.1 | TE | 5 | 21 129 | 3.4121256 | 0.301905 | 0.3019 | +0.05 | 2.5 |
| 0.1 | TM | 5 | 21 129 | 3.4107442 | 0.267456 | 0.2674 | +0.56 | 2.5 |
| 0.3 | TE | 5 | 21 129 | 3.4124919 | 0.311041 | 0.3110 | +0.41 | 2.2 |
| 0.3 | TM | 5 | 21 129 | 3.4110525 | 0.275144 | 0.2751 | +0.44 | 2.3 |
| 0.5 | TE | 5 | 20 289 | 3.4131321 | 0.327012 | 0.3270 | +0.12 | 1.7 |
| 0.5 | TM | 5 | 20 289 | 3.4116082 | 0.289001 | 0.2890 | +0.01 | 1.7 |
| 0.7 | TE | 5 | 21 129 | 3.4141005 | 0.351178 | 0.3512 | −0.22 | 2.0 |
| 0.7 | TM | 5 | 21 129 | 3.4124782 | 0.310699 | 0.3107 | −0.01 | 1.9 |
| 0.9 | TE | 10 | 34 209 | 3.4155881 | 0.388311 | 0.3883 | +0.11 | 4.5 |
| 0.9 | TM (leaky) | 10 | 34 209 | 3.4138712 | 0.345456 | 0.3455 | −0.44 | 4.5 |

**Assessment.**

All nine guided reference values are reproduced within $0.8\cdot 10^{-4}$ in $B$
($\Delta n_\text{eff} \le 3\cdot 10^{-6}$), which is the precision of a four-digit reference;
our own values are converged an order of magnitude further. The two caveats concern the
reference, not the solver: the quasi-TE mode at $t = 0.9$ µm needs a lateral domain of about
10 µm (the source's claim of wall independence is too optimistic there), and the leaky quasi-TM
mode at $t = 0.9$ µm is not a well-defined eigenvalue of a closed box; with the 10 µm wall our
value happens to lie $0.4\cdot 10^{-4}$ below the published one, but the table above shows that
this agreement is accidental. No parameter was adjusted. Cost: the CI table at $p = 4$ takes
about 2 s per mode on 20 000–34 000 unknowns; the long study (p up to 6, $10^5$ unknowns)
takes about 30 minutes in total.

The CI variant of the test checks the nine guided cases at the discretisation shown to be
converged above and the $p$-sequence for $t = 0.5$ µm, with the tolerance
$|B - B_\text{ref}| \le 2\cdot 10^{-4}$ (the reference has four digits).

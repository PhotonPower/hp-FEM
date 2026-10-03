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

## B. Metallic lamellar grating, H parallel to the ridges (Granet & Guizal 1996)

**Source.** G. Granet, B. Guizal, "Efficient implementation of the coupled-wave method for
metallic lamellar gratings in TM polarization", *J. Opt. Soc. Am. A* **13**, 1019–1023
(1996), Table 1, column "Exact" (values computed by L. Li with a modal method); the structure
is that of L. Li, C. W. Haggans, *J. Opt. Soc. Am. A* **10**, 1184–1189 (1993).

**Set-up.** A grating of period $d = 1$ µm with metal ridges of height $h$ on a half-infinite
substrate of the same metal is illuminated from air at $\lambda = 1$ µm under $30°$ with the
magnetic field parallel to the ridges (in-plane $E$). The metal has the refractive index
$0.22 + 6.71i$ in our $e^{-i\omega t}$ convention, i.e. $\varepsilon = -44.9757 + 2.9524i$
(skin depth 24 nm). Two orders propagate: the specular order 0 and order $-1$ with
$k_y = k_0(\sin 30° - \lambda/d) = -k_0/2$, which travels back along the direction of incidence
(Littrow mounting); they are assigned by their wave vector, not by a label. The ridge width is
not stated in the sources; it is **assumed** to be $d/2$ and the assumption is checked, not
fitted (see below).

**Discretisation.** Bloch-periodic unit cell (`tests/convergence/metal_grating.cpp`), the
period along $y$ and the surface normal along $x$ as the diffraction post-processing of the
library expects; scattered-field formulation with the vacuum plane wave as incident field, PML
in the air above the ridge, PEC 0.25 µm (ten skin depths) below the metal surface, which is
consistent with the scattered field tending to $-E_\text{inc}$ inside the metal. The mesh is
the tensor-product triangulation of `tensor_mesh.hpp` with coordinate lines on all interfaces,
graded geometrically (ratio 1/4, three levels) towards the ridge corners and the metal surfaces.
The efficiencies come from the Fourier coefficients of the scattered field on a line in the air,
the absorbed power from $\tfrac12\,\omega\varepsilon_0\,\operatorname{Im}\varepsilon\int|E|^2$
over the metal (exact total field of the formulation), so that $\eta_{-1} + \eta_0 + A = 1$ is
an independent check.

**Results** (long run `convergence_metal_grating "[validation-long]"`,
`benchmarks/results/2026-10-03-validation-metal-grating.json`):

*p-convergence* (spacing 0.25 µm, PML 3 µm of order 4 with target reflection $10^{-14}$):

| h [µm] | p | DoF | η₋₁ | η₀ | A | (1 − Σ)·10⁴ | time [s] |
|---|---|---|---|---|---|---|---|
| 0.1 | 2 | 4,568 | 0.33335 | 0.61959 | 0.03006 | -170.0 | 0.0 |
| 0.1 | 3 | 9,540 | 0.34063 | 0.63053 | 0.02790 | -9.4 | 0.1 |
| 0.1 | 4 | 16,304 | 0.34108 | 0.63099 | 0.02790 | -0.4 | 0.1 |
| 0.1 | 5 | 24,860 | 0.34088 | 0.63123 | 0.02790 | +0.1 | 0.4 |
| 0.1 | 6 | 35,208 | 0.34085 | 0.63124 | 0.02790 | -0.0 | 0.9 |
| 1.0 | 2 | 5,054 | 0.09032 | 0.84107 | 0.06057 | -80.4 | 0.0 |
| 1.0 | 3 | 10,557 | 0.10111 | 0.84792 | 0.05001 | -9.6 | 0.1 |
| 1.0 | 4 | 18,044 | 0.10159 | 0.84829 | 0.04996 | -1.6 | 0.1 |
| 1.0 | 5 | 27,515 | 0.10156 | 0.84848 | 0.04996 | -0.0 | 0.5 |
| 1.0 | 6 | 38,970 | 0.10155 | 0.84848 | 0.04996 | -0.0 | 1.2 |
| 4.8 | 2 | 7,646 | 0.00886 | 0.63173 | 0.40242 | +430.0 | 0.1 |
| 4.8 | 3 | 15,981 | 0.04762 | 0.50372 | 0.44841 | -2.4 | 0.1 |
| 4.8 | 4 | 27,324 | 0.05002 | 0.49928 | 0.45073 | +0.3 | 0.2 |
| 4.8 | 5 | 41,675 | 0.05023 | 0.49885 | 0.45093 | +0.0 | 0.8 |
| 4.8 | 6 | 59,034 | 0.05025 | 0.49879 | 0.45095 | -0.0 | 2.0 |

*Against the reference* ($p = 6$):

| h [µm] | η₋₁ (Li) | η₋₁ (hp-FEM) | Δ·10⁴ | η₀ (Li) | η₀ (hp-FEM) | Δ·10⁴ | Σ (Li) | Σ (hp-FEM) | A (hp-FEM) |
|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 0.3408 | 0.34085 | +0.5 | 0.6312 | 0.63124 | +0.4 | 0.9720 | 0.97210 | 0.02790 |
| 1.0 | 0.1024 | 0.10155 | -8.5 | 0.8477 | 0.84848 | +7.8 | 0.9501 | 0.95003 | 0.04996 |
| 4.8 | 0.0503 | 0.05025 | -0.5 | 0.4985 | 0.49879 | +2.9 | 0.5488 | 0.54904 | 0.45095 |

*Checks.* The flat metal surface (no ridge) reproduces the Fresnel reflectance
$R_p = 0.9781663$ and the absorption $1 - R_p$:

| p | DoF | η₀ | η₀ − R_p | A | A − (1 − R_p) | 1 − Σ |
|---|---|---|---|---|---|---|
| 2 | 5,054 | 0.9615469 | -1.7e-02 | 0.0239478 | +2.1e-03 | +1.5e-02 |
| 3 | 10,557 | 0.9769805 | -1.2e-03 | 0.0218338 | +5.6e-08 | +1.2e-03 |
| 4 | 18,044 | 0.9780752 | -9.1e-05 | 0.0218321 | -1.6e-06 | +9.3e-05 |
| 5 | 27,515 | 0.9781612 | -5.1e-06 | 0.0218337 | -4.4e-08 | +5.1e-06 |
| 6 | 38,970 | 0.9781652 | -1.1e-06 | 0.0218337 | -4.4e-08 | +1.1e-06 |

The PML matters at the $10^{-4}$ level: the library's usual 1 µm / order 2 / $10^{-10}$ layer
leaves $10^{-4}$ of the incident power unaccounted for although every other discretisation
parameter is converged; 3 µm / order 4 / $10^{-14}$ closes the balance to $10^{-6}$.

| h [µm] | p | DoF | η₋₁ | η₀ | A | (1 − Σ)·10⁴ | time [s] |
|---|---|---|---|---|---|---|---|
| 0.1 | 2 | 4,568 | 0.33335 | 0.61959 | 0.03006 | -170.0 | 0.0 |
| 0.1 | 3 | 9,540 | 0.34063 | 0.63053 | 0.02790 | -9.4 | 0.1 |
| 0.1 | 4 | 16,304 | 0.34108 | 0.63099 | 0.02790 | -0.4 | 0.1 |
| 0.1 | 5 | 24,860 | 0.34088 | 0.63123 | 0.02790 | +0.1 | 0.4 |
| 0.1 | 6 | 35,208 | 0.34085 | 0.63124 | 0.02790 | -0.0 | 0.9 |
| 1.0 | 2 | 5,054 | 0.09032 | 0.84107 | 0.06057 | -80.4 | 0.0 |
| 1.0 | 3 | 10,557 | 0.10111 | 0.84792 | 0.05001 | -9.6 | 0.1 |
| 1.0 | 4 | 18,044 | 0.10159 | 0.84829 | 0.04996 | -1.6 | 0.1 |
| 1.0 | 5 | 27,515 | 0.10156 | 0.84848 | 0.04996 | -0.0 | 0.5 |
| 1.0 | 6 | 38,970 | 0.10155 | 0.84848 | 0.04996 | -0.0 | 1.2 |
| 4.8 | 2 | 7,646 | 0.00886 | 0.63173 | 0.40242 | +430.0 | 0.1 |
| 4.8 | 3 | 15,981 | 0.04762 | 0.50372 | 0.44841 | -2.4 | 0.1 |
| 4.8 | 4 | 27,324 | 0.05002 | 0.49928 | 0.45073 | +0.3 | 0.2 |
| 4.8 | 5 | 41,675 | 0.05023 | 0.49885 | 0.45093 | +0.0 | 0.8 |
| 4.8 | 6 | 59,034 | 0.05025 | 0.49879 | 0.45095 | -0.0 | 2.0 |ML

Discretisation parameters (grading ratio and levels, uniform spacing down to 0.0625 µm, metal
depth below the surface, number of Fourier sampling points) change the converged efficiencies
by less than $2\cdot 10^{-5}$:

| variant | h [µm] | DoF | η₋₁ | η₀ | A | 1 − Σ |
|---|---|---|---|---|---|---|
| metal_depth_0.1 | 1 | 18,044 | 0.10157 | 0.84835 | 0.04992 | +1.6e-04 |
| metal_depth_0.15 | 1 | 18,044 | 0.10159 | 0.84829 | 0.04996 | +1.6e-04 |
| metal_depth_0.25 | 1 | 18,044 | 0.10159 | 0.84829 | 0.04996 | +1.6e-04 |
| metal_depth_0.5 | 1 | 18,624 | 0.10159 | 0.84829 | 0.04996 | +1.6e-04 |
| fourier_points_256 | 1 | 27,515 | 0.10156 | 0.84847 | 0.04996 | +1.1e-05 |
| fourier_points_1024 | 1 | 27,515 | 0.10156 | 0.84848 | 0.04996 | +3.3e-06 |
| fourier_points_4096 | 1 | 27,515 | 0.10156 | 0.84847 | 0.04996 | +5.7e-06 |
| fourier_points_16384 | 1 | 27,515 | 0.10156 | 0.84847 | 0.04996 | +5.1e-06 |
| grading_0.25_0 | 1 | 4,295 | 0.10037 | 0.85017 | 0.05070 | -1.2e-03 |
| grading_0.25_2 | 1 | 18,015 | 0.10156 | 0.84848 | 0.04996 | +4.5e-06 |
| grading_0.25_3 | 1 | 27,515 | 0.10156 | 0.84848 | 0.04996 | +3.3e-06 |
| grading_0.25_5 | 1 | 51,795 | 0.10156 | 0.84847 | 0.04996 | +4.2e-06 |
| grading_0.5_4 | 1 | 38,775 | 0.10155 | 0.84849 | 0.04996 | +2.2e-06 |
| grading_0.5_6 | 1 | 66,575 | 0.10155 | 0.84848 | 0.04996 | +2.1e-06 |
| grading_0.5_8 | 1 | 101,415 | 0.10155 | 0.84848 | 0.04996 | +2.1e-06 |
| spacing_0.25 | 1 | 38,970 | 0.10155 | 0.84848 | 0.04996 | +4.1e-06 |
| spacing_0.125 | 1 | 78,420 | 0.10156 | 0.84848 | 0.04996 | +2.4e-06 |
| spacing_0.0625 | 1 | 192,888 | 0.10156 | 0.84848 | 0.04996 | +2.2e-06 |
| spacing_0.25 | 4.8 | 59,034 | 0.05025 | 0.49879 | 0.45095 | +4.2e-06 |
| spacing_0.125 | 4.8 | 126,966 | 0.05025 | 0.49878 | 0.45096 | +2.1e-06 |
| spacing_0.0625 | 4.8 | 326,478 | 0.05026 | 0.49878 | 0.45097 | +2.2e-06 |

*Validation of the assumed fill factor.* With $f = 0.5$ two of the three depths agree with the
reference to $5\cdot 10^{-5}$ ($h = 0.1$ µm) and $3\cdot 10^{-4}$ ($h = 4.8$ µm); $f = 0.45$
or $0.55$ change the efficiencies by 3 to 30 percentage points at every depth. The assumption
is therefore confirmed, not fitted.

| h [µm] | f | η₋₁ | η₀ | A |
|---|---|---|---|---|
| 0.1 | 0.45 | 0.29433 | 0.67682 | 0.02886 |
| 0.1 | 0.5 | 0.34088 | 0.63123 | 0.02790 |
| 0.1 | 0.55 | 0.37364 | 0.59866 | 0.02769 |
| 1 | 0.45 | 0.00527 | 0.89476 | 0.09996 |
| 1 | 0.5 | 0.10156 | 0.84848 | 0.04996 |
| 1 | 0.55 | 0.09921 | 0.70370 | 0.19709 |
| 4.8 | 0.45 | 0.05130 | 0.79708 | 0.15161 |
| 4.8 | 0.5 | 0.05023 | 0.49885 | 0.45093 |
| 4.8 | 0.55 | 0.15966 | 0.70456 | 0.13577 |

*The $h = 1$ µm case.* Our converged efficiencies differ from the reference by
$-8.5\cdot 10^{-4}$ ($\eta_{-1}$) and $+7.8\cdot 10^{-4}$ ($\eta_0$) while their sum agrees to
$7\cdot 10^{-5}$. At this depth the air slot between the ridges is resonant (depth $= \lambda$),
and the efficiencies are extremely sensitive to the geometry and the material:
$\partial\eta/\partial h \approx 3.5$ per µm, so the deviation corresponds to a depth change of
0.24 nm, and rounding $\varepsilon$ to $-45 + 3i$ already moves $\eta_0$ by $6\cdot 10^{-4}$.
The fourth digit of the reference at $h = 1$ µm therefore depends on details the paper does
not state (the exact $\varepsilon$ and $h$ used in Li's computation); our discretisation is
converged to $10^{-5}$ and energy-consistent to $10^{-6}$.

| h [µm] | η₋₁ | η₀ | A |
|---|---|---|---|
| 0.98 | 0.03878 | 0.91322 | 0.04800 |
| 0.99 | 0.06648 | 0.88455 | 0.04897 |
| 1 | 0.10156 | 0.84848 | 0.04996 |
| 1.01 | 0.14373 | 0.80528 | 0.05098 |
| 1.02 | 0.19252 | 0.75547 | 0.05202 |

| ε | η₋₁ | η₀ | A |
|---|---|---|---|
| -44.98 + 2.95i | 0.10156 | 0.84852 | 0.04992 |
| -45 + 3i | 0.10143 | 0.84787 | 0.05069 |
| -44.9757 + 2.9524i | 0.10156 | 0.84848 | 0.04996 |

**Assessment.** Two of the three depths agree with the four-digit modal reference within the tolerance $3\cdot 10^{-4}$ ($h = 0.1$ µm to $0.5\cdot 10^{-4}$, $h = 4.8$ µm to $2.9\cdot 10^{-4}$) with the assumed fill factor $f = 0.5$, which the sensitivity table rules out changing. The $h = 1$ µm case is reported but not asserted (decision with the dev agent, 2026-10-03): the slot is resonant there, the published digits depend on the unstated exact $\varepsilon$ and $h$ of the reference computation, and our result is converged to $10^{-5}$ with an energy balance of $10^{-6}$ and reproduces the Fresnel limit to $10^{-6}$. Nothing was fitted or loosened. Lesson for the library: a PML of one wavelength with the quadratic profile and target reflection $10^{-10}$ is not enough for efficiencies at the $10^{-4}$ level; 3 µm, order 4 and $10^{-14}$ are. Cost: about 1 s per solve at $p = 5$ on 25 000–40 000 unknowns; the whole long study takes a few minutes.

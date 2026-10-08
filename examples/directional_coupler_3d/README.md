# Directional coupler in 3D (SOI, modal ports)

**Physics.** Two parallel silicon strip waveguides (450 nm × 220 nm, silica cladding, gap
150 nm, 1550 nm) exchange power along their coupling length: the fundamental modes of the
isolated cores combine into an even and an odd supermode of the two-core cross-section with
propagation constants $\beta_e > \beta_o$, and light launched into one core beats between
the cores with the coupling coefficient $\kappa = (\beta_e - \beta_o)/2$. Coupled-mode theory
gives the powers after the length $L$,

$$
P_{\mathrm{cross}} = \sin^2(\kappa L), \qquad P_{\mathrm{bar}} = \cos^2(\kappa L),
\qquad L_c = \frac{\pi}{2\kappa},
$$

exact in the two-mode picture up to the small mismatch between the launched single-core mode
and the superposition of the supermodes. This is the coupling element of ring resonators,
Mach–Zehnder interferometers and switches, and the first 3D device of the port framework
(`docs/theory/maxwell.md#waveguide-ports-and-s-parameters`).

**What the program does.** Builds a structured tetrahedral mesh of the coupler with the
cores tagged by cell centroid and grid lines on the core edges (`hpfem.box`), PEC walls at
the cladding margins (the guided modes have decayed to below $10^{-2}$ there), and four modal
ports on the half faces $z = 0$ and $z = L$ (port 1: core A at the input, 2: core B at the
input, 3 and 4 at the output), whose modes come from the 2D mode solver on the extracted
half sections. Port 1 is fed with unit amplitude; the outgoing modal powers of the four ports
give $P_{\mathrm{bar}}$ (port 3), $P_{\mathrm{cross}}$ (port 4), the reflection and the
back-coupling. The reference $\kappa$ comes from `hpfem.PropagatingMode` on the full two-core
cross-section at the same cell size and order. Results go to `directional_coupler_3d.json`.

```bash
python examples/directional_coupler_3d/run.py --quick                       # CPU, < 1 min
python examples/directional_coupler_3d/run.py --length 12 --order 2 --backend cudss
python examples/directional_coupler_3d/run.py --length 30 --order 2 --cell 100 --cell-z 250 --backend cudss --ports full
```

**Expected result.** The coupler is lossless between PEC walls: the four outgoing modal
powers sum to one to $10^{-5}$ and the reflection stays below $10^{-4}$ from p = 2 on. For
L = 2 µm and gap 200 nm (100 nm cells, MUMPS, 24 threads):

| p | DoFs | time | $\kappa$ [1/µm] | $P_{\mathrm{cross}}$ FEM | CMT $\sin^2(\kappa L)$ |
|---|---|---|---|---|---|
| 1 (`--quick`) | 50 k | 3 s | 0.115 | 0.080 | 0.052 |
| 2 | 266 k | 52 s | 0.077 | 0.0297 | 0.0234 |
| 3 | 769 k | 300 s | 0.075 | 0.0307 | 0.0222 |

Both sides converge, but to values about 0.008 apart, and the offset stays (0.011 at gap
300 nm / L = 4 µm, 0.007 at gap 400 nm / L = 6 µm): the half-face ports put a PEC plane
through the centre of the gap, so the launched and the measured port modes are the modes of
the half sections, whose evanescent tails across the plane are mis-assigned to the other
core; the two-mode theory assumes isolated-core modes. The clean comparison is `--ports
full`: one two-mode port per end face, the S-matrix in the basis of the supermodes of the
full section, and the single-core launch of the two-mode picture reconstructed from the 3D
phases, $P_{\mathrm{cross}} = \sin^2ig((rg t_e - rg t_o)/2ig)$:

| p | DoFs | time (4 solves) | $|t_e|$, $|t_o|$ | phase error [rad] | leakage | $P_{\mathrm{cross}}$ FEM | CMT |
|---|---|---|---|---|---|---|---|
| 1 | 50 k | 20 s | 0.997, 0.998 | 0.25 / 0.20 | 6·10⁻² | 0.0626 | 0.0523 |
| 2 | 266 k | 293 s | 1.00000, 1.00000 | 0.014 / 0.015 | 1.6·10⁻³ | 0.02359 | 0.02338 |

The phase error $|rg t - eta L|$ against the section solver's $eta$ and the leakage
between the supermodes are the convergence quantities of the 3D solver; at p = 2 the cross
power agrees with coupled-mode theory to 0.9 %. The production validation (gap 150 nm,
L = 10–30 µm, p = 2–3, 1–1.5 M DoFs with cuDSS, `--ports full`) is the GPU agent's record in
`benchmarks/results/` and `docs/validation.md`. `--quick` asserts the loss balance, the
reflection and agreement with coupled-mode theory within a factor two (half ports) and 25 %
(supermode ports) at p = 1.

**Runtime.** `--quick` 3 s (50 k DoFs, MUMPS); p = 2 on the same mesh 52 s, p = 3 five
minutes (770 k DoFs); the production runs minutes per solve on the RTX 3090 with cuDSS.

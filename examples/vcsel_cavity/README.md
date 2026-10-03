# VCSEL micro-cavity: resonance wavelength and quality factor

**Physics.** A vertical-cavity surface-emitting laser is a one-wavelength GaAs cavity
between two distributed Bragg reflectors of quarter-wave AlAs / GaAs pairs ($n = 2.95$ /
$3.52$, designed for $\lambda = 850$ nm), with air above and the GaAs substrate below. The
cavity mode is a quasi-normal mode: it exists at a complex frequency, decays by radiation
through the mirrors, and its quality factor $Q = \mathrm{Re}\,\omega / (-2\,\mathrm{Im}\,\omega)$
grows exponentially with the number of mirror pairs. For the planar, laterally uniform mode
the stack is one-dimensional, so the resonance is also the pole of the stack's transfer
matrix (no incoming wave on either side), which the script computes as a reference.

**What the program does.** It builds a narrow strip mesh with nodes on every layer interface
(two triangles per interval, PEC side walls, air and substrate margins and a two-wavelength
PML at both ends), solves the resonance eigenproblem with `hpfem.Resonance2D` at $p = 3$
around the design frequency (`docs/theory/maxwell.md#resonances`), and compares wavelength
and $Q$ with the transfer-matrix pole for several numbers of top pairs. Results go to
`vcsel_cavity.json`, the mode profile $|E_y(x)|$ of the last configuration to
`vcsel_cavity.png`.

**Expected result** (14 bottom pairs, 4.2–4.8k DoF):

| top pairs | $\lambda$ [nm] | $Q$ | $\lambda$ TMM [nm] | $Q$ TMM |
|---|---|---|---|---|
| 4 | 849.999 | 133.2 | 850.000 | 133.2 |
| 6 | 849.999 | 259.3 | 850.000 | 259.3 |
| 8 | 849.999 | 457.5 | 850.000 | 457.5 |
| 10 | 849.999 | 719.1 | 850.000 | 719.1 |

The finite-element resonance agrees with the transfer-matrix pole to $10^{-6}$ in the
wavelength and to four digits in $Q$; $Q$ grows by a factor of about 1.6–1.9 per two added
pairs until the bottom mirror (14 pairs) limits it. With a uniform mesh that does not resolve
the layer interfaces the wavelength is off by half a per cent and $Q$ by ten per cent — the
interface-aligned 1D grid is what makes the comparison exact.

**Runtime.** About one second for the four configurations (release build); `--quick` runs
two configurations with 8 bottom pairs.

```bash
pip install -e ".[dev]"
python examples/vcsel_cavity/run.py [--quick]
```

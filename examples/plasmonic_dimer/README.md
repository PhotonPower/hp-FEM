# Plasmonic dimer: hp-adaptivity at metal corners

**Physics.** Two metal rods ($0.3\lambda \times 0.3\lambda$, gold-like
$\varepsilon = -9 + 1.2i$) with a $0.1\lambda$ gap are illuminated by a plane wave polarised
along the dimer axis. The gap concentrates the field, and at the eight $90^\circ$ metal
corners the field is singular, $|E| \sim r^{\mathrm{Re}\,\nu - 1}$ with $\nu = 0.573 - 0.015i$
from $\tan(3\pi\nu/4) + \varepsilon\tan(\pi\nu/4) = 0$ (a PEC corner has $\nu = 2/3$) — the
situation of plasmonic sensors and tips. No closed form exists; the hp loop must find the
corners by itself.

**What the program does.** Starts from a $40 \times 40$ mesh with $p = 2$ (rods and gap on the
grid, cells of $0.05\lambda$ for the skin depth, PML of $0.3\lambda$ around a $1.4\lambda$ box,
scattered-field formulation) and runs seven steps of SOLVE → ESTIMATE → MARK → DECIDE →
REFINE: the residual estimator gives the indicators, Dörfler marking ($\theta = 0.5$) the
cells, the error prediction (`adaptivity::hp_decide_by_prediction`) the choice between h-
and p-refinement, and `adaptivity::hp_refine` performs it on the `mesh::AdaptiveMesh` with
hanging nodes. The dual-weighted residual estimate (`physics::dwr_estimate`) reports the
error of the gap field $E_x(0,0)$ in every step. The last step writes the scattered field,
the cell orders, refinement levels and both indicators to `plasmonic_dimer.vtu`.

**Expected result.** The gap enhancement converges to $|E_x/E_0| = 1.346$ (from $1.321$ on
the start mesh; $1.3459$, $1.3463$ in the last two steps) while the residual estimate drops
from $50$ to $16$ with the DoFs growing from $16$k to $28$k; the goal estimate stays around
$0.02$–$0.04$, a conservative bound here because the adjoint of a point value next to the
singular corners is itself hard to approximate. The final mesh shows geometric grading at
the corners and $p$ up to $4$ along the rod faces. `tests/convergence/plasmonic_wedge.cpp`
checks the exponential convergence of the loop on the manufactured corner solution
(convergence test #7, second part).

**Runtime.** About 25 seconds (release build, up to 28k DoFs, seven solves with their
$p+1$ adjoints).

```bash
cmake --build --preset release --target example_plasmonic_dimer
./build/release/examples/example_plasmonic_dimer
```

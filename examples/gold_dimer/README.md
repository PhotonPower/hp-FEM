# Gold sphere dimer: field in a 1 nm gap (M10 benchmark, under a documented assumption)

**Physics.** Two gold spheres of 80 nm diameter with a 1 nm gap form a plasmonic nano-antenna:
a plane wave polarised along the dimer axis drives the gap plasmon and the electric field at
the gap centre is enhanced by orders of magnitude (surface-enhanced spectroscopy, single
molecule detection). Hoffmann, Hafner, Leidenberger, Hesselbarth and Burger (Proc. SPIE 7390,
73900J, 2009; [arXiv 0907.3570](https://arxiv.org/abs/0907.3570)) used this case to compare
Maxwell solvers and give the multiple-multipole reference
$|E|^2 = 5.47624\cdot10^5\ \mathrm{V^2/m^2}$ at the gap centre for $|E_{inc}| = 1$ V/m at
632 nm, "at least five digits correct". **The paper does not state the permittivity of gold
it used.** This example takes the Johnson & Christy data of the material library
($\varepsilon_{\mathrm{Au}}(632\,\mathrm{nm}) = -11.685 + 1.267i$) as a documented assumption
and reports how the result depends on $\varepsilon$ (see `docs/validation.md`, section G).

**What the program does.** The dimer is a body of revolution. The incident wave (perpendicular
to the axis, polarised along it) is expanded in azimuthal orders (`hpfem.oblique_plane_wave`,
$\theta_i = 90^\circ$, p polarisation) and every order is solved on the meridian mesh with
`hpfem.AxisymmetricScattering` (cylindrical PML, PEC outer wall). On the axis only the orders
$m = 0$ ($E_z$) and $m = \pm1$ ($E_x$, $E_y$) are non-zero, so three 2D solves give the gap
field exactly. The meridian mesh comes from Gmsh (OCC, second-order elements on the sphere
surfaces, graded from 0.1 nm at the gap to 25 nm in the box). `--orders` runs the polynomial
orders on one mesh, `--size` scales the mesh, `--eps RE IM` overrides the gold permittivity,
and without `--no-sensitivity` the derivatives of $|E|^2$ with respect to $\mathrm{Re}\,\varepsilon$
and $\mathrm{Im}\,\varepsilon$ are taken by central differences (four extra three-order
solves). Results go to `gold_dimer.json`; needs the `gmsh` package.

```bash
python examples/gold_dimer/run.py --quick                 # p = 2, 3 on a coarser mesh, 5 s
python examples/gold_dimer/run.py --orders 2 3 4          # 6 715 cells, 10 min incl. sensitivity
python examples/gold_dimer/run.py --orders 3 --size 2 --no-sensitivity --eps -10.3 0.8
```

**Expected result.** With Johnson & Christy gold the gap intensity is converged to five digits
in $p$ and independent of the mesh:

| mesh | p | DoFs (m = 0) | $|E|^2$ [V²/m²] |
|---|---|---|---|
| coarse (1 843 cells) | 2 / 3 | 13 k / 28 k | 2.97662e5 / 2.97242e5 |
| fine (6 715 cells) | 2 / 3 / 4 | 48 k / 101 k / 176 k | 2.97372e5 / 2.97239e5 / 2.97239e5 |

i.e. $2.9724\cdot10^5$, **46 % below the reference**. The field at the centre is along the axis
($|E_z| = 545$, $|E_x| = 10^{-4}$, $E_y = 0$). The sensitivity is $-4.2$ % per 1 % of
$\mathrm{Re}\,\varepsilon$ and $-0.9$ % per 1 % of $\mathrm{Im}\,\varepsilon$ at the Johnson &
Christy point, but the dependence on $\mathrm{Re}\,\varepsilon$ is resonant (maximum
$3.74\cdot10^5$ near $\mathrm{Re}\,\varepsilon = -10.3$ at $\mathrm{Im}\,\varepsilon \approx 1.2$):
no value of $\mathrm{Re}\,\varepsilon$ reaches the reference with the Johnson & Christy loss.
The reference is reproduced for $\varepsilon \approx -10.3 + 0.8i$ ($5.43\cdot10^5$), a gold
with about 35 % lower loss than Johnson & Christy at 632 nm (single-crystal data such as Olmon
et al. 2012 lie in that range). So the solver's result for a given $\varepsilon$ is converged
and the deviation from the published number is the material datum, which the paper does not
give; the five-digit comparison stays out of reach. `--quick` is the regression test
(`python/tests/test_examples.py`: convergence in $p$, the field along the axis, deviation
below 60 %).

**Runtime.** `--quick` 5 s; the fine mesh with $p = 2, 3, 4$ two minutes, the sensitivity
eight more (MUMPS, 24 threads).

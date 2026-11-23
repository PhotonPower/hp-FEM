# Crossed grating: a FourierGrating hole array recomputed with hp-FEM

**Physics.** A square array of air holes (radius 125 nm, period 500 nm in both directions) in a
150 nm glass film on glass ($n = 1.5$), lit from the air at normal incidence with
$\lambda = 405$ nm and the electric field along one lattice direction. The period exceeds
$\lambda / n$ in the glass, so the substrate carries the orders $(\pm 1, 0)$, $(0, \pm 1)$ and
$(\pm 1, \pm 1)$ and the air the first orders as well: a crossed grating with nine transmitted
and five reflected propagating orders, the setting of scatterometry and of metasurfaces of
finite height.

**What the program does.** The model is FourierGrating's own configuration
(`hole_array_glass_r125.json`, stack normal y, periodic in x and z). `hpfem.fouriergrating`
imports it into `hpfem.crossed` (stack normal z), which meshes the unit cell with Gmsh (second-order
tetrahedra at the hole), solves the scattered field on the layered background with PML above
and below and Bloch-periodic side faces, and measures the orders on planes in air and glass. The
efficiencies are compared order by order with FourierGrating's RCWA ($N_x = N_z = 20$,
`reference_rcwa_N20.csv`) for $p = 3, 4$ on one mesh (2.5 elements per local wavelength).
With `--fg-nearfield` the near fields are compared on FourierGrating's cuts as well (run
`fgsolve hole_array_glass_r125_nearfield.json` first); `--plots` draws them.

**Expected result** (p polarisation; efficiencies, relative L2 deviation of the near field
away from the material edges):

| | unknowns | time | R(0,0) | T(0,0) | max \|Δη\| vs RCWA | ΣR + ΣT − 1 | near field |
|---|---|---|---|---|---|---|---|
| RCWA, N = 20 | | 16 min | 0.024680 | 0.804148 | | 6e-13 | |
| hp-FEM p = 3 | 39 564 | 35 s | 0.024994 | 0.802281 | 1.9e-3 | −1.2e-3 | 1.4–1.6e-2 |
| hp-FEM p = 4 | 84 888 | 4.8 min | 0.024655 | 0.804491 | 3.4e-4 | 5.5e-5 | 4.5–4.9e-3 |

The diffracted orders agree to $5\cdot10^{-5}$ at $p = 4$, the largest deviation is in
$T_{00}$; the near-field differences are largest at the 90° edges of the hole, where the field
is singular and the RCWA shows its Gibbs ripple (docs/validation.md N, with the plots).

**Runtime.** About 5 min for $p = 3, 4$ with SparseLU on a 6-core laptop (the factorisation
dominates); `--quick` ($p = 2$ on a coarse mesh, every order within $2\cdot10^{-2}$) takes
about 10 s.

```bash
pip install "hpfem[meshing]"          # gmsh
python examples/crossed_hole_array/run.py [--quick] [--fg-nearfield DIR --plots DIR]
hpfem fg examples/crossed_hole_array/hole_array_glass_r125.json -p 4 -e 2.5   # the CSV only
```

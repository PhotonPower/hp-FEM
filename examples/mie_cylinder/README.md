# Scattering off a dielectric cylinder (Mie)

**Physics.** A plane wave with in-plane electric field ($H_z$ polarisation) hits an
infinite dielectric cylinder of radius $R$ and index $n$. The scattered power per unit
length divided by the incident intensity is the scattering width $\sigma_{sca}$, known in
closed form from the Mie series (`physics/mie.hpp`).

**What the program does.** Meshes a box with the circular inclusion resolved by curved
cell edges (`mesh::square_with_disc`), solves the scattered-field formulation with a PML
around the box ($p = 3$), and computes $\sigma_{sca}$ twice — from the Poynting flux of the
scattered field through the cylinder surface and from the Stratton–Chu far field — next to
the series value; it also prints the far-field pattern and writes the scattered field to
`mie_cylinder.vtu`.

**Expected result.** $k_0R = 1.5$, $n = 1.5$: $\sigma_{sca} = 0.50641$ m from the series,
the two FEM values within $3\cdot10^{-4}$ (near field) and $10^{-2}$ (far field, limited by
the quadrature on the coarse surface) of it, an absorption cross-section of order $10^{-4}$
(discretisation error of a lossless scatterer). See `tests/convergence/mie_cylinder.cpp`
for the convergence with $p$.

**Runtime.** About five seconds (release build, 86k DoFs).

```bash
cmake --build --preset release --target example_mie_cylinder
./build/release/examples/example_mie_cylinder
```

The same setup as a project file for the Python command line: `hpfem run examples/mie_cylinder/project.json`
(see `docs/python.md`).

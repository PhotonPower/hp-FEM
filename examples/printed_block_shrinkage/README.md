# Shrinkage of a two-photon printed block on glass

**Physics.** Resin printed by two-photon polymerisation shrinks during development and
post-curing by a few per cent. On a printed optical surface this shrinkage shows up as a shape
error. The model is small-strain linear elasticity with an isotropic eigenstrain −s I
(ADR-0016, milestone M20). The shrinkage s follows from the local degree of polymerisation,
which a process model (Voxelspaetzle, issue #213) delivers as a volume field.

The example is a flat block on a rigid glass substrate:
- **Geometry.** 40 µm × 40 µm × 2 µm, bonded at its base, with free side walls and a free top.
- **Resin.** An assumed IP-Dip-like resin with E = 4.5 GPa and ν = 0.35. These are stand-in
  values, not data.
- **Shrinkage.** A stand-in profile rising linearly from 2 % at the base to 4 % at the top, as
  for a top layer that is less cross-linked.

**What the program does.**
- **Model.** One quarter of the block (`hpfem.Elasticity3D`): the symmetry planes x = 0 and
  y = 0, the base clamped, static condensation.
- **Shrinkage input.** The shrinkage enters as a data-port field, a `GridField3D` that is
  invariant in x and y.
- **Outputs:**
  - the displacement along the top face;
  - the sag between the centre and the corner of the top face;
  - the pull-in of the side wall at the top edge;
  - the strain energy of the whole block.
- **Check.** It first solves a uniform shrinkage of 3 %. In the centre of the wide block it
  compares with the laterally constrained layer, u_z = −s H (1 + ν)/(1 − ν).
- **Files.** `printed_block_shrinkage.json`, and the profile in `printed_block_shrinkage.png`
  (if matplotlib is installed).

**Expected result.** The constrained-layer check gives −124.59 nm against −124.62 nm, a
difference of 2e-4: the walls are 10 heights away. For the graded shrinkage, at p = 3 on
6 × 6 × 3 cells of the quarter (10.8k DoF):
- **Top centre.** It sinks by 124.6 nm. This equals the uniform case with the mean shrinkage
  of 3 %, because the constrained layer sees only the integral of s over the height.
- **Corners and walls.** The corners of the top face lift relative to the centre: the sag is
  −117 nm, the edge curling known from printed structures. The side walls pull in by 163 nm.
- **Energy.** The strain energy is 1.74e4 pJ.

Convergence of the quarter model:

| p | cells | DoF | sag [nm] | wall pull-in [nm] | energy [pJ] |
|---|---|---|---|---|---|
| 2 | 6 × 6 × 4 | 4.6k | −85.2 | −149.6 | 17593 |
| 2 | 10 × 10 × 4 | 11.9k | −108.7 | −159.3 | 17459 |
| 3 | 6 × 6 × 3 | 10.8k | −116.8 | −162.7 | 17404 |
| 2 | 16 × 16 × 4 | 29.4k | −118.2 | −162.7 | 17389 |

The corner value converges slowly, and so does the sag that depends on it. The clamped base
meets the free walls along an edge where the stress is singular, and the top corner lies only
one height above that edge. A mesh graded towards the walls or hp-refinement (M20 S5) would
speed this up. The quick configuration (p = 2, 6 × 6 × 4) is only a smoke test of the pipeline.

**Runtime.** About 35 s for the full run (the check and the graded case at p = 3) and 10 s for
`--quick`, on a laptop with SparseLU.

```bash
python examples/printed_block_shrinkage/run.py [--quick]
```

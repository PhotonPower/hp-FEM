# Ring resonator in 3D between modal ports (M12 stage B)

**Physics.** The add-drop filter of integrated photonics in full 3D: a silicon ring (SOI strip,
450 nm × 220 nm, centre-line radius 2 µm) between two straight bus waveguides of the same
cross-section, gap 200 nm, in silica, around 1550 nm. Light launched into bus A passes the ring
(through port) except near the resonances, where the ring takes it over and hands it to bus B
running backwards (drop port). The 2D effective-index version is `examples/ring_resonator`;
here the full cross-section, the bend and the coupling regions are resolved, which is what
stage A (`examples/directional_coupler_3d`, `docs/validation.md` F) prepared.

**Discretisation.** `ring_mesh.py` builds the Gmsh model (OCC: ring as the difference of two
cylinders, buses as boxes, fragmented with the cladding box) and writes second-order (curved)
tetrahedra, graded from `--cell-core` (100 nm) at the cores to `--cell-far` (300 nm) 600 nm
away; PEC walls 1 µm beyond the ring along the buses, 750 nm beside the buses and 605 nm
below and above the cores. The two end faces are split at y = 0 into four single-mode ports
(1: input, bus A at x = −X; 2: through, bus A at +X; 3: drop, bus B at −X; 4: add, bus B at
+X). The cells within 250 nm of a core get `--order-core`, the others `--order-far`.

**Reference.** The isolated ring is a body of revolution: `hpfem.AxisymmetricCavity` on the
meridian section (axis to the PEC wall at the bus distance, the same cladding margins) gives
the resonance wavelength of the fundamental TE-like mode for each azimuthal order m. For the
default ring, m = 19 resonates at 1557.39 nm (p = 4–5 on 50 nm cells and p = 4 on 25 nm cells
agree to 0.01 nm; the dielectric corners limit the rate), m = 18 at 1605.23 nm and m = 20 at
1512.21 nm (free spectral range 45–48 nm).

**What the program does.** It computes the reference, then the S-parameters of a launch at
port 1 at each wavelength, by default nine points within ±4 nm of the reference resonance and
nine within ±2 FWHM of the fitted resonance. The resonance is the common complex pole of the
through and drop amplitudes, S(λ) = A / (λ − λ_p) + B + C (λ − λ_c) fitted by linear least
squares (`fit_pole`): Re λ_p is the resonance wavelength, 2 |Im λ_p| the FWHM and
Re λ_p / (2 |Im λ_p|) the loaded Q. With PEC walls and real permittivities the ring is
lossless, so the four port powers must sum to one and the drop power reaches one at the
resonance of the symmetric filter. `--record` writes the JSON after every wavelength and
resumes from it; the mesh can be built in another environment
(`python ring_mesh.py --out ring.msh`) and passed with `--mesh`.

**Expected result.** The 3D resonance converges in p and lies near the isolated-ring
resonance, shifted by the coupling to the buses; the production runs (p = 2, 3, 4 on the
100 nm mesh: 0.36 M, 0.83 M and 1.7 M DoFs, MUMPS) are recorded in `benchmarks/results/` and
`docs/validation.md`. The CI case (`--quick`: R = 1.5 µm, p = 1 on 150 nm cells, two
wavelengths) checks the energy balance (to 1e-6) and the port modes only: p = 1 resolves
nothing of the resonator (the port n_eff is 7 % low, and a scan over 1590–1660 nm shows
spurious resonances every few nanometres with back-reflections up to 0.47 instead of one
clean drop peak).

**Runtime.** `--quick` about 30 s (21 k DoFs, SparseLU); the reference about 10 s per
azimuthal order. The production mesh needs a sparse direct solver with much memory (MUMPS on a
server): on a 16 GB workstation SparseLU takes 218 s for p = 2 / 1 on a 150 nm mesh (98 k
DoFs) and runs out of memory at p = 2 throughout.
The timings of the server runs are in their records.

```bash
pip install -e ".[dev]"          # includes gmsh
python examples/ring_resonator_3d/run.py --quick
python examples/ring_resonator_3d/run.py --order-core 3 --order-far 2 --backend mumps \
    --record ring-p3.json
```

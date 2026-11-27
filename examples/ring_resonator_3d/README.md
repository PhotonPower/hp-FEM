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

**Half model** (`--half`, for the mesh and the solver). The cladding is silica above and
below, so the structure is mirror-symmetric about the mid-plane z = h/2 of the cores, and the
TE-like modes launched here have an even tangential electric field about it: the plane is a
magnetic wall. `--half` meshes only z ≥ h/2, leaves the plane natural in the volume problem
and makes the port rims magnetic there (`WaveguidePort.pmc_tags`, `docs/theory/maxwell.md`).
The power-normalised S-parameters are those of the full device. On the production mesh this
saves a third of the unknowns (p = 3: 0.55 M instead of 0.83 M, p = 4: 1.1 M instead of
1.7 M) and more of the factor memory. A local check on the 150 nm mesh at p = 2 / 1 gives the
port n_eff 2.3501 (half) against 2.3530 (full) — the two meshes differ — in 49 s instead of
218 s.

**Quarter model** (`--quarter` with `--half`, for the mesh and the solver). The device is
also mirror-symmetric about x = 0 (ports 1 ↔ 2, 3 ↔ 4) and y = 0 (1 ↔ 3, 2 ↔ 4). The launch at
port 1 is the mean of the four mirror parities (s_x, s_y) = (±1, ±1), each a one-port problem
on the quarter x, y ≤ 0: the plane of an even parity (+1) is a magnetic wall (natural), that of
an odd one (−1) a PEC wall; the port rim on y = 0 between the buses stays PEC for both, as the
split end faces of the full model. With the four reflection coefficients r of port 1 the
S-parameters of the full device are s_1 = ⟨r⟩, s_2 = ⟨s_x r⟩, s_3 = ⟨s_y r⟩ and
s_4 = ⟨s_x s_y r⟩ (⟨·⟩ the mean over the parities); the amplitudes of ports 3 and 4 carry a
fixed sign against the full model, the powers are the same. A quarter has a quarter of the
unknowns, and its sparse factor grows faster than linearly in them, at four factorisations
per wavelength. The decomposition is exact: on the quick geometry the quarter mesh mirrored into
the half model (`mirror_quarter`, the same discretisation) gives the same port powers to
3.5e-12 (`python/tests/test_examples.py`). The production quarter-half mesh (100 nm cells)
has 9041 cells; p = 2 / 2 gives 63.6 k DoFs per parity, p = 3 / 2 about 140–150 k (against
0.55 M for the half model).

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
On an 8 GB laptop the quarter-half model at p = 2 / 2 (63.6 k DoFs) takes 5–9 min per
wavelength for the four parities with SparseLU at a peak of 4.7 GB; p = 3 / 2 (about 12–15 GB
with SparseLU) does not fit there. The timings of the server runs are in their records.

**Local result at p = 2 / 2** (quarter-half model, 19 wavelengths in 1.8 h,
`benchmarks/results/2026-11-27-validation-ring3d-quarter-p2.json`). The four port powers sum to
one to six digits at every wavelength (lossless), but p = 2 does not resolve the resonator yet:
instead of one drop peak near the isolated-ring resonance (1557.39 nm) the 1 nm scan shows
small, scattered rises of the drop power (1.4 % at 1555.4 nm, 3.6 % at 1558.4 nm, the latter
with 2.8 % back-reflection — the coupling of the two senses of circulation by the discretisation
seen at p = 1 in a stronger form). The pole fit then lands at 1556.12 nm (Q 9900), outside the
refined points, and the program flags it as an extrapolation. The resonance needs p = 3 / 2 at
least (server).

```bash
pip install -e ".[dev]"          # includes gmsh
python examples/ring_resonator_3d/run.py --quick
python examples/ring_resonator_3d/run.py --order-core 3 --order-far 2 --backend mumps \
    --record ring-p3.json
# the quarter-half model: four one-port solves per wavelength, a quarter of the unknowns
python examples/ring_resonator_3d/ring_mesh.py --half --quarter --out ring-hq.msh
python examples/ring_resonator_3d/run.py --half --quarter --mesh ring-hq.msh \
    --order-core 3 --order-far 2 --backend mumps --record ring-hq-p3.json
```

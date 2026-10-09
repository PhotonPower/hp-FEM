# Dipole emitters in periodic structures (milestone M17)

Status: proposal of 9 October 2026 from the GUI work (FEM model builder, mode "periodic
structure"), reviewed by the coordinating dev agent the same day (review notes marked
**Review**). Audience: the development agents. Tracked in `docs/roadmap.md` as milestone M17;
the IDs S0 to S4 below are used there. Decision record: ADR-0013 (S0). Statements marked
*hypothesis* are not verified; everything else was checked against `main`.

## 1. The gap

1. `docs/theory/maxwell.md` (end of the conical section) lists as *planned*: "Purcell factor
   F_P = P_emitted / P_bulk for a point dipole". For bodies of revolution it exists (M11,
   `examples/micropillar_qd`), for 2D non-periodic problems as a line current
   (`examples/quantum_dot_purcell`); for periodic cells there is no front end.
2. `ConicalScatteringSetup.current` already accepts a volume source f = iωμ0 J in the scaled
   components (f_x, f_y, −i f_z) (total-field formulation), with Bloch pairs, PML and the layered
   background. A current in a Bloch-periodic cell is, however, physically **an infinite array of
   coherent line sources** with the phase e^{i kx P n} from cell to cell and the dependence
   e^{iβz} along the lines, not one emitter. Users who put a "dipole" into the cell get the
   emission of that array.
3. The quantities users ask for (LEDs, OLEDs, single-photon sources on gratings,
   metasurface-enhanced emission, quantum dots in photonic-crystal waveguides, plasmonic
   gratings) are those of a **single** emitter: total decay-rate enhancement (Purcell factor),
   radiated fraction into the cover / substrate per direction or diffraction order, power into
   guided modes of the layer stack / grating, absorbed power (quenching).

## 2. Physics and conventions

Frame of the conical solver: x along the period P, y normal to the layers, z along the lines.
Time dependence e^{−iωt}. A point dipole with current moment **p** (A·m) at **x0** = (x0, y0, 0):
J = p δ(x − x0) δ(y − y0) δ(z).

**Array scanning (Floquet–Bloch superposition).** With E_{kx,β}(x, y) e^{iβz} the solution of the
cell problem whose source is the Bloch-periodic array J_{kx,β} = p δ(x − x0) δ(y − y0) (one source
per cell, phase e^{i kx P n}) and longitudinal wavenumber β,

    E(x, y, z) = (P / 2π) ∫_{−π/P}^{π/P} dkx  (1 / 2π) ∫_{−∞}^{∞} dβ  E_{kx,β}(x, y) e^{iβz}

(the Poisson sum gives (P/2π) ∫_BZ Σ_n δ(x − x0 − nP) e^{i kx nP} dkx = δ(x − x0), and
δ(z) = (1/2π) ∫ e^{iβz} dβ). The emitted power of the single dipole is P_em = −½ Re[p* · E(x0)],

    P_em = (P / 2π) ∫_BZ dkx (1/2π) ∫ dβ  P_cell(kx, β),     P_cell = −½ Re[p* · E_{kx,β}(x0)],

and the Purcell factor F_P = P_em / P_bulk with P_bulk the power of the same (smeared) dipole in
the homogeneous medium at x0 (`dipole_vacuum_power` × n × e^{−(n k0 σ)²} for the Gaussian
smearing, as in M11). Equivalently, and better conditioned, P_cell is the Poynting flux of
E_{kx,β} through a closed surface around the source cells, which avoids the self-field at x0.

**Decomposition.** For every (kx, β) the conical post-processing gives the power through the
cover line and the substrate line per diffraction order (`conical_diffraction_orders`,
`conical_power_balance`) and the absorbed power (`absorbed_power_by_tag`). Integrated over
(kx, β) inside the light cones these are the radiated powers into the cover and the substrate,
resolved by direction (kx + 2πm/P, β) ↔ (θ, φ); the rest of P_em is guided (poles on the real
β / kx axes outside the light cones) or absorbed.

**Singularities.** For lossless guided modes of the grating / stack the integrand has poles on
the real (kx, β) integration domain (for every kx a set of β_g(kx) > k n_max-clad), and
square-root branch points at the light lines β² + (kx + 2πm/P)² = (k n_c)², (k n_s)². This is the
hard part of the milestone (Section 3.2).

**Review — symmetries.** For a dipole moment along x, y or z the integrand is even in β (the
structure is invariant and mirror-symmetric in z, the power quadratic in p); for a cell that is
mirror-symmetric about the dipole's x0 it is even in kx as well. The β integral therefore runs
over [0, ∞) and, with a symmetric cell, the kx integral over half the zone. An isotropic emitter
is the mean of the three orientations, whose cross terms vanish by the same symmetry.

## 3. Proposal (three stages, each useful on its own)

### 3.1 Stage A: Bloch-periodic emitter array (cell problem with a dipole source) — S1

A front end for the cell problem that exists in principle today:

```python
hpfem.conical_gaussian_dipole(position, moment, sigma, omega, beta)
# f = i omega mu0 J of a Gaussian-smeared dipole in the 2D cell, scaled components
# (f_x, f_y, -i f_z), moment 3-vector [A m]
res = hpfem.grating.emit(mesh, materials, stack, dipole=dict(position=(x0, y0), moment=(px, py, pz), sigma=...),
                         omega=..., kx=..., beta=..., order=4, pml=..., bottom="pml", orders_max=3,
                         progress=..., cancel=..., keep_factorisation=False)
# EmissionResult: P_cell (power per cell and per unit beta, flux through the closed surface
#   around the source), orders_up / orders_down: list of Order with power per order,
#   A_by_tag (absorbed), flux_balance, guided = P_cell - up - down - absorbed,
#   field(points, quantity), timing, diagnostics
```

- Source in the total-field formulation (`ConicalScatteringSetup.current`), background stack
  only for the PML design and the measurement lines (no incident wave).
- The closed surface around the source: a tagged box of cells around x0 and the existing
  `Surface2D.around_cells` (as the GUI does for bodies of revolution). **Review:** a helper
  `around_point` does not exist; the tagged box is the route (a mesh generator option for the
  box, σ-scaled, as for `axisymmetric_gaussian_dipole`).
- `kx`, `beta` real in this stage. Stage A alone answers "emission of a phased emitter array"
  and is the building block of B and C.
- **Review — three orientations, one factorisation.** The matrix of the cell problem does not
  depend on the moment: the three orientations are three right-hand sides of one factorised
  system (`solve_many`, the kept factorisation of M16 S1). An isotropic emitter costs one
  factorisation per (kx, β), not three.

### 3.2 Stage C before B: angle-resolved emission by reciprocity — S2

**Review — order.** Stage C (Section 3.4 below) is cheap, free of singularities and gives what
LED and OLED users mostly need (the far-field pattern and the extraction into an aperture). It
comes right after A (roadmap S2), B follows (S3) and C then cross-checks B's radiated part.

### 3.3 Stage B: single dipole by array scanning — S3

```python
res = hpfem.grating.dipole_emission(mesh, materials, stack, dipole, omega, *, kx_points=..., beta_rule=..., tolerance=...,
                                    order=4, processes=None, progress=None, cancel=None)
# DipoleResult: purcell, P_em, P_bulk, radiated_up, radiated_down, guided, absorbed (each also
#   / P_bulk), angular emission dP/dOmega in cover and substrate on a (theta, phi) grid,
#   contributions per (kx, beta) sample, convergence estimate (difference of two quadrature
#   levels), number of cell solves, timing
```

Requirements and design questions for ADR-0013:

1. **kx integral** over the Brillouin zone: smooth and periodic in kx except at guided-mode poles
   and light-line branch points; Gauss–Legendre or trapezoid on sub-intervals between the
   light-line crossings; the kx ↔ −kx symmetry of a mirror-symmetric cell halves the work.
2. **β integral** over [0, ∞) (even integrand): the integrand decays for |β| ≫ k n_max (evanescent
   coupling only, decay length ~ distance d of the dipole to the nearest interface); the cut-off
   must be chosen from d and the smearing σ. **Review:** the cut-off grows like 1/d, so a
   dipole very close to an interface drives the number of solves; the cost estimate must show it.
3. **Poles of lossless guided modes** on the real axis. Options (decision in ADR-0013):
   (a) **contour deformation** into complex β (and/or complex kx). **Review:** this needs complex
       β in the conical forms (`assembly/conical_forms.hpp` takes `Real beta` today; the
       *hypothesis* that the assembly is linear in β and β² is plausible) and a PML design and
       `layered_conical_wave` for complex β — a C++ change with its own risks;
   (b) **singularity subtraction** with the guided modes, the remainder integrated on the real
       axis; gives the guided power per mode directly (the β-factor into a waveguide mode).
       **Review:** the guided modes are needed as β_g(kx) at fixed ω, which neither
       `ConicalResonance` nor `grating.bands` give directly (they give ω(kx, β));
       `PropagatingMode` gives β at fixed ω but has no Bloch constraints. Two routes: Newton on
       the bands, β ← β − (ω(kx, β) − ω0) / (∂ω/∂β) with the group velocity of M16 S4
       (`hpfem.group_velocity`, band derivatives), or a Bloch-periodic propagating-mode
       eigenproblem (new). The residue follows from the mode field at x0 and the group velocity;
   (c) a small artificial loss with extrapolation to zero loss (simplest, least accurate, only as
       a fallback and as a cross-check).
   Recommendation: (b) by Newton on the bands for the guided part plus adaptive quadrature on the
   real axis for the radiative part; (a) as a later option once the forms support complex β.
4. **Cost.** Every sample is one factorisation (three orientations as right-hand sides, see
   3.1); the matrix depends on (kx, β) only through the Bloch phase and β, so the symbolic
   analysis is reused (`ConicalSweep`, `LinearSolver::refactorize`, M15 F8) and samples are
   independent (`hpfem.sweep.solve_sweep` over processes). The front end reports the number of
   solves and the memory before running (`estimate_memory` × samples).
5. **Mesh.** A single mesh for all samples (the PML designed for the largest angle of the sampled
   (kx, β), `PmlProfile.for_angle`); a dipole close to a metal needs local refinement around x0.

### 3.4 Stage C: angle-resolved emission by reciprocity (roadmap S2)

For the far-field emission of a single dipole in a given direction (θ, φ) and polarisation,
reciprocity gives

    dP/dΩ (θ, φ, pol) ∝ |p · E_pw(x0; θ, φ, pol)|²,

with E_pw the total field at x0 of the plane wave incident from that direction (`grating.solve`,
sampled with `problem.sample`). One solve per direction, no singularities, exact for the radiated
part; it does not give the total Purcell factor, the guided or the absorbed power. Proposed helper:

```python
hpfem.grating.emission_pattern(mesh, materials, stack, dipole, omega, directions=[(theta, phi, side), ...],
                               pol=("s", "p"), **solve_kwargs)
# dP/dOmega per direction and polarisation, normalised to P_bulk (absolute normalisation from the
# reciprocity constant)
```

Stage C also cross-checks B: the radiated part of B integrated over the light cone must equal the
integral of C over the hemispheres.

## 4. Tests (Definition of Done, CLAUDE.md §7)

1. **Homogeneous cell** (no structure, no stack interfaces): B gives F_P = 1 within the quadrature
   tolerance for dipoles along x, y, z; A at β = 0 compared with the 2D Larmor power of the line
   source.
2. **Dipole above a perfect mirror / PEC** (flat): image-dipole closed form for parallel and
   perpendicular dipoles as a function of the distance (oscillating F_P).
3. **Planar multilayer** (stack only, the grating removed): Chance–Prock–Silbey /
   Sommerfeld-integral reference (one-dimensional integral over the in-plane wavenumber,
   independent code in the test with SciPy): Purcell factor, radiated up/down and guided power of
   a dipole in a slab waveguide (guided-mode poles present, so this tests 3.3 (3)).
4. **Period independence**: a flat stack meshed with periods P and 2P gives the same single-dipole
   result (array scanning removes the artificial periodicity).
5. **Reciprocity**: Stage C against the radiated far field of Stage B for a lamellar grating.
6. **Line-source limit**: the β integral replaced by β = 0 reproduces the 2D line-source result of
   `Scattering<2>` with `gaussian_current` on the same structure.
7. **Plasmonic quenching**: dipole near a flat silver film, total vs radiated F_P against the
   Sommerfeld reference (the loss moves the surface-plasmon pole off the real axis: a regular but
   sharply peaked integrand, a test of the adaptive quadrature).
8. Regression: a lamellar grating case with a stored reference (Purcell factor, up/down
   fractions), quick variant in CI.

## 5. What the GUI needs (FEM model builder)

- A job-runner task `emitter` (schema version 2, shared with the M16 S9 tasks) with the dipole
  (position in the cell, moment as orientation or the three orientations averaged = isotropic
  emitter), the wavelength sweep and the stage (A, B or C); events per (kx, β) sample with
  `progress` and cancellation between samples; the cost estimate (number of solves × memory)
  before the run.
- Results as plain numbers and arrays: F_P, the fractions up / down / guided / absorbed, dP/dΩ on
  a (θ, φ) grid for cover and substrate (polar plots, extraction efficiency into a numerical
  aperture), the guided power per mode with 3.3 (3b), a convergence measure, and the field of the
  single dipole on the cell or on several periods (`field(points)` of the superposition).
- Same conventions as M11: σ-smeared dipole, P_bulk with the refractive index at the emitter, F_P
  normalised to it; the emitter must lie in a lossless medium (or a documented definition for
  lossy hosts).

## 6. Out of scope (first step)

- Two-dimensionally periodic structures (metasurfaces, photonic-crystal slabs) with the 3D solver:
  the same array scanning over a 2D Brillouin zone (no β integral), as a later stage.
- Incoherent ensembles of emitters distributed over the cell (an average over positions of B; the
  GUI can do the averaging).
- Non-local / quantum corrections for emitters closer than ~1 nm to metals.
- Time-domain or broadband excitation; the frequency sweep stays a loop over ω (with the sweep
  acceleration of M15 F8).

## 7. Context in the GUI

The GUI's mode "body of revolution" implements the analogous workflow on top of M11: emitter box
tagged in the Gmsh mesh (tag offset), `Surface2D.around_cells` for the emitted power, Poynting flux
through measurement planes for β, Purcell factor against the smeared bulk dipole (homogeneous
check F_P = 1.00, micropillar of `examples/micropillar_qd` reproduced to 1 % at p = 3), modal
decomposition with `AxisymmetricRieszProjection`. The same structure of inputs and outputs is
wanted for the periodic case, so that both modes of the GUI present emitters alike.

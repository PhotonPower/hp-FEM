# 0013 — Dipole emitters in periodic structures by array scanning of the conical cell problem

**Status:** accepted; §3a amendment accepted (2026-10-10)
**Date:** 2026-10-09

## Context

Milestone M17 (`docs/dipole-emitters-features.md`, items S0–S4) asks for a single point dipole
in a structure that is periodic in x (period P) and invariant in z: Purcell factor, radiated
power into cover and substrate per direction, guided and absorbed power. The conical solver
(`physics::ConicalScattering`) solves the cell problem for one Bloch wavenumber kx (Bloch pairs)
and one longitudinal wavenumber β (fields ∝ e^{iβz}), with PML, the layered background and a
volume source `ConicalScatteringSetup::current` in the total-field formulation. A source in the
cell is an infinite, phased array of line sources, not one emitter. Facts from the code (`main`
after M16 S7):

- `assembly/conical_forms.hpp` takes `Real beta`; complex β is not supported.
- `ConicalResonance` and `grating.bands` give ω(kx, β); M16 S4 adds band derivatives and the group
  velocity (`hpfem.group_velocity`). `PropagatingMode` gives β at fixed ω but without Bloch
  constraints.
- The M11 emitter (`axisymmetric_gaussian_dipole`) is a dipole smeared over the normalised 3D
  Gaussian `g = exp(−|x − x0|²/2σ²)/((2π)^{3/2}σ³)`, whose free-space power is
  `dipole_vacuum_power · n · e^{−(n k0 σ)²}` exactly in a lossless medium.
- The matrix of the cell problem depends on (kx, β) only through the Bloch phase and β; the
  symbolic analysis is reused by `LinearSolver::refactorize` (`ConicalSweep`, M15 F8); the kept
  factorisation (M16 S1) solves several right-hand sides at once.

## Decision

### 1. Source and normalisation

The emitter is the **3D Gaussian dipole of M11**, J = p g(x − x0). Array scanning splits it into
cell problems with the source

```
J_{kx,β}(x, y) = p g2(x − x0, y − y0) e^{−σ²β²/2},      g2 = exp(−ρ²/2σ²)/(2πσ²),
```

one per cell with the Bloch phase e^{i kx P} (`conical_gaussian_dipole(position, moment, sigma,
omega, beta)` returns f = iωμ0 J in the scaled components (f_x, f_y, −i f_z)): the z-smearing is
the Fourier factor e^{−σ²β²/2} of the β sample. The single dipole is then exactly the M11
source, its bulk power `P_bulk = dipole_vacuum_power · n · e^{−(n k0 σ)²}` is known in closed
form, F_P = P_em / P_bulk follows the M11 convention, and the factor bounds the β integrand
independently of the structure. The Gaussian must lie inside the cell (6σ from the Bloch faces,
checked) and in a lossless medium (as M11).

### 2. Power of a sample

The primary quantity of a sample is the **power delivered by the source**,
`P_cell(kx, β) = −½ Re ∫_cell J*_{kx,β} · E_{kx,β} dA` (a volume integral over the cells under the
Gaussian; finite and discretely consistent thanks to the smearing; no tagged box needed). The
decomposition of a sample uses the existing post-processing: the power per diffraction order
through the cover and the substrate lines (`conical_diffraction_orders`), the absorbed power per
tag (`absorbed_power_by_tag`), and the guided remainder `P_cell − up − down − absorbed`; their sum
against `P_cell` (`flux_balance`) is the per-sample check. A closed Poynting surface around a
tagged source box (`Surface2D.around_cells`) is an optional diagnostic, not required.

### 3. Array scanning and quadrature

```
P_em = (P / 2π) ∫_BZ dkx (1/π) ∫_0^∞ dβ P_cell(kx, β)
```

for a moment along x, y or z (the integrand is even in β; an isotropic emitter is the mean of
the three orientations, whose cross terms vanish by the z-mirror symmetry). For a cell that is
mirror-symmetric about x0 the kx integral runs over half the zone. Quadrature:

- β outer, kx inner. For fixed β the light-line crossings `kx = −2πm/P ± (k²n_c,s² − β²)^{1/2}`
  are explicit and split the zone into panels; Gauss–Legendre panels of two levels give the error
  estimate (the difference of the levels), refined where it exceeds the tolerance.
- β: panels split at k n_c, k n_s and the guided-mode positions (§4); cut-off where both the
  Gaussian factor e^{−σ²β²} and the evanescent decay `exp(−2 d (β² − k²n²)^{1/2})` (d the distance
  of x0 to the nearest interface with a higher index) fall below the tolerance. The cut-off grows
  like 1/d and is reported in the cost estimate (§5).

#### 3a. Amendment (2026-10-10, accepted): the kx integral on a complex contour

On the real kx axis the integrand has integrable 1/k_y singularities at the light-line crossings.
There the field consists of grazing orders, which a PML of finite thickness does not absorb, so
the FEM integrand near every crossing is wrong by far more than the discretisation error.
Measured, homogeneous cell, y dipole:

- the β slice at 0.245 k: −0.64 % with the 80° PML of #156, −7 … −11 % before #156;
- the full scan: F_P (y) = 0.979, and more nodes do not make it converge.

**Decision.** The kx integral of the delivered power runs on a contour in the complex plane:

- It leaves the real axis around every light-line crossing. Each crossing gets a bump
  `kx = t − i δ w cos²(πu/2)`, with u = (t − c)/w, w half the distance to the neighbouring
  crossings, and δ ≈ 0.5.
- The bump passes below the crossings `c = −2πm/P + q` and above `c = −2πm/P − q`, where
  q = (k²n² − β²)^{1/2} for the cover and for a lossless substrate. This is the side to which a
  small loss moves the branch points for exp(−iωt).
- Between the crossings the contour is the real axis, and the panels and the β quadrature of §3
  stay as they are.
- The integrand is pᴴ A(kx, β) p, with the power matrix
  A_ij = −½ ∫ g₂ e^{−σ²β²/2} (E_j)_i dA. Neither the load nor A contains a conjugate, so A is
  analytic in kx, and P_em is the real part of the contour integral.

**This needs three things:**

1. *An analytic Bloch elimination.* With the reduction Pᴴ A P, the test functions carry the phase
   conj(λ), λ = e^{i kx P}, and conj(λ) is not analytic once |λ| ≠ 1.
   - The fix: a separate test space Q with the phases 1/conj(λ) (`fespace::Constraints::set_test`).
     The reduction becomes Qᴴ A P, and Qᴴ then holds 1/λ.
   - On the unit circle Q = P, so real kx is unchanged.
   - `ConicalScattering` builds Q when a phase lies off the unit circle. The kept factorisation
     reduces with Qᴴ, and its adjoint expands with conj(Q).
2. *One PML profile per β for all kx samples.* `grating.emit` designs the profile from the
   direction of (Re kx, β), which makes the integrand depend on Re kx alone, so it is not
   analytic. The scan uses one profile, designed for the 80° cap.
3. *A complex Bloch phase in the cell problem.* `dipole_emission` sets up the cell problem
   itself, with the phase e^{i kx P} for a complex kx. `emit`, whose order post-processing is
   for real kx only, stays as it is.

**Scope.**

- The contour gives the total delivered power and the Purcell factor.
- The channel powers (flux per order, absorption) are quadratic in the field and not analytic.
  They are not integrated on the contour. Instead:
  - up and down come from the reciprocity pattern (§6), integrated over each half-space. The
    grazing directions are harmless there: the plane-wave problems carry the grazing wave in
    the analytic background, and the PML only has to absorb the field scattered by the
    structure;
  - the non-radiated part P_em − up − down is the absorbed power in a lossy structure and the
    guided power in a lossless one (§4).
- The real-axis rule (`depth = 0`) remains as a diagnostic.
- Guided-mode poles on the real axis (§4) are unaffected: the contour returns to the real axis
  between the crossings.

**Verification** (homogeneous cell, y dipole, β = 0.245 k, p = 4, 6 nodes per panel):

- Cauchy–Riemann at a real and at a complex kx: the defect falls like h² (3.5e-3, 3.9e-4, 3.5e-5
  for h = 3e-2, 1e-2, 3e-3 π/P).
- The slice integral against the closed form: −0.0750 % at δ = 0.5 and −0.0751 % at δ = 1.0.
  The result does not depend on the path to 5e-7. The −0.075 % is the discretisation error.
- With the Pᴴ elimination the same detour gave +9.9 % and +19.4 %, which was the failure that
  led to this amendment.

**Found during the implementation (M17 S3).** These refine the contour and the channels; the
decision itself is unchanged.

- *Bump height.* For β above k0·n a medium is evanescent, and its branch points are complex:
  2πm/P ± iκ, with κ = (β² − k0²n²)^{1/2}. The PML replaces their cuts by strings of poles of
  the truncated cell problem, and those strings run down towards the real axis.
  - The failure case: a dipole 400 nm above glass, period 0.8 µm, β = 1.013 k0. The slice
    changed by a factor of 0.6 to 5 with the depth, and F_P was 3–5 % off.
  - The fix: the height of a bump is capped at 0.05 times its distance to the nearest complex
    branch point.
- *kx window and panels.* The cell problem depends on kx only through e^{i kx P}. Without
  symmetry, the window is therefore the period that starts in the middle of the widest gap
  between crossings. The real stretches between bumps get panels that grow geometrically away
  from each bump.
- *Node counts.* β needs more nodes than kx: the factor e^{2ik_y d} of a dipole at a distance d
  from an interface oscillates in β. There are separate counts, `nodes` for β and `kx_nodes`
  for kx. For a dipole 400 nm above glass at 1 µm, the rule integrates the Fresnel spectrum to
  5e-4 with 6/4 and to 2e-4 with 8/4.
- *Mirror symmetries.* The rule uses β ≥ 0 and, for a symmetric cell, half the zone. The matrix
  entries that are odd under these symmetries are set to zero after the integration (xz and yz
  always, xy for a symmetric cell). Otherwise a moment mixing the axes would double them instead
  of cancelling them.
- *Channels.* The θ quadrature of a half-space is split at its critical angles, with the points
  crowded towards them, because the forbidden light makes a square-root kink there. The
  material map is completed cell by cell before the substrate problem is mirrored.
- *Validation* (`benchmarks/m17_dipole_scan.py`, p = 3, 8/4 points). The Purcell factors of a
  dipole 400 nm above glass agree with the Sommerfeld integral to 8e-5 for a period of 1 µm and
  to 1e-3 for 0.8 µm, and up and down to 1e-5.

### 4. Guided-mode poles: subtraction with the modes from the bands

For a lossless structure the integrand has poles on the real axes at the guided modes
β_g(kx) (and their kx images). They are **subtracted with the modes**:

- β_g(kx) at the fixed ω0 by Newton on the bands, `β ← β − (ω(kx, β) − ω0) / v_g`, with ω and the
  group velocity `v_g = ∂ω/∂β` of the conical eigenproblem (M16 S4), started from the peaks of a
  coarse β scan of P_cell; the mode field at x0 from the eigenvector.
- The excitation amplitude of a mode by the dipole follows from Lorentz reciprocity with the
  counter-propagating mode, normalised by the cross-section flux integral over the cell; the
  guided power per mode and per kx is `P_g(kx) = |a_g|² P_mode`, without sampling the pole.
- The remainder (P_cell minus the pole terms of the modes found) is smooth and integrated on the
  real axis; its panels avoid the pole by a symmetric (principal-value) split.

The guided power per mode is an output (the β-factor into a waveguide mode). **Fallback and
check:** a small artificial loss ε'' → 0 with Richardson extrapolation over three loss values,
used in the tests to cross-check the subtraction. **Deferred:** contour deformation into complex
β (needs complex β in the conical forms, a PML design and `layered_conical_wave` for complex β;
a separate decision once the assembly supports it).

### 5. Cost and execution

- One factorisation per (kx, β) sample; the three orientations are three right-hand sides of it
  (`solve_many` on the kept factorisation), the symbolic analysis is reused across samples
  (`refactorize`), samples are independent (`hpfem.sweep.solve_sweep` over processes, each with a
  thread budget as ADR-0012 §7).
- Before the run, the front end reports the number of samples of the first quadrature level, the
  memory per factorisation (`estimate_memory`) and the extrapolated time; `cancel` is polled
  between samples, `progress` reports per sample.
- One mesh for all samples: PML designed for the largest angle of the sampled (kx, β)
  (`PmlProfile.for_angle`), local refinement around x0 scaled with σ.

### 6. Stage C: reciprocity

`emission_pattern` computes dP/dΩ(θ, φ, pol) from the total field at x0 of the plane wave
incident from (θ, φ) with polarisation pol, `dP/dΩ = C |p · E_pw(x0)|²` (one `grating.solve` per
direction, Gaussian-weighted field at x0). The constant C (per medium of the direction) is fixed
by reciprocity and checked by the homogeneous case, where the pattern must integrate to
`P_bulk`. Stage C gives the radiated part only; Stage B's radiated part must agree with its
integral over the hemispheres.

### 7. Interfaces

`conical_gaussian_dipole` (C++ and Python), `grating.emit` (Stage A, one sample),
`grating.emission_pattern` (Stage C), `grating.dipole_emission` (Stage B) with the results of
`docs/dipole-emitters-features.md` §3; the job-runner task `emitter` in schema version 2 (shared
with the M16 S9 tasks).

## Consequences

- The single dipole is the M11 emitter: Purcell factors of the periodic and the axisymmetric
  front ends are directly comparable, and P_bulk needs no numerics.
- No C++ change for complex β now; the price is the mode subtraction, which needs the bands near
  ω0 for every kx panel (a few eigen-solves per kx, small against the β samples).
- The mode amplitude via reciprocity needs the backward mode (−β): for a z-mirror-symmetric
  structure it is the mirror image of the forward mode, no extra solve.
- Cost: N_kx × N_β factorisations per frequency; for a dipole close to an interface N_β grows
  like 1/d. The cost estimate makes this visible before the run.
- Lossy hosts are excluded (as M11); emitters in lossy media need a definition of P_bulk first.

## Alternatives considered

- **Contour deformation into complex β** — avoids the poles altogether and is standard in
  Sommerfeld-integral codes, but needs complex β throughout the conical assembly, the PML design
  and the layered background wave; deferred.
- **Artificial loss with extrapolation only** — simple, but slow convergence near the poles and
  no guided power per mode; kept as a check.
- **3D supercell with a finite emitter** — no Bloch decomposition, but the cell must be large
  against the decay length of the guided modes, and the PML in z breaks the invariance; far more
  expensive.
- **Line source (β = 0 only)** — the 2D emitter of `examples/quantum_dot_purcell`; not a point
  dipole (it is the β = 0 slice of the integrand and serves as a limit test).
- **Point dipole without smearing** — the delivered power needs the flux through a closed surface
  and the self-field is singular on the mesh; the Gaussian keeps the M11 convention and the
  volume integral consistent.

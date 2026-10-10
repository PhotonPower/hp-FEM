# 0013 — Dipole emitters in periodic structures by array scanning of the conical cell problem

**Status:** accepted
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

# 0015 — Optically anisotropic materials (substrates, layers, structured parts)

**Status:** proposed
**Date:** 2026-11-01

## Context

Semiconductor metrology and device optics need optically anisotropic media in substrates, in
unstructured layers and in structured parts: sapphire, SiC, GaN, LiNbO₃/LiTaO₃, TiO₂ (uniaxial),
biaxial films, crystals cut off-axis (the optic axis rotated against the interface and against the
grating lines), and, as a weak effect, strained silicon. Cross-polarised reflection
(r_sp, r_ps) and Mueller-matrix scatterometry are the quantities of interest.

Facts from the code (`main` after M17 S3):

- The assembly is tensor-ready: `assembly::MaxwellForm<Dim>` carries complex tensors ε(x) and
  μ⁻¹(x); the PML is already implemented as an anisotropic material (ADR-0005, validated by
  `tests/convergence/pml_reflection.cpp`); `ConicalForm` and the axisymmetric forms carry
  **diagonal** tensors only.
- `materials::Material` is isotropic: scalar `eps_r`, `mu_r` (`include/hpfem/materials/material.hpp`
  names anisotropy as "later"). `Scattering::form_of_cell` builds `ε_r·I`, `μ_r⁻¹·I`.
  `eps_r` appears in 51 files of `src/`, `include/`, `python/` (203 hits), `mu_r` in 37.
- `physics::LayerStack` (ADR-0009) assumes isotropic layers (scalar k_z per layer, s/p
  polarisations); `docs/axisymmetric-layered-features.md` lists anisotropic layers as out of
  scope. `diffraction`, `farfield`, `waveguide_port`, `propagating_mode`, `mie`, `absorption` and
  the dipole scan (ADR-0013) use scalar indices.
- The direct solvers choose their path by `detect_symmetry` (ADR-0008): complex-symmetric systems
  take the LDLᵀ paths, non-symmetric material tensors the general LU.
- `hpfem.materials` (`Dispersive`: `Constant`, `Tabulated`, `Sellmeier`, `DrudeLorentz`) returns
  scalars.

## Decision

### 1. Material model

`materials::Material` keeps `eps_r`, `mu_r` as the isotropic description and gains an optional
tensor part (`std::shared_ptr<const Tensors>` with complex 3×3 `eps`, `mu`), so that isotropic
materials keep their size and speed. `is_isotropic()`, `eps_tensor()`, `mu_tensor()` (return
`eps_r·I` for isotropic) and `mu_inv_tensor()` are the tensor-aware accessors. **Scalar access to
an anisotropic material throws** (`eps_scalar()`, used by every path not yet migrated): an
unsupported feature fails loudly instead of computing with a wrong scalar. Every such path also
reports a `hpfem.diagnostics` entry `anisotropic-unsupported` with the feature name before it
fails. The tensors are given in the **laboratory frame** (x, y along the lattice, z the layer
normal); the crystal frame and its rotation are a Python concern (§6).

### 2. Volume solvers (Scattering 2D/3D)

`form_of_cell` returns tensors; with a layered background the source is
`k0² (ε_c(x) − ε_stack(z_c)) E_inc` as a matrix product, so it still vanishes where the structure
equals the stack, in particular inside the PML. The PML stretches by
`ε̃_ij = ε_ij · s_x s_y s_z / (s_i s_j)` (and `μ̃` likewise); the diagnostics warn for media that
carry **backward waves** along a stretching direction (phase and group velocity opposed, e.g.
hyperbolic media), where a PML can amplify instead of absorb. Residual and goal-oriented
estimators use `ε` in the divergence term and the `n·εE` jump. Per-cell overrides
(`MaterialMap::set_cell`) accept anisotropic materials, which also gives position-dependent
tensors (strain, graded axes) and a path for `ThermoOptical`.

### 3. Solver symmetry

A material with ε = εᵀ (all non-gyrotropic crystals, also rotated by a real rotation and
absorbing) keeps the curl–curl system complex-symmetric; `detect_symmetry` already routes such
systems to LDLᵀ (MUMPS, cuDSS). Gyrotropic and optically active media (ε_ij = −ε_ji parts) are
**out of scope of the first step** (they break reciprocity, which the far field by reciprocity
relies on); they would use the general LU through the same detection.

### 4. Conical (grating) solver

`ConicalForm` goes from diagonal to full 3×3 tensors. With full ε the coupling terms ε_xz, ε_yz
mix the β terms and the E_z pairing of the formulation (M13): the form and its interface
conditions (continuity of E_t and of (εE)_z) must be re-derived for the Nédélec × H1 pair before
coding; the derivation goes into the conical section of `docs/theory/maxwell.md` and is checked against the 3D
solver (§9, item 5). The conical estimators follow §2.

### 5. Anisotropic layer stack

`LayerStack` layers and the substrate may carry anisotropic materials; the incidence medium stays
isotropic and lossless (as today). Per layer the in-plane wave vector (k_x, k_y) fixes a 4×4
system for the tangential fields (Berreman form); its four eigenmodes (two up, two down) are
sorted by decay and, for propagating modes, by the sign of the Poynting flux. Reflection and
transmission follow from a **numerically stable S-matrix recursion** from the substrate upwards
(layer-local amplitudes, decaying exponentials only, as in ADR-0009 §2) — no transfer-matrix
products through thick or absorbing layers. Outputs: the **Jones matrix** (r_ss, r_sp, r_ps,
r_pp, and the transmission analogues), the mode amplitudes and the analytic field (value and
curl) per layer for the scattered-field source. Degenerate eigenvalues (uniaxial axis along z,
normal incidence) are treated analytically for the uniaxial case (ordinary / extraordinary
closed forms) and by a symmetric-pair projection in the general case; the tests cover them.

### 6. Python layer and data

`hpfem.materials.Anisotropic` combines three principal `Dispersive` models (or the uniaxial
shortcut `Uniaxial(n_o, n_e)`) with orientation by Euler angles (crystal → laboratory frame,
documented convention with a figure), `.at(omega)` returns the tensor `Material`. The tensor is
`R diag(ε_x, ε_y, ε_z) Rᵀ` with real R, hence symmetric (§3). The refractiveindex.info loader
reads the separate ordinary / extraordinary entries of uniaxial materials where the database
has them.

### 7. Post-processing

- **Diffraction orders in an anisotropic half-space** are decomposed on the four eigenmodes of
  that half-space (tangential E_t, H_t), not on s/p; every order has an ordinary and an
  extraordinary channel with its own k_z and power. In isotropic half-spaces the existing s/p
  decomposition stays.
- **Absorption** uses `Q = ω/2 · Eᴴ (ε − εᴴ)/(2i) E`; the **Poynting flux** uses `H = (iωμ)⁻¹ curl E`
  with the μ tensor.
- **Far field and cross-sections** only in isotropic cover media (the reciprocity far field
  needs the Green's function of the half-space); anisotropic substrates are measured through the
  mode-resolved orders and fluxes.
- New output: **Jones and Mueller matrix** of each order and of the specular channel.

### 8. Axisymmetric solver and emitters

The axisymmetric solver takes **diagonal tensors in (r, φ, z)** (c-axis along z, or radial /
azimuthal axes); an optic axis tilted against z breaks the rotational symmetry and is
rejected with a diagnostic. The dipole scan (ADR-0013) needs the Green's function of the
anisotropic stack (Sommerfeld integral with the 4×4 formalism of §5, poles and branch cuts of
the extraordinary and ordinary modes); it is a **separate later stage** with its own ADR
amendment once §5 and §7 are validated.

### 9. Validation (each stage with an independent reference)

1. **Isotropic limit:** a tensor ε·I must reproduce every existing benchmark and test to
   round-off (guards the migration).
2. **Homogeneous plate:** uniaxial with the optic axis in the plane, along the normal and tilted;
   reference an independent 4×4 transfer-matrix implementation in the tests (NumPy, thin layers
   only) plus the closed forms (normal incidence: two Fabry–Pérot problems with n_o and n_e).
3. **Energy balance:** R + T + A = 1, lossless and absorbing, all four Jones channels.
4. **Form birefringence:** a sub-wavelength grating of isotropic materials against the homogeneous
   uniaxial plate of effective-medium theory (structured against unstructured), then the same
   with an anisotropic grating material.
5. **Cross-consistency:** the grating and the crystal rotated by the same azimuth, conical against
   3D solver; rotation of the crystal by 90° swaps r_sp and r_ps as symmetry demands.
6. **PML:** `pml_reflection` extended to media with a full tensor.
7. **Reciprocity:** the Jones matrix of a reciprocal (non-gyrotropic) structure satisfies
   r_sp(θ, φ) = r_ps(θ', φ') under the reversed path; checked on a rotated crystal.

## Consequences

- **Safe migration:** because scalar access throws, each of the ~50 files is either migrated or
  refuses anisotropic input with a diagnostic; there is no silent wrong answer. The migration
  order (stages in the roadmap, M19) follows the use: volume solvers and the stack first, then
  the conical solver, then the rest.
- **No cost for isotropic runs:** isotropic materials stay scalar; the tensor path is taken only
  when a cell carries a tensor. The solver symmetry is kept for non-gyrotropic crystals.
- **Larger numerical work:** the conical re-derivation (§4) and the degenerate modes of the
  stack (§5) are the risks; both are checked against independent references before the
  dependent stages start.
- **Limits stay visible:** gyrotropy, tilted axes in the axisymmetric solver, anisotropic cover
  media and the anisotropic emitter scan are listed as out of scope and fail with diagnostics.

## Alternatives considered

- **Always-tensor `Material`** (3×3 for every material): simplest to reason about, but touches
  every one of the 203 `eps_r` uses at once, costs memory and time for the dominant isotropic case
  and makes the migration an all-or-nothing change.
- **Only diagonal anisotropy:** covers on-axis uniaxial crystals, but not the rotated crystals
  (c-plane / a-plane / r-plane cuts at an angle to the grating) that metrology needs, and gives
  no cross-polarisation.
- **Effective isotropic media for form birefringence only:** correct for deep sub-wavelength
  structures, wrong for real crystals and for structures near the diffraction limit.
- **Transfer-matrix (4×4) instead of the S-matrix recursion:** short, but loses digits in thick
  or absorbing layers, which ADR-0009 already rejected for the isotropic stack.
- **Gyrotropy in the first step:** breaks the complex-symmetric solver path and the
  reciprocity-based far field; deferred.

References for the formalisms (Berreman 4×4 formalism, stable S-matrix for anisotropic
multilayers) to be added with checked citations.

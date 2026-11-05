# 0015 — Optically anisotropic materials (substrates, layers, structured parts)

**Status:** accepted
**Date:** 2026-11-01 (proposed and accepted after review the same day)

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
- `materials::Material` is isotropic: the public fields `eps_r`, `mu_r`
  (`include/hpfem/materials/material.hpp`, which names anisotropy as "later").
  `Scattering::form_of_cell` builds `ε_r·I`, `μ_r⁻¹·I`. `eps_r` appears in 51 files of `src/`,
  `include/`, `python/` (203 hits), `mu_r` in 37.
- `physics::LayerStack` (ADR-0009) assumes isotropic layers (scalar k_z per layer, s/p
  polarisations); `docs/axisymmetric-layered-features.md` lists anisotropic layers as out of
  scope. `diffraction`, `farfield`, `waveguide_port`, `propagating_mode`, `mie`, `absorption` and
  the dipole scan (ADR-0013) use scalar indices.
- The direct solvers choose their path by `detect_symmetry` (ADR-0008): complex-symmetric systems
  take the LDLᵀ paths, the others the general LU.
- Several conical paths rely on the symmetry of the unreduced matrix A(β): the adjoint of the
  goal-oriented estimator is assembled from A instead of Aᵀ (`src/physics/conical_goal.cpp`), and
  the emission pattern of M17 S2 (`grating.emission_pattern`, ADR-0013 §6) is computed by
  reciprocity at the same β.
- `hpfem.materials` (`Dispersive`: `Constant`, `Tabulated`, `Sellmeier`, `DrudeLorentz`) returns
  scalars.

## Decision

### 1. Material model

`materials::Material` gains an optional tensor part (`std::shared_ptr<const Tensors>` with a
complex 3×3 `eps`), so that isotropic materials keep their size and speed. `is_isotropic()`,
`eps_tensor()` (returns `eps_r·I` for isotropic) and `mu_inv_tensor()` are the tensor-aware
accessors. **Scalar access to an anisotropic material throws** (`eps_scalar()`, used by every path
not yet migrated): an unsupported feature fails loudly instead of computing with a wrong scalar.
Every such path also reports a `hpfem.diagnostics` entry `anisotropic_unsupported` with the
feature name before it fails.

A public data member cannot throw, so `eps_r` and `mu_r` become **private** with the accessors
`eps_scalar()` / `mu_scalar()` and the constructors `Material::isotropic(eps, mu = 1)` and
`Material::anisotropic(eps_tensor)`. This is a **breaking change** of the C++ API (reads and
writes of `Material::eps_r` / `mu_r`). Transition: S1 replaces all uses inside the repository in
one change; the two-argument constructor `Material(eps_r, mu_r)` keeps `Material{eps, mu}`
compiling.

*Amendment (S1, 2026-11-01):* the Python API does not break. A Python property can raise, so
`Material.eps_r` stays as the scalar property (it raises `hpfem.AnisotropicUnsupported` for a
tensor material, and setting it makes the material isotropic) instead of becoming a deprecated
alias. The diagnostic code is `anisotropic_unsupported`, in the snake_case of the existing
codes. The diagnostic is reported by the Python entry points that run diagnostics
(`grating.validate` / `solve`, `validate_scattering`); C++ paths and other Python paths fail
with the exception. The isotropic-limit regression (§9 item 1) moves to S2, the first stage with
a tensor path to compare.

**μ stays scalar.** Optical crystals have μ = 1; the assembly keeps full μ⁻¹ tensors for the PML,
but `Material` carries no μ tensor. This keeps the 4×4 system of §5 in its simple form and halves
the test matrix; a μ tensor (metamaterial homogenisations) needs an amendment of this ADR.

The tensors are given in the **laboratory frame** (x, y along the lattice, z the layer normal of
the 3D and stack solvers; the conical solver's own frame in §4); the crystal frame and its
rotation are a Python concern (§6).

### 2. Volume solvers (Scattering 2D/3D)

`form_of_cell` returns tensors; with a layered background the source is
`k0² (ε_c(x) − ε_stack(z_c)) E_inc` as a matrix product, so it still vanishes where the structure
equals the stack, in particular inside the PML. The PML stretches by
`ε̃_ij = ε_ij · s_x s_y s_z / (s_i s_j)` (and `μ̃` likewise). Residual and goal-oriented
estimators use `ε` in the divergence term and the `n·εE` jump. Per-cell overrides
(`MaterialMap::set_cell`) accept anisotropic materials, which also gives position-dependent
tensors (strain, graded axes) and a path for `ThermoOptical`.

**PML stability.** The coordinate-stretching PML absorbs a wave only if the component of its
group velocity along the stretching direction has the same sign as the component of its phase
velocity, the geometric condition of Bécache, Fauqueux & Joly (2003). This fails not only in
hyperbolic media but already in an ordinary uniaxial or biaxial crystal **whose axis is tilted
against the PML normal**: an extraordinary wave can carry power into the PML with its phase
travelling out of it, and the stretch then amplifies it. Consequences:

- `form_of_cell` computes, per PML layer, the slowness curve of the medium along the stretching
  direction for the in-plane wave vectors of interest and reports a diagnostic
  `pml-backward-wave` with the medium and the direction where the condition fails; axes aligned
  with the PML normal or lying in the PML plane never trigger it.
- For periodic structures over a tilted anisotropic substrate the substrate side is truncated by
  a **modal transparent boundary** (Dirichlet-to-Neumann map of the 4×4 eigenmodes of the
  half-space for every Fourier order, from §5) instead of a PML; this also gives the
  mode-resolved orders of §7 directly. For isolated structures in such a substrate the first step
  stops with the diagnostic.

*Amendment (S2, 2026-11-05):* the statement above that aligned axes never trigger the
diagnostic holds for **uniaxial** crystals (optic axis along the normal or in the layer plane)
only. An aligned **biaxial** crystal violates the geometric condition along its intermediate
principal axis: near the optic axes the group velocity of one sheet sweeps the cone of internal
conical refraction, which reaches across the layer plane (min k_d v_d ≈ −0.01 for
ε = diag(2, 3, 4.5), checked independently with the gradient of ω(k)). The diagnostic therefore
also fires there; it is a warning, since only waves near the optic axes are affected. The check
samples the wave directions (`pml::backward_wave`, `docs/theory/materials.md`). In the 2D solver
the tensor is the in-plane block; a tensor coupling the plane to z is refused there
(`anisotropic_unsupported`), not approximated.

### 3. Solver symmetry

A material with ε = εᵀ (all non-gyrotropic crystals, also rotated by a real rotation and
absorbing) keeps the 3D and the 2D in-plane curl–curl system complex-symmetric; `detect_symmetry`
routes such systems to LDLᵀ (MUMPS, cuDSS). **This does not hold for the conical solver** once
the tensor couples the longitudinal to the in-plane components (§4). Gyrotropic and optically
active media (antisymmetric parts ε_ij = −ε_ji) are **out of scope of the first step**: they break
reciprocity, on which the far field and the emission pattern by reciprocity rely; they would use
the general LU through the same detection.

### 4. Conical (grating) solver

`ConicalForm` goes from diagonal to full 3×3 tensors (ε full, μ⁻¹ full for the PML only). The
derivation is short and goes into the conical section of `docs/theory/maxwell.md`:

- **Interface conditions need no re-derivation.** They are natural in the weak form: the
  continuity of the tangential field (the in-plane tangential component and E_z, since z is
  tangential to every interface of a z-invariant structure) is built into the Nédélec × H1 pair,
  and the continuity of n·(εE) with n in the (x, y) plane and of the tangential H is natural. The
  gradient kernel `conical_gradient` does not depend on the material and stays exact.
- **The scaled components break the symmetry.** With `E = (E_t, iv) e^{iβz}`, test functions
  `V = (V_t, −iw) e^{−iβz}` and the curls `curl E = D c_E`, `curl V = D̄ c_V`,
  `D = diag(i, −i, 1)` (`c` the real-structured curl components of the conical section), the cell
  matrices are `c_Vᵀ (D̄ μ⁻¹ D) c_E` and `V̂ᵀ (D'' ε D') Ê` with `D' = diag(1, 1, i)`,
  `D'' = diag(1, 1, −i)`. Their entries are, for a symmetric tensor:

  | coupling | stiffness factor | mass factor | symmetric |
  |---|---|---|---|
  | xx, yy, zz | 1 | 1 | yes |
  | xy | −1 (both entries) | 1 | yes |
  | xz, yz | ∓i (antisymmetric pair) | ±i (antisymmetric pair) | **no** |

  So A(β) is complex-symmetric only if ε_xz = ε_yz = 0 (and μ⁻¹_xz = μ⁻¹_yz = 0, which the
  diagonal PML stretch keeps). In general **A(β)ᵀ = S A(−β) S** with `S = diag(I, −I)` (sign of
  the scaled E_z block): reciprocity pairs β with −β. Physically, an optic axis with a component
  along the grating lines that is not perpendicular to them breaks the mirror symmetry z → −z.
- **Consequences in the code.** `kDetect` already routes the non-symmetric matrices to the
  general LU. Every path that uses symmetry must use the transposed system instead:
  the adjoint of `conical_goal` (assemble `Aᵀ`, i.e. `S A(−β) S`, not `A`), the emission pattern
  by reciprocity (ADR-0013 §6: the reciprocal problem at −β), the power matrix on a kept
  factorisation (`ConicalSweep`, transposed solves of the same factorisation) and the material
  and shape sensitivities (ADR-0011). Each gets a unit test with a tilted axis that compares the
  transposed path to a forward solve.
- **The scalar E_z path breaks.** At β = 0 the terms ε_xz, ε_yz couple E_z to the in-plane field,
  so the E_z / H_z polarisations no longer decouple; `ConicalScatteringSetup::scalar_ez` rejects
  such materials with a diagnostic.
- **Staging.** S3 first takes tensors with ε_xz = ε_yz = 0 (optic axis in the cross-section plane
  or along the lines: symmetric, no change of the symmetry-based paths), then the general tensor
  with the transposed paths.

The conical estimators follow §2. Checked against the 3D solver (§9, item 5).

### 5. Anisotropic layer stack

`LayerStack` layers and the substrate may carry anisotropic materials; the incidence medium stays
isotropic and lossless (as today). Per layer the in-plane wave vector (k_x, k_y) fixes a 4×4 system
for the tangential fields (Berreman 1972, Yeh 1979), `∂_z ψ = i K ψ`, `ψ = (E_x, E_y, H_x, H_y)`,
obtained by eliminating E_z and H_z; this requires ε_zz ≠ 0, and an epsilon-near-zero ε_zz is
rejected with a diagnostic.

**Degenerate and defective cases.** The eigenvectors of K are not a reliable basis: every
isotropic layer is doubly degenerate (the common case in a mixed stack), a uniaxial axis near z or
near-normal incidence makes the individual eigenvectors ill-conditioned, and absorbing biaxial
media have exceptional points (Voigt waves) where K is defective. The stack therefore never
diagonalises K. It computes the **ordered complex Schur form** `K = U T Uᴴ` with the two
upward eigenvalues first; the leading two columns of U span the upward invariant subspace, which
is well-conditioned under degeneracy and defined at a defective point. Propagation through a layer
uses the 2×2 matrix exponentials of the diagonal blocks of T, and the **stable S-matrix
recursion** from the substrate upwards works with these subspaces only (layer-local amplitudes,
decaying exponentials only, as in ADR-0009 §2) — no transfer-matrix products through thick or
absorbing layers. Up / down is decided by Im k_z > 0 for evanescent and absorbing modes and by the
sign of the Poynting flux S_z for propagating lossless modes; grazing modes (k_z = 0, Rayleigh
anomalies) are handled as in the isotropic stack. Isotropic layers keep the closed s/p forms as the
fast path and as the test reference.

Individual modes are needed only in the substrate and the incidence medium, for the output: in a
uniaxial substrate the ordinary mode is defined through the optic axis (D ⊥ axis), the
extraordinary one as the other mode of the subspace; in a biaxial substrate the two modes are the
eigenvectors of the upward 2×2 block, made unique at a degeneracy by power orthogonality.
Outputs: the **Jones matrix** (r_ss, r_sp, r_ps, r_pp and the transmission channels into the
substrate modes), the mode amplitudes and the analytic field (value and curl) per layer for the
scattered-field source.

### 6. Python layer and data

`hpfem.materials.Anisotropic` combines three principal `Dispersive` models with an orientation,
`.at(omega)` returns the tensor `Material`. The tensor is `R diag(ε_1, ε_2, ε_3) Rᵀ` with real R,
hence symmetric (§3). Orientation conventions, fixed here:

- **Uniaxial** (`Uniaxial(n_o, n_e, theta, phi)`): the optic axis by its polar angle θ from the
  laboratory z axis and its azimuth φ from the x axis towards y, as used in metrology
  (c-plane: θ = 0; a- and m-plane: θ = 90°).
- **Biaxial**: intrinsic z-x-z Euler angles (α, β, γ), `R = R_z(α) R_x(β) R_z(γ)`, mapping the
  principal (crystal) frame to the laboratory frame; documented with a figure in
  a new page `docs/theory/materials.md` (written in S1).

The refractiveindex.info loader reads the separate ordinary / extraordinary entries of uniaxial
materials where the database has them.

### 7. Post-processing

- **Diffraction orders in an anisotropic half-space** are decomposed on the two modes of that
  half-space (§5, tangential E_t, H_t), not on s/p; every order has an ordinary and an
  extraordinary (or two biaxial) channels with their own k_z and power. In isotropic half-spaces
  the existing s/p decomposition stays.
- **Absorption** uses `Q = ω/2 · Eᴴ (ε − εᴴ)/(2i) E`; the **Poynting flux** uses
  `H = (iωμ)⁻¹ curl E` (μ scalar, §1).
- **Far field and cross-sections** only in isotropic cover media (the reciprocity far field
  needs the Green's function of the half-space); anisotropic substrates are measured through the
  mode-resolved orders and fluxes.
- New output: **Jones and Mueller matrix** of each order and of the specular channel. The s/p
  basis of the incoming and outgoing waves follows the convention of
  `conical_polarisation(k, n̂, s|p)` (s = k̂ × n̂ normalised, p = s × k̂ for both waves), stated in
  `docs/theory/maxwell.md` together with the reciprocity relations of §9.

### 8. Axisymmetric solver and emitters

The axisymmetric solver takes **diagonal tensors in (r, φ, z)** (c-axis along z, or radial /
azimuthal axes); an optic axis tilted against z breaks the rotational symmetry and is
rejected with a diagnostic. The dipole scan (ADR-0013) needs the Green's function of the
anisotropic stack (Sommerfeld integral with the 4×4 formalism of §5, poles and branch cuts of
the extraordinary and ordinary modes) and the β ↔ −β reciprocity of §4; it is a **separate later
stage** with its own ADR amendment once §5 and §7 are validated.

### 9. Validation (each stage with an independent reference)

1. **Isotropic limit:** a tensor ε·I must reproduce every existing benchmark and test to
   round-off (guards the migration).
2. **Homogeneous plate:** uniaxial with the optic axis in the plane, along the normal and tilted;
   reference an independent 4×4 transfer-matrix implementation in the tests (NumPy,
   eigenvector-based, thin layers only) plus the closed forms (normal incidence with the axis in
   the plane: two Fabry–Pérot problems with n_o and n_e).
3. **Degenerate cases of the stack:** isotropic layers between anisotropic ones, axis exactly
   along z at normal incidence, axis 10⁻⁶ rad off z (results continuous in the angle), and an
   absorbing biaxial layer at a Voigt direction (results continuous across it).
4. **Energy balance:** R + T + A = 1, lossless and absorbing, all four Jones channels.
5. **Form birefringence:** a sub-wavelength grating of isotropic materials against the
   homogeneous uniaxial plate of effective-medium theory (structured against unstructured), then
   the same with an anisotropic grating material; published Fourier-modal results for anisotropic
   gratings (L. Li 2003) where they cover the case.
6. **Cross-consistency and symmetry:** the grating and the crystal rotated by the same azimuth,
   conical against the 3D solver. Symmetry checks on a plate with the axis in the plane at the
   azimuth α against the plane of incidence: r_sp(−α) = −r_sp(α) and r_ps(−α) = −r_ps(α) with
   r_ss, r_pp even in α (mirror at the plane of incidence); r_sp = r_ps = 0 for the axis in or
   perpendicular to the plane of incidence.
7. **PML:** `pml_reflection` extended to media with a full tensor, axis aligned (no diagnostic,
   reflection as today) and tilted (the diagnostic `pml-backward-wave` fires where the geometric
   condition fails; the modal boundary of §2 reproduces the stack reference there).
8. **Reciprocity:** for a reciprocal (non-gyrotropic) structure the Jones matrix of the reversed
   path satisfies r_ps(k_in → k_out) = −r_sp(−k_out → −k_in) and r_ss, r_pp equal, with the sign
   fixed by the s/p convention of §7; checked on a rotated crystal in the stack and in the conical
   solver (which exercises the β ↔ −β pairing of §4).
9. **Conical symmetry:** the assembled conical matrix is complex-symmetric for ε_xz = ε_yz = 0 and
   satisfies A(β)ᵀ = S A(−β) S for a general symmetric tensor (unit test on the cell matrices).

## Consequences

- **Safe migration:** because scalar access throws, each of the ~50 files is either migrated or
  refuses anisotropic input with a diagnostic; there is no silent wrong answer. The price is a
  breaking change of the C++ `Material` (private fields), absorbed in one change; the Python
  API keeps `eps_r` as a property. The migration order (stages in the roadmap, M19) follows the use: volume
  solvers and the stack first, then the conical solver, then the rest.
- **No cost for isotropic runs:** isotropic materials stay scalar; the tensor path is taken only
  when a cell carries a tensor. The solver symmetry is kept for non-gyrotropic crystals in the 3D
  and 2D solvers and for the conical solver with ε_xz = ε_yz = 0.
- **Conical symmetry loss:** a general tilted crystal in the conical solver costs the general LU
  (roughly twice the factorisation memory of LDLᵀ) and requires the transposed paths of §4 in
  the adjoint, reciprocity and sensitivity code.
- **Larger numerical work:** the Schur-based stack (§5), the transposed conical paths (§4) and the
  modal substrate boundary (§2) are the risks; each is checked against an independent reference
  before the dependent stages start.
- **Limits stay visible:** gyrotropy, a μ tensor, ε_zz → 0 in the stack, tilted axes in the
  axisymmetric solver, anisotropic cover media, isolated structures in a tilted substrate (PML
  condition) and the anisotropic emitter scan are out of scope and fail with diagnostics.

## Alternatives considered

- **Always-tensor `Material`** (3×3 for every material): simplest to reason about, but touches
  every one of the 203 `eps_r` uses at once, costs memory and time for the dominant isotropic case
  and makes the migration an all-or-nothing change.
- **Keeping `eps_r` public** and documenting that it is meaningless for anisotropic materials:
  no API break, but exactly the silent wrong answer the throwing accessor prevents.
- **Only diagonal anisotropy:** covers on-axis uniaxial crystals, but not the rotated crystals
  (c-plane / a-plane / r-plane cuts at an angle to the grating) that metrology needs, and gives
  no cross-polarisation.
- **Effective isotropic media for form birefringence only:** correct for deep sub-wavelength
  structures, wrong for real crystals and for structures near the diffraction limit.
- **Transfer-matrix (4×4) instead of the S-matrix recursion:** short, but loses digits in thick
  or absorbing layers, which ADR-0009 already rejected for the isotropic stack.
- **Eigenvector basis per layer** (closed forms for uniaxial layers, a projection for degenerate
  pairs): the textbook route, but ill-conditioned near degeneracies and undefined at the
  exceptional points of absorbing biaxial media; the ordered Schur form covers all cases with one
  code path.
- **A different conical scaling** (e.g. `V_z = iw`) to restore symmetry: it changes only the sign
  of the zz block and moves the antisymmetry elsewhere; the loss of symmetry is physical (z → −z
  mirror) and cannot be scaled away.
- **Gyrotropy in the first step:** breaks the complex-symmetric solver path and the
  reciprocity-based far field; deferred.

## References

- D. W. Berreman, "Optics in stratified and anisotropic media: 4×4-matrix formulation",
  J. Opt. Soc. Am. 62, 502 (1972), doi:10.1364/JOSA.62.000502.
- P. Yeh, "Electromagnetic propagation in birefringent layered media", J. Opt. Soc. Am. 69,
  742 (1979).
- L. Li, "Fourier modal method for crossed anisotropic gratings with arbitrary permittivity and
  permeability tensors", J. Opt. A: Pure Appl. Opt. 5, 345–355 (2003).
- E. Bécache, S. Fauqueux, P. Joly, "Stability of perfectly matched layers, group velocities and
  anisotropic waves", J. Comput. Phys. 188, 399–433 (2003).

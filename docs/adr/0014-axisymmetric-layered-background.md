# 0014 — Layered background for the axisymmetric solver

**Status:** accepted
**Date:** 2026-10-10

## Context

Milestone M18 (`docs/axisymmetric-layered-features.md`, items S0–S4) asks for plane-wave
scattering by bodies of revolution on substrates and in layer stacks: particles on substrates,
nanoparticle-on-mirror gap plasmons, holes and grooves in films. The axisymmetric solver of
ADR-0010 forms the scattered-field source against one homogeneous background material; a substrate
then becomes a scatterer of infinite radial extent, and no cross-section exists. ADR-0009 solved
the same problem for `Scattering<2>` / `<3>` with a `LayerStack<Dim>` background. Facts from the
code (`main` after M16 S7):

- `LayerStack<3>::plane_wave(k0, angle, pol, amplitude, azimuth)` gives the analytic field (value
  and curl) of a plane wave on a stack whose normal is z, with the per-layer amplitudes and
  vertical wavenumbers; it refuses |angle| ≥ π/2 and requires a lossless incidence medium.
- `oblique_plane_wave(amplitude, k, theta_i, ...)` expands one plane wave of real wave vector into
  azimuthal orders by Jacobi–Anger (Bessel functions of k sin θ · r).
- The PML stretches each cell's own material (ADR-0005), so layers continue into the radial PML.
- `absorbed_power_by_tag` and `combined_field` (ADR-0009 §5) exist for `Scattering<Dim>`; the
  axisymmetric solver has no absorbed-power routine.

## Decision

### 1. Background stack and source

`AxisymmetricScatteringSetup::background` (`std::optional<LayerStack<3>>`; the axis of the body is
the stack normal, z of the meridian plane = y of the mesh). When present, `form_of_cell` forms the
source against the stack material at the cell centroid, `k0² (ε_c − ε_stack(z_c)) E_inc`, as
ADR-0009 §3: it lives only where the body deviates from the stack. The sign of the deviation does
not matter — holes and grooves (air in a layer) are sources like particles. The constructor checks
that no cell straddles a stack interface and that **no cell of the radial PML deviates from the
stack** (the deviation must be radially bounded); both raise `InvalidArgument` naming the cell.

### 2. Order-m expansion of the stack field

`layered_axisymmetric_wave(stack, k0, theta, pol, m)` returns the order-m component of the stack's
plane wave (in-plane wave vector along +x, azimuth 0) in the scaled components
(E_r, v = −i r E_φ, E_z) and its curl. Analytically: in layer j the field is the sum of an up and a
down partial wave sharing the real in-plane wavenumber k_ρ = k0 n_inc sin θ (phase matching) with
the vertical wavenumber ±k_z,j (complex in lossy layers and beyond the critical angle); each
partial wave is expanded by the Jacobi–Anger formulas of `oblique_plane_wave`, generalised from
(k, θ) to (k_ρ, k_z) with the polarisation vectors s = ŷ and p = (∓k_z, 0, k_ρ)/k_j (complex for
complex k_z) and the amplitudes of the `LayeredPlaneWave` (u = E_s for s, H for p, converted to E).
The Bessel functions keep the real argument k_ρ r. The **test reference** is independent of this
algebra: the numerical Fourier transform over φ of the 3D field `LayerStack<3>::plane_wave(...).field`
on rings of radius r, projected on the cylindrical unit vectors, to round-off for every layer, m,
polarisation and angle (including beyond the critical angle).

### 3. Incidence from the substrate side

A wave from below is the wave from above on the **reversed stack** — incidence medium the old
substrate (which must then be lossless), layers in reverse order, substrate the old incidence
medium — mirrored z → −z. `layered_axisymmetric_wave` takes `side = "top" | "bottom"` and builds the
reversed stack internally; the mirror maps (E_r, v, E_z)(r, z) to (E_r, v, −E_z)(r, −z) and flips
the curl accordingly. Test: R and T of the reversed bare stack against the forward one
(reciprocity), and total internal reflection beyond the critical angle (R = 1 for a lossless
stack, an evanescent field in the cover).

*Polarisation reference (clarified with M18 S1):* the mirror alone would turn the magnetic field
of the p wave from +y to −y. `layered_axisymmetric_wave` undoes that sign, so that p has H along
+y from both sides and the homogeneous-stack limit from below equals `oblique_plane_wave(θ)`, as
from above it equals `oblique_plane_wave(π − θ)`. Only the phase reference of the p amplitude is
affected; fields, R and T are unchanged.

### 4. Post-processing channels

All normalised by the incident intensity n_inc |E0|² / (2 Z0):

- **Axisymmetric absorbed power** (new): the Joule heating of a field given by its orders,
  summed over m (the orders are orthogonal in φ: the factor 2π and the per-order integrals with
  the weight r), by tag and by cell; a variant adds the analytic stack field per quadrature point
  (total field = stack + scattered, the ADR-0009 §5 pattern for the axisymmetric orders). The body's
  absorption is that of the total field in the cells deviating from the stack; around a hole the
  **absorption change** is the total-field absorption in a bounded region minus the stack field's
  in the same cells (can be negative).
- **Scattered power by channel**: the Poynting flux of the scattered field through a closed surface
  of mesh lines around the body, split into the parts above the top interface, below the bottom
  interface and the radial part through the layers (lateral = guided). In a lossy substrate the
  part inside it depends on the depth; the API documents it and offers the surface restricted to
  the lossless media.
- **Aperture transmission**: `axisymmetric_disc_flux` — the flux of the total field through a
  horizontal disc r ≤ R at height z, from which the stack's transmission through the same area is
  subtracted analytically.
- **Extinction** = absorption + scattering (no layered optical theorem).
- **Far field per half-space** (S3): by reciprocity, the overlap of the equivalent surface currents
  of the scattered field with the layered plane wave arriving from the direction (θ, φ, pol) in that
  half-space (§2 again); the constant is fixed by the homogeneous-stack limit against the existing
  near-to-far transform.

### 5. Python

`AxisymmetricScatteringSetup.background`, `hpfem.layered_axisymmetric_wave(stack, k0, theta, pol, m,
side="top")`, the absorbed-power and disc-flux helpers; `scatter_orders` unchanged (the user passes
the order-m incident field and the background).

## Consequences

- Bodies of revolution on and in stacks get the same treatment as the 2D / 3D solvers: the source is
  bounded, the PML sees no source, cross-sections exist per channel.
- The analytic expansion is cheap per quadrature point (a few Bessel functions per partial wave);
  the numerical φ-transform is only a test reference.
- Substrate-side incidence needs a lossless substrate (as the incidence medium of ADR-0009); a lossy
  substrate can only be illuminated from above.
- The axisymmetric absorbed power is new code for all axisymmetric problems, not only layered ones
  (emitters in lossy hosts benefit as well).
- The 3D cross-checks of the specification are long local runs (`validation-long`) with stored
  regression records; CI runs the analytic and order-expansion tests.

## Alternatives considered

- **Total-field formulation with the stack field as boundary data** — the incident field would have
  to be imposed on the radial PML boundary, where it does not decay; the scattered-field
  formulation of ADR-0009 keeps the PML source-free.
- **The numerical φ-transform of the 3D stack field as the production path** — simple and exact but
  needs N_φ ≳ 2(|m| + k_ρ r) samples per quadrature point; kept as the test reference.
- **3D solver with `LayerStack<3>`** — exists, but orders of magnitude more expensive for bodies of
  revolution; it is the cross-check.
- **Sommerfeld-integral Green's function of the stack** (boundary-element style) — no FEM path for
  inhomogeneous bodies; out of scope.

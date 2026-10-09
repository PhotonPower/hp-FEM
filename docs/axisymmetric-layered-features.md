# Layered background for the axisymmetric solver (milestone M18)

Status: proposal of 10 October 2026 from the GUI work (FEM model builder, mode "body of
revolution"), reviewed by the coordinating dev agent the same day (review notes marked
**Review**). Audience: the development agents. Tracked in `docs/roadmap.md` as milestone M18; the
IDs S0 to S4 below are used there. Decision record: ADR-0014 (S0), extending ADR-0009 (layered
background) to ADR-0010 (axisymmetric solver). Statements marked *hypothesis* are not verified;
everything else was checked against `main`.

## 1. The gap

1. `AxisymmetricScatteringSetup` has no `background`: `AxisymmetricScattering::form_of_cell`
   (`src/physics/axisymmetric.cpp`) forms the contrast source `k0² (ε_cell − ε_background) E_inc`
   against one homogeneous background material, and the only plane waves are
   `oblique_plane_wave` / `axial_plane_wave` in that medium.
2. A substrate or a planar layer is therefore a scatterer of infinite radial extent: the contrast
   source runs through the radial PML to the wall, the "scattered" field contains the reflected
   and transmitted plane waves of the stack, which do not decay radially, and the scattered power
   through a closed surface grows with the domain (no cross-section exists). ADR-0009 solved
   exactly this for the 2D / 3D solvers.
3. Problems that need it are among the most common uses of body-of-revolution solvers:
   nanoparticles on substrates (dark-field spectroscopy, single-particle absorption, SERS);
   nanoparticle-on-mirror (NPoM) gap plasmons; micropillars, microdisks and bullseye antennas
   illuminated from outside; tips above surfaces (SNOM / TERS); **holes and grooves in the stack**
   (negative deviations, 2.1): a nanohole in a metal film, bullseye grooves around a hole or an
   emitter, pits and pillars etched into the substrate, a particle in a well.
4. Pieces that already exist and fit: `LayerStack<3>::plane_wave` (s and p, stable recursion,
   R / T / A), the per-cell material in the PML (`axisymmetric_pml_form` stretches the cell's own
   material, so layers continue into the radial PML), the Jacobi–Anger expansion of a plane wave
   into orders m (`oblique_plane_wave`), `scatter_orders`, `axisymmetric_poynting_flux`. The GUI
   verified that planar layers running through the radial PML work for sources (dipole in a GaAs
   membrane against the Sommerfeld solution to 4e-4 / 1.4e-3, independent of the PML thickness).

## 2. Proposal

### 2.1 Background and incident field — S1

- `AxisymmetricScatteringSetup::background` (`std::optional<LayerStack<3>>`, normal along the
  axis: z of the meridian plane = y of the mesh), as `ScatteringSetup::background` of ADR-0009.
  When present, `form_of_cell` takes the background permittivity from the stack at the cell
  centroid, `k0² (ε_c − ε_stack(z_c)) E_inc`, so that the source lives only where the body differs
  from the stack (nowhere in the PML). The constructor checks that no cell straddles a stack
  interface (as ADR-0009).
- `layered_axisymmetric_wave(stack, k0, theta, pol, m)` → the order-m component of the stack's
  plane-wave field in the scaled components (E_r, v = −i r E_φ, E_z) and its curl, in every layer:
  all partial waves share the in-plane wavenumber k_ρ = k0 n_inc sin θ (phase matching), so each up
  / down wave of layer j is an oblique plane wave with that k_ρ and the layer's k_z,j, expanded by
  the Jacobi–Anger formulas of `oblique_plane_wave` (Bessel functions of the real argument k_ρ r).
  Amplitudes from the `LayeredPlaneWave` of `LayerStack<3>::plane_wave`.
  **Review:** `oblique_plane_wave(amplitude, k, theta_i, ...)` takes a real k and a real angle; the
  partial waves need a generalised expansion with a real k_ρ, a complex k_z,j (lossy layers, beyond
  the critical angle) and the complex p-polarisation vector (k_z/k enters it). The Jacobi–Anger
  part is unchanged because k_ρ is real — new code, but no new mathematics.
- **Negative deviations (holes) are first-class.** The scatterer is the deviation of the cell
  material from the stack: a cell of air inside the substrate or a metal layer has the contrast
  ε_air − ε_stack(z) ≠ 0 and is a source like any particle. Nothing depends on the sign of the
  contrast; the deviation must be radially bounded (it must not reach the radial PML). A hole
  through a whole layer (aperture), a ring groove (r_inner > 0) and a hole with a particle in it
  are all valid inputs. 2D precedent: the slit and groove of the slit–groove benchmark are air in a
  silver layer of `LayerStack<2>` (ADR-0009, `docs/validation.md`).
- **Incidence from the substrate side** (θ > 90°, total-internal-reflection / evanescent
  illumination of particles on a prism). **Review:** this does not fall out of `LayerStack<3>`:
  `plane_wave` refuses |angle| ≥ π/2 and the incidence medium must be lossless. It is a reversed
  stack (substrate as incidence medium, layers in reverse order, the substrate lossless) mirrored
  z → −z; ADR-0014 fixes the construction and a test covers it (reflectance of the reversed bare
  stack against the forward one by reciprocity, TIR beyond the critical angle).
- `scatter_orders(...)` unchanged; the user passes the order-m incident field and sets
  `setup.background`. Python: `AxisymmetricScatteringSetup.background`,
  `hpfem.layered_axisymmetric_wave`.

### 2.2 Post-processing in a layered background — S2

Cross-sections of a body on a substrate are defined per channel; the API returns them separately,
normalised by the incident intensity n_inc |E0|² / (2 Z0):

1. **Scattered power through a closed surface** around the body (`axisymmetric_poynting_flux` of
   the scattered field), split into the part above the top interface, the part below the bottom
   interface, and the radial part through the layers. In a lossless stack the sum is the total
   scattered power; in a lossy substrate the part inside the substrate depends on the depth (to be
   documented, or the surface restricted to the lossless media).
2. **Absorption in the body**: absorbed power of the TOTAL field (stack field + scattered field) in
   the cells whose material differs from the stack (or by tag), not the stack's own absorption. For
   a hole the body is lossless air; the meaningful quantity is the *change* of the absorption in
   the surrounding lossy layer: absorption of the total field in a bounded region around the hole
   minus that of the stack field in the same cells (Δσ_abs, can be negative).
   **Review:** the axisymmetric solver has no absorbed-power routine at all yet (the 2D / 3D
   `absorbed_power_by_tag` add the incident or background field to the unknown per quadrature
   point; there is no helper named `combined_field`). S2 first adds the axisymmetric absorbed power
   — the orders are orthogonal in φ, so the power is the sum over m of the per-order integrals with
   the factor 2π — and then the variant with the analytic stack field per quadrature point.
3. **Extinction**: σ_abs + σ_sca (the homogeneous optical theorem does not hold; the generalised
   version with the specular directions of the stack is out of scope).
4. **Far field per half-space** (S3): dP/dΩ in the cover and in a lossless substrate by
   reciprocity — the far field in direction (θ, φ, pol) of either half-space is the overlap of the
   equivalent surface currents with the layered plane wave arriving from that direction
   (`LayerStack<3>::plane_wave`, expanded into orders); integrated over a collection cone it gives
   the dark-field signal for a numerical aperture. Power into guided modes of the layers =
   scattered power − radiated up − down (radial flux at large r as a check).
5. Near field: total = stack field + scattered field for maps and fluxes.
6. **Transmission through an aperture**: for a hole through an opaque film, the flux of the TOTAL
   field through a disc r ≤ R at height z below the film minus the stack's own transmission
   through the same area, normalised by the incident power on the hole area (T/T_geom of
   extraordinary-transmission studies): helper `axisymmetric_disc_flux`.

### 2.3 Emitters (no change needed)

Dipole sources in the total-field formulation already work with layers and substrates (no incident
field). The new background is only for plane-wave excitation. A combined test uses the dipole path
as an independent check via reciprocity.

## 3. Tests (Definition of Done, CLAUDE.md §7)

1. **No body**: mesh of the bare stack (lossy substrate, two layers), oblique s and p: the
   scattered field is zero to solver precision for every order m, and the total-field flux through
   planes above / below equals the stack's R and T (as ADR-0009 in 2D).
2. **Homogeneous stack limit**: all layers and the substrate set to the cover material → identical
   results to today's homogeneous solver (Mie sphere, existing test).
3. **Sphere far above a substrate** (gap ≫ λ, lossless): scattered power against the Mie value plus
   the interference term of the reflected wave; at least the dipole limit analytically.
4. **Small sphere on a substrate (quasi-static limit)**: polarisability with the image-dipole
   correction (Wind–Vlieger / Bobbert–Vlieger dipole limit) for σ_abs of a 10 nm Au sphere on glass
   at normal incidence (tolerance from the neglected retardation, O((k d)²)).
5. **Cross-check with the 3D solver**: the same sphere on a two-layer stack with `Scattering<3>` +
   `LayerStack<3>` at oblique p incidence: σ_sca, σ_abs and the near field at a few points.
   **Review:** a long local run (ctest label `validation-long`) with the result stored in
   `benchmarks/results/` and a quick regression against the stored numbers in CI.
6. **Reciprocity of the far field**: the reciprocity-based far field (2.2 (4)) against a
   near-to-far transform in the homogeneous-stack limit, and against a dipole source.
7. **Single hole in a gold film** (negative deviation): a 200 nm hole through a 100 nm Au film on
   glass, normal incidence, 600–1000 nm: (a) transmitted power through a disc below the film against
   `Scattering<3>` + `LayerStack<3>` with the same hole (long local run, stored as a regression
   record); (c) power balance: reflected + transmitted + absorption change = 0 to the discretisation
   tolerance. **Review:** the proposed (b), Bethe's small-hole limit T/T_geom = (64/27π²)(ka)⁴ for
   a perfectly conducting screen of vanishing thickness, is dropped: PEC is not a layer material,
   a finite film damps the hole transmission exponentially with its thickness, and T ~ (ka)⁴ is
   below the discretisation error at the sizes a mesh can resolve; (a) and (c) test the same code.
8. **Bullseye groove** (regression): a ring groove (r_inner > 0) in a metal film around an emitter
   on the axis, emitter path against the plane-wave path by reciprocity.
9. **NPoM sanity case** (regression, not a reference): Au sphere 60 nm on a 2 nm spacer over an Au
   film, normal incidence: the coupled gap mode red-shifted from the isolated sphere, converging in
   p.
10. **Review — substrate-side incidence:** the reversed bare stack against the forward one (R and T
    by reciprocity), and total internal reflection beyond the critical angle (R = 1 for a lossless
    stack, evanescent field in the cover).

## 4. What the GUI needs (FEM model builder)

- Scattering task in the "body of revolution" mode with substrate and radially infinite layers (the
  GUI already builds the meshes: layers through the radial PML, interfaces on mesh lines, a closed
  measurement box of mesh lines), angle and polarisation of the incidence from above (optionally
  from the substrate side). Holes need nothing new: a part of air drawn over a layer or the
  substrate already overrides it in the mesh.
- For apertures: the transmitted power through a disc below the film (minus the stack's
  transmission), normalised to the hole area.
- Per wavelength: σ_sca split into up / down / lateral (guided), σ_abs of the body, σ_ext = sum, the
  radiation pattern in both half-spaces, the collected power for a numerical aperture (dark field:
  illumination cone and collection cone as two angle ranges), the near field (total and scattered).
- Same conventions as the homogeneous case (orders summed by `scatter_orders` with its stopping
  tolerance, scaled components, the field export through `FieldExporter2D`).

## 5. Out of scope (first step)

- Anisotropic or magnetic layers, graded-index layers.
- Bodies of revolution whose axis is not normal to the layers.
- Periodic arrays of bodies on a substrate (the periodic 3D solver, or M17 for emitters).
- The generalised optical theorem for layered media; σ_ext is the sum of the measured channels.

## 6. Context in the GUI

The GUI mode "body of revolution" (`fem_axi.py`, `fem_axi_worker.py`) offers resonances, emitters
(Purcell factor, β, Riesz modal expansion) and plane-wave scattering in a homogeneous medium
(validated against the Mie series: Au sphere in water and Si sphere at 45° to 1e-4 … 1e-3). It
supports a substrate and radially infinite layers for resonances and emitters and blocks them for
scattering; with M18 the block is lifted and the cross-section channels of 2.2 are shown.

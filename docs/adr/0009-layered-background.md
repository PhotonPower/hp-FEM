# 0009 — Layered background for the scattered-field formulation
**Status:** proposed
**Date:** 2026-10-03

## Context
The scattered-field formulation (`physics::Scattering`, ADR-0002, `docs/theory/maxwell.md`)
assumes one homogeneous background medium: the incident field is an analytic solution of the
*homogeneous* Maxwell equations with `materials.background()`, the volume source is
$k_0^2(\varepsilon(x) - \varepsilon_b)E^{inc}$ in every cell whose material differs from the
background, and the PML absorbs the scattered field only. Many targets of this project
(gratings and masks on substrates, slits and grooves in metal films, antennas on layered
media, the slit–groove benchmark of M10) consist of a *structure embedded in a layer stack*.
With a homogeneous background every layer becomes a scatterer: the source does not vanish
inside the PML (the EUV example lives with a small silicon contrast there), the "scattered"
field carries the whole reflected and transmitted plane waves, and quantities such as the
power in the substrate have to be reconstructed from a field that is dominated by the
background response.

Today every consumer of the incident field (volume source, PEC data $E^{sc} = -E^{inc}$,
`total_field`, `cross_sections`, `poynting_flux`, `hcurl_error`, the sweeps) calls
`IncidentField::value(x)` / `curl(x)` at physical points; the only hard-wired homogeneity is
the contrast against `materials.background()` in `Scattering::form_of_cell`, the incident
intensity in `cross_sections` and the scalar background index of `pml::PmlBox`.

## Decision
1. **Background field = analytic plane-wave solution of a layer stack.** A new class
   `physics::LayerStack<Dim>` describes planar layers perpendicular to the last coordinate
   ($y$ in 2D, $z$ in 3D): a semi-infinite incidence medium on top, $N \ge 0$ finite layers
   with thickness and `materials::Material`, a semi-infinite substrate. For a plane wave of
   given angle, polarisation and amplitude it returns an `IncidentField<Dim>` (value *and*
   curl, piecewise per layer) plus the reflection, transmission and absorption of the bare
   stack. In 2D only the in-plane-E polarisation exists (the field of `Scattering<2>`); in 3D
   both s and p. Interfaces must coincide with mesh facets (the user builds the mesh with
   coordinate lines on the interfaces, as all examples already do).
2. **Numerically stable recursion.** Reflection and transmission by the Airy / S-matrix
   recursion from the substrate upwards and the per-layer wave amplitudes referenced to the
   layer's own interfaces (downward wave at the top, upward wave at the bottom of each layer),
   so that only decaying exponentials $e^{i k_{y,j} d_j}$, $\operatorname{Im} k_{y,j} \ge 0$,
   appear. Thick metal layers neither overflow nor lose digits (transfer matrices do).
3. **`ScatteringSetup::background`** (`std::optional<LayerStack<Dim>>`): when present,
   `form_of_cell` takes the background permittivity of a cell from the stack at the cell
   centroid, $k_0^2(\varepsilon_c - \varepsilon_{bg}(y_c))$, so that the source lives only where the
   structure deviates from the stack (in particular nowhere inside the PML), and
   `cross_sections` normalises by the intensity of the incidence medium. The user sets
   `setup.incident = stack.plane_wave(...)`; the constructor checks that the incident field
   belongs to the stack (same object) and that no cell straddles an interface.
4. **PML unchanged.** The stretched tensors are built per cell from the cell's own material
   (ADR-0005), so each layer continues into the lateral and the top/bottom PML with its own
   $\varepsilon$; the stack's field is defined there analytically and produces no source. The
   scalar `background_index` of `PmlBox` only sets $\sigma_{max}$; it is documented to be
   chosen as the smallest real index of the layers the PML touches (strongest profile), a
   per-side index can follow later without changing this decision.
5. **Post-processing.** The total field stays $E^{bg} + E^{sc}$ assembled on the fly:
   `combined_field(discrete_field(dofs, e_sc), analytic_field(stack field), 1)` for fluxes,
   `e_sc + interpolate(dofs, physical_sampler(stack.value))` for coefficient-based quantities,
   exactly as today. No new field type.
6. **Python**: `LayerStack2D/3D`, `LayerStackPlaneWave` result (R, T, A, field) and
   `ScatteringSetup.background`; a C++ closure, never a Python callback, carries the field.

## Consequences
- Structures on substrates become genuine scattered-field problems: the unknown is small,
  the PML sees only outgoing scattered waves, and fluxes of the total field through planes
  in the substrate give transmission directly (D2 slit–groove benchmark).
- `Scattering` gets one more optional member and two branch points (`form_of_cell`,
  `cross_sections`); everything else (Dirichlet data, sweeps, evaluation, far field,
  diffraction) works unchanged because the stack field is an `IncidentField`.
- The incidence-medium intensity replaces `materials.background()` in `cross_sections`
  when a stack is present; without a stack nothing changes.
- The mesh must resolve the interfaces by facets; `tensor_mesh.hpp`-style and
  `rectangle`-based meshes with interface-aligned lines do. A check in the constructor
  reports a straddling cell with its index.
- Tests: transfer solution against Fresnel for one interface, $R + T + A = 1$ for a lossy
  stack, FEM without perturbation gives $E^{sc} \approx 0$ on a layered mesh, and the
  Poynting flux of the total field through a plane in the substrate equals the analytic
  transmission. Theory section in `docs/theory/maxwell.md`.

## Alternatives considered
- **Total-field formulation with the stack field as boundary data.** Needs the incident
  field on the outer boundary *inside* the PML, where the PML stretches the scattered part
  only; incompatible with ADR-0005 without a two-field PML.
- **Keep the homogeneous background and treat the layers as scatterers** (status quo).
  Sources inside the PML, large "scattered" fields, loss of accuracy for transmission;
  acceptable for weak contrasts only.
- **Numerical 1D background** (solve the stack with a 1D FEM). Exact enough but adds a
  second discretisation error and gives no closed-form R/T to validate against; the
  analytic recursion is a few dozen lines.
- **A dedicated `TotalField` object.** Rejected: all consumers already combine incident and
  scattered parts on the fly; a new type would duplicate the post-processing API.

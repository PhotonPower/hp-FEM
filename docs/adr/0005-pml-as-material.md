# 0005 — PML as complex material tensor inside the FEM
**Status:** accepted · **Date:** 2026-10-02

## Decision
Implement PML as complex coordinate stretching expressed through modified
$\tilde\varepsilon,\tilde\mu$ tensors in the outer layer of cells; no special element type.

## Consequences
PML support reduces to (a) a material callback evaluating the stretch at quadrature
points and (b) higher quadrature order in PML cells. Resonance problems become
non-Hermitian eigenproblems. See `docs/theory/pml.md`.

## Alternatives considered
Infinite elements / boundary-element coupling: exact but much more code and poor fit
with periodic structures. Absorbing boundary conditions: too reflective for the
$10^{-6}$ accuracy goals of scatterometry.

## Amendment 2026-11-25: radial layers and `pml::Pml` (issue #244 S4)
**Context.** A sector of a structure with cyclic symmetry C_n can carry an absorbing layer only
if the layer is invariant under the rotation by 2π/n. The axis-aligned `PmlBox` is invariant
only for C_2, C_4 (about its centre) and C_3 about a cube diagonal, so C_3, C_6, C_8 … in 2D and
about any axis in 3D had no open boundary.

**Decision.** A second layer shape, `pml::RadialPml<Dim>`, stretches the distance from a centre
(2D circle, 3D sphere) or from an axis (3D cylinder, with optional slab layers along the axis)
with the same profile as the box. It stays a material tensor (this ADR): the Jacobian
$J = \partial\tilde x/\partial x$ is now a full symmetric matrix in Cartesian components,
$\tilde\varepsilon = \det J\,J^{-1}\varepsilon J^{-T}$, $\tilde\mu^{-1} = J\mu^{-1}J^T/\det J$.
The problem setups (`ScatteringSetup::pml`, `ResonanceSetup::pml`) hold a `pml::Pml<Dim>`, a
value type wrapping either shape (`std::variant`) with the operations the solvers use (layer
membership, Jacobian, tensors, resolution); it converts implicitly from both, so existing code
assigning a `PmlBox` is unchanged, and `box()` / `radial()` give the concrete layer. The
conical and axisymmetric solvers keep their `PmlBox<2>` (their cross-section geometry has no
use for a radial layer). The symmetry reductions check that the layer of the reduced mesh
continues into its images (`Pml::is_invariant` on the sector sides and mirror planes).

**Consequences.** The stretched tensors are full instead of diagonal (the assembly already
takes full tensors). A crystal inside a radial layer is checked for the PML stability
condition on a fan of stretching directions, since the stretching direction turns with the
point (`backward_wave_along`). A circular layer wants a mesh whose circles are mesh lines:
`mesh::polar_disc`, a C_q-symmetric ring mesh with curved circles.

**Alternatives considered.** A sector mesher that puts the box layer into the corners of each
sector: the box is still not invariant under the rotation, so the coupled sector sides would
see different stretches; it does not work. A polymorphic `Pml` base class behind a pointer:
the setups are copied by value everywhere (sweeps, symmetry parts), a variant keeps value
semantics without allocation.

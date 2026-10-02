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

# 0003 — Entity orientation by global vertex order
**Status:** accepted · **Date:** 2026-10-02

## Context
High-order edge and face shape functions are orientation dependent; neighbouring cells
must agree on the orientation of shared entities, also across Bloch-periodic faces and
after refinement.

## Decision
An edge is globally oriented from the smaller to the larger global vertex index; a face by
its vertex indices sorted ascending. Each cell stores, per local entity, the permutation
relating local to global orientation; the DoF map applies the resulting sign/permutation
to the local basis. Periodic faces are matched by vertex correspondence, so the rule
extends to them.

## Consequences
Deterministic, generator-independent, refinement-stable. Slight per-cell overhead at
basis evaluation (sign flips), negligible.

## Alternatives considered
Orientation from mesh file ordering (fragile), explicit orientation sweeps (complex,
breaks on refinement).

# 0004 — Hierarchical Schöberl–Zaglmayr Nédélec basis
**Status:** accepted · **Date:** 2026-10-02

## Decision
Use the hierarchical first-kind Nédélec basis of Schöberl & Zaglmayr with explicit
gradient/non-gradient splitting, order assignable per entity.

## Consequences
- Enables $p$-adaptivity with the minimum rule and trivial gauging.
- Conditioning degrades with $p$ (mitigated by static condensation and, later, preconditioning).
- Interpolation of boundary data and constraint computation use the entity-wise structure.

## Alternatives considered
Interpolatory (Lagrange-type) Nédélec bases: better conditioning, but fixed uniform $p$
per cell and no hierarchy — incompatible with principle §2.3 of CLAUDE.md.

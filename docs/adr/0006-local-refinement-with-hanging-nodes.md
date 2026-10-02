# 0006 — Local h-refinement: red refinement, hanging nodes, constraints by interpolation
**Status:** accepted · **Date:** 2026-10-02

## Decision
Local *h*-refinement is **red (regular) refinement of individual cells** organised in a
refinement tree (`mesh::AdaptiveMesh`) whose leaves form the current mesh. The leaf mesh is
**one-irregular**: before a cell is split, every coarser leaf cell sharing a vertex with it
and every leaf cell owning the parent of one of its facets is split first (2:1 balance by
vertices *and* facets). The non-conforming *hanging* edges and faces are registered in the
`Mesh` (`set_hanging`); they are not boundary facets, and the H1 / H(curl) spaces are kept
conforming by **constraints of the hanging DoFs** (`assembly::hanging_constraints`): each
slave DoF is the hierarchical interpolant of the parent entity's functions, with coefficients
computed numerically by interpolating every master shape function onto the child entities
(`assembly::interpolate`). The constrained system is reduced by $P^H A P$
(`fespace::Constraints`), Dirichlet data is imposed on the free DoFs afterwards, and
solutions are transferred between meshes by the same interpolation (`assembly::prolongate`).

## Consequences
- The same child patterns serve uniform (`refine_uniform`) and local refinement; curved
  roots keep their quadratic surface. Shape regularity is that of red refinement (Bey's
  rule in 3D: at most three congruence classes).
- The DoF map's minimum rule extends to hanging entities: a parent entity takes at most the
  order of its children, so the parent's trace is always representable on the fine side;
  child functions of higher degree are constrained to zero. Variable orders and hanging nodes
  therefore combine without special cases.
- Coefficients by interpolation cost a few small projections per hanging entity and avoid
  closed-form tables that would depend on orientation codes, orders and the basis family.
- The `Constraints` object is shared with the Bloch-periodic constraints (`append`); the
  reduced system is built after assembly rather than by condensation during assembly.
- Facet loops (estimator jumps, flux surfaces) treat a hanging child facet against the cell
  of its parent facet; PMC / natural conditions on hanging facets are not needed.

## Alternatives considered
Conforming bisection (newest-vertex in 2D, Arnold–Mukherjee–Pouly in 3D) needs no
constraints but a different element pattern per dimension, a more delicate 3D closure and
no reuse of the red children; red–green closure in 3D is notoriously fragile. Hanging nodes
match the data structures planned since M1 (entity-based DoFs, `Constraints`) and keep the
mesh class, the DoF maps and the assembler unchanged apart from the hanging-entity tables.

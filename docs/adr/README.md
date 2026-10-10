# Architecture Decision Records

Decisions that are expensive to reverse are recorded here. Numbered, never deleted; a
superseded ADR gets a status line pointing to its successor.

Template:

```markdown
# NNNN — Title
**Status:** proposed | accepted | superseded by NNNN
**Date:** YYYY-MM-DD

## Context
## Decision
## Consequences
## Alternatives considered
```

| # | Title | Status |
|---|-------|--------|
| 0001 | Language, build system and core dependencies | accepted |
| 0002 | E-field curl–curl formulation and sign conventions | accepted |
| 0003 | Entity orientation by global vertex order | accepted |
| 0004 | Hierarchical Schöberl–Zaglmayr Nédélec basis | accepted |
| 0005 | PML as complex material tensor inside the FEM | accepted |
| 0006 | Local h-refinement: red refinement, hanging nodes, constraints by interpolation | accepted |
| 0007 | Direct solver backends behind one interface (SparseLU, MUMPS) | accepted, amended by 0008, 0012 |
| 0008 | GPU backend: cuDSS direct solver behind `LinearSolver`, no GPU assembly | accepted |
| 0009 | Layered background for the scattered-field formulation | accepted |
| 0010 | Axisymmetric (body-of-revolution, 2.5D) Maxwell solver | accepted |
| 0011 | Shape derivatives by the discrete adjoint on the mesh | accepted |
| 0012 | Optimisation, calibration and UQ in `hpfem.opt` on the discrete sensitivities | accepted |
| 0013 | Dipole emitters in periodic structures by array scanning of the conical cell problem | accepted |
| 0014 | Layered background for the axisymmetric solver | accepted |

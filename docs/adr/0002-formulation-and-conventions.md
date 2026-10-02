# 0002 — E-field curl–curl formulation and sign conventions
**Status:** accepted · **Date:** 2026-10-02

## Decision
- Time dependence $e^{-i\omega t}$ (physics convention), hence $\operatorname{Im}\varepsilon>0$
  for loss and PML stretch $s = 1 + i\sigma/\omega$.
- Primary unknown: electric field $\mathbf E$ in second-order form; $\mathbf H$ by post-processing.
- SI units in the C++ core; unit conversion only in the Python layer.
- Test functions are complex-conjugated in the weak form; system matrices are
  complex-symmetric, not Hermitian.

## Consequences
Every solver, estimator and post-processor assumes these; a change requires a new ADR and
a sweep of all theory pages. See `docs/theory/maxwell.md`.

## Alternatives considered
Engineering convention $e^{+j\omega t}$ (used by many EM textbooks and COMSOL): rejected to
match the optics literature and JCMsuite-style inputs ($n + i\kappa$ with $\kappa>0$).

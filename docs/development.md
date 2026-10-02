# Development guide

Everything binding is in `CLAUDE.md` at the repository root (conventions, workflow,
Definition of Done). This page only adds pointers:

- **Adding a module**: create `include/hpfem/<m>/`, `src/<m>/`, `tests/unit/<m>/`, add
  sources to `src/CMakeLists.txt`, add a row to `docs/architecture.md`.
- **Adding a dependency**: `cmake/Dependencies.cmake` + README table + ADR note.
- **Adding a convergence test**: `tests/convergence/`, print a DoF/error/rate table,
  assert on the rate (tolerance ≈ 0.2), register with label `convergence`.
- **Writing docs**: MathJax is enabled; keep formulas consistent with `docs/theory/`.

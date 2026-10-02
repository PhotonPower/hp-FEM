# Contributing

1. Read `CLAUDE.md` (conventions, workflow, Definition of Done).
2. Pick an item from `docs/roadmap.md` or open an issue first for anything larger.
3. Branch from `main` (`feat/…`, `fix/…`, `docs/…`), use Conventional Commits.
4. Run `./scripts/format.sh`, build with `--preset release` and `--preset asan`, run `ctest`.
5. Open a PR using the template; CI must be green.

Numerical changes without a convergence or regression test will not be merged.

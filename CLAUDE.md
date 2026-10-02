# CLAUDE.md — hp-FEM

This file is the operating manual for Claude Code (and humans) working in this repository.
Read it fully before touching code. When in doubt, follow this file, then the ADRs in
`docs/adr/`, then `docs/roadmap.md`.

## 1. What this project is

**hp-FEM** is an open-source finite-element suite for **nano-optics / computational
electromagnetics**: time-harmonic Maxwell solvers with adaptive *hp*-refinement,
high-order Nédélec (edge) elements, rigorous a-posteriori error estimation and
perfectly matched layers (PML). The scientific target is the feature set of commercial
tools such as JCMsuite, applied to:

- scatterometry / EUV-lithography masks (periodic gratings, Bloch BCs)
- metasurfaces and meta-lenses (unit-cell simulation, phase maps)
- photovoltaics (light trapping, absorption → heat as later multiphysics step)
- integrated photonics (waveguide modes, ring resonators, directional couplers)
- VCSELs, LEDs, OLEDs (resonance/eigenmode problems, outcoupling)
- plasmonics and biosensors (field singularities at metal corners)
- quantum optics (quantum dots in cavities, Purcell factor)

Core library: **C++20**. User API: **Python** (pybind11). Build: **CMake ≥ 3.25 + Ninja**.
See `docs/adr/0001-language-and-stack.md` for the rationale.

## 2. Non-negotiable principles

1. **Correctness before performance.** Every numerical feature ships with a
   *convergence test* against an analytic or manufactured solution. No exceptions.
2. **No spurious modes.** H(curl)-conforming Nédélec elements for all vector fields.
   Never use nodal Lagrange elements for E or H.
3. **Exponential convergence is the goal.** Design data structures so that both
   *h*-refinement (hanging nodes / constrained DoFs) and *p*-refinement (variable
   polynomial order per element, hierarchical bases) are first-class citizens from
   the start — retrofitting *p* into an *h*-only code is extremely painful.
4. **Everything is a complex number.** Fields, permittivities, PML stretches and
   eigenvalues are `std::complex<double>`. Real-valued shortcuts are an optimization
   that must be opt-in and tested, never the default.
5. **Dimension-agnostic where possible.** 2D (triangles) and 3D (tetrahedra) share
   the same abstractions; 2D is the fast test bed, 3D is the product.
6. **Small, reviewable commits** following the workflow in §7. Keep CI green.

## 3. Repository layout

```
hp-FEM/
├── CLAUDE.md                 ← you are here
├── CMakeLists.txt            ← top-level build (library + tests + bindings)
├── CMakePresets.json         ← debug / release / asan / coverage presets
├── cmake/                    ← compiler warnings, dependency fetching
├── include/hpfem/            ← public C++ headers (one subdir per module)
│   ├── core/                 ← types, errors, logging, units, numeric helpers
│   ├── mesh/                 ← topology, geometry, refinement, Gmsh I/O
│   ├── fespace/              ← reference elements, Nédélec/Lagrange bases, DoF maps
│   ├── assembly/             ← quadrature, local matrices, global sparse assembly
│   ├── materials/            ← ε(ω), μ, dispersion models, material library
│   ├── pml/                  ← complex coordinate stretching, adaptive PML
│   ├── solvers/              ← linear (direct/iterative), eigen, parameter sweeps
│   ├── adaptivity/           ← residual estimators, marking, hp-decision
│   ├── physics/              ← problem definitions: scattering, eigenmode, propagating-mode
│   └── io/                   ← VTK/XDMF export, JSON config, checkpoints
├── src/                      ← implementations, mirroring include/hpfem/
├── python/hpfem/             ← Python package + pybind11 bindings
├── tests/
│   ├── unit/                 ← fast, isolated (Catch2)
│   ├── convergence/          ← analytic benchmarks, run in CI (minutes)
│   └── regression/           ← golden-file tests for full pipelines
├── benchmarks/               ← performance benchmarks (not run in CI)
├── examples/                 ← one directory per application area
├── docs/                     ← MkDocs site: theory, architecture, ADRs, roadmap
├── scripts/                  ← dev helpers (format, setup, release)
└── .github/workflows/        ← CI (build/test matrix), docs deployment
```

Rule: **a header in `include/hpfem/<module>/` has its implementation in
`src/<module>/` and its tests in `tests/unit/<module>/`.** Keep modules dependency-
ordered: `core ← mesh ← fespace ← assembly ← {materials, pml} ← solvers ←
adaptivity ← physics ← io ← python`. Never introduce a cyclic include.

## 4. Build, test, run

```bash
# one-time
./scripts/setup-dev.sh                       # installs toolchain + pre-commit hooks (Ubuntu)

# configure + build (Ninja, release with asserts)
cmake --preset release
cmake --build --preset release

# tests
ctest --preset release                        # unit + convergence
ctest --preset release -L unit                # only fast unit tests
ctest --preset release -R nedelec             # by name

# debug / sanitizers / coverage
cmake --preset asan && cmake --build --preset asan && ctest --preset asan   # unit tests only
cmake --preset coverage && cmake --build --preset coverage && ctest --preset coverage

# python package (editable)
pip install -e ".[dev]"
pytest python/tests

# docs
pip install -r docs/requirements.txt && mkdocs serve

# formatting / linting (also runs via pre-commit)
./scripts/format.sh          # clang-format + ruff
```

CMake options: `HPFEM_BUILD_TESTS`, `HPFEM_BUILD_EXAMPLES`, `HPFEM_BUILD_PYTHON`,
`HPFEM_BUILD_BENCHMARKS`, `HPFEM_BUILD_DOCS`, `HPFEM_ENABLE_MPI` (off), `HPFEM_ENABLE_MUMPS`
(preset `mumps`), `HPFEM_ENABLE_OPENMP` (on by default).

Third-party libraries are pulled via `FetchContent` in `cmake/Dependencies.cmake`
(Eigen, Catch2, fmt, spdlog, nlohmann_json, pybind11). Heavy optional deps (MUMPS,
PETSc/SLEPc, Gmsh SDK) are found with `find_package` and guarded by options.

### 4a. Local development on Windows (MSYS2-GCC UCRT64 + CMake + Ninja)

The maintainer develops natively on Windows. Toolchain: **MSYS2 GCC (UCRT64)**,
CMake and Ninja (installed via winget), Git for Windows. **Not MSVC.**

- Use a shell where `C:\msys64\ucrt64\bin` is on `PATH` (`g++ --version` must
  report the MSYS2 GCC). Do **not** build from a Visual Studio *Developer / Cross
  Tools Command Prompt* — CMake may then pick `cl.exe`.
- If CMake picked the wrong compiler, delete `build/<preset>` and reconfigure with
  `CC=gcc CXX=g++` set in the environment.
- `./scripts/setup-dev.sh` is Ubuntu-only. On Windows, install extra tools from the
  *MSYS2 UCRT64* shell, e.g.
  `pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-gdb mingw-w64-ucrt-x86_64-clang-tools-extra`.
- Build and test exactly as above; the presets already use Ninja:

  ```bash
  cmake --preset release
  cmake --build --preset release
  ctest --preset release
  ```

  Without presets (equivalent quick build):

  ```bash
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
  cmake --build build
  ```

- The `asan` test preset runs the unit tests only (label `unit`): the convergence
  tests take hours under sanitizers at -O0 and exercise no code the unit tests do not;
  they run in the release configurations.
- The `asan` preset does not work with MinGW GCC (no AddressSanitizer on
  Windows/MinGW). Sanitizer runs happen in CI (Ubuntu) or in the devcontainer/WSL.
- The `fast` preset uses `-march=native`; binaries are not portable to other CPUs.
- MUMPS backend (`HPFEM_ENABLE_MUMPS`, preset `mumps`): install
  `pacman -S mingw-w64-ucrt-x86_64-mumps` in the UCRT64 shell; the test targets get
  `C:\msys64\ucrt64\bin` (the DLLs) prepended to `PATH` automatically, programs started
  by hand need it on `PATH`. On Ubuntu: `libmumps-seq-dev libmumps-headers-dev`.
- **GCC runtime is linked statically on MinGW** (`HPFEM_STATIC_RUNTIME`, default ON):
  Git for Windows ships an older `libstdc++-6.dll` in `C:\Program Files\Git\mingw64\bin`,
  which Git Bash (and tools started from it) puts *before* `C:\msys64\ucrt64\bin` on
  `PATH`. A dynamically linked test exe then fails with exit code `0xc0000139` /
  "Der Prozedureinsprungpunkt … wurde nicht gefunden" although the build succeeded.
  With the static runtime the binaries import only Windows system DLLs. If you switch the
  option off, run tests from PowerShell or the MSYS2 UCRT64 shell, or prepend
  `/c/msys64/ucrt64/bin` to `PATH` in Git Bash.
- CI runs on Ubuntu only. Code must stay portable: no Windows-only headers or
  APIs in the library, use `std::filesystem` for paths, and keep line endings
  as configured in `.editorconfig`.

## 5. Coding conventions (C++)

- **Standard:** C++20. No compiler extensions. Must compile with GCC ≥ 12, Clang ≥ 15.
- **Namespaces:** everything in `hpfem::`, submodules `hpfem::mesh`, `hpfem::fespace`, …
- **Naming:** `snake_case` for functions/variables/files, `PascalCase` for types,
  `kConstant` for compile-time constants, `member_` trailing underscore for private
  members. Template parameters `PascalCase`. Dimension template parameter is `Dim`.
- **Types:** `Real = double`, `Complex = std::complex<double>`, `Index = std::int64_t`,
  `LocalIndex = std::int32_t` — all in `include/hpfem/core/types.hpp`. Use them.
  Dense linear algebra: Eigen. Sparse: Eigen::SparseMatrix<Complex, RowMajor>
  wrapped behind `hpfem::SparseMatrix` so backends can be swapped.
- **Ownership:** value semantics by default; `std::unique_ptr` for polymorphic
  ownership; `std::shared_ptr` only when shared ownership is semantically required
  (meshes shared by several spaces). Never raw `new`/`delete`.
- **Errors:** throw `hpfem::Error` (or subclasses) with a message that names the
  offending entity. Use `HPFEM_ASSERT(cond, msg)` for invariants (active in Debug and
  Release-with-asserts). Never `assert()` directly.
- **Logging:** `spdlog` via `hpfem::log()`; levels: trace (per-element), debug
  (per-refinement-step), info (per-solve), warn, error.
- **Headers:** `#pragma once`, include-what-you-use, no `using namespace` in headers.
  Doxygen `///` comments on every public symbol: what it does, units, preconditions,
  complexity if non-obvious, and a `@f$ ... @f$` formula where the math matters.
- **Formatting:** `.clang-format` (LLVM-based, 100 cols). CI fails on unformatted code.
- **Const / noexcept / [[nodiscard]]** wherever applicable. Mark functions that
  compute something `[[nodiscard]]`.
- **No** global mutable state, no singletons (except the logger), no exceptions in
  hot loops, no `std::endl`, no `printf` debugging left behind.

## 6. Mathematical & physical conventions (binding for all code and docs)

- Time dependence **`exp(-iωt)`**. A lossy medium therefore has **Im(ε) > 0**.
  A PML stretch is `x̃ = x + (i/ω) ∫σ(s) ds`. State this convention in every file
  where a sign would otherwise be ambiguous.
- Units: **SI** internally (metres, seconds, Farad/m, …). The Python layer may
  accept `nm`, `µm`, `eV`, `THz` and converts at the boundary. Vacuum constants live
  in `core/constants.hpp`: `c0`, `eps0`, `mu0`, `Z0`.
- Primary unknown: **electric field E** in second-order form
  `curl(μ⁻¹ curl E) − ω² ε E = iω J`. The magnetic field is post-processed.
  (Exception: a dual H-formulation may be added later for error-bound purposes.)
- Permittivity `ε(ω)` is a complex tensor (3×3 diagonal or full) per material;
  anisotropy and dispersion models (Drude, Lorentz, tabulated n,k) live in
  `materials/`.
- Weak form: find `E ∈ H(curl)` such that for all `v ∈ H(curl)`
  `(μ⁻¹ curl E, curl v) − ω²(ε E, v) + ⟨BC terms⟩ = (iω J, v)`.
- Elements: **Nédélec first kind**, hierarchical (Schöberl–Zaglmayr style) basis so
  that order can vary per edge/face/cell. Orientation of shared edges/faces is
  resolved by **global vertex-number ordering** (lowest global vertex index first).
- Quadrature: Gauss–Jacobi on simplices, order ≥ 2p+1 for mass terms; dedicated
  rules for curved elements and PML (high order, because stretched coefficients are
  non-polynomial).
- Reference elements: unit triangle `(0,0),(1,0),(0,1)` and unit tetrahedron
  `(0,0,0),(1,0,0),(0,1,0),(0,0,1)`; local edge/face numbering is **fixed in
  `mesh/simplex_topology.hpp` (`SimplexTopology<Dim>`, re-used by
  `fespace/reference_element.hpp`) and documented in `docs/theory/nedelec.md`** —
  never redefine it elsewhere. Global entity numbering and orientation flags:
  `docs/theory/mesh.md`.
- Boundary conditions: PEC (tangential E = 0, via DoF elimination), PMC (natural),
  Bloch-periodic (`E(x+a) = E(x)·exp(ik·a)` via constrained DoFs), transparent
  (PML), and impedance/Robin as first-order absorbing fallback.
- Error estimator: residual-based, element-wise `η_K² = h_K² ‖R_K‖² + h_K ‖J_F‖²`
  with the curl-jump and normal-flux-jump face residuals; marking by Dörfler
  (bulk) criterion; *hp*-decision via smoothness indicator (Legendre-coefficient
  decay) — see `docs/theory/hp-adaptivity.md`.

## 7. Development workflow

- **Branches:** `main` should stay buildable. Direct commits and pushes to `main` are
  allowed (single-maintainer project). Before pushing to `main`, build and run the
  tests locally (`cmake --build --preset release && ctest --preset release`).
  For larger or riskier work, use `feat/<topic>`, `fix/<topic>`, `docs/<topic>`
  branches with a PR so CI runs before merging.
- **Commits:** [Conventional Commits](https://www.conventionalcommits.org):
  `feat(fespace): add order-2 Nédélec basis on tetrahedra`,
  `test(mesh): verify edge orientation consistency`, `docs(adr): …`.
- **Pull requests:** fill the template; CI must pass (build matrix, tests, format,
  clang-tidy, docs build). Link the roadmap item.
- **Definition of done** for any feature:
  1. unit tests in `tests/unit/`
  2. a convergence or regression test if numerics are involved
  3. Doxygen on public API + a docs page or section in `docs/`
  4. `docs/roadmap.md` checkbox ticked, `CHANGELOG.md` entry under *Unreleased*
  5. if an architectural decision was made: a new ADR in `docs/adr/`
- **Architecture Decision Records:** any choice that is expensive to reverse
  (data layout, basis family, solver interface, file format) gets an ADR
  (`docs/adr/NNNN-title.md`, template in `docs/adr/README.md`). Read existing
  ADRs before proposing a change; supersede, don't silently diverge.

## 8. Testing strategy

| Layer        | Location             | Runs in CI | Examples                                        |
|--------------|----------------------|-----------|-------------------------------------------------|
| unit         | `tests/unit/`        | always    | reference-element shape functions, DoF numbering, quadrature exactness, mesh topology |
| convergence  | `tests/convergence/` | always (≤ 10 min total) | h- and p-convergence rates vs. analytic solutions |
| regression   | `tests/regression/`  | always    | golden JSON results of example pipelines (tolerance-based) |
| benchmarks   | `benchmarks/`        | manual    | assembly throughput, solver scaling |

**Analytic benchmarks that must exist and stay green** (add as milestones land):

1. Lagrange Poisson on square/cube with manufactured solution — rate `p+1` in L2.
2. Maxwell cavity eigenvalues (PEC box, 2D & 3D) — must match `π²(m²+n²+l²)`,
   **zero spurious modes** (count eigenvalues near 0 → must equal 0 after gauge).
3. Plane wave in homogeneous medium with PML — reflection coefficient < 1e-6.
4. Mie scattering from a dielectric cylinder (2D) / sphere (3D) — scattering cross
   section vs. series solution.
5. Dielectric slab waveguide modes (2D) — effective index vs. transcendental eq.
6. Lamellar grating (periodic, Bloch) — efficiencies vs. RCWA reference.
7. hp-adaptivity on an L-shaped / re-entrant metal corner — exponential error decay
   in `#DoF^(1/3)` (2D) must be visible in the convergence plot.

Convergence tests print a table (DoF, error, rate) and assert on the rate with a
tolerance, never on absolute magic numbers.

## 9. Documentation

- `docs/` is an MkDocs-Material site (`mkdocs.yml` at root); math via MathJax.
- `docs/theory/` is the *single source of truth* for formulas and conventions.
  Code comments reference these pages instead of re-deriving.
- API docs: Doxygen (`Doxyfile`), published under `docs/api/` by CI.
- Every example in `examples/` has a README with: physics, expected result, runtime.
- Keep `docs/roadmap.md` current — it is the project's task list.

## 10. Things Claude Code must NOT do

- Never commit secrets, tokens, or `.env` files. Never print the GitHub token.
- Never commit generated files (`build/`, `*.msh` > 1 MB, `*.vtu`, `site/`).
  Large test meshes go through `tests/data/` with a generation script, not binaries.
- Never change the mathematical conventions in §6 without an ADR.
- Never silence a failing test by loosening tolerances without a documented reason.
- Never add a dependency without updating `cmake/Dependencies.cmake`, the README
  dependency table and an ADR note if the dependency is load-bearing.
- Never push to `main` with a failing local build or failing tests.

## 11. How to pick up work

1. `git pull`, read `docs/roadmap.md`, pick the lowest unchecked item of the
   current milestone (or the item you were asked to do).
2. Read the relevant `docs/theory/*.md` and the headers of the modules you touch.
3. Write the test first (shape of the API), then the implementation.
4. Run `./scripts/format.sh && cmake --build --preset release && ctest --preset release`.
5. Update docs/roadmap/changelog, open a PR.

## 12. Current status

Milestones **M0 (scaffold)**, **M1 (mesh infrastructure)** and **M2 (scalar FEM)** are
complete: build system, CI, docs; `mesh::Mesh<Dim>` with oriented entities,
connectivity, tags, Gmsh input, generators, affine and quadratic geometry, red
refinement, VTK export; reference element, Gauss–Jacobi quadrature, hierarchical H1
basis with `DofMap`, sparse assembly, Dirichlet elimination, SparseLU solver and the
Poisson convergence test. **M3 (Nédélec elements and Maxwell eigenproblems)** is complete
as well: hierarchical Nédélec basis with `NedelecDofMap`, curl–curl forms with complex
tensors, PEC/PMC, discrete gradients and the gauged shift-invert eigensolver (Spectra),
the PEC cavity convergence test, point location with field evaluation at arbitrary
points, and VTK export of fields (cell averages and subdivision). **M4 (time-harmonic
scattering and waveguides)** is complete: `physics::Scattering` (total / scattered
field, plane-wave and dipole sources, materials by tag), PML as stretched material
tensors, Bloch-periodic constraints, curved elements (second-order Gmsh input, disc /
ball generators), post-processing (fluxes, cross-sections, far field, diffraction
orders), `physics::PropagatingMode`, convergence tests #3–#6 and the four examples.
**M5 (adaptivity)** is complete: residual and dual-weighted (goal-oriented) estimators,
Dörfler marking, local h-refinement with hanging nodes (`mesh::AdaptiveMesh`,
`assembly::hanging_constraints`), p-refinement, hp decision by error prediction (and
coefficient decay), solution transfer, convergence test #7 (L-shape and plasmonic wedge,
exponential in $N^{1/3}$) and the `plasmonic_dimer` example.
Next: **M6 — solvers and performance**.
See `docs/roadmap.md`.

# hp-FEM

[![CI](https://github.com/PhotonPower/hp-FEM/actions/workflows/ci.yml/badge.svg)](https://github.com/PhotonPower/hp-FEM/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Adaptive **hp-finite-element** suite for nano-optics: time-harmonic Maxwell solvers with
high-order **Nédélec (edge) elements**, rigorous **a-posteriori error estimation**,
automatic **hp-refinement** and **perfectly matched layers**.

Target applications: scatterometry & EUV masks, metasurfaces, photovoltaics, integrated
photonics, VCSELs/LEDs, plasmonic sensors, quantum-dot emitters.

## Status
Milestone **M0 (scaffold)** done. See [docs/roadmap.md](docs/roadmap.md).

## Quick start
```bash
./scripts/setup-dev.sh                       # Ubuntu toolchain + pre-commit
cmake --preset release && cmake --build --preset release && ctest --preset release
pip install -e ".[dev]"                      # Python package
mkdocs serve                                 # documentation
```

## Repository
| Path | Content |
|---|---|
| `CLAUDE.md` | conventions, workflow, Definition of Done — read first |
| `include/hpfem`, `src` | C++20 core library |
| `python/` | pybind11 bindings + Python package |
| `tests/` | unit / convergence / regression tests (Catch2, pytest) |
| `docs/` | MkDocs site: theory, architecture, ADRs, roadmap |
| `examples/` | application examples |

## Dependencies
| Library | Purpose | How |
|---|---|---|
| Eigen 3.4 | dense/sparse linear algebra | system or FetchContent |
| fmt / spdlog | logging | FetchContent |
| nlohmann_json | configs, results | FetchContent |
| Spectra 1.2 | shift-invert Lanczos / Arnoldi eigensolvers (header-only) | FetchContent |
| Catch2 v3 | C++ tests | FetchContent |
| pybind11 | Python bindings | FetchContent / pip |
| Gmsh | mesh generation (external tool) | system |
| MUMPS 5 (sequential, complex) | optional direct solver backend (`HPFEM_ENABLE_MUMPS`, preset `mumps`) | `cmake/FindMUMPS.cmake`: apt `libmumps-seq-dev libmumps-headers-dev`, MSYS2 `mingw-w64-ucrt-x86_64-mumps` |
| PETSc/SLEPc, MPI | optional (later milestones) | `find_package`, CMake options |

## License
MIT — see [LICENSE](LICENSE).

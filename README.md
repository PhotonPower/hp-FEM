# hp-FEM

[![CI](https://github.com/PhotonPower/hp-FEM/actions/workflows/ci.yml/badge.svg)](https://github.com/PhotonPower/hp-FEM/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Docs](https://github.com/PhotonPower/hp-FEM/actions/workflows/docs.yml/badge.svg)](https://photonpower.github.io/hp-FEM/)

Adaptive **hp-finite-element** suite for nano-optics: time-harmonic Maxwell solvers with
high-order **Nédélec (edge) elements**, rigorous **a-posteriori error estimation**,
automatic **hp-refinement** and **perfectly matched layers**.

Target applications: scatterometry & EUV masks, metasurfaces, photovoltaics, integrated
photonics, VCSELs/LEDs, plasmonic sensors, quantum-dot emitters.

## Documentation
Theory, architecture, Python guide and API reference:
**<https://photonpower.github.io/hp-FEM/>** (built from `docs/` by MkDocs on every push to `main`).

## Status
Release **0.2.0**: milestones **M0–M9** are complete (mesh infrastructure, scalar FEM,
Nédélec elements and Maxwell eigenproblems, scattering and waveguides, hp-adaptivity, solvers,
Python API, application examples, multiphysics); only the optional MPI item of M6 is open.
See [docs/roadmap.md](docs/roadmap.md) and [CHANGELOG.md](CHANGELOG.md).

## Quick start
```bash
./scripts/setup-dev.sh                       # Ubuntu toolchain + pre-commit
cmake --preset release && cmake --build --preset release && ctest --preset release
pip install -e ".[dev]"                      # Python package (or a wheel from the wheels workflow)
pip install "hpfem[gui]" && hpfem-gui app.py # Streamlit front end with this interpreter
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
| OpenMP | parallel assembly and estimation (`HPFEM_ENABLE_OPENMP`, on by default) | compiler |
| MUMPS 5 (sequential, complex) | optional direct solver backend (`HPFEM_ENABLE_MUMPS`, preset `mumps`) | `cmake/FindMUMPS.cmake`: apt `libmumps-seq-dev libmumps-headers-dev`, MSYS2 `mingw-w64-ucrt-x86_64-mumps` |
| cuDSS 0.8 + CUDA 12/13 | optional GPU direct solver backend (`HPFEM_ENABLE_CUDA`); separate DLL `hpfem_gpu` built with nvcc, loaded at run time | `gpu/README.md` (not needed to build the library) |
| BoTorch | optional: multi-objective and multi-fidelity Bayesian optimisation (`hpfem.opt.pareto_optimize`, `multi_fidelity_optimize`; experimental) | pip extra `opt-bo` |
| PETSc/SLEPc, MPI | optional (later milestones) | `find_package`, CMake options |

## License
MIT — see [LICENSE](LICENSE).

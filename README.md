# hp-FEM

[![CI](https://github.com/PhotonPower/hp-FEM/actions/workflows/ci.yml/badge.svg)](https://github.com/PhotonPower/hp-FEM/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/PhotonPower/hp-FEM/blob/main/LICENSE)
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
Release **0.5.0**: milestones **M0–M20** are complete. Beyond the Maxwell core (mesh
infrastructure, Nédélec elements, scattering and waveguides, hp-adaptivity, solvers, Python API,
multiphysics) this covers validation against the literature, axisymmetric and conical solvers,
ports and sensitivities, optimisation and uncertainty quantification, dipole emitters in periodic
structures, anisotropic materials, and linear elasticity with eigenstrain. Open are the optional
MPI item and the full 3D ring resonator. See the
[roadmap](https://github.com/PhotonPower/hp-FEM/blob/main/docs/roadmap.md) and the
[changelog](https://github.com/PhotonPower/hp-FEM/blob/main/CHANGELOG.md).

## Installation
```bash
pip install hpfem            # binary wheels: Linux x86_64, Windows x86_64, macOS arm64 (>= 14)
pip install "hpfem[gui]"     # plus the Streamlit front end (hpfem-gui)
hpfem info                   # version, threads, solver backends
hpfem reference              # compact command reference of the Python API (for agents)
```
The wheels use the SparseLU direct solver (`hpfem info` shows the backends and OpenMP threads);
the MUMPS and cuDSS backends need a source build. Other platforms build from the sdist (C++20 compiler, CMake ≥ 3.25, Ninja).

## Quick start (development)
```bash
./scripts/setup-dev.sh                       # Ubuntu toolchain + pre-commit
cmake --preset release && cmake --build --preset release && ctest --preset release
pip install -e ".[dev]"                      # Python package from the source tree
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
| Boost.Math 1.86 (standalone) | Bessel functions where the standard library lacks the C++17 special functions (libc++, Apple Clang); not used with libstdc++ / MSVC | FetchContent, only if needed |
| Catch2 v3 | C++ tests | FetchContent |
| pybind11 | Python bindings | FetchContent / pip |
| Gmsh | mesh generation (external tool) | system |
| OpenMP | parallel assembly and estimation (`HPFEM_ENABLE_OPENMP`, on by default) | compiler |
| MUMPS 5 (sequential, complex) | optional direct solver backend (`HPFEM_ENABLE_MUMPS`, preset `mumps`) | `cmake/FindMUMPS.cmake`: apt `libmumps-seq-dev libmumps-headers-dev`, MSYS2 `mingw-w64-ucrt-x86_64-mumps` |
| cuDSS 0.8 + CUDA 12/13 | optional GPU direct solver backend (`HPFEM_ENABLE_CUDA`); separate DLL `hpfem_gpu` built with nvcc, loaded at run time | `gpu/README.md` (not needed to build the library) |
| BoTorch | optional: multi-objective and multi-fidelity Bayesian optimisation (`hpfem.opt.pareto_optimize`, `multi_fidelity_optimize`; experimental) | pip extra `opt-bo` |
| emcee 3 | optional: MCMC sampling of the parameter posterior (`hpfem.opt.sample`); in the `dev` extra for the tests | pip extra `opt-mcmc` |
| PETSc/SLEPc, MPI | optional (later milestones) | `find_package`, CMake options |

## License
MIT — see [LICENSE](https://github.com/PhotonPower/hp-FEM/blob/main/LICENSE).

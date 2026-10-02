# Python API

The Python package `hpfem` wraps the C++ core with pybind11 (`python/bindings/`). It
exposes the same pipeline as the C++ examples — mesh, DoF maps, materials and PML,
problems, solvers, post-processing, adaptivity, export — with NumPy / SciPy types at the
boundary. The C++ library stays the single implementation; the bindings add no numerics.

## Installation

```bash
pip install -e ".[dev]"        # builds the extension with scikit-build-core (CMake + Ninja)
pytest python/tests
```

For development against an existing CMake build: `cmake --preset python`,
`cmake --build --preset python`, then copy `build/python/python/_hpfem*.{so,pyd}` next to
`python/hpfem/__init__.py` (git ignores it) and run `pytest` from `python/`. On Windows
the extension must be built for the interpreter CMake found; with the MSYS2 toolchain that is
`C:\msys64\ucrt64\bin\python3` (install `mingw-w64-ucrt-x86_64-python-numpy`, `-scipy`,
`-pytest` with pacman).

## Conventions

- **Units are SI** and the time dependence is $e^{-i\omega t}$ (CLAUDE.md §6): lengths in
  metres, `omega` in rad/s, fields in V/m, lossy media have `Im eps_r > 0`. `hpfem.units`
  (next roadmap item) converts nm, µm, eV and THz at the boundary.
- **Dimension by name.** Dimension-templated classes exist as `<Name>2D` / `<Name>3D`:
  `Mesh2D`, `NedelecDofMap3D`, `Scattering2D`, `PmlBox3D`, `FieldExporter2D`, … Free
  functions are overloaded on the argument type, so `refine_uniform(mesh)`,
  `extract(mesh, cells)`, `hp_refine(adaptive, …)`, `interpolate(dofs, g)` work for either
  dimension; `read_gmsh(file, scale, dim)` takes the dimension explicitly.
- **Arrays.** Bulk data (vertices, cells, tags, DoF lists, indicators, coefficient vectors)
  are NumPy arrays; sparse matrices are `scipy.sparse.csr_matrix` with complex entries;
  per-entity queries return lists. Inputs accept lists or arrays.
- **Callbacks.** Coefficients, sources and samplers are Python callables of the physical
  point (`lambda x: ...` returning a scalar, a length-`dim` vector or a tensor). They are
  called with the GIL held, also from the OpenMP loops of the assembly, so they are correct
  but slow; prefer the analytic sources (`plane_wave`, `dipole_field`) and materials by tag
  for anything in the hot path. A Python exception inside a callback propagates out of the
  parallel loop as the original exception.
- **Lifetimes.** Objects that reference others (DoF map → mesh, problem → DoF map,
  locator → mesh, exporter → mesh) keep them alive. `AdaptiveMesh.mesh` returns a *copy* of
  the current leaf mesh: take it once per step and build the DoF maps, locators and exporters
  of that step on that object (the C++ pointer identity checks require it).
- **Errors.** `hpfem::InvalidArgument` → `ValueError`, `hpfem::NotImplemented` →
  `NotImplementedError`, other `hpfem::Error` → `RuntimeError`.
- Long-running calls release the GIL; `set_num_threads` / `num_threads` control the OpenMP
  loops, `set_log_level("warn")` silences the per-solve logging.

## Units and materials

`hpfem.units` holds the multipliers (`nm`, `um`, `eV`, `THz`, `deg`, …): multiply to go
into SI, divide to come out, and converts between the spectral variables:

```python
from hpfem import units, materials
omega = units.angular_frequency(wavelength=633 * units.nm)   # or frequency=, energy=, wavenumber=
units.photon_energy(omega) / units.eV                         # 1.96
```

`hpfem.materials` provides dispersive materials — `Tabulated` n, k (linear interpolation in
wavelength, no extrapolation), `Sellmeier`, `DrudeLorentz` / `Drude`, `Constant` — whose
`at(omega)` returns the frequency-independent core `Material` for one solve, so a
frequency sweep re-sets the material per frequency. The library (`materials.library`,
`materials.get(name)`) carries its sources in `python/hpfem/data/*.csv` (refractiveindex.info
database, CC0):

| name | source | range |
|---|---|---|
| `Si` | Green 2008, intrinsic c-Si at 300 K | 0.25 – 1.45 µm |
| `SiO2` | Malitson 1965 Sellmeier, fused silica | 0.21 – 6.7 µm |
| `Au`, `Ag` | Johnson & Christy 1972 | 0.188 – 1.94 µm |
| `Al` | Rakić 1995 (Kramers–Kronig consistent) | 0.000124 – 200 µm |
| `TiO2` | Devore 1951 Sellmeier, rutile, ordinary ray | 0.43 – 1.53 µm |
| `GaAs` | Aspnes et al. 1986 | 0.207 – 0.827 µm |
| `MAPbI3` | Phillips et al. 2015, CH3NH3PbI3 perovskite film | 0.30 – 1.50 µm |
| `water` | Daimon & Masumura 2007 Sellmeier | 0.2 – 2.0 µm |
| `vacuum`, `air` | constants | — |

```python
setup.materials.set(2, materials.get("Au").at(omega))      # eps_r = (n + ik)^2, Im > 0
```

## Pipeline

```python
import numpy as np
import hpfem

# mesh with a curved circular inclusion (tag 2), space of order 3
mesh = hpfem.square_with_disc(4, radius=0.25, half_width=1.0, outer=2.0, inclusion_tag=2)
dofs = hpfem.NedelecDofMap2D(mesh, 3)

# scattering of a plane wave, scattered-field formulation, PML on all sides
k = 6.0
setup = hpfem.ScatteringSetup2D()
setup.omega = k * hpfem.constants.c0
setup.materials.set(2, hpfem.Material.dielectric(1.5))
setup.incident = hpfem.plane_wave([0.0, 1.0], [k, 0.0])
setup.formulation = hpfem.Formulation.SCATTERED_FIELD
setup.pml = hpfem.PmlBox2D.uniform([-1, -1], [1, 1], 1.0, k)
setup.pec_tags = [hpfem.box_tag.X_MIN, hpfem.box_tag.X_MAX, hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX]
problem = hpfem.Scattering2D(dofs, setup)
solution = problem.solve()

# post-processing: cross-sections, far field, field values, VTK export
surface = hpfem.Surface2D.around_cells(mesh, 2)
cs = hpfem.cross_sections(problem, solution, surface, incident_amplitude=1.0)
print(cs.scattering, hpfem.mie_cylinder_scattering_width(k, 0.25, 1.5))
far = hpfem.FarField2D(mesh, surface, hpfem.discrete_field(dofs, solution.unknown),
                       setup.omega, hpfem.Material.vacuum(), order=8)
pattern = far.pattern([1.0, 0.0])
locator = hpfem.PointLocator2D(mesh)
E = problem.total_field(solution, locator, [0.5, 0.1])      # None outside the mesh
hpfem.FieldExporter2D(mesh, 3).hcurl("E", dofs, solution.unknown).write("mie.vtu")

# adaptivity: estimate, mark, refine (h or p), transfer
estimate = problem.estimate(solution)
marked = hpfem.dorfler_marking(estimate.indicators, 0.5)
```

The hp loop (`tests/convergence/adaptive_hp_refinement.cpp`) reads the same in Python:

```python
adaptive = hpfem.AdaptiveMesh2D(root_mesh)
orders = np.ones(adaptive.mesh.num_cells, dtype=int)
predicted = np.zeros(0)
for step in range(20):
    mesh = adaptive.mesh                        # copy of the current leaf mesh
    dofs = hpfem.NedelecDofMap2D(mesh, orders.tolist())
    problem = hpfem.Scattering2D(dofs, setup)
    solution = problem.solve()
    estimate = problem.estimate(solution)
    marked = hpfem.dorfler_marking(estimate.indicators, 0.5)
    decision = hpfem.hp_decide_by_prediction(estimate.indicators, predicted, marked)
    hp = hpfem.hp_refine(adaptive, orders.tolist(), decision.h_marked, decision.p_marked)
    predicted = hpfem.predict_indicators(estimate.indicators, orders.tolist(), hp)
    orders = hp.orders
```

Other entry points follow the C++ headers one to one: `assemble_h1` / `assemble_maxwell`
with `ScalarForm2D` / `MaxwellForm2D` callbacks and `solve_direct` or `make_direct_solver`
(SparseLU / MUMPS), `dirichlet_values` / `apply_dirichlet`, `gauged_curl_curl_eigenpairs`
for cavity modes, `PropagatingMode` for waveguide cross-sections, `ScatteringOperator2D` /
`plane_wave_sweep` / `ReducedBasis` for sweeps, `bloch_constraints` / `PeriodicPair2D` /
`fourier_coefficients` / `diffraction_efficiencies` for gratings, `dwr_estimate` with
`point_value_functional` or a Python functional for goal-oriented estimation, `VtkWriter2D`
for meshes with data arrays. `help(hpfem.<name>)` shows the bound signature and docstring.

## Tests

`python/tests/` runs in CI (`pip install ".[dev]" && pytest python/tests`, under a minute):
mesh and DoF-map invariants, Poisson with manufactured solution (rate $p+1$), PEC cavity
eigenvalues (no spurious modes), the Mie cylinder cross-section against the series, the
slab waveguide effective index, the lamellar grating's energy balance, the hp loop on the
L-shape, the goal-oriented estimate, callbacks raising inside parallel loops, VTK export.
Convergence is established by the C++ suite; the Python tests check that the bound pipeline
reproduces those numbers.

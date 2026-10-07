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

## Project files and the command line

A simulation can be described without Python in a JSON (or YAML) project file and run with
`hpfem run project.json -o results.json` (also `python -m hpfem run …`); `hpfem validate`
builds the mesh and the setups without solving, `hpfem info` and `hpfem materials` print
the build and the material library. `examples/*/project.json` hold the four examples;
the Mie cylinder reads:

```json
{
  "problem": "scattering", "dim": 2, "length_unit": "m",
  "mesh": {"type": "square_with_disc", "n": 4, "radius": 0.25, "half_width": 1.0, "outer": 2.0, "inclusion_tag": 2},
  "order": 3,
  "wavelength": {"value": 1.0471975511965976, "unit": "m"},
  "materials": {"background": "vacuum", "2": {"n": 1.5}},
  "formulation": "scattered_field",
  "source": {"type": "plane_wave", "angle": 0.0, "angle_unit": "deg", "polarisation": "TE"},
  "pml": {"lower": [-1, -1], "upper": [1, 1], "thickness": 1.0, "profile": {"order": 2, "reflection": 1e-10}},
  "boundaries": {"pec": ["x_min", "x_max", "y_min", "y_max"]},
  "outputs": {"cross_sections": {"around_tag": 2}, "far_field": {"around_tag": 2, "directions": 36},
              "points": [[0.5, 0.1]], "vtk": {"file": "mie_cylinder.vtu", "subdivisions": 3}}
}
```

Keys (see the docstring of `hpfem.project` for the full list): `problem` (`scattering`,
`waveguide`, `cavity`), `mesh.type` (`rectangle`, `box`, `disc`, `ball`, `square_with_disc`,
`gmsh` with `file`; `regions` re-tag cells whose centroid lies in a box), `wavelength` /
`frequency` / `energy` with a `unit` (a list runs a sweep), `materials` by tag (library
name, `{"n": 1.5}`, `{"eps_r": [re, im]}`), `source` (`plane_wave` with `angle` in 2D or
`direction` + `polarisation` in 3D, `dipole`), `pml`, `boundaries.pec` / `incident` (side
names `x_min` … `z_max` or tags), `periodic` (Bloch phase from the source), `solver`,
`outputs` (`cross_sections`, `far_field`, `points`, `flux`, `diffraction`, `estimate`,
`vtk`). `outputs.diffraction` takes, besides the `x_above` / `x_below` lines, a `line` of any
orientation (`origin`, `tangent`, `normal`, optional `index` and `subtract_incident`) whose
orders are those of the total field with the incident wave subtracted, and a `balance`
(`axis`, `reflection` and optional `transmission` plane coordinates) that writes the
flux-based energy balance with its relative residual (`hpfem.diffraction_orders`,
`hpfem.power_balance`; on a layered background set `setup.incident_wave` to the stack's
`incident_wave`). The results are JSON with one entry per spectral point; from Python,
`hpfem.project.run(spec)` takes the same mapping.

## Notebooks

`examples/notebooks/` holds Jupyter notebooks over this API (Mie cylinder, hp-adaptivity on
the L-shape, gold nanowire spectrum with the material library and a project-file sweep).
They are committed without outputs and executed by the test suite, so they double as
living documentation.

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
values, cells = problem.sample(solution, locator, points)   # (n, 2) complex, (n,) cells; NaN / -1 outside
h_z, _ = problem.sample(solution, locator, points, quantity="H")   # "E" (default), "H" or "S" (Poynting)
tri = problem.triangulate(solution, subdivisions=3)         # points, simplices, values, cell, tag
absorbed = hpfem.absorbed_power_by_tag(problem, solution)   # .total, .by_tag {tag: W/m}, .per_cell
density = hpfem.absorption_density(problem, solution)       # Joule heating at the quadrature points
# matplotlib.tri.Triangulation(tri.points[:, 0], tri.points[:, 1], tri.simplices)
hpfem.FieldExporter2D(mesh, 3).hcurl("E", dofs, solution.unknown).write("mie.vtu")

# adaptivity: estimate, mark, refine (h or p), transfer
estimate = problem.estimate(solution)
marked = hpfem.dorfler_marking(estimate.indicators, 0.5)
```

## Job runner (`python -m hpfem.run`)

`python -m hpfem.run job.json [--out DIR] [--cancel-file F] [--threads N] [--quiet]` runs a
grating job described by a JSON document (schema version 1: `model` as a `UnitCell`, `mesh`
as `structured` / `gmsh` / `file`, `materials` by library name or `{"eps"}` / `{"n"}`,
`stack`, `incidence`, `sweep` over wavelengths or angles, `solver`, `maps`; the full example
is in the module docstring) and streams JSON-lines events on stdout: `start` (with
`version_info()`), `mesh` (the report), `diagnostics` (when they change), one `point` per
sweep point with R, T, A, the balance, the orders and the timing, `map` per field map
(`maps_<point>_<map>.npz` with `x`, `y`, `values`), `cancelled`, `error` and `done`. The
results go to `results.json`. SIGTERM / SIGINT or the appearance of the cancel file stop the
run after the current point (exit code 2; 1 on an error). From Python,
`hpfem.run.run_job(job, out_dir, emit, cancel)` does the same with callbacks;
`hpfem.version_info()` reports version, platform, OpenMP, threads, backends and whether gmsh
is importable. `hpfem.meshing.structured_unit_cell(cell, nx, rows)` meshes a `UnitCell`
without Gmsh (columns over the period, rows of cells stacked from the bottom, tags by
priority), which the runner uses for `"mesh": {"structured": ...}`.

## Diagnostics (`hpfem.diagnostics`)

`hpfem.grating.validate(...)` (the arguments of `solve`) and
`hpfem.diagnostics.validate_scattering(mesh, setup, orders, materials=None)` return a list of
`Diagnostic(code, severity, text, hint)` with stable codes a GUI can translate:
`mesh_invalid_cells`, `mesh_poor_angles`, `mesh_untagged_cells`, `tag_without_material`,
`material_without_cells`, `interface_off_mesh`, `periodic_partner_missing`,
`periodic_faces_differ`, `pml_under_resolved`, `pml_thin`, `pml_missing`,
`too_few_elements_per_wavelength`, `material_out_of_range` (tabulated data outside its
range, before anything is assembled), `lossy_incidence_medium`, `grazing_order` (an order
propagating at more than 80° from the normal), `pec_wall_too_close` (a PEC wall closer than
six field decay lengths in a lossy substrate, with the amplitude fraction reaching it) and
`setup` (a failing set-up). `grating.solve` runs them first (`check=True`): errors raise
`GratingError`, warnings and infos land in `result.diagnostics`;
`diagnostics.raise_on_errors(found)` does the same for the generic validator. The individual
checks (`validate_mesh`, `validate_interfaces`, `validate_periodic`, `validate_pml`,
`validate_resolution`, `validate_materials`, `validate_stack`, `validate_orders`,
`validate_bottom_wall`) are available on their own. `grating.solve` also accepts dispersive
materials (`hpfem.materials.get("Ag")`) in its material dict and evaluates them at `omega`.

## Unit-cell meshing (`hpfem.meshing`)

`hpfem.meshing` builds the unit cell of a grating or metasurface with the Gmsh Python API
(`pip install gmsh`): a `UnitCell(period, y_bottom, y_top, slabs=[Slab(tag, y0, y1), ...],
shapes=[Shape(kind, tag, params), ...], background_tag=...)` with slabs over the full period
(substrate, films, cover; the PML space is part of `y_bottom` / `y_top`) and scatterers of
kind `"rectangle"`, `"trapezoid"`, `"ellipse"` or `"polygon"` that are copied by ±period and
clipped to the cell; a shape wins over a slab, a later shape over an earlier one
(`cell.tag_at(x, y)`). `element_sizes(materials, omega, p)` gives the size per tag from
`12 / p` elements per local wavelength and, in absorbing media, the field decay length
`1 / (k0 Im n)`. `unit_cell_mesh(cell, sizes, interface_factor=0.5, periodic=True, order=None)`
writes the MSH 4.1 file (physical groups: surfaces = material tags, curves = the four sides
with the `box_tag` numbers, `$Periodic` for the x faces) and reads it back as
`(mesh, periodic_links)`; curved cells (order 2) are used when a shape is an ellipse. The
model is built in nanometres (OpenCASCADE's absolute tolerances) and read with the matching
scale. `meshing.report(mesh, materials)` adds `tags_without_material` and
`materials_without_cells` to `hpfem.mesh_report`, and `hpfem.check_periodic` checks a pair
of faces.

```python
cell = meshing.UnitCell(period=400 * nm, y_bottom=-800 * nm, y_top=950 * nm,
                        slabs=[meshing.Slab(SUB, -800 * nm, 0.0)],
                        shapes=[meshing.Shape("rectangle", RIDGE, dict(x=-100 * nm, y=0.0,
                                                                          width=200 * nm, height=148 * nm))],
                        background_tag=AIR)
sizes = meshing.element_sizes({AIR: air, SUB: glass, RIDGE: glass}, omega, p=3)
mesh, links = meshing.unit_cell_mesh(cell, sizes)
result = hpfem.grating.solve(mesh, {AIR: air, SUB: glass, RIDGE: glass}, stack, "p", theta, phi, omega)
```

## Gratings in one call (`hpfem.grating.solve`)

`hpfem.grating.solve(mesh, materials, stack, polarisation, theta, phi, omega, order=4, *,
pml=None, bottom="pml", orders_max=3, ...)` runs the conical solver on a grating unit cell
(period along x, stack normal along y, Bloch faces tagged `box_tag.X_MIN` / `X_MAX`, PML
regions at the top and bottom of the mesh) and returns a `GratingResult`: `R_orders` and
`T_orders` (per order: efficiency, complex vector amplitude, tangential and normal
wavenumber, propagating flag), the sums `R`, `T`, the absorbed fraction `A` with `A_by_tag`,
`power_balance_residual = R + T + A − 1`, the bare-stack `wave`, the `problem` and `solution`,
`field(points)` (vectorised sampling of E, H or S) and a `timing` dict. It snaps mesh
vertices onto the stack interfaces (`snap_tolerance` × period), designs the PML from the
largest propagating-order angle in cover and substrate (`PmlProfile.for_angle`, reference
index `min(n_cover, n_substrate)`, thickness half a local wavelength rounded to whole cells,
or `pml={"top": t, "bottom": t}`), places the measurement lines between the structure and
the PML, and raises `GratingError` with a hint for cells straddling an interface or bad
options. `bottom="pec"` ends a thick lossy substrate on the PEC wall without a bottom PML
(no transmitted orders).

```python
stack = hpfem.LayerStack2D(hpfem.Material.dielectric(1.0), [], glass, 0.0)   # air over glass
result = hpfem.grating.solve(mesh, {SUB: glass, RIDGE: glass}, stack, "p",
                             50 * units.deg, 30 * units.deg, omega, order=4)
print({o.m: o.efficiency for o in result.R_orders if o.propagating}, result.power_balance_residual)
```

`python/tests/test_grating_solve.py` checks the glass grating of the conical validation
against the conical RCWA (s 40°/30°, p 50°/30°, reflected and transmitted orders to 2e-3 at
p = 3), the silver grating with the PEC bottom and the absorbed power, and the snapping.

The loop is also available as a generator, `hpfem.adaptive_solve`, which streams one
`AdaptiveStep` per iteration (DoFs, `eta`, observables and their change, goal value and
estimated goal error) and stops on a tolerance:

```python
def factory(mesh, orders):
    nd = hpfem.NedelecDofMap2D(mesh, orders.tolist())
    h1 = hpfem.DofMap2D(mesh, orders.tolist())
    problem = hpfem.ConicalScattering(nd, h1, setup)
    return problem, problem.solve()

goal = lambda p, s: hpfem.conical_dwr_estimate(p, s, functional)   # DWR of an order amplitude
for step in hpfem.adaptive_solve(adaptive, factory, observe=orders_of, goal=goal,
                                 tolerance=1e-4, max_dofs=100_000):
    print(step.dofs, step.eta, step.observables, step.goal_error, step.converged)
```

(`hpfem.refine_at_points(adaptive, corners, levels)` pre-refines material corners before the
loop; `functional = hpfem.conical_order_functional(origin, tangent, period, kx, m, points, e)`
with `e = conj(A_m) / |A_m|` of the current amplitude linearises the efficiency of order m.)

The hp loop (`tests/convergence/adaptive_hp_refinement.cpp`) reads the same in Python, and
with `ConicalScattering` in place of `Scattering2D` (its `estimate` / `error` take the
`ConicalSolution`, both DoF maps get the same `orders`; the Bloch faces may be refined
independently, `AdaptiveMesh2D.set_periodic` keeps them mirrored if wanted):

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
(SparseLU / MUMPS / cuDSS), `dirichlet_values` / `apply_dirichlet`, `gauged_curl_curl_eigenpairs`
for cavity modes, `complex_eigenpairs_near` / `complex_eigenpairs_near_gauged` for complex
pencils (resonances; the Krylov basis lives on the GPU with the cuDSS backend), `RieszProjection2D/3D` /
`AxisymmetricRieszProjection` for the modal expansion of a source problem around the resonances
(`RieszSetup`, `spectrum`, `expand`), `PropagatingMode` for waveguide cross-sections, `BandStructure2D` for
Floquet–Bloch bands of a unit cell, `ScatteringOperator2D` /
`plane_wave_sweep` / `ReducedBasis` for sweeps, `bloch_constraints` / `PeriodicPair2D` /
`fourier_coefficients` / `diffraction_efficiencies` for gratings, `PmlProfile.for_angle(theta_max_deg, target)` /
`PmlBox2D.max_resolution` / `recommended_thickness(..., profile, p)` for the PML at oblique
incidence (project files: `pml.profile.theta_max` and `target`), `dwr_estimate` with
`point_value_functional` or a Python functional for goal-oriented estimation, `VtkWriter2D`
for meshes with data arrays. `help(hpfem.<name>)` shows the bound signature and docstring.

## GPU backend

With the separately built GPU library (`gpu/README.md`, `HPFEM_ENABLE_CUDA`) the direct
solvers can run on cuDSS: `hpfem.available(hpfem.DirectSolverBackend.CUDSS)` tells whether the
library and a device are present, `hpfem.cudss_status()` why not, and
`hpfem.gpu_min_unknowns()` the size from which `AUTO` prefers the GPU. Every problem class
with a `solver` field accepts `DirectSolverBackend.CUDSS`; `TimeDomain2D/3D` then keep the
whole Newmark loop on the device (identical results to the host loop, state downloaded only
for observers and at the end). The building blocks are bound as well: `DeviceMatrix(csr)`
with `apply` / `apply_many` keeps a sparse matrix on the GPU for repeated products, and
`DeviceStepper(newmark, damping, stiffness, load, dt, beta, gamma)` with `set_state`, `step`
and `get_state` runs the Newmark recursion on the device for a Newmark operator factorised
by cuDSS; `solver.backend` names the backend `AUTO` chose, `solver.details` its factor
statistics. `python/tests/test_gpu.py` compares all of this with the host and is skipped
without the library.

## Tests

`python/tests/` runs in CI (`pip install ".[dev]" && pytest python/tests`, under a minute):
mesh and DoF-map invariants, Poisson with manufactured solution (rate $p+1$), PEC cavity
eigenvalues (no spurious modes), the Mie cylinder cross-section against the series, the
slab waveguide effective index, the lamellar grating's energy balance, the hp loop on the
L-shape, the goal-oriented estimate, callbacks raising inside parallel loops, VTK export.
Convergence is established by the C++ suite; the Python tests check that the bound pipeline
reproduces those numbers.

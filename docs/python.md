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

**Wheels (M15 F15).** The workflow `.github/workflows/wheels.yml` builds binary wheels of the
SparseLU build (self-contained, no MUMPS / cuDSS) for CPython 3.10–3.13: `manylinux_2_28`
x86_64 wheels with cibuildwheel and Windows `win_amd64` wheels for the python.org CPython,
compiled with the MSYS2 UCRT64 GCC (static GCC runtime, `pyproject.toml`
`[tool.cibuildwheel]`), plus the sdist. It runs on `workflow_dispatch`, on tags `v*` (the
wheels are attached to the GitHub release) and, with one Python version, on pull requests
that touch the packaging files; the artefacts are downloadable from the run. Install a wheel
with `pip install hpfem-<version>-<tag>.whl`; `pip install "hpfem[gui]"` adds Streamlit, Gmsh,
Matplotlib, meshio and PyYAML for the GUI, and `hpfem-gui [fem_app.py] [streamlit args]`
starts the Streamlit front end with this interpreter (the app comes from the argument, the
environment variable `HPFEM_GUI_APP` or an installed package `hpfem_gui`; `--dry-run` prints
the command). The MSYS2 build of the "Installation" section stays the developer path.

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

Dispersive materials can also live in a `materials.DispersiveMap` (M15 F13):
`dmap = DispersiveMap("air").set(2, "Ag").set(3, glass)` takes models, core materials, library
names or numbers, `dmap.at(omega)` freezes them into a `MaterialMap`, and
`dmap.apply(setup, omega)` sets `setup.omega` and `setup.materials` in one call, so a sweep is
one line per frequency. Tabulated data outside its range raises by default; `out_of_range =
"clamp"` (on the map or via `materials.with_policy(m, "clamp")`) uses the nearest sample and
warns once. `materials.fit_drude_lorentz(m, (lam0, lam1, count), oscillators=n)` fits a
passive Drude–Lorentz model to tabulated n, k over a wavelength range and returns the model
with its maximal relative error of ε (silver over 350–800 nm with two poles: a few per cent);
the fit is smooth in ω, for adaptive runs, resonances and sweeps beyond the samples.

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
sweep = hpfem.ConicalSweep(nd, h1, setup0)                  # affine operator: one analysis per sweep
solutions = [sweep.solve(setup_at(w)) for w in wavelengths]  # omega, beta, materials, incident may change
results = hpfem.sweep.solve_sweep(point, wavelengths, processes=4)   # any task over worker processes
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
(`maps_<point>_<map>.npz` with `x`, `y`, `values`), `cancelled`, `error` and `done`; since F9
also `estimate` (the predicted sizes of a factorisation, after `mesh`) and `progress` (`i`,
`phase`, `step`, `num_steps`, `seconds` at the start of every phase of a solve), the point's
`timing`, and cancellation between the phases of a solve as well as between points. The
results go to `results.json`. SIGTERM / SIGINT or the appearance of the cancel file stop the
run after the current point (exit code 2; 1 on an error). From Python,
`hpfem.run.run_job(job, out_dir, emit, cancel)` does the same with callbacks;
`hpfem.version_info()` reports version, platform, OpenMP, threads, backends and whether gmsh
is importable. `hpfem.meshing.structured_unit_cell(cell, nx, rows)` meshes a `UnitCell`
without Gmsh (columns over the period, rows of cells stacked from the bottom, tags by
priority), which the runner uses for `"mesh": {"structured": ...}`.

**Study tasks (schema version 2, M16 S9).** A document with `"version": 2` may also have
`"task": "optimize"`, `"reconstruct"` or `"uq"` (version-1 documents run unchanged and still
report version 1). Its `"parameters"` are geometry parameters of the model's shapes
(`{"name", "shape", "trapezoid": "cd" | "height" | "angle"}` or `{"name", "shape", "field":
"width"}`) and permittivities (`{"name", "material": tag, "part": "re" | "im"}`), each with
`"bounds"`; `"configurations"` list the measured efficiencies (`wavelength`, `theta_deg`,
`phi_deg`, `polarisation`, `orders`); `"morph"` sets the band of fixed nodes, the quality
threshold and remeshing of a structured mesh. Values are in job units (lengths times `unit`,
angles in degrees). The runner builds an `hpfem.opt.GratingEvaluator` on the morphed reference
mesh and drives a study whose store `<name>.study.jsonl` lies next to `results.json`, so
running the job again replays the evaluations and continues. `"optimize"` takes `objective`,
`maximize`, `method` (`"L-BFGS-B"`, `"Nelder-Mead"`, `"differential-evolution"`,
`"bayesian"`), `max_evaluations`, `x0`; `"reconstruct"` takes `measured` (or `synthetic`
`{"params", "noise", "seed"}`), `sigma`, `x0` and optionally `posterior` (emcee, extra
`opt-mcmc`); `"uq"` takes `inputs` (`{"normal": [mean, std]}` / `{"uniform": [lo, hi]}`),
`propagation` (`"linear"` or `"surrogate"` with `points`, `active`, `samples`) and `sobol`.
Events: `evaluation` per evaluation (also in `results["points"]`), `remesh`, `cancelled`,
`done`; the results hold the task's block with the parameters in job units. The full schema is
in the module docstring; `python/tests/test_run_study.py` runs all three tasks, the replay,
cancellation and the version-1 compatibility.

**Emitter task (schema version 2, M17 S4).** `"task": "emitter"` computes the emission of a
single Gaussian dipole in the cell over a wavelength sweep. The `"emitter"` block holds
`position` (x, y in the cell), `sigma`, `moment` (`"isotropic"` — the mean of the three
orientations —, `"x"`, `"y"`, `"z"` or `[px, py, pz]`), `wavelength` (a value, a list or
`{"start", "stop", "count"}`) and `stage`:

- `"A"`: the Bloch array at `kx_over_k0`, `beta_over_k0` (`grating.emit`): `P_cell`, `up`,
  `down`, `absorbed`, `guided`, the order powers.
- `"B"`: the single dipole (`grating.dipole_emission` with `scan` = `nodes`, `kx_nodes`,
  `depth`, `angle_nodes`, `symmetric`, `beta_max_over_k0`): `purcell`, the `fractions` up / down
  / nonradiated of the emitted power and, per numerical `aperture`, the collected power and
  fraction (`grating.emission_cone`, `aperture_side`).
- `"C"`: dP/dΩ on the grid `directions` = `{"theta_deg", "phi_deg", "sides"}`
  (`grating.emission_pattern`) per side and polarisation, with the power into each `aperture`
  integrated from the grid.

Before the solves the event `emission_cost` (and `results["cost"]`) gives the solves per
wavelength and in total, the memory of one factorisation and, with `calibrate` (one cell sample
and one plane-wave solve), the predicted wall time; `"dry_run": true` stops there. Further
events: `progress` per solve (`phase` `"scan"`, `"aperture …"`, `"pattern"` or `"emit"`), one
`point` per wavelength, `cancelled` (checked between the solves) and `done`. The full schema is
in the module docstring; `python/tests/test_run_emitter.py` runs the three stages, the dry run,
cancellation and an isotropic dipole in a homogeneous medium against the closed form, and
`examples/grating_emitter` uses the task for a quantum dot above a glass grating.

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

`result.flux_balance` is the flux-based balance through the PML boundaries
(`conical_power_balance`; `None` on meshes whose PML boundaries are no mesh lines), the
cross-check of the order-based `power_balance_residual`. For isolated scatterers in the
conical solver, `conical_cross_sections(problem, solution, surface)` and
`ConicalFarField(problem, solution, surface)` give σ_sca / σ_abs / σ_ext and the far-field
pattern of the 2.5D field (F10 / F11); `conical_diffraction_orders` takes orders on any
`OrderLine`, `to_literature_frame` converts the vector amplitudes to the x-period / y-invariant
/ z-normal frame of the grating literature.

**Shape sensitivities.** `grating.shape_sensitivity(result, velocity, order=0, side="R")`
returns the derivative of an efficiency with respect to a geometry parameter given by its
mesh velocity, an array `(hpfem.num_geometry_nodes(mesh), 2)` of node displacements per unit
of the parameter (`hpfem.region_normal_velocity(mesh, tag)` for the uniform normal growth of a
region; `hpfem.move_nodes(mesh, velocity, t)` applies a step). The general functions
`shape_gradient` / `shape_derivative` (`Scattering2D` / `3D`) and `conical_shape_gradient` /
`conical_shape_derivative` take any `Functional` (ADR-0011, docs/theory/maxwell.md "Shape
derivatives").

**Material sensitivities.** `grating.sensitivity(result, tag, order=0, side="R")` returns
the derivatives of the efficiency of a reflected or transmitted order with respect to the
real and the imaginary part of the permittivity of the cells tagged `tag`, from one adjoint
solve on the problem of the result (`hpfem.conical_adjoint_solution`,
`hpfem.conical_material_sensitivity`; `adjoint_solution` / `material_sensitivity` for
`Scattering2D` / `3D` with any `Functional`; docs/theory/maxwell.md, "Sensitivities").

**Jacobians (M16 S1).** `grating.solve(..., keep_factorisation=True)` keeps the factorised
system in `result.solution.factorisation`; `grating.sensitivity` and
`grating.shape_sensitivity` then cost one transposed solve each, and
`grating.jacobian(result, [("eps", tag), ("shape", velocity), ...], observables=None,
mode="auto")` returns `(J, rows, columns)` with the derivatives of several efficiencies
(default: every propagating R and T order) with respect to several parameters (two columns
for the real and imaginary part of a permittivity, one per mesh velocity) from one
multi-right-hand-side solve: `"direct"` solves once per parameter, `"adjoint"` once per
observable, `"auto"` picks the cheaper. The building blocks are
`conical_material_residual_derivative`, `conical_shape_residual_derivative` and
`conical_functional_shape_derivative` (and their `Scattering2D` / `3D` counterparts) with
`KeptFactorisation.solve_many` / `solve_adjoint_many` (ADR-0012, docs/theory/maxwell.md
"Direct mode").
The parameters `"theta"`, `"phi"` (per radian), `"omega"` (per rad/s) and `"wavelength"`
(vacuum, per metre), also as `(name, step)`, are columns for the incidence and the frequency:
they move the incident wave, the Bloch phases, β, k0 and the dispersive materials (models in
the `materials` given to `solve` are evaluated at the new frequency) at a fixed PML box and
fixed measurement lines, by the tangent with the derivative of the Bloch constraints plus the
explicit dependence of the order post-processing (`hpfem.conical_parameter_tangent`,
`conical_residual`, `conical_transported_solution`; docs/theory/maxwell.md "Frequency and
angle derivatives"). They are always computed in the direct mode, cost two assemblies and two
post-processings each and need no further factorisation; `"phi"` needs the vector path
(`solve(..., scalar=False)`). `result.inputs` keeps what `solve` was called with for this.

**Design parameters and mesh morphing (`hpfem.opt`, M16).** `import hpfem.opt` (not imported
by `import hpfem`). A `MaterialParameter(name, tag, part="re" | "im", lower, upper)` is a part
of the permittivity of a tag (`get(materials)`, `apply(materials, value)`); a
`GeometryParameter` is a scalar of one `Shape` of a `meshing.UnitCell`:
`GeometryParameter.field(name, shape, "height")` for an entry of the shape's `params`, and
`trapezoid_parameters(shape)` for the scatterometry triple mid-height width `cd`, `height`
and side-wall `angle` (radians), each changed with the other two fixed.
`shape_velocity(cell, mesh, parameter)` is the mesh velocity of a geometry parameter: the
nodes on the shape's boundary (and on its copies shifted by a period) move exactly with it,
the cell boundary and every other material interface stay fixed, the rest follows by the
harmonic extension of the mesh graph; it raises if the shape touches the cell boundary.
`Morph(cell, mesh, parameters, quality_threshold=0.3)` keeps the reference mesh:
`mesh_at({name: value})` moves it: the nodes on the shapes' boundaries go to the same place on
the changed shape (edge and fraction, ellipse angle: the exact geometry of `cell_at(values)`
for any combination of values), the others by the same harmonic extension (same topology,
tags and DoF numbering, so the objective is smooth and consistent with the shape
derivatives). `velocity_at(values, name)` is the mesh velocity at that point (central
differences of the node map) and goes into
`grating.jacobian(result, [("shape", morph.velocity_at(values, name)), ...])`;
`velocity(name)` is the one at the reference. (A linear morph `x_ref + Σ (p - p_ref) V_p`
misses the cross terms — the wall moves with `h cot α` — and biased a reconstruction by 2.5
standard errors; ADR-0012 §3 amendment.) `check(mesh)` compares
the signed cell quality (`cell_quality`: inscribed over circumscribed radius, 1 for the
equilateral triangle) with the reference and raises `MeshQualityError` (`ratio`,
`inverted`) for inverted cells or a ratio below the threshold — the signal to remesh
(ADR-0012 §3). `python/tests/test_opt_parameters.py` checks that the morphed boundary lies on
the new trapezoid exactly for every parameter and for all of them together, `velocity_at`
away from the reference against differences of the node map, and the derivatives along the
velocities against finite differences of solves on morphed meshes (to 1e-6).
`Morph(..., band=(y_low, y_high))` (and `shape_velocity(..., band=...)`) keeps every vertex on
and beyond the band's edges fixed. Use it to keep the measurement lines and the PML out of
the deformation: a measurement line that lies on mesh facets (the default line midway to the
PML often does on structured cells) must not touch deforming cells, because the order
amplitude sampled on facets is not differentiable along such a velocity (the normal Nédélec
component jumps across facets, the located cell changes; the Jacobian was off by a factor of
ten for the height of a p-polarised line). `grating.jacobian` and `grating.shape_sensitivity`
refuse such velocities (`GratingError`); put the band at least one cell row inside the lines.

**Studies (`hpfem.opt`, M16 S2).** A `DesignSpace([...], constraints)` holds `Continuous(name,
lower, upper, log=False, unit="", scale=None)`, `Integer(name, lower, upper)` and
`Categorical(name, choices)` parameters (the S1 `MaterialParameter` / `GeometryParameter` count
as continuous with their bounds and scale), `LinearConstraint({name: a}, lower, upper)` and
`NonlinearConstraint(function, lower, upper)` on the SI values. It validates points
(`validate`, `check` with the constraints, `feasible`), maps them to the unit cube and back
(`encode` / `decode`, logarithmic for `log=True`) and to SI vectors (`to_vector` /
`from_vector`, `bounds()`, `linear_constraints()` as `(A, lower, upper)`), and draws initial
designs (`sample(n, "lhs" | "sobol" | "random", seed)`, infeasible points rejected). An
evaluator follows ADR-0012 §2: it returns an `Evaluation` (`params`, real `values` (m,),
`jacobian` (m, n) in parameter order, `error`, `cost` in seconds, `fidelity`, `mesh_id`,
`status` `"ok"` / `"failed"` / `"cancelled"`, `meta`); `FunctionEvaluator(f, space,
observables, settings=...)` wraps a function returning a number, an array, `(values,
jacobian[, error])` or a dict, splits complex values into real and imaginary part
(`split_complex`) and turns an exception into a failed evaluation with NaN values and the
message in `meta["error"]`.

```python
import hpfem.opt as opt

space = opt.DesignSpace([opt.Continuous("x", -2, 2), opt.Continuous("y", -1, 3)],
                        [opt.LinearConstraint({"x": 1, "y": 1}, upper=3)])
def rosenbrock(p, jacobian=False):
    x, y = p["x"], p["y"]
    return (1 - x)**2 + 100*(y - x*x)**2, [-2*(1 - x) - 400*x*(y - x*x), 200*(y - x*x)]
study = opt.Study(rosenbrock, "rosenbrock.study.jsonl", space, emit=print, cancel=None)
study.run(space.sample(8, seed=1))                    # propose and evaluate, sequentially
e = study.evaluate({"x": 1.0, "y": 1.0}, jacobian=True)
study.best(), study.history().values, study.failed
```

`Study(evaluator, path, space)` evaluates sequentially (ADR-0012 §7), caches by a canonical key
(values divided by their scale and rounded to 12 significant digits, plus the requested
`fidelity` and the SHA-256 of the evaluator `settings`, so a changed setting never reuses old
values) and appends every evaluation to the JSON-lines store of ADR-0012 §5: a header line
(`schema` 1, version, design space, evaluator name / settings / hash / observables), then
`evaluation`, `proposal`, `state` (`checkpoint(method, iteration, state)` for optimisers),
`remesh` (from `meta["remesh"]` of an evaluation) and `note` lines; NaN as `null`, ±∞ as
`"inf"`, complex values as `{"__complex__": [re, im]}`. Opening an existing store resumes it:
the cache, the last `state` and the open proposals (`open_proposals()`, `evaluate_open()`)
are rebuilt, nothing is evaluated again, a truncated last line is dropped; `Study.load(path)`
opens it read-only. `emit(event)` receives JSON-ready events (`study`, `proposal`,
`evaluation` with `i`, `n`, `cached`, `remesh`, `state`, `note`, `cancelled`); `cancel()` is
polled before every evaluation and passed to the evaluator, a true value raises
`hpfem.Cancelled` with the store consistent. Failed evaluations are recorded and cached and do
not stop the study (`raise_errors=True` raises `EvaluationFailed` after recording,
`retry_failed=True` evaluates them again). `python/tests/test_opt_study.py` covers the
encodings, constraints, the cache key, the store round trip, resume after a cancellation,
failures, events, an L-BFGS-B run on Rosenbrock replayed from the store and a grating
evaluator with its Jacobian.

**Optimisers and least squares (`hpfem.opt`, M16 S3).** `minimize(study, objective, method=...)`
runs SciPy's `"L-BFGS-B"` (gradient from the evaluator's Jacobian), `"Nelder-Mead"` or
`"differential-evolution"` (seeded, `seed=`) on a `Study` — or on an evaluator / function with
`space=` and `path=` — so every point is cached, stored and replayed. The objective is an
observable (index or name) or a function of the value vector returning `F` or `(F, dF/dy)`
(chain rule `dF/dp = (dF/dy)ᵀ J`); `maximize=True`, `fixed={name: value}` (integer and
categorical parameters must be fixed), `fidelity`, `max_evaluations`, `options` for SciPy,
`callback(info)`. The optimisers work on the unit-cube coordinates of the design space
(logarithmic for `log=True`) with bounds `[0, 1]`; linear and nonlinear constraints go to
differential evolution as constraints and act as an extreme barrier for the other two. A
failed evaluation (ADR-0012 §2) gets a failure value above every value seen, so the step is
rejected (line-search backtracking, simplex contraction); `max_failures` stops the run. A
remesh (ADR-0012 §3) restarts L-BFGS-B with an empty memory from the remesh point. After every
iteration a `state` checkpoint (current and best point, counts, the population of differential
evolution) is stored; a cancelled run (`hpfem.Cancelled`) resumes by calling `minimize` again —
with the same `x0` and `seed` it replays the stored points from the cache, with `x0=None` it
restarts from the best stored point (differential evolution: from the stored population). The
`OptimizeResult` has `params` (SI), `value`, `evaluation`, `success`, `message`, `iterations`,
`evaluations` / `new_evaluations` / `cache_hits`, `failures`, `infeasible`, `restarts` and
`history()`.

```python
result = opt.minimize(study, "R0", maximize=True, x0={"cd": 60e-9, "height": 80e-9})
rec = opt.fit(evaluator, y_meas, sigma, x0=..., fixed={"swa": 88 * units.deg})
rec.params, rec.std, rec.correlation, rec.chi2_red, print(rec.summary())
```

`fit(study_or_evaluator, y_meas, sigma=None, x0=..., observables=..., method="lm" | "gn")`
reconstructs parameters by minimising `½‖W^{1/2}(y(p) − y_meas)‖²` with `W = diag(1/σ²)`
(ADR-0012 §6; `sigma` a number, an array — observables of very different magnitudes — `None`
for unweighted or `"relative"` for `σ = |y_meas|`): Levenberg–Marquardt with Moré scaling and
Nielsen's damping update, or Gauss–Newton with step halving, on the unit-cube coordinates,
each step a least-squares solve of the stacked system (no normal equations). Bounds by an active
set and projection (a parameter at a bound with an outward gradient is held, the trial point is
projected, the predicted decrease uses the projected step); constraint violations and failed
evaluations are rejected steps; a remesh restarts the damping. It stops on the scaled gradient
(`gtol`), the step (`xtol`) or the relative decrease (`ftol`) and checkpoints after every
accepted step. The `FitResult` carries `params`, `x`, `std`, `covariance`, `correlation` (SI),
`cost`, `chi2`, `chi2_red`, `dof`, `values`, `residual`, `jacobian`, `at_bounds`, the counts, the
`study` and the `laplace` object. `laplace(jacobian, residual, sigma=None, names=...)` is the
Laplace approximation on its own: `C = s² (JᵀWJ)⁻¹` from the SVD of the column-equilibrated
weighted Jacobian, with `s² = χ²_red` when σ is unknown (`sigma=None` or `"relative"`, as the
NIST certified standard deviations) and `s² = 1` when it is given; rank deficiency (relative
singular values below `rcond = 1e-8`) or a condition number above `1e8` raise an
`IdentifiabilityWarning`, non-identifiable parameters get an infinite standard error. Tests:
`python/tests/test_opt_optimize.py` (Rosenbrock 2D/4D, Nelder–Mead, Branin, replay,
cancellation and resume, failures, constraints, remesh) and `python/tests/test_opt_lsq.py`
(an exact linear model, NIST StRD MGH17 from both starts — parameters and standard deviations
to `1e-6` of the certified values —, weights, bounds, non-identifiability, study integration).

**Bayesian optimisation (`hpfem.opt`, M16 S5).** `bayesian_optimize(study, objective,
acquisition="ei" | "lcb", use_gradients=False, constraints=(), max_evaluations=50, seed=0)`
drives a `Study` like `minimize` (same objective forms, `fixed`, `maximize`, `fidelity`, unit-cube
coordinates of the free parameters): an initial design of `n_initial = 2(d + 1)` points
(`initial="lhs"` or `"sobol"` from `DesignSpace.sample`, seeded, plus `initial_points`), then one
point per iteration, proposed by maximising the acquisition on a Gaussian-process surrogate of
the objective fitted to every successful evaluation of the study with the same fixed values and
fidelity. Acquisitions: expected improvement in a numerically stable log form (LogEI; `xi`) and
the lower confidence bound `μ − κσ` (`kappa=2`), maximised from Sobol' and local candidates by
multi-start L-BFGS-B with the analytic gradients of the GP prediction. Known constraints of the
design space restrict the candidates and the local search (SLSQP); unknown, expensive ones are
`OutcomeConstraint(observable_or_function, lower, upper)` on the observables, each with its own
GP, entering as `log EI + Σ log P(feasible)`. `use_gradients=True` evaluates with the Jacobian
and fits the gradient-enhanced GP (values and partial derivatives chain-ruled to the unit cube).
Stopping: `max_evaluations` (initial design and cache hits included), `patience`, `tol` on the
predicted gain, `max_failures`, `callback(info)`. Failed evaluations stay in the surrogate with
the worst successful value and are never proposed again; remeshes are counted, the surrogate
keeps every point (its learned noise absorbs the jump). `Evaluation.error` (the DWR estimate) is
not used: it is not observation noise (ADR-0012 §6). A `state` checkpoint after every iteration
(seed, counts, hyperparameters of the surrogates, incumbent) lets a cancelled run resume on the
stored study exactly where it stopped: the open proposal is evaluated first and every random
choice is seeded by `(seed, iteration)`, so the resumed run proposes the same points as an
uninterrupted one. The `BOResult` extends `OptimizeResult` by `gp`, `constraint_gps`,
`acquisition` (predicted gain per iteration), `remeshes` and `feasible`.

```python
res = opt.bayesian_optimize(study, "R0", maximize=True, use_gradients=True,
                            constraints=[opt.OutcomeConstraint("T0", upper=0.05)],
                            max_evaluations=30, seed=1)
mean, var, dmean, dvar = res.gp.predict(u, grad=True)  # u: (q, d) unit-cube points
front = opt.pareto_front(study, ["R0", "A"], maximize=[True, False])
```

The surrogate is `GaussianProcess(kernel="matern52" | "matern32" | "se", mean="constant",
noise="learn" | variance)` (NumPy/SciPy only, dense, up to a few hundred points; with
derivatives `n(d + 1)` observations): ARD length scales, signal variance and noise from the log
marginal likelihood with analytic gradients (multi-start L-BFGS-B in log space, weak log-normal
priors, bounds for unit-cube inputs and standardised outputs), Cholesky with jitter escalation,
`fit(x, y, dy=None)` with optional partial derivatives (NaN entries left out),
`predict(x, grad=..., full_cov=...)`, `sample`, `log_marginal_likelihood(theta, gradient=True)`;
`MultiOutputGP` fits one process per output. `pareto_front(study, objectives, maximize=...)` and
`non_dominated(values)` give the non-dominated evaluations of any study. With the optional extra
`opt-bo` (BoTorch, imported lazily, an `ImportError` names the extra), `pareto_optimize` runs
multi-objective BO (qLogNEHVI) and `multi_fidelity_optimize(study, objective, fidelities=[...],
costs=...)` cost-aware multi-fidelity BO (multi-fidelity knowledge gradient over discrete
fidelity levels) on the same study. These two BoTorch drivers are **experimental**: BoTorch is
not installed on the development machine nor in CI, so they have not been run yet (their
tests skip without it). Tests: `python/tests/test_opt_gp.py` (interpolation, the
likelihood and prediction gradients against finite differences, hyperparameter recovery, the
gradient-enhanced process) and `python/tests/test_opt_bo.py` (EI/LCB/feasibility formulas,
Branin to `1e-3` within 40 evaluations with EI and LCB, gradient-enhanced against plain BO on
Hartmann-3, known and outcome constraints, cancellation and exact resume, failures, remeshes,
the DWR estimate ignored, Pareto front; the BoTorch tests skip without BoTorch).

**Scatterometry evaluator (`hpfem.opt`, M16 S3).** `GratingEvaluator(morph, materials, stack,
configurations, material_parameters=(), order=3, pml=None, mesher=None, solve_options=None)`
is the evaluator of a grating measurement: the parameters are the geometry parameters of the
`Morph` followed by `MaterialParameter`s, the observables the efficiencies of every
`Configuration(wavelength, theta, phi=0, polarisation="s", orders=(("R", 0),))` (labels such
as `"s 405nm 65deg: R0"`). Each evaluation morphs the reference mesh and runs one
`grating.solve` per configuration with the kept factorisation and one `grating.jacobian`
along the morph velocities and the permittivity columns. The PML boxes and the measurement
lines are designed once on the reference mesh and kept, so nothing in the discretisation
jumps with the parameters. `stack` may be a function of ω (dispersive substrates); a fidelity
`{"order": p}` overrides the order. When the morph's quality guard trips, `mesher(cell) ->
mesh` builds a new reference mesh at the current geometry (the study records a `remesh`);
without one the evaluation fails. The settings hash covers the order, the configurations,
the materials, the stack, the mesh size and the frames. The constructor refuses a morph that
deforms the cells at a measurement line on mesh facets (see the band above).
`python/tests/test_opt_scatterometry.py` checks the values against a plain solve, the Jacobian
(CD, height, side-wall angle, permittivity) against finite differences of the evaluator to
`1e-5`, the store reuse and the remesh. `examples/grating_reconstruction` reconstructs CD,
height and side-wall angle of a silicon line grating from synthetic spectroscopic data with
`fit` and reports the Laplace uncertainties.

**Posterior beyond Laplace (`hpfem.opt`, M16 S6).** `sample(fit_result, surrogate=True,
prior=None, walkers=None, steps=3000, burn=None, thin=1, seed=0)` samples the posterior of the
free parameters of a `fit` — likelihood `exp(-½ Σ (y_i(p) - y_meas,i)² / (s σ_i)²)` with the
noise scale `s² = χ²_red` when σ was unknown (the scaling of `laplace`), a uniform prior on
the bounds times optional Gaussians `prior={name: (mean, std)}` — with the affine-invariant
ensemble sampler of emcee (optional extra `opt-mcmc`, imported lazily; an `ImportError` names
the extra). The walkers start from the Laplace Gaussian. With `surrogate=True` the model is
replaced by `build_surrogate(fit_result, width=4, points=2n+2, validation=n+2)`: a
Latin-hypercube design in `x̂ ± 4 std` (clipped to the bounds, the optimum included)
evaluated **with the Jacobian** through the study, a gradient-enhanced `MultiOutputGP` per
observable without noise, its variance added to the noise in the likelihood (the posterior
widens where the surrogate is unsure), and `validation` extra evaluations measuring its error
in noise standard deviations (`Surrogate.validation`, a `PosteriorWarning` above 0.3).
`surrogate=False` calls the evaluator at every step without the study's cache and store
(cheap models only). `PosteriorResult` holds the samples, mean, std, covariance, correlation,
the 2.5/16/50/84/97.5 % quantiles, the acceptance fraction and the autocorrelation time
(a `PosteriorWarning` when the chain is shorter than 50 of them or the acceptance below
0.15); `compare()` lists per parameter the Laplace mode and error against the posterior mean
and std (`shift` in Laplace errors, `ratio` of the stds), `summary()` prints it.
`python/tests/test_opt_posterior.py`: on a linear Gaussian model the posterior (direct and on
the surrogate) reproduces the Laplace mean, errors and correlations to 0.1 standard errors /
10 %, with a Gaussian prior it matches the exact `(C⁻¹ + D)⁻¹`; on a skewed one-parameter
model the 16/50/84 % quantiles match a brute-force grid posterior where the Laplace Gaussian
does not (mean shifted by more than 0.15 errors); on a two-parameter exponential model the
surrogate posterior equals the direct one. `examples/grating_reconstruction/run.py
--posterior` samples the posterior of the silicon-grating reconstruction.

**Uncertainty propagation and sensitivity (`hpfem.opt`, M16 S7).** Uncertain inputs are
independent `Normal(mean, std)` or `Uniform(lower, upper)` per parameter name (SI); the other
parameters keep `fixed` (default: the centre of their bounds). `linear_propagation(study,
inputs)` evaluates once with the Jacobian at the input means and returns the delta-method
output covariance `J Σ Jᵀ`, the standard deviations and the share of every input in every
output variance (`contributions`, rows sum to 1). `build_global_surrogate(study, inputs,
points=4n+4, active=0, width=4, gradients=True)` evaluates a Latin hypercube in the input box
(`mean ± 4 std`, the bounds of a uniform input; clipped to the design space) through the
study, with the Jacobian, fits a gradient-enhanced `MultiOutputGP` without noise and adds
`active` points one at a time where the predicted standard deviation relative to the spread
of the training values is largest (`max_relative_std` reports it for the final model; it is
set by the corners of the box and conservative). `monte_carlo(surrogate_or_function, inputs,
samples=10000)` gives the output mean, std, quantiles and samples (normal samples outside the
surrogate box are clipped and counted); `sobol_indices(surrogate_or_function, inputs,
samples=4096, bootstrap=200)` the first-order (Saltelli 2010) and total (Jansen) Sobol'
indices with bootstrap 95 % half-widths from scrambled Sobol' points, `N (n + 2)` model calls
(free on the surrogate). SALib is not used: it does not install on the MSYS2 Python, and the
own estimators are checked against the analytic indices instead.
`python/tests/test_opt_uq.py`: the Ishigami indices directly (2¹⁴ points, 0.02) and through a
surrogate of 40 + 30 actively learned points (0.02); on a linear model linearised propagation,
Monte Carlo and Sobol' indices against the closed form; on a nonlinear model Monte Carlo on
the surrogate against Monte Carlo on the function (1–2 %) and the linearised std as the
small-tolerance limit. `examples/fabrication_tolerance` propagates CD, height and side-wall
angle tolerances of the silicon grating to its reflectance spectrum.

**Dipole emitters in the cell (M17 S1).** `grating.emit(mesh, materials, stack, dipole,
omega, kx=0, beta=0, order=4, pml=..., bottom="pml", orders_max=3, ...)` solves one cell problem
of a dipole array — one Gaussian dipole per period with the phase `exp(i kx P)` from cell to cell
and `exp(i beta z)` along the lines, the cell problem of the array scanning of a single dipole
(ADR-0013). `dipole = {"position": (x0, y0), "moment": (px, py, pz), "sigma": s}` (SI, current
moment [A m], solver frame: x along the period, y the stack normal, z along the lines); the
Gaussian must lie 6σ inside the cell and outside the PML, in a lossless medium. Geometry, PML and
measurement lines are those of `grating.solve`; cells whose tag is not in `materials` take the
stack material at their centroid. The `EmissionResult` holds `P_cell` (the delivered power,
`-½ Re ∫ conj(J)·E`, [W/m per unit β]), `orders_up` / `orders_down` (`EmissionOrder` with the
radiated `power` per order), `up`, `down`, the PML-boundary fluxes `flux_up` / `flux_down`,
`absorbed` and `A_by_tag`, the `guided` remainder and `field(points)`. The building blocks are
`hpfem.conical_gaussian_dipole` and `hpfem.conical_source_power` (docs/theory/maxwell.md "Dipole
emitters").

**Emission pattern by reciprocity (M17 S2).** `grating.emission_pattern(mesh, materials, stack,
dipole, omega, directions=[(theta, phi, side), ...], pol=("s", "p"), normalized=False,
quadrature_points=6, order=4, pml=..., **solve_kwargs)` gives the far-field power per unit
solid angle of a single Gaussian dipole (the `dipole` dict of `emit`) in each direction and
polarisation: `side="up"` is the direction (sinθ cosφ, cosθ, sinθ sinφ) into the cover,
`"down"` (sinθ cosφ, −cosθ, sinθ sinφ) into a lossless substrate. Each value is one
`grating.solve` with the plane wave incident from that direction, `dP/dΩ = n k0² Z0 |p·⟨E⟩|² /
(32π²)` with the total field averaged over the dipole's Gaussian (docs/theory/maxwell.md "Stage
C"); directions into the substrate are solved on the problem mirrored at y = 0 (straight meshes,
`pml` as `None` or a `{"top", "bottom"}` dict). The `EmissionPattern` holds `dP_dOmega`
(directions × polarisations, [W/sr], divided by `P_bulk` with `normalized=True`), `total` (summed
over the polarisations), `P_bulk` (the dipole in its homogeneous host medium), the reciprocity
`amplitude` and the index `n` of each direction's medium. `moment` may also be `"x"`, `"y"`,
`"z"` or `"isotropic"` (the mean of dP/dΩ over the three unit moments, `amplitude` NaN);
`progress` / `cancel` act per solve. It is the radiated part only; the guided and absorbed
power and the Purcell factor need Stage B.

**Single dipole by array scanning (M17 S3).** `grating.dipole_emission(mesh, materials, stack,
dipole, omega, nodes=6, kx_nodes=4, depth=0.5, beta_max=None, symmetric=False, order=4,
channels=True, angle_nodes=(8, 16), pml=..., progress=None, cancel=None, ...)` integrates the
cell problem over kx (one period, on the complex contour around the light lines) and β
(`grating.array_scan_rule`, docs/theory/maxwell.md "Stage B"). The `DipoleEmission` holds, per
orientation key `"x"`, `"y"`, `"z"`, `"isotropic"` and `"moment"` (the dipole's own, if
given), the emitted power `P_em`, the Purcell factor `purcell` = P_em / P_bulk, the radiated
channels `up` / `down` (by reciprocity over the half-spaces, `angle_nodes` = θ points, φ points)
and `nonradiated` = P_em − up − down (absorbed in a lossy structure, guided in a lossless one;
guided-mode poles are not treated yet, so use structures without lossless waveguide layers),
with `samples`, `directions`, the power and radiation matrices and the timing; `symmetric`
halves the scan for a cell mirror-symmetric about the dipole. `grating.emission_cost(...)` with
the same arguments counts the cell problems and plane-wave solves without running them, gives
the memory of one factorisation and, with `calibrate=True`, times one of each and extrapolates
the wall time (`EmissionCost`). `grating.emission_cone(mesh, materials, stack, dipole, omega,
aperture, side="up", angle_nodes=(8, 16), symmetric=False, ...)` is the power radiated into a
collection cone θ ≤ asin(NA / n) per orientation key (`EmissionCone.power`); divided by
`P_em` it is the extraction efficiency into the aperture of an objective.

**Scalar E_z path.** For the s polarisation at `phi = 0` (`scalar="auto"`, the default)
`grating.solve` lets `ConicalScattering` factorise only the H1 block (`setup.scalar_ez`), about
a third of the unknowns with identical results; `result.scalar` says whether it was used,
`scalar=False` forces the block solve.

**Progress, cancellation, timing, memory (F9).** `solve(..., progress=callback,
cancel=flag)` calls `callback(event)` with an `hpfem.ProgressEvent` (`phase`, `step`,
`num_steps`, `seconds`) at the start of every phase of the solve (assembly, constraints,
factorisation, solve, post) and when it is done, and polls `cancel()` at the same moments;
a true value raises `hpfem.Cancelled`. The same callback is `setup.progress` of
`Scattering2D` / `Scattering3D` / `ConicalScattering`, whose solutions carry `timing` (seconds
per phase and total); `result.timing` of `grating.solve` adds them as `solver.<phase>`.
`grating.estimate_memory(mesh, order, solver)` (or `hpfem.estimate_memory(mesh, order,
backend, conical)` and the map-based overloads) predicts DoFs, matrix nonzeros, factor entries
and bytes of the factorisation before the run; `LinearSolver.factor_entries` is the measured
number after `factorize` (docs/theory/solvers.md, "Memory estimate").

**Resonances and bands of the cell (F14).** `grating.resonances(mesh, materials, stack,
omega_target, kx=0.0, beta=0.0, num_modes=4, order=4, pml=None, bottom="pml", ...)` solves
the conical eigenproblem of the unit cell (`hpfem.ConicalResonance`: PEC top and bottom, PML
designed at the target and the angle of the Bloch wavenumber `kx` [1/m], both polarisations
at `beta = 0`) and returns a `ResonanceResult` with `modes` (`omega` complex with Im < 0 for
a decaying mode, `wavelength`, `Q`, `residual`, `beta`, `field(points, quantity)` with `"E"`,
`"H"` or `"S"`), `omegas`, `dofs` and `timing`; `progress` / `cancel` as in `solve` (phases
assembly, constraints, eigensolve, post). `grating.bands(mesh, materials, stack,
omega_target, kx_values, **kwargs)` repeats this along a list of Bloch wavenumbers (the
complex band structure of the open cell; closed photonic crystals stay with
`hpfem.BandStructure2D`). In the job runner `"task": "resonances"` with a `"resonance"` block
(`wavelength`, `num_modes`, `kx_over_g` or `theta_deg`, `beta`) and `"task": "bands"` with
`"sweep": {"kx_over_g": ...}` emit `mode` and `point` events and write the maps per mode.
`python/tests/test_grating_resonances.py` checks the Fabry–Pérot slab against the exact complex
wavenumber through the front end and the runner.

**Resonance derivatives and dispersive resonances (M16 S4).**
`grating.resonance_sensitivity(result, mode, [("eps", tag), ("shape", velocity), "beta",
"kx"])` returns `{label: hpfem.ResonanceDerivative}` with `domega` (complex: the real part
moves the resonance, the imaginary part its width), `dquality`, `dwavelength` and `dlambda`,
from the left eigenvector of the Bloch-reduced pencil (`hpfem.conical_resonance_adjoint`) —
no further eigensolve; `"eps"` gives the entries `eps[tag].re` and `eps[tag].im`. Dispersive
models in `materials` are evaluated at the target by `resonances`;
`grating.refine_resonance(result, mode)` solves for the self-consistent resonance (Newton on
`λ̂(ω) = (ω/c0)²` with ε at the mode's own complex ω — the analytic continuation of
`DrudeLorentz` and `Constant`, the real part of ω for tabulated data) and marks the result
`self_consistent`; its derivatives then carry the `dε/dω` term. The building blocks
(`resonance_adjoint`, `resonance_material_derivative`, `resonance_shape_derivative` for
`Resonance2D` / `3D`; `conical_resonance_*_derivative`; `resonance_derivative_from`) are bound
as well (docs/theory/maxwell.md, "Resonance derivatives"). `python/tests/test_resonance_sensitivity.py`
checks them on the Fabry–Pérot slab against the exact derivatives and re-solved resonances.

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

Bodies of revolution on substrates and in layer stacks (ADR-0014): set
`AxisymmetricScatteringSetup.background` to a `LayerStack3D` normal to the axis (its z is the y of
the meridian mesh, interfaces on mesh lines, the layer cells tagged with the stack's materials)
and take the incident order from `layered_axisymmetric_wave(stack, k0, theta, pol, m,
side="top")` (`pol` `"s"` / `"p"`, `side` `"top"` or `"bottom"` for a lossless substrate; the
result has `value` and `curl` callables and R, T, A of the bare stack). The source then lives
only where a cell deviates from the stack, particles and holes alike:

```python
stack = hpfem.LayerStack3D(hpfem.Material.vacuum(), [hpfem.Layer(gold, 50e-9)], glass, 0.0)
setup.background = stack
for m in range(-m_max, m_max + 1):
    setup.azimuthal_order = m
    setup.incident = hpfem.layered_axisymmetric_wave(stack, k0, theta, "p", m).value
    fields.append(hpfem.AxisymmetricScattering(nd, h1, setup).solve())
```

Per order, `problem.absorbed_power(field)` gives the absorption of the total field (per cell and
tag; the body is `problem.scatterer_cells()`), `problem.incident_absorbed_power()` that of the bare
stack in the same cells (their difference over a region around a hole is the absorption change),
`axisymmetric_flux_channels(..., surface, stack)` the scattered power split into `up` / `down` /
`lateral`, and `axisymmetric_disc_flux(..., z, radius, -1, wave.value, wave.curl)` the transmission
through a disc with the stack's own part (`background`, `change()`); `axisymmetric_poynting_flux`
takes the same `added_value` / `added_curl` for the flux of the total field. Sum every quantity over
the orders.
The far field in both half-spaces comes from `axisymmetric_layered_far_field(..., surface, stack,
theta_up, theta_down)` (reciprocity with the stack's plane waves; `.up` / `.down` are
`AxisymmetricFarField`s with `radiated_power()` and `power_between(theta_min, theta_max)` for a
collection cone, e.g. `power_between(0, asin(NA / n))` for an objective above the sample).

**Band derivatives and group velocity.** With `setup.keep_modes = True` a
`BandStructure2D/3D` keeps the eigenvectors in `Bands.modes`, and the bands can be
differentiated without another eigensolve: `band_permittivity_derivative(problem, bands, tag)`
and `band_permeability_derivative` for the material of a tag, `band_shape_derivative(problem,
bands, velocity)` for a mesh velocity (e.g. `region_normal_velocity(mesh, tag)` for a rod
radius; it must vanish on the periodic faces) and `band_wave_vector_derivative(problem, bands,
direction)` along k. Each returns a `BandDerivative` with `eigenvalue` (d k0²/dp), `wavenumber`
(d k0/dp, NaN at k0 = 0), `angular_frequency` (c0 d k0/dp) and `multiplicity`; degenerate bands
(relative gap below `degeneracy_tolerance`, 1e-6) are resolved as a cluster.
`group_velocity(problem, bands)` returns dω/dk in m/s as an array (num_bands, Dim):

```python
setup.keep_modes = True
crystal = hpfem.BandStructure2D(nd, h1, setup)
bands = crystal.bands([1.3, 0.6])
v_g = hpfem.group_velocity(crystal, bands)                         # (num_bands, 2) [m/s]
d_eps = hpfem.band_permittivity_derivative(crystal, bands, 2).wavenumber
d_r = hpfem.band_shape_derivative(crystal, bands, hpfem.region_normal_velocity(mesh, 2))
```

See docs/theory/maxwell.md, "Band derivatives and group velocity".

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

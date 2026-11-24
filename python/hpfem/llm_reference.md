# hpfem command reference (for agents)

hpfem is an adaptive hp-FEM suite for nano-optics: time-harmonic Maxwell with Nédélec elements,
PML, gratings (Bloch, conical incidence), waveguide modes, resonances, emitters, band structures,
time domain, plus heat conduction and linear elasticity. The core is C++ (`hpfem._hpfem`); the
Python API is NumPy / SciPy at the boundary.

Discover at runtime:

- `hpfem.reference()` returns this text; `hpfem reference` prints it.
- `help(hpfem.Scattering2D)`, `help(hpfem.grating.solve)`: bound signatures and docstrings.
  `dir(hpfem)` lists the API; overloaded functions list every overload in `help`.
- `hpfem.__version__`, `hpfem.version_info()` (version, platform, OpenMP, threads, backends,
  gmsh), `hpfem.available_backends()`, `hpfem.has_openmp()`.
- CLI: `hpfem info | materials | reference | validate P | run P [-o out.json] [-t N] [-q]`
  (also `python -m hpfem ...`); job runner `python -m hpfem.run job.json --out DIR`.
- Not imported by `import hpfem`: `import hpfem.opt`, `from hpfem import planar, dataports, gui`.

## 1. Conventions (get these right)

- Time dependence **exp(-iωt)**. Lossy media have **Im ε_r > 0**; a decaying resonance has
  **Im ω < 0**; Q = Re ω / (−2 Im ω).
- **SI units everywhere** in the core: metres, rad/s, V/m, W. Convert at the boundary with
  `hpfem.units`: multiply into SI (`633 * units.nm`), divide out (`omega / units.THz`).
  Angles passed to `hpfem.grating.*` are radians (`50 * units.deg`);
  `hpfem.PmlProfile.for_angle` takes **degrees**.
- `units.angular_frequency(wavelength=|frequency=|energy=|wavenumber=)` (exactly one keyword),
  `units.wavelength(omega)`, `units.frequency(omega)`, `units.photon_energy(omega)`,
  `units.vacuum_wavenumber(omega)`. Multipliers: `m cm mm um nm pm angstrom s ns ps fs Hz kHz
  MHz GHz THz PHz J eV meV rad deg V_per_m W mW`. Constants: `hpfem.constants.c0`, `eps0`,
  `mu0`, `Z0`, `h_planck`, `e_charge`, `pi`.
- Unknown: the electric field E, curl(μ⁻¹ curl E) − ω² ε E = iω J. H is post-processed.
- **2D vs 3D by name**: `Mesh2D/3D`, `NedelecDofMap2D/3D`, `Scattering2D/3D`, `PmlBox2D/3D`, ...
  Free functions are overloaded on the argument type (`hpfem.refine_uniform(mesh)` works in both).
- **`Scattering2D` solves the in-plane field (E_x, E_y)** (H_z polarisation). E_z polarisation and
  oblique / conical incidence use `hpfem.ConicalScattering` (β = 0 for E_z) or `hpfem.grating`.
- Arrays: coefficient vectors are complex NumPy arrays of length `dofs.num_dofs`; sparse
  matrices are complex `scipy.sparse.csr_matrix`; points are (n, dim) float arrays.
- Elasticity / multi-component H1 vectors are **component-major**: (u_x block | u_y block | u_z
  block), length dim × `DofMap.num_dofs`.
- Tags: cells untagged = tag 0 (`hpfem.NO_TAG`) → background material. Generators tag their
  sides `hpfem.box_tag.X_MIN=1, X_MAX=2, Y_MIN=3, Y_MAX=4, Z_MIN=5, Z_MAX=6` (`tag_sides=True`);
  disc / ball boundary `hpfem.DISC_BOUNDARY`; inclusions of `square_with_disc` / `box_with_ball`
  get `inclusion_tag` (default 2).
- Materials by cell tag: `setup.materials.set(tag, hpfem.Material.dielectric(n))`;
  `hpfem.Material(eps_r, mu_r)`, `hpfem.Material.vacuum()`, `hpfem.Material.anisotropic(eps3x3)`.
- Solvers return: `Scattering*.solve()` → solution with `.unknown` (coefficients), `.timing`;
  eigen solvers → list of mode objects; `Elasticity*.solve()` / `Thermal*.solve()` → coefficient
  array; `hpfem.grating.solve` → `GratingResult` dataclass.
- Formulation default is `TOTAL_FIELD`; with a PML use `SCATTERED_FIELD`, then `.unknown` is the
  scattered field (total = `unknown + hpfem.interpolate(dofs, setup.incident.value)`).
- Errors: invalid input → `ValueError`, not implemented → `NotImplementedError`, other core
  errors → `RuntimeError`; cancellation → `hpfem.Cancelled`.

## 2. Object map

`name(args) -> result — purpose`. "2D" names have a "3D" twin unless stated.

### Mesh
- `hpfem.rectangle(nx, ny, lower=[0,0], upper=[1,1], tag_sides=True) -> Mesh2D`
- `hpfem.box(nx, ny, nz, lower, upper, tag_sides=True) -> Mesh3D`
- `hpfem.disc(n, center, radius=1, curved=True)`, `hpfem.ball(n, center, radius, curved)`
- `hpfem.square_with_disc(n, radius, half_width, outer, inclusion_tag=2) -> Mesh2D` — curved
  disc in a square [−half_width, half_width]², padded to [−outer, outer]² (room for a PML).
- `hpfem.box_with_ball(n, radius, half_width, outer, inclusion_tag=2) -> Mesh3D`
- `hpfem.Mesh2D(vertices (n,2), cells (m,3) int, cell_tags=None)` — from arrays.
- `hpfem.read_gmsh(file, scale=1.0, dim=3)` — Gmsh 4.1 ASCII; **pass dim=2 for 2D**; scale to
  metres (1e-9 for nm). `hpfem.read_gmsh_periodic(...) -> (mesh, [(master, slave, shift)])`,
  `hpfem.read_gmsh_string(text, scale, dim)`.
- Mesh members: `num_cells num_vertices num_facets vertices cells cell_tags facet_tags
  cell_centroids cell_volumes boundary_facets geometry_order`, `cell_centroid(c)`,
  `cell_vertices(c)`, `vertex(v)`, `set_vertex(v, x)`, `set_cell_tag(c, t)`, `set_facet_tag(f,
  t)`, `cells_with_tag(t)`, `facets_with_tag(t)`, `facet_vertices(f)`, `facet_cells(f)`, `copy()`.
- `hpfem.extract(mesh, cells | keep(centroid)->bool) -> Mesh` — submesh (e.g. an L-shape).
- `hpfem.refine_uniform(mesh) -> Refined2D` (`.mesh`, `.parent_cell`); `hpfem.subdivide(mesh, n)`.
- `hpfem.mesh_report(mesh) -> dict` (angles, aspect ratio, invalid / untagged cells).
- `hpfem.check_periodic(mesh, master, slave, shift)`; `hpfem.move_nodes(mesh, velocity, t)`;
  `hpfem.region_normal_velocity(mesh, tag)`; `hpfem.num_geometry_nodes(mesh)`.
- `hpfem.PointLocator2D(mesh)`; `.locate(x) -> LocatedPoint2D | None` (`.cell`, `.xi`).
- `hpfem.meshing` (unit cells of gratings): `UnitCell(period, y_bottom, y_top, slabs=[Slab(tag,
  y0, y1)], shapes=[Shape(kind, tag, params)], background_tag=0)` with kinds `"rectangle"`,
  `"trapezoid"`, `"ellipse"`, `"polygon"`; `hpfem.meshing.structured_unit_cell(cell, nx, rows)`
  (no Gmsh; rows = [[y0, y1, n], ...]); `hpfem.meshing.unit_cell_mesh(cell, sizes) -> (mesh,
  links)` (needs `pip install gmsh`); `hpfem.meshing.element_sizes(materials, omega, p)`;
  `hpfem.meshing.report(mesh, materials)`.

### Spaces
- `hpfem.NedelecDofMap2D(mesh, order | cell_orders)` — H(curl) space for E (never Lagrange for E).
- `hpfem.DofMap2D(mesh, order | cell_orders)` — H1 space (scalars, longitudinal parts,
  temperature, displacement components).
- Members: `num_dofs`, `mesh`, `max_order`, `cell_orders`, `cell_dofs(c)`, `facet_dofs(f)`,
  `dofs_on_tag(tag)`.
- `hpfem.interpolate(dofs, g) -> coeffs`, `hpfem.evaluate_h1(h1, u, locator, x)`,
  `hpfem.evaluate_hcurl(nd, e, cell, xi)`, `hpfem.sample_hcurl(nd, e, cells, xi, curl=False)`,
  `hpfem.discrete_gradient(h1, nd) -> csr`, `hpfem.free_dofs(n, constrained)`,
  `hpfem.cell_averages(dofs, coeffs)`.

### Materials
- Core: `hpfem.Material`, `hpfem.MaterialMap(background)` (`.set(tag, m)`, `.set_cell(c, m)`,
  `.at(tag)`, `.of_cell(mesh, c)`). Lossy metal: Im eps_r > 0.
- `hpfem.materials.get(name)` — library `Si SiO2 Au Ag Ti Cr Al TiO2 Al2O3 GaAs MAPbI3 water
  vacuum air` (tabulated / Sellmeier with sources; `hpfem materials` lists ranges).
- Dispersive models (`.at(omega) -> hpfem.Material`, `.eps_r(omega)`, `.refractive_index(omega)`,
  `.at_wavelength(lam)`):
  `hpfem.materials.Tabulated`, `hpfem.materials.Sellmeier`,
  `hpfem.materials.DrudeLorentz(eps_inf, omega_p, gamma, oscillators)`,
  `hpfem.materials.Drude(eps_inf, omega_p, gamma)`, `hpfem.materials.Constant(value)`.
- `hpfem.materials.DispersiveMap(background)` — `.set(tag, model | name | number)`, `.at(omega)
  -> MaterialMap`, `.apply(setup, omega)` (sets omega and materials).
- `hpfem.materials.with_policy(m, "clamp")` — tabulated data outside range: default raises.
- `hpfem.materials.fit_drude_lorentz(m, (lam0, lam1, count), oscillators=n) -> (model, max_err)`.
- Crystals: `hpfem.materials.Uniaxial(ordinary, extraordinary, theta, phi)`,
  `hpfem.materials.Anisotropic((e1, e2, e3), hpfem.materials.euler_rotation(a, b, c))`.

### Sources, PML, boundaries
- `hpfem.plane_wave(amplitude, wave_vector) -> IncidentField2D` (`.value(x)`, `.curl(x)`);
  amplitude ⟂ wave_vector, |k| = n·ω/c0.
- `hpfem.dipole_field(position, moment, k)`; `hpfem.gaussian_current(position, moment, sigma,
  omega)` (smooth current for `setup.current`); `hpfem.IncidentField2D(value, curl)` (custom).
- `hpfem.PmlBox2D(lower, upper, thickness[x_min, x_max, y_min, y_max(, z_min, z_max)], k0,
  background_index=1, profile)`; `hpfem.PmlBox2D.uniform(lower, upper, t, k0, n, profile)`.
  [lower, upper] is the **interior**; the layers lie outside it and must be meshed.
- `hpfem.PmlProfile(order=3, reflection=1e-8)`, `hpfem.PmlProfile.for_angle(theta_max_deg,
  target)`; `PmlBox2D.recommended_thickness(...)`, `PmlBox2D.max_resolution(...)`.
- Setup fields: `pec_tags` (tangential E = 0), PMC is natural (untagged boundary),
  `incident_tags` (prescribe the incident trace), `periodic = [hpfem.PeriodicPair2D(master,
  slave, shift, phase)]` with `hpfem.bloch_phase(k, shift)`, `ports = [hpfem.WaveguidePort(tag,
  num_modes)]`, `background = LayerStack2D` (layered scattered-field background).
- Layered media: `hpfem.Layer(material, thickness)`, `hpfem.LayerStack2D(incidence, [layers],
  substrate, top=0.0)` (normal along y in 2D, z in 3D), `.plane_wave(k0, angle, pol)` (Fresnel
  R / T); `hpfem.AnisotropicStack(...)` (Berreman 4×4); `hpfem.Polarisation.S` / `.P`.

### Physics
- `hpfem.ScatteringSetup2D()` fields: `omega materials incident formulation pml pec_tags
  incident_tags periodic ports current background solver condense keep_factorisation progress`.
  `hpfem.Scattering2D(nd, setup)`: `.solve()`, `.estimate(sol)`, `.error(sol, exact)`,
  `.total_field(sol, locator, x)` (None outside), `.scattered_field(...)`,
  `.sample(sol, locator, points, quantity="E"|"H"|"S") -> (values, cells)` (NaN / −1 outside),
  `.triangulate(sol, subdivisions)`, `.assemble()`, `.interior_cells()`.
  `hpfem.Formulation.SCATTERED_FIELD` / `.TOTAL_FIELD`.
- Mirror symmetry (any excitation): mesh the reduced side (`hpfem.extract(mesh, keep)`), tag the
  cut with `hpfem.tag_mirror_plane(mesh, hpfem.MirrorPlane2D(point, normal, tag))`, describe the
  reduced domain in the setup (PML thickness 0 at the planes, plane tags NOT in `pec_tags`),
  then `sym = hpfem.MirrorSymmetricScattering2D(nd, setup, planes)` (≤ Dim mutually orthogonal
  planes, any orientation): `.solve()`, `.field(sol, locator, points, scattered=False)` anywhere
  in the full domain, `.cross_sections(sol, surface, amp)`, `.absorbed_power(sol)`,
  `hpfem.MirrorSymmetricFarField2D(sym, sol, surface, order)` — all for the FULL problem.
- Cyclic symmetry C_n (any excitation): cut one sector (`hpfem.extract`), describe it with
  `sym = hpfem.CyclicSymmetry2D(n, center, master=20, slave=21)` (3D: `axis=`), tag its sides with
  `hpfem.tag_sector_sides(mesh, sym, direction_of_master_side)`, setup for the sector (side tags
  NOT in `pec_tags`; a `PmlBox` is symmetric only for n = 2, 4), then
  `prob = hpfem.CyclicSymmetricScattering2D(nd, setup, sym)`: `.solve()`, `.field(...)`,
  `.cross_sections(...)`, `.absorbed_power(...)`, `hpfem.CyclicSymmetricFarField2D(...)` — FULL
  problem. Low level: `hpfem.cyclic_pair(master, slave, n, m, center, axis)` in `setup.periodic`.
  `hpfem.hexagon(n, radius, inner, inner_tag)` is a C_6-symmetric mesh.
- `hpfem.ScatteringOperator2D(problem)` — one factorisation, `.solve(incident)`,
  `.solve_many([...])`; `hpfem.plane_wave_sweep(problem, wave_vectors, polarisation_fn)`;
  `hpfem.ReducedBasis(num_dofs)`; `hpfem.s_parameters(nd, setup)` (ports).
- `hpfem.WaveguideSetup()` (`omega materials pec_tags num_modes`) +
  `hpfem.PropagatingMode(nd, h1, setup).solve() -> [WaveguideMode]` (`effective_index`, `beta`,
  `transverse`, `longitudinal`) — modes of a 2D cross-section.
- `hpfem.ResonanceSetup2D()` (`target_omega materials pml pec_tags num_modes krylov_dimension`)
  + `hpfem.Resonance2D(nd, setup).solve() -> [ResonantMode]` (`omega` complex, `quality`,
  `wavelength`, `residual`, `field`) — quasi-normal modes with PML.
- Closed cavities: `hpfem.assemble_maxwell(nd, hpfem.MaxwellForm2D())` (`.stiffness`, `.mass`)
  + `hpfem.gauged_curl_curl_eigenpairs(K, M, G, free_nd, free_h1, hpfem.EigenOptions(...))`
  (no spurious zero modes). Generic pencils: `hpfem.complex_eigenpairs_near(A, B, sigma, opts)`.
- `hpfem.RieszProjection2D(resonance, hpfem.RieszSetup())` — modal expansion of a source problem.
- `hpfem.BandStructureSetup2D()` (`lattice=[PeriodicPair2D...]`, `num_bands`, `keep_modes`) +
  `hpfem.BandStructure2D(nd, h1, setup)`: `.bands(k) -> Bands2D` (`wavenumber`, `normalised(a)`),
  `.path(corners, segments)`; `hpfem.group_velocity(problem, bands)`,
  `hpfem.band_permittivity_derivative`, `hpfem.band_shape_derivative`.
- `hpfem.TimeDomainSetup2D()` (`dt pec_tags absorbing_tags current signal`) +
  `hpfem.TimeDomain2D(nd, setup)`: `.initialize(u0, v0)`, `.run(state, steps, observer)`,
  `.energy(state)`; signals `hpfem.gaussian_pulse(t0, w)`, `hpfem.modulated_gaussian(omega, t0,
  w)`, `hpfem.TimeSignal(f, df)`. Newmark-β.
- `hpfem.ConicalScatteringSetup()` (`omega beta materials incident pml pec_tags periodic
  background scalar_ez`) + `hpfem.ConicalScattering(nd, h1, setup)` — 2.5D: x-y cross-section,
  invariant along z with exp(iβz); solution has `.transverse`, `.longitudinal` blocks (scaled);
  read physical (E_x, E_y, E_z) via `.total_field`, `.sample`. Waves:
  `hpfem.conical_plane_wave(amp3, k3)`, `hpfem.layered_conical_wave(stack, k0, angle, azimuth,
  pol)` (`.field`, `.beta`, `.kx`, `.reflectance`). `hpfem.ConicalSweep(nd, h1, setup)` reuses
  the assembly over a sweep; `hpfem.ConicalResonance` for cell resonances.
- Axisymmetric 2.5D (meridian mesh, x = r ≥ 0): `hpfem.AxisymmetricScatteringSetup()` (`omega
  materials axis_tag azimuthal_order incident pml background`) +
  `hpfem.AxisymmetricScattering(nd, h1, setup)`; `hpfem.axial_plane_wave(amp, k, m)`,
  `hpfem.layered_axisymmetric_wave(stack3d, k0, theta, "s"|"p", m)`,
  `hpfem.axisymmetric_poynting_flux`, `hpfem.axisymmetric_far_field`,
  `hpfem.axisymmetric_fields_at`; `hpfem.AxisymmetricCavity`, `hpfem.AxisymmetricResonance`.
  Sum results over the azimuthal orders m.
- Heat: `hpfem.ThermalSetup()` (`background_conductivity`, `conductivity={tag: k}`,
  `fixed_temperature=[(tag, T)]`) + `hpfem.Thermal2D(h1, setup)`: `.solve(q)` (q = source
  density coeffs), `.solve_load(load)`; optical load from `hpfem.absorbed_power_load(nd, E_total,
  omega, materials, h1)`.
- `hpfem.ThermoOpticalSetup2D()` (`optical` = ScatteringSetup2D, `thermal` = ThermalSetup,
  `thermo_optic={tag: dε/dT}`, `tolerance`) + `hpfem.ThermoOptical2D(nd, h1, setup).solve()` →
  state (`temperature`, `materials`, `absorbed_power`, `converged`).
- Elasticity (small strain, eigenstrain; 2D = plane strain):
  `hpfem.ElasticitySetup3D()` fields `background=hpfem.ElasticMaterial(E, nu)`, `materials={tag:
  ElasticMaterial}`, `displacements=[hpfem.DisplacementCondition3D(tag, [bx, by, bz], value=None)]`,
  `tractions=[hpfem.TractionCondition3D(tag, t(x))]`, `body_force`, `eigenstrain(x)` (3×3),
  `shrinkage(x)` (isotropic −s I), `set_shrinkage_grid(hpfem.GridField3D(axes, values))`,
  `temperature`, `thermal_expansion`, `reference_temperature`, `remove_rigid_modes`,
  `static_condensation`, `axisymmetric` + `axis_tag` (2D meridian, (u_r, u_z)).
  `hpfem.Elasticity3D(h1, setup)`: `.solve() -> u`, `.displacements_at(u, pts)` (NaN outside),
  `.stress(u, cell, xi)`, `.strain(...)`, `.strain_energy(u)`, `.deformed_mesh(u, scale=1)`,
  `.mesh_velocity(u)`, `.hoop_stress(...)`, `.estimate(u)`, `.energy_error(u, exact, grad)`.
- Analytic references: `hpfem.mie_cylinder_scattering_width(k, r, n)`, `hpfem.mie_sphere(k, r,
  eps_r, background_index)` (`.scattering_cross_section()`, ...).

### Post-processing
- `hpfem.Surface2D.around_cells(mesh, tag)`, `.boundary(mesh, facet_tag)`, `.plane(mesh, axis,
  coord, direction)`, `.whole_boundary(mesh)`.
- `hpfem.cross_sections(problem, sol, surface, incident_amplitude) -> CrossSections`
  (`scattering`, `absorption`, `extinction`; per unit length in 2D).
- `hpfem.FarField2D(mesh, surface, hpfem.discrete_field(nd, sol.unknown), omega, background,
  order)`: `.pattern(direction)`, `.radiated_power()`, `.scattering_cross_section(amp)`.
- `hpfem.poynting_flux(mesh, surface, field, omega, materials, order)`.
- `hpfem.absorbed_power(problem, sol)`, `hpfem.absorbed_power_by_tag(problem, sol)` (`.total`,
  `.by_tag`, `.per_cell`), `hpfem.absorption_density(problem, sol)`,
  `hpfem.absorbed_power_per_cell(nd, e, omega, materials)`.
- Gratings by hand: `hpfem.diffraction_orders`, `hpfem.fourier_coefficients`,
  `hpfem.diffraction_efficiencies`, `hpfem.power_balance`, `hpfem.bloch_constraints`,
  `hpfem.conical_fourier_coefficients`, `hpfem.conical_diffraction_efficiencies`,
  `hpfem.conical_cross_sections`, `hpfem.ConicalFarField`, `hpfem.conical_power_balance`.
- Crossed gratings (3D, periodic along two lattice vectors): `hpfem.OrderPlane(origin, a1, a2,
  normal)` (normal away from the structure) and `hpfem.diffraction_orders(field, plane, k0,
  index_plane, k_tangential, kn_incident, incident=None, max_order=2)` -> list of
  `hpfem.DiffractionOrder3D` (`m`, `n`, `k_tangential`, `kn`, `propagating`, `amplitude`,
  `efficiency`; m outer, n inner). Reflected: `field = lambda x: problem.total_field(sol, loc, x)`,
  `incident = problem.incident_wave`, plane above; transmitted: total field, plane below,
  `index_plane` of the substrate.
- Errors vs exact: `hpfem.hcurl_error(nd, e, field, curl)`, `hpfem.h1_error(h1, u, g, grad_g)`.
- Export: `hpfem.FieldExporter2D(mesh, subdivisions).hcurl("E", nd, e).h1("T", h1, u)
  .write("f.vtu")`; `hpfem.VtkWriter2D(mesh)`; `hpfem.write_vtu_facets(mesh, file)`.
- `hpfem.interop`: `to_meshio`, `from_meshio`, `field_to_meshio`, `to_pyvista`,
  `field_to_pyvista`, `plot_mesh`, `plot_field(dofs, e)`, `plot_convergence`, `plot_far_field`
  (`hpfem.interop.plot_field` etc.; optional meshio / pyvista / matplotlib).

### Adaptivity
- `problem.estimate(sol) -> Estimate` (`.indicators` per cell, `.total()`).
- `hpfem.dorfler_marking(indicators, theta)`, `hpfem.maximum_marking(indicators, gamma)`.
- `hpfem.hp_decide_by_prediction(indicators, predicted, marked) -> HpDecision` (`h_marked`,
  `p_marked`); `hpfem.hp_decide(h1, u, marked)` (coefficient decay).
- `hpfem.AdaptiveMesh2D(root)` (`.mesh` = **copy** of the leaf mesh, `.max_level`,
  `.set_periodic`); `hpfem.hp_refine(adaptive, orders, h_marked, p_marked) -> HpStep` (`.orders`,
  `.step`); `hpfem.predict_indicators(indicators, orders, hp)`; `hpfem.p_refine(orders, marked)`;
  `hpfem.refine_at_points(adaptive, points, levels)`.
- `hpfem.prolongate(old_dofs, old_coeffs, new_dofs, hp.step)` — transfer a solution.
- Goal-oriented: `hpfem.dwr_estimate(problem, sol, hpfem.point_value_functional(x, w))`,
  `hpfem.conical_dwr_estimate(problem, sol, hpfem.conical_point_functional(x, w))`,
  `hpfem.conical_order_functional(...)` → `GoalEstimate` (`value`, `error`, `indicators`).
- Driver: `hpfem.adaptive_solve(adaptive, factory(mesh, orders) -> (problem, sol), observe=,
  goal=, tolerance=, max_dofs=)` yields `hpfem.AdaptiveStep` (`dofs`, `eta`, `observables`,
  `goal_error`, `converged`); needs a problem with `free_dofs` (ConicalScattering).
- Guaranteed bounds: `hpfem.dual_solution`, `hpfem.hypercircle_estimate`.

### Solvers, threads, memory
- Every setup has `solver = hpfem.DirectSolverBackend.AUTO | SPARSE_LU | MUMPS | CUDSS`;
  PyPI wheels ship SPARSE_LU only — check `hpfem.available_backends()`, `hpfem.available(b)`,
  `hpfem.cudss_status()`.
- `hpfem.solve_direct(A, b)`, `hpfem.make_direct_solver()` (`.factorize(A)`, `.solve(b)`,
  `.solve_many`); `hpfem.EigenOptions(num_eigenvalues, shift, krylov_dimension, tolerance)`.
- `hpfem.estimate_memory(mesh, order)` / `(nd)` → `MemoryEstimate` (`.describe()`) — run before
  large 3D solves; `hpfem.grating.estimate_memory(mesh, order)`.
- `hpfem.set_num_threads(n)`, `hpfem.num_threads()`, `hpfem.set_log_level("warn")`.
- Progress / cancel: `setup.progress = callback(hpfem.ProgressEvent)`; grating functions take
  `progress=`, `cancel=` (raises `hpfem.Cancelled`).
- `hpfem.sweep.solve_sweep(task, values, processes=4)` — parallel sweeps over processes (task at
  module level, picklable results).

### High-level: gratings (`hpfem.grating`)
Unit cell: period along x, stack normal along y, Bloch faces tagged `box_tag.X_MIN` / `X_MAX`,
PML regions included at top and bottom of the mesh, z = direction of the lines.
- `hpfem.grating.solve(mesh, materials{tag: Material|model}, stack, "s"|"p", theta, phi, omega,
  order=4, *, pml=None|{"top": t, "bottom": t}, bottom="pml"|"pec"|"modal", orders_max=3,
  keep_factorisation=False, progress, cancel) -> GratingResult`: `R_orders`, `T_orders`
  (`m`, `efficiency`, `amplitude`, `kt`, `kn`, `propagating`), `R`, `T`, `A`, `A_by_tag`,
  `power_balance_residual`, `flux_balance`, `field(points, quantity)`, `diagnostics`, `timing`.
- `hpfem.grating.validate(...)` (same args) → list of `hpfem.diagnostics.Diagnostic` (`code`,
  `severity`, `text`, `hint`); errors raise `hpfem.grating.GratingError`.
- `hpfem.grating.jones(...) -> JonesResult` (2×2 Jones, `mueller`), `hpfem.grating.mueller(J)`,
  `hpfem.grating.operator(...)` (cover + substrate incidence on one factorisation).
- Derivatives: `hpfem.grating.sensitivity(result, tag, order=0, side="R")`,
  `hpfem.grating.shape_sensitivity(result, velocity)`, `hpfem.grating.jacobian(result,
  [("eps", tag), ("shape", V), "theta", "wavelength"]) -> (J, rows, cols)`.
- Resonances: `hpfem.grating.resonances(mesh, materials, stack, omega_target, kx=0, beta=0,
  num_modes=4) -> ResonanceResult` (`modes[i].omega`, `.Q`, `.wavelength`, `.field(points)`),
  `hpfem.grating.bands(..., kx_values)`, `hpfem.grating.refine_resonance`,
  `hpfem.grating.resonance_sensitivity`.
- Emitters in the cell (dipole dict `{"position": (x, y), "moment": (px, py, pz) | "x" |
  "isotropic", "sigma": s}`): `hpfem.grating.emit` (Bloch array), `hpfem.grating.dipole_emission`
  (single dipole, Purcell, up / down / nonradiated), `hpfem.grating.emission_pattern` (dP/dΩ),
  `hpfem.grating.emission_cone(…, aperture)`, `hpfem.grating.emission_cost` (dry run cost).
- Diagnostics alone: `hpfem.diagnostics.validate_scattering(mesh, setup, orders)`,
  `hpfem.diagnostics.raise_on_errors(found)`.

### High-level: crossed gratings in 3D (`hpfem.crossed`, needs `gmsh`)
- Model: `hpfem.crossed.CrossedGrating(period=(px, py), superstrate, substrate, layers=[
  hpfem.crossed.Layer(thickness, material, shapes=[hpfem.crossed.Shape(kind, material, center,
  radius | rx, ry | width, height | points, angle)])])`; kinds circle, ellipse, rectangle,
  polygon; stack normal z, superstrate above z = 0, layers downwards, cell [0, px] x [0, py];
  materials as Material, index (complex n + ik), library name or dispersive model.
- `hpfem.crossed.solve(grating, wavelength, theta, phi, polarisation="s" | "p" | (a_s, a_p),
  order=3) -> CrossedResult`: `.reflected` / `.transmitted` (lists of `CrossedOrder`: `m`, `n`,
  `efficiency`, `amplitude`, `direction`), `.order(m, n, "R" | "T")`, `.R`, `.T`,
  `.absorption`, `.balance` (ΣR + ΣT + A - 1), `.table()`, `.polarised(pol)` (no new solve).
  Superstrate lossless, substrate a lossless dielectric. s: E perpendicular to the plane of
  incidence (normal incidence, phi = 0: E along -y), p: E in it (E along +x).

- FourierGrating models (RCWA config, `"dimension": "2D"`): `hpfem.fouriergrating.load(path)`
  -> `Model` (`.grating`, `.incidences`, `.fg_to_hpfem(points_um)`, `.hpfem_to_fg_field(e)`),
  `hpfem.fouriergrating.solve(model, order=3)` -> [(incidence, CrossedResult)],
  `hpfem.fouriergrating.write_efficiencies(results, path)` (FourierGrating's CSV and order labels);
  CLI `hpfem fg model.json -p 4 -o outdir`. Frames: FG (x, y, z) = hpfem (x, -z, y - H); FG psi
  = -phi; FG delta -> (cos d, -sin d); FG order (n, m) = hpfem (n, -m).

### High-level: other modules
- `hpfem.planar.dipole_above_stack(stack, k0, height, moment) -> Purcell factor` (semi-analytic).
- `hpfem.project`: `load(path)`, `validate(spec)`, `run(spec | path) -> dict` (JSON-able,
  complex as [re, im]); `hpfem.project.ProjectError`. Problems: scattering, waveguide, cavity,
  resonance.
- `hpfem.run.run_job(job, out_dir, emit, cancel) -> dict` — grating jobs (schema v1: model /
  mesh / materials / stack / incidence / sweep / solver / maps; v2 tasks `optimize`,
  `reconstruct`, `uq`, `emitter`, plus `resonances`, `bands`); full schema in
  `help(hpfem.run)`. `hpfem.run.JobError`.
- `hpfem.opt` (needs `import hpfem.opt`): `DesignSpace`, `Continuous`, `Integer`,
  `Categorical`, `LinearConstraint`, `Study(evaluator, path, space)` (cached, resumable JSON-lines
  store), `minimize`, `fit` (Levenberg–Marquardt + Laplace errors), `bayesian_optimize`,
  `OutcomeConstraint`, `pareto_front`, `GaussianProcess`, `GratingEvaluator(morph, materials,
  stack, [Configuration(wavelength, theta, phi, polarisation, orders)])`, `Morph(cell, mesh,
  params)`, `MaterialParameter`, `GeometryParameter`, `trapezoid_parameters`,
  `linear_propagation`, `build_global_surrogate`, `monte_carlo`, `sobol_indices`, `Normal`,
  `Uniform`, `sample` (emcee, extra `opt-mcmc`). E.g. `hpfem.opt.minimize`, `hpfem.opt.fit`.
- `hpfem.dataports` (BindOptiX envelopes): `planar_emitter_response(stack, wavelength, heights,
  detectors=[Detector(name, side, theta_min, theta_max)])` → `.table.to_json()`,
  `planar_field_map`, `grating_field_map(result, RegularGrid)`, `axisymmetric_emitter_response`,
  `axisymmetric_field_map`, `meridian_mesh` (e.g. `hpfem.dataports.planar_emitter_response`).
- `hpfem.pv`: `generation_per_cell(nd, e, omega, materials)`, `spectral_weights`,
  `spectral_sum`, `depth_profile`, `write_generation` (carrier generation for device solvers).
- `hpfem.gui.main` — `hpfem-gui` launcher (Streamlit; `pip install "hpfem[gui]"`).

## 3. Recipes

### R1 Mie cylinder, 2D scattering with PML (in-plane E)
```python
import hpfem
hpfem.set_log_level("warn")
k = 6.0                                                   # vacuum wavenumber [1/m]
mesh = hpfem.square_with_disc(4, 0.25, 1.0, 2.5, 2)       # disc r=0.25 tag 2, PML room to |x|=2.5
nd = hpfem.NedelecDofMap2D(mesh, 3)
s = hpfem.ScatteringSetup2D()
s.omega = k * hpfem.constants.c0
s.materials.set(2, hpfem.Material.dielectric(1.5))
s.incident = hpfem.plane_wave([0.0, 1.0], [k, 0.0])
s.formulation = hpfem.Formulation.SCATTERED_FIELD
s.pml = hpfem.PmlBox2D.uniform([-1, -1], [1, 1], 1.5, k, 1.0, hpfem.PmlProfile(2, 1e-8))
s.pec_tags = [hpfem.box_tag.X_MIN, hpfem.box_tag.X_MAX, hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX]
problem = hpfem.Scattering2D(nd, s); sol = problem.solve()
cs = hpfem.cross_sections(problem, sol, hpfem.Surface2D.around_cells(mesh, 2), 1.0)
print(cs.scattering, hpfem.mie_cylinder_scattering_width(k, 0.25, 1.5))  # agree to ~2e-4
E = problem.total_field(sol, hpfem.PointLocator2D(mesh), [0.5, 0.1])
```

### R2 Slab waveguide mode (effective index)
```python
import hpfem
mesh = hpfem.rectangle(96, 2, [-12.0, 0.0], [12.0, 0.5])
for c, x in enumerate(mesh.cell_centroids):
    if abs(x[0]) < 0.5:
        mesh.set_cell_tag(c, 2)                           # core of thickness 1
setup = hpfem.WaveguideSetup()
setup.omega = 2.0 * hpfem.constants.c0                    # k0 = 2
setup.materials.set(2, hpfem.Material.dielectric(1.5))
setup.pec_tags = [hpfem.box_tag.X_MIN, hpfem.box_tag.X_MAX, hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX]
setup.num_modes = 2
modes = hpfem.PropagatingMode(hpfem.NedelecDofMap2D(mesh, 3), hpfem.DofMap2D(mesh, 3), setup).solve()
print(modes[0].effective_index, modes[0].beta)
```

### R3 Resonance (quasi-normal mode) of a Fabry–Pérot slab with PML
```python
import hpfem
mesh = hpfem.rectangle(32, 1, [-4.0, 0.0], [4.0, 0.125])
for c, x in enumerate(mesh.cell_centroids):
    if abs(x[0]) < 0.5:
        mesh.set_cell_tag(c, 2)
k_ref = 0.97 * 3.14159 * 4 / 3.5
setup = hpfem.ResonanceSetup2D()
setup.target_omega = k_ref * hpfem.constants.c0
setup.materials.set(2, hpfem.Material.dielectric(3.5))
setup.pec_tags = [hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX]
setup.pml = hpfem.PmlBox2D([-1.0, 0.0], [1.0, 0.125], [3.0, 3.0, 0.0, 0.0], k_ref, 1.0,
                           hpfem.PmlProfile(2, 1e-10))
setup.num_modes, setup.krylov_dimension = 3, 40
for m in hpfem.Resonance2D(hpfem.NedelecDofMap2D(mesh, 3), setup).solve():
    print(m.omega / hpfem.constants.c0, m.quality)        # Im < 0: decaying
```

### R4 Closed PEC cavity (gauged, no spurious modes)
```python
import numpy as np, hpfem
mesh = hpfem.rectangle(6, 6)
nd, h1 = hpfem.NedelecDofMap2D(mesh, 2), hpfem.DofMap2D(mesh, 2)
def boundary(d):
    return np.unique(np.concatenate([d.facet_dofs(f) for f in mesh.boundary_facets]))
sys_ = hpfem.assemble_maxwell(nd, hpfem.MaxwellForm2D())
res = hpfem.gauged_curl_curl_eigenpairs(
    sys_.stiffness, sys_.mass, hpfem.discrete_gradient(h1, nd),
    hpfem.free_dofs(nd.num_dofs, boundary(nd)), hpfem.free_dofs(h1.num_dofs, boundary(h1)),
    hpfem.EigenOptions(num_eigenvalues=5, krylov_dimension=30))
print(res.eigenvalues / np.pi**2)                         # 1, 1, 2, 4, 4
```

### R5 Grating efficiencies (conical incidence) with `hpfem.grating.solve`
```python
import hpfem
from hpfem import grating, units
nm = units.nm; P, H, c = 400 * nm, 148 * nm, 148 * nm / 6; SUB, RIDGE = 2, 3
mesh = hpfem.rectangle(16, 71, [-P / 2, -33 * c], [P / 2, 38 * c])   # y = stack normal
for i in range(mesh.num_cells):
    x, y = mesh.cell_centroid(i)
    if y < 0: mesh.set_cell_tag(i, SUB)
    elif y < H and abs(x) < 100 * nm: mesh.set_cell_tag(i, RIDGE)
glass = hpfem.Material.dielectric(1.5)
stack = hpfem.LayerStack2D(hpfem.Material.dielectric(1.0), [], glass, 0.0)
omega = units.angular_frequency(wavelength=405 * nm)
r = grating.solve(mesh, {SUB: glass, RIDGE: glass}, stack, "p", 50 * units.deg, 30 * units.deg,
                  omega, order=3, pml={"top": 17 * c, "bottom": 17 * c}, orders_max=2)
print({o.m: o.efficiency for o in r.R_orders if o.propagating}, r.T, r.power_balance_residual)
```
(Dispersive: pass `hpfem.materials.get("Au")` in the dict; it is evaluated at `omega`.)

### R6 hp-adaptive loop (L-shaped domain, exact corner field as boundary data)
```python
import numpy as np, hpfem
def E(x):                                     # grad(r^(2/3) sin(2t/3)): re-entrant corner field
    r, t = np.hypot(x[0], x[1]), np.arctan2(x[1], x[0]) % (2 * np.pi)
    if r == 0: return np.zeros(2, complex)
    dr, dt = (2 / 3) * r ** (-1 / 3) * np.sin(2 * t / 3), (2 / 3) * r ** (-1 / 3) * np.cos(2 * t / 3)
    return np.array([np.cos(t) * dr - np.sin(t) * dt, np.sin(t) * dr + np.cos(t) * dt], complex)
root = hpfem.extract(hpfem.rectangle(4, 4, [-1, -1], [1, 1]), lambda c: not (c[0] > 0 and c[1] < 0))
for f in root.boundary_facets: root.set_facet_tag(f, 9)
s = hpfem.ScatteringSetup2D(); s.omega = hpfem.constants.c0            # k = 1
s.incident = hpfem.IncidentField2D(E, lambda x: np.zeros(1, complex))  # exact field, curl 0
s.incident_tags = [9]; s.current = lambda x: -E(x)                    # curl curl E - k^2 E = f
adaptive = hpfem.AdaptiveMesh2D(root)
orders, predicted = np.ones(root.num_cells, dtype=int), np.zeros(0)
for step in range(6):
    mesh = adaptive.mesh                                  # a copy: take it once per step
    problem = hpfem.Scattering2D(hpfem.NedelecDofMap2D(mesh, orders.tolist()), s)
    sol = problem.solve(); est = problem.estimate(sol); err = problem.error(sol, s.incident)
    print(problem.dofs.num_dofs, est.total(), np.hypot(err.l2, err.curl))
    marked = hpfem.dorfler_marking(est.indicators, 0.5)
    d = hpfem.hp_decide_by_prediction(est.indicators, predicted, marked)
    hp = hpfem.hp_refine(adaptive, orders.tolist(), d.h_marked, d.p_marked)
    predicted = hpfem.predict_indicators(est.indicators, orders.tolist(), hp)
    orders = np.asarray(hp.orders)
```
Same loop for `ConicalScattering` (both maps get `orders`) and `Elasticity2D/3D` (`estimate(u)`);
transfer old solutions with `hpfem.prolongate(old_nd, old_coeffs, new_nd, hp.step)`. With a PML
(e.g. R1 as root mesh) the loop converges too, but the residual estimator measures the error of
the stretched problem: an under-resolved layer (the "PML is under-resolved" warning) takes most
of eta and of the new unknowns; resolve the layer (milder `PmlProfile` target, thicker layer) or
compare with uniform p for quantities of the physical domain.

### R7 Elasticity: shrinkage of a block bonded at its base (quarter model)
```python
import numpy as np, hpfem
a, h = 20e-6, 2e-6
mesh = hpfem.box(6, 6, 3, [0, 0, 0], [a, a, h])
tag = hpfem.box_tag
st = hpfem.ElasticitySetup3D()
st.background = hpfem.ElasticMaterial(4.5e9, 0.35)                 # E [Pa], nu
st.displacements = [hpfem.DisplacementCondition3D(tag.X_MIN, [True, False, False]),  # symmetry
                    hpfem.DisplacementCondition3D(tag.Y_MIN, [False, True, False]),
                    hpfem.DisplacementCondition3D(tag.Z_MIN, [True, True, True])]   # clamped
z = np.linspace(0, h, 5)
st.set_shrinkage_grid(hpfem.GridField3D([[0.0], [0.0], list(z)], 0.02 + 0.02 * z / h))
st.static_condensation = True
problem = hpfem.Elasticity3D(hpfem.DofMap3D(mesh, 2), st)
u = problem.solve()                                       # component-major coefficients
print(problem.displacements_at(u, np.array([[0, 0, h], [a, a, h]])).real, problem.strain_energy(u))
eta = problem.estimate(u).total(); moved = problem.deformed_mesh(u)
```

### R8 Gold nanowire: absorbed power, then temperature
```python
import hpfem
from hpfem import materials, units
r, omega = 50 * units.nm, units.angular_frequency(wavelength=520 * units.nm)
k0 = units.vacuum_wavenumber(omega)
mesh = hpfem.square_with_disc(2, r, 4 * r, 8 * r, 2)
nd, h1 = hpfem.NedelecDofMap2D(mesh, 3), hpfem.DofMap2D(mesh, 3)
sides = [hpfem.box_tag.X_MIN, hpfem.box_tag.X_MAX, hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX]
s = hpfem.ScatteringSetup2D(); s.omega = omega; s.pec_tags = sides
s.materials.set(2, materials.get("Au").at(omega))                  # Im eps_r > 0
s.incident = hpfem.plane_wave([0.0, 1.0e6], [k0, 0.0])               # 1 MV/m
s.formulation = hpfem.Formulation.SCATTERED_FIELD
s.pml = hpfem.PmlBox2D.uniform([-4 * r, -4 * r], [4 * r, 4 * r], 4 * r, k0, 1.0, hpfem.PmlProfile(2, 1e-10))
sol = hpfem.Scattering2D(nd, s).solve()
total = sol.unknown + hpfem.interpolate(nd, s.incident.value)      # scattered -> total field
load = hpfem.absorbed_power_load(nd, total, omega, s.materials, h1)  # W/m per H1 DoF
th = hpfem.ThermalSetup(); th.background_conductivity = 0.6; th.conductivity = {2: 315.0}
th.fixed_temperature = [(t, 300.0) for t in sides]                 # K on the outer walls
T = hpfem.Thermal2D(h1, th).solve_load(load)
print(hpfem.evaluate_h1(h1, T, hpfem.PointLocator2D(mesh), [0.0, 0.0]).real)
```
Self-consistent coupling: `hpfem.ThermoOptical2D(nd, h1, setup).solve()` with
`setup.optical = s`, `setup.thermal = th`, `setup.thermo_optic = {2: deps_dT}`.

### R9 Project file + CLI (no Python modelling)
```json
{"problem": "scattering", "dim": 2, "length_unit": "m",
 "mesh": {"type": "square_with_disc", "n": 4, "radius": 0.25, "half_width": 1.0, "outer": 2.0, "inclusion_tag": 2},
 "order": 3, "wavelength": {"value": 1.0471975511965976, "unit": "m"},
 "materials": {"background": "vacuum", "2": {"n": 1.5}}, "formulation": "scattered_field",
 "source": {"type": "plane_wave", "angle": 0.0, "angle_unit": "deg", "polarisation": "TE"},
 "pml": {"lower": [-1, -1], "upper": [1, 1], "thickness": 1.0, "profile": {"order": 2, "reflection": 1e-10}},
 "boundaries": {"pec": ["x_min", "x_max", "y_min", "y_max"]},
 "outputs": {"cross_sections": {"around_tag": 2}, "far_field": {"around_tag": 2, "directions": 36},
             "points": [[0.5, 0.1]]}}
```
`hpfem validate p.json && hpfem run p.json -o results.json`, or
`hpfem.project.run(hpfem.project.load("p.json"))["results"][0]["cross_sections"]`.
Materials: library name, `{"n": 1.5}`, `{"eps_r": [re, im]}`; a list in `wavelength.value` sweeps.

### R10 Grating job through the runner (JSON-lines events)
```python
from hpfem import run
job = {"version": 1, "problem": "grating", "unit": 1e-9,
       "model": {"period": 400, "y_bottom": -800, "y_top": 950, "background_tag": 1,
                 "slabs": [{"tag": 2, "y0": -800, "y1": 0}],
                 "shapes": [{"kind": "rectangle", "tag": 3,
                             "params": {"x": -100, "y": 0, "width": 200, "height": 148}}]},
       "mesh": {"structured": {"nx": 16, "rows": [[-800, 0, 32], [0, 148, 6], [148, 950, 32]]}},
       "materials": {"1": "air", "2": {"n": [1.5, 0.0]}, "3": {"eps": [2.25, 0.0]}},
       "stack": {"incidence": "air", "substrate": {"n": [1.5, 0.0]}, "layers": [], "top": 0},
       "incidence": {"polarisation": "p", "theta_deg": 50, "phi_deg": 30},
       "sweep": {"wavelength": [405]},
       "solver": {"order": 3, "pml": {"top": 419, "bottom": 419}, "orders_max": 2}}
results = run.run_job(job, "out", print)            # or: python -m hpfem.run job.json --out out
print(results["points"][0]["R"], results["points"][0]["T"])
```

### R11 Optimisation study (resumable)
```python
import hpfem.opt as opt
def rosenbrock(p, jacobian=False):
    x, y = p["x"], p["y"]; f = (1 - x) ** 2 + 100 * (y - x * x) ** 2
    return (f, [-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)]) if jacobian else f
space = opt.DesignSpace([opt.Continuous("x", -2, 2), opt.Continuous("y", -1, 3)])
study = opt.Study(rosenbrock, "rb.study.jsonl", space)   # rerun = replay from the store
result = opt.minimize(study, 0, x0=[-1.2, 1.0])            # L-BFGS-B with the Jacobian
print(result.params, result.value, study.best().values)
```

### R12 Crossed grating: hole array in a glass film (3D)
```python
import hpfem
from hpfem import crossed as cg
um = 1e-6
g = cg.CrossedGrating((0.5 * um, 0.5 * um), superstrate=1.0, substrate=1.5, layers=[
    cg.Layer(0.15 * um, 1.5, [cg.Shape("circle", 1.0, center=(0.25 * um, 0.25 * um),
                                       radius=0.125 * um)])])
r = cg.solve(g, 0.405 * um, theta=0.0, phi=0.0, polarisation="s", order=3)  # ~1 min, 40k DoF
print(r.table(), r.R, r.T, r.balance)       # balance -> 0 as order grows
print(r.polarised("p").order(1, 0, "T").efficiency)
```

## 4. Gotchas

- `hpfem.read_gmsh` defaults to `dim=3`: a 2D mesh needs `dim=2`; set `scale` to metres.
- `AdaptiveMesh2D.mesh` returns a new copy each access: take it once per step and build all maps,
  locators and exporters of that step on that object.
- PML: [lower, upper] is the interior; the mesh must contain the layers outside it; end the
  outer boundary with `pec_tags`. A warning "the PML is under-resolved" means the stretch is too
  steep for the cells: thicken the layer (`PmlBox2D.recommended_thickness(k0, n, h, profile, p)`)
  or take a milder profile (`PmlProfile(2, 1e-8)`) instead of ignoring it. Thickness list order x_min, x_max, y_min, y_max(, z_min, z_max).
  Use `SCATTERED_FIELD` with a PML; `TOTAL_FIELD` is the default.
- `Scattering2D` has no E_z; for E_z use `ConicalScattering` (β = 0) or `grating.solve(..., "s")`.
- Tensor materials (`hpfem.Material.anisotropic`, `hpfem.materials.Uniaxial`): `Scattering2D`
  refuses a tensor that couples the plane to z (use `ConicalScattering`); a feature that does not
  take crystals reports the diagnostic `anisotropic_unsupported`.
- Plane-wave amplitude must be perpendicular to the wave vector; |k| = n ω / c0 in the medium.
- Dispersive models must be frozen with `.at(omega)` before going into a core setup;
  `hpfem.grating.*` and `DispersiveMap.apply` do it for you. Tabulated data outside its range
  raises unless `materials.with_policy(m, "clamp")`.
- Python callbacks (`current`, `eigenstrain`, custom fields) run under the GIL inside the
  assembly: correct but slow; prefer `plane_wave`, materials by tag, `set_shrinkage_grid`.
- Scattered field on a layered background (`setup.background = LayerStack3D(...)`): give the
  cells of every layer and of the substrate their material by tag as well; unlisted tags are
  `materials.background` (vacuum), and a substrate meshed as vacuum becomes a scatterer.
- In `hpfem.grating` the mesh must have the period along x and the stack normal along y, with
  PML cells inside the mesh; cells straddling a stack interface raise `GratingError` (put
  interfaces on mesh lines). Angles in radians.
- Resonance eigen solvers return the `num_modes` closest to `target_omega`; filter PML modes by
  residual / quality / field location. Raise `krylov_dimension` if not converged.
- 3D direct solves are memory-bound: check `hpfem.estimate_memory(mesh, order)` first; PyPI wheels
  have only SparseLU.
- A free elastic body needs `remove_rigid_modes = True` (loads must balance).
- `hpfem.opt`, `hpfem.planar`, `hpfem.dataports` must be imported explicitly. BoTorch drivers
  (`pareto_optimize`, `multi_fidelity_optimize`) are experimental and need the `opt-bo` extra.
- Results are per unit length in 2D (W/m, cross-sections in m); 3D in W, m².
- Silence per-solve logging with `hpfem.set_log_level("warn")`.

# Roadmap

The project task list. Claude Code picks the lowest unchecked item of the current
milestone (CLAUDE.md §11). Each item is done when the Definition of Done (CLAUDE.md §7)
holds. Estimated effort is in rough "focused sessions".

## M0 — Scaffold ✅
- [x] CMake + presets, strict warnings, FetchContent deps
- [x] CI matrix (gcc/clang/asan), clang-format, clang-tidy, docs deploy
- [x] Python package skeleton (scikit-build-core + pybind11)
- [x] Docs site, theory pages, ADRs, this roadmap

## M1 — Mesh infrastructure (≈ 6) ✅
- [x] `mesh::Mesh<Dim>`: vertices, cells (tri/tet), **derived edges and faces** with
      consistent global orientation (lowest-vertex-first rule, ADR-0003)
- [x] entity connectivity tables (cell→edge, cell→face, face→cell, edge→cells)
- [x] boundary and material tags (physical groups)
- [x] Gmsh `.msh` v4 reader (ASCII), simple structured generators for tests
- [x] affine geometry mapping: Jacobian, det, inverse-transpose, per cell
- [x] curved (order-2) geometry hook (interface only; implementation in M4)
- [x] uniform refinement (red refinement in 2D and 3D, Bey's rule) → prepares M5
- [x] unit tests: Euler characteristic, orientation consistency, boundary extraction
- [x] VTK (`.vtu`) export of mesh + cell data

## M2 — Scalar FEM on the infrastructure (≈ 4)
- [x] `fespace::ReferenceElement<Dim>` with **fixed local numbering** (theory/nedelec.md)
- [x] Gauss–Jacobi quadrature on triangle/tetrahedron, exactness tests up to order 20
- [x] hierarchical H1 Lagrange basis, arbitrary order p (Legendre/Jacobi based)
- [x] `fespace::DofMap`: vertex/edge/face/cell DoFs, variable p per entity
- [ ] global sparse assembly (COO → CSR), element loop with `std::span`
- [ ] Dirichlet via DoF elimination
- [ ] solver interface + Eigen SparseLU backend
- [ ] **convergence test #1**: Poisson, manufactured solution, rate p+1 in L2, p in H1

## M3 — Nédélec elements and Maxwell eigenproblems (≈ 8)
- [ ] hierarchical H(curl) Nédélec-I basis, 2D (p ≤ 6) and 3D (p ≤ 4), per
      Schöberl–Zaglmayr (theory/nedelec.md)
- [ ] orientation-aware local→global mapping of edge/face functions
- [ ] curl–curl stiffness and mass matrices with complex tensor ε, μ
- [ ] PEC boundary condition, PMC (natural)
- [ ] Maxwell eigenproblem: shift-invert Arnoldi (Spectra) on `(S − σM)`
- [ ] gauge handling / kernel filtering (discrete gradients); spurious-mode test
- [ ] **convergence test #2**: PEC box eigenvalues, rate 2p; zero spurious modes
- [ ] field evaluation at arbitrary points (point location + reference-coordinate inversion)
- [ ] VTK export of vector fields (cell-averaged + high-order via subdivision)

## M4 — Time-harmonic scattering & waveguides (≈ 10)
- [ ] `physics::Scattering`: total/scattered-field formulation, plane-wave and
      dipole sources (theory/maxwell.md)
- [ ] PML as complex coordinate stretching in boundary layer cells (theory/pml.md),
      polynomial profile, adaptive σ from wavelength and cell size
- [ ] Bloch-periodic constraints (`E(x+a) = e^{ik·a} E(x)`) via constrained DoFs
- [ ] curved elements (isoparametric order 2) + PML-compatible quadrature
- [ ] post-processing: Poynting flux through surfaces, absorption, scattering
      cross-section, far field (Stratton–Chu), Fourier/diffraction coefficients
- [ ] `physics::PropagatingMode`: 2D waveguide cross-section eigenproblem for
      effective index (quadratic → linearized)
- [ ] **convergence test #3**: PML reflection < 1e-6 for plane wave
- [ ] **convergence test #4**: Mie cylinder (2D) cross section vs. series
- [ ] **convergence test #5**: slab waveguide n_eff
- [ ] **convergence test #6**: lamellar grating efficiencies vs. RCWA
- [ ] examples: `cavity_modes`, `mie_cylinder`, `slab_waveguide`, `lamellar_grating`

## M5 — Adaptivity: a-posteriori estimation and hp-refinement (≈ 10)
- [ ] residual-based estimator for curl–curl (theory/error-estimation.md):
      element residual + tangential-curl jump + normal-displacement jump
- [ ] Dörfler marking
- [ ] conforming h-refinement with **hanging edges/faces → constrained DoFs**
      (one-irregular rule), re-numbering, transfer of solution (prolongation)
- [ ] p-refinement: per-entity order increase, minimum rule on shared entities
- [ ] hp-decision: Legendre-coefficient decay / analyticity estimate per element
- [ ] goal-oriented (dual-weighted) estimator for a scalar quantity of interest
      (e.g. a Fourier coefficient) — needed for scatterometry accuracy claims
- [ ] **convergence test #7**: exponential convergence at re-entrant corner (L-shape,
      then plasmonic wedge)
- [ ] example `plasmonic_dimer`

## M6 — Solvers & performance (≈ 8)
- [ ] MUMPS / PARDISO backend behind `solvers::DirectSolver`
- [ ] static condensation of interior (cell-bubble) DoFs
- [ ] OpenMP parallel assembly (colouring or per-thread COO buffers)
- [ ] parameter sweeps / reduced basis hooks (frequency, angle, geometry parameters)
- [ ] optional: MPI domain decomposition (own ADR before starting)
- [ ] benchmarks recorded in `benchmarks/results/`

## M7 — Python API & usability (≈ 6)
- [ ] bindings for mesh, spaces, materials, problems, solvers, post-processing
- [ ] `hpfem.units` (nm, µm, eV, THz → SI) and `hpfem.materials` library
      (Si, SiO₂, Au, Ag, Al, TiO₂, GaAs, perovskite — tabulated n,k with sources)
- [ ] JSON/YAML project files (`hpfem run project.json`) + CLI
- [ ] meshio / pyvista interop, matplotlib helpers
- [ ] Jupyter example notebooks

## M8 — Application examples (≈ 6)
- [ ] `metasurface_unitcell`: phase/transmission map over pillar diameter
- [ ] `vcsel_cavity`: DBR micro-cavity mode, Q-factor, resonance wavelength
- [ ] `quantum_dot_purcell`: dipole in micropillar, Purcell factor, β-factor
- [ ] `euv_mask`: 3D absorber on multilayer, oblique incidence, near-field export
- [ ] `ring_resonator`: coupling + resonance (2D effective-index model first)

## M9 — Multiphysics (≈ 8)
- [ ] absorbed-power density → heat-conduction solve on the same mesh (H1)
- [ ] temperature-dependent ε feedback loop
- [ ] carrier-generation profile export for PV device solvers

## Backlog / ideas
- dual H-formulation for guaranteed error bounds
- Floquet-Bloch band structure solver
- transient (time-domain) solver via implicit time stepping
- GPU assembly

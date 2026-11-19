# 0016 — Linear elasticity with eigenstrain on a vector H1 space

**Status:** proposed
**Date:** 2026-11-19

## Context

Issue #213 (proposal from Voxelspaetzle, the two-photon printing process model, its decision D4 of
2026-11-07) asks for the shrinkage of printed polymer parts during development and post-curing: the
shape error and focal shift of printed micro-optics, observed for example as a changed dispersion of
a printed spectrometer (Toulouse et al., Light Adv. Manuf. 2 (2021) 2). The model is small-strain
linear elasticity with an eigenstrain:

```
−∇·σ = 0,   σ = C : (ε(u) − ε*),   ε(u) = ½(∇u + ∇uᵀ),   ε* = −s(α) I
```

- **Shrinkage.** s is the isotropic shrinkage; it follows from the degree of polymerisation α, which
  Voxelspaetzle delivers as a volume field.
- **Material.** C is isotropic, given by (E, ν) per element.
- **Boundary conditions.** The part is clamped on the substrate face and traction-free elsewhere.
- **Wanted output.** The displacement at the nodes and at surface points.
- **Acceptance.** Three references: a free sphere with homogeneous shrinkage (exact u = −s r), a
  bonded thin plate (analytic), and the patch test.

Facts from the code (`main` at `115befe`):

- **Scalar H1 space.** `fespace::DofMap<Dim>` and `H1Basis<Dim>` give the hierarchical H1 space with
  variable order per entity and hanging-node constraints (`assembly::hanging_constraints`).
  - The vertex functions are the P1 hat functions, so a vertex coefficient is the value at the
    vertex.
  - `assembly::block_constraints` combines the constraints of two spaces on a block vector; the
    axisymmetric solver uses it for its (e | v) block.
- **Model problem.** `physics::Thermal<Dim>` (M9) solves an H1 problem with a per-tag coefficient,
  Dirichlet data on tagged facets and a load from another field on the same mesh. It is the
  pattern to follow.
- **Linear algebra is complex only.** `SparseMatrix` is complex, and so are the `LinearSolver`
  backends: SparseLU, MUMPS (zmumps) and cuDSS. `Symmetry::kComplexSymmetric` and `kDetect` select
  LDLᵀ in MUMPS and cuDSS. There is no real-valued sparse path.
- **Evaluation.** `assembly::evaluate_h1` evaluates a scalar H1 field at a reference or physical
  point (point location), and `assembly::StaticCondensation` eliminates the cell-interior DoFs.
- **Shape derivatives.** ADR-0011 describes a geometry change as a mesh velocity V on the geometry
  nodes: `shape_sensitivity(gradient, V) = Σ ∂Q/∂x · V`. A displacement field u on the same mesh is
  exactly such a V.
- **Data ports.** `hpfem.dataports.RegularGrid` is the regular grid of the data ports in µm in the
  sensor frame.

## Decision

### 1. Formulation

The unknown is the displacement u ∈ [H¹(Ω)]^Dim, Dim = 3, or 2 in plane strain. The weak form:

```
find u with u = g on Γ_D such that for all v with v = 0 on Γ_D:
    ∫ C ε(u) : ε(v) dx = ∫ C ε* : ε(v) dx + ∫ f · v dx + ∫_{Γ_N} t · v ds
```

- **Eigenstrain.** ε* is a general symmetric tensor field. Isotropic shrinkage (−s I) and thermal
  expansion (α_th ΔT I) are special cases.
- **Stiffness.** C is given per cell as the 6×6 Voigt matrix (3×3 in 2D). Isotropic materials enter
  through (E, ν), with λ = Eν/((1+ν)(1−2ν)) and μ = E/(2(1+ν)). The interface admits anisotropic C
  later without a change.
- **Units.** SI: u in m, E and σ in Pa.
- **Theory.** The weak form and the conventions go to `docs/theory/elasticity.md`.

Body forces f and tractions t are part of the form from the start. They cost almost nothing, and
the manufactured-solution test needs f.

### 2. Space and numbering

The vector space is **Dim copies of the existing scalar `DofMap<Dim>`**, with the same orders on
every entity.

- **Numbering.** The block vector is component-major: [u_x | u_y | u_z].
- **Hanging nodes.** The constraints are the scalar ones, repeated per component through
  `block_constraints`.
- **Condensation.** Static condensation works per cell on the interior DoFs of all components.
- **p-refinement.** It changes all components of a cell together.

No new basis, no new DoF map, and no new constraint code are needed.

### 3. Number type

The system is real symmetric positive definite. We **assemble it in `Complex`, with a zero imaginary
part**, and solve it with the existing backends under `Symmetry::kComplexSymmetric`, which selects
LDLᵀ in MUMPS and cuDSS.

- **Why.** The principle "everything is complex" (CLAUDE.md §2) holds. Elasticity uses the whole
  solver stack as it is: MUMPS, cuDSS, the kept factorisation and the optimisation layer.
- **Later uses.** A complex modulus (viscoelastic damping) or time-harmonic elastodynamics (acousto-
  and opto-mechanics) need no change of type.
- **Cost.** Memory doubles and the factorisation takes about 4× the flops of a real LDLᵀ.
- **Real path.** A real-valued backend is an optimisation behind `LinearSolver` (an amendment to
  ADR-0007). It is added only when a measured problem needs it.

### 4. Boundary conditions

- **Dirichlet per component on tagged facets.** A setup entry gives (facet tag, components, value):
  - all components zero clamps the part on the substrate face;
  - one component zero on an axis-aligned plane is a symmetry plane, used for the 1/8 sphere and
    the laterally constrained layer.

  The elimination is the one of `Thermal`, done per component.
- **Neumann.** Tractions on tagged facets. Untagged boundaries are traction-free, which is the
  natural condition.
- **Free bodies.** The rigid-body modes are removed by `remove_rigid_modes`: 6 Lagrange multipliers
  in 3D and 3 in 2D, bordering the system, which fix the mean translation and the mean rotation to
  zero. The bordered system is symmetric indefinite, and the direct solvers handle it. We choose
  this over pinning single DoFs because it does not depend on the mesh and leaves the stresses
  untouched.

### 5. Eigenstrain and data input

The eigenstrain enters as a **callable evaluated at the quadrature points**: `ε*(x)` as a 3×3
symmetric matrix, or `s(x)` for isotropic shrinkage. There are three sources:

- **Python callables.** Functions and NumPy-vectorised data.
- **Data port** (`hpfem.dataports`). A volume field on a `RegularGrid`, for example Voxelspaetzle's
  α or s on voxels, interpolated trilinearly. Points outside the grid raise an error.
- **Fields on the same mesh.** A temperature from `Thermal` gives the thermal strain α_th (T − T₀) I.
  This is the thermo-mechanical coupling, at no extra cost.

The map from α to s, the shrinkage law of the resin, belongs to Voxelspaetzle and is not part of
hp-FEM.

### 6. Output

- **Displacement.** The coefficients, and the vertex values, which are the vertex coefficients of
  the hierarchical basis.
- **Point values.** The displacement at arbitrary points through point location, for example on
  the surface for the shape error.
- **Strain and stress.** ε(u) and σ at points, and the strain energy ½∫ε : C : ε.
- **Deformed mesh.** The mesh with its geometry nodes moved by u, for an optical solve on the
  shrunk part.
- **Mesh velocity.** u passed to `shape_sensitivity` as the ADR-0011 velocity V. This gives the
  first-order change of an optical quantity caused by the shrinkage without remeshing, for example
  the focal shift or the phase of a lens.

### 7. Locking

The formulation is purely in displacements. The printed polymers have ν ≈ 0.35–0.45, where the
hierarchical space with p ≥ 2 does not lock in practice. The setup **warns for ν > 0.45** and
**refuses ν ≥ 0.5**. A mixed u–p formulation is an alternative (see below) and is not part of this
decision.

### 8. Verification

These are acceptance criteria. Each is a unit or convergence test as in CLAUDE.md §8.

1. **Patch test.** Every linear displacement field, with body force 0 and matching tractions, is
   reproduced to round-off for p ≥ 1, also on meshes with hanging nodes.
2. **Manufactured solution.** A smooth u with the corresponding f and t. The convergence test
   requires rate p + 1 in L² and p in the energy norm, for p = 1 … 4 on cubes and triangles.
3. **Free sphere with homogeneous shrinkage.** The exact field is linear, u = −s r with zero
   stress. It is reproduced with `remove_rigid_modes`, and also on the 1/8 sphere with symmetry
   planes. On curved second-order meshes the geometry error remains, so the test checks against
   it.
4. **Laterally constrained layer on a rigid substrate.** The sides are symmetry planes and the base
   is clamped. The exact 1D solution is u_z = −s z (1 + ν)/(1 − ν), with in-plane stress
   σ_xx = σ_yy = s E/(1 − ν).
5. **Bonded finite plate.** The film on a substrate (bimetal / Stoney curvature), compared with
   Timoshenko's analytic plate solution away from the edges. The tolerance is set by the
   plate-theory error, and the test is documented as a validation in `docs/validation.md`.

### 9. Stages

A new milestone **M20 — Linear elasticity (ADR-0016)** in `docs/roadmap.md`:

| Stage | Content |
|---|---|
| S1 | Vector H1 assembly and `physics::Elasticity<Dim>` (3D and 2D plane strain): Dirichlet per component, tractions, body force. Tests 1 and 2. |
| S2 | Eigenstrain fields, the data-port input and `remove_rigid_modes`. Tests 3 and 4. |
| S3 | Output: points, stress, energy, the deformed mesh. Python API (`hpfem.Elasticity3D` / `2D`). The bonded plate (test 5). An example: a printed micro-lens or cube shrinking. |
| S4 | Coupling to optics: an optical solve on the deformed mesh, and u as the shape velocity (focal shift to first order), checked against the optical solve on the deformed mesh. |
| S5 (optional) | Axisymmetric elasticity (u_r, u_z) for printed lenses, which are often bodies of revolution. A residual estimator and hp-adaptivity for elasticity. A real-valued backend if measured necessary. |

## Consequences

- **Little new code.** The space, the constraints, the condensation, the solvers and the point
  evaluation already exist. The new code is the vector assembly with the C : ε forms, the
  per-component Dirichlet data, the bordered rigid-mode system, and the outputs.
- **A second physics on the same meshes.** Voxelspaetzle can drop its own P1 tetrahedral solver
  once S2 passes its references. The meshes come from Nurbforge or gmsh, as for the optical
  problems.
- **Memory and time** are about 2–4× those of a real solver at the same size. Target problems
  (printed parts of 10–100 µm at p = 2–3) are around 10⁵–10⁶ unknowns, so MUMPS or cuDSS is needed
  at the upper end. This is the main cost of decision 3.
- **The optical coupling (S4)** turns shrinkage into optical tolerances directly. The ADR-0011
  velocity interface needs no change, because u lives on the same geometry nodes.
- **Locking** for ν close to 0.5 is not addressed. The checks of §7 make that visible instead of
  failing silently.
- **Small strain only.** Shrinkage of a few percent is within linear elasticity. Large rotations or
  strains would need a geometrically nonlinear formulation, which is not planned.

## Alternatives considered

- **A real-valued assembly and solver path from the start.** It halves memory and cuts the time.
  But it duplicates the solver interface and the backends (dmumps, real cuDSS), breaks the uniform
  stack (kept factorisation, `hpfem.opt`), and contradicts CLAUDE.md §2.4 without a measured need.
  Deferred to §3 as an opt-in optimisation.
- **A dedicated vector H1 basis and DoF map** (interleaved, node-major numbering). Interleaving
  helps band solvers, but the direct solvers reorder anyway. It would duplicate the hanging-node
  and p-refinement machinery that the component-major block reuses as it is.
- **Pinning DoFs to remove the rigid-body modes.** It is simple, but it depends on the mesh and
  creates local stress concentrations. The Lagrange multipliers are exact and cost 6 extra rows.
- **A mixed u–p (displacement–pressure) formulation** against locking. It is necessary only near
  ν = 0.5, it needs inf–sup stable pairs and a saddle-point solver, and it is not needed for the
  target polymers. It can be added later as a separate ADR.
- **An external FEM package** (FEniCS, deal.II) called from Python. It brings a second mesh
  pipeline and a second set of conventions, and loses the coupling to the optical solvers on the
  same mesh, which is the point of #213 (§6, S4).

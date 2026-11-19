# Linear elasticity with eigenstrain

Small-strain linear elasticity on the vector H1 space (ADR-0016, milestone M20). The model is
the shrinkage of two-photon printed polymer parts (issue #213). Code:
`assembly/elasticity_forms.hpp` and `physics/elasticity.hpp`. SI units: displacement u in m, E and
σ in Pa, body force f in N/m³, traction t in Pa.

## Equations

Find the displacement u with

$$
-\nabla\cdot\sigma = f,\qquad \sigma = C : (\varepsilon(u) - \varepsilon^*),\qquad
\varepsilon(u) = \tfrac12(\nabla u + \nabla u^\top)
$$

and the following boundary conditions:

- prescribed displacement components on tagged facets (Dirichlet, per component);
- tractions σn = t on tagged facets;
- σn = 0 on the rest of the boundary, which is the natural condition.

The eigenstrain ε\* is a symmetric tensor field. Isotropic shrinkage is ε\* = −s I, thermal
expansion is ε\* = α (T − T₀) I.

The weak form: find u with u = g on Γ_D such that for all v with v = 0 on Γ_D

$$
\int_\Omega C\,\varepsilon(u) : \varepsilon(v)\,dx
= \int_\Omega C\,\varepsilon^* : \varepsilon(v)\,dx + \int_\Omega f\cdot v\,dx
+ \int_{\Gamma_N} t\cdot v\,ds .
$$

## Voigt notation

Strains and stresses are written as vectors. In 3D the order is (xx, yy, zz, yz, xz, xy), in 2D
it is (xx, yy, xy). Shear strains are engineering strains, γ_ij = 2ε_ij, so that σ = C ε holds
with a symmetric C.

The isotropic stiffness, with λ = Eν/((1+ν)(1−2ν)) and μ = E/(2(1+ν)), is

$$
C = \begin{pmatrix} \lambda + 2\mu & \lambda & \lambda & & & \\
\lambda & \lambda + 2\mu & \lambda & & & \\ \lambda & \lambda & \lambda + 2\mu & & & \\
& & & \mu & & \\ & & & & \mu & \\ & & & & & \mu \end{pmatrix}.
$$

In 2D the code uses the plane-strain block, (λ + 2μ, λ; λ, λ + 2μ) and μ. The out-of-plane stress
σ_zz = λ(ε_xx + ε_yy) is carried by the constraint ε_zz = 0 and does not appear in the in-plane
stress.

For a scalar basis function φ_a with gradient g, the strain–displacement matrix of φ_a e_x in 3D
is the column

$$
B_{a,x} = (g_x, 0, 0, 0, g_z, g_y)^\top .
$$

The other components follow by symmetry. Cell by cell, the element matrix is
∫ Bᵀ C B dx and the eigenstrain load is ∫ Bᵀ C ε\*_Voigt dx.

## Space and numbering

The space is Dim copies of the scalar hierarchical `DofMap<Dim>`
([H1 basis and DoF map](h1-basis.md)), numbered component-major: component i of scalar DoF k is
the DoF i·N + k, where N is the scalar size. This lets the vector space reuse the scalar
machinery:

- **Vertex values.** The vertex coefficients are the values at the vertices. A linear
  displacement is reproduced exactly, which is the patch test.
- **Dirichlet data.** Each component is interpolated hierarchically with the scalar
  `dirichlet_values`, and its DoFs are shifted by i·N.
- **Hanging nodes.** The constraints are the scalar ones, repeated per component through
  `block_constraints`.
- **Static condensation.** The exterior DoFs of all components of a cell are ordered first and
  the interior bubbles last; the condensation then works as in the scalar case
  ([linear solvers](solvers.md)).
- **Tractions.** They act on facet DoFs only, because the interior bubbles vanish on the facets.
  The traction load is therefore added after the condensation.

## Eigenstrain sources and free bodies (M20 S2)

`ElasticitySetup` adds up three sources of eigenstrain:

- **`eigenstrain`:** a symmetric tensor field ε\*(x).
- **`shrinkage`:** an isotropic shrinkage s(x), giving −s I. The data path for it is a
  `GridField<Dim>`: a volume field on a regular grid, interpolated linearly per axis (bilinear or
  trilinear), with values in C order as in NumPy. An axis of length 1 means invariance along it,
  and points outside the grid are refused.
- **Temperature:** coefficients on the same scalar map, for example from `Thermal`, give the
  thermal strain α (T − T₀) I. α is given per cell tag. The temperature is evaluated in
  reference coordinates of the cell, so no point location is needed.

**Free bodies (`remove_rigid_modes`, ADR-0016 §4a).**

1. Three far-apart vertices are pinned by the 3-2-1 rule.
2. The load is checked against the rigid-body modes: an unbalanced load is refused.
3. After the solve, the L2 projection onto the six modes (three in 2D) is subtracted, so that the
   mean translation and the mean rotation vanish.

## Output (M20 S3)

- **`displacements_at(u, points)`.** Point location and evaluation, NaN outside the mesh.
- **`strain_energy(u)`.** ½∫ (ε − ε\*) : C : (ε − ε\*).
- **`deformed_mesh(u, scale)`.** The vertices move by their coefficients, which are their
  values. For p ≥ 2 or a curved mesh, one node per edge moves with the displacement at the edge
  midpoint, which gives geometry order 2. The deformed mesh carries quadratic displacements
  exactly and higher ones as their quadratic interpolant.

## Coupling to the optics (M20 S4)

**The velocity.** `mesh_velocity(u)` is Re u on the geometry nodes of the mesh: the vertex
values and, on a second-order mesh, the displacement at the edge midpoints, whose images are the
edge nodes. This is the mesh velocity V of the shape derivatives (ADR-0011).

**First-order change.** For an optical goal Q computed on the same mesh, the shrinkage changes Q
to first order by
`shape_derivative(problem, solution, functional, V)`, or `shape_sensitivity(shape_gradient(…),
V)` when the functional does not move. This takes one adjoint solve and needs no remeshing, for
example for the focal shift or the phase of a printed lens.

**The deformed mesh.** `deformed_mesh(u, t)` is `move_nodes(mesh, V, t)`. An affine mesh with
p ≥ 2 is first raised to geometry order 2. The optical solve on `deformed_mesh(u)` is the full
reference.

**Test** (`test_elasticity.cpp`):
- **Setup.** A resin disc in a soft surrounding, clamped on the outer box, shrinks by 1 %. A
  uniform temperature with α = −s on the disc tag only produces the shrinkage.
- **Goal.** The scattered field at a point, from the in-plane Maxwell solver on the same
  second-order mesh.
- **Result.**
  - The shape derivative along V equals the central difference of the optical solve on
    `deformed_mesh(u, ±10⁻⁶)` to 2e-4.
  - It predicts the change under the full shrinkage to 0.9 %, a remainder of order s.

## Bodies of revolution (M20 S5)

With `axisymmetric = true` (2D only) the mesh is the meridian half-plane x = r ≥ 0, y = z. The
unknowns are (u_r | u_z): azimuthal order 0, without torsion.

**Kinematics.** The strains in Voigt order (rr, zz, θθ, rz) are

$$
\varepsilon_{rr} = \partial_r u_r,\quad \varepsilon_{zz} = \partial_z u_z,\quad
\varepsilon_{\theta\theta} = \frac{u_r}{r},\quad \gamma_{rz} = \partial_z u_r + \partial_r u_z .
$$

**Integrals and stiffness.** The integrals carry the weight r; the factor 2π is dropped, as in the
axisymmetric Maxwell forms, so energies and loads are per radian. The isotropic stiffness is the
3D one restricted to (xx, yy, zz, xy) ↔ (rr, zz, θθ, rz).

**The axis.** u_r = 0 is imposed on the facets with `axis_tag`. The remaining basis functions of
u_r vanish linearly there, so (φ/r)(φ/r) r is a polynomial and the quadrature (2p + extra + 1)
integrates it.

**Eigenstrains.** Those of the shrinkage and the temperature are isotropic, hoop part included.
`eigenstrain` gives (rr, zz, rz), and `hoop_eigenstrain` gives its θθ part.

**Free bodies.** The only rigid-body mode is the translation along z: one pin, then the
r-weighted projection.

**Output.** `stress` returns (rr, rz; rz, zz). `hoop_strain` and `hoop_stress` return the θθ
parts; on the axis the hoop strain is the limit ∂u_r/∂r.

**Tests** (`tests/unit/physics/test_elasticity_axisymmetric.cpp`):
- **Patch test.** u_r = a r, u_z = b z + d on the meridian rectangle reproduces the exact uniform
  stresses, hoop included, for p = 1 … 3.
- **Free sphere.** On the curved half disc, u = −s x is reproduced to 1e-12 for p = 2 and 3.
- **Hollow sphere under internal pressure** against Lamé's solution:
  - On 4 × 12 cells the relative L² error is 1.3e-3, 9.1e-5 and 1.5e-5 for p = 2, 3, 4.
  - On 8 × 24 cells at p = 4 it is 8.4e-7, and the inner pole agrees to 2.9e-6.
  - The quadratic geometry of the arcs bounds the error from below. That bound falls like h_θ⁴:
    1.3e-5 → 8.0e-7 at p = 5 when the angular step is halved.

## Number type

The matrix is real symmetric positive definite once the Dirichlet data is eliminated. It is
assembled as complex with a zero imaginary part and solved with the direct backends
(ADR-0016 §3).

## Verification

**Unit tests** (`tests/unit/physics/test_elasticity.cpp`):

- **Patch test.** Linear displacements with a rotation part, prescribed on the whole boundary,
  are reproduced to round-off for p = 1 … 3. This holds in 2D and 3D, with and without static
  condensation, and on meshes with hanging nodes.
- **Rigid-body modes.** The six rigid-body modes lie in the kernel of the stiffness.
- **Uniaxial bar.**
  - Setup: two materials in series with equal ν/E, a traction on one end, three symmetry planes.
  - Result: σ = σ₀ e_x e_x, with u_x piecewise linear.
  - Plane strain: ε_xx = (1 − ν²)σ₀/E and ε_yy = −ν(1 + ν)σ₀/E.
- **Uniform shrinkage.** An eigenstrain ε\* = −s I in a block held only by three symmetry planes
  gives u = −s x and σ = 0.

**Unit tests for S2:**

- **Free sphere.** On the curved ball mesh (geometry order 2) with homogeneous shrinkage, the
  exact field u = −s x is reproduced to 1e-12 for p = 2 and 3, with and without condensation.
  This holds for a constant s and for the same s from a `GridField`. The mean translation and
  rotation are zero.
- **Laterally constrained layer.** On a clamped base, u_z = −s z (1 + ν)/(1 − ν) and
  σ_xx = σ_yy = s E/(1 − ν).
- **Free cube with a linear temperature** from `Thermal`.
  - The thermal strain α g x I is compatible: ε = α g x I and σ = 0.
  - A second material with a different α makes it incompatible, and stresses appear.
- **`GridField`.** It reproduces linear functions, is invariant along an axis of length 1, and
  refuses points outside the grid and malformed input.
- **Free-body checks.** An unbalanced load is refused.

**Convergence test** (`tests/convergence/elasticity.cpp`):

- Manufactured solution u_i = a_i Π sin(πx_k), clamped on the whole boundary.
- Body force f = μ Dim π² a s − (λ + μ) H a, where H is the Hessian of s.
- h-refinement gives the rates p + 1 in L² and p in the H1 seminorm: p = 1 … 4 in 2D and p = 1 … 3
  in 3D.

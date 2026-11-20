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

**Convergence test** (`tests/convergence/elasticity.cpp`):

- Manufactured solution u_i = a_i Π sin(πx_k), clamped on the whole boundary.
- Body force f = μ Dim π² a s − (λ + μ) H a, where H is the Hessian of s.
- h-refinement gives the rates p + 1 in L² and p in the H1 seminorm: p = 1 … 4 in 2D and p = 1 … 3
  in 3D.

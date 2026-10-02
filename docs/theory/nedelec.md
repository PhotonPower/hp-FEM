# Nédélec (edge) elements and reference-element conventions

## Why not nodal elements

Nodal Lagrange elements approximate each Cartesian component in $H^1$, i.e. they enforce
full continuity of $\mathbf{E}$ across element faces. Physically only the *tangential*
component is continuous at material interfaces; the normal component jumps by
$\varepsilon_1/\varepsilon_2$. Forcing it to be continuous (a) pollutes the eigenvalue
spectrum with non-physical *spurious modes* and (b) converges to the wrong solution
at re-entrant corners. $H(\mathrm{curl})$-conforming elements fix both: they are tangentially
continuous and reproduce the exact de Rham sequence
$H^1 \xrightarrow{\nabla} H(\mathrm{curl}) \xrightarrow{\nabla\times} H(\mathrm{div}) \xrightarrow{\nabla\cdot} L^2$
on the discrete level, so the discrete kernel of the curl is exactly the gradient of the
discrete $H^1$ space.

## Basis family

hpfem uses the **hierarchical high-order Nédélec first-kind basis of Schöberl & Zaglmayr**
(Zaglmayr, PhD thesis, JKU Linz 2006; Schöberl & Zaglmayr, COMPEL 24 (2005) 374–384).
Properties required by CLAUDE.md §2:

- functions are grouped by entity: **edge** functions (lowest order Whitney + higher
  order gradients of edge-bubbles), **face** functions (gradient and non-gradient type),
  **cell** functions;
- the order $p$ is assignable **per entity**, enabling $p$-adaptivity with a minimum rule
  on shared entities;
- gradient functions are explicit, which makes gauging/kernel filtering trivial.

Shape functions on the reference element are built from Legendre $\ell_i$ and integrated
Legendre $L_i$ polynomials in barycentric coordinates $\lambda_j$; see Zaglmayr Sect. 5.2
for the formulas. The implementation lives in `fespace/nedelec.hpp` and must cite the
equation numbers it implements.

## Reference elements and local numbering (binding)

**Triangle** $\hat T$: vertices $v_0=(0,0)$, $v_1=(1,0)$, $v_2=(0,1)$.
Edges: $e_0=(v_0,v_1)$, $e_1=(v_1,v_2)$, $e_2=(v_2,v_0)$ (counter-clockwise).

**Tetrahedron** $\hat K$: vertices $v_0=(0,0,0)$, $v_1=(1,0,0)$, $v_2=(0,1,0)$, $v_3=(0,0,1)$.
Edges (6): $e_0=(v_0,v_1)$, $e_1=(v_0,v_2)$, $e_2=(v_0,v_3)$, $e_3=(v_1,v_2)$, $e_4=(v_1,v_3)$, $e_5=(v_2,v_3)$.
Faces (4), face $i$ opposite vertex $i$, vertices in increasing local order:
$f_0=(v_1,v_2,v_3)$, $f_1=(v_0,v_2,v_3)$, $f_2=(v_0,v_1,v_3)$, $f_3=(v_0,v_1,v_2)$.

These tables are the single source of truth in `mesh/simplex_topology.hpp`
(`mesh::SimplexTopology<Dim>`, one layer below `fespace` in the module order);
`fespace/reference_element.hpp` re-uses them. Global numbering and orientation of the
derived mesh entities: [Mesh topology](mesh.md).

## Orientation

Edge and face functions depend on the orientation of the entity. Two neighbouring cells
must agree on it. Rule (ADR-0003): **the global orientation of an edge points from the
vertex with the smaller global index to the larger one; a face is oriented by sorting its
global vertex indices ascending.** The local→global DoF map stores, per cell, whether each
local entity agrees with the global orientation and applies sign flips / permutation of
the higher-order face functions accordingly. This rule is independent of mesh generator
output and survives refinement, which is why it is preferred over "orientation from file".

## Mapping

Vector shape functions are mapped covariantly (Piola transform for $H(\mathrm{curl})$):

$$
\phi(x) = J^{-T}\hat\phi(\hat x), \qquad
\nabla\times\phi = \frac{1}{\det J}\, J\, \hat\nabla\times\hat\phi \quad (3D), \qquad
\nabla\times\phi = \frac{1}{\det J}\, \hat\nabla\times\hat\phi \quad (2D).
$$

In 2D the vector field has two components and the curl is a scalar; the $H(\mathrm{curl})$
and $H(\mathrm{div})$ spaces are rotations of each other — do not exploit this silently,
keep the 2D implementation an honest $H(\mathrm{curl})$ space.

## Verification

- Tangential continuity: for every interior face, $\mathbf{n}\times(\phi^+-\phi^-)=0$ at
  quadrature points (unit test on random meshes).
- Discrete de Rham: $\nabla\times\nabla \phi_{H^1} = 0$ exactly, and `rank(curl)` equals
  `dim(H(curl)) − dim(H¹) + #components`.
- PEC cavity eigenvalues $\pi^2(m^2+n^2+l^2)$ with **no** eigenvalues clustering near zero
  beyond the gradient kernel.

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

`fespace::ReferenceElement<Dim>` adds the geometry of the reference simplex: vertices
$e_0 = 0, e_i$, barycentric coordinates $\lambda_0 = 1 - \sum_i \xi_i$, $\lambda_i = \xi_i$
with their constant gradients, the outward unit normal and measure of every local
facet, the vertex opposite a facet, and affine parametrisations of edges ($t \in [0,1]$
from the first to the second local vertex) and facets (reference interval / triangle
onto the facet with its vertices in table order). Quadrature on facets composes
`facet_point` with the rules of [Quadrature](quadrature.md).

## Orientation

Edge and face functions depend on the orientation of the entity. Two neighbouring cells
must agree on it. Rule (ADR-0003): **the global orientation of an edge points from the
vertex with the smaller global index to the larger one; a face is oriented by sorting its
global vertex indices ascending.** The local→global DoF map stores, per cell, whether each
local entity agrees with the global orientation and applies sign flips / permutation of
the higher-order face functions accordingly. This rule is independent of mesh generator
output and survives refinement, which is why it is preferred over "orientation from file".

## Hierarchical basis (`fespace/nedelec_basis.hpp`)

The implemented space is Nédélec's **first kind** $ND_p$ (lowest order $p = 1$: Whitney
functions), $\dim ND_p = p(p+2)$ on the triangle and $p(p+2)(p+3)/2$ on the tetrahedron.
With barycentric coordinates $\lambda$, the Whitney function
$w_{ab} = \lambda_a\nabla\lambda_b - \lambda_b\nabla\lambda_a$ (curl $2\nabla\lambda_a
\times\nabla\lambda_b$), the scaled integrated Legendre kernel
$u_i = L_i^S(\lambda_b - \lambda_a, \lambda_a + \lambda_b)$ and the bubble factor
$v_j(\lambda) = \lambda\,P_j(2\lambda - 1)$ of the [H1 basis](h1-basis.md):

| entity, order $p$ | functions | count |
|---|---|---|
| edge $(a,b)$ | $w_{ab}$; gradients $\nabla u_i$, $i = 2..p$ | $p$ |
| face $(a,b,c)$ / triangle interior | Type 1 $\nabla(u_i v_j(\lambda_c))$ for $i \ge 2$, $j \ge 0$, $i+j+1 \le p$; Type A $w_{ab}\,\lambda_c P_i(2\lambda_a-1)P_k(2\lambda_c-1)$ for $i+k \le p-2$; Type B $w_{bc}\,\lambda_a P_k(2\lambda_a-1)$ for $k \le p-2$ | $(p-1)(p-2)/2 + p(p-1)/2 + (p-1) = p(p-1)$ |
| tetrahedron interior | Type 1 $\nabla(u_i v_j(\lambda_2) v_k(\lambda_3))$ for $i+j+k+2 \le p$; Types A, B $w_{01}\lambda_2\lambda_3\,q$, $w_{02}\lambda_1\lambda_3\,q$ with $q = P_i(2\lambda_1-1)P_j(2\lambda_2-1)P_k(2\lambda_3-1)$, $i+j+k \le p-3$; Type C $w_{03}\lambda_1\lambda_2\,P_j(2\lambda_1-1)P_k(2\lambda_2-1)$, $j+k \le p-3$ | $p(p-1)(p-2)/2$ |

Why these lie in $ND_p = P_{p-1}^d \oplus \{F \in \tilde P_p^d : x\cdot F = 0\}$: gradients of
polynomials of degree $\le p$ are in $ND_p$, and a Whitney function times any polynomial
of degree $\le p-1$ is too, because the linear part of $w_{ab}$ is orthogonal to $x$. The
Whitney families are *complete* on the bubble space ($w_{ab}\lambda_c P_{p-2}$ and
$w_{bc}\lambda_a P_{p-2}$ together already span all $p(p-1)$ interior functions of the
triangle); to keep the gradients explicit, the last family is reduced to polynomials in
one variable fewer, which removes exactly $\dim \nabla W_p^{\text{int}}$ functions. The
obvious alternative, the non-gradient twins $\nabla u\,v - u\nabla v$ of Zaglmayr's
second-kind basis, is *not* usable here: its lowest member is a combination of the
Whitney products (found by the rank test). The counts match $\dim ND_p$, and the
tests verify linear independence (SPD mass matrix) and that every vector monomial of
degree $\le p-1$ and every field $x^\perp q$ / $x \times e_m q$ of degree $p$ is
reproduced exactly, hence the span *is* $ND_p$. The de Rham property
$\nabla W_p \subset ND_p$ is tested by projecting every gradient of the H1 basis.

The gradient functions (edge gradients, Type 1) are marked as such in the function order,
which makes gauging / kernel filtering a matter of index bookkeeping. Zaglmayr's
Jacobi-weighted factors for better conditioning may replace $P_j$ later.

**Function order per cell**: edge 0, 1, … (Whitney, then $\nabla u_2 \dots \nabla u_p$),
then faces (Type 1 in $(i,j)$, Type A in $(i,k)$, Type B in $k$), then the interior
(Type 1 in $(i,j,k)$, then Types A, B, C in $(i,j,k)$). Orientation follows ADR-0003
exactly as for the H1 basis: edges in ascending global vertex order (`edge_flipped`),
faces by the sorted triple (`face_permutations`), so shared entities carry identical
functions from all adjacent cells and the DoF map stores plain ids. `NedelecDofMap`
(`fespace/dof_map.hpp`) numbers $p$ DoFs per edge, $p(p-1)$ per face and the interior
functions per cell with the minimum rule; tangential continuity across every interior
facet is checked by the tests with the covariant Piola map below.

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

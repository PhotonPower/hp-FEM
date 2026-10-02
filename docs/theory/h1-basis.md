# Hierarchical H1 basis and DoF map

The scalar space of M2 (Poisson benchmark, later heat conduction and the gradient kernel of
the Nédélec space) is an **H1-conforming, hierarchical** basis of arbitrary order built
from Legendre and integrated Legendre polynomials in barycentric coordinates, in the
spirit of Zaglmayr (2006, Sect. 5.1) and Šolín et al. (2004). Hierarchical means that the
basis of order $p$ contains the basis of order $p-1$, so the order can be raised per
entity without rebuilding anything (the `hp` of hp-FEM).

## Building blocks (`fespace/polynomials.hpp`)

- Legendre polynomials $P_j$ on $[-1,1]$ by the three-term recurrence.
- Integrated Legendre polynomials $L_1(x) = x$, $L_i(x) = \int_{-1}^x P_{i-1}(s)\,ds$
  for $i \ge 2$, hence $L_i(\pm 1) = 0$: these are the edge "bubbles".
- Their **scaled** versions $L_i^S(x,t) = t^i L_i(x/t)$, homogeneous polynomials of degree
  $i$ in $(x,t)$, evaluated by the recurrence
  $(i+1)\,L^S_{i+1} = (2i-1)\,x\,L^S_i - (i-2)\,t^2 L^S_{i-1}$ with $L^S_1 = x$,
  $L^S_2 = (x^2 - t^2)/2$, and the corresponding recurrences for $\partial_x$ and
  $\partial_t$. No division by $t$ occurs, so the functions and their gradients are exact
  at the vertex opposite an edge, where $t = 0$.

## Shape functions on the reference simplex (`fespace/h1_basis.hpp`)

With barycentric coordinates $\lambda_0, \dots, \lambda_d$ of the
[reference element](nedelec.md#reference-elements-and-local-numbering-binding):

| entity | functions | count |
|---|---|---|
| vertex $v$ | $\lambda_v$ | 1 |
| edge $(a,b)$, order $p_e$ | $L_i^S(\lambda_b - \lambda_a,\ \lambda_a + \lambda_b)$, $i = 2..p_e$ | $p_e - 1$ |
| face $(a,b,c)$, order $p_f$ (3D) | $L_i^S(\lambda_b - \lambda_a,\ \lambda_a + \lambda_b)\ \lambda_c\, P_j(2\lambda_c - 1)$, $i \ge 2,\ j \ge 0,\ i + j + 1 \le p_f$ | $(p_f-1)(p_f-2)/2$ |
| triangle interior, order $p$ | as the face functions with $(a,b,c) = (0,1,2)$ | $(p-1)(p-2)/2$ |
| tetrahedron interior, order $p$ | $L_i^S(\lambda_1 - \lambda_0, \lambda_0 + \lambda_1)\ \lambda_2 P_j(2\lambda_2-1)\ \lambda_3 P_k(2\lambda_3-1)$, $i + j + k + 2 \le p$ | $(p-1)(p-2)(p-3)/6$ |

Why this works: on any facet that does not contain both $a$ and $b$ one of
$\lambda_a, \lambda_b$ vanishes, so the edge kernel becomes $t^i L_i(\pm 1) = 0$; the
factors $\lambda_c$ (and $\lambda_3$) kill the remaining facets. Hence every entity
function vanishes on all entities that do not contain its own, which is exactly the
locality needed for H1 conformity. The counts add up to $\dim P_p$:
$(p+1)(p+2)/2$ on the triangle and $(p+1)(p+2)(p+3)/6$ on the tetrahedron; the tests
verify completeness by projecting every monomial of degree $\le p$ exactly.

**Orientation.** The edge kernel is odd in $x$ for odd $i$, so neighbouring cells must
agree on which vertex is $a$. Following ADR-0003, $a$ is always the vertex with the
*smaller global index*: `H1Layout::edge_flipped` tells the basis when the local edge
direction is reversed, and for faces `face_permutations` gives the ascending global order
of the three face vertices. Both cells then evaluate literally the same function on a
shared entity, and the DoF map stores plain global ids without sign or permutation
information. The DoFs of an entity are ordered by ascending degree ($i$, then $j$, then
$k$), identically in all cells.

**Function order inside a cell**: the $d+1$ vertex functions, then edge $0, 1, \dots$,
then (3D) face $0, 1, \dots$, then the interior functions. `H1Basis::evaluate` returns
values and *reference* gradients $\nabla_\xi \phi$; physical gradients are
$J^{-T} \nabla_\xi \phi$ with the [cell geometry](mesh.md#geometry).

The Jacobi-weighted variants of the face and interior kernels that Zaglmayr uses for
better sparsity are not needed for correctness and can replace $P_j$ later without
changing the interface.

## DoF map (`fespace/dof_map.hpp`)

`DofMap<Dim>(mesh, orders)` numbers the global degrees of freedom: all vertices
($0 \dots V-1$), then the edge functions edge by edge, then (3D) the face functions, then
the interior functions cell by cell. The order of an edge or face is the **minimum** of
the orders of its cells (minimum rule), so the trace spaces of neighbouring cells match
and the global space is conforming for arbitrary per-cell orders. Per cell the map
provides the `H1Layout` (orders and orientation codes straight from the mesh) and the
global ids in basis order, so assembly is a plain gather / scatter. `facet_dofs(f)` lists
the DoFs supported on a facet and `dofs_on_tag(t)` their union over a tagged boundary,
which is what Dirichlet elimination needs.

Verification (`tests/unit/fespace/`): recurrences against explicit polynomials, finite
difference gradients, trace properties, SPD mass matrix and exact reproduction of all
monomials up to $p = 5$ (2D) / $4$ (3D), sign behaviour under flipped edges, DoF counts,
the minimum rule, and the conformity test: on every interior facet of a rectangle and a box
with uniform and random orders, the global functions seen from both cells coincide on the
facet DoFs and vanish for all others.

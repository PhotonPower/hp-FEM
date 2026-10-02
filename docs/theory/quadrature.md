# Quadrature on simplices

`assembly/quadrature.hpp` provides Gauss rules of arbitrary order on the interval, the
reference triangle and the reference tetrahedron of the
[reference element](nedelec.md#reference-elements-and-local-numbering-binding). Mass
terms of order-$p$ bases need exactness $\ge 2p + 1$ (CLAUDE.md §6); PML cells use more
because the stretched coefficients are not polynomial ([PML](pml.md)).

## One-dimensional rules

For $\alpha, \beta > -1$ the $n$-point **Gauss–Jacobi** rule on $[0,1]$ satisfies

$$
\sum_{j=1}^{n} w_j\, g(s_j) = \int_0^1 g(s)\,(1-s)^\alpha s^\beta \, ds
\qquad \text{for all polynomials } g \text{ of degree} \le 2n-1 .
$$

Nodes and weights come from the **Golub–Welsch** algorithm: the nodes are the eigenvalues
of the symmetric tridiagonal Jacobi matrix of the three-term recurrence of the orthonormal
Jacobi polynomials, the weights are $\mu_0 v_{0j}^2$ with $v_{0j}$ the first component of
the $j$-th normalised eigenvector and $\mu_0 = \int (1-x)^\alpha (1+x)^\beta\,dx$ on
$[-1,1]$. The rule is then mapped to $[0,1]$. Gauss–Legendre is the case
$\alpha = \beta = 0$. There is no table and hence no maximum order; the $O(n^3)$ eigenvalue
problem is negligible for the $n \lesssim 30$ used in practice.

## Collapsed (Duffy) rules on the simplex

The triangle is the image of the unit square under
$y = s$, $x = t\,(1-s)$ with $dx\,dy = (1-s)\,ds\,dt$, so

$$
\int_T f \, dA = \int_0^1\!\!\int_0^1 f\big(t(1-s),\, s\big)\,(1-s)\,dt\,ds
\approx \sum_{i,j} w^{(1,0)}_i\, w^{\mathrm{GL}}_j\; f\big(t_j(1-s_i),\, s_i\big),
$$

where $w^{(1,0)}$ is the Gauss–Jacobi rule with $\alpha = 1$ that absorbs the factor
$(1-s)$. A polynomial of total degree $p$ in $(x,y)$ has degree $\le p$ in each of $s$ and
$t$, so $n = \lceil (p+1)/2 \rceil$ points per direction give exactness for total degree
$p$ with $n^2$ points. The tetrahedron collapses twice:

$$
z = s,\quad y = t\,(1-s),\quad x = u\,(1-s)(1-t), \qquad dV = (1-s)^2 (1-t)\, ds\,dt\,du ,
$$

with Jacobi weights $\alpha = 2$ in $s$, $\alpha = 1$ in $t$ and Gauss–Legendre in $u$;
$n^3$ points for total degree $2n-1$.

Properties used by the code and checked by the tests:

- all weights are positive and all points lie strictly inside the simplex (the collapsed
  vertex is never sampled), which keeps the discrete norms definite;
- the weights sum to the reference measure $1/2$ and $1/6$;
- every monomial $x^a y^b z^c$ with $a+b+c \le p$ is reproduced to relative precision
  $10^{-12}$ for $p \le 20$; the exact values are
  $\int_T x^a y^b\,dA = a!\,b!/(a+b+2)!$ and $\int_K x^a y^b z^c\,dV = a!\,b!\,c!/(a+b+c+3)!$;
- `simplex_quadrature<1>` is Gauss–Legendre on $[0,1]$, so facet integrals in 2D and
  cell integrals in 1D share the interface.

Rule selection in the assemblers: degree $2p + \text{extra\_order}$ per cell (`extra_order`
defaults to 2, scattering problems use 4 for the non-polynomial incident fields), plus 2
on curved (order-2) cells whose Piola factors are rational, and `MaxwellForm::quadrature_order`
overrides the degree per cell — the PML layer cells of `physics::Scattering` use
$2p + 6$ (`pml_extra_quadrature_order`) for their rational stretched tensors.

Collapsed rules use more points than symmetric rules of the same degree (e.g. $n^2$ vs.
roughly $(p+1)(p+2)/6$); they are chosen for simplicity, arbitrary order and positivity.
Symmetric rules can be added behind the same `QuadratureRule` type if assembly time ever
matters.

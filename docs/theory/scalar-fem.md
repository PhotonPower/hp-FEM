# Scalar FEM: assembly, Dirichlet conditions, solvers

The scalar pipeline of M2 exercises the whole infrastructure on the model problem
$-\nabla\cdot(\alpha\nabla u) + \beta u = f$ before the Maxwell operators of M3 reuse the
same pieces. Everything is complex-valued (CLAUDE.md §2.4); for the Poisson benchmark the
imaginary parts are simply zero.

## Weak form and element integrals

Find $u \in V_h \subset H^1$ with $u = g$ on $\Gamma_D$ such that for all
$v \in V_h$ vanishing on $\Gamma_D$

$$
a(u, v) = \int_\Omega \alpha\,\nabla u\cdot\nabla v + \beta\,u\,v \,dx
\;=\; \int_\Omega f\,v\,dx = \ell(v).
$$

Test functions are **not** conjugated (ADR-0002): with real basis functions the matrix is
complex symmetric, $A_{ij} = a(\phi_j, \phi_i) = A_{ji}$. On a cell with
[geometry map](mesh.md#geometry) $x(\xi)$ and the
[H1 basis](h1-basis.md) $\hat\phi_i(\xi)$,

$$
A^K_{ij} = \sum_q w_q\,|\det J(\xi_q)|\,\Big[\alpha(x_q)\,
(J^{-T}\hat\nabla\hat\phi_j)\cdot(J^{-T}\hat\nabla\hat\phi_i) + \beta(x_q)\,\hat\phi_j\hat\phi_i\Big],
\qquad
b^K_i = \sum_q w_q\,|\det J|\, f(x_q)\,\hat\phi_i ,
$$

with a [simplex rule](quadrature.md) exact for degree $2p + 2$ by default (`extra_order`
covers non-polynomial coefficients and curved Jacobians). `element_h1` computes
$A^K, b^K$; `assemble_h1` runs the element loop with per-order cached rules.

## Assembly

`SparseAssembler` collects the element entries as COO triplets
`(cell_dofs[i], cell_dofs[j], A^K_{ij})` and `finalize()` compresses them into the
row-major CSR matrix `hpfem::SparseMatrix` with duplicates summed ($O(\mathrm{nnz}\log
\mathrm{nnz})$). `scatter` / `gather` move element vectors. Because the DoF map already
orders the local DoFs like the basis functions and the basis evaluates shared entities
in the global orientation, the element loop contains no sign or permutation logic.

## Dirichlet conditions

**Values.** Dirichlet data $g$ is represented in the trace space by *hierarchical
interpolation* (`dirichlet_values`):

1. vertex DoFs take $g(x_v)$;
2. for every edge $(a,b)$ of the Dirichlet facets, the edge DoFs are the $L^2(e)$
   projection of the remainder $g(x(t)) - (1-t)g_a - t\,g_b$ onto the edge functions
   $L_i(2t-1)$, $i = 2..p_e$ (a $(p_e-1)\times(p_e-1)$ Gram system per edge);
3. in 3D, the face DoFs are the $L^2(F)$ projection of the remainder (after vertices and
   edges) onto the face functions. The trace of the 3D basis on a face is exactly the 2D
   basis on that triangle (with its third local edge reversed), which the implementation
   uses literally.

If $g$ lies in the trace space the result is exact; otherwise it is an interpolant with the
optimal order $p+1$, so it does not spoil the convergence rates.

**Elimination.** `apply_dirichlet` keeps the system size and symmetry:
$b \leftarrow b - A_{:,c}\,g_c$, rows and columns $c$ are cleared, $A_{cc} = 1$,
$b_c = g_c$. The solution of the modified system equals $g$ on the constrained DoFs and
solves the reduced problem on the free ones; the matrix stays usable for direct and
iterative solvers alike.

## Solvers

`solvers::LinearSolver` separates factorisation from solves (many right-hand sides per
factorisation in parameter sweeps). The first backend is Eigen's supernodal `SparseLU`
with COLAMD ordering on the complex CSR matrix; MUMPS / PARDISO and iterative solvers
arrive in M6 behind the same interface.

## Error norms and convergence test #1

`h1_error` integrates $\|u_h - u\|_{L^2}$ and $\|\nabla u_h - \nabla u\|_{L^2}$ (and the
norms of $u$ for relative errors) with the same per-cell rules. The convergence test
(`tests/convergence/poisson.cpp`) solves $-\Delta u = f$ for the manufactured solution
$u = \sin(\pi x)\,e^{y}$ (2D, $f = (\pi^2 - 1)u$) and $u = \sin(\pi x)\,e^{y}\cos z$ (3D,
$f = \pi^2 u$) with Dirichlet data from $u$ on all sides, prints DoF / h / error / rate
tables and asserts

- h-refinement: rate $\ge p + 1 - 0.2$ in $L^2$ and $\ge p - 0.2$ in $H^1$ for
  $p = 1..3$ (2D) and $p = 1..2$ (3D);
- p-refinement on a fixed $2\times2$ mesh: the $L^2$ error at least halves per order and
  reaches $10^{-8}$ by $p = 8$ (exponential convergence for the analytic solution).

Unit tests additionally check the element matrices (mass sums, constants in the stiffness
kernel, symmetry), the patch test (linear solutions reproduced to $10^{-11}$ for
$p = 1..3$), the symmetric elimination against a dense reference, exactness of the boundary
interpolation for trace-space data in 2D and 3D, and the solver on random complex systems.

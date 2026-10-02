# hp-adaptivity

## Goal

For solutions that are piecewise analytic with isolated singularities (material corners,
metal tips, PML interfaces) the error of an appropriately graded $hp$-mesh decays as

$$
\|E - E_{hp}\| \le C \exp\left(-b\, N^{1/(d+1)}\right)
$$

in the number of DoFs $N$ (Babuška & Guo 1986, Schwab 1998), versus algebraic
$N^{-p/d}$ for $h$-refinement and stagnation for pure $p$-refinement near singularities.
This is the headline feature; convergence test #7 (CLAUDE.md §8) must show it.

## Loop

```
SOLVE → ESTIMATE → MARK → DECIDE (h or p) → REFINE → (repeat)
```

1. **Estimate**: element indicators $\eta_K$ from [error-estimation.md](error-estimation.md).
2. **Mark**: Dörfler criterion — smallest set $\mathcal M$ with
   $\sum_{K\in\mathcal M}\eta_K^2 \ge \theta \sum_K \eta_K^2$, $\theta\approx 0.5$.
   Implemented as `adaptivity::dorfler_marking` (`maximum_marking` as the simpler alternative).
3. **Decide** per marked $K$:
   - compute the decay of the expansion coefficients of $E_{hp}|_K$ in an orthogonal
     (Legendre/Jacobi) basis; estimate the Sobolev regularity exponent $\sigma_K$
     (Mavriplis 1994; Houston & Süli 2005; Eibner & Melenk 2007);
   - if $\sigma_K > p_K + 1$ (smooth): **p-refine** ($p_K \leftarrow p_K+1$);
   - else: **h-refine**.
   Alternative to be implemented later: the "reference solution" strategy of Demkowicz
   (solve on $h/2, p+1$ and project) — more expensive but very robust.
4. **Refine**:
   - $h$: red refinement of the marked cells with the **one-irregular rule** (a face or
     edge may be shared by at most two levels), `mesh::AdaptiveMesh::refine`. Hanging
     edges/faces produce **constrained DoFs**: the slave DoFs are linear combinations of
     master DoFs computed by interpolation of the master shape functions on the refined
     entity (`assembly::hanging_constraints`, see below). The constraints reduce the
     assembled system ($P^H A P$) before the Dirichlet data is imposed on the free DoFs.
   - $p$: raise the cell order; shared edges/faces take the **minimum** order of adjacent
     cells (minimum rule) so the space stays conforming.
   - PML cells follow the refinement of the adjacent interior cells; never refine the
     PML alone.
5. **Transfer**: prolongate the previous solution (`assembly::prolongate`, exact by
   hierarchical interpolation) as the initial guess for iterative solvers; direct solvers
   ignore it.

## Hanging nodes and constraints

A hanging edge $(a,b)$ with midpoint $m$ faces the half edges $(a,m)$, $(m,b)$ of the
refined side; a hanging face $(a,b,c)$ faces its four child faces, the three half edges of
each of its edges and the three edges between the midpoints. The DoFs of the fine entities
are **slaves** of the **masters** on the parent entity and its boundary (for a face: the
face DoFs, the DoFs of its three edges and, in H1, its three vertices):

$$
u_s = \sum_m c_{sm}\, u_m, \qquad c_{sm} = \text{coefficient of fine function } s
\text{ in the hierarchical interpolant of master function } \phi_m .
$$

The coefficients are computed numerically: every master function (evaluated through the
coarse cell in global orientation) is interpolated onto the child entities with
`assembly::interpolate`, which is exact because the parent's trace is a polynomial of
degree $p_{\text{parent}} \le p_{\text{child}}$ (the DoF map's minimum rule includes the
children). Fine functions of higher degree than the parent get the coefficient $0$ — they
are constrained to vanish. Hanging faces are processed first, then hanging edges whose
half edges are not yet constrained; a slave never has two constraints. The resulting
space is exactly the conforming subspace: tangential (H(curl)) or full (H1) continuity
across every hanging facet, and a global polynomial of the space is reproduced exactly.
See ADR-0006.

Verification (`tests/unit/assembly/test_hanging_constraints.cpp`, `test_prolongation.cpp`):
random constrained vectors are continuous / tangentially continuous across all hanging
facets of twice-refined corners in 2D and 3D for $p \le 4$ / $3$ and for random per-cell
orders; the number of constrained DoFs equals the count of fine-entity DoFs; a quadratic
field of $\mathrm{ND}_3$ is solved exactly by `physics::Scattering` on the hanging mesh with
vanishing residual estimate; prolongated functions agree with the originals pointwise and
satisfy the constraints of the refined mesh.

### Adaptive h-refinement on the L-shape (`tests/convergence/adaptive_h_refinement.cpp`)

$E = \nabla(r^{2/3}\sin\tfrac{2\theta}{3})$ on $[-1,1]^2 \setminus [0,1]\times[-1,0]$ with
$k = 1$ and the exact trace on the boundary, Dörfler marking with $\theta = 0.5$:

| | uniform, $p = 2$ | adaptive, $p = 1$ | adaptive, $p = 2$ |
|---|---|---|---|
| rate in $N$ (last steps) | $-0.35$ | $-0.58$ | $-1.11$ |
| optimal $-p/2$ | — | $-0.5$ | $-1.0$ |
| effectivity $\eta / \|E - E_h\|$ | 4.1 … 4.5 | 2.1 … 3.8 | 3.4 … 4.7 |

The largest indicator sits at the corner in every step; 16 steps reach $10^{-3}$ relative
error with 12 600 DoFs at $p = 2$ in 3 s. The singular solution limits uniform refinement to
$N^{-1/3}$ for every $p$ (the $h$ part of convergence test #7; the $hp$ part follows with
the $hp$ decision).

## Data-structure consequences (why M1/M2 must prepare this)

- DoF numbering is **entity-based** (vertex → edge → face → cell) with a per-entity order,
  not a per-cell block of fixed size.
- The mesh keeps parent/child relations and level numbers.
- `fespace::Constraints` (`slave = Σ c_i master_i`) serves Bloch-periodic boundaries and
  hanging nodes alike (`append` combines them) and reduces the assembled system by
  $P^H A P$; `physics::Scattering` imposes the Dirichlet data on the free DoFs after the
  reduction.

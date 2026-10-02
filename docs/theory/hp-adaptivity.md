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
3. **Decide** per marked $K$ (see [hp decision](#hp-decision) below):
   - **error prediction** (Melenk & Wohlmuth 2001, `adaptivity::hp_decide_by_prediction`):
     a cell whose indicator fell to the value predicted for a smooth solution after the
     last refinement is **p-refined**, one that fell short is **h-refined** — the default;
   - **coefficient decay** (Mavriplis 1994; Houston & Süli 2005,
     `adaptivity::hp_decide`): the decay rate $\sigma_K$ of the Legendre (Dubiner)
     coefficients of $E_{hp}|_K$, p-refinement above a threshold — reliable only once the
     local singularity is resolved;
   - to be implemented later: the "reference solution" strategy of Demkowicz (solve on
     $h/2, p+1$ and project) — more expensive but very robust.
4. **Refine**:
   - $h$: red refinement of the marked cells with the **one-irregular rule** (a face or
     edge may be shared by at most two levels), `mesh::AdaptiveMesh::refine`. Hanging
     edges/faces produce **constrained DoFs**: the slave DoFs are linear combinations of
     master DoFs computed by interpolation of the master shape functions on the refined
     entity (`assembly::hanging_constraints`, see below). The constraints reduce the
     assembled system ($P^H A P$) before the Dirichlet data is imposed on the free DoFs.
   - $p$: raise the cell order (`adaptivity::p_refine`, or `hp_refine` together with the
     h-step: children inherit the parent's order, p-marked cells are raised); shared
     edges/faces take the **minimum** order of adjacent cells (minimum rule) so the space
     stays conforming. The spaces are nested, so `assembly::prolongate` (with
     `identity_step` for a pure p-step) transfers solutions exactly.
   - PML cells follow the refinement of the adjacent interior cells; never refine the
     PML alone.
5. **Transfer**: prolongate the previous solution (`assembly::prolongate`, exact by
   hierarchical interpolation) as the initial guess for iterative solvers; direct solvers
   ignore it.

## hp decision

### Error prediction (`adaptivity/prediction.hpp`)

After a refinement step every new cell carries the indicator it would have if the solution
were smooth there (Melenk & Wohlmuth 2001):

$$
\eta^{\text{pred}}_{K'} = \gamma_p\,\eta_K \ \text{(p-refined)},\qquad
\eta^{\text{pred}}_{K'} = \frac{\gamma_h\, 2^{-p_K}}{\sqrt{n_{\text{children}}}}\,\eta_K \ \text{(child of } K),\qquad
\eta^{\text{pred}}_{K'} = \gamma_n\,\eta_K \ \text{(unchanged)},
$$

with $\gamma_p = 0.63 \approx \sqrt{0.4}$, $\gamma_h = 2$, $\gamma_n = 1$ (`PredictionOptions`,
the defaults of deal.II). A marked cell with $\eta_{K'} \le \eta^{\text{pred}}_{K'}$ achieved
the smooth rate and is p-refined next, otherwise it is h-refined; in the first step (no
prediction) everything is h-refined. The corner cells of a singular solution never reach
the predicted $2^{-p}$ reduction, so they are h-refined for ever, while cells at a fixed
distance-to-size ratio gain the predicted factor from each p-step.

`hp_refine` with `spread_p` (default) also raises the lower-order facet neighbours of a
p-refined cell by one: under the minimum rule the shared edges would otherwise stay at the
neighbour's order and the raised cell would only gain interior functions, which the
estimator keeps marking. Spreading raised the exponential rate $b$ of the L-shape run from
$0.23$ to $0.28$ and halved the error at $12\,000$ DoFs.

### Coefficient decay (`adaptivity/smoothness.hpp`)

$E_{hp}|_K$ is expanded component-wise in the $L^2$-orthonormal Dubiner basis of the
reference cell (`fespace::DubinerBasis`, Legendre and Jacobi polynomials in collapsed
coordinates). With $a_n^2 = \sum_{|\alpha| = n}|c_\alpha|^2$ the least-squares fit
$\log a_n \approx C - \sigma n$ over $n = 0 \dots p_K$ gives the decay rate; a field
represented exactly ($a_{p} = 0$) counts as infinitely smooth. A cell is p-refined if
$\sigma_K \ge \tau(p_K) = $ `smooth_threshold` $+$ `threshold_scale`$/p_K$ with the
defaults $1 + 3.5/p$, and cells below `min_decision_order` $= 2$ are p-refined without a
decision. The $1/p$ term was calibrated on the interpolant of the corner singularity
$\nabla(r^{2/3}\sin\tfrac{2\theta}{3})$: a sequence that decays algebraically looks
exponential over few modes, and its fitted rate drops with $p$ (corner cells: $4.1, 2.0,
1.4, 1.1, 0.9, 0.7$ for $p = 1 \dots 6$, cells away from the corner stay above
$1.4$).

**Measured limitation.** The decay of the *Galerkin solution* is not the decay of the
exact field: on an unresolved corner cell the discrete solution is a smooth polynomial,
with rates $3.1$–$3.9$ at $p = 2$ and $2.0$–$2.5$ at $p = 4$, above the thresholds, so the
corner is p-refined up to $p = 6$ before the first h-step and the loop converges no
faster than h-adaptivity. This is why the error prediction is the default; the decay
indicator remains available for solutions whose singularities are resolved, and as the
smoothness measure of the prolongated solution in later strategies.

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

### Adaptive p-refinement on a smooth solution (`tests/convergence/adaptive_p_refinement.cpp`)

Plane wave with $k = 6$ on a $4 	imes 4$ mesh of the unit square, exact trace on the
boundary, Dörfler marking ($	heta = 0.8$) of the residual indicators, $p \leftarrow p + 1$
on the marked cells (minimum rule on the shared edges): the error decays like
$\exp(-0.41\sqrt{N})$, from $1$ at $56$ DoFs ($p = 1$) to $2.5\cdot 10^{-8}$ at $2694$ DoFs
($p \le 9$) in 16 steps; the effectivity index stays within $60 \dots 118$ (its constant
grows with $p$). The transferred solution agrees pointwise with the previous one in every
step.

### hp-adaptivity on the L-shape (`tests/convergence/adaptive_hp_refinement.cpp`, test #7)

Same problem, starting from $p = 1$ on 24 cells, Dörfler $\theta = 0.5$, decision by error
prediction, p-spreading, hanging-node h-refinement:

| DoF | 44 | 1 300 | 4 100 | 8 000 | 11 800 | 16 700 | 21 800 | 26 300 |
|---|---|---|---|---|---|---|---|---|
| $\|E - E_{hp}\|_{H(\mathrm{curl})}$ | $6.2\cdot10^{-1}$ | $2.9\cdot10^{-2}$ | $2.4\cdot10^{-3}$ | $9.1\cdot10^{-4}$ | $3.9\cdot10^{-4}$ | $1.6\cdot10^{-4}$ | $9.8\cdot10^{-5}$ | $6.2\cdot10^{-5}$ |

Fit $\exp(-b N^{1/3})$ over the last ten steps: $b = 0.28$; the algebraic slope for
$N \ge 4000$ is $-2.2$, steeper than any fixed-$p$ h-adaptivity with $p \le 4$ reaches,
and the error at $12\,000$ DoFs is $2.7\times$ below the $p = 2$ h-adaptive run. The final
mesh has 20 levels of geometric grading at the corner with $p = 2$ in the innermost rings
and $p$ up to $6$–$7$ outside; the corner cells are never p-refined; the effectivity index
stays between $2$ and $4$. Beyond $\approx 40\,000$ DoFs the error stalls near $10^{-5}$:
the direct solver's accuracy on the graded high-order system (M6).

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
- The mesh keeps parent/child relations and level numbers (`mesh::AdaptiveMesh`).
- `fespace::Constraints` (`slave = Σ c_i master_i`) serves Bloch-periodic boundaries and
  hanging nodes alike (`append` combines them) and reduces the assembled system by
  $P^H A P$; `physics::Scattering` imposes the Dirichlet data on the free DoFs after the
  reduction.

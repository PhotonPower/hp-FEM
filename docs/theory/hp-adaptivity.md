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
   - $h$: bisection (2D) / red refinement (3D) with the **one-irregular rule** (a face or
     edge may be shared by at most two levels). Hanging edges/faces produce
     **constrained DoFs**: the slave DoFs are linear combinations of master DoFs computed
     by interpolation of the master shape functions on the refined entity. Constraints
     are applied during assembly (condensing rows/columns), not by post-hoc elimination.
   - $p$: raise the cell order; shared edges/faces take the **minimum** order of adjacent
     cells (minimum rule) so the space stays conforming.
   - PML cells follow the refinement of the adjacent interior cells; never refine the
     PML alone.
5. **Transfer**: prolongate the previous solution as the initial guess for iterative
   solvers; direct solvers ignore it.

## Data-structure consequences (why M1/M2 must prepare this)

- DoF numbering is **entity-based** (vertex → edge → face → cell) with a per-entity order,
  not a per-cell block of fixed size.
- The mesh keeps parent/child relations and level numbers.
- `DofMap` exposes a `Constraints` object (`slave = Σ c_i master_i`) used by the assembler.
  (`fespace::Constraints` exists since M4 for Bloch-periodic boundaries and currently
  reduces the assembled system by $P^T A P$; condensation during assembly comes with the
  hanging nodes.)

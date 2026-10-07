# A-posteriori error estimation

## Residual estimator for the curl–curl problem

For $E_{hp}$ solving the discrete problem with data $f = i\omega J$ (Beck, Hiptmair,
Hoppe & Wohlmuth 2000; Schöberl 2008 for the $hp$ setting), the element indicator is

$$
\eta_K^2 = \frac{h_K^2}{p_K^2}\,\big\| f - \nabla\times(\mu^{-1}\nabla\times E_{hp}) + \omega^2\varepsilon E_{hp} \big\|^2_{L^2(K)}
+ \frac{h_K^2}{p_K^2}\,\ell^2\big\| \nabla\cdot(f + \omega^2\varepsilon E_{hp}) \big\|^2_{L^2(K)}
+ \sum_{F\subset\partial K}\frac{h_F}{2p_F}\Big( \big\| [\![\mathbf n\times\mu^{-1}\nabla\times E_{hp}]\!] \big\|^2_{L^2(F)}
+ \ell^2\big\| [\![\mathbf n\cdot(f+\omega^2\varepsilon E_{hp})]\!] \big\|^2_{L^2(F)} \Big).
$$

Terms: (1) element residual, (2) divergence residual (Gauss law, captures the gradient
part of the error), (3) jump of the tangential magnetic field across faces, (4) jump of
the normal displacement flux. The length scale $\ell$ of the Gauss-law terms is not in the
cited papers, which work with lengths of order one: $\nabla\cdot d$ carries one inverse length
more than the element residual, so in SI units (cells of nanometres, $k \sim 10^7$/m) the
unscaled terms (2) and (4) exceed (1) and (3) by $1/(kh)^2 \sim 10^{15}$ and $\eta$ measures only
the Gauss-law residual of the smallest cells (observed on the silver grating of
[validation.md](../validation.md#e-hp-adaptive-silver-grating-with-the-conical-solver-m15-f1)).
With $\ell = 1/k$ (the default, `EstimatorOptions::length_scale = 0`) the four terms are
measured in the same units, the wavelength being the length scale of the $H(\mathrm{curl})$
norm; problems with $k = 1$ are unchanged. Pass $\ell = c_0/\omega$ when the mass coefficient is
$\omega^2$ with SI tensors. The estimator is **reliable**
$\|E-E_{hp}\|_{H(\mathrm{curl})} \le C\,(\sum_K \eta_K^2)^{1/2}$ up to data oscillation
and higher-order terms (the Helmholtz-type problem is indefinite: reliability holds once
the mesh resolves the wavelength). In $hp$ the constants depend on $p$; this is accepted.

Complex coefficients: all norms use $|\cdot|^2 = (\cdot)\overline{(\cdot)}$; PML cells use
the stretched $\tilde\varepsilon, \tilde\mu$ in the residual so the estimator is consistent
with the equation actually solved there.

### Implementation (`adaptivity::residual_estimate`)

The estimator works on the same per-cell forms as the assembler
(`assembly::MaxwellForm`: $\mu^{-1}$, $\varepsilon$, $f$, $g$) and the mass coefficient
$k^2$ ($k_0^2$ for `physics::Scattering`, whose `estimate()` wraps the call). With the
curl source $g$ of the scattered-field formulation the strong equation reads
$\nabla\times(\mu^{-1}\nabla\times E) - k^2\varepsilon E = f + \nabla\times g$, so the
implementation uses

$$
w = \mu^{-1}\nabla\times E_{hp} - g, \qquad d = f + k^2\varepsilon E_{hp}, \qquad
R_K = d - \nabla\times w,
$$

and the jumps $[\![\mathbf n\times w]\!]$, $[\![\mathbf n\cdot d]\!]$ (both $w$ and $d$ are
continuous across interfaces for the exact solution, including the jump of $g$ and $f$ at
the scatterer boundary).

- **Derivatives inside a cell.** The Nédélec basis provides values and curls only. The
  curl of $w$ and the divergence of $d$ are formed from central differences of $w(\xi)$,
  $d(\xi)$ in *reference* coordinates (step `difference_step`, default $10^{-4}$) and the
  chain rule $\partial_{x_j} = \sum_k (J^{-T})_{jk}\,\partial_{\xi_k}$ with $J^{-T}$ at the
  quadrature point. On affine cells with constant coefficients both fields are polynomials
  in $\xi$, so the differences are exact up to rounding ($\sim 10^{-12}$ relative); on
  curved or PML cells the $O(\delta^2)$ truncation error is far below the estimator's own
  constants. The product rule for variable tensors (PML) is thereby included automatically.
- **Facets.** Interior facets only. Each facet is parametrised from its first cell, the
  physical quadrature point is pulled back into the second cell with the cell geometry
  (`to_reference`, Newton on curved cells), and both sides are sampled with their own
  forms. $h_F$ is the edge length (2D) or the longest edge of the face (3D),
  $p_F = \max(p_{K_0}, p_{K_1})$; each adjacent cell receives $h_F/(2p_F)$ times the jump
  integral. In 2D the tangential jump of the scalar curl is $|[\![w]\!]|$.
- **Hanging facets** (locally refined meshes): the jump is integrated over each child
  facet against the cell of the parent facet; the parent facet itself carries no term.
- **Boundary facets.** PEC and prescribed tangential traces are imposed exactly and carry no
  term; PMC (natural) facets are not accounted for. **Bloch facets** (`periodic` argument,
  passed by `Scattering::estimate` and `ConicalScattering::estimate`): the jump
  $[\![n\times w]\!] = w_\text{slave} - e^{ik\cdot a}\,w_\text{master}$ (and the flux jump) is
  integrated on each slave facet against the master cells under its shifted quadrature points
  (`assembly::PeriodicLocator`, non-matching faces included) and credited to the slave cell and
  to each master cell met, so cells at the periodic faces are marked like interior cells.
- **Quadrature** degree $2p + $ `extra_order` (+2 on curved cells), or the cell's own
  `quadrature_order` (PML cells); facet rules of the same degree.
- The Gauss-law terms can be switched off (`divergence_terms = false`), e.g. for
  magnetostatic-like problems.

`Estimate` returns $\eta_K$ per cell and the four squared contributions (`parts`), the
global $\eta$ (`total()`) and the worst cell (`argmax()`).

### Bodies of revolution (`adaptivity::axisymmetric_residual_estimate`)

The same estimator for the equation of one azimuthal order on the meridian mesh, with the
cylindrical curl and divergence and the weight $r$ in every integral; formulas and
verification in [axisymmetric.md](axisymmetric.md#hp-adaptivity-on-the-meridian-plane).

### Conical incidence (`adaptivity::conical_residual_estimate`)

The estimator of the 2.5D problem of [maxwell.md](maxwell.md#conical-incidence-and-the-e_z-polarisation),
$E(x, y)e^{i\beta z}$ with the in-plane field on the Nédélec map and $v = -iE_z$ on the H1 map.
It is the residual of the three-dimensional equation with $\partial_z = i\beta$, in Cartesian
components: $w = \mu^{-1}\nabla\times E_{hp}$ with
$\nabla\times E = \big(i(\partial_yv - \beta E_y),\ -i(\partial_xv - \beta E_x),\ \partial_xE_y - \partial_yE_x\big)$,
$d = f + k_0^2\varepsilon E_{hp}$, $R_K = d - \nabla\times w$ with
$(\nabla\times w)_x = \partial_yw_z - i\beta w_y$, $(\nabla\times w)_y = i\beta w_x - \partial_xw_z$,
$\nabla\cdot d = \partial_xd_x + \partial_yd_y + i\beta d_z$, and on a facet with the in-plane normal
$n$ the jumps $|[n\times w]|^2 = |[w_z]|^2 + |[n_xw_y - n_yw_x]|^2$ and $[n\cdot d]$. The
$z$-component of the tangential jump is the jump of the in-plane curl (the $H_z$ polarisation),
the in-plane components contain $\partial_n v$ (the normal derivative of $E_z$) and the
$\beta E_t$ coupling; the normal-flux jump is that of $\varepsilon E_t$. No $r$ weights, the same
$h/p$ factors as `residual_estimate`, PEC and Bloch facets carry no term. At $\beta = 0$ the
indicators are those of the in-plane and the $E_z$ block side by side, so the same marking and
hp decision apply to all three polarisations. Verified in `tests/unit/adaptivity/test_conical_estimator.cpp`
(zero for an exact gradient mode on conforming and one-irregular meshes, localisation) and by
the convergence test `conical_hp_corner` (re-entrant PEC corner at $\beta = 1.3$: effectivity
3.3–5.2, error $\sim \exp(-0.26\,N^{1/3})$, algebraic slope $-2.0$, the corner h-refined);
`physics::ConicalScattering::estimate` and `error` wrap it with the problem's forms.

## Goal-oriented estimation (dual-weighted residual)

Scatterometry and metasurface design need accuracy in a *functional* $Q(E)$ (a Fourier
coefficient, a flux, a Purcell factor), not in the energy norm. The DWR method solves the
adjoint problem $(S-\omega^2 M)^{\!*} z = Q'$ and weights the primal residuals with the
adjoint solution:

$$
Q(E) - Q(E_{hp}) \approx \sum_K \big( R_K(E_{hp}), z - z_{hp} \big)_K + \ldots
$$

### Conical solver (`physics::conical_dwr_estimate`)

The same construction for the coupled system of
[maxwell.md](maxwell.md#conical-incidence-and-the-e_z-polarisation): a functional is a pair
$(q_e, q_v)$ on the two DoF maps, $Q(E_h) = q_e^\top e + q_v^\top v$, returned by a
`ConicalFunctional` for any pair of maps of the mesh (`conical_point_functional` for
$E(x)\cdot w$ of the physical field, `conical_order_functional` for the vector amplitude
$A_m\cdot e$ of a diffraction order along a polarisation vector $e$, the same Gauss–Legendre
line rule as `conical_fourier_coefficients`; both carry the factor $i$ of $E_z = iv$ in $q_v$).
The adjoint of the enriched conical system (orders $+1$ on both maps, the PEC facets and the
hanging / Bloch constraints of the primal problem, conjugated prolongation for the test space)
gives the weight $z - I_pz$ block by block, and `adaptivity::conical_weighted_residual`
pairs the Cartesian residual with the physical test vector $W = (W_x, W_y, -i\,w)$ of the
scaled test function (the pairing of the conical forms, no conjugation):
$r_K(W) = \int_K R_K\cdot W + \tfrac12\sum_F\int_F (n\times[\![w_h]\!])\cdot W$ with the in-plane
normal, boundary facets contributing their one-sided term, so that
$\sum_K r_K(W) = \ell(W) - a(E_h, W)$ for every tangentially continuous $W$ (checked to
$10^{-8}$ in `test_conical_goal.cpp` on an enriched space with random coefficients). For the
efficiency $R_m = c\,|A_m|^2$ of an order the linearised goal is $Q(E) = A_m(E)\cdot
\overline{A_{m,h}}/|A_{m,h}|$, so that $\Delta R_m \approx 2R_m\,\mathrm{Re}(\Delta Q)/|A_{m,h}|$;
the Python generator `hpfem.adaptive_solve` re-evaluates the direction every step. The
convergence test `conical_goal_oriented` (point value of the singular gradient mode of the
conical corner, h-refinement at $p = 2$) finds effectivities 0.6–1.0 in the resolved regime
(below 3 k DoFs the signed goal error crosses zero on some meshes) and a goal error 35 times
smaller than the energy-driven loop at 8 k DoFs.

### Implementation (`physics::dwr_estimate`)

- **Functionals** are vectors $q$ on a DoF map with $Q(E_h) = q^\top e_h$ (no
  conjugation, like the forms): `assembly::point_functional` builds them for point-sampled
  functionals $Q(E) = \sum_j E(x_j)\cdot w_j + (\nabla\times E)(x_j)\cdot v_j$, which
  cover point values (`point_value_functional`), Fourier coefficients of diffraction orders
  (`fourier_coefficient_functional`, Gauss–Legendre along the line) and far-field patterns.
  A `physics::Functional` is a callable returning $q$ for any DoF map of the mesh, because
  the adjoint lives on the enriched space.
- **Adjoint.** The orders of all cells are raised by one (`NedelecDofMap` with $p_K + 1$),
  the same forms are assembled (`Scattering::form_of_cell`, so PML and materials carry
  over), and the adjoint is solved in the *test* space of the constrained primal problem:
  with the prolongation $P$ of the hanging-node and Bloch constraints the primal reduced
  system is $P^H A P$, its test functions are $\bar P v_r$, and the adjoint reads
  $(P^\top A \bar P)\, z_r = P^\top q$, $z = \bar P z_r$ (for real $P$ this is the
  symmetric system $P^\top A P$). Dirichlet facets of the primal problem (PEC and
  prescribed traces) get homogeneous data on the free DoFs.
- **Weight and residual.** $w = z - I_p z$ with the hierarchical interpolation $I_p$
  (`assembly::interpolate`, prolongated back to the enriched map), and the cell
  contributions are the signed weighted residuals of `adaptivity::weighted_residual`,

  $$
  r_K(w) = \int_K (d - \nabla\times W)\cdot w
  + \tfrac12\sum_{F\subset\partial K\setminus\partial\Omega}\int_F \big(n\times[\![W]\!]\big)\cdot w
  + \sum_{F\subset\partial K\cap\partial\Omega}\int_F (n\times W)\cdot w ,
  $$

  with $W = \mu^{-1}\nabla\times E_h - g$, $d = f + k^2\varepsilon E_h$ as in the
  residual estimator (hanging child facets against the cell of their parent). The
  boundary term is the residual of a natural condition; it vanishes where $w$ has no
  tangential trace (Dirichlet facets). The identity $\sum_K r_K(w) = \ell(w) - a(E_h, w)$
  holds for every tangentially continuous $w$ and is unit-tested against the algebraic
  residual $w^\top(b_{p+1} - A_{p+1} P_{p\to p+1} e_h)$ on hanging meshes in 2D and 3D.
- `GoalEstimate` returns the signed estimate $\sum_K r_K$ of $Q(E) - Q(E_h)$, the
  contributions, the indicators $|r_K|$ for marking and $Q(E_h)$.

## Dual formulation and guaranteed bounds (`adaptivity/hypercircle.hpp`)

The residual estimator above is reliable only up to an unknown constant. A bound *without*
constants comes from the hypercircle (Prager–Synge) argument, which needs a second,
*dual* approximation of the magnetic field. Write the primal problem as

$$
\nabla\times(\mu^{-1}\nabla\times E) - k^2\varepsilon E = f + \nabla\times g
$$

and introduce the dual unknown $\sigma = \mu^{-1}\nabla\times E - g$ (the scaled magnetic
field: $\sigma = i\omega H$ for $g = 0$). It satisfies $\nabla\times\sigma = f + k^2\varepsilon E$,
so eliminating $E = (k^2\varepsilon)^{-1}(\nabla\times\sigma - f)$ gives the **dual
curl–curl problem**

$$
\nabla\times\big((k^2\varepsilon)^{-1}\nabla\times\sigma\big) - \mu\,\sigma
= \nabla\times\big((k^2\varepsilon)^{-1} f\big) + \mu\,g ,
$$

with the roles of the materials swapped and the boundary conditions exchanged: where $E$ is
PEC (essential) the dual condition is natural, where $E$ is PMC the dual field has the
essential condition $n\times\sigma = 0$. In 3D $\sigma$ lives in the same Nédélec space as
$E$; in 2D the curl of a vector is a scalar, so $\sigma \in H^1$ with
$\nabla\times\sigma = (\partial_y\sigma, -\partial_x\sigma)$ and the dual problem is a
scalar reaction–diffusion problem with the rotated tensor $R^T(k^2\varepsilon)^{-1}R$
(`assembly::ScalarForm::diffusion_tensor`) and the source $(R^T(k^2\varepsilon)^{-1} f,
\nabla\tau)$ (`gradient_source`). `adaptivity::dual_form` performs this transformation per
cell and `dual_solution` solves the dual Galerkin problem on a `DofMap` (2D) or a
`NedelecDofMap` (3D).

For *any* pair of fields $E_h \in H(\mathrm{curl})$ with the primal and $\sigma_h$ with the
dual essential conditions, the **constitutive-relation error**

$$
\eta_K^2 = \big\|\sigma_h - (\mu^{-1}\nabla\times E_h - g)\big\|^2_{\mu,K}
+ \big\|(k^2\varepsilon)^{-1}(\nabla\times\sigma_h - f) - E_h\big\|^2_{-k^2\varepsilon,K}
$$

is computable without any constant (`adaptivity::hypercircle_estimate`, weighted norms
$|w^H T w|$ with $T = \mu$ and $T = -k^2\varepsilon$). For the **coercive** problem — real
symmetric positive $\mu$, $\varepsilon$ and $k^2 < 0$, i.e. $\nabla\times\mu^{-1}\nabla\times E
+ \kappa^2\varepsilon E = f$ — integration by parts of the error $e = E - E_h$ against the
exact relations gives

$$
\|e\|_a^2 = (\sigma_h - \mu^{-1}\nabla\times E_h + g,\ \nabla\times e)
+ (f - \nabla\times\sigma_h + k^2\varepsilon E_h,\ e)
\le \Big(\sum_K \eta_K^2\Big)^{1/2} \|e\|_a ,
\qquad
\|e\|_a^2 = \|\mu^{-1/2}\nabla\times e\|^2 - k^2\|\varepsilon^{1/2} e\|^2 ,
$$

so $\|E - E_h\|_a \le \eta$ is a **guaranteed upper bound** of the energy error, for every
mesh and every $\sigma_h$; the Galerkin dual solution gives the sharpest one, with
effectivity at most $1 + \|\sigma - \sigma_h\|_b / \|E - E_h\|_a$ in the dual energy norm.
For the indefinite time-harmonic problem ($k^2 = k_0^2 > 0$, lossy or PML materials) the
same quantity is the constitutive-relation estimator of Ladevèze type: it still vanishes for
the exact pair and converges at the rate of the error, but the inequality holds only up to the
inf-sup constant of the problem.

**Verification** (`tests/convergence/hypercircle_bound.cpp`,
`tests/unit/adaptivity/test_hypercircle.cpp`): for $\nabla\times\nabla\times E + E = f$ on
the unit square with PEC walls and $E = (\sin\pi y, \sin\pi x)$ the bound holds on every mesh
for $p = 1, 2, 3$ and $h = 1/4 \ldots 1/16$, $\eta$ converges with the rate $p$ of the energy
error, and the effectivity index is $4.5$ / $3.9$ / $4.4$ with the dual space of the same
order (here $\sigma = \nabla\times E$ is one derivative rougher than $E$, so the dual error is
about $\pi$ times the primal one) and $1.00 \ldots 1.10$ with the dual order $p + 1$. On the
cube with $E = (\sin\pi y\sin\pi z, \sin\pi z\sin\pi x, \sin\pi x\sin\pi y)$ the bound holds
for $p = 1, 2$ on $3^3$ cells; a quadratic solution in $\mathrm{ND}_3$ with its dual in
$P_3$ gives $\eta < 10^{-9}$.

## Verification

- Effectivity index $\theta = \eta / \|E - E_{hp}\|$ on problems with analytic
  solutions (cavity mode, Mie): must stay bounded (0.5…5) across refinement.
- The estimator must localise: on the L-shaped corner the largest $\eta_K$ must sit at
  the corner for every step.
- The DWR estimate must track the true goal error (effectivity bounded) and goal-driven
  refinement must beat energy-driven refinement for the goal.
- The hypercircle estimate must bound the energy error from above on every mesh of the
  coercive test problem and converge at the rate of the error.

### Results (`tests/convergence/estimator_effectivity.cpp`, `tests/unit/adaptivity/`)

- A quadratic field in $\mathrm{ND}_3$ (non-vanishing $\nabla\times\nabla\times E$,
  non-trivial $f$) solved on $\mathrm{ND}_3$ is reproduced exactly; all four residual parts
  vanish to rounding ($\eta < 10^{-7}$ at $\|f\| \sim k^2 \approx 9$) in 2D and 3D. A
  perturbed edge DoF raises the indicator only in the cells of that edge (element terms)
  and their facet neighbours (jump terms), with the maximum in a cell of the edge.
- Oblique plane wave on the unit square / cube with $k_0 = 3$ and the exact tangential
  trace prescribed: $\eta$ converges with the same rate $p$ as the $H(\mathrm{curl})$ error
  and the effectivity index settles between 6 and 14 (2D: 9.3 / 11.1 / 11.2 for
  $p = 1, 2, 3$ on $h = 1/16$; 3D: 13.5 / 11.5 for $p = 1, 2$ on $h = 1/8$). The
  constant is larger than the 0.5…5 of the H1 textbook case because the $h/p$ weights
  carry no interpolation constants; it is stable, which is what adaptivity needs.
- **Goal-oriented** (`tests/unit/physics/test_goal_oriented.cpp`,
  `tests/convergence/goal_oriented.cpp`): for the point value of a plane-wave solution
  the DWR estimate has effectivity 0.3…3 on $4\times4$ and $8\times8$ meshes with $p = 1, 2$,
  and the right sign beyond the coarsest mesh; on the L-shape with the goal
  $E_x(-0.55, 0.45)$ the goal-driven h-adaptive loop ($p = 2$, Dörfler 0.5 on $|r_K|$)
  reaches a goal error of $1.5\cdot10^{-5}$ at 6 800 DoFs with effectivity 0.5…0.9 in every
  step, while the energy-driven loop sits at $1.3\cdot10^{-3}$ at 6 200 DoFs: the adjoint
  weight concentrates the refinement between the corner and the evaluation point.

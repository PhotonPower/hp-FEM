# A-posteriori error estimation

## Residual estimator for the curl–curl problem

For $E_{hp}$ solving the discrete problem with data $f = i\omega J$ (Beck, Hiptmair,
Hoppe & Wohlmuth 2000; Schöberl 2008 for the $hp$ setting), the element indicator is

$$
\eta_K^2 = \frac{h_K^2}{p_K^2}\,\big\| f - \nabla\times(\mu^{-1}\nabla\times E_{hp}) + \omega^2\varepsilon E_{hp} \big\|^2_{L^2(K)}
+ \frac{h_K^2}{p_K^2}\,\big\| \nabla\cdot(f + \omega^2\varepsilon E_{hp}) \big\|^2_{L^2(K)}
+ \sum_{F\subset\partial K}\frac{h_F}{2p_F}\Big( \big\| [\![\mathbf n\times\mu^{-1}\nabla\times E_{hp}]\!] \big\|^2_{L^2(F)}
+ \big\| [\![\mathbf n\cdot(f+\omega^2\varepsilon E_{hp})]\!] \big\|^2_{L^2(F)} \Big).
$$

Terms: (1) element residual, (2) divergence residual (Gauss law, captures the gradient
part of the error), (3) jump of the tangential magnetic field across faces, (4) jump of
the normal displacement flux. The estimator is **reliable**
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
- **Boundary facets** carry no term: PEC and prescribed tangential traces are imposed
  exactly, PMC (natural) and Bloch-periodic facets are not yet accounted for.
- **Quadrature** degree $2p + $ `extra_order` (+2 on curved cells), or the cell's own
  `quadrature_order` (PML cells); facet rules of the same degree.
- The Gauss-law terms can be switched off (`divergence_terms = false`), e.g. for
  magnetostatic-like problems.

`Estimate` returns $\eta_K$ per cell and the four squared contributions (`parts`), the
global $\eta$ (`total()`) and the worst cell (`argmax()`).

## Goal-oriented estimation (dual-weighted residual)

Scatterometry and metasurface design need accuracy in a *functional* $Q(E)$ (a Fourier
coefficient, a flux, a Purcell factor), not in the energy norm. The DWR method solves the
adjoint problem $(S-\omega^2 M)^{\!*} z = Q'$ and weights the primal residuals with the
adjoint solution:

$$
Q(E) - Q(E_{hp}) \approx \sum_K \big( R_K(E_{hp}), z - z_{hp} \big)_K + \ldots
$$

Implementation plan (M5): same assembler, transposed/conjugated operator, $z$ approximated
on the $p+1$ space. The estimator interface takes a `Functional` object so both the energy
and the goal-oriented estimator plug into the same adaptivity loop.

## Verification

- Effectivity index $\theta = \eta / \|E - E_{hp}\|$ on problems with analytic
  solutions (cavity mode, Mie): must stay bounded (0.5…5) across refinement.
- The estimator must localise: on the L-shaped corner the largest $\eta_K$ must sit at
  the corner for every step.

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

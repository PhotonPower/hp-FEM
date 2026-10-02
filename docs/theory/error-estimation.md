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

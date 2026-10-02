# Perfectly matched layers as complex coordinate stretching

## Idea

Outside the computational domain $\Omega_{\mathrm{int}}$ the (scattered) field is outgoing.
Analytically continue the coordinate normal to the boundary, $\xi \mapsto \tilde\xi$,
into the complex plane:

$$
\tilde\xi(\xi) = \xi + \frac{i}{\omega}\int_0^{\xi}\sigma(s)\,ds, \qquad \sigma \ge 0,
$$

so that an outgoing wave $e^{ik\xi}$ becomes $e^{ik\xi} e^{-\frac{k}{\omega}\int\sigma}$:
exponentially decaying, while the transformation is exact (no reflection) in the
continuous setting. (With the $e^{-i\omega t}$ convention the sign of the imaginary part is
**plus**; getting this wrong produces exponential growth — test it.)

## In the FEM

Rather than modifying the equations, hpfem applies the stretching as a change of
variables in the weak form, which is equivalent to replacing the material tensors in the
PML cells by

$$
\tilde\varepsilon = \det(\Lambda)\,\Lambda^{-1}\varepsilon\Lambda^{-T}, \qquad
\tilde\mu = \det(\Lambda)\,\Lambda^{-1}\mu\Lambda^{-T}, \qquad
\Lambda = \operatorname{diag}(s_x, s_y, s_z),\ s_\xi = 1 + \frac{i\sigma_\xi(\xi)}{\omega}.
$$

The PML is therefore just an anisotropic, complex, spatially varying material — the
Nédélec assembly handles it without special code paths. Corners and edges of the box
get products of the one-dimensional stretches.

## Profile and adaptivity

- Polynomial profile $\sigma(\xi) = \sigma_{\max}(\xi/d)^m$, $m = 2\ldots3$, over thickness
  $d$ (typically $\lambda/2 \ldots \lambda$ in the surrounding medium).
- $\sigma_{\max}$ from the target theoretical reflection $R_0$:
  $\sigma_{\max} = -\tfrac{(m+1)\,c}{2\,d\,n}\ln R_0$.
- **Adaptive PML** (M4): choose $d$ and the number of layers from the local wavelength
  $\lambda/n$ and the mesh size so that the stretched field stays resolved; because
  $\tilde\varepsilon$ is non-polynomial, PML cells use higher quadrature order (≥ 2p+4).
- Alternative for layered/periodic backgrounds: a **pole-condition / radial PML** in
  $\xi$ only, keeping Bloch periodicity in the tangential directions.
- For resonance problems the stretched operator is non-Hermitian; its complex
  eigenvalues $\omega$ give resonance frequency ($\operatorname{Re}\omega$) and Q-factor
  ($Q = \operatorname{Re}\omega / (2|\operatorname{Im}\omega|)$). Spurious PML modes are
  identified by the fraction of energy inside the PML.

## Verification

- Convergence test #3: plane wave at normal and 60° incidence into a PML-terminated
  homogeneous box, reflection $< 10^{-6}$ for $p \ge 3$.
- Resonance test: 2D dielectric disk whispering-gallery mode, Q vs. analytic
  (Hankel-function) solution.

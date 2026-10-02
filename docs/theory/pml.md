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

## Implementation (`pml/pml.hpp`)

`pml::PmlBox<Dim>` describes layers of given thickness on the sides of an axis-aligned
interior box (thickness 0 = no layer on that side, e.g. a periodic direction). It works
with the dimensionless profile $\hat\sigma = \sigma/\omega$, so $s_\xi = 1 + i\hat\sigma_\xi$
with the polynomial $\hat\sigma(d) = \hat\sigma_{\max}(d/\text{thickness})^m$ and

$$
\hat\sigma_{\max} = -\frac{(m+1)\ln R_0}{2\,k_0\,n\,\text{thickness}},
$$

the dimensionless form of the formula below ($c/\omega = 1/k_0$); $n$ is the real
refractive index of the medium the layer continues. The box provides, per point, the
stretch factors, the **stretched coordinate** $\tilde x_\xi = x_\xi + i\int\hat\sigma$ in
closed form (so that an outgoing analytic field evaluated at $\tilde x$ is the exact
solution inside the layer, which the convergence test uses), and the effective tensors of
an isotropic material, $\tilde\varepsilon = \varepsilon_r\det\Lambda\,\Lambda^{-2}$ and
$\tilde\mu^{-1} = \mu_r^{-1}\Lambda^2/\det\Lambda$ (in 2D the out-of-plane curl sees the
scalar $1/(\mu_r s_x s_y)$). The sign is checked by the unit tests: a wave $e^{ik\tilde x}$
leaving through a layer arrives at its far end with amplitude $\sqrt{R_0}$.

`PmlBox::recommended_thickness(k0, n, h)` adapts the thickness to the local wavelength
(half a wavelength by default) rounded up to whole cells of a structured mesh. Keep the
layer resolved: the stretched field varies like $e^{iks\xi}$, so $|k s| h$ must stay
moderate ($\lesssim 3$ for $p \ge 4$), which favours thick layers with mild profiles over
thin, strongly absorbing ones; see the verification section.

`physics::Scattering` takes an optional `PmlBox`; its per-cell forms then evaluate the
stretched tensors pointwise (identity inside the box, so no cell tagging is needed, but
cell facets should coincide with the box faces and PML cells use the raised quadrature
order of the incident-field terms, `extra_quadrature_order`).

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

- Convergence test #3 (`tests/convergence/pml_reflection.cpp`): plane wave at normal
  and 60° incidence into the unit box terminated by a PML in $+x$ with a PEC behind it,
  the exact field prescribed on the other sides. The exact solution of the
  anisotropic-material problem is $\Lambda\,\mathbf{E}^{inc}(\tilde x)$ — the incident
  wave at the stretched coordinate with the normal component scaled by the stretch (the
  material formulation solves for $\Lambda\mathbf{E}$, verified against the tensors by
  finite differences), so the error against it is discretisation error plus PML
  reflection. In the interior it converges with rate $p$ under h-refinement (2D at both
  angles, 3D at normal incidence; observed 2D rates 1.0 / 2.0) and under p-refinement at
  $h \approx \lambda/8$ it decreases exponentially to $5\cdot10^{-8}$ (normal) and
  $2\cdot10^{-7}$ (60°) at $p = 6$: no reflection floor above $10^{-6}$.

  Three lessons from setting this test up, all relevant to real simulations:

  1. **Resolve the complex wavenumber.** Inside the layer the field behaves like
     $e^{ik s\xi}$, so the mesh must resolve $|k s| h$, not $kh$: $|ks|h \lesssim 3$ at
     $p \ge 4$. A thick layer with a mild profile (here three wavelengths, $m = 2$,
     $\hat\sigma_{\max} \approx 4$) converges where half a wavelength with
     $\hat\sigma_{\max} \approx 20$ stalls at $10^{-4}$.
  2. **The far wall sees the one-way attenuated field.** At incidence angle $\theta$ the
     round-trip reflection is $R_0^{\cos\theta}$ and the PEC behind the layer receives the
     field attenuated by $R_0^{\cos\theta/2}$ only; a mismatch there is re-radiated into
     the interior. The test uses $R_0 = 10^{-20}$ (one-way $10^{-5}$ at 60°).
  3. **Avoid cavity resonances of the test domain.** With the exact field prescribed on
     the sides of the unit square, $k_0 = 2\pi$ coincides with the PEC-cavity eigenvalue
     $\pi^2(2^2 + 0^2)$; the discrete near-resonance (which moves with $p$) produced a
     plateau of $10^{-4}$ at $p = 5$ that disappears at $k_0 = 6$.
- Resonance test: 2D dielectric disk whispering-gallery mode, Q vs. analytic
  (Hankel-function) solution.

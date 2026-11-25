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
thin, strongly absorbing ones; `PmlBox::max_resolution(h, n)` evaluates $|k s| h$ at the far
end of the layers, `recommended_thickness(k0, n, h, profile, p)` returns the smallest
whole-cell thickness that keeps a given profile below `resolution_limit(p)`, and
`physics::Scattering` warns when a PML cell violates the limit (see "Oblique incidence").

## Oblique incidence

$R_0$ is the **round-trip field reflection at normal incidence**: the one-dimensional
attenuation $e^{-2k_0 n\int\hat\sigma}$ equals $R_0$ for a wave travelling along the layer
normal. A wave at the angle $\theta$ against the normal travels the layer with the normal
wavenumber $k\cos\theta$, so its round trip leaves $R_0^{\cos\theta}$ and the far wall (PEC
behind the layer) receives the one-way field $R_0^{\cos\theta/2}$, which it re-radiates into
the interior. For a reflectance $R = |r|^2$ computed with such a layer the error is bounded by

$$
|\Delta R| \lesssim 2\,|r|\,R_0^{\cos\theta/2},
$$

and doubling the thickness changes only the phase of this error, not its size — the
exponent is fixed by $R_0$, the thickness sets the steepness of the profile. The library
default $R_0 = 10^{-8}$ (and the $10^{-10}$ of the examples, adequate at normal incidence)
therefore leaves errors of $10^{-3}$ to $10^{-2}$ at 50° to 70°. Measured on a flat silicon
surface ($\lambda = 405$ nm, in-plane $E$, $p = 5$, Fresnel reference; test report of
2026-10-05):

| material, $\theta$ | $R_0 = 10^{-10}$ | $10^{-16}$ | $10^{-24}$ | bound at $10^{-10}$ / $10^{-16}$ |
|---|---|---|---|---|
| Si, 0° | $5.6\cdot10^{-6}$ | – | – | $1.4\cdot10^{-5}$ / – |
| Si, 20° | $8.6\cdot10^{-6}$ | – | – | $2.7\cdot10^{-5}$ / – |
| Si, 35° | $4.8\cdot10^{-5}$ | – | – | $1.0\cdot10^{-4}$ / – |
| Si, 50° | $2.8\cdot10^{-4}$ | $1.3\cdot10^{-6}$ | – | $6.8\cdot10^{-4}$ / $8\cdot10^{-6}$ |
| Si, 60° | $2.6\cdot10^{-3}$ | $7.9\cdot10^{-5}$ | – | $3.0\cdot10^{-3}$ / $9.4\cdot10^{-5}$ |
| Si, 70° | $3.6\cdot10^{-3}$ | $4.1\cdot10^{-4}$ | $1.7\cdot10^{-5}$ | $1.2\cdot10^{-2}$ / $1.1\cdot10^{-3}$ |

`PmlProfile::for_angle(theta_max, target, r_amplitude = 1, order = 2)` inverts the bound:
$R_0 = (\text{target}/2|r|)^{2/\cos\theta_{\max}}$ for the largest angle that occurs (for the
target $10^{-4}$ and $|r| = 1$: $4\cdot10^{-14}$ at 50°, $6\cdot10^{-18}$ at 60°,
$7\cdot10^{-26}$ at 70°, $1.5\cdot10^{-28}$ at 72°, the edge of a microscope pupil of NA 0.95).
Such steep profiles must still be resolved: with $|s| = \sqrt{1+\hat\sigma_{\max}^2}$ at the
far end, $|k s| h \le 3$ for $p \ge 4$ (`PmlBox::resolution_limit`, $0.75p$ below). A PML in
air of three wavelengths (1215 to 1480 nm at 405 nm, cells of 37 nm) gives 1.5 to 2.8 for
$R_0 = 10^{-16}$; the same layer meshed with silicon ($n = 5.4$) gives 12 to 38 and converges
nowhere, which is what happens when a substrate of high index is continued into a layer
designed with `background_index = 1`. `Scattering` warns about it; the recommended
structure for a substrate is a `LayerStack` background with a PEC wall several attenuation
lengths deep and no PML there (identical results for substrate depths of 1776 nm and
2368 nm in the report), or `recommended_thickness(k0, n, h, profile, p)` for the layer.

**Verification** (`tests/convergence/flat_surface_fresnel.cpp`): flat silicon at 50° and 70°
and flat silver at 50° ($\lambda = 405$ nm, in-plane $E$, Bloch cell of 400 nm, substrate as
scatterer down to a PEC wall, the specular order from the scattered field on a line in the
air) against the Fresnel reflectances 0.31366825, 0.09595201 and 0.95156272 with the profile
of `for_angle`: errors $4.6\cdot10^{-6}$ ($R_0 = 1.9\cdot10^{-16}$, $p = 5$),
$2.6\cdot10^{-5}$ ($6.7\cdot10^{-23}$, three wavelengths of PML) and $2.1\cdot10^{-5}$
($3.5\cdot10^{-17}$, $p = 4$), all with $|k s| h \le 2$; the library default $R_0 = 10^{-8}$
at 70° gives $4.5\cdot10^{-3}$ on the same mesh.

`physics::Scattering` takes an optional `pml::Pml` (a `PmlBox`, or a `RadialPml` below); its
per-cell forms then evaluate the
stretched tensors pointwise (identity inside the box, so no cell tagging is needed, but
cell facets should coincide with the box faces and PML cells use the raised quadrature
order of the incident-field terms, `extra_quadrature_order`).

## Radial layers (`pml::RadialPml`)

A layer of radial shape stretches the distance $r$ from a centre $c$ (2D circle, 3D sphere) or
from an axis (3D cylinder) beyond the inner radius $r_0$ with the profile above,

$$
\tilde r = r + i\int_{r_0}^{r}\hat\sigma(s)\,ds, \qquad
\tilde x = c + \frac{\tilde r}{r}\,(x - c)
$$

(for the cylinder only the part of $x - c$ perpendicular to the axis is scaled). Its Jacobian is

$$
J = \frac{\partial\tilde x}{\partial x} = s_r\,\hat r\hat r^T + \frac{\tilde r}{r}\,(P - \hat r\hat r^T)
+ (I - P), \qquad s_r = 1 + i\hat\sigma(r),
$$

with $P$ the projector onto the stretched plane (circle, sphere: $P = I$; cylinder:
$P = I - \hat a\hat a^T$). The radial direction is absorbed with the stretch $s_r$ as in the
box, the tangential directions with $\tilde r / r$, which carries the accumulated stretch and
keeps the transformation a smooth map (it is what makes the radial layer exact for outgoing
cylindrical and spherical waves). $J$ is symmetric but not diagonal in Cartesian components,
and the materials become the full tensors

$$
\tilde\varepsilon = \det J\,J^{-1}\varepsilon J^{-T}, \qquad
\tilde\mu^{-1} = \frac{J\mu^{-1}J^{T}}{\det J}
\quad (\text{2D: } \tilde\mu^{-1}_{zz} = 1/(\mu\det J)),
$$

the general form of which the box formulas are the diagonal case. $\hat\sigma_{\max}$ follows
the same design rule, so a radial wave leaves the far end with the one-way amplitude
$\sqrt{R_0}$. Beyond $r_0 + d$ the final stretch $\hat\sigma_{\max}$ continues, so a square
or cubic mesh around a circular or spherical layer is allowed (its corners are absorbed
further). The cylinder takes, in addition, slab layers below and above an interval along the
axis; at the edges both stretches multiply as at the corners of a box.

The point of the radial shape is its invariance: $J(Ax) = R\,J(x)\,R^T$ for every rotation
$A x = c + R(x - c)$ about the centre (axis), so a sector of a structure with cyclic symmetry
$C_n$ of **any** order carries the layer, which the box does only for $C_2$ and $C_4$ about its
centre and $C_3$ about a diagonal of a cube (`physics::CyclicSymmetricScattering` and
`MirrorSymmetricScattering` check that the layer on the sector sides and mirror planes
continues into the images, `Pml::is_invariant`). A circular layer wants circles as mesh lines:
`mesh::polar_disc(sides, n, radii, tags)` meshes concentric annuli by rings (ring $j$ with
$j\cdot$`sides` vertices), curves the edges on every circle of `radii` and is invariant under
the rotations by $2\pi/$`sides`.

The problem setups hold a `pml::Pml<Dim>`, which is either a `PmlBox` or a `RadialPml`
(ADR-0005 amendment); assigning either works. The stability condition of anisotropic media
(ADR-0015 §2) is checked along a fan of stretching directions for crystals inside a radial
layer (`backward_wave_along`), since the radial direction sweeps the whole plane.

**Verification.** The Jacobian against finite differences of the closed-form stretched point
for all three shapes and the tensors against the box where they coincide
(`tests/unit/pml/test_pml.cpp`); the Mie cylinder under oblique incidence in a circular layer
on `polar_disc`, full disc and $C_6$ sector (`tests/convergence/mie_cylinder_radial_pml.cpp`):

| $p$ | DoF (full) | $\sigma_{\text{sca}}$ rel. error | sector vs full |
|---|---|---|---|
| 1 | 20 880 | $9.7\cdot10^{-2}$ | $4\cdot10^{-15}$ |
| 2 | 69 408 | $2.5\cdot10^{-3}$ | $2\cdot10^{-15}$ |
| 3 | 145 584 | $2.8\cdot10^{-4}$ | $2\cdot10^{-14}$ |

The error is that of the discretisation, not of the layer ($R_0 = 10^{-14}$ instead of
$10^{-10}$ changes the $p = 3$ value by $2\cdot10^{-6}$ relative). In 3D the Mie sphere on the
octant with a spherical layer in the box mesh (`tests/convergence/mie_sphere_mirror.cpp`) gives
a field error of $8.1\cdot10^{-2}$ at $p = 2$, the box layer $9.5\cdot10^{-2}$ on the same mesh.

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

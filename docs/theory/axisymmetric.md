# Axisymmetric (2.5D) problems

Bodies of revolution (VCSELs, micropillars, nanowires, round apertures) are computed on the
meridian plane with the azimuthal Fourier decomposition

$$
E(r, \varphi, z) = \sum_{m=-\infty}^{\infty} E_m(r, z)\,e^{im\varphi},
\qquad E_m = (E_r, E_\varphi, E_z)(r, z),
$$

so that every order $m$ is an independent two-dimensional problem (ADR-0010). The mesh is the
meridian domain with coordinates $x = r \ge 0$ and $y = z$; the axis $r = 0$ is a boundary of
the mesh with its own facet tag. The convention $e^{-i\omega t}$ of `maxwell.md` applies.

## Unknowns and spaces (`assembly/axisymmetric_forms.hpp`)

The meridian components $(E_r, E_z)$ are a 2D vector field in $H(\mathrm{curl})$ of the
$(r, z)$ plane and live in the Nédélec space; the azimuthal component is a scalar in $H^1$.
The code uses the **scaled azimuthal unknown**

$$
v = -i\,r\,E_\varphi, \qquad E_\varphi = \frac{i\,v}{r},
$$

in the H1 space of the same order. Two things follow. The gradient of a potential mode,
$\nabla(\psi e^{im\varphi}) = (\partial_r\psi,\ im\psi/r,\ \partial_z\psi)e^{im\varphi}$, becomes
$(\nabla_{rz}\psi,\ v = m\psi)$: exactly representable in the discrete spaces, so the
**order-$m$ discrete gradient** $K_m = [G;\ mI]$ ($G$ the 2D discrete gradient) spans the
kernel of the curl–curl matrix and the gauged eigensolvers remove it without spurious modes.
And the factor $-i$ turns the $\pm im$ couplings of the curl into real numbers: for lossless
media the matrices are real symmetric, otherwise complex symmetric.

With the cylindrical curl

$$
(\nabla\times E)_r = \frac{im E_z - \partial_z(rE_\varphi)}{r}, \qquad
(\nabla\times E)_\varphi = \partial_z E_r - \partial_r E_z, \qquad
(\nabla\times E)_z = \frac{\partial_r(rE_\varphi) - imE_r}{r},
$$

the volume element $r\,dr\,d\varphi\,dz$ and the $\varphi$ integration done (the factor
$2\pi$ is dropped), the bilinear forms of order $m$ with diagonal material tensors
$\mu^{-1} = \mathrm{diag}(\mu_r^{-1}, \mu_\varphi^{-1}, \mu_z^{-1})$ and $\varepsilon$ read

$$
a(E, V) = \int \Big[ \mu_r^{-1}\frac{(mE_z - \partial_z v)(mV_z - \partial_z w)}{r}
+ \mu_\varphi^{-1}(\partial_z E_r - \partial_r E_z)(\partial_z V_r - \partial_r V_z)\,r
+ \mu_z^{-1}\frac{(\partial_r v - mE_r)(\partial_r w - mV_r)}{r}\Big]\,dr\,dz ,
$$

$$
b(E, V) = \int \big[\varepsilon_r E_r V_r\,r + \varepsilon_\varphi\,v\,w / r
+ \varepsilon_z E_z V_z\,r\big]\,dr\,dz .
$$

The test functions are the complex conjugates of the trial basis (sesquilinear pairing
with $e^{-im\varphi}$), which is what makes the forms symmetric in $(E, v) \leftrightarrow
(V, w)$; $m \to -m$ gives the same spectrum. `assemble_axisymmetric` returns the stiffness
(the $a$ form) and the mass (the $b$ form) of the block vector $(e, v)$: the first
`nedelec.num_dofs()` entries are the Nédélec coefficients, the rest the H1 coefficients of
$v$. Per-cell diagonal tensors in $(r, \varphi, z)$ cover anisotropic media and the
cylindrical PML (stretched tensors of Teixeira–Chew, stage 2).

## The axis

The $1/r$ terms are singular on the axis unless their numerators vanish there. The
regularity of a field $E_m e^{im\varphi}$ at $r = 0$ requires

| order | conditions at $r = 0$ |
|---|---|
| $m = 0$ | $E_\varphi = 0$, $E_r = 0$; $E_z$ free |
| $m = \pm1$ | $E_z = 0$; $E_r \pm iE_\varphi = 0$ (the Cartesian components are regular) |
| $|m| \ge 2$ | all components zero |

Of these only $v = 0$ (all $m$; automatic for $r E_\varphi$ and imposed on the H1 DoFs of
the axis facets) and $E_z = 0$ for $m \ne 0$ (tangential Nédélec DoFs of the axis facets)
are Dirichlet data. The remaining conditions ($E_r$ on the axis, the $m = \pm1$ coupling)
involve the normal component of the edge elements and are not imposed; the quadrature of the
cells touching the axis (rule two degrees higher, Gauss points strictly inside) weights
them, which is the established body-of-revolution practice. The gauge potential $\psi$
vanishes on the axis for $m \ne 0$ ($v = m\psi$) and is free there for $m = 0$. PEC walls fix
the Nédélec trace and $v$ (`physics::AxisymmetricCavity` builds all index sets from the axis
tag and the PEC tags).

## Eigenmodes (`physics/axisymmetric.hpp`)

`AxisymmetricCavity` assembles the order-$m$ pencil, applies the constraints and calls
`solvers::gauged_curl_curl_eigenpairs` with $K_m$; it returns the lowest $k_0$ with the
meridian and azimuthal coefficients. The modes of the PEC cylinder (radius $a$, height $h$)
are the analytic reference,

$$
\mathrm{TM}_{mnp}:\ k^2 = \Big(\frac{j_{mn}}{a}\Big)^2 + \Big(\frac{p\pi}{h}\Big)^2,\ p \ge 0;
\qquad
\mathrm{TE}_{mnp}:\ k^2 = \Big(\frac{j'_{mn}}{a}\Big)^2 + \Big(\frac{p\pi}{h}\Big)^2,\ p \ge 1,
$$

with the zeros $j_{mn}$ of $J_m$ and $j'_{mn}$ of $J'_m$.

**Verification** (`tests/convergence/axisymmetric_cavity.cpp`,
`tests/unit/physics/test_axisymmetric.cpp`): on the cylinder $a = 1$, $h = 1.5$ the four
lowest eigenvalues of $m = 0, 1, 2$ converge with rate $2p$ under h-refinement
($p = 1$: $1.9 / 1.8 / 1.9$; $p = 2$: $3.8 / 3.8 / 3.9$ between $h = 1/8$ and $1/16$, maximum
relative errors $3\cdot10^{-5}$, $1.3\cdot10^{-5}$, $1.0\cdot10^{-5}$ at $p = 2$, $h = 1/16$)
and no eigenvalue appears below the first physical one. The unit tests check that $K_m\psi$
lies in the kernel of the stiffness matrix to rounding for every $m$, the symmetry and
positivity of the matrices, the decoupling of the blocks for $m = 0$, the $m$-dependent
axis conditions, and $\mathrm{TM}_{010}$ / $\mathrm{TE}_{111}$ on a coarse mesh.

## Resonances with the cylindrical PML (`physics::AxisymmetricResonance`)

Open resonators (micropillars, VCSELs, spheres) have quasi-normal modes with complex
$\omega$ (`maxwell.md`, Resonances). The PML of ADR-0005 carries over as a *material*: with
the stretch factors $s_r(r)$, $s_z(z)$ of a `pml::PmlBox<2>` (layers on the $r$-max, $z$-min
and $z$-max sides, none on the axis) and the stretched radius $\tilde r = r + i\int\hat\sigma_r$
the cylindrical PML of Teixeira and Chew is the diagonal tensor

$$
\Lambda = \mathrm{diag}\Big(\frac{s_\varphi s_z}{s_r},\ \frac{s_r s_z}{s_\varphi},\
\frac{s_r s_\varphi}{s_z}\Big), \qquad s_\varphi = \frac{\tilde r}{r},
\qquad \tilde\varepsilon = \varepsilon_r\Lambda, \quad \tilde\mu^{-1} = \mu_r^{-1}\Lambda^{-1},
$$

evaluated per quadrature point (`axisymmetric_pml_form`); the curl operator and the forms
above are untouched. Inside the box $\Lambda = I$. The order-$m$ pencil becomes complex
symmetric, and `AxisymmetricResonance` solves it with the complex gauged shift-invert
solver (`solvers::complex_eigenpairs_near_gauged`, the gradient $K_m$ restricted to the
free DoFs) around the target $k_0^2 = (\omega_{\text{target}}/c_0)^2$. Each mode carries
$\omega$, $\lambda_{\mathrm{res}}$, $Q = \mathrm{Re}\,\omega / (-2\,\mathrm{Im}\,\omega)$, the
Arnoldi residual and both coefficient vectors.

**Verification** (`tests/convergence/axisymmetric_sphere_resonance.cpp`,
`tests/unit/physics/test_axisymmetric_resonance.cpp`): the quasi-normal modes of a
dielectric sphere of radius $a$ and index $n$ are the zeros of the Mie denominators in the
complex size parameter $x = ka$; for $l = 1$ (the lowest modes of order $m = 1$) with the
Riccati–Bessel functions $\psi_1$, $\xi_1$,

$$
\mathrm{TE}_1:\ \psi_1(nx)\,\xi_1'(x) - n\,\xi_1(x)\,\psi_1'(nx) = 0, \qquad
\mathrm{TM}_1:\ n\,\psi_1(nx)\,\xi_1'(x) - \xi_1(x)\,\psi_1'(nx) = 0 ,
$$

solved by Newton in the test. For $n = 3$ the poles are $x_{\mathrm{TE}} = 0.98712 -
0.06058i$ and $x_{\mathrm{TM}} = 1.44174 - 0.17947i$. On the half-disc meridian mesh of
`square_with_disc` (curved interface, four cells per radius, PML of three radii beyond
$r, |z| = 3a$) the computed poles converge exponentially under p-refinement: relative
errors $2.9\cdot10^{-2}$, $2.4\cdot10^{-4}$, $3.5\cdot10^{-6}$, $1.8\cdot10^{-6}$ (TE) and
$3.7\cdot10^{-2}$, $2.0\cdot10^{-3}$, $2.3\cdot10^{-5}$, $2.0\cdot10^{-6}$ (TM) for
$p = 1 \ldots 4$, with Arnoldi residuals below $10^{-8}$. The unit test also checks the PML
tensors against the formula and that a layer on the axis side is rejected.

## Scattering (`physics::AxisymmetricScattering`)

The scattered-field formulation of `maxwell.md` carries over order by order: the incident
field is a solution in the background medium, its $m$-th Fourier component in the scaled
components $(E_r, v, E_z)$ drives the volume source $k_0^2(\varepsilon_r -
\varepsilon_{bg})E^{inc}_m$ in the cells whose material differs from the background (nowhere
inside the PML), and the operator $S_m - k_0^2 M_m$ with the cylindrical PML is solved for the
scattered field of that order. The source enters the forms through the same pairing as the
mass term, $\int (f_r V_r + f_z V_z)\,r + f_v\,w / r$ with $f_v = -i\,r f_\varphi$
(`AxisymmetricForm::source`). An $x$-polarised plane wave along the axis,
$E_0\hat x\,e^{ikz}$, has the orders $m = \pm1$ only:

$$
(E_r, E_\varphi, E_z)_{\pm1} = \tfrac{E_0}{2}\,(1,\ \pm i,\ 0)\,e^{ikz}, \qquad
(E_r, v, E_z)_{\pm1} = \tfrac{E_0}{2}\,(1,\ \pm r,\ 0)\,e^{ikz}
$$

(`axial_plane_wave`); the two orders are mirror images and carry the same power, so one
solve suffices. Orders do not mix in quadratic quantities: the power of the full field through
a surface of revolution is the sum over $m$ of

$$
P_m = \int_\Gamma 2\pi r\,\tfrac12\,\mathrm{Re}\big(E_m\times H_m^*\big)\cdot n\,ds, \qquad
H_m = \frac{\nabla\times E_m}{i\omega\mu},
$$

with the cylindrical curl of the mode (the azimuthal component is minus the 2D scalar curl
of the meridian field) over the meridian curve $\Gamma$ of `postprocess::Surface`
(`axisymmetric_poynting_flux`); the scattering cross-section is $\sigma = P^{sca} / I^{inc}$
with $I^{inc} = |E_0|^2 / (2Z_0)$ in vacuum.

**Verification** (`tests/convergence/axisymmetric_mie_sphere.cpp`,
`tests/unit/physics/test_axisymmetric_scattering.cpp`): for the dielectric sphere $n = 2$,
$ka = 1.5$ the Mie series (Bohren–Huffman $a_l$, $b_l$ with Riccati–Bessel functions by
downward recurrence, computed in the test) gives $\sigma / \pi a^2 = 4.2315$; the scattered
power of the order $m = 1$ through the sphere interface, doubled, converges to it under
p-refinement with relative errors $1.9\cdot10^{-1}$, $3.9\cdot10^{-3}$, $1.3\cdot10^{-3}$,
$3.9\cdot10^{-6}$ for $p = 1 \ldots 4$ — the 2.5D counterpart of convergence test #4. The unit
tests check the plane-wave orders, the Rayleigh limit of the Mie series, the equality of the
$m = \pm1$ powers, and that the discrete Poynting flux through the shell between the sphere
and the box $r, |z| < 2a$ vanishes up to the discretisation error.

## Dipoles on the axis (`axisymmetric_gaussian_dipole`)

A point emitter on the axis is a current moment $p$ [A m] at $(r, z) = (0, z_0)$. With the
total-field formulation (`AxisymmetricScatteringSetup::current`, the source
$f = i\omega\mu_0 J$ in every cell, the PML absorbing the emitted field) and the Gaussian
smearing of `maxwell.md` the orders are

$$
\text{axial } (p \parallel \hat z):\ m = 0,\quad (f_r, f_v, f_z) = i\omega\mu_0\,p\,g\,(0, 0, 1);
\qquad
\text{transverse } (p \parallel \hat x):\ m = \pm1,\quad
(f_r, f_v, f_z) = i\omega\mu_0\,\tfrac{p}{2}\,g\,(1, \pm r, 0),
$$

with $g = e^{-(r^2 + (z - z_0)^2)/2\sigma^2} / ((2\pi)^{3/2}\sigma^3)$. The emitted power is
the Poynting flux through any surface enclosing the source, and the Purcell factor its
ratio to the free-space value. For the smeared dipole in vacuum the total power is
*exactly* $P_0\,e^{-k^2\sigma^2}$ with the Larmor power $P_0 = Z_0 k_0^2 |p|^2 / (12\pi)$
(`dipole_vacuum_power`): the Gaussian form factor $e^{-k^2\sigma^2/2}$ multiplies the far
field in every direction, and in a lossless medium the emitted power equals the radiated
one. This gives an analytic reference without any limit $\sigma \to 0$.

**Verification** (`tests/convergence/axisymmetric_dipole.cpp`,
`tests/unit/physics/test_axisymmetric_dipole.cpp`): for $ka = 1.5$ and $\sigma = 0.08a$ the
axial ($m = 0$) and the transverse dipole ($m = \pm1$, doubled) converge to
$P_0 e^{-k^2\sigma^2}$ with relative errors $1.7\cdot10^{-1}$ / $3.9\cdot10^{-2}$,
$1.9\cdot10^{-3}$ / $4.8\cdot10^{-3}$, $2.0\cdot10^{-4}$ / $6.4\cdot10^{-5}$ and
$2.4\cdot10^{-5}$ / $2.2\cdot10^{-5}$ for $p = 1\ldots4$. The unit test places the axial
dipole at the centre of the sphere $n = 3$: the electric dipole couples to the
$\mathrm{TM}_1$ modes, and the Purcell factor peaks at $\mathrm{Re}\,x_{\mathrm{TM}}$ of the
Mie pole found by the resonance solver, which ties the three problem classes together.

## Far field (`axisymmetric_far_field`)

The near-to-far transform of `maxwell.md` (Stratton–Chu with the equivalent currents
$J = n\times H$, $M = -n\times E$ on a closed surface in the background medium,
$F = \frac{ik}{4\pi}[Z N_t - \hat r\times L]$ with $N = \int J e^{-ik\hat r\cdot x'}$,
$L = \int M e^{-ik\hat r\cdot x'}$) is applied to one order: on a surface of revolution the
azimuthal integration of $e^{im\varphi'}$ against the plane-wave phase
$e^{-ik\rho\sin\theta\cos(\varphi' - \varphi)}$ is analytic,

$$
\int_0^{2\pi} e^{im\varphi'}\,e^{-ik\rho\sin\theta\cos(\varphi'-\varphi)}\,d\varphi'
= 2\pi\,(-i)^m J_m(k\rho\sin\theta)\,e^{im\varphi},
$$

and the Cartesian components of a cylindrical vector bring the factors $\cos\varphi'$,
$\sin\varphi'$, i.e. the orders $m \pm 1$. Only the meridian curve is integrated numerically
(same quadrature as the flux), and the pattern of order $m$ is
$E \approx F(\theta)\,e^{im\varphi}\,e^{ikR}/R$ with the spherical components $F_\theta$,
$F_\varphi$ evaluated at $\varphi = 0$. The radiated power $\int |F|^2 d\Omega / (2Z)$ of the
order (trapezoidal rule over the sampled polar angles) must equal the Poynting flux of the
order through the same surface; the total pattern of a field with several orders is the sum
of the $F_m e^{im\varphi}$.

**Verification** (`tests/unit/physics/test_axisymmetric_farfield.cpp`,
`tests/convergence/axisymmetric_mie_sphere.cpp`): the axial dipole in vacuum radiates the
Larmor pattern, $|F_\theta| \propto \sin\theta$ to $2\cdot10^{-3}$ with $F_\varphi = 0$, and
its far-field power equals the near-field flux to $2\cdot10^{-3}$ (and the analytic
$P_0 e^{-k^2\sigma^2}$ to $5\cdot10^{-3}$) at $p = 3$; the transverse dipole radiates along
the axis and conserves power likewise; for the sphere scattering the cross-section from the
far-field power agrees with the flux and converges to the Mie value (the convergence test
prints both columns).

## Oblique incidence (`oblique_plane_wave`, `scatter_orders`)

A plane wave at the angle $\theta_i$ to the axis is not a single order: with the wave vector
$k(\sin\theta_i, 0, \cos\theta_i)$ and the polarisation $\hat p$ ($\hat y$ for s, $(\cos\theta_i,
0, -\sin\theta_i)$ for p, both relative to the plane of incidence $x$–$z$) the Jacobi–Anger
expansion of the transverse phase,

$$
e^{ik_\perp\rho\cos\varphi} = \sum_{n=-\infty}^{\infty} i^n J_n(k_\perp\rho)\,e^{in\varphi},
\qquad k_\perp = k\sin\theta_i,\quad a_n = i^n J_n(k_\perp\rho),
$$

spreads the field over all orders. The Cartesian polarisation enters the cylindrical components
through $\cos\varphi$ and $\sin\varphi$, which shift the index by one:

$$
\begin{aligned}
E_{\rho,m} &= \tfrac{p_x}{2}(a_{m-1} + a_{m+1}) + \tfrac{p_y}{2i}(a_{m-1} - a_{m+1}),\\
E_{\varphi,m} &= -\tfrac{p_x}{2i}(a_{m-1} - a_{m+1}) + \tfrac{p_y}{2}(a_{m-1} + a_{m+1}),\\
E_{z,m} &= p_z\,a_m,
\end{aligned}
\qquad\text{all times } E_0\,e^{ikz\cos\theta_i},
$$

returned in the scaled components $(E_r, v = -ir E_\varphi, E_z)$ by `oblique_plane_wave`. At
$\theta_i = 0$ only $a_0 = 1$ survives and the orders $m = \pm 1$ of `axial_plane_wave` remain.
The orders are *not* coupled by a body of revolution: the operator commutes with rotations
about the axis, so every $m$ is solved separately with its own incident component and the
response is the sum $\sum_m E_m e^{im\varphi}$. The series converges once $|m|$ exceeds
$k_\perp R$ for the radius $R$ of the scatterer ($J_m(k_\perp\rho)$ decays super-exponentially
beyond $m \approx k_\perp\rho$), and the orders are orthogonal in every power integral over a
surface of revolution, so the total scattered power is $\sum_m P_m$.

`scatter_orders` runs this loop: it solves $m = 0, \pm1, \pm2, \dots$ with the incident
component supplied per order, measures the power of every order through a surface and stops
when the pair $\pm m$ ($|m| \ge 2$) contributes less than a tolerance of the total so far, or
at `max_order`. `superpose_far_field` sums the patterns $F_m(\theta)e^{im\varphi}$ at an azimuth
so that the full three-dimensional pattern of the oblique problem is available from the
one-dimensional patterns of the orders. The same loop serves any source that is not a single
order (a tilted dipole, a focused beam): the user supplies its decomposition per $m$.

**Verification** (`tests/unit/physics/test_axisymmetric_orders.cpp`,
`tests/convergence/axisymmetric_oblique_sphere.cpp`): the orders summed over $|m| \le 25$
restore the plane wave at a point to $10^{-12}$ for both polarisations, and at $\theta_i = 0$
the p-polarised orders coincide with `axial_plane_wave`. The dielectric sphere ($n = 2$,
$ka = 1.5$) at $\theta_i = 50^\circ$ scatters the same cross-section as on the axis, which the
sum over the orders reproduces with $\pm m$ powers equal to $10^{-6}$, the far field of the
superposed orders carrying the total power to $10^{-2}$, and the cross-section converging
exponentially in $p$ (both polarisations, relative errors $3\cdot10^{-1}$, $8\cdot10^{-3}$,
$1.5\cdot10^{-3}$, $8\cdot10^{-5}$ for $p = 1, \dots, 4$ with seven to nine orders).

## Roadmap

The micropillar example (`examples/micropillar_qd`) exercises resonance, Purcell factor and
β factor together. Oblique incidence is covered by the sum over the orders
(`scatter_orders`). Still open: adaptivity on the meridian plane with the $r$-weighted
estimator. The Python bindings
(`AxisymmetricCavity`, `AxisymmetricResonance`, `AxisymmetricScattering`, `axial_plane_wave`,
`axisymmetric_gaussian_dipole`, `axisymmetric_poynting_flux`, `axisymmetric_far_field`,
`oblique_plane_wave`, `scatter_orders`, `superpose_far_field`) follow the C++ API one to one.

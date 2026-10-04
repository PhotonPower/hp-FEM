# Time-harmonic Maxwell equations

**Convention: time dependence $e^{-i\omega t}$, SI units.** Lossy media have $\operatorname{Im}\varepsilon > 0$.

## Strong form

With $\mathbf{D} = \varepsilon \mathbf{E}$, $\mathbf{B} = \mu \mathbf{H}$ and source current $\mathbf{J}$:

$$
\nabla \times \mathbf{E} = i\omega \mu \mathbf{H}, \qquad
\nabla \times \mathbf{H} = -i\omega \varepsilon \mathbf{E} + \mathbf{J}.
$$

Eliminating $\mathbf{H}$ gives the second-order (curl–curl) equation solved by hpfem:

$$
\nabla \times \left( \mu^{-1} \nabla \times \mathbf{E} \right) - \omega^2 \varepsilon \mathbf{E} = i\omega \mathbf{J}.
$$

$\varepsilon,\mu$ are complex $3\times 3$ tensors (diagonal in most applications). The magnetic field is recovered as $\mathbf{H} = (i\omega\mu)^{-1}\nabla\times\mathbf{E}$.

## Weak form

Find $\mathbf{E} \in H(\mathrm{curl};\Omega)$ with the prescribed tangential trace on PEC boundaries such that for all test functions $\mathbf{v}$:

$$
\int_\Omega \mu^{-1} (\nabla\times\mathbf{E})\cdot(\nabla\times\bar{\mathbf{v}}) \,dx
- \omega^2 \int_\Omega \varepsilon \mathbf{E}\cdot\bar{\mathbf{v}} \,dx
- \int_{\partial\Omega} \left( \mu^{-1}\nabla\times\mathbf{E}\right)\times\mathbf{n} \cdot \bar{\mathbf{v}} \,ds
= i\omega \int_\Omega \mathbf{J}\cdot\bar{\mathbf{v}} \,dx .
$$

Discretisation: $S \mathbf{e} - \omega^2 M \mathbf{e} = \mathbf{f}$ with $S_{ij} = (\mu^{-1}\nabla\times\phi_j, \nabla\times\phi_i)$, $M_{ij} = (\varepsilon\phi_j,\phi_i)$.
The basis functions are real, so conjugating the test function changes nothing; with complex $\varepsilon$ the matrices are complex-symmetric (not Hermitian), which solvers must respect.

## Discrete forms (`assembly/maxwell_forms.hpp`)

`element_maxwell` integrates the forms on one cell with the [Nédélec basis](nedelec.md),
the covariant Piola map $\phi = J^{-T}\hat\phi$, $\nabla\times\phi = J\,\hat\nabla\times\hat\phi/\det J$
(3D) resp. $\hat\nabla\times\hat\phi/\det J$ (2D) and a [simplex rule](quadrature.md) exact for
degree $2p + 2$:

$$
S^K_{ij} = \sum_q w_q |\det J|\; (\nabla\times\phi_i)\cdot\mu^{-1}(x_q)\,(\nabla\times\phi_j), \qquad
M^K_{ij} = \sum_q w_q |\det J|\; \phi_i\cdot\varepsilon(x_q)\,\phi_j, \qquad
b^K_i = \sum_q w_q |\det J|\; \phi_i\cdot f(x_q).
$$

$\varepsilon$ is a complex $d\times d$ tensor, $\mu^{-1}$ a complex tensor acting on curls
($3\times3$ in 3D, a scalar in 2D); PML cells supply stretched tensors ([pml.md](pml.md)).
`assemble_maxwell` returns $S$, $M$ and $b$ separately so that scattering ($S - \omega^2 M$)
and eigenproblems ($S e = \lambda M e$) share one assembly. `hcurl_error` measures
$\|E_h - E\|_{L^2}$ and $\|\nabla\times(E_h - E)\|_{L^2}$, `evaluate_hcurl` /
`evaluate_hcurl_curl` the physical field and its curl (for $H = (i\omega\mu)^{-1}\nabla\times
E$) at a reference point of a cell or, with a `mesh::PointLocator`, at an arbitrary
physical point ([mesh.md](mesh.md#point-location)). `io::FieldExporter` writes $E_h$ and
$\nabla\times E_h$ on a subdivided mesh and `io::cell_averages` the cell means for VTK
([mesh.md](mesh.md#field-export)).
**PEC** is `homogeneous_dirichlet` on the Nédélec DoF map (all edge and face DoFs of the
tagged facets, i.e. the tangential trace) followed by `apply_dirichlet`; **PMC** needs no
action. Verified by unit tests: symmetric and positive element matrices, gradient
functions and discrete gradients of the vertex functions in the stiffness kernel
(discrete de Rham), exact reproduction of constant fields, $L^2$ projection converging
with rate $p$, tensor coefficients entering linearly, and a vanishing tangential trace on
PEC boundaries of a solved problem.

## Eigenproblems and gauging (`solvers/eigen_solver.hpp`)

The discrete curl operator has the exact kernel $\nabla W_h$ (discrete de Rham complex),
so $S e = \lambda M e$ has the eigenvalue $0$ with multiplicity $\dim W_h^0$ (the
H1 space with the same orders and boundary condition). Nédélec elements produce no
*other* non-physical eigenvalues; the only task is to keep the Lanczos iteration away
from this kernel. hpfem does this with the **discrete gradient** $G$
(`assembly/discrete_gradient.hpp`): because the hierarchical Nédélec basis contains the
gradients of the H1 basis explicitly, $G$ has entries $\pm1$ (Whitney functions of the
edges at a vertex) and $1$, and $S G = 0$ holds to rounding; moreover
$G^T M G = K_{H^1}$, the Poisson stiffness matrix. The shift-invert operator
$(S - \sigma M)^{-1} M$ is applied together with the $M$-orthogonal projector
$P = I - G (G^T M G)^{-1} G^T M$ onto the complement of the gradients, which is an
invariant subspace; with $\sigma < 0$ the matrix $S - \sigma M$ is positive definite
and the eigenvalues closest to $\sigma$ are the smallest resonances
(`gauged_curl_curl_eigenpairs`, Spectra `SymGEigsShiftSolver` with Eigen SparseLU).
PEC enters by restricting all matrices to the free DoFs (`extract`, `free_dofs`).
Lossless media only for now: lossy or PML eigenproblems are complex symmetric and need
an Arnoldi variant (M4+).

Convergence test #2 (`tests/convergence/maxwell_cavity.cpp`): the PEC unit square
($\lambda = \pi^2(m^2+n^2)$) and cube ($\pi^2(m^2+n^2+l^2)$, at least two non-zero
indices) with $p = 1, 2$; the maximal relative error of the first eigenvalues must
decay with rate $2p$, no eigenvalue may lie below half the first exact one (zero
spurious modes), and the unit test compares against a dense generalized eigensolver
whose zero count equals $\dim W_h^0$.

## Problem classes

| Class | Unknown | Equation | Used for |
|---|---|---|---|
| Scattering | $\mathbf{E}$ at fixed $\omega$ | $(S-\omega^2M)\mathbf{e} = \mathbf{f}$ | scatterometry, metasurfaces, PV, sensors |
| Resonance (eigenmode) | $(\omega, \mathbf{E})$ | $S\mathbf{e} = \omega^2 M\mathbf{e}$, complex $\omega$ with PML | VCSEL, cavities, Purcell |
| Propagating mode | $(k_z, \mathbf{E}_\perp)$ | quadratic eigenproblem in $k_z$ at fixed $\omega$ | waveguides, PICs |

## Scattered-field formulation

For an incident field $\mathbf{E}^{\mathrm{inc}}$ that solves Maxwell in the background medium $\varepsilon_b, \mu_b$, write $\mathbf{E} = \mathbf{E}^{\mathrm{inc}} + \mathbf{E}^{\mathrm{sc}}$. Then $\mathbf{E}^{\mathrm{sc}}$ solves the curl–curl equation with the source
$\mathbf{f} = \omega^2 (\varepsilon - \varepsilon_b)\mathbf{E}^{\mathrm{inc}} - \nabla\times\big((\mu^{-1} - \mu_b^{-1})\nabla\times\mathbf{E}^{\mathrm{inc}}\big)$ supported only on the scatterer, and PML absorbs $\mathbf{E}^{\mathrm{sc}}$ cleanly. In the weak form the curl term is integrated by parts, $\ell(\mathbf{v}) = \omega^2\int(\varepsilon - \varepsilon_b)\mathbf{E}^{\mathrm{inc}}\cdot\mathbf{v} - \int(\mu^{-1} - \mu_b^{-1})\nabla\times\mathbf{E}^{\mathrm{inc}}\cdot\nabla\times\mathbf{v}$, which `MaxwellForm::source` and `MaxwellForm::curl_source` carry. For layered backgrounds (gratings, masks) the incident field is the analytic multilayer solution.

## Layered background (`physics/layer_stack.hpp`, ADR-0009)

Structures on substrates, in films or on multilayers are scattered-field problems whose
background is a planar layer stack rather than a homogeneous medium. `physics::LayerStack<Dim>`
holds an incidence medium (semi-infinite, lossless) above the coordinate `top`, finite
non-magnetic layers and a semi-infinite substrate, perpendicular to the last coordinate ($y$
in 2D, $z$ in 3D). `plane_wave(k_0, \theta, \text{pol})` returns the exact field of a unit
plane wave incident from above at the angle $\theta$ from the normal as an `IncidentField`
(value and curl) together with the reflectance $R$, the transmittance $T$ (power flux into the
substrate, also when it is lossy) and the absorption $A = 1 - R - T$ of the finite layers.

In every region $j$ the scalar wave function $u$ ($E_s$ for s polarisation, $H$ for p; in 2D
only p exists because `Scattering<2>` solves for the in-plane $E$) is
$$
u_j = e^{i k_\parallel x}\Big(a_j\, e^{-i k_{z,j}(z - z_j^{top})} + b_j\, e^{i k_{z,j}(z - z_j^{bot})}\Big),
\qquad k_{z,j} = \sqrt{k_0^2\varepsilon_j - k_\parallel^2},\ \operatorname{Im} k_{z,j} \ge 0,
$$
with the downward amplitude $a_j$ referenced to the top and the upward amplitude $b_j$ to the
bottom interface of the region, so that every exponential inside a layer has modulus $\le 1$.
With the admittance-like quantities $q_j = k_{z,j}$ (s) or $k_{z,j}/\varepsilon_j$ (p), the
interface coefficients $r_{j} = (q_j - q_{j+1})/(q_j + q_{j+1})$, $t_j = 2q_j/(q_j + q_{j+1})$
and the round trip $\phi_j = e^{2 i k_{z,j} d_j}$, the total reflection at interface $j$ follows
the Airy recursion from the substrate upwards,
$$
R_j = \frac{r_j + R_{j+1}\phi_{j+1}}{1 + r_j R_{j+1}\phi_{j+1}},\qquad R_N = r_N,
$$
and the amplitudes downwards, $a_{j+1} = a_j^{bot}\, t_j / (1 + r_j R_{j+1}\phi_{j+1})$,
$b_{j+1} = R_{j+1}\, a_{j+1}\, e^{i k_{z,j+1} d_{j+1}}$. This is the S-matrix form of the
transfer problem: a 50 µm silver layer gives the same $R$ as the half-infinite metal and a
finite field everywhere, where transfer matrices overflow. $E$ and $\nabla\times E = i\omega\mu_0 H$
follow per plane-wave component from $H = k\times E/(\omega\mu_0)$ (s) and
$E = -k\times H/(\omega\varepsilon_0\varepsilon_j)$ (p), $R = |R_0|^2$,
$T = \operatorname{Re} q_{sub}\,|a_{sub}|^2 / \operatorname{Re} q_0$.

Unit tests check Fresnel's coefficients for one interface (s and p, 2D and 3D), $R + T = 1$
for a Bragg mirror and $R + T + A = 1$ for lossy stacks, the thick-metal limit, the
continuity of the tangential $E$, of $\varepsilon E_z$ and of the tangential $H$ across every
interface and the curl against finite differences. Used as background of the scattered-field
formulation, the stack's field is the incident field and the source lives only where the
permittivity deviates from the stack (see `Scattering`, next section, and the slit–groove
benchmark in [validation.md](../validation.md)).

## Scattering problems (`physics/scattering.hpp`)

`physics::Scattering<Dim>` solves, at a fixed angular frequency $\omega$ with
$k_0 = \omega / c_0$ and SI lengths, the curl–curl equation divided by $\mu_0$,

$$
\nabla\times(\mu_r^{-1}\nabla\times\mathbf{E}) - k_0^2\,\varepsilon_r\,\mathbf{E} = \mathbf{f},
\qquad \mathbf{f} = i\omega\mu_0\mathbf{J},
$$

on a Nédélec space with isotropic materials $(\varepsilon_r, \mu_r)$ assigned by cell tag
(`materials::MaterialMap`, the background for unlisted tags). Two formulations share the
operator $A = S - k_0^2 M$ assembled with per-cell coefficients
(`assemble_maxwell` with a cell-form factory):

- **total field**: the unknown is $\mathbf{E}$; sources are currents and the incident
  field prescribed as tangential trace on `incident_tags` facets;
- **scattered field**: the unknown is $\mathbf{E}^{sc} = \mathbf{E} - \mathbf{E}^{inc}$ for an
  analytic incident field of the background medium; the sources
  $k_0^2(\varepsilon_r - \varepsilon_{r,b})\mathbf{E}^{inc}$ (`source`) and
  $-(\mu_r^{-1} - \mu_{r,b}^{-1})\nabla\times\mathbf{E}^{inc}$ (`curl_source`) live on the
  scatterer only; PEC facets get $\mathbf{n}\times\mathbf{E}^{sc} = -\mathbf{n}\times\mathbf{E}^{inc}$
  and incident facets $\mathbf{n}\times\mathbf{E}^{sc} = 0$.

Analytic incident fields (`physics/sources.hpp`) come as value and curl: the plane wave
$\mathbf{E}_0 e^{i\mathbf{k}\cdot\mathbf{x}}$ with $\mathbf{E}_0\perp\mathbf{k}$ (curl
$i\mathbf{k}\times\mathbf{E}$), and the dipole field, the outgoing solution of
$\nabla\times\nabla\times\mathbf{E} - k^2\mathbf{E} = \mathbf{p}\,\delta(\mathbf{x} - \mathbf{x}_0)$,
$\mathbf{E} = (I + \nabla\nabla/k^2)\,g\,\mathbf{p}$ with the scalar Green's function
$g = e^{ikr}/(4\pi r)$ (3D) or $g = \tfrac{i}{4}H_0^{(1)}(kr)$ (2D, line dipole with in-plane
moment; Hankel functions from `core/special_functions.hpp`). Explicitly, with
$\mathbf{n} = (\mathbf{x} - \mathbf{x}_0)/r$,

$$
\mathbf{E}_{3D} = g\Big[\big(1 + \tfrac{i}{kr} - \tfrac{1}{(kr)^2}\big)\mathbf{p}
+ \big(-1 - \tfrac{3i}{kr} + \tfrac{3}{(kr)^2}\big)\mathbf{n}(\mathbf{n}\cdot\mathbf{p})\Big],
\qquad \nabla\times\mathbf{E} = \nabla g\times\mathbf{p},
$$

and in 2D $\mathbf{E} = (g + g'/(k^2 r))\,\mathbf{p} + (g'' - g'/r)/k^2\;\mathbf{n}(\mathbf{n}\cdot\mathbf{p})$.

The analytic dipole field is an incident field of the *homogeneous* background; an emitter
inside a structure (a quantum dot in a cavity, `examples/quantum_dot_purcell`) is modelled
instead as a volume current in the total-field formulation, `physics::gaussian_current`:
$\mathbf{f} = i\omega\mu_0\,\mathbf{p}\,g_\sigma(\mathbf{x} - \mathbf{x}_0)$ with the
normalised Gaussian $g_\sigma$ of a width $\sigma$ below the cell size and far below the
wavelength. The emitted power is the Poynting flux of the total field through a closed
surface around the current, and ratios such as the Purcell factor $P/P_0$ (the same current
in the homogeneous host, on the same mesh) are insensitive to $\sigma$: the example checks a
current in front of a PEC mirror against the image-dipole solution to better than one per
cent.
Both are checked by finite differences (curl of the value, and
$\nabla\times\nabla\times\mathbf{E} = k^2\mathbf{E}$) in the unit tests.

The solution is evaluated as total or scattered field at reference points of cells or, via
`mesh::PointLocator`, at physical points (the incident field is added or subtracted
according to the formulation), and `error` measures the unknown against an analytic field
of the same kind. An optional `pml::PmlBox` replaces the material tensors by their
stretched versions ([pml.md](pml.md)). Bloch-periodic constraints, curved elements and
post-processing (fluxes, cross-sections, far fields) are the following M4 items.

**Mie reference** (`physics/mie.hpp`): for the infinite dielectric cylinder under a plane
wave with in-plane electric field ($H_z$ polarisation) the scattered field is
$H_z^s = \sum_n c_n H_n^{(1)}(kr)e^{in\varphi}$ with
$c_n = i^n\,[n_c J_n'(kR)J_n(n_ckR) - J_n(kR)J_n'(n_ckR)] / [H_n^{(1)}(kR)J_n'(n_ckR) -
n_c H_n^{(1)\prime}(kR)J_n(n_ckR)]$ (continuity of $H_z$ and of $\varepsilon^{-1}\partial_r H_z$),
and the scattering width is $\sigma_{sca} = \tfrac{4}{k}(|c_0|^2 + 2\sum_{n\ge1}|c_n|^2)$.
Convergence test #4 (`tests/convergence/mie_cylinder.cpp`): a lossless cylinder
($kR = 1.5$, $n_c = 1.5$) in a PML-terminated box on the `square_with_disc` mesh with the
curved interface, scattered-field formulation; the scattering width from the flux of the
scattered field through the cylinder surface converges to the series value under
p-refinement and the absorption cross-section vanishes within the discretisation error.

**Mie sphere** (`physics::mie_sphere`, Bohren & Huffman ch. 4): a sphere of radius $a$ and
complex permittivity $\varepsilon_r$ in a lossless background of index $n_b$ under the plane
wave $\hat x\,e^{ikz}$, $k = n_b k_0$, size parameter $x = ka$, relative index
$m = \sqrt{\varepsilon_r}/n_b$. With the Riccati–Bessel functions $\psi_n(\rho) = \rho j_n(\rho)$,
$\xi_n(\rho) = \rho h_n^{(1)}(\rho)$ and the logarithmic derivative
$D_n(mx) = \psi_n'(mx)/\psi_n(mx)$, evaluated for complex $mx$ by the downward recurrence of
$j_n$ (`spherical_bessel_j(max_order, z)` in `core/special_functions.hpp`),
$$
a_n = \frac{(D_n/m + n/x)\psi_n(x) - \psi_{n-1}(x)}{(D_n/m + n/x)\xi_n(x) - \xi_{n-1}(x)},\qquad
b_n = \frac{(mD_n + n/x)\psi_n(x) - \psi_{n-1}(x)}{(mD_n + n/x)\xi_n(x) - \xi_{n-1}(x)},
$$
$Q_{sca} = \tfrac{2}{x^2}\sum_n (2n+1)(|a_n|^2 + |b_n|^2)$,
$Q_{ext} = \tfrac{2}{x^2}\sum_n (2n+1)\,\mathrm{Re}(a_n + b_n)$, $Q_{abs} = Q_{ext} - Q_{sca}$,
cross-sections $\sigma = Q\pi a^2$. The fields are the vector spherical harmonic expansions
$E^{sca} = \sum_n E_n(i a_n N^{(3)}_{e1n} - b_n M^{(3)}_{o1n})$ outside and
$E^{int} = \sum_n E_n(c_n M^{(1)}_{o1n} - i d_n N^{(1)}_{e1n})$ inside,
$E_n = i^n(2n+1)/(n(n+1))$, with the internal coefficients
$c_n = m\,W_n/(\psi_n(mx)\xi_n'(x) - m\,\xi_n(x)\psi_n'(mx))$,
$d_n = m\,W_n/(m\,\psi_n(mx)\xi_n'(x) - \xi_n(x)\psi_n'(mx))$, $W_n = \psi_n\xi_n' - \xi_n\psi_n' = i$.
The unit tests check the optical theorem, the Rayleigh limit, the lossless limit and the
continuity of the tangential field and of $\varepsilon E_r$ across the surface, which ties the
four coefficient families and the harmonics together. The 3D half of convergence test #4 and
validation benchmark C (`tests/convergence/mie_sphere.cpp`, [validation.md](../validation.md))
compare $Q_{sca}$, $Q_{abs}$ and the total field at points inside and outside against the
series on the `box_with_ball` mesh.

**Verification** (`tests/convergence/maxwell_scattering.cpp`): with the exact tangential
trace prescribed on all sides, a plane wave and a dipole field whose source lies outside
the domain are reproduced with rate $p$ in the $H(\mathrm{curl})$ norm for $p = 1, 2$ in 2D
and 3D (the first-kind space converges with rate $p$ in both the $L^2$ and the curl part);
the scattered- and total-field formulations of a dielectric disc under the same boundary
data converge to the same total field. Unit tests cover setup validation, materials by tag,
the assembled operator against a manual assembly, a zero-contrast scatterer (zero
scattered field), PEC data in the scattered-field formulation (vanishing total trace) and
current sources.

## Boundary conditions

- **PEC** $\mathbf{n}\times\mathbf{E} = 0$: eliminate edge/face DoFs (`homogeneous_dirichlet`).
- **Prescribed tangential trace** $\mathbf{n}\times\mathbf{E} = \mathbf{n}\times\mathbf{g}$ (incident
  field on the boundary of a test domain, Dirichlet coupling to analytic solutions):
  `tangential_dirichlet_values` projects $\mathbf{g}$ hierarchically onto the trace space, edge by
  edge onto the $p_e$ edge functions (whose tangential traces are the Legendre polynomials
  up to degree $p_e - 1$, so the projection is exact for traces in the discrete space), then
  in 3D the tangential remainder onto the face functions; eliminated with `apply_dirichlet`.
- **PMC** $\mathbf{n}\times(\mu^{-1}\nabla\times\mathbf{E}) = 0$: natural, do nothing.
- **Bloch-periodic** $\mathbf{E}(x+a) = e^{i\mathbf{k}\cdot\mathbf{a}}\mathbf{E}(x)$: slave-facet DoFs
  constrained to the master-facet DoFs with the Bloch phase, see
  [below](#bloch-periodic-constraints).
- **Transparent**: PML, see [pml.md](pml.md).
- **Waveguide port**: modal absorption and excitation on boundary facets, the low-rank
  term of the mode expansion, see [below](#waveguide-ports-and-s-parameters-physicswaveguide_porthpp).

## Bloch-periodic constraints (`assembly/periodic.hpp`)

For a periodic direction with lattice vector $\mathbf{a}$ (`PeriodicPair`: master tag, slave
tag, shift, phase $e^{i\mathbf{k}\cdot\mathbf{a}}$), the facets on the slave side are the master
facets translated by $\mathbf{a}$ (matched by centroids; the two sides must be meshed
identically). Every slave-facet DoF is expressed through the master-facet DoFs:
`bloch_constraints` takes each master basis function, shifts it by $\mathbf{a}$ and
multiplies by the phase, and projects its tangential trace onto the slave trace space with
`tangential_dirichlet_values`. Because the trace spaces coincide, the projection is exact
and the coefficients come out as the phase times $\pm1$ for edge functions (the sign is
the orientation flip of ADR-0003 when the global vertex order differs between the two
sides) and as the face permutation matrices in 3D — no orientation bookkeeping is needed.
Two periodic directions share the corner edges: those chain (slave of a slave) and are
resolved by substitution.

`fespace::Constraints` holds such linear relations $x_s = \sum_i c_i x_{m_i}$ generally
(hanging nodes of irregular refinement will use the same object), resolves chains, rejects
cycles, and applies them to an assembled system through the prolongation $P$ of the free
DoFs: $A_f = P^H A P$, $b_f = P^H b$, solve, $x = P x_f$. The conjugate transpose matters:
the weak form is bilinear (test functions are not conjugated), and the boundary terms on
the two periodic faces cancel only when the test functions carry the *inverse* phase
$e^{-i\mathbf{k}\cdot\mathbf{a}}$, which for real Bloch vectors is the conjugate (a plain
transpose gives a wrong, non-convergent solution; complex Bloch vectors — evanescent
Bloch waves — would need the inverse instead and are not supported yet). For real
coefficients (hanging nodes) the two coincide. `physics::Scattering` does this when `ScatteringSetup::periodic`
is set; Dirichlet data is applied before the reduction, which is consistent for DoFs that
are both constrained and prescribed (box corners) as long as the data itself is periodic.

Verified by unit tests (one-to-one coefficients with $|c| = 1$ on structured and randomly
renumbered meshes, corner edges of two directions resolving to the product of the phases,
the constrained solution satisfying its constraints) and by the convergence test
`tests/convergence/bloch_plane_wave.cpp`: a plane wave at oblique incidence on a unit cell
with Bloch-periodic sides and the exact trace on the remaining sides converges with rate
$p$ in $H(\mathrm{curl})$ (2D with one periodic direction, 3D with two) and exponentially
under p-refinement.

## Propagating modes (`physics/propagating_mode.hpp`)

For a waveguide with a 2D cross-section and propagation $e^{i\beta z}$, write
$\mathbf{E} = (\mathbf{E}_t + \hat z E_z)\,e^{i\beta z}$ with $\mathbf{E}_t$ in the Nédélec space
and the scaled longitudinal field $e_z = iE_z/\beta$ in the H1 space of the same order. The
curl–curl weak form then separates into the generalized eigenproblem (Lee–Sun–Cendes)

$$
\begin{pmatrix} S - k_0^2 M_\varepsilon & 0 \\ 0 & 0 \end{pmatrix}
\begin{pmatrix} e_t \\ e_z \end{pmatrix} = -\beta^2
\begin{pmatrix} M_\mu & M_\mu G \\ G^T M_\mu & G^T M_\mu G - k_0^2 M^{H1}_\varepsilon \end{pmatrix}
\begin{pmatrix} e_t \\ e_z \end{pmatrix},
$$

with the curl–curl matrix $S$ (weight $\mu_r^{-1}$), the Nédélec mass matrices $M_\varepsilon$,
$M_\mu$ (weights $\varepsilon_r$, $\mu_r^{-1}$), the discrete gradient $G$ (so that
$\nabla_t e_z$ is exactly $G e_z$ in the Nédélec basis) and the H1 mass matrix with weight
$\varepsilon_r$. The matrix on the right is indefinite and the one on the left singular, so the
pencil is solved with the real nonsymmetric shift-invert Arnoldi of
`solvers::generalized_eigenpairs_near` on $(A - \sigma B)^{-1}B$ with
$\sigma = -1.05\,k_0^2 n_{\max}^2$: the eigenvalues $-\beta^2$ closest to $\sigma$ are the guided
modes, largest $\beta$ first; spurious solutions sit at $\beta^2 = 0$ and never appear. PEC
walls remove the tangential Nédélec DoFs and the H1 boundary DoFs. `PropagatingMode::solve`
returns $\beta$, $n_{\mathrm{eff}} = \beta/k_0$ and the coefficients of $\mathbf{E}_t$ and
$E_z = -i\beta e_z$. Lossless media only (the quadratic → linear step and the real Arnoldi
assume real pencils); leaky and lossy modes need the complex solver of M6.

**Verification** (convergence test #5, `tests/convergence/slab_waveguide.cpp`): the symmetric
slab (core $n = 1.5$, $d = 1$, cladding $n = 1$, $k_0 d = 2$, a single even TE mode) on a strip
with PEC walls in $y$, which admits exactly the TE modes $E = E_y(x)$; the effective index
converges to the root of $\tan(\kappa d/2) = \gamma/\kappa$ with rate $2p$ under h-refinement
(least-squares rates 2.0 and 3.7 for $p = 1, 2$; eigenvalue errors wobble from mesh to mesh)
and exponentially under p-refinement ($2\cdot10^{-3}$, $10^{-7}$, $6\cdot10^{-9}$, $5\cdot10^{-13}$
for $p = 1..4$ on four cells per unit length; the strip is $\pm12$ long so that the exponential
tails do not limit the accuracy). The unit test also checks that
the mode has no longitudinal field and that the setup rejects lossy materials and mismatched
spaces.


On a locally refined cross-section (`mesh::AdaptiveMesh`, hanging nodes) the Nédélec and the H1
space are constrained with `assembly::hanging_constraints`, restricted to the DoFs left after
the PEC elimination (`restrict_constraints`) and combined block-wise
(`block_constraints`); the pencil is reduced with the prolongation and the eigenvectors are
expanded, exactly as in the resonance solver, so the modes are conforming across the hanging
edges (unit test on the slab with the core refined on one side).
## Waveguide ports and S-parameters (`physics/waveguide_port.hpp`)

A port is a set of boundary facets through which the guided modes of an attached waveguide
enter and leave the domain. On the port $\Gamma$ the tangential field is expanded in the
modes of the cross-section, $E_t = \sum_m (a_m + b_m)\,\hat e_m$ with prescribed incoming
amplitudes $a_m$ and unknown outgoing $b_m$. The natural boundary term of the weak form,
in 2D $\oint_\Gamma (\mu^{-1}\nabla\times E)\,(v\cdot t')\,ds$ with the tangent
$t' = (n_y, -n_x)$ (from $\int_\Omega \nabla\times(c)\cdot v = \int_\Omega c\,\nabla\times v +
\oint c\,(v\cdot t')$), is written with the modal expansion of $w = \mu^{-1}\nabla\times E =
\sum_m (b_m - a_m)\,\hat w_m$: the magnetic field of a mode flips sign with its direction of
propagation, the tangential electric field does not. The modes are bi-orthogonal,
$\int_\Gamma \hat e_m\hat w_n\,ds = \delta_{mn}N_m$, so the unknown sum $a_m + b_m$ is the
projection $c_m = \int_\Gamma E_t\hat w_m\,ds / N_m$ of the solution itself, and the port
contributes

$$
\sum_m \frac{1}{N_m}\,q_m q_m^\top \quad\text{to the operator},\qquad
2\sum_m a_m\,q_m \quad\text{to the load},\qquad
q_{m,i} = \int_\Gamma (\phi_i\cdot t')\,\hat w_m\,ds ,
$$

a low-rank term on the port DoFs (edge DoFs of the port facets; the static condensation of
the interior DoFs is unaffected). Modes in the expansion leave without reflection; modes not
included see a PEC, so a port far enough from any scatterer needs only the guided modes, and
a near field at the port is absorbed by adding evanescent modes (`num_modes`). After the
solve, $b_m = c_m - a_m$; the power-normalised scattering matrix over the propagating
channels $(p, m)$ is $S_{ij} = b_i\sqrt{P_i}/(a_j\sqrt{P_j})$ with the modal powers $P_m$
(`s_parameters` solves once per channel with unit incoming amplitude on that channel).

**2D port modes.** For the in-plane field $E = (E_x, E_y)$ with $H_z$ out of plane the port
is a straight chain of boundary edges with the cross-section coordinate $s$ (increasing
along $t'$) and the outward normal coordinate $\xi$. A mode $H_z = h(s)\,e^{i\beta\xi}$ solves

$$
\partial_s\big(\varepsilon_r^{-1}\partial_s h\big) + k_0^2\mu_r\,h = \beta^2\varepsilon_r^{-1}h ,
$$

discretised with a hierarchical 1D p-FEM on the port edges (vertex functions and integrated
Legendre bubbles, the orders of the inside cells, materials of the inside cells; the dense
generalized eigenproblem $A h = -\beta^2 B h$ with $A = K_{1/\varepsilon} - k_0^2M_\mu$,
$B = M_{1/\varepsilon}$). The ends of the port lie on PEC walls, where $E_\xi \propto
\partial_s h = 0$ is the natural condition, so no essential condition is imposed. Modes are
ordered by $-\beta^2$ ascending: guided modes first (largest $\beta$), then the propagating
box modes of a closed cross-section, then the evanescent ones with the slowest decay
($\beta = i|\beta|$). From Ampère's law, $E = (i/\omega\varepsilon)\nabla\times H$, the outgoing
mode has the tangential trace and the curl

$$
\hat e_m = E\cdot t' = -\frac{\beta_m}{\omega\varepsilon_0\varepsilon_r(s)}\,h_m(s),\qquad
\hat w_m = \mu_r^{-1}\nabla\times E = i\omega\mu_0\,h_m(s),
$$

so that $\hat w_m = -i(k_0^2\varepsilon_r/\beta_m)\,\hat e_m$ (for the TEM mode this is the
Silver–Müller condition $\nabla\times E = -ik_0\,E\cdot t'$ with the convention
$e^{-i\omega t}$), $N_m = \int\hat e_m\hat w_m\,ds \propto h_m^\top B h_m$ (the
bi-orthogonality is the $B$-orthogonality of the eigenvectors) and the power of the
unit-amplitude mode is $P_m = \beta_m/(2\omega\varepsilon_0)\int h_m^2/\varepsilon_r\,ds$ (zero
for evanescent modes). The eigenvectors are $B$-normalised and their sign is fixed
independently of the port's orientation: the tangential electric field of the mode along the
direction from the lexicographically smaller end point of the port to the larger one is
positive at the port midpoint (positive slope for a mode vanishing there), so that parallel
ports of a straight guide carry identical mode fields and $S_{21} = e^{i\beta L}$. Ports
require the total-field formulation and lossless materials on the port; 3D ports (modes of
the 2D cross-section from `PropagatingMode`) are a later item of M12.

**Verification** (`tests/unit/physics/test_waveguide_port.cpp`,
`tests/convergence/waveguide_port.cpp`): the port modes of the PEC parallel plate are
$\cos(n\pi s/a)$ with $\beta_n = \sqrt{k_0^2 - (n\pi/a)^2}$ ($10^{-6}$ at $p = 4$, evanescent
$n = 2$ with $\beta = i|\beta|$ and zero power), the even TM mode of the slab reproduces the
root of $\kappa\tan(\kappa d/2) = (\varepsilon_{\mathrm{core}}/\varepsilon_{\mathrm{clad}})\gamma$ to
$10^{-6}$; a straight parallel plate with two propagating modes has $S_{ij} = \delta_{m_im_j}
e^{i\beta_mL}$ between the ports and no reflection or mode conversion ($2\cdot10^{-5}$), $S$
symmetric and unitary; the slab section transmits its guided mode with $e^{i\beta L}$. The
convergence test refines the order on a fixed mesh of the slab section: both the error of
$S_{21}$ against $e^{ik_0n_{\mathrm{eff}}L}$ with the analytic $n_{\mathrm{eff}}$ and the
residual reflection $|S_{11}|$ decay exponentially in $p$ (the 1D port modes converge with
the same order as the 2D field).

## Resonances (`physics/resonance.hpp`)

An open structure (a micro-cavity between Bragg mirrors, a plasmonic particle, a ring) has no
bound states but *quasi-normal modes*: solutions of the source-free curl–curl equation with
outgoing waves, which exist only at complex frequencies. With the $e^{-i\omega t}$ convention a
mode decays in time as $e^{\mathrm{Im}(\omega)\,t}$, so $\mathrm{Im}\,\omega < 0$, and its
quality factor is

$$
Q = \frac{\mathrm{Re}\,\omega}{-2\,\mathrm{Im}\,\omega},
\qquad \lambda_{\mathrm{res}} = \frac{2\pi c_0}{\mathrm{Re}\,\omega}.
$$

The PML of the scattering problems turns the outgoing-wave condition into complex-symmetric
absorbing layers (the stretched material tensors of `pml/pml.hpp`, designed at the target
frequency), so the modes become eigenpairs of the discrete pencil

$$
S\,e = k^2\,M\,e, \qquad k^2 = \omega^2/c_0^2 \in \mathbb{C},
$$

with the stiffness matrix $S$ (weight $\mu_r^{-1}$, PML-stretched) and the mass matrix $M$
(weight $\varepsilon_r$, PML-stretched, possibly lossy). PEC walls remove the tangential DoFs,
hanging-node constraints are applied as in the scattering problem. `physics::Resonance`
assembles this pencil and calls `solvers::complex_eigenpairs_near` with the shift
$\sigma = k_{\mathrm{target}}^2$: a shift-invert Arnoldi iteration in complex arithmetic on
$(S - \sigma M)^{-1} M$ (direct factorisation with the chosen backend, modified Gram–Schmidt
with re-orthogonalisation, explicit restarts from the wanted Ritz vectors, the relative
residual $|h_{m+1,m}\,y_m| / |\theta|$ as convergence test). The result is the list of modes
ordered by the distance of $\omega$ to the target, each with $\omega$, $\lambda_{\mathrm{res}}$,
$Q$, the Arnoldi residual and the field coefficients. The pencil also contains the gradient
kernel at $k^2 = 0$ and, on a strip with PEC walls, lossless guided modes of the layers;
both lie away from a well-chosen target, and a narrow strip pushes the latter far up.

**Verification** (`tests/convergence/fabry_perot_resonance.cpp`): a dielectric slab ($n = 3.5$,
$d = 1$) in vacuum on a strip with PEC walls (so that the $y$-uniform modes $E = E_y(x)$ are
the one-dimensional Fabry–Pérot modes) has the exact resonances

$$
k_m = \frac{\pi m}{n d} - \frac{i}{n d}\ln\frac{n+1}{n-1},
\qquad Q_m = \frac{\pi m}{2\ln\frac{n+1}{n-1}} ,
$$

from the round-trip condition $r^2 e^{2iknd} = 1$ with $r = (n-1)/(n+1)$. With a PML of about
two wavelengths on both sides the computed $k_4$ converges exponentially under p-refinement
(relative errors $1.4\cdot10^{-1}$, $5.8\cdot10^{-2}$, $1.0\cdot10^{-3}$, $1.6\cdot10^{-4}$,
$8.5\cdot10^{-7}$ for $p = 1..5$ on four cells per unit length; $Q_4 = 10.690$), the
neighbours $k_3$ and $k_5$ come out of the same run, and every returned pair satisfies the
discrete pencil to $10^{-15}$. The unit tests check the complex solver against a dense
reference (scale invariant from $O(1)$ to $O(10^{13})$ matrices) and the closed PEC square
(real eigenfrequencies $c_0\pi\sqrt{m^2 + n^2}$, also on a hanging-node mesh).

## Modal expansion by Riesz projection (`physics/riesz_projection.hpp`)

How much of a scattering or emission problem is carried by a given quasi-normal mode? The
solution $x(\omega) = A(\omega)^{-1} b(\omega)$ of the discrete source problem on the pencil
of the resonance solver, $A(\omega) = S - (\omega/c_0)^2 M$ with the PML frozen at its design
frequency, is a meromorphic function of $\omega$: the stiffness and mass matrices do not depend
on $\omega$, the source does so analytically (a current enters as $i\omega\mu_0 b_J$), and the
poles are exactly the eigenvalues $\omega_n$ of `physics::Resonance` /
`physics::AxisymmetricResonance`. Cauchy's integral formula on a contour $C_0$ around the real
frequencies of interest, with small circles $C_n$ around the enclosed poles, therefore splits
the solution into modal contributions and a background (Zschiedrich, Binkowski, Nikolay,
Burger, Lockau, Schmidt, *Phys. Rev. A* **98**, 043806 (2018)):

$$
x(\omega) = \frac{1}{2\pi i}\oint_{C_0}\frac{x(\omega')}{\omega'-\omega}\,d\omega'
 - \sum_n \frac{1}{2\pi i}\oint_{C_n}\frac{x(\omega')}{\omega'-\omega}\,d\omega' .
$$

For a simple pole the modal term is exactly $R_n/(\omega-\omega_n)$ with the residue
$R_n = \frac{1}{2\pi i}\oint_{C_n} x(\omega')\,d\omega'$: one vector per mode, valid at every
$\omega$ (inside the circle as well, so high-$Q$ modes next to the real axis need no special
treatment) and obtained without normalising the eigenvector — the residue *is* the eigenvector
times its excitation coefficient. Poles too close to each other (circles below a fraction of
$|\omega|$) share a contour, whose contribution keeps the $\omega$-dependent integral
(for $\omega$ inside such a group contour the direct solution minus the integral).

**Contours and quadrature.** `RieszSetup` lists the poles (from the resonance solver — every
pole inside the background contour must be listed, which `Resonance` guarantees when its
farthest returned mode lies outside), the range $[\omega_{\min}, \omega_{\max}]$ and the point
counts. The background contour is an ellipse around the range (semi-axes: half the width plus
a margin, times `background_aspect` in the imaginary direction — a flat ellipse keeps the
heavily damped PML modes outside), every pole inside it gets a circle of radius
`radius_factor` times its distance to the nearest other pole or to the background contour.
All contours are integrated by the trapezoidal rule, which converges exponentially for
periodic analytic integrands: the error of a contour with $N$ points decays like $\rho^N$
with $\rho < 1$ the ratio between the contour and the distance of the nearest singularity of
the integrand. For the pole circles that is the next pole ($\rho \le$ `radius_factor`, so
16 points give $10^{-8}$); for the background integrand the enclosed poles themselves would
set $\rho$, so the modal parts $\sum_n R_n/(\omega'-\omega_n)$ are subtracted from it before
integration — their own Cauchy integral over $C_0$ vanishes because both poles are enclosed —
and the rate is then governed by the poles *outside* $C_0$ and by the evaluation frequency
$\omega$ (the factor $1/(\omega'-\omega)$ is singular at $\omega$, so the expansion converges
best well inside the range). Every contour reports the relative difference between its $N$-
and $N/2$-point rules (the latter is free, every second point with doubled weights) as a
convergence check, with a warning above `convergence_warning`. Each contour point costs one
factorisation of $A(\omega')$ and one batched solve for all sources (`solve_many`); on cuDSS
a 194 k-DoF problem takes about 1.5 s per point.

**Observables.** Any linear functional $Q(x) = q^\top x$ — point values
(`add_point_value`), the emitted power of a current source
$Q(x) = -\tfrac12\int E\cdot J^* = -\tfrac12\,\overline{b_J}^{\,\top} x$ (`add_emitted_power`;
bodies of revolution carry the azimuthal factor $2\pi$), Fourier coefficients and far-field
amplitudes through the functionals of `assembly/functionals.hpp` — is sampled on the contours,
so its modal contributions $Q(R_n)/(\omega-\omega_n)$ and background follow for a whole
spectrum without further solves (`spectrum(source, functional, omegas)`: one row per contour,
the background last; `total` is their sum). Because $\mathrm{Re}$ is linear, the real parts of
the modal shares of the emitted power add up to the physical power exactly; the Purcell
spectrum of an emitter is thus a sum of Lorentzians plus a smooth background. Quadratic
quantities (fluxes through surfaces, $|E|^2$) are not expanded: `expand(source, omega)` returns
the exact field as the sum of the modal fields and the (stored) background field, and the modal
fields `field(contour, source, omega)` may be post-processed individually, aware that the
cross terms are then lost.

**Verification** (`tests/unit/physics/test_riesz_projection.cpp`): on the Fabry–Pérot strip
of the resonance test (dipole line current inside the slab, 12 listed modes) the sum of the
modal terms and the background reproduces the directly solved emitted power and a point value
at 20 frequencies across the range to $10^{-8}$ (24 points per pole, 64 on the background),
the expanded field equals the direct field, the residue of the $m = 4$ mode is parallel to the
eigenvector of the resonance solver, and the background integral converges like $0.4^N$
(errors $3.9\cdot10^{-4}$, $2.5\cdot10^{-7}$, $1.7\cdot10^{-10}$, $1.1\cdot10^{-13}$ for 8, 16,
24, 32 points, the half-rule estimate tracking them). The same identity holds on the order-1
block pencil of a dielectric sphere with a transverse dipole at its centre (2.5D, Mie poles of
the sphere resonance test). `examples/micropillar_qd` compares the Purcell spectrum of the
quantum dot from the modal sum with the frequency sweep of the example.

## Band structures (`physics/band_structure.hpp`)

A photonic crystal is a lossless periodic structure with lattice vectors $a_j$. By Bloch's
theorem its eigenmodes are Bloch waves, $E(x + a_j) = e^{ik\cdot a_j} E(x)$ for a wave vector
$k$ of the first Brillouin zone, so one unit cell with the Bloch-periodic constraints of the
previous section (phases $e^{ik\cdot a_j}$, one `PeriodicPair` per lattice vector) carries the
source-free eigenproblem

$$
\nabla\times(\mu_r^{-1}\nabla\times E) = k_0^2\,\varepsilon_r\,E ,
\qquad \omega_n(k) = c_0\,k_{0,n}(k),
$$

whose eigenvalues as functions of $k$ are the bands; the usual units are the normalised
frequencies $\omega a/(2\pi c_0) = k_0 a/(2\pi)$. `physics::BandStructure` assembles the
stiffness and mass matrices (relative tensors, real and symmetric for lossless materials)
and the discrete gradient $G$ once; per wave vector it builds the Bloch constraints of the
Nédélec space *and* of the H1 space (`bloch_constraints` for `DofMap`: the scalar trace of
the shifted master functions, same hierarchical projections on both sides), reduces the
pencil with the prolongation $P$ of the Nédélec constraints,

$$
S_k = P^H S P, \qquad M_k = P^H M P, \qquad
G_k = (P^H P)^{-1} P^H\,G\,P_{\mathrm{H1}} ,
$$

and removes PEC DoFs if walls are present. $S_k$ and $M_k$ are complex Hermitian; $G_k$ is the
discrete gradient of the Bloch-periodic H1 space, i.e. the exact kernel of $S_k$ (for
$k \neq 0$ the constant field is no longer a gradient of a periodic function and the lowest
band leaves zero). The eigenpairs come from `solvers::complex_eigenpairs_near_gauged`, the
complex shift-invert Arnoldi iteration of the resonance solver with every Krylov vector
projected onto the $M_k$-orthogonal complement of the gradients,

$$
\Pi = I - G_k\,(G_k^H M_k G_k)^{-1} G_k^H M_k ,
$$

so the eigenvalues at zero never appear and no spurious modes are produced — the same gauge
as for the real cavity solver, in complex arithmetic. The shift sits below the lowest band
($\sigma = -(2\pi/a)^2$ by default), the result lists $k_0$ of the lowest bands in ascending
order with the Arnoldi residuals; `path(corners, segments)` walks a polyline through the
Brillouin zone (Γ–X–M–Γ for the square lattice).

**Verification** (`tests/convergence/empty_lattice_bands.cpp`): on the empty lattice (vacuum
unit cell) the bands are the folded free-space dispersion $k_0 = |k + G|$ over the reciprocal
lattice vectors $G = 2\pi(m, n)/a$. At a generic wave vector the four lowest bands converge
exponentially under p-refinement (maximum relative errors $2.9\cdot10^{-2}$,
$4.8\cdot10^{-3}$, $2.0\cdot10^{-4}$, $4.4\cdot10^{-6}$ for $p = 1..4$ on $4\times4$ cells) and
with rate $2p$ under h-refinement ($2.0$ for $p = 1$, $3.9$ for $p = 2$). The unit tests
check the zero of the lowest band at Γ (twofold: both polarisations, nothing else at zero),
the symmetry $\omega(-k) = \omega(k)$, that dielectric rods ($\varepsilon_r = 8.9$) lower the
bands, the H1 Bloch constraints on the interpolant of a Bloch function, and the gauged solver
against a dense reference (kernel skipped, eigenvectors $B$-orthogonal to the kernel).

## Time domain (`physics/time_domain.hpp`)

The transient solver integrates the second-order wave equation for the electric field,
obtained from Faraday's and Ampère's laws with a conductivity $\sigma$ and the current
density $J(x, t) = J(x)\,g(t)$,

$$
\varepsilon\,\partial_t^2 E + \sigma\,\partial_t E + \nabla\times(\mu^{-1}\nabla\times E)
= -\partial_t J ,
$$

in SI units with real $\varepsilon = \varepsilon_0\varepsilon_r$ and $\mu = \mu_0\mu_r$
(dispersive materials would need auxiliary differential equations and are not covered). The
Nédélec discretisation gives the system $M\ddot u + C\dot u + S u = f(t)$ with the mass
matrix $M$ (weight $\varepsilon$), the stiffness matrix $S$ (weight $\mu^{-1}$), the load
$f(t) = -g'(t)\,(J, \phi_i)$ and the damping matrix $C$, which collects the conductivity
mass (weight $\sigma$) and the boundary term of the **first-order Silver–Müller absorbing
condition** $n\times\mu^{-1}\nabla\times E = -Z^{-1}\partial_t E_t$: the tangential boundary
mass $\int_\Gamma Z^{-1}\phi_{i,t}\cdot\phi_{j,t}\,\mathrm dS$ with the wave impedance
$Z = \sqrt{\mu/\varepsilon}$ of the adjacent cell. It absorbs normally incident waves and
reflects a few percent at oblique incidence (a time-domain PML is future work). PEC facets
remove their tangential DoFs; the state vectors keep full length with zeros there.

Time stepping is the implicit **Newmark-β** scheme: with the predictors
$\tilde u = u + \Delta t\,v + \Delta t^2(\tfrac12 - \beta)\,a$ and
$\tilde v = v + \Delta t(1 - \gamma)\,a$ the new acceleration solves

$$
\big(M + \gamma\Delta t\,C + \beta\Delta t^2 S\big)\,a^{n+1}
= f(t^{n+1}) - C\tilde v - S\tilde u ,
\qquad u^{n+1} = \tilde u + \beta\Delta t^2 a^{n+1},
\quad v^{n+1} = \tilde v + \gamma\Delta t\,a^{n+1},
$$

and the operator is factorised once (direct solver backend of `solvers::LinearSolver`),
so a step costs one forward/backward substitution. The default $\beta = 1/4$, $\gamma = 1/2$
is the trapezoidal rule: unconditionally stable, second order, with a phase error of
$(\omega\Delta t)^2/12$ per radian and exact conservation of the discrete energy
$\tfrac12(v^T M v + u^T S u)$ of the undamped system, so $\Delta t$ is chosen for
accuracy (about 20 steps per period for 1 % phase error per period). $\gamma > 1/2$ (with
$\beta = (\gamma + 1/2)^2/4$) adds numerical damping of the high modes. The initial
acceleration is the consistent one, $M a^0 = f(t_0) - C v^0 - S u^0$. Sources are
`TimeSignal`s with analytic derivatives (`gaussian_pulse`, `modulated_gaussian`), and
`TimeDomain::run` calls an observer after every step for probes, energies or snapshots
(`assembly::evaluate_hcurl` on `state.u`).

**Verification** (`tests/convergence/time_domain_cavity.cpp`,
`tests/unit/physics/test_time_domain.cpp`): the TE$_{11}$ mode of the PEC unit square,
$E(x, t) = \nabla\times(\cos\pi x\cos\pi y)\cos\omega t$ with $\omega = c_0\pi\sqrt2$, is
an exact solution. Started from its interpolant with zero velocity and evaluated after
$2.25$ periods (where the phase error is fully visible), the relative $L^2$ error converges
with order $2$ in $\Delta t$ ($1.1\cdot10^{-1}$, $2.9\cdot10^{-2}$, $7.3\cdot10^{-3}$ for
$20, 40, 80$ steps per period on $\mathrm{ND}_5$) and with order $p$ in $h$ at $1600$ steps
per period ($1.9$ for $p = 1$, $2.3\ldots3.2$ for $p = 2$); the energy drift stays at
$10^{-15}$. The unit tests check the signals against finite differences, the setup
validation, the monotone energy decay with conductivity and with $\gamma > 1/2$, the 3D
cavity, and a modulated pulse radiated by a current sheet in a strip with absorbing ends,
whose energy drops below $2\,\%$ of its peak once the pulse has left (and stays with PEC
ends).

## Post-processing quantities

Implemented in `physics/postprocess.hpp`:

- **Surfaces** are sets of mesh facets with an inside cell each (`Surface::around_cells`
  for the interface of a tagged region, `Surface::boundary` for a tagged boundary,
  `whole_boundary`); `surface_quadrature` maps Gauss rules through the cell geometry, so
  curved facets get their curved measure and outward normal.
- **Poynting flux** $\int_S \tfrac12\operatorname{Re}(\mathbf{E}\times\bar{\mathbf{H}})\cdot\mathbf{n}\,ds$
  with $\mathbf{H} = (i\omega\mu_0\mu_r)^{-1}\nabla\times\mathbf{E}$ of discrete, analytic or
  combined fields (`poynting_flux`); in 2D the flux is per unit length.
- **Absorbed power** $P_{\mathrm{abs}} = \tfrac{\omega\varepsilon_0}{2}\int \operatorname{Im}(\varepsilon_r)|\mathbf{E}|^2\,dx$
  (`absorbed_power`).
- **Cross-sections** of a `Scattering` solution on a closed surface around the scatterer:
  $\sigma_{\mathrm{sca}} = P_{\mathrm{sca}}/I$ from the outward flux of the scattered field,
  $\sigma_{\mathrm{abs}} = -P_{\mathrm{tot}}/I$ from the inward flux of the total field,
  $\sigma_{\mathrm{ext}} = \sigma_{\mathrm{sca}} + \sigma_{\mathrm{abs}}$, with the incident
  intensity $I = |E_0|^2/(2Z)$, $Z = Z_0\sqrt{\mu_r/\varepsilon_r}$ of the background
  (`cross_sections`, `plane_wave_intensity`).

Verified by unit tests: surface measures and outward normals (also on the curved sphere),
the flux of an analytic plane wave through a box side equals $I\cos\theta$ times the side
and vanishes through the closed boundary (2D and 3D), the discrete plane-wave solution
reproduces it, and for a lossy disc in a PML box the flux-based absorption agrees with the
volume integral of the total field (energy balance).

- **Far field** (`physics/farfield.hpp`): on a closed surface $S$ in the homogeneous
  background that encloses all sources and scatterers, the equivalent currents
  $\mathbf{J} = \mathbf{n}\times\mathbf{H}$, $\mathbf{M} = -\mathbf{n}\times\mathbf{E}$ radiate
  (Stratton–Chu / Huygens). With $\mathbf{N} = \int_S \mathbf{J}e^{-ik\hat r\cdot x'}ds'$ and
  $\mathbf{L} = \int_S \mathbf{M}e^{-ik\hat r\cdot x'}ds'$ the pattern is
  $\mathbf{F} = \tfrac{ik}{4\pi}[Z\mathbf{N}_t - \hat r\times\mathbf{L}]$ in 3D
  ($\mathbf{E}\approx\mathbf{F}e^{ikr}/r$) and $\mathbf{F} = -\tfrac{k}{4}\sqrt{2/(\pi k)}\,
  e^{-i\pi/4}[Z\,\mathbf{N}\cdot\hat t + L_z]\,\hat t$ in 2D ($\mathbf{E}\approx\mathbf{F}e^{ik\rho}/\sqrt\rho$,
  $\hat t = \hat z\times\hat r$); `FarField::pattern`, `radiated_power` ($\int|F|^2 d\Omega/2Z$)
  and `scattering_cross_section`. Verified against the closed-form dipole far fields
  $\mathbf{p}_t/(4\pi)$ (3D) and $\tfrac{i}{4}\sqrt{2/(\pi k)}e^{-i\pi/4}\mathbf{p}_t$ (2D) sampled
  on curved spheres / circles, the radiated power against the Poynting flux, and the Mie
  cylinder's far-field cross-section against the flux-based one.
- **Diffraction orders** (`physics/diffraction.hpp`, 2D): on a line $x = x_0$ in a
  homogeneous region of a Bloch-periodic problem the field is $\sum_m A_m e^{ik_{y,m}y}$,
  $k_{y,m} = k_{y,0} + 2\pi m/a$; `fourier_coefficients` samples the field with composite
  Gauss rules along one period and `diffraction_efficiencies` gives
  $\eta_m = \mathrm{Re}(k_{x,m})|A_m|^2/(k_x^{inc}|E_0|^2)$ with $k_{x,m} = \sqrt{k^2n^2 - k_{y,m}^2}$
  (evanescent orders: $\eta_m = 0$). Verified with a Bloch plane wave (only the zeroth order,
  efficiency one, analytically and from the discrete solution) and by convergence test #6
  (`tests/convergence/lamellar_grating.cpp`): a lamellar grating (period 1, fill 0.5,
  thickness 0.5, ridge index 2 on a substrate of index 1.5, $\lambda = 0.8$, normal
  incidence) in a Bloch-periodic unit cell with PML in $\pm x$, scattered-field
  formulation; the reflected orders come from the scattered field above the grating and
  the transmitted ones from the total field below. The reference is an RCWA for the
  $H_z$ polarisation with Li's factorisation rules (Laurent's rule for $\varepsilon^{-1}
  \partial_x H$, the inverse rule for $\varepsilon^{-1}\partial_y H$), written in the test and
  checked by energy conservation ($\sum R + T = 1$ to $10^{-6}$) and truncation
  independence; the FEM efficiencies converge to it under p-refinement (maximal
  deviation $1.1\cdot10^{-1}$, $3.1\cdot10^{-2}$, $2.0\cdot10^{-3}$ for $p = 1, 2, 3$ on eight
  cells per unit length, energy sum $1.002$ at $p = 3$).

Planned: Purcell factor $F_P = P_{\mathrm{emitted}}/P_{\mathrm{bulk}}$ for a point dipole.

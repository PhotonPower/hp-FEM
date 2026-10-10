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
- **Transparent**: PML, see [pml.md](pml.md). A scattered-field source that extends into
  the PML (a substrate modelled as a scatterer) leaves an error of the order of 1e-3 in
  the efficiencies; model substrates as a [layered background](#layered-background-physicslayer_stackhpp-adr-0009).
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

**Non-matching faces (M15 F16).** The two sides need not be meshed identically. The facets
of a periodic direction are grouped by overlap after the shift, and each group is coupled as
a conforming constraint: the *coarser* facet of the group carries the trace of the coupled
space, truncated to the lowest polynomial order $p^\ast$ occurring in the group (the minimum
rule of the hp spaces), its surplus hierarchical modes are constrained to zero, and every DoF
of the finer facets is the interpolation of that trace (as for hanging nodes), so that the
discrete space is exactly the Bloch-periodic subspace. This is exact for identical facets with
different orders, for facets refined on one side only (nested, either side finer) and for
their combinations; the direction of the constraint follows the coarser side, coincident
vertices and edges keep the default direction slave $= e^{ik\cdot a}\,$master so that no cycles
arise, and the Bloch phase enters as $e^{-ik\cdot a}$ when the master side is the finer one.
Facets that are neither identical nor nested (unrelated meshes on the two sides, sheared
lattices) fall back to interpolating the master trace from the master cells under each point,
which is exact only up to the slave's order (a warning is logged once). `AdaptiveMesh::set_periodic`
(identical faces by mirrored refinement) is therefore optional. `assembly::PeriodicLocator`
answers, for a point on a slave facet, which master cell lies under its shifted image, with the
phase; the residual estimators use it to include the jump across the Bloch faces
([error-estimation.md](error-estimation.md)). Verified in
`tests/unit/assembly/test_periodic_nonmatching.cpp` (interpolants of Bloch functions obey the
constraints to $10^{-10}$, the solution's tangential trace is continuous up to the phase, the
mixed-order space is no worse than the uniform space of the common order) and by case (d) of
`conical_grating_hp` ([validation.md](../validation.md)).

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
(`s_parameters` factorises the operator once — the modal boundary terms do not depend on
the excitation — and solves once per channel with unit incoming amplitude on that channel,
`ScatteringOperator::solve_port`; on the 3D coupler of `examples/directional_coupler_3d`
that is 47 s instead of 167 s for four channels on cuDSS).

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
require the total-field formulation and lossless materials on the port.

**3D port modes.** The port facets form a planar cross-section. It is extracted as a 2D
mesh in the frame $(t_1, t_2, n)$ ($n$ the outward normal, $t_1$ the global axis least
aligned with $n$ projected onto the plane, $t_2 = n\times t_1$, so $t_1\times t_2 = n$ and
the cross-section's $z$ is the outward direction), with the cell tags and orders of the
inside cells and a PEC rim, and its guided modes $(E_t + \hat n E_z)e^{i\beta\xi}$ come from
`PropagatingMode` (propagating modes only; the solver drops the gradient kernel at
$\beta^2 = 0$ by a relative threshold). The transverse curl of such a mode is
$(\nabla\times E)_t = (\nabla_tE_z - i\beta E_t)\times\hat n$, hence

$$
\hat w_m = \hat n\times\mu_r^{-1}\nabla\times E = \frac{\nabla_tE_z - i\beta_m E_t}{\mu_r},\qquad
h_t = \frac{(\nabla\times E)_t}{i\omega\mu_0\mu_r},\qquad
P_m = \tfrac12\,\mathrm{Re}\int_\Gamma(\hat e_m\times h_t^*)\cdot\hat n\,dS ,
$$

and $q_{m,i} = \int_\Gamma\phi_i\cdot\hat w_m\,dS$, $N_m = \int_\Gamma\hat e_m\cdot\hat w_m\,dS$
are integrated with the surface quadrature of the port facets (3D Nédélec basis of the
inside cell at the quadrature point, mode fields of the section at its frame coordinates).
The 2D relation $\hat w = i\omega\mu_0 h$ is the special case of this formula (with the mode
equation $h'' = (\beta^2 - k_0^2\varepsilon_r\mu_r)h$). The sign of a 3D mode is fixed by the
largest Cartesian component of its mean transverse field $\int E_t\,dS$ (of the first moment
about the port centre for modes with vanishing mean), again independent of the port
orientation. The modes' unit amplitude is the unit 2-norm of the `PropagatingMode`
coefficients; the S-parameters are power-normalised and do not depend on it.

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
the same order as the 2D field). In 3D (`tests/convergence/waveguide_port_3d.cpp`) the PEC
rectangular waveguide $2\times1$ at $k_0 = 2.5$ (TE$_{10}$ only) has the port mode with
$\beta = \sqrt{k_0^2 - (\pi/a)^2}$ ($2\cdot10^{-3}$ at $p = 2$ on the coarse section), the
mode field $\propto\hat y\sin(\pi x/a)$ with the same sign on both ports, and the section of
length 2 transmits with $S_{21} = e^{i\beta L}$: on a fixed $4\times2\times4$ box the error of
$S_{21}$ decays $4\cdot10^{-2}$, $3.7\cdot10^{-3}$, $6.8\cdot10^{-5}$ and $|S_{11}|$
$4\cdot10^{-2}$, $3.6\cdot10^{-4}$, $1.7\cdot10^{-6}$ for $p = 1, 2, 3$.

## Conical incidence and the E_z polarisation (`physics/conical_scattering.hpp`)

A structure invariant in $z$ under a plane wave with the wave vector $(k_x, k_y, \beta)$ has a
field of the form $E = (E_x, E_y, E_z)(x, y)\,e^{i\beta z}$ for every source with that
$z$-dependence: the longitudinal wavenumber $\beta$ is conserved and the problem is
two-dimensional (2.5D, M13). The in-plane components live in the Nédélec space and the
scaled longitudinal component $v = -iE_z$ in the H1 space of the same order, exactly as in
the axisymmetric solver with $\beta$ in place of the azimuthal order and without the
$r$ weights (`assembly/conical_forms.hpp`). With the test functions carrying $e^{-i\beta z}$
and $V_z = -iw$, the curl of the mode

$$
\nabla\times E = \big(\,i(\partial_y v - \beta E_y),\ -i(\partial_x v - \beta E_x),\
\nabla_t\times E_t\,\big)
$$

gives real-structured forms with diagonal tensors in $(x, y, z)$,

$$
a(E, V) = \int \mu_x^{-1}(\partial_y v - \beta E_y)(\partial_y w - \beta V_y)
+ \mu_y^{-1}(\partial_x v - \beta E_x)(\partial_x w - \beta V_x)
+ \mu_z^{-1}(\nabla_t\times E_t)(\nabla_t\times V_t)\,dx\,dy ,
\qquad
b(E, V) = \int \varepsilon_xE_xV_x + \varepsilon_yE_yV_y + \varepsilon_z vw\,dx\,dy ,
$$

real symmetric for lossless media and complex symmetric otherwise; the gradient of a
potential, $(\nabla_t\psi,\ v = \beta\psi)$, spans the kernel of $a$ exactly
(`conical_gradient`). Sources are paired as $\int f_xV_x + f_yV_y + f_vw$ with the scaled
$f_v = -if_z$. The 2D PML enters as the tensors $\Lambda = \mathrm{diag}(s_y/s_x, s_x/s_y,
s_xs_y)$, $\tilde\varepsilon = \varepsilon\Lambda$, $\tilde\mu^{-1} = \mu^{-1}\Lambda^{-1}$
(`conical_pml_form`), i.e. $\tilde\varepsilon_{zz} = \varepsilon s_xs_y$ and the in-plane
$\tilde\mu_t^{-1} = \mu^{-1}\mathrm{diag}(s_x/s_y, s_y/s_x)$ for the longitudinal block.

**The E_z polarisation.** At $\beta = 0$ the coupling terms vanish and the system splits
into the in-plane block of `Scattering<2>` (the $H_z$ polarisation) and the scalar block
$\int \mu_t^{-1}\nabla v\cdot\nabla w - k_0^2\varepsilon_z vw$ of $E = E_z\hat z$, the "TE"
of the grating literature (E parallel to the lines): an incident field with only a
$z$-component keeps the in-plane coefficients at exactly zero, so `ConicalScattering` with
$\beta = 0$ is the E_z solver (the spec's `ScatteringEz`); the project files name the
polarisations by the field component, `Ez` and `Hz`.

**Problem class.** `ConicalScattering(transverse, longitudinal, setup)` takes ω, β, the
materials, PEC facets (tangential in-plane DoFs and $E_z$ eliminated), Bloch-periodic pairs
(constraints on both spaces), the PML box and either the incident field of the background
(scattered-field formulation, source $k_0^2(\varepsilon - \varepsilon_{bg})E^{inc}$ in the cells
whose permittivity deviates; μ must equal the background's) or a current
$f = i\omega\mu_0J$ (total field); hanging nodes of locally refined meshes are constrained.
`conical_plane_wave(E_0, k)` gives the scaled components of a plane wave ($E_0\perp k$),
`conical_polarisation(k, n̂, s|p)` the unit amplitudes with respect to the plane of incidence
spanned by $k$ and the structure normal, and `layered_conical_wave(stack, k_0, θ, φ, s|p)` the
field of a plane wave on a `LayerStack<2>` (normal along $y$) at the azimuth φ about the
normal, taken from the 3D stack solution with $\beta = k_0n\sin\theta\sin\varphi$, together
with the downward incident wave alone (separated by sampling the stack field at two heights)
for the extraction of reflected orders. Fields are evaluated as physical vectors
$(E_x, E_y, E_z = iv)$; `conical_poynting_flux` integrates $\tfrac12\mathrm{Re}(E\times H^*)\cdot n$
per unit length with the conical curl, `conical_fourier_coefficients` samples a 3-vector field
along any line and `conical_diffraction_efficiencies` gives
$\eta_m = \mathrm{Re}(k_{n,m})|A_m|^2/(k_n^{inc}|E_0|^2)$ with
$k_{n,m} = \sqrt{k_0^2n^2 - k_{t,m}^2 - \beta^2}$ and the complex vector amplitudes $A_m$.
**Scalar E_z path** (`ConicalScatteringSetup::scalar_ez`). At $\beta = 0$ the block system
decouples and an E_z-only excitation (the s polarisation at $\varphi = 0$, or a current
along $z$) leaves the in-plane block without a source, so its solution is zero. With
`scalar_ez` the solver assembles the block system as before but extracts, constrains
(hanging nodes and Bloch phases of the H1 space) and factorises only the H1 block, about a
third of the DoFs: the discrete solution is identical (unit test: $10^{-10}$ on a disc and on
a Bloch strip; the E_z Mie test prints both factorisation times), the in-plane coefficients
are returned as zeros, and every post-processing, estimator and sampler applies unchanged.
The constructor checks $\beta = 0$ and that the excitation has no in-plane components at the
cell centroids. `hpfem.grating.solve` takes the path automatically for s at $\varphi = 0$.
This replaces the separately planned scalar `ScatteringEz` solver.

**Post-processing of isolated and periodic structures** (`physics/conical_postprocess.hpp`,
M15 F10 / F11). `conical_diffraction_orders(field, line, k_0, n, k_{t0}, β, k_n^{inc}, incident)`
takes the orders of `field − incident` on any `OrderLine` with the composite Gauss–Legendre
rule of `conical_fourier_coefficients`; `to_literature_frame` maps a solver-frame vector
$(v_x, v_y, v_z)$ to the grating literature's $(x', y', z') = (x, -z, y)$ (period, invariant,
normal). `conical_power_balance` is the flux-based balance (incident power
$\tfrac12|E_0|^2 k_n^{inc}a/(k_0Z_0)$ per period, the reflected field = total minus the
downward wave through a line in the cover with H of the analytic wave by central differences,
the total field through a line in the substrate, the volumetric absorbed power), whose
relative residual is the reference-free quality indicator. `conical_cross_sections` gives
$\sigma_{sca}$ from the flux of the scattered field through a closed surface in the lossless
background, $\sigma_{abs}$ from the volumetric absorbed power and $\sigma_{ext}$ as their sum,
per unit length. `ConicalFarField` is the far field of the 2.5D scattered field: every
Cartesian component of $E(x,y)e^{i\beta z}$ radiates as a cylindrical wave with the transverse
wavenumber $k_t = \sqrt{k^2 - \beta^2}$, and integrating the 3D Stratton–Chu far field along z
gives, with $\hat k = (k_t\cos\varphi, k_t\sin\varphi, \beta)/k$, the equivalent currents
$N = \oint (n\times H)e^{-ik_t\hat\varphi\cdot x'}ds$, $L = -\oint (n\times E)e^{-ik_t\hat\varphi\cdot x'}ds$
(in-plane normal $n$) and $N_\perp = N - (N\cdot\hat k)\hat k$,

$$
E \to F(\varphi)\,\frac{e^{ik_t\rho}}{\sqrt\rho}\,e^{i\beta z}, \qquad
F = \frac{k}{4}\sqrt{\frac{2}{\pi k_t}}\,e^{-i\pi/4}\big(\hat k\times L - Z\,N_\perp\big),
\qquad P = \frac{k_t}{k}\oint\frac{|F|^2}{2Z}\,d\varphi .
$$

At $\beta = 0$ with an in-plane field it reproduces `FarField<2>` to $10^{-8}$; the radiated
power equals the flux of the scattered field through the surface (checked at $\beta = 0.4k$
with all three components, `test_conical_postprocess.cpp`); the convergence test
`conical_cross_sections` drives the Mie cylinder ($kR = 1.5$, $n = 1.5$) in the $E_z$ and the
in-plane polarisation to the series widths by flux and far field to $10^{-4}$.

`estimate` and `error` make the solver part of the hp loop
([hp-adaptivity.md](hp-adaptivity.md#conical-incidence), estimator in
[error-estimation.md](error-estimation.md#conical-incidence-adaptivityconical_residual_estimate)).

**A caveat for E_z fluxes.** The Poynting flux of the longitudinal block needs the normal
derivative of the H1 field, $H_t \propto \partial_nE_z$. On a material interface the
discrete normal derivative is only weakly continuous and converges one order slower than
the field; its error there is far larger than in the interior (on the E_z Mie cylinder below,
3.6 % at $p = 3$ on the cylinder surface while the field is accurate to $5\cdot10^{-4}$ and the
flux one cell away to $4\cdot10^{-5}$). Take flux surfaces through homogeneous cells, e.g.
`Surface::around_cells` of a tagged region one cell away from the scatterer. The in-plane
(Nédélec) block does not share this problem: its $H_z$ is the discrete curl.

**Verification** (`tests/unit/physics/test_conical_scattering.cpp`,
`test_conical_layered.cpp`, `tests/convergence/conical_mie_cylinder_ez.cpp`,
`conical_lamellar_grating_ez.cpp`): the stiffness annihilates the gradient of every
potential and the blocks decouple at $\beta = 0$; manufactured solutions ($E_z = \sin\pi x
\sin\pi y$ with a current, and the curl-free mode $\nabla(\psi e^{i\beta z})$ at $\beta = 1.3$)
converge in $p$; the PML tensors follow the stretch factors; a flat interface under conical
incidence (θ = 35°, φ = 50°, s and p, Bloch cell with PML) has a vanishing scattered field
and reflected / transmitted zeroth orders equal to the stack's R and T to $10^{-6}$. The E_z
Mie cylinder ($n = 1.5$, $ka = 1.5$) reaches the series scattering width
$Q_{sca} = (2/x)\sum_n|b_n|^2$ to $4\cdot10^{-5}$ at $p = 3$ (flux through a surface in the
vacuum) with the in-plane block exactly zero, and the E_z lamellar grating (period 1, fill
0.5, thickness 0.5, $n_g = 2$ on $n_2 = 1.5$, λ = 0.8, 20°) with the substrate as layered
background converges to an RCWA written in the test (the Fourier modal method of the scalar
equation, checked by energy conservation and truncation independence).

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

### Conical resonances (`physics/conical_resonance.hpp`)

The same eigenproblem for the 2.5D field of the conical solver (M15 F14): with the block
forms of `assembly/conical_forms.hpp`,

$$
S(\beta)\,(e, v) = k_0^2\,M\,(e, v), \qquad k_0^2 = \omega^2/c_0^2 \in \mathbb{C},
$$

with the longitudinal wavenumber $\beta$, PEC walls on both spaces, Bloch constraints along
the period (the phase of the pair is the Bloch wavenumber $k_x$) and the PML designed at the
target frequency. At $\beta = 0$ the pencil decouples into the in-plane ($H_z$) and the $E_z$
family, so one call returns the resonances of both polarisations of a grating unit cell; at
$\beta \ne 0$ the modes are conical. The gradient kernel of $S(\beta)$ is spanned by
$K_\beta = [G;\ \beta I]$ applied to the (PEC-free, Bloch-reduced) H1 potentials and is
projected out of the Krylov space (`solvers::complex_eigenpairs_near_gauged` with the reduced
kernel $\tilde K = (P^HP)^{-1}P^H K P_\psi$, as for the band structures), so a target near
$k_0 = 0$ or a large H1 space does not flood the result with eigenvalues at zero
(`ConicalResonanceSetup::remove_gradients`). `physics::ConicalResonance` returns the modes
ordered by the distance of $\omega$ to the target with the block coefficients; `field`,
`h_field` (with the mode's complex $\omega$) and `poynting` evaluate them, `sample_field` and
`triangulate_field` have overloads for modes. Checks: the PEC square waveguide at $\beta \ne 0$
reproduces $k_0^2 = \pi^2(m^2 + n^2) + \beta^2$ of both families (TE$_{10}$, TE$_{01}$,
TE$_{11}$, TM$_{11}$ to $2\cdot10^{-3}$ at $p = 3$ on $4\times4$ cells), a Bloch-periodic strip
the folded free-space modes $(k_x + 2\pi m/a)^2 + (n\pi/h)^2$, and the convergence test
`conical_resonance` drives the Fabry–Pérot resonance of a slab between PMLs ($n = 3.5$,
$Q = 10.7$) in $p$ to the floor of about $1.5\cdot10^{-4}$ set by the PML's reflection of the
outward-growing quasi-normal mode. The periodic-cell front end is
`hpfem.grating.resonances` / `bands` (docs/python.md).

### Resonance derivatives (`physics/eigen_sensitivity.hpp`)

A simple eigenvalue $\lambda = k_0^2$ of the reduced pencil $A x = \lambda B x$ moves with a
parameter $p$ by

$$
\frac{d\lambda}{dp} = \frac{y^\top(\partial_p A - \lambda\,\partial_p B)\,x}{y^\top B\,x},
\qquad A^\top y = \lambda B^\top y ,
$$

with the *left* eigenvector $y$, unconjugated: with PML and losses the pencil is
non-Hermitian. The element matrices are complex symmetric (diagonal PML stretches, scalar or
symmetric material tensors), so without Bloch phases $y = x$ — also with hanging nodes, whose
constraint coefficients are real (`resonance_adjoint`). The Bloch-reduced pencil
$P^H A P$ of `ConicalResonance` has the transpose $P^\top A \bar P$, the pencil at $-k_x$; its
left vector comes from two steps of transposed inverse iteration at $\lambda(1 + 10^{-10})$
(one factorisation, `conical_resonance_adjoint`, residual about $10^{-16}$), and in full
coefficients $\tilde y = \bar P y$, $x = P x_r$:
$d\lambda/dp = \tilde y^\top(\partial_p S - \lambda\,\partial_p M)\,x / (\tilde y^\top M x)$.
The parameters:

- permittivity of a tag: $\partial_\varepsilon M = M_{\text{tag}}$ (the stretched tensor at
  $\varepsilon = 1$ in the PML), holomorphic, so $d/d\,\mathrm{Im}\,\varepsilon = i\,d/d\varepsilon$;
- a mesh velocity (ADR-0011): one central directional difference of the element matrices
  $S_K - \lambda M_K$ per moving cell (helpers shared with the shape derivatives);
- β: central difference of the assembled pencil (absolute step, about $10^{-6}\max(|\beta|,
  k_0)$);
- the Bloch wavenumber $k_x$ (`conical_resonance_bloch_derivative`): the derivative of $P$
  from the problems at $k_x \mp h$, $d\lambda = [\tilde y^\top(S - \lambda M)\,\partial P x_r +
  (\overline{\partial P}\,y)^\top(S - \lambda M)\,x]/(\tilde y^\top M x)$ — the complex
  dispersion $d\omega/dk_x$ of a leaky mode.

Then $d\omega/dp = c_0\,(d\lambda/dp)/(2k_0)$, complex (the real part moves the resonance, the
imaginary part its width), $Q = \mathrm{Re}\,\omega/(-2\,\mathrm{Im}\,\omega)$ with
$dQ = (\mathrm{Re}\,\omega\,d\,\mathrm{Im}\,\omega - \mathrm{Im}\,\omega\,d\,\mathrm{Re}\,\omega)/
(2\,\mathrm{Im}^2\omega)$.

**Dispersive materials.** The resonance solvers evaluate $\varepsilon(\omega)$ once, at the
target. The resonance proper solves $\hat\lambda(\omega) = (\omega/c_0)^2$, where
$\hat\lambda(\omega)$ is the eigenvalue of the pencil with $\varepsilon$ at the mode's own
complex ω (the analytic continuation of Drude–Lorentz and constant models; the real part of ω
for tabulated data). `hpfem.grating.refine_resonance` finds it by Newton on
$f(\omega) = \hat\lambda(\omega) - (\omega/c_0)^2$ with
$f' = \sum_t (\partial\hat\lambda/\partial\varepsilon_t)\,\varepsilon_t'(\omega) - 2\omega/c_0^2$
(the material derivatives of the dispersive tags), one eigensolve and one adjoint per step.
At the self-consistent mode the dispersion enters the denominator,

$$
\frac{d\omega}{dp} = \frac{\partial_p\hat\lambda}{2\omega/c_0^2 - \partial_\omega\hat\lambda},
\qquad \partial_\omega\hat\lambda = \sum_t \frac{\partial\hat\lambda}{\partial\varepsilon_t}
\,\varepsilon_t'(\omega)
$$

(`resonance_derivative_from(omega, dlambda, dlambda_domega)`).

**Checks.** `test_eigen_sensitivity.cpp`: a lossy block in a PML box (`Resonance<2>`) — ε,
Im ε, Q and the growth of the block against re-solved modes to all six printed digits, also on
a hanging-node mesh; a Bloch strip with PML at $\beta \ne 0$ (`ConicalResonance`) — ε, β, $k_x$
and the shape against re-solved modes, the left vector's residual $8\cdot10^{-17}$ and not the
mode itself. `test_resonance_sensitivity.py`: the Fabry–Pérot slab with the exact
$k = (m\pi - i\ln\frac{n+1}{n-1})/(nd)$ — $d\omega/d\varepsilon$ and $d\omega/dd$ against the
exact derivatives (to the discretisation, $5\cdot10^{-3}$) and against re-solved resonances
($10^{-6}$, $10^{-5}$), β and $k_x$ against re-solved resonances (the reference's $O(h^2)$
falls to the eigensolver's noise of $10^{-5}$); a Drude–Lorentz slab: `refine_resonance` hits
the exact self-consistent resonance within the discretisation in a few Newton steps, and the
thickness derivative with the dispersion term matches refined resonances on moved meshes while
the frozen-pencil derivative does not.

## Sensitivities by the adjoint solve (`physics/sensitivity.hpp`)

For the discrete problem $A(\theta)\,e = b(\theta)$ and a linear goal $Q(e) = q^\top e$ (the
functionals of the goal-oriented estimator, no conjugation) the derivative with respect to a
parameter $\theta$ is

$$
\frac{dQ}{d\theta} = z^\top\Big(\frac{\partial b}{\partial\theta} - \frac{\partial A}{\partial\theta}\,e\Big),
\qquad A^\top z = q ,
$$

one adjoint solve per goal, any number of parameters. With hanging-node and Bloch constraints
the adjoint lives in the test space of the reduced system, $(P^\top A\bar P)\,z_r = P^\top q$,
$z = \bar P z_r$, with homogeneous Dirichlet data on the PEC and incident facets, exactly as
in `dwr_estimate` but on the primal space (`adjoint_solution`, `conical_adjoint_solution`).

**Material derivatives.** For the relative permittivity of all cells of a tag (outside the
PML) $\partial_\varepsilon A = -k_0^2 M_{\text{tag}}$ and, in the scattered-field
formulation, $\partial_\varepsilon b = k_0^2\,\ell_{\text{tag}}(E^{inc})$ with
$\ell_{\text{tag}}(v) = \int_{\text{tag}} E^{inc}\cdot v$, so

$$
\frac{dQ}{d\varepsilon_{\text{tag}}} = k_0^2\int_{\text{tag}} E_{\text{tot}}\cdot z\,dx
= k_0^2\,z^\top\big(M_{\text{tag}}\,e + \ell_{\text{tag}}(E^{inc})\big)
$$

(`material_sensitivity`, `conical_material_sensitivity`; the conical block mass pairs the
scaled longitudinal part as $\varepsilon_z v w$). The derivative is holomorphic in
$\varepsilon$, so one complex number gives the sensitivities to the real and the imaginary
part. It is the exact derivative of the *discrete* goal: `test_sensitivity.cpp` checks it
against central finite differences to $10^{-6}$ for the in-plane solver (disc with PML), the
conical solver at $\beta \ne 0$ and a Bloch-periodic strip with a current source; its
convergence to the continuous derivative follows that of the primal and adjoint solutions.
Quadratic observables follow by the chain rule from the complex amplitude: for the
efficiency $R_m = c\,|A_m|^2$ of an order the linearised goal $Q = A_m\cdot\bar A_m/|A_m|$
gives $dR_m = 2R_m\,\mathrm{Re}(dQ)/|A_m|$ (`hpfem.grating.sensitivity(result, tag, order)`
returns $\partial R_m/\partial\mathrm{Re}\,\varepsilon$ and
$\partial R_m/\partial\mathrm{Im}\,\varepsilon$, checked against finite differences of
`grating.solve` to $10^{-3}$).

### Shape derivatives

The geometry enters the discrete problem only through the node coordinates $x$ of the mesh
(vertices, and the edge nodes of second-order meshes), so the same adjoint gives the
derivative with respect to every node (ADR-0011, `physics/shape_sensitivity.hpp`):

$$
\frac{\partial Q}{\partial x_{n,d}} = \sum_{K\ni n} z_K^\top\Big(\frac{\partial b_K}{\partial x_{n,d}}
- \frac{\partial A_K}{\partial x_{n,d}}\,e_K\Big),
\qquad \frac{dQ}{dp} = \sum_{n,d}\frac{\partial Q}{\partial x_{n,d}}\,V_{n,d}
+ \Big(\frac{\partial q}{\partial x}\cdot V\Big)^{\!\top} e ,
$$

with the element contributions differentiated by central differences of the element
integrals (`shape_gradient`, `conical_shape_gradient`; step $10^{-6}$ cell diameters, in
parallel over the cells) and a geometry parameter $p$ represented by its mesh velocity
$V = \partial x/\partial p$ on the nodes (`region_normal_velocity` for the uniform normal
growth of a tagged region, `move_nodes` to apply a step). `shape_derivative` /
`conical_shape_derivative` add the derivative of the functional vector $q$ along $V$ (a point
value or a line integral changes with the cells it lives in) and return the exact derivative
of the discrete goal: `test_shape_sensitivity.cpp` compares it with finite differences of the
solve on moved meshes for the radius of a disc (in-plane and conical solver, second-order
mesh) and of a ball (3D). The discrete goal is smooth but, on coarse meshes with stretched
cells next to the interface, strongly nonlinear in the node motion (a shift of $10^{-4}$
changes those element matrices by per cents), so finite checks need steps well below the
local cell size; the derivative is the limit. This is the discrete counterpart of the
Hadamard formula, whose interface integrals of the coefficient jumps are the continuous limit.
`hpfem.grating.shape_sensitivity(result, velocity, order)` gives $dR_m/dp$ of an efficiency
for any node velocity (checked for the ridge height of the glass grating against finite
differences to $10^{-3}$).

One exception to the smoothness: a goal sampled on mesh facets. The discrete Nédélec field
is only tangentially continuous, its normal component jumps across facets, and a point on a
facet is evaluated in one of the two cells. When the velocity deforms the cells at such a
point, the sampled value is not differentiable in $p$ — the one-sided changes differ, and
a difference quotient grows like the jump over the step. This is the case for the order
amplitudes of a grating whose measurement line lies on a row of mesh facets (the default line
midway between the structure and the PML often does on structured cells): the Jacobian of a
p-polarised line grating along a morph velocity that reached the line was off by a factor of
ten for the line height. Lines that cross cells transversally are fine (a point stays in its
cell for small steps). `hpfem.grating.jacobian` / `shape_sensitivity` therefore refuse a
velocity that deforms a cell with a vertex on a measurement line, and
`hpfem.opt.Morph(..., band=(y_low, y_high))` keeps everything on and beyond the band edges
fixed, so that the measurement lines and the PML stay outside the deformation; with the band
the derivatives match finite differences of the solve to $10^{-8}$.

### Kept factorisation

`adjoint_solution(problem, q)` assembles and factorises the adjoint system anew. With
`setup.keep_factorisation = true` (`ScatteringSetup`, `ConicalScatteringSetup`,
`hpfem.grating.solve(..., keep_factorisation=True)`) the solve hands its factorised system to
the solution (`solution.factorisation`, a `physics::KeptFactorisation`, ADR-0012 §4), and
every further solve with the same operator is a pair of triangular solves:

- the **tangent** (direct) solve $s = A^{-1} r$ of a full-size residual $r$ — the solution
  change $de/d\theta = A^{-1}(\partial_\theta b - \partial_\theta A\,e)$ of one parameter;
- the **adjoint** solve $z = A^{-\top} q$ of a full-size functional vector — one goal.

The forward solve reaches the factorised matrix in fixed steps: static condensation of the
interior DoFs ($\tilde f_E = f_E - K_{EB}K_{BB}^{-1}f_B$), selection of the unknowns (the
conical solver drops its Dirichlet DoFs, the scalar $E_z$ path keeps the H1 block),
constraints ($P^H$), zero data on the eliminated Dirichlet unknowns; back through $P$ and the
interior recovery $u_B = K_{BB}^{-1}(f_B - K_{BE}u_E)$. The adjoint applies the transposes:

$$
\tilde q_E = q_E - (K_{BB}^{-1}K_{BE})^\top q_B,\quad
y_r = (P^H \tilde A P)^{-\top} P^\top \tilde q,\quad
z_E = \bar P y_r,\quad
z_B = K_{BB}^{-\top}(q_B - K_{EB}^\top z_E),
$$

the middle one a transposed solve on the forward factors (`LinearSolver::solve_transposed`;
the LDLᵀ paths solve with $A$ itself, see [solvers](solvers.md)). The result is the exact
transpose of the discrete solution operator, $q^\top s = z^\top r$ to round-off for every
pair, so the adjoint includes everything the solve included — ports, condensation, Bloch and
hanging-node constraints. Verified in `test_kept_factorisation.cpp` (disc with PML, PEC and
condensation at order 3; the Bloch strip of the conical solver; the scalar $E_z$ path against
the full block system): the identity to $10^{-10}$, the tangent solve against the assembled
equations, and adjoints and material sensitivities equal to the assembling path to
$10^{-9}$; `test_grating_solve.py` checks that `grating.sensitivity` and
`grating.shape_sensitivity` are unchanged. The factors stay in memory as long as the solution
lives (`estimate_memory` predicts their size); `keep_factorisation` is therefore opt-in.

### Direct mode

With a kept factorisation the derivative of the solution with respect to one parameter is a
single tangent solve, $de/d\theta = A^{-1} r_\theta$, with the residual derivative at fixed
coefficients

$$
r_\theta = \frac{\partial}{\partial\theta}\big(b - A e\big)\Big|_{e\ \text{fixed}},
\qquad \frac{dQ_k}{d\theta} = q_k^\top\frac{de}{d\theta}
+ \Big(\frac{\partial q_k}{\partial\theta}\Big)^{\!\top} e ,
$$

for every observable $k$ at once — the adjoint mode $z_k^\top r_\theta$ with one solve per
observable gives the same numbers. The residual derivatives (full size, on the DoF map; the
conical ones stacked as in-plane | scaled longitudinal):

- material: $r_\varepsilon = k_0^2\big(M_{\text{tag}}\,e + \ell_{\text{tag}}(E^{inc})\big)$
  (`material_residual_derivative`, `conical_material_residual_derivative`);
- shape: one central difference of the element residuals $b_K - A_K e_K$ per cell whose nodes
  move, along the mesh velocity $V$ with the largest node displacement $10^{-6}$ cell
  diameters (`shape_residual_derivative`, `conical_shape_residual_derivative`), and the
  functional term $\partial q/\partial x\cdot V$ from `functional_shape_derivative` /
  `conical_functional_shape_derivative` (as in `shape_derivative`). A directional difference
  per cell is $2\,\dim$ (nodes per cell) times cheaper than the node gradient of the adjoint
  mode.

The cheaper mode follows from the counts per factorisation: $n$ parameters cost $n$ tangent
solves, $m$ observables $m$ adjoint solves (`hpfem.grating.jacobian` picks
$\min(n, m)$; scatterometry with many orders and few parameters is the direct case).
Parameters that change the constraints — the frequency and the angles of a Bloch-periodic
problem, whose phases depend on $k_x$ — need the derivative of $P$ as well (next section).
`test_residual_derivative.cpp` checks the
direct against the adjoint mode for the disc permittivity (to $10^{-9}$) and the disc radius
(to $10^{-6}$, two different central differences) in the in-plane and the conical solver,
$z^\top r_V$ against the node-gradient pairing of ADR-0011, and a $2\times 2$ Jacobian in both
modes; `test_grating_solve.py` checks `grating.jacobian` in both modes against
`grating.sensitivity` and `grating.shape_sensitivity` entry by entry.

### Frequency and angle derivatives

The frequency $\omega$ and the angles $\theta$, $\varphi$ of the incident wave are parameters
of the whole conical setup (`physics/parameter_sensitivity.hpp`): they change $k_0^2$,
$\beta = k_z$, dispersive permittivities, the incident field of the scattered-field
formulation and the Bloch phases $e^{ik_x a}$ of the constraints $P$ — the solution space
itself. With the reduced coefficients $u$ (the constraint masters) the discrete equations are
$G(u,\theta) = P^H(b - A P u) = 0$; differentiating at the solution gives

$$
P^H A P\,\frac{du}{d\theta} = P^H\rho' + (\partial_\theta P)^H\rho_0,
\qquad
\frac{de}{d\theta} = P\,\frac{du}{d\theta} + (\partial_\theta P)\,u ,
$$

with the full residual of the solution $\rho_0 = b - Ae$ (only $P^H\rho_0$ vanishes; the rows
of the Bloch slaves carry the reaction of the periodic coupling) and the residual derivative
along the transported coefficients $\rho' = \tfrac{d}{d\theta}\big[b - A P u\big]_{u\ \text{fixed}}$.
`conical_parameter_tangent` takes the problems at $\theta\pm h$ on the same DoF maps and forms
$\rho'$, $(\partial_\theta P)^H\rho_0$ and $(\partial_\theta P)u$ as central differences (the
slaves recomputed with the neighbour's phases, `conical_transported_solution`; the residuals
by `conical_residual`, one assembly each). No solve is repeated: the tangent is one solve on
the kept factorisation with the extra term added after the constraints
(`KeptFactorisation::solve_many(loads, system_loads)`). The differences are of assembled
residuals, smooth in $\theta$ and free of solver noise, so $h = 10^{-6}$ (relative for
$\omega$) is accurate to $O(h^2)$; the round-off of the parallel assembly, amplified by
$1/2h$, leaves about $10^{-8}$ relative. An observable $Q(e,\theta)$ — a diffraction
efficiency depends on $\theta$ also through the order directions, the Fourier kernel and the
incident power — has

$$
\frac{dQ}{d\theta} = \partial_e Q\cdot\frac{de}{d\theta} + \partial_\theta Q\big|_e ,
$$

the explicit term again a central difference of the post-processing at fixed $e$.
`hpfem.grating.jacobian` takes `"theta"`, `"phi"`, `"omega"` and `"wavelength"` columns this
way (always in the direct mode; at a fixed PML box and fixed measurement lines; dispersive
materials given as models are evaluated at the new frequency). Without the constraint term
the tangent misses the change of the solution space and is wrong at the percent level
(checked in the test). `test_parameter_sensitivity.cpp` checks $de/dt$ for a parameter that
moves the Bloch phase, $\omega$, $\beta$ and the source at once against central differences
of full solves (to $3\cdot 10^{-7}$, the $O(h^2)$ of the reference) on the vector and the
scalar $E_z$ path; `test_grating_solve.py` checks the $\theta$, $\varphi$ and wavelength
columns of a conical grating with a dispersive ridge against differences of full solves at the
same PML box (deviations $10^{-6}$–$10^{-5}$, shrinking as $h^2$ of the reference).

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

### Band derivatives and group velocity

`physics/band_sensitivity.hpp` differentiates the bands with respect to material and shape
parameters and the Bloch wave vector without another eigensolve (M16 S4). At the wave vector
$k$ the reduced pencil $A(p)\,u = \lambda B(p)\,u$, $A = P^H S P$, $B = P^H M P$,
$\lambda = k_0^2$, is Hermitian for lossless media, so the left eigenvector of a simple
eigenvalue is the right one and the derivative is the Hellmann–Feynman quotient

$$
\frac{d\lambda}{dp} = \frac{u^H(\partial_p A - \lambda\,\partial_p B)\,u}{u^H B u},
\qquad
\frac{dk_0}{dp} = \frac{1}{2k_0}\frac{d\lambda}{dp},
\qquad
\frac{d\omega}{dp} = c_0\,\frac{dk_0}{dp} .
$$

The eigenvectors are needed: with `BandStructureSetup::keep_modes` the bands carry the
full-size modes $w = P u$ (Bloch slaves filled in, zero on PEC DoFs, $w^H M w = 1$); they are
off by default, since a long path would hold a dense block per wave vector. The parameters:

- **Permittivity / permeability of a tag** (`band_permittivity_derivative`,
  `band_permeability_derivative`): $\partial B = P^H M_{\text{tag}} P$ with the mass matrix of
  the tagged cells at $\varepsilon_r = 1$, i.e. $d\lambda/d\varepsilon_r = -\lambda\,
  w^H M_{\text{tag}} w$; for $\mu_r$, $\partial A = -\mu_r^{-2} P^H S_{\text{tag}} P$.
- **Shape** (`band_shape_derivative`, a mesh velocity $V$ as in ADR-0011): $\partial S$ and
  $\partial M$ by one central directional difference of the element matrices of every cell with
  a moving node (largest node displacement `relative_step` $= 10^{-6}$ times the cell
  diameter), projected on the modes cell by cell. The Bloch constraints do not depend on the
  geometry only if $V$ vanishes on the periodic faces; this is checked (vertices and edge
  nodes of the master and slave facets) and violated velocities are rejected.
- **Wave vector** (`band_wave_vector_derivative` along a direction $d$, `group_velocity` for
  the Cartesian directions): $S$ and $M$ do not depend on $k$, only $P$ does, so

  $$
  \partial_d A - \lambda\,\partial_d B
  = \partial_d P^H (S - \lambda M)\,P + P^H (S - \lambda M)\,\partial_d P ,
  $$

  projected as $Z^H(S - \lambda M)W + W^H(S - \lambda M)Z$ with $Z = \partial_d P\,U$ and the
  reduced coefficients $U$ (the values of $W$ on the unconstrained DoFs, where $P$ has identity
  rows). $\partial_d P$ is a central difference of the Bloch prolongation at
  $k \pm h\,d$, $h = 10^{-5}\cdot 2\pi/(a|d|)$: the entries are geometric coefficients times
  $e^{ik\cdot s}$ (s the lattice shift of the slave), so the relative truncation is
  $(h\,d\cdot s)^2/6 \approx 10^{-10}$; no solve, $2\,\mathrm{Dim}$ constraint builds for the
  group velocity $v_g = \nabla_k\omega = c_0\nabla_k k_0$.

**Degenerate bands.** Hellmann–Feynman does not apply to a multiple eigenvalue (folded bands,
high-symmetry points), and for a nearly degenerate pair it is ill-conditioned in the
eigenvectors. Consecutive bands whose eigenvalues differ relatively by at most
`degeneracy_tolerance` ($10^{-6}$ by default; bands at $k_0 = 0$ always) form a cluster $I$
with modes $W_I$, and its derivatives are the eigenvalues of the small Hermitian pencil

$$
W_I^H\big(\partial A - \tfrac12(\Lambda\,\partial B + \partial B\,\Lambda)\big)W_I\,y
= \mu\,W_I^H B W_I\,y ,
$$

assigned in ascending order to the bands of the cluster (`BandDerivative::multiplicity` gives
the cluster size). These are the one-sided derivatives of the branches leaving the degenerate
point as $p$ increases — the sorted eigenvalues of $\lambda(p + h)$ for $h \to 0^+$. For a
simple band the pencil is $1\times1$ and reduces to Hellmann–Feynman. For the group velocity
at a degenerate point every Cartesian component is the sorted cluster spectrum of its own
direction (the branches differ per direction), so branch velocities along a path need
`band_wave_vector_derivative` along that path. At $k_0 = 0$ (the light cone at Γ)
$k_0 = \sqrt\lambda$ is not differentiable: $d\lambda/dp$ is returned, $dk_0/dp$ is NaN.

**Verification** (`tests/unit/physics/test_band_sensitivity.cpp`, square lattice of rods of
radius $0.2a$, $\varepsilon_r = 8.9$, $8\times8$ squares, $p = 2$, four bands at
$k = (1.3, 0.6)/a$): against central differences of re-solved bands the derivatives agree to
$1.0\cdot10^{-8}$ ($\varepsilon_r$), $1.3\cdot10^{-8}$ ($\mu_r$), $7\cdot10^{-8}$ (rod radius by
`region_normal_velocity` against moved meshes; the deviation is the $O(h^2)$ of the reference:
$1.2\cdot10^{-6}$, $2.9\cdot10^{-7}$, $7.3\cdot10^{-8}$ for $h = 10^{-4}$, $5\cdot10^{-5}$,
$2.5\cdot10^{-5}$) and $5\cdot10^{-9}$ (wave vector along $(0.6, -0.8)$), relative to the
largest derivative. In a uniform medium $n = 1.5$ ($4\times4$ cells, $p = 4$) the group
velocities of the four lowest folded bands match $c_0 (k + G)/(n|k + G|)$ to
$3.9\cdot10^{-5}$ (the discretisation error of the bands). On the empty lattice at Γ the
fourfold band $k_0 = 2\pi/a$ forms one cluster whose $x$-derivatives are $-1, 0, 0, 1$ (the
branches $G = (-1,0)$, $(0,\pm1)$, $(1,0)$) to $10^{-5}$ and agree with one-sided differences
of the sorted bands; the twofold zero band gives NaN. A 3D smoke test (an inclusion of
$\varepsilon_r = 4$, the cells within $0.3a$ of the centre of a cubic cell, $p = 2$) checks $\varepsilon_r$ and $k_z$ derivatives to
$2\cdot10^{-8}$ and $5\cdot10^{-9}$.

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
  incidence) in a Bloch-periodic unit cell with PML in $\pm y$, scattered-field
  formulation on the layered background superstrate | substrate (`LayerStack`, so that the
  ridge alone is the scatterer and no source reaches into the PML — with the substrate as
  scatterer the efficiencies plateau at $\sim 2\cdot10^{-3}$); the reflected orders are those
  of the total field minus the incident wave above the grating and the transmitted ones
  those of the total field below (`diffraction_orders`). The reference is an RCWA for the
  $H_z$ polarisation with Li's factorisation rules (Laurent's rule for $\varepsilon^{-1}
  \partial_x H$, the inverse rule for $\varepsilon^{-1}\partial_y H$), written in the test and
  checked by energy conservation ($\sum R + T = 1$ to $10^{-6}$) and truncation
  independence; the FEM efficiencies converge to it under p-refinement (maximal
  deviation $1.1\cdot10^{-1}$, $3.1\cdot10^{-2}$, $2.0\cdot10^{-3}$ for $p = 1, 2, 3$ on eight
  cells per unit length, energy sum $1.002$ at $p = 3$).

- **Gratings on layered backgrounds** (M14-C, test report of 2026-10-05): `diffraction_orders`
  takes the orders on an `OrderLine` of any orientation (origin, tangent along the period,
  normal away from the structure) by composite Gauss–Legendre blocks (exact for the FEM
  field when they align with the cells) and subtracts an incident wave, so the reflected
  orders of a total field on a `LayerStack` background follow directly from
  `Scattering::incident_wave` (the incident plane wave alone, which `LayeredPlaneWave` now
  carries next to its full field); the result holds the complex vector amplitude of every
  order with the line origin as phase reference. `Surface::plane` makes a flux surface of the
  facets on a coordinate plane, and `power_balance` gives the energy balance of a periodic
  problem: incident power per period, the reflected flux of total minus incident wave, the
  transmitted flux and the absorbed power of the *total* field over the lossy cells
  (`absorbed_power(problem, solution)`, background included); its relative residual is a
  reference-free quality indicator of the efficiencies. `total_field` on an interface of the
  background evaluates the background branch of the side of the located cell, so the two
  one-sided limits are obtained from cells above and below. Verified
  (`tests/unit/physics/test_grating_postprocessing.cpp`,
  `tests/convergence/grating_postprocessing.cpp`): analytic quasi-periodic fields in four
  orientations to $10^{-12}$; the silicon lamellar grating of the report (400 nm period,
  200 × 148 nm ridge, 405 nm, 50°, in-plane $E$) with the stack background gives
  $R_0 = 0.143380$, $R_{-1} = 0.142383$ at $p = 5$ against the RCWA 0.143381 / 0.142382 —
  the report's own extraction had stalled at $+1.2\cdot10^{-4}$ in $R_{-1}$ — with a balance
  residual of $1.7\cdot10^{-6}$; the same grating in glass closes $R + T = 1$ to
  $2\cdot10^{-6}$ once the PML is designed for the steepest *propagating order* on its side
  (order $-2$ at 57° in the glass, not the 50° of the incident wave: designed for 50° the
  layer reflected $2.5\cdot10^{-5}$ of it and the balance stalled at $1.7\cdot10^{-5}$).

- **Field sampling and triangulated export** (`physics/field_sampling.hpp`, M15 F3): maps and
  line scans no longer need one call per point. `sample_field(problem, solution, locator,
  points, options)` evaluates the total or scattered field of a `Scattering<Dim>` or
  `ConicalScattering` solution at many points in parallel: the points are located first (in
  blocks), grouped by cell, and every cell evaluates its points with basis, geometry and
  coefficients set up once (affine cells reuse their Jacobian). Points outside the mesh along
  a Bloch-periodic direction of the setup are mapped back into the unit cell and the value
  carries the Bloch phase $e^{i n k\cdot a}$; points outside the mesh otherwise give NaN and
  no cell; `interface_side` resolves a point on a facet to the cell above or below along the
  last coordinate (the stack normal), so both one-sided limits of a discontinuous normal
  component are reachable. `triangulate_field(problem, solution, n)` returns the field on the
  $n$-fold subdivided mesh (`mesh::subdivide`, curved cells included): points, simplices,
  values per point, parent cell and tag per simplex, with points not shared between parent
  cells so that discontinuities across facets are preserved — the input of
  `matplotlib.tri.Triangulation` or of any viewer. The basis kernels
  (`fespace/detail/kernels.hpp`) keep their Legendre data in fixed-capacity arrays since this
  work, so a basis evaluation allocates nothing; this halved the cost of every point
  evaluation and quadrature loop. Measured on the maintainer's machine (16 threads, 2D,
  $p = 2$ and $4$): 50–60 ns per point against 2.5–5 µs for the Python loop over
  `total_field`, 50–80× faster; single-threaded 250–350 ns, 8–14×. Unit tests
  (`test_field_sampling.cpp`): values identical to the point-wise evaluation to $10^{-14}$,
  Bloch wrapping against the phase, NaN outside, interface side, the subdivision against
  `mesh::subdivide`, both solvers.

- **Absorbed power per tag, cell and quadrature point** (`physics/absorption.hpp`, M15 F4):
  `absorbed_power_by_tag(problem, solution)` integrates the Joule heating
  $Q = \tfrac{\omega\varepsilon_0}{2}\,\mathrm{Im}(\varepsilon_r)\,|E|^2$ of the *total* field
  (incident or background field included) over every lossy cell with rules of degree
  $2p + 2$ and returns the total, the power of every material tag and the power of every
  cell; `absorption_density` returns $Q$ at the quadrature points with their weights for
  maps and carrier-generation profiles. Both exist for `Scattering<Dim>` and
  `ConicalScattering` ($|E|^2 = |E_x|^2 + |E_y|^2 + |E_z|^2$ of the physical field). The
  volumetric quadrature replaces the raster integration of a field map, which is wrong by
  about a pixel at every material boundary (4–8 % in the GUI), and avoids the interface
  caveat of the E$_z$ Poynting flux. Divide by the incident power per period,
  `plane_wave_intensity(|E_0|, medium) cos θ · a`, for the absorptance. Verified
  (`test_absorption.cpp`): the absorptance of a flat lossy film on a layered background from
  the exact stack field to $10^{-6}$ (both solvers, the conical one at β ≠ 0 in s and p), the
  per-cell and per-point sums against the total, two lossy tags of a grating; the Si grating
  of `grating_postprocessing.cpp` closes $\sum_{\text{tags}} = 1 - R_0 - R_{-1}$ to $10^{-4}$.

- **Magnetic field and Poynting vector** (M15 F12): `ConicalScattering::h_field` returns
  $H = \nabla\times E/(i\omega\mu_0\mu_r)$ of the total field with the conical curl
  $(\partial_yE_z - i\beta E_y,\ i\beta E_x - \partial_xE_z,\ \partial_xE_y - \partial_yE_x)$
  of the unknown (`curl_field`) plus the curl of the incident field, and `poynting` the
  time-averaged $S = \tfrac12\mathrm{Re}(E\times\bar H)$ for energy-flow maps. The incident
  curl is analytic for the built-in waves — `conical_plane_wave_curl`, and
  `LayeredConicalWave::field_curl` / `incident_curl` from the 3D stack curl (the conical plane
  $(x, y, z) = (X, Z, Y)$ is a reflection of the stack's frame, so the pseudo-vector changes
  sign) — through `ConicalScatteringSetup::incident_curl`; without it `incident_curl` falls
  back to central differences of the incident field (relative accuracy about $10^{-8}$).
  `sample_field` / `triangulate_field` take a `SampledQuantity` (E, H or S) for both solvers
  (`Scattering<2>`: $H_z$ alone and the in-plane $S$; `Scattering<3>` and the conical solver:
  three components), so the GUI's maps of $|H|$ and of the energy flow come from the same
  vectorised path. Verified (`test_conical_h_field.cpp`): a plane wave in a homogeneous cell
  against $H = k\times E_0\,e^{ik\cdot x}/(\omega\mu_0)$ and $S = |E_0|^2\hat k/(2Z_0)$ to
  $10^{-9}$, the analytic curls against central differences, and on a lossy film stack the
  normal energy flow of the exact stack wave, $-(1 - R)\,I\cos\theta$ above and
  $-T\,I\cos\theta$ below, to $10^{-9}$ in s and p.

### Dipole emitters

A point dipole in a structure periodic in $x$ and invariant in $z$ (milestone M17, ADR-0013,
`docs/dipole-emitters-features.md`) is the M11 emitter: current moment $p$ [A m] smeared over
the normalised 3D Gaussian $g$ of width $\sigma$. Array scanning splits it into cell problems
of the conical solver with the source

$$
J_{k_x,\beta}(x, y) = p\,\frac{e^{-\rho^2/2\sigma^2}}{2\pi\sigma^2}\,e^{-\sigma^2\beta^2/2}
$$

(one per period with the Bloch phase $e^{ik_xP}$; the $z$-smearing is the Fourier factor of the
$\beta$ sample), `conical_gaussian_dipole`, and the single dipole's power is
$P_{em} = \frac{P}{2\pi}\int_{BZ}dk_x\,\frac1{2\pi}\int d\beta\,P_{cell}(k_x,\beta)$ with the
power delivered in one cell problem,
$P_{cell} = -\tfrac12\,\mathrm{Re}\int_{cell}\bar J_{k_x,\beta}\cdot E_{k_x,\beta}\,dA$
(`conical_source_power`, [W/m per unit $\beta$]).

**Stage A, the phased emitter array** (`hpfem.grating.emit`, M17 S1): one cell problem with the
geometry, materials, stack, PML and measurement lines of `grating.solve` (cells whose tag is not
in the material map take the stack material at their centroid), the PML designed for the
direction of $(k_x, \beta)$ in the cover. It reports $P_{cell}$, the power of every Floquet
order up and down, $P_m = P\,\mathrm{Re}(k_{n,m})\,|A_m|^2/(2k_0Z_0)$ from the order amplitudes
on the measurement lines, the Poynting fluxes through the PML boundaries, the absorbed power by
tag and the guided remainder $P_{cell} - $ fluxes $-$ absorbed. In a homogeneous medium of index
$n$ ($k = nk_0$) the array is a sum of current sheets, one per order $k_m = k_x + 2\pi m/P$, and

$$
P_{cell} = \sum_{m,\pm} \frac{\omega\mu_0\,|p_{\perp,m,\pm}|^2}{8P\,k_{y,m}}\,e^{-\sigma^2k^2},
\qquad k_{y,m} = \sqrt{k^2 - k_m^2 - \beta^2},
$$

summed over the propagating orders and both sides, with $p_\perp$ the part of $p$ normal to
$\hat k = (k_m, \pm k_{y,m}, \beta)/k$ (the Gaussian factor is the same for every order because
$k_m^2 + k_{y,m}^2 + \beta^2 = k^2$). `python/tests/test_grating_emit.py` checks $P_{cell}$, every
order and the fluxes against it for four dipole moments and five $(k_x, \beta)$ — at normal
incidence to $10^{-9}$, at 45° to $5\cdot10^{-3}$; beyond the light cone nothing is delivered —
and the energy balance (delivered = fluxes + absorbed to 1 %) on a lossy ridge. Near grazing
directions the PML limits the accuracy: at 67° a PML of 1 µm (in a medium of index 1.5 at
1 µm) still reflects 0.8 % back onto the source, 0.5 µm 1.7 %; the PML thickness, not the mesh,
has to grow there.

**Measurement lines and the PML near an anomaly.** The order powers measure the radiated far
field only if the lines see outgoing waves alone: `emit` places the default lines at least
$6\sigma$ beyond the dipole's Gaussian and refuses a given line that cuts it (a cover line through
the Gaussian had made the $m = 0$ power 8 % wrong and mesh-dependent). Near a Rayleigh anomaly an
order is barely evanescent ($\kappa = (k_m^2 + \beta^2 - k^2)^{1/2}$ small, e.g. 1/280 nm for a
400 nm period at 405 nm and $\beta = 0.2k_0$) and its near field reaches the PML; the PML then
exchanges power with the source through that order (an evanescent pair carries power), which
shifts the Poynting flux through the PML boundary and $P_{cell}$ by up to about 1 % for a PML
300 nm away, while the order powers do not change and are the converged radiation. `emit` takes
the guided remainder from the order powers ($P_{cell}$ − up − down − absorbed, ADR-0013 §2) and
reports the difference of fluxes and order powers as `pml_leak`, with a warning above
$10^{-3}P_{cell}$ (move the PML away from the source); in the flat glass case of the test it is
0.9 % with 296 nm of PML and below $10^{-3}$ with 740 nm. Both findings came from the helper
agent that wrote Stage C.

**Stage C, angle-resolved emission by reciprocity** (`hpfem.grating.emission_pattern`, M17
S2, ADR-0013 §6). Lorentz reciprocity between the emitter and a distant dipole $p_2 \parallel
\hat e$ in the direction $\hat r$ (medium of index $n$) gives the far-field amplitude of the
emitter in that direction and polarisation, $\hat e\cdot F = \frac{i\omega\mu_0}{4\pi}\,
p\cdot E_{pw}(x_0)$, with $E_{pw}$ the total field of the unit plane wave incident from
$\hat r$ (travelling along $-\hat r$) with polarisation $\hat e$; hence

$$
\frac{dP}{d\Omega}(\hat r, \hat e) = \frac{n k_0^2 Z_0}{32\pi^2}
\left|p\cdot\langle E_{pw}\rangle_g\right|^2 ,
\qquad \langle E_{pw}\rangle_g = e^{-\sigma^2\beta^2/2}\int g_2\,E_{pw}\,dA ,
$$

the field averaged over the dipole's Gaussian: in-plane by a tensor Gauss–Hermite rule on the
solved field, along $z$ by the Fourier factor of the direction's β. In a homogeneous medium
$E_{pw} = \hat e\,e^{ik\cdot x}$, the sum over s and p gives
$n k_0^2 Z_0 |p_\perp|^2 e^{-(nk_0\sigma)^2}/(32\pi^2)$, and its integral over the sphere is
$P_{bulk} = $ `dipole_vacuum_power` $\cdot\, n\, e^{-(nk_0\sigma)^2}$ — the constant needs no
further calibration. A direction $(\theta, \varphi)$ into the cover is
$\hat r = (\sin\theta\cos\varphi, \cos\theta, \sin\theta\sin\varphi)$ (the specular direction of
a wave incident at $(\theta, \varphi)$), so the plane wave is `grating.solve` at
$(\theta, \varphi + \pi)$; directions into a lossless substrate are solved on the problem
mirrored at $y = 0$ (mesh, stack, PML, $p_y \to -p_y$). One solve per direction and
polarisation, no singularities; Stage C gives the radiated part only. By the array scanning a
Floquet order of Stage A maps to the same quantity,
$dP/d\Omega = \frac{P}{4\pi^2}(nk_0)^2\cos\theta\,P_m(k_x, \beta)$ with
$(k_x + 2\pi m/P, \beta) = nk_0\sin\theta\,(\cos\varphi, \sin\varphi)$.
`python/tests/test_emission_pattern.py` checks a homogeneous cell against the closed form per
direction and the integral over a spherical 3-design (the eight cube corners) against
$P_{bulk}$ (both to $10^{-6}$), a dipole above glass against the plane-wave Fresnel far field
computed independently (direct and reflected wave in air, transmitted wave in glass, also
beyond the critical angle; $10^{-6}$), and a glass ridge against the $m = 0$ order of Stage A at
$(k_x, \beta) = (0.3, 0.2)\,k_0$ ($2\cdot10^{-5}$; the opposite direction differs by a factor of
three). Stage B (array scanning with the guided-mode poles) is the next step of M17.

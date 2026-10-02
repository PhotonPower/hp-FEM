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
  efficiency one, analytically and from the discrete solution).

Planned: Purcell factor $F_P = P_{\mathrm{emitted}}/P_{\mathrm{bulk}}$ for a point dipole.

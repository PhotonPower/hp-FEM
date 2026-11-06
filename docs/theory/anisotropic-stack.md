# Anisotropic layer stacks

Plane-wave reflection and transmission of planar stacks with crystal layers and a crystal
substrate (`physics::AnisotropicStack`, `include/hpfem/physics/anisotropic_stack.hpp`; decision
record [ADR-0015 §5](../adr/0015-anisotropic-materials.md), milestone M19 S4). Conventions as
everywhere: time dependence $e^{-i\omega t}$, a lossy medium has a positive semi-definite
$\operatorname{Im}\varepsilon$; laboratory frame with $z$ the layer normal
([materials.md](materials.md)). The wave comes from an isotropic, lossless incidence medium
above the top interface and travels towards $-z$. The isotropic `LayerStack`
([maxwell.md](maxwell.md#layered-background)) stays the fast path of the solvers and the
reference of the isotropic case.

## The 4×4 system

With the in-plane wavevector $k_0(a, b)$ fixed by the incidence, every field in a homogeneous
region is $F(z)\,e^{ik_0(ax + by)}$. Write $\tilde H = Z_0H$, so that
$\nabla\times E = ik_0\tilde H$ and $\nabla\times\tilde H = -ik_0\varepsilon E$. The $z$ rows are
algebraic,
$$
\tilde H_z = aE_y - bE_x, \qquad
E_z = \frac{-(a\tilde H_y - b\tilde H_x) - \varepsilon_{zx}E_x - \varepsilon_{zy}E_y}{\varepsilon_{zz}},
$$
(this needs $\varepsilon_{zz} \ne 0$; an epsilon-near-zero $\varepsilon_{zz}$ is refused), and the
tangential field $\psi = (E_x, E_y, \tilde H_x, \tilde H_y)$ obeys the Berreman system
(Berreman 1972, Yeh 1979)
$$
\psi' = ik_0K\psi, \qquad
\begin{aligned}
E_x' &= ik_0(\tilde H_y + aE_z), & E_y' &= ik_0(bE_z - \tilde H_x),\\
\tilde H_x' &= ik_0\big(a\tilde H_z - (\varepsilon E)_y\big), &
\tilde H_y' &= ik_0\big(b\tilde H_z + (\varepsilon E)_x\big).
\end{aligned}
$$
`AnisotropicStack::berreman_matrix` builds $K$ column by column from this linear map. Its
eigenvalues are the normalised vertical wavenumbers $q = k_z/k_0$; for an isotropic medium they
are $\pm\sqrt{\varepsilon - a^2 - b^2}$, each twice. The power flux along $z$ is
$2Z_0S_z = \operatorname{Re}(E_x\bar{\tilde H}_y - E_y\bar{\tilde H}_x)$.

## Subspaces instead of eigenvectors

The eigenvectors of $K$ are not a reliable basis: every isotropic region is doubly degenerate,
an optic axis near $z$ or near-normal incidence makes individual eigenvectors ill-conditioned
(the eigenvector matrix of a degenerate region can be numerically singular), and absorbing
biaxial media have **Voigt (exceptional) points** where $K$ is defective. The stack therefore
uses, per region, two **ordered complex Schur forms** $K = UTU^H$ (LAPACK `ztrexc`-style Givens
swaps of adjacent diagonal entries): one with the two upward eigenvalues first, one with the two
downward ones first. The leading two columns of $U$ are an orthonormal basis $W_u$ (resp. $W_d$)
of the upward (downward) invariant subspace, and the leading $2\times2$ block $T_u$ ($T_d$) is
$K$ on that subspace, upper triangular. Both stay well defined under degeneracy and at a
defective point.

**Up and down.** An eigenvalue is upward if $\operatorname{Im}q > 0$ (decaying upwards:
evanescent or absorbing); for $\operatorname{Im}q = 0$ (propagating in a lossless medium) by the
sign of $S_z$ of its eigenvector — the only use of eigenvectors, and a sign does not suffer from
their conditioning. A region without two modes of each kind (grazing, $k_z = 0$) is refused.

**Propagation.** In region $j$ of thickness $d$ the upward amplitudes are referred to its bottom,
the downward ones to its top, so the propagators
$$
P_u = \exp(ik_0dT_u), \qquad P_d = \exp(-ik_0dT_d)
$$
only decay (or keep their modulus). They are the closed-form exponentials of upper triangular
$2\times2$ matrices,
$\exp\begin{pmatrix}\lambda_1 & \beta\\ 0 & \lambda_2\end{pmatrix} =
\begin{pmatrix}e^{\lambda_1} & \beta\,\frac{e^{\lambda_1} - e^{\lambda_2}}{\lambda_1 - \lambda_2}\\
0 & e^{\lambda_2}\end{pmatrix}$, with the divided difference by its Taylor series for
$|\lambda_1 - \lambda_2| < 10^{-3}$ — exact at a defective block.

**S-matrix recursion.** At interface $i$ (region $i$ above, $i+1$ below) the upward amplitudes
are $\alpha_i = R_i\delta_i$ (all at the interface). Below, the upward field of region $i+1$ at
the interface is $\Gamma\delta_{i+1}$ with $\Gamma = P_uR_{i+1}P_d$ ($\Gamma = 0$ in the
substrate). Continuity of $\psi$,
$$
\begin{pmatrix} W_u^{(i)} & -\big(W_u^{(i+1)}\Gamma + W_d^{(i+1)}\big) \end{pmatrix}
\begin{pmatrix}\alpha_i\\ \delta_{i+1}\end{pmatrix} = -W_d^{(i)}\delta_i ,
$$
gives $R_i$ and the transmission $\delta_{i+1} = \tau_i\delta_i$ by one $4\times4$ solve per
interface, from the substrate upwards. No transfer matrix crosses a layer: thick and absorbing
layers neither overflow nor cancel (a 100-wavelength absorbing crystal reflects like the
half-space).

## Output

For incident s ($E \parallel \hat s = \hat z\times\hat k_\parallel$) and p
($E \parallel \hat s\times\hat k$, so that $H \parallel +\hat s$ as in `LayerStack`) of unit
$|E|$: the downward amplitudes $\delta_0 = W_d^{(0)H}\psi_{inc}$, the reflected field
$W_u^{(0)}R_0\delta_0$ decomposed on the reflected s and p waves gives the **Jones matrix**
$$
\begin{pmatrix} r_{ss} & r_{sp}\\ r_{ps} & r_{pp}\end{pmatrix}
$$
(columns incident, rows reflected; E amplitudes at the top interface), the transmitted field at
the top of the substrate is decomposed on the two downward substrate modes (s and p of an
isotropic substrate; the eigenvectors of $T_d$ for a crystal, or the orthonormal Schur basis
where the modes coincide), each with unit $|E|$. Reflectance and transmittance follow from $S_z$
of the incident, reflected and transmitted fields, $A = 1 - R - T$.

For an isotropic stack $r_{ss}$, $r_{pp}$ equal `LayerStack`'s s and p reflection coefficients
(the p one is the ratio of the H amplitudes, which with this p vector equals the ratio of the E
amplitudes in the incidence medium) and $r_{sp} = r_{ps} = 0$. At normal incidence and azimuth 0,
$\hat s = \hat y$, the incident p is $-\hat x$ and the reflected p is $+\hat x$, so
$r_{xx} = -r_{pp}$, $r_{yx} = -r_{sp}$, $r_{yy} = r_{ss}$, $r_{xy} = r_{ps}$.

## Substrate modes

The transmitted field is decomposed on two downward modes of the substrate, each with unit
$|E|$ (`StackResponse::substrate_modes`, `substrate_mode_kinds`):

- **isotropic:** s and p;
- **uniaxial** (two equal principal values of $\operatorname{Re}\varepsilon$, optic axis
  $\hat c$ the third eigenvector): the **ordinary** mode is the one with $D = \varepsilon E \perp
  \hat c$ (its $k_z = -k_0\sqrt{\varepsilon_o - a^2 - b^2}$), the **extraordinary** one the other
  eigenmode of the downward block;
- **biaxial:** the two eigenvectors of the downward block $T_d$;
- **coinciding modes** (k along an optic axis): the basis of the downward subspace that
  diagonalises the Hermitian cross power flux
  $P_{ij} = \tfrac12(\bar E_{x,i}\tilde H_{y,j} + \bar{\tilde H}_{y,i}E_{x,j} - \bar E_{y,i}\tilde H_{x,j}
  - \bar{\tilde H}_{x,i}E_{y,j})$ (power-orthogonal modes).

`channel_transmittance` is the power per mode, $|t_m|^2$ times the mode's flux over the incident
flux; it adds up to $T$ when the modes are power-orthogonal (always for a lossless substrate).

## The stack as layered background

`AnisotropicStack::plane_wave(k0, angle, pol, amplitude, azimuth)` returns the analytic total
field in every region,
$$
\psi(z) = e^{ik_0(ax + by)}\Big(W_u\,e^{ik_0T_u(z - z_b)}\alpha + W_d\,e^{ik_0T_d(z - z_t)}\delta\Big),
$$
with the upward amplitudes at the bottom and the downward ones at the top of each region (so all
exponentials decay), the full E from $\psi$ and $\nabla\times E = ik_0\tilde H$, the branch of a
given region (`value_in_region`, for points on an interface) and the incident wave alone. The
incident wave has the phase 1 at the origin of the top interface, as in `LayerStack`.

**3D solver.** `layer_stack()` gives the geometry as a `LayerStack<3>` (which now holds crystal
layers for lookup; its own scalar `plane_wave` refuses them). With it as
`ScatteringSetup<3>::background` and the stack field as `incident`, the scattered-field source
is the tensor contrast $k_0^2(\varepsilon_c - \varepsilon_b)E^{inc}$, zero wherever a cell
equals its layer.

**Conical solver.** `anisotropic_conical_wave(stack, k0, angle, azimuth, pol)` turns a stack given
in the laboratory frame into the solver frame of the conical solver (x across the lines, y the
normal, z along them): the background `LayerStack<2>` carries $P\varepsilon P^T$ with P swapping y
and z, the solver point $(x, y)$ is the laboratory point $(x, 0, y)$, the field is $PE$ in scaled
components and its physical curl $\det(P)\,P\,\nabla\times E = -P\,\nabla\times E$ (the curl is a
pseudovector and P a reflection). The angles and the s / p vectors coincide with
`layered_conical_wave`.

## Mode-resolved diffraction orders

`half_space_modes(eps, a, b, upward)` gives the two upward or downward modes of a homogeneous
half-space at the normalised in-plane wavevector $(a, b)$, labelled as the substrate modes above,
with unit $|E|$, their $k_z/k_0$, the full $E$ and the flux $2Z_0S_z$ along their direction. The
substrate modes of `response` are the downward ones.

`conical_mode_orders` (ADR-0015 §7) decomposes the orders of a conical field on an `OrderLine` in
a crystal on these modes. With the unit tangent $\hat t$ and the normal $\hat n$ away from the
structure, the local frame $(\hat t, \hat n\times\hat t, \hat n)$ is right-handed in the solver
coordinates; with R the matrix of its rows the order $m$ sees the medium $R\varepsilon R^T$ at
$a = k_{t,m}/k_0$, $b = \beta\,(\hat n\times\hat t)_z/k_0$, and only outgoing waves (along
$\hat n$) on the line. The tangential part of $R A_m$ then fixes both amplitudes $c_{1,2}$ by a
2×2 solve, and

$$
\eta_{m,j} = \frac{|c_j|^2 S_j}{|E_0|^2 k_n^{inc}/k_0}, \qquad
\eta_m = \frac{S(c_1\psi_1 + c_2\psi_2)}{|E_0|^2 k_n^{inc}/k_0}
$$

with the fluxes $S$ in units of $2Z_0$. In a lossless medium with distinct modes the channels are
power-orthogonal and $\eta_m = \eta_{m,1} + \eta_{m,2}$; in an isotropic one the channels are s and
p and $\eta_m$ equals `conical_diffraction_orders`. In a lossy medium the efficiencies are the
powers through the line. A channel propagates when $|\operatorname{Im}k_n| < |\operatorname{Re}k_n|$.

**Jones convention and reciprocity.** The reflection Jones matrix of the stack uses
$\hat s = \hat z\times\hat k_\parallel$ for both waves and $\hat p = \hat s\times\hat k/n$ in the
laboratory frame. `hpfem.grating.jones` uses `conical_polarisation` in the solver frame,
$\hat s = \hat k\times\hat n$, $\hat p = \hat s\times\hat k$; mapped back by the reflection P
(y ↔ z), cross products change sign, so its ŝ is the stack's and its p̂ the negative of the
stack's for both waves: $J = D\,r\,D$ with $D = \operatorname{diag}(1, -1)$, i.e. equal
diagonals and opposite off-diagonal elements (checked to $10^{-8}$ on a flat crystal stack).
Reversing the specular path at the azimuth φ is the incidence at φ + π, and for every symmetric
tensor

$$
r_{ss}(\phi + \pi) = r_{ss}(\phi),\quad r_{pp}(\phi + \pi) = r_{pp}(\phi),\quad
r_{ps}(\phi) = -r_{sp}(\phi + \pi).
$$

A plate with its axis in the layer plane at the azimuth α against the plane of incidence is
mirrored by α → −α: $r_{ss}, r_{pp}$ are even, $r_{sp}, r_{ps}$ odd in α (and zero for the axis
in or perpendicular to the plane of incidence).

## Modal transparent boundary

A coordinate-stretching PML amplifies a *backward* wave, whose phase travels into the layer while
its energy leaves it ($k_nv_n < 0$, `pml-backward-wave`). In a crystal with a tilted axis an
outgoing diffraction order can be such a wave. `ConicalScatteringSetup::modal_boundaries`
truncates the substrate (or the cover) of a periodic conical problem instead with the exact
radiation condition of the half-space: the boundary term of the weak form,
$\oint (n\times\nabla\times E)\cdot F\,ds = ik_0\oint (n\times\tilde H)\cdot F\,ds$
($\nabla\times E = ik_0\tilde H$, the test function in physical components
$F = (V_x, V_y, -iw)\,e^{-i\beta z}$), is evaluated order by order. In the local frame
$(\hat t, \hat n\times\hat t, \hat n)$ of the line (n outward) the unknown is outgoing, so for
the order $k_m = k_{t0} + 2\pi m/a$ its tangential magnetic field is
$\tilde H_t = Y_mE_t$ with the admittance $Y_m = W_HW_E^{-1}$ of the two outgoing modes of
`half_space_modes` (labelling-independent, also for coinciding modes). With
$J = \left[\begin{smallmatrix}0&-1\\1&0\end{smallmatrix}\right]$ for $n\times$,

$$
B_{ij} = ik_0\sum_{m=-M}^{M}\hat F_{i,m}^T\,J\,Y_m\,\hat E_{j,m},\qquad
\hat E_{j,m} = \frac1a\int_\Gamma E_t(\phi_j)e^{-ik_ms}ds,\quad
\hat F_{i,m} = \int_\Gamma F_t(\phi_i)e^{ik_ms}ds
$$

(`conical_modal_boundary_matrix`, dense on the DoFs of the line; the tangential traces are the
tangential Nédélec component and $\pm E_z = \pm iv$). The Bloch constraints combine the two ends
of the line as for the volume terms. Orders beyond M see a natural boundary; the default
$M = \lceil n_fp/2\rceil + 2$ covers the resolution of the trace of n_f facets of order p.
Like the volume operator, B satisfies $B(\beta, k_t)^T = S\,B(-\beta, -k_t)\,S$
(reciprocity of a symmetric tensor). The line must be straight, along the period, in one
homogeneous non-magnetic medium outside the PML; the estimators, the resonance and the sweep
solvers do not take the term.

## Crystal stacks in `hpfem.grating`

`grating.solve` / `validate` / `jones` take the stack as an `AnisotropicStack` in the laboratory
frame (as the crystals of the structured parts); it is turned into the solver frame with
`anisotropic_conical_wave`. A `LayerStack2D` with crystals is refused (its frame would be
ambiguous). In a crystal substrate the transmitted orders are measured with
`conical_mode_orders` and carry `channels`; `bottom="modal"` truncates the substrate by the
modal transparent boundary, and `bottom="pml"` is refused when a transmitted order is a backward
wave. `grating.jones` solves for s and p and returns per propagating order the power-normalised
Jones matrix (amplitudes referred to the top / bottom interface of the stack, rows s / p or the
mode channels) and its Mueller matrix $M = A(J\otimes\bar J)A^{-1}$ (Stokes vectors
(I, Q, U, V), $Q = |E_s|^2 - |E_p|^2$; $M_{00}$ is the efficiency for unpolarised light). The
reciprocity relation above holds for the specular order of a grating too (the reversed path is
the incidence at φ + π).

**Form birefringence.** A lamellar grating of period P ≪ λ (fill f, lines along y) reflects
like the uniaxial plate of effective-medium theory, $\varepsilon_y = \varepsilon_z =
f\varepsilon_1 + (1 - f)\varepsilon_2$ along the lines and the normal, $\varepsilon_x =
(f/\varepsilon_1 + (1 - f)/\varepsilon_2)^{-1}$ across them (component by component for crystal
lamellae with the axes of the lamellae; L. Li 2003). The plate's reflection refers to its top,
the grating's to the bottom of the lines: $J = e^{-2ik_0\cos\theta\,h}DrD$. The difference falls
like P/λ, not like the (P/λ)² of the bulk correction: the evanescent orders form boundary layers
of thickness ~P at the top and the bottom of the grating, which the plate lacks. The Richardson
extrapolation $2J(P/2) - J(P)$ removes that term and falls like (P/λ)².

## A dipole above the stack

`AnisotropicStack::reflection_at(k0, a, b)` gives the Jones reflection matrix at any in-plane
wavevector $k_0(a, b)$, also beyond the light cone of the cover, where the incident and the
reflected wave are evanescent ($q = \sqrt{n^2 - a^2 - b^2}$, Im q ≥ 0, ŝ and p̂ complex with
$\hat s\cdot\hat s = \hat p\cdot\hat p = 1$). A current moment m at the height h above the top
interface delivers (exp(−iωt), Weyl expansion of the Green's dyadic,
$I - \hat k\hat k^T = \hat s\hat s^T + \hat p\hat p^T$)

$$
\frac{P}{P_{bulk}} = 1 + \frac{3}{4\pi n|m|^2}\,\mathrm{Re}\iint \frac{da\,db}{q}\,
e^{2ik_0qh}\; m^H\,[\hat s, \hat p_u]\; r\; [\hat s, \hat p_d]^T\, m,
\qquad P_{bulk} = \frac{Z_0k_0^2n|m|^2}{12\pi}
$$

(`hpfem.planar.dipole_above_stack`). For an isotropic stack and m ∥ ẑ it reduces to the
Chance–Prock–Silbey integral $1 + \tfrac32\mathrm{Re}\int s^3/s_z\,r_p\,e^{2ik_1hs_z}ds$ (the
stack's $r_{pp}$ has the convention $r_p = +1$ for a perfect conductor). The Gaussian dipole of
ADR-0013 multiplies every spectral component by $e^{-\sigma^2k\cdot k} = e^{-(nk_0\sigma)^2}$,
the factor of $P_{bulk}$, so the Purcell factor of the point dipole is that of the FEM emitter.
The radial integral runs on the real axis ($s = n\sin t$, $s = n\cosh u$), the azimuth by the
trapezoidal rule. Guided modes of lossless stacks are poles on the real axis (S7e-2a,
`depth` > 0): bound modes are evanescent in both half-spaces, so beyond the largest principal
index $s_b$ of the cover and the substrate the radial path leaves the axis,
$s = s_b + (S - s_b)\tau - i\,\text{depth}\,\sin\pi\tau$ up to S beyond all indices (a small loss
moves the poles above the axis, exp(−iωt), so the path passes them on the causal side), with the
stack at complex in-plane wavevectors (`reflection_at`; ŝ normalised by the non-conjugating norm,
the two modes with the largest Im k_z upward, unambiguous where both half-spaces are
evanescent). Closer to the axis a tilted crystal may have upward modes with Im k_z < 0 (its
in-plane group velocity can oppose the wavevector), which is why the path stays real up to $s_b$.

**One sample of the array scan.** The cell problem of `grating.emit` (a Bloch array of
Gaussian dipoles with $e^{ik_xP}$ and $e^{i\beta z}$) samples the same spectrum on the orders
$k_m = k_x + 2\pi m/P$ at $k_y^{lab} = \beta$: with $\iint d^2k = \sum_m\int_{BZ}dk_x\int d\beta$ and
the array-scan weight $P/(4\pi^2)$,

$$
P_{cell}(k_x, \beta) = \frac{\omega\mu_0}{4P}\sum_m \mathrm{Re}\Big[\frac{1}{k_z}\Big(
\tfrac12\textstyle\sum_\pm|m_\perp^\pm|^2\,[k_z\ \text{real}] + e^{2ik_zh}\,m^HM_mm\Big)\Big]
e^{-(nk_0\sigma)^2}
$$

(`hpfem.planar.array_sample_above_stack`; the homogeneous term is the closed form of the M17 S1
test). The Gaussian must stay 6σ clear of the stack interfaces (`emit` warns otherwise): a source
reaching into the next medium is no longer the point dipole of the reference (the first
comparison missed by 2.5 % for σ = 0.05 µm at 0.12 µm until the dipole was raised to 0.32 µm).

**The array scan above crystals (S7c).** The (kx, β) integrand of `grating.dipole_emission` is
even in β and in kx only for mirror-symmetric media. With a crystal in the cell or the stack, β
runs over (−β_max, β_max) with half the weight, kx over the full zone, and every entry of the
power matrix counts. The light line of a crystal half-space (k_y = 0 in the solver frame) is
the zero set of the Fresnel quartic $F(k_x) = \det(k_0^2\varepsilon + kk^T - |k|^2I)$,
$k = (k_x, 0, \beta)$: its real roots are the crossings of the kx contour, on the side of the
group velocity $v_x = -F_{k_x}/F_{k_0}$ (a small loss moves the root into the upper half plane
when $v_x > 0$), its complex roots the branch points of the evanescent sheets; β is split at the
turning points of the two sheets, the largest $k\sin t$ of the generalised eigenvalues
$\det(k_0^2\varepsilon - k^2(I - uu^T)) = 0$, $u = (\cos t, 0, \sin t)$. The Wood-anomaly
pinch breaks are kept for the isotropic media only. `symmetric` is not available with
crystals. The guided-mode poles (S7e-2b) are detected per β node by the AAA fit on the kx
contour and bumped to the causal side as for isotropic cells, which needs no kx mirror; the β
pre-pass that splits the β rule at the pinch points (pole pairs meeting at kx = 0 or π/P) does
assume it and is skipped, so a crystal structure's β rule is not split there. This path is not
validated yet: `benchmarks/m19_crystal_guided_scan.py` (a tilted crystal slab against
`dipole_above_stack(depth=...)`, a long run) is pending, and `dipole_emission` warns.

**Emission by reciprocity (S7d).** dP/dΩ into the direction r̂ is
$n k_0^2Z_0|p\cdot\langle E_{pw}\rangle|^2/(32\pi^2)$ with the plane wave incident from r̂ (ADR-0013
§6); reciprocity needs only a symmetric ε, so it holds for crystals. A direction into the
substrate is computed on the cell mirrored at y = 0: the mirror maps a solution for ε onto one
for $M\varepsilon M$, $M = \operatorname{diag}(1, -1, 1)$ (solver frame), so every cell's
tensor is mirrored and the stack becomes the `AnisotropicStack` of the reversed layers with
$M_z\varepsilon M_z$ (laboratory frame), the former substrate as its incidence medium. A crystal
substrate becomes a crystal incidence medium of the mirrored stack (S7e below). The Gaussian average at the dipole carries $e^{-(n_{host}k_0\sigma)^2}$
for every direction (the plane wave at the dipole lives in the host), the prefactor n is that of
the direction's medium.

**Radiation into a crystal substrate (S7e).** The stack takes a lossless crystal as its
incidence medium; `AnisotropicStack::mode_wave(k0, a, b, mode)` gives the analytic field of one
of its downward modes (`half_space_modes`: ordinary / extraordinary), and
`anisotropic_conical_mode_wave` the same in the conical solver frame. A direction into a crystal
is not described by an index, so the power is integrated over the in-plane wavevector of the
substrate modes: with the field ⟨E_j⟩ at the dipole of mode j incident from the substrate with
unit |E| and its z flux $F_j$ ($2Z_0S_z$),

$$
\frac{dP}{d^2k_\parallel} = \frac{Z_0\,|p\cdot\langle E_j\rangle|^2}{32\pi^2F_j},
$$

which for an isotropic medium is $dP/d\Omega = nk_0^2Z_0|p\cdot E|^2/(32\pi^2)$ with
$d\Omega = d^2k_\parallel/(kk_z)$ and $F = k_z/k_0$. Per azimuth ψ the wavenumber runs to the
cutoff $\kappa_{max,j}(\psi)$ of the mode (bisection on `half_space_modes`) as
$\kappa = \kappa_{max}\sin\tau$ (the $1/F_j$ singularity cancels), split at the light line of the
cover (the near-field coupling peaks just beyond it) and at the cutoff of the other mode (its
reflected wave turns evanescent), each panel with the cosine-mapped Gauss rule: a plain Gauss
rule converged only like $n^{-3}$ on the square-root points (0.5 % at 8 points per panel). The
incident mode is solved on the mirrored cell (`grating._crystal_substrate_radiation`);
`emission_cone(side="down")` (the whole half-space) and the `channels` of `dipole_emission` use
it. Uniaxial substrates only (the o / e labelling); biaxial ones are refused.

## Verification

`tests/unit/physics/test_anisotropic_stack.cpp`:

- isotropic stacks (absorbing layer, a metal film, lossy substrate, three angles at an azimuth)
  reproduce `LayerStack` to $10^{-12}$, with zero cross-polarisation;
- thin crystal layers (an absorbing tilted biaxial layer, an isotropic one, a tilted uniaxial
  one) on a tilted uniaxial substrate agree with an independent transfer-matrix reference
  ($\exp(-ik_0Kd)$ by Taylor series with scaling and squaring, no eigenvectors in the layers) to
  $10^{-10}$, cross-polarisation included;
- a plate with the axis in the plane at normal incidence gives the two Fabry–Pérot problems with
  $n_e$ and $n_o$ in closed form, $r_{xx} = \cos^2\phi\,r_e + \sin^2\phi\,r_o$,
  $r_{yx} = \sin\phi\cos\phi\,(r_e - r_o)$;
- an optic axis exactly along $z$ at normal incidence (both pairs degenerate) equals the
  isotropic plate with $n_o$, and $10^{-6}$ rad off it changes the result by less than $10^{-9}$;
  the same for a crystal substrate with the axis along $z$;
- lossless crystals conserve energy, $R + T = 1$ to $10^{-12}$; a 100-wavelength absorbing
  crystal reflects as the half-space;
- at a Voigt point of an absorbing biaxial medium (found by Newton on the discriminant
  $(q_1 - q_2)^2$ of the downward eigenvalues), the layer agrees with the eigenvector-free
  reference to $10^{-9}$, and the half-space reflection is continuous across the point;
- the analytic field solves $\nabla\times\nabla\times E = k_0^2\varepsilon E$ in every region
  (finite differences), its tangential components are continuous at the interfaces, and for an
  isotropic stack it equals `LayerStack`'s field and curl;
- the ordinary mode of a tilted uniaxial substrate has the closed-form $k_z$; the power channels
  add up to $T$, also for coinciding modes;
- `tests/convergence/anisotropic_stack_background.cpp`: a tilted crystal plate meshed on the
  analytic half-space of a c-cut substrate converges to the analytic field of the full stack, in
  the 3D solver (Bloch cell, $2\cdot10^{-2} \to 9\cdot10^{-5}$ for p = 2…4) and in the conical
  solver ($5\cdot10^{-2} \to 1\cdot10^{-5}$ for p = 2…5); a structure equal to the stack has no
  scattered field (exactly zero), in both solvers;
- `anisotropic_conical_wave` of an isotropic stack equals `layered_conical_wave` (field, curl,
  β, R), and its curl matches the field by finite differences for a crystal stack;
- half-space modes: s / p with $k_z = \pm\sqrt{\varepsilon - a^2 - b^2}$ and flux $\operatorname{Re}k_z$,
  eigenvectors of K and the closed-form ordinary $k_z$ of a tilted crystal, both directions, and
  the substrate modes of `response`; the mirror symmetry in α and the reciprocity relation above
  (also for an axis tilted out of the plane) to $10^{-12}$;
- `conical_mode_orders` (`test_anisotropic_conical.cpp`): a field of known ordinary and
  extraordinary waves (one evanescent) in a tilted crystal, verified to solve Maxwell's equations
  in the solver frame by finite differences, gives back its amplitudes to $10^{-10}$; in an
  isotropic medium the order efficiencies equal `conical_diffraction_orders`;
- `anisotropic_stack_background` (convergence): on a lossless c-cut substrate the ordinary and
  extraordinary channels of the transmitted zeroth order of the conical FEM field converge to the
  per-mode transmittances of the stack ($1.5\cdot10^{-4} \to 3.5\cdot10^{-7}$ for p = 2…5; the
  error of these quadratic quantities alternates in sign, so it falls in pairs of p);
- `conical_modal_boundary_matrix` (`test_anisotropic_conical.cpp`): the transposition symmetry
  above to $10^{-12}$ for a tilted crystal, rows only on the line;
- `anisotropic_modal_boundary` (convergence), a tilted lossless crystal substrate truncated by
  the modal boundary: a crystal plate converges to the analytic stack field
  ($5.8\cdot10^{-2} \to 1.5\cdot10^{-5}$, the o / e channels to $6.5\cdot10^{-7}$, p = 2…5); a
  lossless lamellar grating whose transmitted order −1 is an extraordinary backward wave closes
  the energy balance R + T = 1 to $1.6\cdot10^{-4} \to 3.3\cdot10^{-8}$, while a PML in the
  same substrate stalls at $1.1\cdot10^{-4}$ for every p. (The plate alone is no test of the
  PML: only its zeroth order propagates, a forward wave, and the PML there converges as well.)
- `python/tests/test_grating_jones.py`: a flat crystal stack through `grating.jones` gives the
  stack's reflection Jones matrix ($J = DrD$) and its o / e channels to $10^{-8}$; a tilted
  crystal ridge on glass satisfies the reciprocity relation of the specular order in the conical
  solver to $10^{-5}$ at p = 3 (with $|J_{ps}| = 0.056$; $1.3\cdot10^{-7}$ at p = 4); a grating
  on a substrate with a backward order −1 is refused with `bottom="pml"` and closes its energy
  balance with `bottom="modal"`; a lamellar grating (fill ½, ε = 4 and diag(4, 3, 2.5), h = 0.3λ,
  θ = 0.5, φ = 0.7, |J_ps| ≈ 0.05–0.08) against the effective-medium plate at P = 0.2, 0.1,
  0.05λ: the difference 1.1·10⁻², 7.5·10⁻³, 4.1·10⁻³ (~P; the FEM error is 4·10⁻⁵ by mesh
  refinement), the Richardson extrapolation 3.5·10⁻³, 8.5·10⁻⁴ (~P²);
- `reflection_at` equals `response` inside the light cone and the Fresnel coefficients with
  complex $k_z$ beyond it; `python/tests/test_planar_dipole.py`: `dipole_above_stack` equals the
  1D Sommerfeld integrals with the Fresnel / Airy coefficients (glass half-space, absorbing layer
  on glass) and with the closed-form $r_p = (\varepsilon_oq_1 - q_{2e})/(\varepsilon_oq_1 +
  q_{2e})$, $q_{2e} = \sqrt{\varepsilon_o(1 - s^2/\varepsilon_e)}$, of a c-cut half-space, to
  $10^{-9}$ (the glass case to $10^{-15}$); a tilted absorbing crystal layer gives the same F
  when crystal and dipole are rotated together about the normal; `grating.emit` above a tilted
  absorbing crystal layer equals `array_sample_above_stack` to $6\cdot10^{-8}$ inside and
  $2\cdot10^{-5}$ beyond the light cone of the cover, above a tilted lossless crystal substrate
  with the modal bottom to $10^{-5}$ with $P_{cell} = $ up + down;
- the crystal light lines (`test_planar_dipole.py`): the roots of the Fresnel quartic and the
  turning points of a diagonal tensor equal the closed forms; the scan rule with the crystal
  light lines integrates the exact sample spectrum above a tilted lossless crystal half-space to
  the 2D reference ($10^{-4}$ at 6 nodes per panel, $2\cdot10^{-6}$ to $10^{-5}$ at 10);
- the emission pattern of a dipole above a tilted crystal layer on a substrate of index 2, up
  and down at azimuths without symmetry, equals $|p\cdot E_{pw}(x_0)|^2$ of
  `AnisotropicStack.plane_wave` (and of the mirrored stack for the down directions) to
  $6\cdot10^{-5}$ per polarisation (`test_planar_dipole.py`); the powers radiated into both
  half-spaces (`emission_cone`, 8×16 directions each) add up to the total of
  `dipole_above_stack` to $4.8\cdot10^{-5}$ (lossless, no guided modes;
  `benchmarks/m19_crystal_emission_balance.py`, 15 min);
- guided modes on the deformed path (`test_planar_dipole.py`): an isotropic waveguide slab
  (n = 2 on 1.45) equals the 1D Airy–Sommerfeld integral to $10^{-8}$ (the M17 benchmark
  `sommerfeld_layered` to $3\cdot10^{-14}$); a tilted lossless crystal slab gives the same F for
  two path depths, and the same slab with a small loss δ on the real axis tends linearly to it
  ($-2.9\cdot10^{-6}$, $-1.5\cdot10^{-6}$, $-7.7\cdot10^{-7}$ for δ = 4, 2, 1·10⁻³);
- the radiation into a tilted lossless crystal substrate: the FEM field of a mode incident from
  the crystal (both modes, also beyond the light line of the cover) equals `mode_wave` to
  $2\cdot10^{-4}$; with the analytic fields the cover and the crystal half-space together give the
  total of `dipole_above_stack` to $1.2\cdot10^{-4}$ at 8 points per panel and 16 azimuths
  ($2.4\cdot10^{-5}$ at 32), and the FEM (`benchmarks/m19_crystal_substrate_balance.py`) to
  7.5e-5 (`test_planar_dipole.py`);
- `benchmarks/m19_crystal_dipole_scan.py` (a long local run, 3840 cell problems, 32 min with 4
  workers): `dipole_emission` of a Gaussian dipole 0.32 µm above that half-space equals
  `dipole_above_stack` to $5\cdot10^{-4}$ for every orientation (x −5.3e-4, y −5.1e-4,
  z +2.9e-5; the rule alone contributes 1–2e-4), `benchmarks/results/2026-11-05-validation-
  m19-crystal-scan.json`;
- `python/tests/test_anisotropic_volume.py` checks the bindings.

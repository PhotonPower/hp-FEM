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

**Not yet:** diffraction orders and the power balance in a crystal substrate (mode-resolved
orders, M19 S5), and therefore crystal stacks in `hpfem.grating`; a PML in a substrate with a
tilted axis fails the stability condition (`pml-backward-wave`), the modal transparent boundary
of ADR-0015 §2 is S5 as well.

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
- `python/tests/test_anisotropic_volume.py` checks the bindings.

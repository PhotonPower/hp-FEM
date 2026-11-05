# Materials: isotropic and anisotropic media

Conventions as everywhere in the project: time dependence $e^{-i\omega t}$, so a lossy medium
has $\operatorname{Im}\varepsilon > 0$; relative parameters $\varepsilon_r$, $\mu_r$ (SI
otherwise). The decision record is [ADR-0015](../adr/0015-anisotropic-materials.md); the stages
of the migration are milestone M19 of the [roadmap](../roadmap.md).

## The core material

`materials::Material` (`include/hpfem/materials/material.hpp`) is the frequency-independent
description of one medium at one frequency:

- **isotropic:** a complex scalar $\varepsilon_r$ and a complex scalar $\mu_r$
  (`Material(eps, mu)`, `Material::isotropic`, `Material::dielectric(n)`);
- **anisotropic:** a complex symmetric $3\times3$ permittivity tensor in the laboratory frame
  and a scalar $\mu_r$ (`Material::anisotropic(eps, mu)`).

The permeability is always scalar: optical crystals have $\mu_r = 1$, and the full
$\mu^{-1}$ tensors of the assembly are used by the PML only.

| accessor | isotropic | anisotropic |
|---|---|---|
| `is_isotropic()` | true | false |
| `eps_tensor()` | $\varepsilon_r I$ | the tensor |
| `mu_inv_tensor()` | $\mu_r^{-1} I$ | $\mu_r^{-1} I$ |
| `eps_scalar()`, `refractive_index()` | $\varepsilon_r$, $\sqrt{\varepsilon_r\mu_r}$ | throw `AnisotropicUnsupported` |
| `mu_scalar()` | $\mu_r$ | $\mu_r$ |

**Why the scalar accessors throw.** Every feature written before M19 reads the scalar
permittivity. A tensor material reaching such a feature must not be reduced silently to some
scalar (its trace, its $zz$ entry, …): the result would look plausible and be wrong. The
exception (`hpfem::AnisotropicUnsupported`, a `NotImplemented`; in Python
`hpfem.AnisotropicUnsupported`, a `NotImplementedError`) names the accessor, and each
feature is migrated to the tensor accessors in its own stage. The Python entry points that run
diagnostics (`hpfem.grating.validate` / `solve`, `diagnostics.validate_scattering`) report the
error `anisotropic_unsupported` with the offending tags before the set-up starts.

**Admissible tensors.** `Material::anisotropic` refuses a non-finite or zero tensor and a
non-symmetric one, $\lVert\varepsilon - \varepsilon^T\rVert_{\max} > 10^{-12}\lVert\varepsilon
\rVert_{\max}$: antisymmetric parts describe gyrotropic or optically active media, which break
reciprocity and are out of scope (ADR-0015 §3). A complex symmetric tensor covers every
non-gyrotropic crystal, lossy or not, in any orientation. A tensor proportional to the identity
stays a tensor material; it is the isotropic limit with which the tensor paths are checked
against the scalar ones.

`MaterialMap::is_isotropic()` tells whether a map (background, tags and cell overrides)
contains a tensor material.

## Crystal orientation

A crystal is described by its principal permittivities
$\varepsilon_1, \varepsilon_2, \varepsilon_3$ (each a dispersive model in Python) and the
rotation $R$ from its principal frame to the laboratory frame ($x, y$ along the lattice, $z$ the
layer normal of the 3D and the stack solvers):

$$
\varepsilon_{\text{lab}} = R\,\operatorname{diag}(\varepsilon_1, \varepsilon_2,
\varepsilon_3)\,R^T .
$$

$R$ is a real rotation, so $\varepsilon_{\text{lab}}$ is symmetric. The columns of $R$ are the
principal axes in laboratory coordinates.

**Uniaxial crystals** ($\varepsilon_1 = \varepsilon_2 = \varepsilon_o$, $\varepsilon_3 =
\varepsilon_e$) are oriented by their optic axis
$\hat c = (\sin\theta\cos\phi,\ \sin\theta\sin\phi,\ \cos\theta)$: $\theta$ is the polar angle
from the laboratory $z$ axis, $\phi$ the azimuth from $x$ towards $y$, and
$R = R_z(\phi)\,R_y(\theta)$ (`materials.axis_rotation`). Then
$\varepsilon_{\text{lab}} = \varepsilon_o I + (\varepsilon_e - \varepsilon_o)\,\hat c\hat c^T$.
A c-cut substrate has $\theta = 0$; an a- or m-cut has $\theta = 90°$, with $\phi$ the angle of
the axis against the $x$ direction (for a grating, against the direction across the lines).

**Biaxial crystals** use intrinsic z-x-z Euler angles,
$R = R_z(\alpha)\,R_x(\beta)\,R_z(\gamma)$ (`materials.euler_rotation`): first a turn by
$\gamma$ about the principal $z$ axis, then by $\beta$ about the new $x$ axis, then by $\alpha$
about the laboratory $z$ axis. With $\beta = 0$ the rotation is a turn by $\alpha + \gamma$ about
$z$.

## Volume solvers (2D / 3D)

Since M19 S2 the volume solvers `physics::Scattering`, `physics::Resonance` and
`physics::BandStructure` take tensor materials (by tag, as background, or per cell with
`MaterialMap::set_cell`, e.g. for graded or strained crystals). All three get their coefficients
from `physics::set_material_tensors` (`include/hpfem/physics/material_form.hpp`); isotropic cells
keep the scalar path, a tensor cell enters the Maxwell forms through
`materials::permittivity_tensor<Dim>`:

- **3D:** the full tensor.
- **2D:** the in-plane block $\varepsilon_{\alpha\beta}$, $\alpha, \beta \in \{x, y\}$. The 2D
  space carries the in-plane field $(E_x, E_y)$ with $\partial_z = 0$; a tensor with
  $\varepsilon_{xz} \ne 0$ or $\varepsilon_{yz} \ne 0$ makes the in-plane field drive $E_z$, which
  that space does not have, and is refused with `AnisotropicUnsupported` (use the conical solver,
  M19 S3, or the 3D solver). $\varepsilon_{zz}$ does not enter.

**Scattered-field source.** With an isotropic background $\varepsilon_b$ the source is the
matrix product $k_0^2(\varepsilon_c - \varepsilon_b I)E^{inc}$; the anisotropic layered
background is M19 S4.

**PML.** The coordinate stretch maps any tensor like a coordinate transformation,
$$\tilde\varepsilon_{ij} = \varepsilon_{ij}\,\frac{s_x s_y s_z}{s_i s_j}$$
(`PmlBox::permittivity(tensor, x)`, in 2D with $s_z = 1$ on the in-plane block); $\tilde\varepsilon$
stays symmetric, so the operator stays complex-symmetric and `detect_symmetry` keeps the LDLᵀ
paths. The PML resolution check uses the largest principal index (`materials::max_real_index`).

**Stability of the PML in a crystal.** The stretch absorbs a wave only if its phase and its group
velocity point into the layer together, $k_d\,v_{g,d} \ge 0$ for the stretched axis $d$ (Bécache,
Fauqueux & Joly 2003). `pml::backward_wave<Dim>(eps, axis)` checks this on sampled wave directions
(2D: slowness curve $k^TR^T\varepsilon^{-1}Rk = k_0^2$ of the in-plane waves, group velocity
$R^T\varepsilon^{-1}Rk$; 3D: both sheets of the Fresnel surface, group velocity along the
Poynting vector $\hat k|E|^2 - E(\hat k\cdot E)$ of the eigenwave). The result:

| medium in the layer | condition |
|---|---|
| isotropic | holds |
| uniaxial, optic axis along the layer normal or perpendicular to it | holds |
| uniaxial, optic axis tilted against the normal (also an ordinary crystal) | fails |
| biaxial, principal axes aligned: normal along the largest or smallest axis | holds |
| biaxial, principal axes aligned: normal along the intermediate axis | fails (weakly) |

The last row is the cone of internal conical refraction: near the optic axes, which lie in the
plane of the largest and the smallest principal axis, the group velocity of one sheet sweeps a
cone that contains $v_{g,y} < 0$ for $k_y > 0$ ($\min k_y v_y \approx -0.01$ for
$\varepsilon = \operatorname{diag}(2, 3, 4.5)$, confirmed by the gradient of $\omega(k)$).
`Scattering` logs the warning `pml-backward-wave` for every crystal and stretched axis that
fails, and `diagnostics.validate_scattering` reports `pml_backward_wave`. A wave that is not
excited is not amplified: an eigenwave along the normal of a tilted crystal is absorbed as usual
(`tests/convergence/pml_reflection.cpp`); the failure matters for grazing and scattered waves.

**Losses.** The time-averaged Joule heating of a symmetric tensor is
$$p = \tfrac12\,\omega\varepsilon_0\,E^H\,\operatorname{Im}(\varepsilon)\,E,$$
because the Hermitian part $(\varepsilon - \varepsilon^H)/2i$ of a symmetric $\varepsilon$ is the
real matrix $\operatorname{Im}\varepsilon$ (`materials::loss_density`); it is used by
`absorbed_power`, the per-cell and per-tag absorption, the absorption density and the heat
source of `physics::Thermal`. `BandStructure` requires a real tensor.

**Estimators.** The residual and goal-oriented estimators already took $\varepsilon E$ from the
form, so the divergence term and the jump of $n\cdot\varepsilon E$ use the tensor without
further change.

The conical solver takes crystals since M19 S3 (full tensors in `ConicalForm`, the loss of
symmetry for $\varepsilon_{xz}, \varepsilon_{yz} \ne 0$ and the transposed adjoints:
[maxwell.md](maxwell.md#crystals-full-tensors-in-the-conical-forms));
`diagnostics.validate_scattering` checks conical setups for the scalar $E_z$ path and the PML condition (the
3D condition, conservative). `ConicalSweep` and `hpfem.grating.solve` / `validate` take crystals
in the structured parts; `hpfem.grating` expects them in the laboratory frame of this page and
turns them into the conical frame (x across the lines, y the layer normal, z along the lines) by
swapping y and z (ADR-0015 S3 amendment); a `MaterialMap` with crystals is refused there.
Crystal layer stacks are `AnisotropicStack` (M19 S4) and, in the laboratory frame as well, the
stack of `hpfem.grating.solve` / `validate` / `jones` (S5c, with mode-resolved transmitted
orders and the modal bottom boundary). The emission paths of `hpfem.grating` take crystals as
well (S7, the emitter in an isotropic host), the axisymmetric solver diagonal tensors in
(r, φ, z) (S6: an optic axis along z, or radially / azimuthally anisotropic media; tilted axes
are refused). Still isotropic only (they refuse with `AnisotropicUnsupported`), among others:
the dipole host medium, the background and the layered background of the axisymmetric solver,
`TimeDomain`, `ThermoOptical`, `PropagatingMode` and the modal ports.

## Python layer

`hpfem.materials.Anisotropic(principal, rotation)` holds three dispersive models (numbers and
library names are accepted) and the rotation; `eps_tensor(omega)` returns the laboratory tensor,
`at(omega)` the core tensor `Material`, `rotated(R)` the crystal turned by a further laboratory
rotation. `Uniaxial(ordinary, extraordinary, theta, phi)` builds the uniaxial case. The scalar
`eps_r(omega)` and `refractive_index(omega)` of the `Dispersive` interface raise
`AnisotropicUnsupported`. `DispersiveMap` accepts anisotropic models and tensor materials.

The library entry `Al2O3` (sapphire, Malitson & Dodge 1972, ordinary and extraordinary
Sellmeier forms, 0.2–5 µm) is a c-cut uniaxial crystal; at 589.3 nm it gives
$n_o = 1.7681$, $n_e = 1.7600$.

```python
from hpfem import materials, units
omega = units.angular_frequency(wavelength=633 * units.nm)
a_cut = materials.get("Al2O3").rotated(materials.axis_rotation(theta=np.pi / 2, phi=0.0))
crystal = materials.Uniaxial(2.0, 2.3, theta=0.4, phi=0.2)    # permittivities, radians
eps = crystal.eps_tensor(omega)                                # 3x3, symmetric
material = crystal.at(omega)                                   # tensor hpfem.Material
```

## Verification

`tests/unit/materials/test_material.cpp`: the tensor accessors of both kinds, the refused
scalar access, the isotropic limit staying a tensor, the refused gyrotropic, non-finite and zero
tensors, `MaterialMap::is_isotropic` over background, tags and cell overrides.
`python/tests/test_units_materials.py`: the uniaxial tensor $\varepsilon_o I +
(\varepsilon_e - \varepsilon_o)\hat c\hat c^T$ for an axis in the plane, the optic axis as the
extraordinary eigenvector for a tilted axis, the Euler rotation against the axis rotation and
the principal values of a biaxial tensor, the sapphire indices at 589.3 nm, `DispersiveMap` with
tensors. `python/tests/test_diagnostics.py`: the error `anisotropic_unsupported` of
`grating.validate` / `solve` and of `validate_isotropic`. The migration itself (private scalar
fields, accessors at every former use) is covered by the whole existing test suite.

Volume solvers (S2): `tests/unit/physics/test_anisotropic_volume.cpp` — $\varepsilon_r I$
through the tensor path reproduces the scalar path to round-off in 2D and 3D (solution, residual
estimate, absorbed power), $\varepsilon_{zz}$ does not enter 2D and $\varepsilon_{xx}$ does, the
refusal of plane-to-$z$ coupling in 2D, the PML stretch of a full tensor, per-cell overrides,
the complex symmetry of the operator with an absorbing tilted crystal in the PML (and its loss
with a gyrotropic form), and the stability condition of the table above.
`tests/convergence/pml_reflection.cpp` — eigenwaves of a crystal with a tilted axis (2D with
$\varepsilon_{xy} \ne 0$, 3D full tensor, both sheets) through the PML converge exponentially in
$p$ with no reflection floor. `tests/convergence/anisotropic_slab.cpp` — a uniaxial plate in 3D
(axis in the plane at 30°) at normal incidence: the reflected Jones entries $r_{xx}$, $r_{yx}$
converge exponentially in $p$ to the two Fabry–Pérot problems with $n_e$ and $n_o$.
`python/tests/test_anisotropic_volume.py` — the bound pipeline and the diagnostics
`pml_backward_wave` and `anisotropic_unsupported` of `validate_scattering`.

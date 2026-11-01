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

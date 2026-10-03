# 0010 — Axisymmetric (body-of-revolution, 2.5D) Maxwell solver
**Status:** accepted · **Date:** 2026-10-03

## Context
VCSELs, micropillar cavities with quantum dots, nanowire antennas and many metalens and
plasmonic benchmarks are bodies of revolution. A full 3D Nédélec computation of a pillar of
tens of wavelengths height does not fit the memory of one machine at the accuracy the
*hp* method can deliver, while the azimuthal Fourier decomposition
$E = \sum_m E_m(r, z)\,e^{im\varphi}$ turns it into independent 2D problems on the meridian
plane, one per order $m$. Commercial tools (JCMsuite) offer this; for micro-cavities it is
the standard approach. The project already has the ingredients: the 2D Nédélec and H1
spaces of the same order, the discrete gradient and the gauged eigensolvers, PEC
elimination and the PML-as-material concept.

## Decision
1. **Unknowns.** For order $m$ the meridian components $(E_r, E_z)$ live in the 2D Nédélec
   space on the $(r, z)$ mesh (mesh coordinates $x = r \ge 0$, $y = z$) and the azimuthal
   component in the H1 space of the same order through the **scaled unknown**
   $v = -i\,r E_\varphi$. The scaling makes the gradient of a potential mode exactly
   representable, $\nabla(\psi e^{im\varphi}) \leftrightarrow (\nabla_{rz}\psi,\ v = m\psi)$,
   so the order-$m$ gradient $K_m = [G;\ mI]$ spans the kernel of the curl–curl matrix and
   the gauge of ADR-0004 / the cavity solver removes it (no spurious modes). The factor
   $-i$ makes all matrices real symmetric for lossless media and complex symmetric
   otherwise (`assembly::assemble_axisymmetric`). Physical fields are recovered as
   $E_\varphi = i v / r$.
2. **Forms with weight $r$ and the $1/r$ terms** (`docs/theory/axisymmetric.md`). The
   azimuthal and radial curl components carry $1/r$. Cells touching the axis use a
   quadrature rule two degrees higher; the integrands are bounded rational functions once
   the axis conditions hold, and the Gauss points (never on the axis) weight the
   regularity conditions that cannot be imposed as Dirichlet data ($E_r$, $E_\varphi$
   coupling for $m = \pm1$). This is the established body-of-revolution practice; the
   convergence test confirms rate $2p$ for $m = 0, 1, 2$.
3. **Axis conditions** are Dirichlet data on the facets with the axis tag: $v = 0$ for all
   $m$, $E_z = 0$ (tangential Nédélec DoFs) for $m \ne 0$; the gauge potential vanishes on
   the axis for $m \ne 0$ and is free there for $m = 0$. PEC walls fix the Nédélec trace and
   $v$. The meridian mesh must have the axis as a boundary with its own tag and no vertex
   with $r < 0$.
4. **Materials and PML** are diagonal tensors in $(r, \varphi, z)$ per cell
   (`AxisymmetricForm`), so the cylindrical PML of Teixeira–Chew,
   $\Lambda = \mathrm{diag}(s_\varphi s_z / s_r,\ s_r s_z / s_\varphi,\ s_r s_\varphi / s_z)$ with
   $s_\varphi = \tilde r / r$, enters as a material without touching the operator (ADR-0005).
5. **Staging.** Stage 1 (this ADR): forms, order-$m$ gradient, `physics::AxisymmetricCavity`
   with the PEC-cylinder convergence test. Stage 2: resonances with the cylindrical PML
   (complex gauged solver), validated against the Mie resonances of a sphere. Stage 3:
   scattering with the axial plane wave ($m = \pm1$) against the Mie cross-section, dipole
   sources on the axis (Purcell), far field from the $m$ contributions, Python bindings and
   the micropillar example. Later: coupled orders for oblique incidence, adaptivity on the
   meridian plane with the $r$-weighted estimator.

## Consequences
- A VCSEL / micropillar resonance costs one 2D solve per azimuthal order instead of a 3D
  solve; memory and time drop by two to three orders of magnitude.
- New module files `assembly/axisymmetric_forms` and `physics/axisymmetric`; no change to
  the Cartesian path. The block ordering (Nédélec DoFs, then H1 DoFs) is part of the API of
  `AxisymmetricSystem`.
- Convergence is of the standard order but the axis treatment is "variational by
  quadrature"; the convergence test guards it and every later stage must keep it.
- The meridian problem is complex symmetric with PML, like the Cartesian one, so the
  existing complex eigensolver and direct solvers (including the cuDSS backend of ADR-0008)
  apply unchanged.

## Alternatives considered
- **Unscaled $E_\varphi$ in H1**: polynomial mass term but a rational gradient
  ($im\psi/r$), so the discrete kernel would not be exactly spanned and the gauge would
  leave small spurious eigenvalues. Rejected.
- **Weighted Sobolev spaces with modified bases near the axis** (Gopalakrishnan–Pasciak,
  Oh–Lee): rigorous but needs special axis elements that do not fit the hierarchical
  Schöberl–Zaglmayr bases. Deferred; the plain approach meets the rate.
- **Full 3D with symmetry-reduced meshes** (a wedge with Bloch-like constraints in
  $\varphi$): still three-dimensional memory. Rejected.

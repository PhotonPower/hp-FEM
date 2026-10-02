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
$\|E_h - E\|_{L^2}$ and $\|\nabla\times(E_h - E)\|_{L^2}$, `evaluate_hcurl` the physical field.
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

For an incident field $\mathbf{E}^{\mathrm{inc}}$ that solves Maxwell in the background medium $\varepsilon_b$, write $\mathbf{E} = \mathbf{E}^{\mathrm{inc}} + \mathbf{E}^{\mathrm{sc}}$. Then $\mathbf{E}^{\mathrm{sc}}$ solves the curl–curl equation with the volumetric source
$\mathbf{f} = \omega^2 (\varepsilon - \varepsilon_b)\mathbf{E}^{\mathrm{inc}}$ supported only on the scatterer, and PML absorbs $\mathbf{E}^{\mathrm{sc}}$ cleanly. For layered backgrounds (gratings, masks) the incident field is the analytic multilayer solution.

## Boundary conditions

- **PEC** $\mathbf{n}\times\mathbf{E} = 0$: eliminate edge/face DoFs.
- **PMC** $\mathbf{n}\times(\mu^{-1}\nabla\times\mathbf{E}) = 0$: natural, do nothing.
- **Bloch-periodic** $\mathbf{E}(x+a) = e^{i k_x a}\mathbf{E}(x)$: master/slave DoF constraints with complex factor; DoF orientation on the two faces must match (ADR-0003).
- **Transparent**: PML, see [pml.md](pml.md).

## Post-processing quantities

- Poynting vector $\mathbf{S} = \tfrac12 \operatorname{Re}(\mathbf{E}\times\bar{\mathbf{H}})$, flux through surfaces.
- Absorbed power $P_{\mathrm{abs}} = \tfrac{\omega}{2}\int \operatorname{Im}(\varepsilon)|\mathbf{E}|^2 dx$.
- Scattering/extinction cross-sections, diffraction-order efficiencies (Fourier transform of the field on a plane above/below a periodic structure).
- Far field via Stratton–Chu on a closed surface inside the PML-free region.
- Purcell factor $F_P = P_{\mathrm{emitted}}/P_{\mathrm{bulk}}$ for a point dipole.

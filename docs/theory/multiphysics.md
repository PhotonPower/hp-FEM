# Multiphysics: optical heating

The first multiphysics step (roadmap M9) couples the time-harmonic solution to a steady
heat-conduction problem on the same mesh. Conventions as everywhere: time dependence
$e^{-i\omega t}$, SI units, lossy media have $\mathrm{Im}\,\varepsilon_r > 0$.

## Absorbed power density

The time-averaged power dissipated per volume by a time-harmonic field is

$$
q(\mathbf{x}) = \frac{\omega\varepsilon_0}{2}\,\mathrm{Im}\big(\varepsilon_r(\mathbf{x})\big)\,
|\mathbf{E}(\mathbf{x})|^2 \qquad [\mathrm{W/m^3}],
$$

the integrand of the absorbed power of `physics/postprocess.hpp` (in 2D the density is per
unit length of the invariant direction, so the integral is W/m). It is discontinuous at
material interfaces (the jump of $\mathrm{Im}\,\varepsilon_r$) and, with the Nédélec field,
also in its normal component across every facet; it is piecewise polynomial of degree $2p$
on affine cells.

`physics::absorbed_power_load` therefore does not represent $q$ in a continuous space but
assembles its load on the H1 space directly, cell by cell at the quadrature points,

$$
b_i = \sum_K \int_K \phi_i\, q_h, \qquad q_h = \tfrac{\omega\varepsilon_0}{2}\,
\mathrm{Im}(\varepsilon_{r,K})\,|\mathbf{E}_h|^2 ,
$$

which is exact for the polynomial integrand with the rule of degree $2p + 2$ (the
`source_reference` member of `assembly::ScalarForm` evaluates a source in reference
coordinates of the cell being assembled). The sum of the vertex entries of $b$ is the
absorbed power (the constant 1 of the hierarchical basis has coefficient 1 on the vertex
functions and 0 on the bubbles). `absorbed_power_density` returns the hierarchical H1
interpolant of $q_h$ instead, for export and plotting; it smears the jump at an interface
over one cell and is not meant for integration.

## Heat conduction

The temperature solves the steady heat equation with the thermal conductivity $\kappa$
[W/(m K)] per material tag,

$$
-\nabla\cdot(\kappa\nabla T) = q \ \text{in } \Omega, \qquad
T = T_D \text{ on } \Gamma_D, \qquad \kappa\,\partial_n T = 0 \text{ on } \partial\Omega\setminus\Gamma_D ,
$$

in the weak form $\int_\Omega \kappa\nabla T\cdot\nabla v = \int_\Omega q\,v$ on the
hierarchical H1 space of `docs/theory/scalar-fem.md`: `physics::Thermal` assembles the
stiffness matrix with the per-cell conductivity (`assemble_h1` with a form per cell), takes
the fixed temperatures on tagged facets as Dirichlet data (`dirichlet_values`, eliminated
symmetrically), leaves the other walls adiabatic (natural condition) and solves with the
direct solver backends. On a locally refined mesh the H1 hanging-node constraints are
applied before the elimination, as in the scattering problem. `solve(q)` takes source
coefficients on the same map (load $Mq$), `solve_load(b)` an assembled load such as the
absorbed power, `total_power(q)` integrates source coefficients. Convective (Robin) walls
and temperature-dependent material data follow with the feedback loop of the next
roadmap item.

A thermal problem on the optical mesh is usually over-resolved in the metal and
under-resolved far away — the optical PML region has no thermal meaning and is simply part
of the conducting domain (the heat sink is wherever the fixed temperature is set).

**Verification.** `tests/unit/physics/test_thermal.cpp`: a uniform field in a lossy block
gives the expected density, the vertex sum of the load equals `absorbed_power` to
$10^{-10}$; a two-material strip with $T = 0$ and $T = 1$ at its ends has the exact
interface temperature $\kappa_2/(\kappa_1 + \kappa_2)$ (piecewise linear, exact for
$p \ge 1$, also on a hanging-node mesh); setup errors. `tests/convergence/heat_conduction.cpp`:
a plane wave $E_y = E_0 e^{iknx}$ with complex $n$ in a lossy slab (interpolated into the
Nédélec space) has the density $q_0 e^{-ax}$, $a = 2k\,\mathrm{Im}\,n$, and with $T = 0$ at
both ends of the strip the closed-form temperature

$$
T(x) = \frac{q_0}{\kappa a^2}\Big[1 - e^{-ax} - \frac{x}{L}\big(1 - e^{-aL}\big)\Big];
$$

the relative $L^2$ error of the computed temperature decays exponentially under
p-refinement ($9.6\cdot10^{-2}$, $1.0\cdot10^{-3}$, $3.5\cdot10^{-5}$, $1.1\cdot10^{-6}$,
$6.1\cdot10^{-9}$ for $p = 1..5$ on eight cells) and with rate $\ge p$ under h-refinement —
the Nédélec interpolant of the wave is $O(h^p)$, so the temperature cannot do better than
the field it is driven by.

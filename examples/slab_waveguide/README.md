# Slab waveguide modes

**Physics.** A dielectric slab of thickness $d$ and index $n_{core}$ in a cladding of index
$n_{clad}$ guides modes $\mathbf{E}(x)\,e^{i\beta z}$ with effective index
$n_{\mathrm{eff}} = \beta/k_0$ between $n_{clad}$ and $n_{core}$. For the TE modes
($E = E_y(x)$) the effective index solves $\kappa\tan(\kappa d/2) = \gamma$ (even modes)
and $-\kappa\cot(\kappa d/2) = \gamma$ (odd modes) with $\kappa = k_0\sqrt{n_{core}^2 - n_{\mathrm{eff}}^2}$
and $\gamma = k_0\sqrt{n_{\mathrm{eff}}^2 - n_{clad}^2}$.

**What the program does.** Builds the cross-section as a strip with PEC walls in $y$
(which admits exactly the TE modes), solves the vector eigenproblem of
`physics::PropagatingMode` ($p = 3$) for the guided modes and prints their effective
indices next to the transcendental-equation values; the fundamental mode is written to
`slab_waveguide.vtu`.

**Expected result.** $k_0 d = 4$, $n = 1.5 / 1$: two guided TE modes, both effective indices
agreeing with the exact ones to better than $10^{-7}$; further eigenvalues below the
cladding index are box-confined radiation modes of the finite strip.

**Runtime.** About one second (release build).

```bash
cmake --build --preset release --target example_slab_waveguide
./build/release/examples/example_slab_waveguide
```

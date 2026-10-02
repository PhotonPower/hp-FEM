# PEC cavity eigenmodes

**Physics.** Resonances of a perfectly conducting box: the curl–curl eigenproblem
$\nabla\times\nabla\times\mathbf{E} = k^2\mathbf{E}$ with $\mathbf{n}\times\mathbf{E} = 0$ on all
walls. The eigenvalues of the unit square are $k^2 = \pi^2(m^2 + n^2)$, those of the unit
cube $\pi^2(m^2 + n^2 + l^2)$ (one TE or TM mode when one index vanishes, both when all
three are non-zero). The gradient kernel of the curl operator is removed by the discrete
gauge projector, so no spurious modes appear.

**What the program does.** Assembles the Nédélec stiffness and mass matrices on an $8\times8$
mesh ($p = 2$) and a $3\times3\times3$ mesh ($p = 1$), solves for the smallest eigenvalues with
the shift-invert Lanczos solver and prints them next to the analytic values; the first mode
of the square is written to `cavity_modes.vtu` (open in ParaView, show `E_re`).

**Expected result.** Relative eigenvalue errors of a few $10^{-4}$ on the square and a few
$10^{-2}$ on the coarse cube, all eigenvalues positive (no zero-frequency spurious modes);
see `tests/convergence/maxwell_cavity.cpp` for the rates $2p$.

**Runtime.** Below one second (release build).

```bash
cmake --build --preset release --target example_cavity_modes
./build/release/examples/example_cavity_modes
```

The same setup as a project file for the Python command line: `hpfem run examples/cavity_modes/project.json`
(see `docs/python.md`).

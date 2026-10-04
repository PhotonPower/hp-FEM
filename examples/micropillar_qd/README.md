# Quantum dot in a micropillar cavity (2.5D)

**Physics.** A micropillar is a GaAs / AlAs distributed-Bragg-reflector cavity etched into a
cylinder of a few micrometres diameter; a single InAs quantum dot at the antinode of the
one-wavelength cavity (design wavelength 940 nm) emits into the fundamental mode. The
structure is a body of revolution, so each azimuthal order m is a two-dimensional problem on
the meridian plane (`docs/theory/axisymmetric.md`, ADR-0010), which is why such cavities
are computed this way in practice: a 3D calculation of the same pillar would need two to
three orders of magnitude more unknowns.

- The **resonance** of the fundamental mode (HE11-like, m = 1) is the quasi-normal mode of
  `hpfem.AxisymmetricResonance` with the cylindrical PML: wavelength and
  Q = Re ω / (−2 Im ω).
- The **Purcell factor** F_P = P / P_bulk is the power the in-plane dipole radiates in the
  pillar over the power it radiates in bulk GaAs. The dipole on the axis has the orders
  m = ±1, equal by symmetry, so one solve of `hpfem.AxisymmetricScattering` with the
  Gaussian current `hpfem.axisymmetric_gaussian_dipole` per wavelength gives the spectrum;
  the bulk reference is the analytic Larmor power of the smeared dipole,
  n P₀ e^{−(nk)²σ²}.
- The **β factor** is the fraction of the emitted power leaving upwards through a plane
  above the pillar (`hpfem.axisymmetric_poynting_flux` through the plane and through the
  surface around the cavity layer).
- The **modal expansion** (`modal_spectrum`, `hpfem.AxisymmetricRieszProjection`) writes
  the same Purcell spectrum as a sum over the quasi-normal modes: the resonance pencil (PML
  frozen at the design wavelength) is expanded by Riesz projections around the modes of the
  resonance solver, and the emitted power at every wavelength follows from the contour
  samples without a further solve (`docs/theory/maxwell.md`, *Modal expansion by Riesz
  projection*). The share of the fundamental mode is a Lorentzian, the rest (the emission
  into the leaky continuum) a smooth background; the modal shares of the power add up to
  the total exactly.

**What the program does.** It builds a structured meridian mesh (x = r, y = z) with nodes
on every layer interface and on the pillar side wall, air around the pillar, GaAs substrate
below, PML on the outer sides (none on the axis), solves the m = 1 resonance nearest the
design wavelength (the highest Q within 3 %), then sweeps the wavelength over ±1.5
linewidths around it with the dipole at the cavity centre and reports F_P and β. The
Purcell spectrum must peak at the resonance with a width set by Q, which ties the
resonance solver and the source formulation together (the regression test checks it), and
the modal sum of the Riesz projection (one pole inside the background contour, 16 points
on its circle, 48 on the background, half-rule difference 4·10⁻⁶) must follow the sweep:
the two differ by the per-wavelength PML of the sweep against the frozen PML of the pencil
(2 % at the peak).

**Expected result** (full run: radius 1.0 µm, 10 top / 16 bottom pairs, p = 3,
170 k DoF per order):

| quantity | value |
|---|---|
| resonance (m = 1) | 934.61 nm, Q = 996 |
| Purcell factor at the resonance | 5.1, falling to 2.1 / 2.0 at ∓1.4 nm |
| full width at half maximum of F_P | 0.95 nm ≈ λ / Q |
| β (top) at the resonance | 0.47 |
| Purcell factor from the modal sum at the resonance | 5.03 (fundamental mode 3.37, background 1.66), within 2 % of the sweep across the range |

The Purcell peak sits on the resonance to the sampling step (0.12 nm) and its width agrees
with the linewidth of the quasi-normal mode; β follows the Purcell curve because the
off-resonant emission goes mostly sideways and into the substrate.

The quick configuration (radius 0.75 µm, 6 / 10 pairs, p = 2, 53 k DoF) gives

| quantity | value |
|---|---|
| resonance (m = 1) | 930.15 nm, Q = 171 |
| Purcell factor at the resonance | 2.3 (1.0 at −8 nm, 1.8 at +8 nm) |
| β (top) at the resonance | 0.24 |

The peak of the Purcell spectrum sits on the resonance wavelength; the spectrum is
asymmetric because the neighbouring higher-order m = 1 modes lie on the long-wavelength
side.

**Runtime.** About five minutes for the full run (resonance plus 25 wavelengths, release
build, MUMPS); `--quick` takes about one minute (resonance 2 s, nine wavelengths at 3 s each). Results go to `micropillar_qd.json`, the
spectrum with the resonance marked to `micropillar_qd.png`.

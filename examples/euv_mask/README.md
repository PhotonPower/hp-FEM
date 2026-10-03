# EUV mask: 3D absorber on a Mo/Si multilayer under oblique incidence

**Physics.** An EUV photomask is a Mo/Si Bragg mirror (bilayers of 6.9 nm, 40 of them in a
real mask, reflective around 13.5 nm) with a patterned absorber on top, illuminated at the
chief-ray angle of 6°. The absorber shadows and scatters the light, so the reflected near
field of a pattern — the input of the imaging optics — differs from the geometric shadow;
rigorous 3D electromagnetic simulation of such cells is the core of EUV scatterometry and
mask modelling. The example solves a unit cell of 32 nm pitch with a 16 nm tantalum pad of
16 nm height on a shortened mirror of six bilayers on the silicon substrate, with the
electric field perpendicular to the plane of incidence. Optical constants at 13.5 nm,
$n = 1 - \delta + i\beta$ from the CXRO tables (Henke, Gullikson & Davis 1993):
Mo $\delta = 0.0774$, $\beta = 0.00644$; Si $\delta = 0.00100$, $\beta = 0.00183$;
Ta $\delta = 0.0588$, $\beta = 0.0410$.

**What the program does.** It builds a structured tetrahedral mesh (six Kuhn tetrahedra per
cube) with nodes on every layer interface, Bloch-periodic in $x$ and $y$ with the phases of
the tilted plane wave, PML of one wavelength in the vacuum above and in the substrate, and
solves the scattered-field formulation at $p = 2$ (`Scattering3D`). The reflectivity of the
cell is the Poynting flux of the scattered field through a plane above the absorber over
the incident flux; the bare mirror is solved first and compared with the transfer-matrix
reflectivity of the same stack. The total near field is exported on the subdivided mesh
(`euv_mask.vtu`, about 10 MB) for ParaView; results go to `euv_mask.json`.

**Expected result** (6 bilayers, 8 cells across the pitch, 167k DoF):

| | reflectivity |
|---|---|
| bare mirror, FEM | 0.157 |
| bare mirror, transfer matrix | 0.150 |
| mirror with the 16 nm Ta pad | 0.127 |

The 4 % difference of the bare mirror at $p = 2$ with 4 nm cells shrinks to 0.2 % at
$p = 3$ (0.0445 against 0.0444 for three bilayers), i.e. the 3D discretisation converges
to the 1D reference; the pad covers a quarter of the cell and absorbs about a fifth of
the reflected power. A full 40-bilayer mirror reflects 0.72 at this angle (transfer
matrix) and needs the same vertical resolution over ten times the height.

**Runtime.** About 40 s for the two solves with the MUMPS backend (release build with
`HPFEM_ENABLE_MUMPS`); with Eigen's SparseLU the 167k-DoF 3D factorisation takes several
minutes. `--quick` (three bilayers, 8 nm lateral cells, 34k DoF) runs in about 6 s.

```bash
pip install -e ".[dev]"
python examples/euv_mask/run.py [--quick]
```

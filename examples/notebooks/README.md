# Example notebooks

Jupyter notebooks over the Python API (`docs/python.md`). They are committed without
outputs; `pytest python/tests/test_notebooks.py` executes their code cells, so they stay
in step with the API.

| notebook | physics | runtime |
|---|---|---|
| `01_mie_cylinder.ipynb` | plane wave on a dielectric cylinder: mesh with curved inclusion, scattered-field formulation with PML, scattering width against the Mie series (near field and far field), field plots, VTK export | ~5 s |
| `02_hp_adaptivity_lshape.ipynb` | corner singularity on the L-shape: estimate → Dörfler marking → hp decision by error prediction → `hp_refine`, exponential convergence in $N^{1/3}$, order map | ~10 s |
| `03_gold_nanowire_spectrum.ipynb` | absorption and scattering spectrum of a 100 nm gold wire with `hpfem.units` and the tabulated Johnson & Christy permittivity (`hpfem.materials`), the same sweep from a project file | ~20 s |

```bash
pip install -e ".[dev]" jupyter
jupyter lab examples/notebooks
```

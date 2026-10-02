"""The example notebooks run: their code cells are executed in order in a fresh namespace
(no Jupyter kernel needed), with the Agg backend and a temporary working directory."""

import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[2] / "examples" / "notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=[n.stem for n in NOTEBOOKS])
def test_notebook_runs(notebook, tmp_path, monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    content = json.loads(notebook.read_text(encoding="utf-8"))
    assert content["nbformat"] == 4
    namespace: dict = {"__name__": "__notebook__"}
    code_cells = [c for c in content["cells"] if c["cell_type"] == "code"]
    assert code_cells
    for cell in code_cells:
        assert cell["outputs"] == [] and cell["execution_count"] is None  # committed clean
        exec(compile("".join(cell["source"]), f"{notebook.name}:cell", "exec"), namespace)  # noqa: S102
    import matplotlib.pyplot as plt

    plt.close("all")
    assert "hpfem" in namespace


def test_notebooks_are_listed_in_the_readme():
    readme = (NOTEBOOKS[0].parent / "README.md").read_text(encoding="utf-8")
    for notebook in NOTEBOOKS:
        assert notebook.name in readme

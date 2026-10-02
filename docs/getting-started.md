# Getting started

```bash
git clone https://github.com/PhotonPower/hp-FEM.git && cd hp-FEM
./scripts/setup-dev.sh                  # Ubuntu; or open in the devcontainer
cmake --preset release && cmake --build --preset release
ctest --preset release
pip install -e ".[dev]" && python -c "import hpfem; print(hpfem.__version__)"
```

Developing with Claude Code: open the repository root; `CLAUDE.md` is picked up
automatically and contains conventions, commands and the Definition of Done.

#!/usr/bin/env bash
# One-time developer setup for Ubuntu 22.04/24.04 (also used by the devcontainer).
set -euo pipefail
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential g++ clang clang-format clang-tidy cmake ninja-build git \
  libeigen3-dev python3 python3-pip python3-venv doxygen graphviz \
  gmsh libmumps-seq-dev libopenblas-dev
pip3 install --user --break-system-packages pre-commit ruff pytest mkdocs-material mkdocs-bibtex pymdown-extensions
pre-commit install
echo "Done. Next: cmake --preset release && cmake --build --preset release && ctest --preset release"

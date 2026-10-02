#!/usr/bin/env bash
# Format all C++ and Python sources in place.
set -euo pipefail
cd "$(dirname "$0")/.."
find include src tests benchmarks examples python -name '*.hpp' -o -name '*.cpp' 2>/dev/null \
  | xargs -r clang-format -i
command -v ruff >/dev/null && ruff format python && ruff check --fix python || true

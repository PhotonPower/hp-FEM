"""Compact command reference for LLM agents and other readers without the repository.

The text is ``llm_reference.md`` next to this module, shipped inside the wheel and the sdist:
``hpfem.reference()`` returns it, ``hpfem reference`` (or ``python -m hpfem reference``) prints
it. Every qualified ``hpfem.*`` name in it is checked against the package by
``python/tests/test_reference.py``.
"""

from __future__ import annotations

from importlib import resources


def reference() -> str:
    """The compact command reference of the package (Markdown)."""
    return resources.files("hpfem").joinpath("llm_reference.md").read_text(encoding="utf-8")

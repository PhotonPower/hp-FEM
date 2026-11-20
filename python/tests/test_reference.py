"""The compact command reference shipped with the package (``hpfem.reference()``): it is part
of the installed package, the command line prints it, and every qualified ``hpfem.*`` name in it
exists, so the reference cannot drift from the API."""

import importlib
import re

import hpfem
from hpfem import cli

# qualified names: hpfem.Name, hpfem.module.name, hpfem.Class.member, ...
_QUALIFIED = re.compile(r"(?<![\w./-])hpfem(?:\.[A-Za-z_]\w*)+")


def _resolve(dotted: str) -> bool:
    parts = dotted.split(".")
    obj = hpfem
    for i, part in enumerate(parts[1:], start=1):
        if hasattr(obj, part):
            obj = getattr(obj, part)
            continue
        try:  # a submodule not imported by the package itself
            obj = importlib.import_module(".".join(parts[: i + 1]))
        except ImportError:
            return False
    return True


def test_reference_ships_with_the_package():
    text = hpfem.reference()
    assert text.startswith("#") and len(text.splitlines()) > 100
    assert "hpfem.reference()" in text


def test_command_line_prints_the_reference(capsys):
    assert cli.main(["reference"]) == 0
    assert capsys.readouterr().out.strip() == hpfem.reference().strip()


def test_every_qualified_name_exists():
    names = sorted(set(_QUALIFIED.findall(hpfem.reference())))
    assert len(names) > 50
    missing = [name for name in names if not _resolve(name)]
    assert not missing, f"names in llm_reference.md missing from the package: {missing}"

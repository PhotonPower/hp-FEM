"""``hpfem-gui`` (M15 F15): starts the Streamlit front end ("FEM model builder") with the
interpreter that has hpfem installed, so the GUI no longer needs a separate "Python with
hpfem" setting.

The app itself is not part of this package. ``hpfem-gui`` finds it, in this order:

1. ``hpfem-gui path/to/fem_app.py`` (an explicit script),
2. the environment variable ``HPFEM_GUI_APP`` (path of the script),
3. an installed package ``hpfem_gui`` (its ``app.py``, or the module attribute ``APP``),

and runs ``python -m streamlit run <app> [streamlit arguments]``. ``--dry-run`` prints the
command instead of running it; ``--version`` prints :func:`hpfem.version_info`. Exit codes:
0 ok, 2 no app or no Streamlit (with a hint what to install: ``pip install hpfem[gui]``).
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ENV_VAR = "HPFEM_GUI_APP"
PACKAGE = "hpfem_gui"


def find_app(explicit: str | None = None, env: dict | None = None) -> Path | None:
    """The app script by the rules of the module docstring, or ``None``."""
    env = os.environ if env is None else env
    if explicit:
        return Path(explicit).expanduser().resolve()
    configured = env.get(ENV_VAR)
    if configured:
        return Path(configured).expanduser().resolve()
    spec = importlib.util.find_spec(PACKAGE)
    if spec is not None:
        module = importlib.import_module(PACKAGE)
        app = getattr(module, "APP", None)
        if app:
            return Path(app).resolve()
        if spec.origin:
            candidate = Path(spec.origin).parent / "app.py"
            if candidate.exists():
                return candidate.resolve()
    return None


def streamlit_available() -> bool:
    return importlib.util.find_spec("streamlit") is not None


def command(app: Path, extra: list[str] | None = None) -> list[str]:
    """``[python, -m, streamlit, run, app, *extra]``."""
    return [sys.executable, "-m", "streamlit", "run", str(app), *(extra or [])]


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--version" in args:
        from hpfem import version_info

        print(json.dumps(version_info(), indent=2))
        return 0
    dry_run = "--dry-run" in args
    args = [a for a in args if a != "--dry-run"]
    explicit = None
    if args and not args[0].startswith("-"):
        explicit = args.pop(0)
    app = find_app(explicit)
    if app is None or not app.exists():
        where = f"{app} does not exist" if app is not None else "no app configured"
        print(
            f"hpfem-gui: {where}. Give the script as argument (hpfem-gui path/to/fem_app.py), "
            f"set {ENV_VAR}, or install the GUI package '{PACKAGE}'.",
            file=sys.stderr,
        )
        return 2
    if not streamlit_available():
        print(
            "hpfem-gui: Streamlit is not installed in this interpreter; "
            "run: pip install 'hpfem[gui]'",
            file=sys.stderr,
        )
        return 2
    cmd = command(app, args)
    if dry_run:
        print(" ".join(cmd))
        return 0
    return subprocess.call(cmd)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

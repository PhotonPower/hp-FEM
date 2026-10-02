"""Command line: ``hpfem run project.json [-o results.json]``, ``hpfem validate``,
``hpfem info``, ``hpfem materials`` (also ``python -m hpfem ...``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import hpfem
from hpfem import materials, project, units


def _info(_args) -> int:
    print(f"hpfem {hpfem.__version__}")
    print(f"OpenMP: {hpfem.has_openmp()}, threads: {hpfem.num_threads()}")
    print(
        "direct solver backends:",
        ", ".join(hpfem.backend_name(b) for b in hpfem.available_backends()),
    )
    return 0


def _materials(_args) -> int:
    for name in sorted(materials.library):
        material = materials.library[name]
        extra = ""
        if isinstance(material, materials.Tabulated):
            lo, hi = material.range
            extra = f" [{lo / units.um:.4g}-{hi / units.um:.4g} um]"
        elif isinstance(material, materials.Sellmeier):
            extra = f" [{material.range_um[0]:.4g}-{material.range_um[1]:.4g} um]"
        print(f"{name:8s} {material.source}{extra}")
    return 0


def _validate(args) -> int:
    try:
        project.validate(project.load(args.project))
    except (project.ProjectError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"{args.project}: ok")
    return 0


def _run(args) -> int:
    if args.quiet:
        hpfem.set_log_level("warn")
    if args.threads:
        hpfem.set_num_threads(args.threads)
    try:
        spec = project.load(args.project)
        results = project.run(spec)
    except (project.ProjectError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    text = json.dumps(results, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"results written to {args.output}")
    else:
        print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hpfem", description="hp-FEM for nano-optics")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a project file (JSON / YAML)")
    run.add_argument("project")
    run.add_argument("-o", "--output", help="write the results as JSON to this file")
    run.add_argument("-t", "--threads", type=int, default=0, help="OpenMP threads (0: default)")
    run.add_argument("-q", "--quiet", action="store_true", help="only warnings from the core")
    run.set_defaults(func=_run)
    validate = sub.add_parser("validate", help="check a project file without solving")
    validate.add_argument("project")
    validate.set_defaults(func=_validate)
    sub.add_parser("info", help="version, threads, solver backends").set_defaults(func=_info)
    sub.add_parser("materials", help="list the material library").set_defaults(func=_materials)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

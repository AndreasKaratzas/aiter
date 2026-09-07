# SPDX-License-Identifier: MIT
"""Run a registered native source generator from the selected AITER package."""

import importlib
import json
import sys
from pathlib import Path

from .context import BuildContext


def registry():
    records = json.loads((Path(__file__).parent / "registry.json").read_text())
    if type(records) is not dict or not records:
        raise ValueError("code generator registry must be a nonempty mapping")
    for name, record in records.items():
        if (
            type(name) is not str
            or not name
            or type(record) is not dict
            or set(record) != {"module", "description"}
            or type(record["description"]) is not str
            or type(record["module"]) is not str
            or not record["module"].startswith("aiter.codegen.")
        ):
            raise ValueError("invalid code generator registration")
    return records


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    records = registry()
    if not argv or argv[0] in ("--help", "-h", "--list"):
        print("Usage: python -m aiter.codegen GENERATOR --output DIRECTORY [OPTIONS]")
        print("\nGenerators:")
        for name, record in sorted(records.items()):
            print(f"  {name:38} {record['description']}")
        return 0
    name = argv.pop(0)
    if name not in records:
        raise ValueError(f"unknown generator {name!r}; use --list")
    if not any(flag in argv for flag in ("--help", "-h")):
        output_flags = ("--output", "--working_path", "--output_dir", "--out-dir", "-w")
        if not any(
            arg in output_flags
            or any(arg.startswith(flag + "=") for flag in output_flags)
            for arg in argv
        ):
            raise ValueError("code generation requires an explicit --output directory")
    context = BuildContext.load()
    module = importlib.import_module(records[name]["module"])
    return module.main(argv, context=context)


if __name__ == "__main__":
    sys.exit(main())

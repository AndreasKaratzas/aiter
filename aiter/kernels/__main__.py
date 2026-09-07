# SPDX-License-Identifier: MIT
"""Inspect, verify or admit an explicitly selected kernel catalog."""

import argparse
import json
import os
import subprocess
import sys

from .catalog import KernelCatalog
from .store import KernelStore


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    program = []
    if "--" in argv:
        separator = argv.index("--")
        program, argv = argv[separator + 1 :], argv[:separator]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "admit", "run"))
    parser.add_argument("--root", help="original managed resource directory")
    parser.add_argument("--target", action="append", dest="targets")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--cache-root")
    args = parser.parse_args(argv)
    if (args.command == "run") != bool(program):
        parser.error("run requires -- followed by an explicit native program")
    if args.root is None:
        from aiter.codegen.context import BuildContext

        args.root = BuildContext.load().resource("kernels")
    catalog = KernelCatalog.load(args.root)
    targets = args.targets or catalog.targets
    report = catalog.verify(targets=targets, require_complete=args.require_complete)
    if args.command in ("admit", "run"):
        receipt = KernelStore(args.cache_root).admit(
            catalog, targets=targets, require_complete=args.require_complete
        )
        report["admission"] = receipt.as_dict()
        report["environment"] = receipt.environment()
        if args.command == "run":
            result = subprocess.run(
                program, env={**os.environ, **receipt.environment()}, check=False
            )
            return (
                result.returncode if result.returncode >= 0 else 128 - result.returncode
            )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

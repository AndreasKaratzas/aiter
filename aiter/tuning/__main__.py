# SPDX-License-Identifier: MIT
"""Run an explicit offline search; importing tuning does not load GPU libraries."""

import runpy
import sys

from .search.registry import searches


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    records = searches()
    if not argv or argv[0] in ("--list", "--help", "-h"):
        print("Usage: python -m aiter.tuning SEARCH [OPTIONS]")
        print("\nOffline searches:")
        for name in sorted(records):
            print(f"  {name}")
        print(
            "\nSearch results remain trial data until an approved manifest selects them."
        )
        return 0
    name = argv.pop(0)
    if name not in records:
        raise ValueError(f"unknown tuning search {name!r}; use --list")
    previous = sys.argv
    try:
        sys.argv = [records[name]["module"], *argv]
        runpy.run_module(records[name]["module"], run_name="__main__", alter_sys=True)
    finally:
        sys.argv = previous
    return 0


if __name__ == "__main__":
    sys.exit(main())

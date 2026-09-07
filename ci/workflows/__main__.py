"""Generate or verify GitHub entrypoints from their hierarchical sources."""

import argparse
import sys
from pathlib import Path

from ci.workflows import generate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--check", action="store_true", help="reject generated drift without writing"
    )
    action.add_argument(
        "--write", action="store_true", help="update generated entrypoints and indexes"
    )
    parser.add_argument("--root", type=Path, help="explicit repository/control root")
    args = parser.parse_args(argv)
    try:
        generate(root=args.root, check=args.check, write=args.write)
    except (ValueError, TypeError, KeyError, OSError, UnicodeError) as error:
        print(f"workflows: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

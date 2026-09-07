# SPDX-License-Identifier: MIT
"""Inspect recorded GPU traces without importing Torch or running a benchmark."""

import argparse
import sys
import tempfile
from pathlib import Path

from .report import CATEGORIES, analyze


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="JSON/JSON.gz trace or directory of traces"
    )
    parser.add_argument(
        "--output", type=Path, help="new output directory outside the checkout"
    )
    parser.add_argument(
        "--kernel", default="all", help="kernel-name substring; default all"
    )
    parser.add_argument("--category", action="append", choices=CATEGORIES)
    args = parser.parse_args(argv)
    output = args.output or Path(tempfile.mkdtemp(prefix="aiter-traces-")) / "report"
    try:
        result = analyze(
            args.input,
            output,
            kernel=args.kernel,
            categories=args.category or CATEGORIES,
        )
    except (ValueError, TypeError, OSError) as error:
        print(f"trace analysis: {error}", file=sys.stderr)
        return 2
    print(
        f"Retained {result['event_count']} events from {len(result['inputs'])} traces in {output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

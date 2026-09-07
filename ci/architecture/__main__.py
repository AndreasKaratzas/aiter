"""Inspect and enforce the repository's dependency and layout rules."""

import argparse
import json
import sys
from pathlib import Path

from ci.common.json import write_json

from .check import check_repository
from .policy import load_policy


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "show"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        policy = load_policy(args.policy)
        if args.command == "show":
            print(json.dumps(policy, indent=2))
            return 0
        report = check_repository(args.root, policy)
        if args.output:
            write_json(args.output, report)
        print(
            f"Architecture: {report['status']} ({report['python_files']} Python files, "
            f"{report['dependency_edges']} dependency edges)"
        )
        for issue in report["violations"]:
            print(
                f"{issue['path']}:{issue['line']}: {issue['rule']}: "
                f"{issue['target'] or ''} — {issue['reason']}"
            )
        print(
            f"Computed dynamic imports requiring review: {len(report['unresolved_dynamic_imports'])}"
        )
        return 0 if report["status"] == "PASS" else 1
    except (ValueError, TypeError, KeyError, OSError) as error:
        print(f"architecture: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

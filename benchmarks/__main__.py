# SPDX-License-Identifier: MIT
"""Measure an operator and retain every observation, including failed cases."""

import argparse
import json
import traceback
from pathlib import Path

from benchmarks.operators.rmsnorm import measure


def shape(value):
    try:
        rows, columns = (int(part) for part in value.lower().split("x"))
        if min(rows, columns) < 1:
            raise ValueError
        return rows, columns
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "shape must be positive MxN, such as 256x4096"
        ) from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", choices=("rmsnorm",), default="rmsnorm")
    parser.add_argument("--shape", type=shape, action="append")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), action="append")
    parser.add_argument(
        "--output", type=Path, help="new directory for raw case records"
    )
    args = parser.parse_args()
    for name, values in (("shape", args.shape), ("dtype", args.dtype)):
        if values is not None and len(values) != len(set(values)):
            parser.error(f"duplicate --{name} would overwrite a case record")
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for rows, columns in args.shape or ((1, 4096), (256, 4096), (1024, 8192)):
        for dtype in args.dtype or ("float16", "bfloat16"):
            try:
                result = measure(rows, columns, dtype)
            except Exception:  # noqa: BLE001 -- retain failures and return nonzero
                result = {
                    "shape": [rows, columns],
                    "dtype": dtype,
                    "comparison": {"passed": False},
                    "error": traceback.format_exc(),
                }
            results.append(result)
            if args.output is not None:
                (args.output / f"{rows}x{columns}-{dtype}.json").write_text(
                    json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
                )
    report = {
        "schema_version": 1,
        "benchmark": args.operator,
        "results": results,
        "passed": all(result["comparison"]["passed"] for result in results),
    }
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        (args.output / "report.json").write_text(text)
    print(text, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

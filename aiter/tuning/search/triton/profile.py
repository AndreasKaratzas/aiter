# SPDX-License-Identifier: MIT
"""Summarize selected kernels from a rocprof trace, retaining every sample."""

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


def summarize(path, keywords=("gemm",)):
    path = Path(path)
    payload = path.read_bytes()
    groups = []
    current = {}
    for row in csv.DictReader(payload.decode().splitlines()):
        name = row["Kernel_Name"]
        if name == "split_dummy":
            if not current:
                raise ValueError("trace group contains no matching kernel samples")
            counts = {len(samples) for samples in current.values()}
            if len(counts) != 1:
                raise ValueError("selected kernels have unequal repetition counts")
            totals = [sum(values) for values in zip(*current.values())]
            groups.append(
                {
                    "kernels_ns": current,
                    "samples_ns": totals,
                    "median_us": statistics.median(totals) / 1000,
                }
            )
            current = {}
        elif any(keyword in name for keyword in keywords):
            duration = int(row["End_Timestamp"]) - int(row["Start_Timestamp"])
            if duration <= 0:
                raise ValueError("trace duration must be positive")
            current.setdefault(name, []).append(duration)
    if current:
        raise ValueError("trace ends without its profiling completion marker")
    if not groups:
        raise ValueError("trace contains no completed profiling groups")
    return {
        "schema_version": 1,
        "scope": "selected kernel durations; excludes host and unmatched kernels",
        "source": str(path.resolve()),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "keywords": list(keywords),
        "groups": groups,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename", type=Path)
    parser.add_argument("-k", nargs="+", default=["gemm"])
    parser.add_argument("-m", type=float, default=0, help="bytes moved, in GB")
    parser.add_argument("-f", type=float, default=0, help="operation count, in TFLOP")
    parser.add_argument("--output", type=Path, help="new JSON file for all samples")
    args = parser.parse_args(argv)
    report = summarize(args.filename, args.k)
    if args.output is not None:
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    for group in report["groups"]:
        print("Kernel detected:")
        for kernel in group["kernels_ns"]:
            print(f"\t{kernel}")
        latency = group["median_us"]
        print(f"{latency:.6f} (us)")
        if args.m > 0:
            print(f"{args.m / latency * 1e6:.2f} (GB/s)")
        if args.f > 0:
            print(f"{args.f / latency * 1e6:.2f} (TFLOP/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

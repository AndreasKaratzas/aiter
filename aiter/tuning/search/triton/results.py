# SPDX-License-Identifier: MIT
"""Collect measured configurations without installing a runtime selection."""

import argparse
import hashlib
import json
from pathlib import Path

from .parameters import get_config_list, read_screen_file
from .registry import resolve


def collect(driver, dimensions, directory, max_m):
    records = []
    for n, k in dimensions:
        m = 1
        while m <= max_m:
            source = directory / f"screen-{driver.name}-{m}-{n}-{k}.log"
            if source.is_file():
                cases = []
                read_screen_file(source, cases)
                if not cases:
                    raise ValueError(f"no completed measurements in {source}")
                observations = [
                    {
                        "selected_kernel_median_us": latency,
                        "config": get_config_list(parameters.split())[0],
                    }
                    for latency, parameters in cases
                ]
                records.append(
                    {
                        "shape": [m, n, k],
                        "source": str(source.resolve()),
                        "source_sha256": hashlib.sha256(
                            source.read_bytes()
                        ).hexdigest(),
                        "observations": observations,
                        "fastest_observation": min(
                            range(len(cases)), key=lambda i: cases[i][0]
                        ),
                    }
                )
            m *= 2
    if not records:
        raise ValueError("no measured workloads match the requested dimensions")
    return {
        "schema_version": 1,
        "driver": driver.name,
        "config_family": driver.config,
        "scope": "offline selected-kernel timing; numerical qualification is separate",
        "runtime_selection": False,
        "workloads": records,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("driver")
    parser.add_argument("--n-list", nargs="+", type=int, required=True)
    parser.add_argument("--k-list", nargs="+", type=int, required=True)
    parser.add_argument("--max-m", type=int, default=131072)
    parser.add_argument("--directory", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, help="new file; otherwise print JSON")
    args = parser.parse_args(argv)
    if (
        len(args.n_list) != len(args.k_list)
        or min(*args.n_list, *args.k_list, args.max_m) <= 0
    ):
        parser.error(
            "N and K lists must pair positive dimensions; max-M must be positive"
        )
    report = collect(
        resolve(args.driver), zip(args.n_list, args.k_list), args.directory, args.max_m
    )
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        with args.output.open("x") as stream:
            stream.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

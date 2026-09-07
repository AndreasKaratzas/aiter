# SPDX-License-Identifier: MIT
"""Profile the currently selected implementation of a registered workload."""

import argparse
import json
import tempfile
from pathlib import Path

from .execution import run
from .profile import summarize
from .registry import resolve


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("driver")
    parser.add_argument("--output", type=Path, help="new directory for trace and logs")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args(argv)
    if min(args.M, args.N, args.K, args.timeout) <= 0:
        parser.error("dimensions and timeout must be positive")
    driver = resolve(args.driver)
    output = args.output
    if output is None:
        output = Path(tempfile.mkdtemp(prefix="aiter-profile-"))
    else:
        output.mkdir(parents=True, exist_ok=False)
    output = output.resolve()
    command = [
        "rocprofv3",
        "--kernel-trace",
        "-f",
        "csv",
        "-d",
        str(output),
        "-o",
        "selected",
        "--",
        *driver.command((args.M, args.N, args.K)),
    ]
    print(f"Profiling output: {output}", flush=True)
    execution = run(command, timeout=args.timeout)
    (output / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
    (output / "execution.log").write_text(execution["stdout"] + execution["stderr"])
    if execution["timed_out"] or execution["returncode"] != 0:
        raise RuntimeError(f"profiling failed; see {output / 'execution.json'}")
    report = summarize(output / "selected_kernel_trace.csv")
    report.update(driver=driver.name, shape=[args.M, args.N, args.K], command=command)
    (output / "profile.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Selected kernel median: {report['groups'][0]['median_us']:.6f} us")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

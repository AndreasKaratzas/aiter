"""Execute a declared vLLM latency case inside its resolved canary image."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from benchmarks.vllm.latency import measure_case
from ci.common.json import load_json, require, write_json
from ci.qualification.environments import framework_observation


def execute(request: Path) -> None:
    value = load_json(request)
    output = request.parent
    require(
        value["classification"] == "rolling-upstream-canary", "invalid canary request"
    )
    subprocess.run(
        [
            sys.executable,
            "/control/docker/common/wheel.py",
            "install",
            "--wheelhouse",
            "/wheelhouse",
            "--framework",
            "vllm",
        ],
        check=True,
    )
    observation = framework_observation("vllm")
    write_json(output / "framework.json", observation)
    measured = measure_case(value["case"], output)
    write_json(
        output / "measurement.json",
        {
            "classification": "rolling-upstream-canary",
            **measured,
            "framework": observation,
            "wheel": value["wheel"],
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    execute(parser.parse_args(argv).request)


if __name__ == "__main__":
    main()

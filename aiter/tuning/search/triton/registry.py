# SPDX-License-Identifier: MIT
"""Explicit Triton profiling programs and their legacy configuration families."""

import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Driver:
    name: str
    module: str
    config: str

    def command(self, dimensions, parameters=()):
        return [
            sys.executable,
            "-m",
            self.module,
            *map(str, dimensions),
            *map(str, parameters),
        ]


def drivers():
    records = json.loads(Path(__file__).with_name("drivers.json").read_text())
    return {
        name: Driver(name, record["module"], record["config"])
        for name, record in records.items()
    }


def resolve(name):
    records = json.loads(Path(__file__).with_name("drivers.json").read_text())
    for key, record in records.items():
        if name in (key, record["legacy"]):
            return Driver(key, record["module"], record["config"])
    raise ValueError(
        f"unknown profiling driver {name!r}; choose from {', '.join(records)}"
    )

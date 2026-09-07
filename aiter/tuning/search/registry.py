# SPDX-License-Identifier: MIT
"""Named offline searches and their explicit native build-module associations."""

import json
from pathlib import Path


def searches():
    records = json.loads(Path(__file__).with_suffix(".json").read_text())
    if type(records) is not dict or not records:
        raise ValueError("tuning registry must be a nonempty mapping")
    seen = set()
    for name, record in records.items():
        if (
            type(name) is not str
            or not name
            or type(record) is not dict
            or set(record) != {"module", "build_modules"}
            or type(record["module"]) is not str
            or not record["module"].startswith("aiter.tuning.search.")
            or type(record["build_modules"]) is not list
        ):
            raise ValueError("invalid tuning registry entry")
        for module in record["build_modules"]:
            if type(module) is not str or not module.isidentifier() or module in seen:
                raise ValueError("invalid or duplicated native tuning module")
            seen.add(module)
    return records


def for_build(module):
    return next(
        (
            record["module"]
            for record in searches().values()
            if module in record["build_modules"]
        ),
        None,
    )

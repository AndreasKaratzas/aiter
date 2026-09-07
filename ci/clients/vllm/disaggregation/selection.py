"""Review model/topology areas independently from GitHub scheduling and Slurm."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from pathlib import Path

from ci.common.json import require


def configuration():
    return json.loads(Path(__file__).with_name("selection.json").read_text())


def cases(pipeline: bytes, image: str) -> list[dict]:
    config = configuration()
    require(
        re.fullmatch(r"[^\s]+@sha256:[a-f0-9]{64}", image),
        "select a pinned image digest",
    )
    require(
        hashlib.sha256(pipeline).hexdigest() == config["source"]["sha256"],
        "upstream disaggregation pipeline differs from the reviewed revision",
    )
    selected, label = {}, None
    # The exact pinned source format is admitted above. This deliberately
    # recognizes its command rows, not a general-purpose YAML/shell language.
    for line in pipeline.decode().splitlines():
        match = re.fullmatch(r'\s*- label: "([^"]+)"', line)
        if match:
            label = match[1]
        elif (
            line.lstrip().startswith("- ")
            and label
            and label.endswith("-vllm-router")
            and "MODEL_NAME=" in line
        ):
            tokens = shlex.split(line.strip().removeprefix("- "))
            split = tokens.index("bash")
            require(
                tokens[split:] == ["bash", config["script"]],
                "unrecognized upstream launcher",
            )
            pairs = [token.split("=", 1) for token in tokens[:split]]
            require(all(len(pair) == 2 for pair in pairs), "invalid command assignment")
            environment = dict(pairs)
            require(len(environment) == len(pairs), "duplicate command assignment")
            environment["IMAGE"] = image
            require(label not in selected, "duplicate upstream case")
            selected[label] = environment
    result = []
    for model in config["models"]:
        require(re.fullmatch(r"[A-Za-z0-9_.-]+", model), "invalid model area")
        for topology, topology_env in config["topologies"].items():
            label = f"{model}-PD-{topology}-MoRIIO-vllm-router"
            environment = {
                "IMAGE": image,
                "MODEL_NAME": model,
                **topology_env,
                **config["environment"],
            }
            require(
                selected.pop(label, None) == environment,
                f"upstream case differs: {label}",
            )
            result.append(
                {
                    "case_name": label,
                    "case_slug": f"{model}-{topology}",
                    "model_name": model,
                    "topology": topology,
                    "nodes": int(environment["NODES"]),
                    "environment": environment,
                    "argv": ["bash", config["script"]],
                }
            )
    require(not selected, "unreviewed upstream router cases remain")
    return result


def select_case(items: list[dict], slug: str) -> dict:
    found = [case for case in items if case["case_slug"] == slug]
    require(len(found) == 1, "select exactly one reviewed disaggregation case")
    return found[0]

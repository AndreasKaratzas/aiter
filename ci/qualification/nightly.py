"""Observe one freshly installed nightly interpreter, before any model workload."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

from ci.common.json import load_json, require, write_json
from ci.qualification.distributions import installed_distributions
from ci.qualification.environments import observe_requirements, validate_lock
from ci.qualification.substitution import (
    bind_policy,
    verify_pip_result,
)
from ci.qualification.substitution import (
    observe as observe_substitution,
)
from ci.release.artifacts import hash_file


def verified_record(distribution) -> str:
    record = distribution.read_text("RECORD")
    require(bool(record), "installed nightly distribution has no RECORD")
    for name, declared_hash, declared_size in csv.reader(io.StringIO(record)):
        if not declared_hash:
            require(
                name.endswith(("/RECORD", ".pyc")), "unhashed installed nightly payload"
            )
            continue
        algorithm, encoded = declared_hash.split("=", 1)
        require(algorithm == "sha256", "unsupported installed artifact digest")
        path = Path(distribution.locate_file(name)).resolve()
        require(
            path.is_relative_to(Path(sys.prefix).resolve()),
            "installed RECORD escaped private environment",
        )
        require(
            path.is_file() and path.stat().st_size == int(declared_size),
            "installed nightly payload size changed",
        )
        actual = bytes.fromhex(hash_file(path)[0])
        require(
            base64.urlsafe_b64encode(actual).decode().rstrip("=") == encoded,
            "installed nightly payload bytes changed",
        )
    return hashlib.sha256(record.encode()).hexdigest()


def inventory() -> list[dict]:
    result = []
    for distribution in installed_distributions().values():
        name = distribution.metadata["Name"].lower().replace("_", "-")
        payload_digest = (
            verified_record(distribution) if name in ("vllm", "amd-aiter") else None
        )
        result.append(
            {
                "payload_record_sha256": payload_digest,
                "name": name,
                "version": distribution.version,
                "metadata_sha256": hashlib.sha256(
                    (distribution.read_text("METADATA") or "").encode()
                ).hexdigest(),
                "record_sha256": hashlib.sha256(
                    (distribution.read_text("RECORD") or "").encode()
                ).hexdigest(),
                "direct_url": json.loads(
                    distribution.read_text("direct_url.json") or "null"
                ),
            }
        )
    require(
        len({item["name"] for item in result}) == len(result),
        "duplicate installed distributions",
    )
    return sorted(result, key=lambda item: item["name"])


def observe(
    request: dict,
    resolution: dict,
    install_report: dict,
    pip_result: dict,
    pip_output: str,
) -> dict:
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm._aiter_ops import rocm_aiter_ops

    import aiter

    require(
        sys.flags.optimize == 0, "nightly import interpreter must preserve assertions"
    )
    require(
        sys.prefix != sys.base_prefix,
        "nightly import must use its private virtual environment",
    )
    require(torch.version.hip is not None, "installed Torch is not a ROCm build")
    actual = observe_requirements(None, "vllm", torch)
    framework = actual["frameworks"]["vllm"]
    require(
        framework["revision"] == resolution["revision"],
        "installed vLLM does not match the resolved nightly commit",
    )
    policy = bind_policy(request["candidate_substitution"], request["artifact"])
    substitution = observe_substitution(policy)
    dependency_acceptance = verify_pip_result(pip_result, pip_output, substitution)
    for name, module in (("amd-aiter", aiter), ("vllm", vllm)):
        distribution = metadata.distribution(name)
        require(
            Path(module.__file__).resolve()
            == Path(
                distribution.locate_file(module.__name__ + "/__init__.py")
            ).resolve(),
            "imported package differs from installed distribution",
        )
        require(
            Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()),
            "import escaped private nightly environment",
        )
    candidate = json.loads(
        metadata.distribution("amd-aiter").read_text("direct_url.json") or "{}"
    )
    require(
        candidate.get("archive_info", {}).get("hashes", {}).get("sha256")
        == request["artifact"]["sha256"],
        "consumer replaced the selected candidate AITER wheel",
    )
    vllm_records = [
        item
        for item in install_report["install"]
        if item["metadata"]["name"].lower() == "vllm"
    ]
    require(
        len(vllm_records) == 1
        and vllm_records[0]["download_info"]["url"] == resolution["wheel_url"],
        "installer selected a different vLLM artifact",
    )
    lock = validate_lock(
        {
            "schema_version": 1,
            "id": "vllm-rolling-nightly",
            "status": "development",
            "image": None,
            "python": ".".join(platform.python_version().split(".")[:2]),
            "rocm": ".".join(torch.version.hip.split(".")[:2]),
            "packages": actual["packages"],
            "torch_revision": actual["torch_revision"],
            "candidate_substitution": policy,
            "frameworks": {
                "vllm": {
                    "version": framework["version"],
                    "revision": framework["revision"],
                }
            },
        }
    )
    return {
        "status": "PASS",
        "environment_lock": lock,
        "observation": actual,
        "dependency_acceptance": dependency_acceptance,
        "candidate": candidate,
        "distributions": inventory(),
        "python": sys.executable,
        "imports": {
            "aiter": aiter.__file__,
            "vllm": vllm.__file__,
            "LLM": LLM.__module__,
            "SamplingParams": SamplingParams.__module__,
            "rms_norm_bridge": callable(rocm_aiter_ops.rms_norm),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--resolution", type=Path)
    parser.add_argument("--install-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--pip-result", type=Path)
    parser.add_argument("--pip-log", type=Path)
    parser.add_argument("--verify-policy", type=Path)
    args = parser.parse_args(argv)
    if args.verify_policy:
        policy = load_json(args.verify_policy)["environment_lock"][
            "candidate_substitution"
        ]
        write_json(
            args.output,
            verify_pip_result(
                load_json(args.pip_result),
                args.pip_log.read_text(),
                observe_substitution(policy),
            ),
        )
        return
    record = (
        inventory()
        if args.inventory
        else observe(
            load_json(args.request),
            load_json(args.resolution),
            load_json(args.install_report),
            load_json(args.pip_result),
            args.pip_log.read_text(),
        )
    )
    write_json(args.output, record)


if __name__ == "__main__":
    main()

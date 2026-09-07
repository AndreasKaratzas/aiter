"""Provision reviewed real-model inputs before entering selected tests."""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

from ci.common.json import require, write_json


def provision_models(
    plan: dict,
    request: dict,
    controls: Path,
    output: Path,
    *,
    python: str | None = None,
) -> None:
    manifests = sorted(
        {
            group["model_manifest"]
            for group in plan["groups"].values()
            if "model_manifest" in group
        }
    )
    if manifests:
        model_cache = output / "cache/models"
        model_cache.mkdir(parents=True, exist_ok=False)
        os.environ["HF_HUB_CACHE"] = str(model_cache)
        provisioning = output / "model-provisioning"
        provisioning.mkdir()
        for index, relative in enumerate(manifests):
            manifest = controls / relative
            require(
                manifest.is_file()
                and manifest.resolve().is_relative_to(controls.resolve())
                and not manifest.is_symlink(),
                "model manifest is missing from reviewed controls",
            )
            write_json(
                provisioning / f"{index}-request.json",
                {
                    "manifest": relative,
                    "manifest_sha256": hashlib.sha256(
                        manifest.read_bytes()
                    ).hexdigest(),
                    "control_source": request["control_source"],
                    "plan_digest": plan["plan_digest"],
                },
            )
            environment = dict(os.environ, PYTHONPATH=str(controls / "tests"))
            with (provisioning / f"{index}.log").open("w") as log:
                subprocess.run(
                    [
                        python or sys.executable,
                        "-m",
                        "common.models",
                        "--manifest",
                        str(manifest),
                        "--download",
                        "--cache-dir",
                        str(model_cache),
                        "--output",
                        str(provisioning / f"{index}-receipt.json"),
                    ],
                    cwd=controls,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=3600,
                    check=True,
                )

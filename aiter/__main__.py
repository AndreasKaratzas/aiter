# SPDX-License-Identifier: MIT
"""Inspect an AITER installation without compiling a kernel."""

import argparse
import importlib.metadata
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path


def installation_info(gpu=False):
    root = Path(__file__).resolve().parent
    versions = {}
    for name in ("amd-aiter", "torch", "triton", "pytorch-triton-rocm", "flydsl"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {
        "package_path": str(root),
        "installed_distribution_versions": versions,
        "native_manifest": (
            str(root / "lib/manifest.json")
            if (root / "lib/manifest.json").is_file()
            else None
        ),
        "jit_cache_override": os.environ.get("AITER_JIT_DIR"),
        "source_revision": None,
    }
    if (root.parent / ".git").exists():
        result["source_revision"] = subprocess.check_output(
            ["git", "-C", str(root.parent), "rev-parse", "HEAD"], text=True
        ).strip()
        result["source_modified"] = bool(
            subprocess.check_output(
                ["git", "-C", str(root.parent), "status", "--porcelain"]
            )
        )
    if gpu:
        import torch

        result["hip_version"] = torch.version.hip
        result["devices"] = [
            {
                "index": index,
                "name": props.name,
                "target": getattr(props, "gcnArchName", None),
                "compute_units": props.multi_processor_count,
                "memory_bytes": props.total_memory,
            }
            for index in range(torch.cuda.device_count())
            for props in (torch.cuda.get_device_properties(index),)
        ]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser(
        "doctor", help="show package, dependency and optional GPU information"
    )
    doctor.add_argument(
        "--gpu",
        action="store_true",
        help="also initialize Torch to inspect available devices",
    )
    commands.add_parser(
        "operators",
        help="show operation domains, dependencies and execution interfaces",
    )
    manifest = commands.add_parser(
        "manifest", help="validate and print a pinned dispatch manifest"
    )
    manifest.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            result = installation_info(args.gpu)
        elif args.command == "operators":
            from .api import operator_domains

            result = [asdict(domain) for domain in operator_domains()]
        else:
            from .tuning import DispatchManifest

            value = DispatchManifest.read(args.path)
            result = {**value.to_dict(), "digest": value.digest}
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ValueError, OSError, ImportError) as error:
        parser.exit(1, f"aiter: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

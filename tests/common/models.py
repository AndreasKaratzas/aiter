# SPDX-License-Identifier: MIT
"""Provision pinned model inputs through the optional Hugging Face adapter."""

import argparse
import json
from pathlib import Path

from ci.common.checkpoints import ModelUnavailable, load_manifest, verify_snapshot

__all__ = ["ModelUnavailable", "load_manifest", "resolve_model", "verify_snapshot"]


def resolve_model(model, *, download=False, cache_dir=None, destination=None):
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError as error:
        raise ModelUnavailable(
            "Model fixtures require huggingface_hub installed."
        ) from error
    try:
        snapshot = snapshot_download(
            model["repository"],
            revision=model["revision"],
            allow_patterns=list(model["files"]),
            local_files_only=not download,
            cache_dir=cache_dir,
        )
    except LocalEntryNotFoundError as error:
        raise ModelUnavailable(
            f"Provision {model['repository']}@{model['revision']} with python -m common.models before running E2E tests."
        ) from error
    return verify_snapshot(snapshot, model, destination=destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    models = load_manifest(args.manifest)
    selected = args.model or list(models)
    unknown = set(selected) - models.keys()
    if unknown:
        parser.error(f"Unknown model fixtures: {sorted(unknown)}")
    receipts = {
        name: resolve_model(
            models[name], download=args.download, cache_dir=args.cache_dir
        )
        for name in selected
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipts, indent=2) + "\n")


if __name__ == "__main__":
    main()

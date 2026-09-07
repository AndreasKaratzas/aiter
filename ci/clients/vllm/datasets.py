# SPDX-License-Identifier: MIT
"""Provision declared public vLLM evaluation data before offline test execution."""

import argparse
import hashlib
import json
from pathlib import Path


PUBLIC_GROUP_INPUTS = {
    "vllm-chartqa": "tests/frameworks/vllm/evaluation/fixtures/chartqa.json",
}


def required_inputs(groups, *, root=None):
    """Only named public inputs are provisioned; GPQA has no automatic admission."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[3]
    inputs = []
    for group in sorted(set(groups) & PUBLIC_GROUP_INPUTS.keys()):
        relative = PUBLIC_GROUP_INPUTS[group]
        manifest = root / relative
        declaration = json.loads(manifest.read_text())
        source = declaration["source"]
        if (
            type(declaration.get("schema_version")) is not int
            or declaration["schema_version"] != 1
            or source.get("repository") != "HuggingFaceM4/ChartQA"
            or len(source.get("revision", "")) != 40
            or any(c not in "0123456789abcdef" for c in source["revision"])
            or len(source.get("sha256", "")) != 64
            or any(c not in "0123456789abcdef" for c in source["sha256"])
            or type(source.get("size")) is not int
            or source["size"] <= 0
            or Path(source["file"]).is_absolute()
            or ".." in Path(source["file"]).parts
        ):
            raise ValueError("Invalid declared public ChartQA input")
        inputs.append(
            {
                "group": group,
                "declaration": relative,
                "declaration_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "source": source,
            }
        )
    return inputs


def provision(groups, *, cache_dir=None, root=None, download=None):
    inputs = required_inputs(groups, root=root)
    if inputs and download is None:
        from huggingface_hub import hf_hub_download

        download = hf_hub_download
    records = []
    for item in inputs:
        source = item["source"]
        path = Path(
            download(
                source["repository"],
                source["file"],
                repo_type="dataset",
                revision=source["revision"],
                cache_dir=cache_dir,
            )
        )
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if (
            path.stat().st_size != source["size"]
            or digest.hexdigest() != source["sha256"]
        ):
            raise ValueError(
                "Provisioned ChartQA bytes differ from the declared source"
            )
        records.append(
            {
                **item,
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    return {
        "schema_version": 1,
        "classification": "public dataset input admission, not model execution",
        "groups": sorted(set(groups)),
        "inputs": records,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups", required=True, help="Comma-separated selected qualification groups"
    )
    parser.add_argument("--cache-dir")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    groups = [value.strip() for value in args.groups.split(",") if value.strip()]
    args.output.write_text(
        json.dumps(provision(groups, cache_dir=args.cache_dir), indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

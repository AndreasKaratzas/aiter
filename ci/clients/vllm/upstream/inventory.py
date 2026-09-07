"""Read upstream Buildkite selections without importing or executing upstream tests."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

LABELS = {
    "entrypoints": "Entrypoints Integration",
    "lm_eval": "LM Eval",
    "qwen3": "Qwen3",
    "language": "Language Models",
    "multimodal": "Multimodal Models",
    "quantized": "Quantized Models",
    "spec_decode": "Spec Decode",
    "v1": "V1 ",
}
WHOLE_AREAS = {
    "lm_eval": "lm_eval.yaml",
    "language": "models_language.yaml",
    "multimodal": "models_multimodal.yaml",
    "spec_decode": "spec_decode.yaml",
}
FIELDS = {
    "key",
    "label",
    "commands",
    "command",
    "working_dir",
    "device",
    "num_devices",
    "num_gpus",
    "parallelism",
    "optional",
    "autorun_on_main",
    "env",
    "mirror",
    "mirror_hardwares",
    "soft_fail",
    "fast_check",
    "fast_check_only",
    "timeout_in_minutes",
}


def inventory(root):
    import yaml

    root = Path(root).resolve()
    result = {
        "schema_version": 1,
        "upstream_revision": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "scope": "Area-file groups include all steps; named-label selections elsewhere are also retained. Legacy AMD selectors remain a separate source form, not additional executed coverage.",
        "areas": {name: [] for name in LABELS},
    }
    sources = sorted((root / ".buildkite/test_areas").glob("*.yaml")) + [
        root / ".buildkite/test-amd.yaml"
    ]
    for source in sources:
        document = yaml.safe_load(source.read_text())
        for index, step in enumerate(document.get("steps", [])):
            if not isinstance(step, dict):
                continue
            for area, label in LABELS.items():
                whole = (
                    source.parent.name == "test_areas"
                    and source.name == WHOLE_AREAS.get(area)
                )
                if not whole and label not in step.get("label", ""):
                    continue
                selected = {key: value for key, value in step.items() if key in FIELDS}
                selected["mirror"] = {
                    name: {key: value for key, value in mirror.items() if key in FIELDS}
                    for name, mirror in step.get("mirror", {}).items()
                }
                commands = list(step.get("commands", [])) + (
                    [step["command"]] if "command" in step else []
                )
                for mirror in step.get("mirror", {}).values():
                    commands += mirror.get("commands", [])
                config_lists = {}
                for command in commands:
                    for token in re.findall(r"--config-list-file[= ]([^\s]+)", command):
                        token = token.strip("\"'").replace(
                            "$$BUILDKITE_PARALLEL_JOB", "*"
                        )
                        for base in (root / "tests", root / ".buildkite"):
                            for path in base.glob("**/" + token):
                                if path.is_file():
                                    config_lists[str(path.relative_to(root))] = {
                                        "sha256": hashlib.sha256(
                                            path.read_bytes()
                                        ).hexdigest(),
                                        "text": path.read_text(),
                                    }
                result["areas"][area].append(
                    {
                        "file": str(source.relative_to(root)),
                        "file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "source_form": (
                            "legacy_amd" if source.name == "test-amd.yaml" else "area"
                        ),
                        "step_index": index,
                        "selection": "whole_area" if whole else "matching_label",
                        "group": document.get("group"),
                        "step": selected,
                        "config_lists": config_lists,
                    }
                )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory(args.repo), indent=2) + "\n")


if __name__ == "__main__":
    main()

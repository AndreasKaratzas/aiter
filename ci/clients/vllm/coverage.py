"""Describe declared model-feature cases; never infer execution from collection."""

import argparse
import ast
import re
from pathlib import Path

from ci.common.checkpoints import load_manifest
from ci.common.json import load_json, require


def load_coverage(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[3]
    directory = root / "ci/clients/vllm"
    value = load_json(directory / "coverage.json")
    require(
        set(value)
        == {
            "schema_version",
            "classification",
            "models",
            "cases",
            "uncovered",
            "upstream_revision",
        }
        and type(value["schema_version"]) is int
        and value["schema_version"] == 1,
        "Invalid model coverage schema",
    )
    require(
        re.fullmatch(r"[a-f0-9]{40}", value["upstream_revision"]),
        "Unpinned upstream inventory",
    )
    assets = load_manifest(directory / "models.json")
    groups = load_json(directory / "groups.json")
    require(
        set(value["models"]) == set(assets),
        "Coverage model inventory differs from admitted assets",
    )
    for model in value["models"].values():
        require(
            set(model)
            == {
                "family",
                "architecture",
                "parameters",
                "modalities",
                "checkpoint_dtype",
            }
            and all(
                type(model[key]) is str and model[key]
                for key in ("family", "architecture", "parameters", "checkpoint_dtype")
            )
            and type(model["modalities"]) is list
            and model["modalities"]
            and set(model["modalities"]) <= {"text", "image"},
            "Invalid model metadata",
        )
    seen = set()
    counts = {}
    for name, case in value["cases"].items():
        require(
            re.fullmatch(r"[a-z][a-z0-9-]*", name)
            and set(case)
            == {
                "selector",
                "group",
                "model",
                "dtypes",
                "tensor_parallel",
                "execution",
                "features",
                "context_capacity",
                "oracle",
            },
            "Invalid model case fields",
        )
        require(
            case["model"] in assets and case["group"] in groups,
            "Unregistered model or group",
        )
        selector = case["selector"]
        path, separator, node = selector.partition("::")
        require(
            separator
            and not Path(path).is_absolute()
            and ".." not in Path(path).parts
            and path.startswith("tests/frameworks/vllm/"),
            "Invalid model selector path",
        )
        functions = {
            item.name
            for item in ast.parse((root / path).read_text()).body
            if isinstance(item, ast.FunctionDef)
        }
        require(
            node.split("[")[0] in functions and selector not in seen,
            "Missing or duplicated model selector",
        )
        seen.add(selector)
        require(
            any(
                selector == target or selector.startswith((target + "::", target + "["))
                for target in groups[case["group"]]["targets"]
            ),
            "Case is not reached by its declared group",
        )
        require(
            type(case["context_capacity"]) is int
            and case["context_capacity"] in (1024, 4096),
            "Invalid context capacity",
        )
        for field, allowed in (
            ("dtypes", {"bfloat16", "float16", "fp8-per-channel"}),
            ("tensor_parallel", {1, 2}),
            ("execution", {"eager", "graph"}),
        ):
            require(
                type(case[field]) is list
                and case[field]
                and len(set(case[field])) == len(case[field])
                and all(
                    item in allowed
                    and type(item) is (int if field == "tensor_parallel" else str)
                    for item in case[field]
                ),
                "Invalid dtype/topology/execution metadata",
            )
        require(
            type(case["features"]) is list
            and case["features"]
            and all(
                type(feature) is str and re.fullmatch(r"[a-z][a-z0-9-]*", feature)
                for feature in case["features"]
            )
            and type(case["oracle"]) is str
            and case["oracle"],
            "Case lacks explicit features or oracle",
        )
        counts[case["group"]] = counts.get(case["group"], 0) + 1
    require(
        set(counts)
        == {name for name, group in groups.items() if "model_manifest" in group},
        "Model groups missing from coverage catalog",
    )
    require(
        all(count == groups[name]["minimum_cases"] for name, count in counts.items()),
        "Declared case counts differ from CI groups",
    )
    require(
        value["uncovered"]
        and all(
            type(reason) is str and reason for reason in value["uncovered"].values()
        ),
        "Missing explicit coverage gaps",
    )
    return value


def inventory(
    *,
    root=None,
    profile="vllm-e2e",
    family=None,
    feature=None,
    dtype=None,
    tensor_parallel=None,
):
    root = Path(root) if root else Path(__file__).resolve().parents[3]
    value = load_coverage(root)
    profiles = load_json(root / "ci/clients/vllm/profiles.json")
    require(profile in profiles, "Unknown vLLM profile")
    groups = set(profiles[profile]["groups"])
    cases = [
        {"id": name, **case}
        for name, case in value["cases"].items()
        if case["group"] in groups
        and (family is None or value["models"][case["model"]]["family"] == family)
        and (feature is None or feature in case["features"])
        and (dtype is None or dtype in case["dtypes"])
        and (tensor_parallel is None or tensor_parallel in case["tensor_parallel"])
    ]
    return {
        "scope": "Declared selector coverage, not execution or support qualification",
        "profile": profile,
        "case_count": len(cases),
        "models": value["models"],
        "cases": cases,
        "uncovered": value["uncovered"],
    }


def main(argv=None):
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="vllm-e2e")
    parser.add_argument("--family")
    parser.add_argument("--feature")
    parser.add_argument("--dtype")
    parser.add_argument("--tensor-parallel", type=int, choices=(1, 2))
    args = parser.parse_args(argv)
    print(json.dumps(inventory(**vars(args)), indent=2))


if __name__ == "__main__":
    main()

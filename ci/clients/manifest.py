"""Read the existing heavyweight client cases from versioned manifests."""

import argparse
import json
import os
from pathlib import Path

from ci.common.json import load_json, require


def client_cases(client: str) -> dict:
    require(client in ("vllm", "sglang"), "unknown client")
    manifest = load_json(Path(__file__).parent / client / "canaries.json")
    require(manifest["client"] == client, "client manifest identity mismatch")
    require(
        manifest["classification"] == "rolling-upstream-canary",
        "legacy consumer environment must remain an explicit canary",
    )
    require(
        isinstance(manifest["cases"], list) and bool(manifest["cases"]),
        "client has no cases",
    )
    for case in manifest["cases"]:
        require(
            isinstance(case, dict) and bool(case.get("model")),
            "invalid client model case",
        )
        require(
            isinstance(case.get("runner"), str) and bool(case["runner"]),
            "client case requires a runner",
        )
        require(
            isinstance(case.get("environment"), dict),
            "client case environment must be a mapping",
        )
        command = case.get("arguments" if client == "vllm" else "command")
        require(
            isinstance(command, list)
            and all(isinstance(token, str) for token in command),
            "client command must be an explicit argv list",
        )
    return {"include": manifest["cases"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", choices=("vllm", "sglang"), required=True)
    args = parser.parse_args()
    text = json.dumps(client_cases(args.client), separators=(",", ":"))
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            output.write("matrix=" + text + "\n")
    print(text)


if __name__ == "__main__":
    main()

"""Control SGLang downstream test selection, patching, and model resolution."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

TESTS = json.loads(Path(__file__).with_name("canaries.json").read_text())["cases"]


SGLANG_CI_PATCHES = [
    {
        "path": "scripts/ci/amd/amd_ci_start_container.sh",
        "old": "HOSTNAME_VALUE=$(hostname)",
        "new": 'HOSTNAME_VALUE="${SGLANG_CI_HOSTNAME_OVERRIDE:-$(hostname)}"',
    },
    {
        "path": "scripts/ci/amd/amd_ci_install_dependency.sh",
        "old": "HOSTNAME_VALUE=$(hostname)",
        "new": 'HOSTNAME_VALUE="${SGLANG_CI_HOSTNAME_OVERRIDE:-$(hostname)}"',
    },
    {
        "path": "scripts/ci/amd/amd_ci_exec.sh",
        "old": "HOSTNAME_VALUE=$(hostname)",
        "new": 'HOSTNAME_VALUE="${SGLANG_CI_HOSTNAME_OVERRIDE:-$(hostname)}"',
    },
    {
        "path": "scripts/ci/amd/amd_ci_install_dependency.sh",
        "old": "docker cp human-eval ci_sglang:/",
        "new": "docker cp human-eval ci_sglang:/\n  docker exec ci_sglang git config --global --add safe.directory /human-eval",
    },
    {
        "path": "scripts/ci/amd/amd_ci_install_dependency.sh",
        "old": "install_with_retry docker exec -w /human-eval ci_sglang pip install --cache-dir=/sgl-data/pip-cache -e .",
        "new": "install_with_retry docker exec -w /human-eval ci_sglang pip install --cache-dir=/sgl-data/pip-cache --no-build-isolation -e .",
    },
    {
        "path": "scripts/ci/amd/amd_ci_start_container.sh",
        "old": "$CACHE_VOLUME \\",
        "new": "$CACHE_VOLUME \\\n  -v /models:/models \\",
    },
    {
        "path": "test/registered/amd/test_qwen3_instruct_mxfp4.py",
        "old": 'QWEN3_MODEL_PATH = "amd/Qwen3-235B-A22B-Instruct-2507-mxfp4"',
        "new": 'QWEN3_MODEL_PATH = os.environ.get("QWEN3_MODEL_PATH", "amd/Qwen3-235B-A22B-Instruct-2507-mxfp4")',
    },
    {
        "path": "test/registered/amd/accuracy/mi35x/test_qwen35_eval_mi35x.py",
        "old": 'QWEN35_MODEL_PATH = "Qwen/Qwen3.5-397B-A17B"',
        "new": 'QWEN35_MODEL_PATH = os.environ.get("QWEN35_MODEL_PATH", "Qwen/Qwen3.5-397B-A17B")',
    },
    {
        "path": "test/registered/amd/accuracy/mi35x/test_deepseek_v32_eval_mi35x.py",
        "old": 'model_path="deepseek-ai/DeepSeek-V3.2",',
        "new": 'model_path=os.environ.get("DEEPSEEK_V32_MODEL_PATH", "deepseek-ai/DeepSeek-V3.2"),',
    },
    {
        "path": "test/registered/amd/accuracy/mi35x/test_deepseek_v32_eval_mi35x.py",
        "old": '        timeout=5400,\n        variant="basic",',
        "new": '        timeout=7200,\n        variant="basic",',
    },
]


def write_output(name: str, value: str) -> None:
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write(f"{name}={value}\n")


def run_cell(test: dict, key: str) -> str:
    if test.get(key, False):
        return "yes"

    comment = test.get("comment")
    if comment:
        return f"no ({comment})"
    return "no"


def write_summary(
    selected: list[dict], skipped: list[dict], disabled: list[dict], event_name: str
) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    with open(summary_path, "a", encoding="utf-8") as summary:
        summary.write("## SGLang Downstream Test Selection\n\n")
        summary.write(f"- Event: `{event_name}`\n")
        summary.write(f"- Selected tests: `{len(selected)}`\n")
        summary.write(f"- Disabled tests: `{len(disabled)}`\n")
        summary.write(f"- Event-skipped tests: `{len(skipped)}`\n\n")
        summary.write("| Model | Test | Run on PR | Run on schedule |\n")
        summary.write("| --- | --- | --- | --- |\n")
        summary.writelines(
            f"| {test['model']} | {test['test_type']} | "
            f"{run_cell(test, 'run_on_pr')} | "
            f"{run_cell(test, 'run_on_schedule')} |\n"
            for test in TESTS
        )


def select_tests() -> None:
    event_name = os.environ.get("EVENT_NAME") or os.environ.get("GITHUB_EVENT_NAME", "")
    run_key = "run_on_schedule" if event_name == "schedule" else "run_on_pr"
    disabled = [
        test
        for test in TESTS
        if not test.get("run_on_pr", False) and not test.get("run_on_schedule", False)
    ]
    runnable = [test for test in TESTS if test not in disabled]
    selected = [test for test in runnable if test.get(run_key, False)]
    skipped = [test for test in runnable if not test.get(run_key, False)]

    write_output("matrix", json.dumps({"include": selected}, separators=(",", ":")))
    write_output("has_tests", "true" if selected else "false")
    write_summary(selected, skipped, disabled, event_name or "unknown")


def replace_once(root: Path, patch: dict[str, str]) -> None:
    path = root / patch["path"]
    text = path.read_text()
    if patch["old"] not in text:
        raise SystemExit(f"Expected snippet not found in {path}: {patch['old']!r}")
    path.write_text(text.replace(patch["old"], patch["new"], 1))


def patch_sglang_checkout() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"Usage: {sys.argv[0]} patch-sglang SGLANG_WORKSPACE")

    root = Path(sys.argv[2])
    for patch in SGLANG_CI_PATCHES:
        replace_once(root, patch)


def main() -> None:
    if len(sys.argv) == 1 or sys.argv[1] == "select-tests":
        select_tests()
    elif sys.argv[1] == "patch-sglang":
        patch_sglang_checkout()
    else:
        raise SystemExit(f"Unknown command: {sys.argv[1]}")


if __name__ == "__main__":
    main()

"""Control SGLang downstream test selection, patching, and model resolution."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from ci.common.json import digest, parse_json, require, write_json
from ci.pipelines.docker import Docker
from ci.pipelines.process import Process
from ci.qualification.isolation import CACHE_DIRECTORIES
from ci.release.artifacts import hash_file
from ci.release.wheels import collect_source_identity

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


def prepare_checkout(workspace: Path, container: str, git: Process) -> dict:
    """Resolve the upstream AMD platform branch, then retain every local patch."""
    git.command(
        [
            "clone",
            "--depth",
            "1",
            "--branch",
            "amd/aiter-ci",
            "https://github.com/sgl-project/sglang.git",
            str(workspace),
        ]
    )
    revision = git.command(["-C", str(workspace), "rev-parse", "HEAD"]).strip()
    require(re.fullmatch(r"[0-9a-f]{40}", revision), "invalid SGLang revision")
    for patch in SGLANG_CI_PATCHES:
        replace_once(workspace, patch)
    for path in (workspace / "scripts/ci/amd").glob("*.sh"):
        text = path.read_text()
        if "ci_sglang" in text:
            path.write_text(text.replace("ci_sglang", container))
    for relative in ("python/sglang/kernels/aot", "python"):
        require((workspace / relative).is_dir(), "upstream SGLang layout changed")
    return {
        "revision": revision,
        "branch": "amd/aiter-ci",
        "patched_files": {
            str(path.relative_to(workspace)): hash_file(path)
            for path in sorted((workspace / "scripts/ci/amd").glob("*.sh"))
        },
        "patches": SGLANG_CI_PATCHES,
    }


def install_candidate(
    source: Path, controls: Path, container: str, docker: Docker, caches: dict
):
    """Install the selected local bytes after upstream dependencies; gate imports."""
    from ci.qualification.isolation import BUILD_OPTIONS, RESERVED

    def execute(arguments, cwd="/tmp"):
        command = ["exec", "-w", cwd]
        for key, value in {**caches, "AITER_USE_SYSTEM_TRITON": "1"}.items():
            command += ["-e", key + "=" + value]
        command += [container, "env"]
        for key in sorted((RESERVED | BUILD_OPTIONS | {"PYTHONPATH"}) - set(caches)):
            command += ["-u", key]
        return docker.command(command + arguments)

    docker.command(
        [
            "cp",
            str(controls / "requirements/clients/sglang-models.txt"),
            container + ":/tmp/aiter-client-requirements.txt",
        ]
    )
    execute(
        ["python3", "-m", "pip", "install", "-r", "/tmp/aiter-client-requirements.txt"]
    )
    docker.command(["cp", str(source), container + ":/tmp/aiter-under-test"])
    execute(["python3", "-m", "pip", "uninstall", "-y", "amd-aiter", "aiter"])
    execute(["python3", "-m", "pip", "install", "-e", "."], cwd="/tmp/aiter-under-test")
    execute(
        [
            "python3",
            "-c",
            (
                "import aiter,sglang,json; from pathlib import Path; "
                "p=Path(aiter.__file__).resolve(); "
                "exec(\"if not p.is_relative_to(Path('/tmp/aiter-under-test/aiter')): raise RuntimeError('candidate import differs')\"); "
                "print(json.dumps({'aiter':str(p),'sglang':sglang.__file__}))"
            ),
        ]
    )
    execute(["python3", "-m", "pip", "freeze", "--all"])


def run(
    *,
    source: Path,
    controls: Path,
    output: Path,
    case: dict,
    git: Process | None = None,
    bash: Process | None = None,
    docker: Docker | None = None,
):
    """Own the complete rolling upstream adapter; its results remain canaries."""
    from ci.pipelines.canaries import selected_case, sglang_model

    selected_case("sglang", case)
    source, controls, output = (path.resolve() for path in (source, controls, output))
    require(
        not output.exists() and source != controls,
        "use fresh evidence and distinct controls",
    )
    require(
        not output.is_relative_to(source) and not output.is_relative_to(controls),
        "evidence must be external",
    )
    identities = {
        "candidate": collect_source_identity(source).to_dict(),
        "controls": collect_source_identity(controls).to_dict(),
    }
    output.mkdir(parents=True)
    write_json(
        output / "request.json",
        {"classification": "rolling-upstream-canary", "case": case, **identities},
    )
    git = git or Process(output / "git", executable="git")
    bash = bash or Process(output / "platform", executable="bash")
    docker = docker or Docker(output / "docker")
    container = "aiter-sglang-" + digest(str(output)).split(":")[1][:20]
    workspace = output / "sglang"
    caches = {
        key: "/tmp/" + container + "/" + directory
        for key, directory in CACHE_DIRECTORIES.items()
    }
    environment = {
        **os.environ,
        "GITHUB_WORKSPACE": str(workspace),
        "GPU_ARCH": "gfx950",
        "SGLANG_CI_HOSTNAME_OVERRIDE": "linux-mi35x-gpu-8",
    }
    status = {
        "status": "FAIL",
        "stage": "checkout",
        "container": container,
        "cleanup_problems": [],
    }
    failure = None
    # Upstream xtrace can expose Docker environment values. Credentials are
    # not evidence; send shell traces to a private null descriptor.
    platform_shell = [
        "-c",
        'exec 9>/dev/null; export BASH_XTRACEFD=9; exec bash "$@"',
        "sglang-platform",
    ]
    try:
        write_json(
            output / "upstream.json", prepare_checkout(workspace, container, git)
        )
        status["stage"] = "platform"
        bash.command(
            platform_shell
            + ["scripts/ci/amd/amd_ci_start_container.sh", "--rocm-version", "rocm720"],
            cwd=workspace,
            env=environment,
        )
        for key, value in (("global.default-timeout", "60"), ("global.retries", "10")):
            docker.command(
                [
                    "exec",
                    "-u",
                    "root",
                    container,
                    "python3",
                    "-m",
                    "pip",
                    "config",
                    "set",
                    key,
                    value,
                ]
            )
        bash.command(
            platform_shell
            + ["scripts/ci/amd/amd_ci_install_dependency.sh", "--skip-aiter-build"],
            cwd=workspace,
            env=environment,
        )
        status["stage"] = "candidate-import"
        install_candidate(source, controls, container, docker, caches)
        status["stage"] = "model"
        sglang_model(
            container=container,
            case=case,
            output=output / "model",
            docker=docker,
            caches=caches,
        )
        summary = workspace / "github_summary.md"
        if summary.is_file() and os.environ.get("GITHUB_STEP_SUMMARY"):
            with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as stream:
                stream.write(summary.read_text())
        require(
            identities
            == {
                "candidate": collect_source_identity(source).to_dict(),
                "controls": collect_source_identity(controls).to_dict(),
            },
            "candidate or controls changed during SGLang execution",
        )
        status["status"] = "PASS"
    except BaseException as error:
        failure = error
        status["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            docker.command(["rm", "--force", "--volumes", container], timeout=120)
        except (ValueError, OSError) as error:
            status["cleanup_problems"].append(str(error))
            status["status"] = "FAIL"
        try:
            write_json(output / "status.json", status)
        except OSError as error:
            if failure is None:
                raise
            if hasattr(failure, "add_note"):
                failure.add_note("Cannot retain SGLang status: " + str(error))
        if failure is None:
            require(not status["cleanup_problems"], "SGLang container cleanup failed")


def run_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run one reviewed upstream SGLang model case"
    )
    for name in ("source", "controls", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--case-json", required=True)
    args = parser.parse_args(argv)
    run(
        source=args.source,
        controls=args.controls,
        output=args.output,
        case=parse_json(args.case_json),
    )


def main() -> None:
    if len(sys.argv) == 1 or sys.argv[1] == "select-tests":
        select_tests()
    elif sys.argv[1] == "patch-sglang":
        patch_sglang_checkout()
    else:
        raise SystemExit(f"Unknown command: {sys.argv[1]}")


if __name__ == "__main__":
    main()

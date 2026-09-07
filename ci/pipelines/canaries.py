"""Retain specialized model canaries without turning them into release evidence."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from ci.clients.manifest import client_cases
from ci.common.json import digest, require, write_json
from ci.pipelines.docker import IMAGE_ID, Docker
from ci.release.artifacts import hash_file


def resolve_environments(
    *, controls: Path, images: dict, output: Path, docker: Docker | None = None
) -> dict:
    """Observe configured upstream images through the shared retained Docker port."""
    from ci.qualification.environments import validate_lock

    require(
        isinstance(images, dict) and set(images) == {"vllm", "sglang"},
        "configure both reviewed rolling framework images",
    )
    require(
        all(isinstance(value, str) and value for value in images.values()),
        "canary images must be references",
    )
    controls, output = controls.resolve(), output.resolve()
    require(
        not output.exists() and not output.is_relative_to(controls),
        "canary resolution requires fresh external evidence",
    )
    output.mkdir(parents=True)
    # Only the observer program is exposed on Python's path, never checkout AITER.
    suite = output / "suite"
    shutil.copytree(
        controls / "ci",
        suite / "ci",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    transport = docker or Docker(output / "docker")
    entries = []
    for client, reference in images.items():
        inspection = transport.resolve(reference)
        write_json(output / (client + "-image.json"), inspection)
        digests = inspection.get("RepoDigests", [])
        require(
            isinstance(digests, list) and digests,
            "canary image has no immutable repository digest",
        )
        immutable = digests[0]
        name = (
            "aiter-observe-"
            + digest({"client": client, "output": str(output)}).split(":")[1][:24]
        )
        for path in (suite, output):
            require(
                ":" not in str(path) and "\n" not in str(path), "invalid canary mount"
            )
        transport.managed_container(
            [
                "run",
                "--rm",
                "--name",
                name,
                "--entrypoint",
                "python3",
                "-v",
                str(suite) + ":/instructions:ro",
                "-v",
                str(output) + ":/evidence",
                "-e",
                "PYTHONPATH=/instructions",
                "-e",
                "PYTHONNOUSERSITE=1",
                "-w",
                "/evidence",
                inspection["Id"],
                "-m",
                "ci.qualification.environments",
                "--image",
                immutable,
                "--client",
                client,
                "--output",
                "/evidence/" + client + "-lock.json",
            ],
            name=name,
            timeout=600,
        )
        lock = validate_lock(json.loads((output / (client + "-lock.json")).read_text()))
        require(
            lock["image"] == immutable
            and set(lock["frameworks"]) == {client}
            and lock["status"] == "canary",
            "canary observer returned a different image, framework or classification",
        )
        entries.append(
            {
                "profile": client,
                "image": immutable,
                "environment_lock": json.dumps(lock),
                "label": client + "-canary",
            }
        )
    matrix = {"include": entries}
    write_json(output / "matrix.json", matrix)
    return matrix


def selected_case(client: str, case: dict) -> dict:
    require(
        case in client_cases(client)["include"],
        "canary case differs from the reviewed manifest",
    )
    environment = case.get("environment")
    require(isinstance(environment, dict), "canary environment must be a mapping")
    for key, value in environment.items():
        require(
            re.fullmatch(r"(?:VLLM|SGLANG)_[A-Z0-9_]+", key)
            and isinstance(value, str)
            and "\n" not in value,
            "invalid framework environment override",
        )
    return case


def vllm_latency(
    *,
    controls: Path,
    wheel_dir: Path,
    image: str,
    case: dict,
    output: Path,
    docker: Docker | None = None,
) -> None:
    selected_case("vllm", case)
    controls, wheel_dir, output = (p.resolve() for p in (controls, wheel_dir, output))
    require(
        not output.exists() and not output.is_relative_to(controls),
        "canary evidence must be a fresh directory outside controls",
    )
    wheels = list(wheel_dir.glob("*.whl"))
    require(
        len(wheels) == 1 and not wheels[0].is_symlink(),
        "latency canary needs exactly one wheel",
    )
    wheel = wheels[0]
    sha, size = hash_file(wheel)
    output.mkdir(parents=True)
    wheelhouse = output / "wheelhouse"
    wheelhouse.mkdir()
    shutil.copy2(wheel, wheelhouse / wheel.name)
    (wheelhouse / "SHA256SUMS").write_text(f"{sha}  {wheel.name}\n")
    transport = docker or Docker(output / "docker")
    inspection = transport.resolve(image)
    write_json(output / "image.json", inspection)
    request = {
        "classification": "rolling-upstream-canary",
        "case": case,
        "wheel": {"filename": wheel.name, "sha256": sha, "size_bytes": size},
        "executor_image": inspection["Id"],
    }
    write_json(output / "request.json", request)
    name = (
        "aiter-vllm-canary-"
        + digest({"output": str(output), "wheel": sha}).split(":")[1][:24]
    )
    command = [
        "run",
        "--rm",
        "--name",
        name,
        "--entrypoint",
        "python3",
        "--device=/dev/kfd",
        "--device=/dev/dri",
        "--group-add",
        "video",
        "--ulimit",
        "core=0:0",
        "--ulimit",
        "memlock=-1:-1",
        "--ulimit",
        "stack=67108864",
        "--cap-add=SYS_PTRACE",
        "--network=host",
        "--security-opt",
        "seccomp=unconfined",
        "--shm-size",
        "16g",
        "-v",
        str(controls) + ":/control:ro",
        "-v",
        str(output) + ":/evidence",
        "-v",
        str(wheelhouse) + ":/wheelhouse:ro",
        "-w",
        "/control",
        "-e",
        "PYTHONPATH=/control",
        "-e",
        "PYTHONNOUSERSITE=1",
        "-e",
        "HF_TOKEN",
        "-e",
        "VLLM_ROCM_USE_AITER=1",
        "-e",
        "AITER_USE_SYSTEM_TRITON=1",
        "-e",
        "AITER_JIT_DIR=/evidence/cache/aiter-jit",
        "-e",
        "AITER_AOT_CACHE_DIR=/evidence/cache/aiter-aot",
        "-e",
        "TRITON_CACHE_DIR=/evidence/cache/triton",
    ]
    if Path("/models").is_dir():
        command.extend(["-v", "/models:/models:ro"])
    for key, value in case["environment"].items():
        command.extend(["-e", key + "=" + value])
    command.extend(
        [
            inspection["Id"],
            "-m",
            "ci.clients.vllm.latency",
            "--request",
            "/evidence/request.json",
        ]
    )
    transport.managed_container(command, name=name, timeout=3600)


def sglang_model(
    *,
    container: str,
    case: dict,
    output: Path,
    docker: Docker | None = None,
    caches: dict | None = None,
) -> None:
    selected_case("sglang", case)
    require(
        re.fullmatch(r"[A-Za-z0-9_.-]+", container),
        "invalid existing SGLang container name",
    )
    require(not output.exists(), "canary evidence must be a new directory")
    require(
        isinstance(case["command"], list)
        and case["command"]
        and all(isinstance(x, str) for x in case["command"]),
        "model command must be an explicit argv list",
    )
    output.mkdir(parents=True)
    transport = docker or Docker(output / "docker")
    identity = transport.command(
        ["inspect", "--format", "{{.Image}}", container]
    ).strip()
    require(
        IMAGE_ID.fullmatch(identity), "SGLang container has no immutable image identity"
    )
    environment = {
        "GITHUB_STEP_SUMMARY": "/sglang-checkout/github_summary.md",
        "AITER_USE_SYSTEM_TRITON": "1",
        **case["environment"],
    }
    if caches is not None:
        from ci.qualification.isolation import CACHE_DIRECTORIES

        require(
            set(caches) == set(CACHE_DIRECTORIES),
            "SGLang adapter must own every compile cache",
        )
        environment.update(caches)
    model_id = case.get("model_id")
    if model_id and (Path("/models") / model_id / "config.json").is_file():
        environment[case["model_path_env"]] = "/models/" + model_id
    command = ["exec", "-w", "/sglang-checkout/test"]
    for key, value in environment.items():
        command.extend(["-e", key + "=" + value])
    command.append(container)
    if caches is not None:
        from ci.qualification.isolation import BUILD_OPTIONS, RESERVED

        command.append("env")
        for key in sorted((RESERVED | BUILD_OPTIONS | {"PYTHONPATH"}) - set(caches)):
            command.extend(["-u", key])
    command.extend(case["command"])
    write_json(
        output / "request.json",
        {
            "classification": "rolling-upstream-canary",
            "scope": "upstream AMD serving harness; environment preparation remains its platform adapter",
            "case": case,
            "executor_image": identity,
            "command": command,
        },
    )
    # This container was created by the upstream platform adapter. Kill it on
    # command timeout/failure so the daemon cannot retain a running model server.
    transport.managed_container(
        command, name=container, timeout=case["timeout_minutes"] * 60
    )

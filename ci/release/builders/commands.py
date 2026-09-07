"""Execute build commands as argument lists, with a reviewable dry-run mode."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ci.common.json import require
from ci.release.builders.environment import (
    Environment,
    architectures,
    torch_dependency,
    wheel_version,
)


@dataclass(frozen=True)
class Runner:
    dry_run: bool = False

    def run(
        self,
        command: list[str],
        *,
        environment: dict[str, str] | None = None,
        check: bool = True,
        cwd: Path | None = None,
    ):
        require(
            command and all(isinstance(value, str) for value in command),
            "command must contain string arguments",
        )
        print(
            json.dumps(
                {"command": command, "environment": environment or {}}, sort_keys=True
            ),
            flush=True,
        )
        if self.dry_run:
            return None
        return subprocess.run(
            command, env={**os.environ, **(environment or {})}, check=check, cwd=cwd
        )


def docker(
    environment: Environment,
    command: list[str],
    *,
    variables: dict[str, str] | None = None,
) -> list[str]:
    argv = ["docker", "exec", "-w", "/control", "-e", "PYTHONPATH=/control"]
    for key, value in (variables or {}).items():
        argv.extend(("-e", f"{key}={value}"))
    return argv + [environment.container] + command


def control_root() -> Path:
    """The checkout that owns this executing controller, never the candidate."""
    return Path(__file__).resolve().parents[3]


def start(
    environment: Environment,
    source: Path,
    directory: Path,
    runner: Runner,
    *,
    control: Path | None = None,
    has_gpu: bool | None = None,
):
    require(bool(environment.image), "container image is required")
    control = (control or control_root()).resolve()
    source, directory = source.resolve(), directory.resolve()
    require(control == control_root(), "control mount must own the executing builder")
    for path in (control, source, directory):
        require(":" not in str(path), "invalid workspace mount")
    require(source.is_dir(), "candidate source directory is missing")
    paths = (control, source, directory)
    require(
        all(
            a != b and a not in b.parents and b not in a.parents
            for number, a in enumerate(paths)
            for b in paths[number + 1 :]
        ),
        "control, candidate source and artifact directory must be separate trees",
    )
    if not runner.dry_run:
        directory.mkdir(parents=True, exist_ok=True)
        require(
            not list(directory.iterdir()),
            "artifact directory must be empty before build",
        )
    argv = ["docker", "run", "-dt", "--shm-size=16G", "--network=host"]
    if has_gpu if has_gpu is not None else Path("/dev/kfd").exists():
        argv.extend(("--device=/dev/kfd", "--device=/dev/dri", "--group-add", "video"))
    argv.extend(
        (
            "--entrypoint",
            "",
            "-v",
            f"{control}:/control:ro",
            "-v",
            f"{source}:/workspace",
            "-v",
            f"{directory}:/artifacts",
            "-w",
            "/control",
            "--name",
            environment.container,
            environment.image,
            "sleep",
            "infinity",
        )
    )
    runner.run(argv)
    for path in ("/control", "/workspace", "/workspace/3rdparty/composable_kernel"):
        runner.run(
            docker(
                environment,
                ["git", "config", "--global", "--add", "safe.directory", path],
            )
        )


def dependencies(
    environment: Environment,
    runner: Runner,
    *,
    torch_pin: str = "",
    torch_index: str = "",
):
    python = environment.python_bin
    pip = [python, "-m", "pip", "install", "--timeout=60", "--retries=10"]
    if environment.flavor == "manylinux":
        spec, index = torch_dependency(environment, torch_pin, torch_index)
        runner.run(docker(environment, pip + ["--upgrade", "pip"]))
        runner.run(docker(environment, pip + ["--index-url", index, spec]))
        runner.run(
            docker(
                environment,
                pip
                + [
                    "--extra-index-url",
                    "https://rocm.frameworks-devreleases.amd.com/whl-staging/gfx942-gfx950/",
                    "-r",
                    "/workspace/requirements/test/product.txt",
                ],
            )
        )
        tools = ["-r", "/control/requirements/build/manylinux.txt"]
    else:
        runner.run(
            docker(
                environment, pip + ["-r", "/workspace/requirements/test/product.txt"]
            )
        )
        tools = ["-r", "/control/requirements/build/release.txt"]
    runner.run(docker(environment, pip + tools))


def latest_version(source: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), "describe", "--tags", "--abbrev=0"],
        capture_output=True,
        text=True,
        check=False,
    )
    return (
        result.stdout.strip()
        if result.returncode == 0 and result.stdout.strip()
        else "0.1.0"
    )


def build(
    environment: Environment,
    runner: Runner,
    *,
    gpu_archs: str,
    version: str = "",
    source: Path,
):
    variables = {
        "PREBUILD_KERNELS": "1",
        "AITER_NATIVE_LIB_DIR": "/tmp/aiter-native-wheel",
        "GPU_ARCHS": architectures(gpu_archs),
    }
    command = [
        environment.python_bin,
        "-m",
        "ci.release.builders",
        "compile",
        "--source",
        "/workspace",
        "--directory",
        "/artifacts",
        "--native-dir",
        variables["AITER_NATIVE_LIB_DIR"],
    ]
    if environment.flavor == "manylinux":
        version = wheel_version(
            version or latest_version(source), rocm_tag=environment.rocm_tag
        )
        variables["AITER_BUILD_PYBIN"] = str(Path(environment.python_bin).parent)
        command = ["bash", "/control/ci/release/builders/toolchain.sh", *command]
    if version:
        variables["SETUPTOOLS_SCM_PRETEND_VERSION"] = wheel_version(version)
    runner.run(docker(environment, command, variables=variables))


def compile_wheel(
    source: Path,
    native_dir: Path,
    directory: Path,
    runner: Runner,
    *,
    python: str,
    jobs: int,
    rocm: Path,
):
    require(jobs > 0, "build parallelism must be positive")
    source, native_dir, directory = (
        source.resolve(),
        native_dir.resolve(),
        directory.resolve(),
    )
    protected = (source, control_root(), directory)
    require(
        all(
            native_dir != path
            and path not in native_dir.parents
            and native_dir not in path.parents
            for path in protected
        ),
        "native build output must be outside and disjoint from source, controls and artifacts",
    )
    require(
        all(
            directory != path
            and path not in directory.parents
            and directory not in path.parents
            for path in (source, control_root())
        ),
        "wheel output must be outside and disjoint from source and controls",
    )
    require(
        not directory.exists()
        or (directory.is_dir() and not list(directory.iterdir())),
        "wheel output directory must be empty before compile",
    )
    required = (
        "CMakeLists.txt",
        "include/aiter/aiter.h",
        "csrc/runtime/runtime.cpp",
        "csrc/runtime/ck_blockscale.cu",
    )
    missing = [name for name in required if not (source / name).is_file()]
    require(
        not missing,
        "candidate does not provide the required native SDK build: "
        + ", ".join(missing),
    )
    runner.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(native_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_TESTING=OFF",
            "-DCMAKE_HIP_COMPILER=" + str(rocm / "lib/llvm/bin/clang++"),
            "-DAITER_BUILD_KERNELS=ON",
            "-DAITER_GPU_ARCH=gfx950",
        ]
    )
    runner.run(
        [
            "cmake",
            "--build",
            str(native_dir),
            "--target",
            "aiter",
            "aiter_rmsnorm_backend",
            "aiter_ck_backend",
            "--parallel",
            str(jobs),
        ]
    )
    runner.run(
        [python, "setup.py", "bdist_wheel", "--dist-dir", str(directory)],
        environment={"AITER_NATIVE_LIB_DIR": str(native_dir)},
        cwd=source,
    )
    if not runner.dry_run:
        require(bool(list(directory.glob("*.whl"))), "build produced no wheel")

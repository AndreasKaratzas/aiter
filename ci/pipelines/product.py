"""Run declared legacy product areas with reviewed tests and installed candidates.

This adapter preserves the historical driver commands and result semantics. It
does not promote their output into modern release qualification evidence.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from ci.common.json import digest, load_json, require, write_json
from ci.pipelines.docker import Docker
from ci.pipelines.process import Process
from ci.qualification.controls import copy_controls
from ci.qualification.isolation import BUILD_OPTIONS, CACHE_DIRECTORIES, RESERVED
from ci.release.artifacts import hash_file
from ci.release.wheels import collect_source_identity


def area(root: Path, name: str) -> dict:
    definitions = load_json(root / "ci/pipelines/product_areas.json")
    require(name in definitions["areas"], "unknown product area")
    return definitions["areas"][name]


def inventory(root: Path, name: str) -> list[str]:
    definition = area(root, name)
    paths = {
        path.relative_to(root).as_posix()
        for pattern in definition["selectors"]
        for path in root.glob(pattern)
        if path.is_file() and path.name not in definition["exclude_names"]
    }
    require(bool(paths), "product area selects no files")
    for relative in paths:
        require(
            (root / relative).resolve().is_relative_to(root.resolve()),
            "product selector escapes controls",
        )
    return sorted(paths)


def selection(
    root: Path, artifacts: Path, name: str, index: int, count: int
) -> list[str]:
    expected = inventory(root, name)
    if name == "multi-gpu":
        return expected
    require(
        type(count) is int and count == 8 and type(index) is int and 0 <= index < count,
        "standard product execution requires one of eight shards",
    )
    shards = [
        (artifacts / "shards" / f"aiter_shard_{item}.list").read_text().split()
        for item in range(count)
    ]
    flat = [path for shard in shards for path in shard]
    require(
        all(shards) and len(flat) == len(set(flat)) and sorted(flat) == expected,
        "shards must cover the reviewed product area exactly once",
    )
    return shards[index]


def execute(
    *,
    source: Path,
    controls: Path,
    output: Path,
    artifacts: Path,
    name: str,
    image: str,
    index: int,
    count: int,
    docker: Docker | None = None,
):
    source, controls, output, artifacts = (
        p.resolve() for p in (source, controls, output, artifacts)
    )
    require(
        source != controls and not output.exists(),
        "use distinct checkouts and new product evidence",
    )
    require(
        not output.is_relative_to(source) and not output.is_relative_to(controls),
        "product evidence must be external",
    )
    selected = selection(controls, artifacts, name, index, count)
    wheels = sorted((artifacts / "aiter_wheels").glob("*.whl"))
    require(len(wheels) == 1, "expected exactly one candidate wheel")
    definition = area(controls, name)
    output.mkdir(parents=True)
    transport = docker or Docker(output / "docker")
    inspection = transport.resolve(image)
    write_json(output / "image.json", inspection)
    request = {
        "schema_version": 1,
        "classification": "legacy-installed-driver-execution",
        "source": collect_source_identity(source).to_dict(),
        "control_source": collect_source_identity(controls).to_dict(),
        "area": name,
        "definition": definition,
        "shard_index": index,
        "shard_count": count,
        "selected": selected,
        "test_sha256": {path: hash_file(controls / path)[0] for path in selected},
        "wheel": wheels[0].name,
        "wheel_sha256": hash_file(wheels[0])[0],
        "executor_image": inspection["Id"],
    }
    request["request_digest"] = digest(request)
    write_json(output / "request.json", request)
    transport.run(
        inspection["Id"],
        controls=controls,
        source=source,
        evidence=output,
        artifacts=artifacts,
        request="request.json",
        controller="ci.pipelines.product",
        timeout=definition["timeout_seconds"] + 1800,
    )
    result = load_json(output / "result.json")
    require(
        result["status"] == "PASS"
        and result["request_digest"] == request["request_digest"],
        "product execution failed or returned another request",
    )
    require(
        hash_file(wheels[0])[0] == request["wheel_sha256"],
        "candidate artifact changed during tests",
    )
    require(
        collect_source_identity(source).to_dict() == request["source"]
        and collect_source_identity(controls).to_dict() == request["control_source"],
        "candidate or controls changed during product execution",
    )


def run(
    request_path: Path,
    *,
    controls=Path("/control"),
    source=Path("/workspace"),
    artifacts=Path("/artifacts"),
):
    request = load_json(request_path)
    require(
        request["request_digest"]
        == digest({k: v for k, v in request.items() if k != "request_digest"}),
        "product request changed",
    )
    require(
        request["definition"] == area(controls, request["area"]), "product area changed"
    )
    require(
        request["selected"]
        == selection(
            controls,
            artifacts,
            request["area"],
            request["shard_index"],
            request["shard_count"],
        ),
        "product selection changed",
    )
    require(
        request["executor_image"] == os.environ.get("AITER_CI_EXECUTOR_IMAGE"),
        "product image changed",
    )
    wheel = artifacts / "aiter_wheels" / request["wheel"]
    require(hash_file(wheel)[0] == request["wheel_sha256"], "candidate wheel changed")
    output = request_path.parent
    suite = copy_controls(controls, output / "suite")
    cache = output / "cache"
    cache.mkdir()
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("AITER_", "TRITON_", "FLYDSL_", "PYTEST_", "PYTHON"))
        and key not in RESERVED | BUILD_OPTIONS
    }
    for key, directory in CACHE_DIRECTORIES.items():
        (cache / directory).mkdir()
        environment[key] = str(cache / directory)
    environment.update(
        {
            "PYTHONPATH": str(suite / "tests") + os.pathsep + str(suite),
            "PYTHONNOUSERSITE": "1",
            "AITER_USE_SYSTEM_TRITON": "1",
            "TRITON_WHEEL_DIR": str(artifacts / "triton_wheels"),
            "AITER_CI_SOURCE_ROOT": str(source),
            "AITER_CI_CONTROL_ROOT": str(controls),
            "AITER_TEST": " ".join(request["selected"]),
            "SHARD_IDX": str(request["shard_index"]),
            "SHARD_TOTAL": str(request["shard_count"]),
            "MULTIGPU": "TRUE" if request["area"] == "multi-gpu" else "FALSE",
        }
    )
    python = Process(output / "installation", executable=sys.executable)
    bash = Process(output / "driver", executable="bash")
    result = {"status": "FAIL", "request_digest": request["request_digest"]}
    failure = None
    try:
        require(
            request["test_sha256"]
            == {path: hash_file(suite / path)[0] for path in request["selected"]},
            "copied product tests changed",
        )
        python.command(
            [
                "-m",
                "pip",
                "install",
                "-r",
                str(suite / "requirements/test/product.txt"),
            ],
            cwd=suite,
            env=environment,
        )
        python.command(
            [
                "-m",
                "pip",
                "install",
                "--upgrade",
                "-r",
                str(suite / "requirements/test/legacy-overrides.txt"),
            ],
            cwd=suite,
            env=environment,
        )
        python.command(
            ["-m", "pip", "install", "--force-reinstall", "--no-deps", str(wheel)],
            cwd=suite,
            env=environment,
        )
        bash.command(
            [str(suite / ".github/scripts/common/install_triton.sh")],
            cwd=suite,
            env=environment,
        )
        python.command(
            [
                "-c",
                (
                    "import aiter,torch,json; from pathlib import Path; "
                    "p=Path(aiter.__file__).resolve(); "
                    "exec(\"if p.is_relative_to(Path('/workspace')) or p.is_relative_to(Path('/control')): raise RuntimeError('source shadows candidate wheel')\"); "
                    "print(json.dumps({'aiter':str(p),'devices':torch.cuda.device_count()})); "
                    f"exec(\"if torch.cuda.device_count() < {request['definition']['minimum_gpus']}: raise RuntimeError('insufficient GPUs')\")"
                ),
            ],
            cwd=suite,
            env=environment,
        )
        bash.command(
            [str(suite / ".github/scripts/library/run_tests.sh")],
            cwd=suite,
            env=environment,
            timeout=request["definition"]["timeout_seconds"],
        )
        require(
            request["test_sha256"]
            == {path: hash_file(suite / path)[0] for path in request["selected"]},
            "product tests changed during execution",
        )
        result["status"] = "PASS"
    except BaseException as error:
        failure = error
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            for name in ("latest_test.log", "tuned_op_bench.csv"):
                if (suite / name).is_file():
                    shutil.copyfile(suite / name, output / name)
            write_json(output / "result.json", result)
        except OSError as error:
            if failure is None:
                raise
            if hasattr(failure, "add_note"):
                failure.add_note("Cannot retain product driver results: " + str(error))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--area", choices=("standard", "multi-gpu"))
    for name in ("source", "controls", "output", "artifacts"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--image")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=8)
    args = parser.parse_args(argv)
    if args.list:
        print(
            "\n".join(
                inventory(
                    args.controls or Path(__file__).resolve().parents[2], args.area
                )
            )
        )
    elif args.request:
        run(args.request)
    else:
        require(
            all(
                (
                    args.source,
                    args.controls,
                    args.output,
                    args.artifacts,
                    args.image,
                    args.area,
                )
            ),
            "product execution arguments are required",
        )
        execute(
            source=args.source,
            controls=args.controls,
            output=args.output,
            artifacts=args.artifacts,
            name=args.area,
            image=args.image,
            index=args.shard_index,
            count=args.shard_count,
        )


if __name__ == "__main__":
    main()

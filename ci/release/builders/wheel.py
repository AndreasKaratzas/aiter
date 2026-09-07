"""Repair candidate wheels and verify their binary dependencies before receipts."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from ci.common.json import require
from ci.release.builders.commands import Runner, control_root

EXCLUDED_LIBRARIES = (
    "libamdhip64.so.7",
    "libamdhip64.so",
    "libhsa-runtime64.so.1",
    "libhsa-runtime64.so",
    "librocblas.so",
    "librocsparse.so",
    "librocsolver.so",
    "libhipblas.so",
    "libhipblaslt.so",
    "librccl.so.1",
    "librccl.so",
    "libMIOpen.so.1",
    "libMIOpen.so",
    "libtorch.so",
    "libtorch_cpu.so",
    "libtorch_hip.so",
    "libtorch_python.so",
    "libc10.so",
    "libc10_hip.so",
)


def wheels(directory: Path) -> list[Path]:
    result = sorted(directory.glob("*.whl"))
    require(bool(result), f"no candidate wheels in {directory}")
    return result


def repair(directory: Path, runner: Runner, *, auditwheel: str = "auditwheel"):
    originals = wheels(directory)
    with tempfile.TemporaryDirectory(prefix="aiter-wheel-repair-") as temporary:
        output = Path(temporary)
        excludes = [
            value for library in EXCLUDED_LIBRARIES for value in ("--exclude", library)
        ]
        for wheel in originals:
            runner.run([auditwheel, "show", str(wheel)], check=False)
            runner.run(
                [
                    auditwheel,
                    "repair",
                    str(wheel),
                    *excludes,
                    "--plat",
                    "manylinux_2_28_x86_64",
                    "-w",
                    str(output),
                ]
            )
        if runner.dry_run:
            return
        repaired = wheels(output)
        require(
            len(repaired) == len(originals),
            "repair changed the number of candidate wheels",
        )
        names = {path.name for path in repaired}
        for wheel in repaired:
            shutil.move(str(wheel), str(directory / wheel.name))
        for original in originals:
            if original.name not in names:
                original.unlink()


def symbol_versions(text: str) -> dict[str, tuple[int, ...]]:
    result = {}
    for family in ("GLIBCXX", "GLIBC"):
        versions = [
            tuple(map(int, version.split(".")))
            for version in re.findall(
                r"\b" + family + r"_([0-9]+(?:\.[0-9]+)+)\b", text
            )
        ]
        result[family] = max(versions, default=(0,))
    return result


def verify_symbols(
    directory: Path,
    runner: Runner,
    *,
    auditwheel: str = "auditwheel",
    objdump: str = "objdump",
    glibcxx: tuple[int, ...] = (3, 4, 29),
    glibc: tuple[int, ...] = (2, 34),
) -> list[dict]:
    observations = []
    for wheel in wheels(directory):
        runner.run([auditwheel, "show", str(wheel)], check=False)
        if runner.dry_run:
            continue
        with zipfile.ZipFile(wheel) as archive, tempfile.TemporaryDirectory(
            prefix="aiter-wheel-symbols-"
        ) as temporary:
            for number, member in enumerate(archive.infolist()):
                if member.is_dir() or not re.search(
                    r"\.so(?:\.[^/]*)?$", member.filename
                ):
                    continue
                # Inspect each member through a generated local filename; ZIP paths
                # are never extracted and cannot escape the temporary directory.
                path = Path(temporary) / f"library-{number}.so"
                with archive.open(member) as source, path.open("wb") as target:
                    shutil.copyfileobj(source, target)
                result = subprocess.run(
                    [objdump, "-p", str(path)],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                versions = symbol_versions(result.stdout)
                require(
                    versions["GLIBCXX"] <= glibcxx and versions["GLIBC"] <= glibc,
                    f"{wheel.name}:{member.filename} exceeds GLIBCXX {glibcxx} / GLIBC {glibc}: {versions}",
                )
                observations.append(
                    {
                        "wheel": wheel.name,
                        "library": member.filename,
                        "versions": versions,
                    }
                )
    return observations


def receipts(
    directory: Path,
    source: Path,
    runner: Runner,
    *,
    python: str,
    image: str,
    python_version: str,
):
    directory, source = directory.resolve(), source.resolve()
    control = control_root()
    execution = {"cwd": control, "environment": {"PYTHONPATH": str(control)}}
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    for wheel in wheels(directory):
        prefix = [python, "-m", "ci.release.wheels"]
        runner.run(
            prefix + ["refresh-native-manifest", "--wheel", str(wheel)], **execution
        )
        runner.run(
            prefix
            + [
                "create",
                "--wheel",
                str(wheel),
                "--source-root",
                str(source),
                "--environment",
                "build_image=" + image,
                "--environment",
                "build_python=" + python_version,
            ],
            **execution,
        )
        runner.run(
            prefix
            + [
                "verify",
                "--wheel",
                str(wheel),
                "--expected-source-revision",
                revision,
                "--require-clean-source",
            ],
            **execution,
        )

"""Check the selected Python development headers before installing a nightly environment."""

from __future__ import annotations

import shutil
import sys
import sysconfig
from pathlib import Path

from ci.common.json import load_json, require, write_json
from ci.pipelines.process import Process
from ci.release.artifacts import hash_file


def source(major: int, minor: int) -> str:
    return f"""#include <Python.h>
#if PY_MAJOR_VERSION != {major} || PY_MINOR_VERSION != {minor}
#error Python development headers do not match the selected interpreter
#endif
int main() {{ return 0; }}
"""


def sha256(path: Path) -> str:
    return hash_file(path.resolve())[0]


def preflight(output: Path, environment: dict) -> dict:
    directory = output / "preflight"
    directory.mkdir()
    paths = {
        name: str(Path(sysconfig.get_path(name)).resolve())
        for name in ("include", "platinclude")
    }
    paths["INCLUDEPY"] = sysconfig.get_config_var("INCLUDEPY")
    includes = list(dict.fromkeys(paths[name] for name in ("include", "platinclude")))
    header = Path(paths["include"]) / "Python.h"
    require(header.is_file(), f"selected Python development headers missing: {header}")
    require(
        all(Path(path).is_dir() for path in includes),
        "selected Python platform include directory missing",
    )
    compiler = shutil.which("c++", path=environment.get("PATH"))
    require(compiler is not None, "nightly worker requires a C++ compiler")
    compiler = Path(compiler).resolve()
    target = directory / "python_headers.cpp"
    version = [sys.version_info.major, sys.version_info.minor]
    target.write_text(source(*version))
    Process(directory, executable=str(compiler)).command(
        [
            "-std=c++17",
            "-fsyntax-only",
            *[argument for path in includes for argument in ("-I", path)],
            str(target),
        ],
        env=environment,
        cwd=directory,
        timeout=60,
    )
    shutil.copyfile(header, directory / "Python.h")
    record = {
        "python_version": version,
        "include_paths": paths,
        "python_h_sha256": sha256(header),
        "compiler": str(compiler),
        "compiler_sha256": sha256(compiler),
        "source_sha256": sha256(target),
        "compile_source": str(target),
    }
    write_json(directory / "receipt.json", record)
    return verify_preflight(output, live=True)


def verify_preflight(output: Path, *, live: bool = False) -> dict:
    directory = output / "preflight"
    record = load_json(directory / "receipt.json")
    includes = list(
        dict.fromkeys(
            record["include_paths"][name] for name in ("include", "platinclude")
        )
    )
    header = Path(record["include_paths"]["include"]) / "Python.h"
    compiler = Path(record["compiler"])
    target = directory / "python_headers.cpp"
    require(
        target.read_text() == source(*record["python_version"])
        and sha256(directory / "Python.h") == record["python_h_sha256"]
        and sha256(target) == record["source_sha256"],
        "nightly Python development toolchain changed",
    )
    if live:
        require(
            sha256(header) == record["python_h_sha256"]
            and sha256(compiler) == record["compiler_sha256"],
            "live Python development toolchain changed",
        )
    require(
        Path(record["compile_source"]).name == target.name
        and Path(record["compile_source"]).is_absolute(),
        "invalid retained compile input path",
    )
    execution = load_json(directory / "001--std=c++17.execution.json")
    require(
        execution.get("command")
        == [
            str(compiler),
            "-std=c++17",
            "-fsyntax-only",
            *[argument for path in includes for argument in ("-I", path)],
            record["compile_source"],
        ]
        and type(execution.get("returncode")) is int
        and execution["returncode"] == 0
        and execution.get("cleanup_problems") == []
        and execution.get("timed_out") is False
        and execution.get("interrupted") is False
        and (directory / "001--std=c++17.log").is_file(),
        "nightly Python header compile prerequisite failed",
    )
    return record

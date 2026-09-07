# SPDX-License-Identifier: MIT
"""Verified FlyDSL AOT artifacts, separated from writable runtime caches.

FlyDSL currently needs writable reader locks next to its cached kernels. Copy
the declared artifact bytes into a private cache before asking FlyDSL to load
them; never point that runtime at an installed package directory.
"""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import re
import sys
import tempfile
from pathlib import Path, PurePosixPath

_MANIFEST = "manifest.json"
_admitted_configuration = None


def _environment():
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "flydsl": importlib.metadata.version("flydsl"),
    }


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _files(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"FlyDSL cache cannot contain symlinks: {path}")
        if path.is_file() and path.suffix == ".pkl":
            data = path.read_bytes()
            result[path.relative_to(root).as_posix()] = {
                "sha256": _digest(data),
                "size_bytes": len(data),
            }
    return result


def seal_bundle(directory):
    """Write a build manifest after every requested AOT compilation succeeds."""
    root = Path(directory).resolve()
    files = _files(root)
    if not files:
        raise ValueError("cannot seal an empty FlyDSL AOT bundle")
    for path in root.rglob("*.lock"):
        path.unlink()
    record = {"schema_version": 1, "environment": _environment(), "files": files}
    (root / _MANIFEST).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def _read_manifest(root):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate FlyDSL manifest field: {key}")
            result[key] = value
        return result

    raw = (root / _MANIFEST).read_bytes()
    record = json.loads(raw, object_pairs_hook=unique)
    if (
        type(record) is not dict
        or set(record) != {"schema_version", "environment", "files"}
        or type(record["schema_version"]) is not int
        or record["schema_version"] != 1
        or record["environment"] != _environment()
        or type(record["files"]) is not dict
        or not record["files"]
    ):
        raise ValueError("invalid or incompatible FlyDSL AOT manifest")
    for name, info in record["files"].items():
        path = PurePosixPath(name)
        if (
            not name
            or "\\" in name
            or path.is_absolute()
            or ".." in path.parts
            or str(path) != name
            or path.suffix != ".pkl"
            or type(info) is not dict
            or set(info) != {"sha256", "size_bytes"}
            or type(info["size_bytes"]) is not int
            or info["size_bytes"] < 1
            or type(info["sha256"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", info["sha256"]) is None
        ):
            raise ValueError(f"invalid FlyDSL artifact record: {name}")
    if _files(root) != record["files"]:
        raise ValueError("FlyDSL AOT bundle bytes do not match its manifest")
    return record, _digest(raw)


def installed_bundle():
    """Return only a declared installed bundle; source build caches are inputs."""
    from aiter.codegen import BuildContext

    context = BuildContext.load()
    root = context.package / "jit" / "flydsl_cache"
    return root if context.kind == "installed" and root.is_dir() else None


def prepare_cache(destination, *, bundle=None, run_only=True):
    """Stage verified artifacts, then configure FlyDSL's supported cache mode.

    Call before importing the FlyDSL compiler. ``run_only=True`` prevents
    compilation and requires the exact bundle
    inventory. Ordinary execution can add specializations in a separate private
    cache. No pickle is deserialized by this function.
    """
    global _admitted_configuration
    if type(run_only) is not bool:
        raise ValueError("run_only must be a boolean")
    if not os.fspath(destination).strip():
        raise ValueError("private FlyDSL cache directory cannot be empty")
    source = installed_bundle() if bundle is None else Path(bundle).resolve()
    if source is None:
        raise FileNotFoundError("the installed wheel has no FlyDSL AOT bundle")
    record, digest = _read_manifest(source)
    destination = Path(destination).expanduser().resolve()
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("writable FlyDSL cache must be separate from its bundle")
    target = destination / ("run-only" if run_only else "runtime") / digest
    configuration = (str(source), str(target), digest, run_only)

    def check_admission():
        if "flydsl.compiler.jit_function" in sys.modules:
            if _admitted_configuration != configuration:
                raise RuntimeError(
                    "prepare the FlyDSL bundle before importing its compiler; "
                    "existing JIT objects cannot be retargeted"
                )
            if os.environ.get("FLYDSL_RUNTIME_CACHE_DIR") != str(
                target
            ) or os.environ.get("FLYDSL_RUNTIME_RUN_ONLY") != (
                "1" if run_only else "0"
            ):
                raise RuntimeError("FlyDSL cache configuration changed after admission")

    check_admission()
    target.mkdir(parents=True, exist_ok=True)
    with (target / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = _files(target)
        if run_only and set(existing) - set(record["files"]):
            raise ValueError("run-only FlyDSL cache contains undeclared artifacts")
        for name, info in record["files"].items():
            output = target / name
            if name in existing:
                if existing[name] != info:
                    raise ValueError(f"staged FlyDSL artifact was modified: {name}")
                continue
            data = (source / name).read_bytes()
            if _digest(data) != info["sha256"] or len(data) != info["size_bytes"]:
                raise ValueError(f"FlyDSL bundle changed during staging: {name}")
            output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=output.parent, delete=False
            ) as temporary:
                temporary.write(data)
                temporary_path = Path(temporary.name)
            os.replace(temporary_path, output)
    receipt = {
        "bundle": str(source),
        "cache_dir": str(target),
        "manifest_sha256": digest,
        "run_only": bool(run_only),
        "artifacts": len(record["files"]),
    }
    verify_cache(receipt)
    check_admission()
    os.environ["AITER_FLYDSL_CACHE_DIR"] = str(destination)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(target)
    os.environ["FLYDSL_RUNTIME_RUN_ONLY"] = "1" if run_only else "0"
    _admitted_configuration = configuration
    return receipt


def verify_cache(receipt):
    """Verify staged bytes after execution; run-only mode permits only locks."""
    record, digest = _read_manifest(Path(receipt["bundle"]))
    if digest != receipt["manifest_sha256"]:
        raise ValueError("FlyDSL bundle manifest changed after staging")
    actual = _files(Path(receipt["cache_dir"]))
    if receipt["run_only"] and actual != record["files"]:
        raise ValueError("run-only FlyDSL artifact inventory changed")
    if any(actual.get(name) != info for name, info in record["files"].items()):
        raise ValueError("staged FlyDSL artifact bytes changed")


def configure_installed_cache():
    """Connect normal FlyDSL operator imports to an installed AOT bundle."""
    if os.environ.get("AITER_AOT_IMPORT") == "1":
        return None
    bundle = installed_bundle()
    if bundle is None:
        return None
    base = os.environ.get("AITER_FLYDSL_CACHE_DIR")
    if base is None:
        base = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    if base is None:
        from aiter.jit.cache import cache_directory

        base = cache_directory().parent / "flydsl"
    return prepare_cache(
        base,
        bundle=bundle,
        run_only=os.environ.get("FLYDSL_RUNTIME_RUN_ONLY", "").lower()
        in ("1", "true", "yes", "on"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination")
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--bundle")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        if args.destination or args.bundle:
            parser.error("verification accepts only --receipt")
        result = json.loads(Path(args.receipt).read_text())
        verify_cache(result)
    else:
        if not args.destination:
            parser.error("staging requires --destination")
        result = prepare_cache(args.destination, bundle=args.bundle, run_only=True)
        Path(args.receipt).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

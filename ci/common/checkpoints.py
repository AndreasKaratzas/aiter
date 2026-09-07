# SPDX-License-Identifier: MIT
"""Admit exact checkpoint inputs independently of test and benchmark applications."""

import hashlib
import json
import shutil
from contextlib import nullcontext
from pathlib import Path, PurePosixPath


class ModelUnavailable(RuntimeError):
    """A declared model has not been provisioned into the selected cache."""


def load_manifest(path):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"Duplicate manifest key: {key}")
            result[key] = value
        return result

    manifest = json.loads(Path(path).read_text(), object_pairs_hook=pairs)
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema_version", "models"}
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or not isinstance(manifest["models"], dict)
        or not manifest["models"]
    ):
        raise ValueError("A model manifest requires schema_version1 and models.")
    for name, model in manifest["models"].items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(model, dict)
            or set(model) != {"repository", "revision", "files"}
        ):
            raise ValueError(
                "Model declarations require repository, revision and files only"
            )
        revision = model.get("revision", "")
        if (
            not isinstance(revision, str)
            or len(revision) != 40
            or any(c not in "0123456789abcdef" for c in revision)
        ):
            raise ValueError(f"{name}: model revision must be an exact Git SHA")
        if (
            not isinstance(model["files"], dict)
            or not model["files"]
            or not isinstance(model["repository"], str)
            or not model["repository"]
        ):
            raise ValueError(f"{name}: repository and exact files are required")
        for filename, expected in model["files"].items():
            path = PurePosixPath(filename)
            if path.is_absolute() or ".." in path.parts or str(path) != filename:
                raise ValueError(f"{name}: unsafe model path {filename}")
            if not isinstance(expected, dict) or set(expected) not in (
                {"size", "sha256"},
                {"size", "git_blob_sha1"},
            ):
                raise ValueError(
                    f"{name}: each file requires size and exactly one digest"
                )
            algorithm = "sha256" if "sha256" in expected else "git_blob_sha1"
            digest = expected.get(algorithm, "")
            length = 64 if algorithm == "sha256" else 40
            if (
                not isinstance(digest, str)
                or len(digest) != length
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError(f"{name}: invalid digest for {filename}")
            if type(expected.get("size")) is not int or expected["size"] < 0:
                raise ValueError(f"{name}: invalid byte count for {filename}")
    return manifest["models"]


def verify_snapshot(snapshot, model, *, destination=None, strict=False):
    """Hash the exact bytes consumed, optionally copying only declared files."""
    snapshot = Path(snapshot)
    if strict:
        observed = {
            p.relative_to(snapshot).as_posix()
            for p in snapshot.rglob("*")
            if p.is_file() or p.is_symlink()
        }
        if observed != set(model["files"]):
            raise ValueError(
                "The verified model view contains missing or undeclared files"
            )
    if destination is not None:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        try:
            return _verify_files(snapshot, model, destination)
        except BaseException:
            shutil.rmtree(destination)
            raise
    return _verify_files(snapshot, model, None)


def _verify_files(snapshot, model, destination):
    records = {}
    for filename, expected in model["files"].items():
        path = snapshot / filename
        if not path.is_file():
            raise ModelUnavailable(f"Missing model file: {path}")
        size = path.stat().st_size
        if size != expected["size"]:
            raise ValueError(f"Wrong byte count in model file: {path}")
        sha256 = hashlib.sha256()
        git_blob = hashlib.sha1(f"blob {size}\0".encode(), usedforsecurity=False)
        output = destination / filename if destination is not None else None
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
        with path.open("rb") as source, (
            output.open("xb") if output else nullcontext()
        ) as target:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                sha256.update(chunk)
                git_blob.update(chunk)
                if target is not None:
                    target.write(chunk)
        observed = sha256.hexdigest() if "sha256" in expected else git_blob.hexdigest()
        if observed != expected.get("sha256", expected.get("git_blob_sha1")):
            raise ValueError(f"Model bytes differ from the pinned revision: {path}")
        if output is not None:
            output.chmod(0o444)
        records[filename] = {"size": size, "sha256": sha256.hexdigest()}
    return {
        "repository": model["repository"],
        "revision": model["revision"],
        "snapshot": str((destination or snapshot).resolve()),
        "files": records,
    }

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Immutable build observations and byte identities; no runtime dependencies.

These records are not attestations, compatibility certificates, or cache keys.
An artifact digest identifies bytes. A recipe identifies declared build inputs;
neither establishes that the recipe actually produced those bytes.
"""

import hashlib
import itertools
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from ci.common.validation import (
    ValidationError,
    canonical_digest,
    require_int,
    require_string,
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def _object(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValidationError(f"{name} must contain exactly {sorted(fields)}")
    return value


def _text(value: Any, name: str) -> str:
    value = require_string(value, name)
    if "\x00" in value:
        raise ValidationError(f"{name} must not contain NUL")
    return value


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValidationError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _optional_sha(value: Any, name: str) -> None:
    if value is not None:
        _sha(value, name)


def _strings(value: Any, name: str, *, unique: bool = False) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise ValidationError(f"{name} must be an immutable tuple")
    for item in value:
        _text(item, name)
    if unique and len(set(value)) != len(value):
        raise ValidationError(f"{name} contains duplicates")
    return value


def _array(value: Any, name: str) -> tuple[Any, ...]:
    if not isinstance(value, list):
        raise ValidationError(f"{name} must be a JSON array")
    return tuple(value)


def _pairs(value: Any, name: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, tuple):
        raise ValidationError(f"{name} must be an immutable tuple of pairs")
    keys = set()
    for pair in value:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValidationError(f"{name} must contain string pairs")
        key, val = pair
        _text(key, name)
        _text(val, name)
        if key in keys:
            raise ValidationError(f"{name} contains duplicate key {key!r}")
        keys.add(key)
    return value


def _mapping_pairs(value: Any, name: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, dict):
        raise ValidationError(f"{name} must be a JSON object")
    if any(type(key) is not str for key in value):
        raise ValidationError(f"{name} must have string keys")
    return tuple(sorted(value.items()))


def validate_relative_path(value: Any, name: str = "path") -> str:
    """Validate a portable logical path without filesystem access."""
    value = _text(value, name)
    if "\\" in value or ":" in value or value.startswith("/"):
        raise ValidationError(f"{name} must be a relative POSIX path")
    if any(part in ("", ".", "..") for part in value.split("/")):
        raise ValidationError(f"{name} must not contain empty, dot or parent segments")
    if str(PurePosixPath(value)) != value:
        raise ValidationError(f"{name} is not normalized")
    return value


def _version(value: Any) -> None:
    if require_int(value, "schema_version", minimum=1) != 1:
        raise ValidationError("unsupported artifact schema_version")


def parse_wheel_filename(filename: str) -> tuple[str, tuple[str, ...]]:
    """Read the distribution/version/tag envelope, without a packaging import."""
    validate_relative_path(filename, "wheel filename")
    if "/" in filename or not filename.endswith(".whl"):
        raise ValidationError("wheel filename must be a .whl basename")
    parts = filename[:-4].split("-")
    if len(parts) not in (5, 6) or parts[0] != "amd_aiter":
        raise ValidationError("expected an amd_aiter wheel filename")
    if len(parts) == 6 and not re.fullmatch(r"[0-9][A-Za-z0-9_]*", parts[2]):
        raise ValidationError("invalid wheel build tag")
    groups = [part.split(".") for part in parts[-3:]]
    if any(
        not re.fullmatch(r"[A-Za-z0-9_]+", tag) for group in groups for tag in group
    ):
        raise ValidationError("invalid wheel filename tags")
    tags = tuple(sorted({"-".join(tag) for tag in itertools.product(*groups)}))
    return parts[1], tags


def _digest(payload: dict[str, Any]) -> str:
    return "sha256:" + canonical_digest(payload)


def hash_file(path: str | Path) -> tuple[str, int]:
    """Hash regular-file bytes, rejecting an observed concurrent modification."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValidationError(f"artifact must be a regular, non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        import os

        before = os.fstat(stream.fileno())
        count = 0
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            count += len(chunk)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValidationError(f"file changed while hashing: {path}")
    if count != before.st_size:
        raise ValidationError(f"file size changed while hashing: {path}")
    return digest.hexdigest(), count


@dataclass(frozen=True)
class BuildInput:
    path: str
    sha256: str
    size_bytes: int
    role: str = "source"

    def __post_init__(self) -> None:
        validate_relative_path(self.path)
        _sha(self.sha256, "input.sha256")
        require_int(self.size_bytes, "input.size_bytes", minimum=0)
        if type(self.role) is not str or self.role not in {
            "source",
            "header",
            "generator",
            "generated",
            "dependency",
        }:
            raise ValidationError("unsupported build input role")

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "BuildInput":
        return cls(**_object(value, {"path", "sha256", "size_bytes", "role"}, "input"))


@dataclass(frozen=True)
class BuildRecipe:
    """Declared inputs, preserving command/flag order and explicit omissions.

    The first schema deliberately supports only incomplete input closure.
    It cannot be used to claim reproducible or compiler-free delivery.
    """

    module_name: str
    inputs: tuple[BuildInput, ...]
    command: tuple[str, ...]
    targets: tuple[str, ...]
    toolchain: tuple[tuple[str, str], ...]
    missing_inputs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.module_name, str) or not _NAME.fullmatch(
            self.module_name
        ):
            raise ValidationError("invalid module_name")
        if not isinstance(self.inputs, tuple) or not self.inputs:
            raise ValidationError("recipe inputs must be a nonempty immutable tuple")
        if not all(isinstance(item, BuildInput) for item in self.inputs):
            raise ValidationError("recipe inputs must be BuildInput records")
        if len({item.path for item in self.inputs}) != len(self.inputs):
            raise ValidationError("duplicate input paths")
        _strings(self.command, "command")
        if not self.command:
            raise ValidationError("command cannot be empty")
        _strings(self.targets, "targets", unique=True)
        _pairs(self.toolchain, "toolchain")
        _strings(self.missing_inputs, "missing_inputs", unique=True)
        if not self.missing_inputs:
            raise ValidationError("schema 1 requires explicit missing input closure")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "module_name": self.module_name,
            "inputs": [
                item.to_dict() for item in sorted(self.inputs, key=lambda x: x.path)
            ],
            "command": list(self.command),
            "targets": sorted(self.targets),
            "toolchain": dict(self.toolchain),
            "input_closure": "incomplete",
            "missing_inputs": sorted(self.missing_inputs),
        }

    @property
    def recipe_digest(self) -> str:
        return _digest(self.to_dict())

    @classmethod
    def from_dict(cls, value: Any) -> "BuildRecipe":
        data = _object(
            value,
            {
                "schema_version",
                "module_name",
                "inputs",
                "command",
                "targets",
                "toolchain",
                "input_closure",
                "missing_inputs",
            },
            "recipe",
        )
        _version(data["schema_version"])
        if data["input_closure"] != "incomplete":
            raise ValidationError("schema 1 cannot claim complete input closure")
        return cls(
            data["module_name"],
            tuple(
                BuildInput.from_dict(item) for item in _array(data["inputs"], "inputs")
            ),
            _array(data["command"], "command"),
            _array(data["targets"], "targets"),
            _mapping_pairs(data["toolchain"], "toolchain"),
            _array(data["missing_inputs"], "missing_inputs"),
        )


@dataclass(frozen=True)
class ArtifactFile:
    filename: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        validate_relative_path(self.filename, "artifact.filename")
        if "/" in self.filename or not self.filename.endswith(".whl"):
            raise ValidationError("artifact.filename must be a wheel basename")
        _sha(self.sha256, "artifact.sha256")
        require_int(self.size_bytes, "artifact.size_bytes", minimum=1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ArtifactFile":
        return cls(**_object(value, {"filename", "sha256", "size_bytes"}, "artifact"))


@dataclass(frozen=True)
class WheelMetadata:
    distribution: str
    version: str
    tags: tuple[str, ...]
    requires_python: str | None
    requires_dist: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.distribution != "amd-aiter":
            raise ValidationError("receipt must describe the amd-aiter distribution")
        _text(self.version, "wheel.version")
        if not re.fullmatch(r"[A-Za-z0-9_.+!]+", self.version):
            raise ValidationError("wheel.version contains invalid filename characters")
        _strings(self.tags, "wheel.tags", unique=True)
        if not self.tags or any(
            not re.fullmatch(r"[A-Za-z0-9_]+-[A-Za-z0-9_]+-[A-Za-z0-9_]+", tag)
            for tag in self.tags
        ):
            raise ValidationError("wheel.tags must contain expanded wheel tags")
        if self.requires_python is not None:
            _text(self.requires_python, "wheel.requires_python")
        _strings(self.requires_dist, "wheel.requires_dist")

    def to_dict(self) -> dict[str, Any]:
        return {
            "distribution": self.distribution,
            "version": self.version,
            "tags": sorted(self.tags),
            "requires_python": self.requires_python,
            "requires_dist": sorted(self.requires_dist),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "WheelMetadata":
        data = _object(
            value,
            {
                "distribution",
                "version",
                "tags",
                "requires_python",
                "requires_dist",
            },
            "wheel",
        )
        return cls(
            data["distribution"],
            data["version"],
            _array(data["tags"], "wheel.tags"),
            data["requires_python"],
            _array(data["requires_dist"], "wheel.requires_dist"),
        )


@dataclass(frozen=True)
class SourceIdentity:
    revision: str | None
    ck_gitlink: str | None
    dirty: bool | None
    tracked_diff_sha256: str | None
    untracked_digest: str | None
    observation: str = "receipt_creation"

    def __post_init__(self) -> None:
        for name in ("revision", "ck_gitlink"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not _GIT_OID.fullmatch(value)
            ):
                raise ValidationError(f"source.{name} must be a full Git object ID")
        if self.dirty is not None and type(self.dirty) is not bool:
            raise ValidationError("source.dirty must be boolean or null")
        _optional_sha(self.tracked_diff_sha256, "source.tracked_diff_sha256")
        _optional_sha(self.untracked_digest, "source.untracked_digest")
        if self.observation != "receipt_creation":
            raise ValidationError("source observation must be receipt_creation")
        values = (self.dirty, self.tracked_diff_sha256, self.untracked_digest)
        if self.revision is None and any(value is not None for value in values):
            raise ValidationError("unidentified source must not claim a worktree state")
        if self.revision is not None and any(value is None for value in values):
            raise ValidationError(
                "identified source requires the full observed worktree state"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "ck_gitlink": self.ck_gitlink,
            "dirty": self.dirty,
            "tracked_diff_sha256": self.tracked_diff_sha256,
            "untracked_digest": self.untracked_digest,
            "observation": self.observation,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "SourceIdentity":
        return cls(
            **_object(
                value,
                {
                    "revision",
                    "ck_gitlink",
                    "dirty",
                    "tracked_diff_sha256",
                    "untracked_digest",
                    "observation",
                },
                "source",
            )
        )


@dataclass(frozen=True)
class ArtifactReceipt:
    artifact: ArtifactFile
    wheel: WheelMetadata
    source: SourceIdentity
    environment: tuple[tuple[str, str], ...]
    recipe: BuildRecipe | None = None
    limitations: tuple[str, ...] = (
        "Source and environment are observations at receipt creation, not build attestations.",
        "Transitive build inputs, external dependency bytes and native target coverage are unverified.",
        "No API/ABI, numerical, performance or compiler-free compatibility is certified.",
    )

    def __post_init__(self) -> None:
        for name, kind in (
            ("artifact", ArtifactFile),
            ("wheel", WheelMetadata),
            ("source", SourceIdentity),
        ):
            if not isinstance(getattr(self, name), kind):
                raise ValidationError(f"{name} must be a {kind.__name__} record")
        _pairs(self.environment, "environment")
        if self.recipe is not None and not isinstance(self.recipe, BuildRecipe):
            raise ValidationError("recipe must be a BuildRecipe or null")
        _strings(self.limitations, "limitations", unique=True)
        if not self.limitations:
            raise ValidationError("receipt limitations must be explicit")
        version, tags = parse_wheel_filename(self.artifact.filename)
        if version != self.wheel.version or tags != tuple(sorted(self.wheel.tags)):
            raise ValidationError("wheel filename and metadata disagree")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "wheel",
            "artifact": self.artifact.to_dict(),
            "wheel": self.wheel.to_dict(),
            "source": self.source.to_dict(),
            "environment": dict(self.environment),
            "recipe": self.recipe.to_dict() if self.recipe is not None else None,
            "limitations": list(self.limitations),
        }

    @property
    def receipt_digest(self) -> str:
        return _digest(self._payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "receipt_digest": self.receipt_digest}

    @classmethod
    def from_dict(cls, value: Any) -> "ArtifactReceipt":
        data = _object(
            value,
            {
                "schema_version",
                "kind",
                "artifact",
                "wheel",
                "source",
                "environment",
                "recipe",
                "limitations",
                "receipt_digest",
            },
            "receipt",
        )
        _version(data["schema_version"])
        if data["kind"] != "wheel":
            raise ValidationError("only wheel receipts are supported")
        receipt = cls(
            ArtifactFile.from_dict(data["artifact"]),
            WheelMetadata.from_dict(data["wheel"]),
            SourceIdentity.from_dict(data["source"]),
            _mapping_pairs(data["environment"], "environment"),
            None if data["recipe"] is None else BuildRecipe.from_dict(data["recipe"]),
            _array(data["limitations"], "limitations"),
        )
        if data["receipt_digest"] != receipt.receipt_digest:
            raise ValidationError("receipt digest does not match its contents")
        return receipt

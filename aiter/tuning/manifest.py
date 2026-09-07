# SPDX-License-Identifier: MIT
"""Exact-workload selection, pinned when a runtime prepares an operation."""

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .._validation import ValidationError, canonical_digest, canonical_json


def digest(value, name):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValidationError(f"{name} must be a lowercase SHA-256 digest")


def identifier(value, name):
    if type(value) is not str or re.fullmatch(r"[a-z0-9][a-z0-9_.:/-]*", value) is None:
        raise ValidationError(f"invalid {name}")


@dataclass(frozen=True)
class Selection:
    request_id: str
    target: str
    backend: str
    artifact_digest: str
    evidence_digest: str

    def __post_init__(self):
        digest(self.request_id, "request_id")
        digest(self.artifact_digest, "artifact_digest")
        digest(self.evidence_digest, "evidence_digest")
        identifier(self.target, "target")
        identifier(self.backend, "backend")

    def to_dict(self):
        return {
            "request_id": self.request_id,
            "target": self.target,
            "backend": self.backend,
            "artifact_digest": self.artifact_digest,
            "evidence_digest": self.evidence_digest,
        }


@dataclass(frozen=True)
class DispatchManifest:
    selections: tuple[Selection, ...]
    environment_digest: str
    protocol: str

    def __post_init__(self):
        if type(self.selections) is not tuple or not self.selections:
            raise ValidationError(
                "manifest selections must be a nonempty immutable tuple"
            )
        if any(not isinstance(value, Selection) for value in self.selections):
            raise ValidationError("manifest entries must be Selection values")
        keys = [(value.request_id, value.target) for value in self.selections]
        if len(keys) != len(set(keys)):
            raise ValidationError("duplicate exact-workload selection")
        digest(self.environment_digest, "environment_digest")
        identifier(self.protocol, "protocol")

    def select(self, request_id, target):
        for value in self.selections:
            if value.request_id == request_id and value.target == target:
                return value
        raise ValidationError(
            "manifest contains no approved implementation for this exact workload and target"
        )

    def to_dict(self):
        return {
            "schema_version": 1,
            "scope": "exact-workload",
            "environment_digest": self.environment_digest,
            "protocol": self.protocol,
            "selections": [
                item.to_dict()
                for item in sorted(
                    self.selections, key=lambda item: (item.request_id, item.target)
                )
            ],
        }

    @property
    def digest(self):
        return canonical_digest(self.to_dict())

    def write(self, path):
        path = Path(path)
        text = canonical_json({**self.to_dict(), "digest": self.digest}) + "\n"
        if path.exists():
            if path.read_text() != text:
                raise ValidationError(
                    "refusing to overwrite a different dispatch manifest"
                )
            return
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValidationError(
                "manifest appeared while writing; refusing replacement"
            ) from error
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def read(cls, path):
        def unique(pairs):
            values = {}
            for key, value in pairs:
                if key in values:
                    raise ValidationError(f"duplicate manifest field: {key}")
                values[key] = value
            return values

        data = json.loads(Path(path).read_text(), object_pairs_hook=unique)
        required = {
            "schema_version",
            "scope",
            "environment_digest",
            "protocol",
            "selections",
            "digest",
        }
        if (
            type(data) is not dict
            or set(data) != required
            or type(data["schema_version"]) is not int
            or data["schema_version"] != 1
            or data["scope"] != "exact-workload"
        ):
            raise ValidationError("invalid manifest envelope")
        if type(data["selections"]) is not list:
            raise ValidationError("selections must be an array")
        try:
            manifest = cls(
                tuple(Selection(**item) for item in data["selections"]),
                data["environment_digest"],
                data["protocol"],
            )
        except TypeError as error:
            raise ValidationError("invalid selection fields") from error
        if manifest.digest != data["digest"]:
            raise ValidationError("dispatch manifest identity mismatch")
        return manifest

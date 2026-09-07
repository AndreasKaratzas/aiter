# SPDX-License-Identifier: MIT
"""Keep raw measurements separate from runtime decisions."""

import csv
import hashlib
import io
import statistics
from dataclasses import dataclass
from pathlib import Path

from .._validation import ValidationError, canonical_digest
from .manifest import DispatchManifest, Selection, digest, identifier


@dataclass(frozen=True)
class Trial:
    request_id: str
    target: str
    backend: str
    artifact_digest: str
    environment_digest: str
    protocol: str
    samples_ns: tuple[int, ...]
    correctness_passed: bool
    evidence_digest: str

    def __post_init__(self):
        for name in (
            "request_id",
            "artifact_digest",
            "environment_digest",
            "evidence_digest",
        ):
            digest(getattr(self, name), name)
        for name in ("target", "backend", "protocol"):
            identifier(getattr(self, name), name)
        if (
            type(self.samples_ns) is not tuple
            or len(self.samples_ns) < 3
            or any(type(value) is not int or value <= 0 for value in self.samples_ns)
        ):
            raise ValidationError(
                "a trial requires at least three positive integer nanosecond samples"
            )
        if type(self.correctness_passed) is not bool:
            raise ValidationError("correctness_passed must be boolean")


def promote(trials, *, environment_digest, protocol):
    """Choose measured implementations only after every supplied trial is correct.

    Callers retain the actual numerical/timing evidence identified by each trial.
    This policy consumes that evidence; it does not authenticate its producer.
    """
    digest(environment_digest, "environment_digest")
    identifier(protocol, "protocol")
    if not trials:
        raise ValidationError("cannot promote an empty trial set")
    groups = {}
    identities = set()
    for trial in trials:
        if not isinstance(trial, Trial):
            raise ValidationError("legacy rows are not qualified trials")
        if not trial.correctness_passed:
            raise ValidationError("a correctness failure blocks this promotion")
        if trial.environment_digest != environment_digest or trial.protocol != protocol:
            raise ValidationError("incomparable environments or timing protocols")
        identity = (
            trial.request_id,
            trial.target,
            trial.backend,
            trial.artifact_digest,
        )
        if identity in identities:
            raise ValidationError(
                "duplicate implementation trial; consolidate its samples explicitly"
            )
        identities.add(identity)
        groups.setdefault((trial.request_id, trial.target), []).append(trial)
    selected = []
    for values in groups.values():
        best = min(
            values,
            key=lambda value: (
                statistics.median(value.samples_ns),
                value.backend,
                value.artifact_digest,
            ),
        )
        evidence = canonical_digest(
            {"trials": sorted(value.evidence_digest for value in values)}
        )
        selected.append(
            Selection(
                best.request_id,
                best.target,
                best.backend,
                best.artifact_digest,
                evidence,
            )
        )
    return DispatchManifest(tuple(selected), environment_digest, protocol)


def import_legacy_csv(path):
    """Preserve each row and its source bytes; missing provenance is never invented."""
    path = Path(path)
    content = path.read_bytes()
    with io.StringIO(content.decode("utf-8"), newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(
            set(reader.fieldnames)
        ):
            raise ValidationError("CSV requires distinct column names")
        rows = list(reader)
        if any(None in row or None in row.values() for row in rows):
            raise ValidationError("CSV row width differs from its header")
    return {
        "schema_version": 1,
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "status": "unverified-measurements",
        "rows": rows,
    }

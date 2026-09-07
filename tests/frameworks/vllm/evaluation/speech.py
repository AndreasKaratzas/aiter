# SPDX-License-Identifier: MIT
"""Word-error scoring against recorded human LibriSpeech test-clean transcripts."""

import hashlib
import json
import re
from pathlib import Path


def words(text):
    # Qwen-ASR separates the language prefix from its actual transcription.
    text = text.rsplit("<asr_text>", 1)[-1]
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower().replace("’", "'"))


def word_errors(reference, candidate):
    expected, actual = words(reference), words(candidate)
    if not expected:
        raise ValueError("Empty speech reference")
    previous = list(range(len(actual) + 1))
    for i, left in enumerate(expected, 1):
        row = [i]
        for j, right in enumerate(actual, 1):
            row.append(
                min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right))
            )
        previous = row
    return previous[-1], len(expected)


def load_utterances(count):
    root = Path(__file__).parent / "fixtures/librispeech"
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    if count not in (manifest["smoke_count"], manifest["full_count"]):
        raise ValueError("Select declared speech smoke or complete retained slice")
    rows = manifest["utterances"][:count]
    if len(rows) != count or len({r["id"] for r in rows}) != count:
        raise ValueError("Missing or duplicate speech rows")
    for row in rows:
        data = (root / row["file"]).read_bytes()
        if (
            len(data) != row["size"]
            or hashlib.sha256(data).hexdigest() != row["sha256"]
        ):
            raise ValueError("Recorded speech bytes changed")
    return root, manifest, rows


def score(rows, predictions):
    if len(rows) != len(predictions) or not rows:
        raise ValueError("Every utterance must remain in the WER denominator")
    records = []
    for row, prediction in zip(rows, predictions, strict=True):
        errors, total = word_errors(row["transcript"], prediction)
        records.append(
            {
                "id": row["id"],
                "prediction": prediction,
                "errors": errors,
                "reference_words": total,
            }
        )
    errors = sum(r["errors"] for r in records)
    total = sum(r["reference_words"] for r in records)
    return {
        "word_error_rate": errors / total,
        "errors": errors,
        "reference_words": total,
        "utterances": records,
    }

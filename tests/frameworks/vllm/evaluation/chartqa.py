# SPDX-License-Identifier: MIT
"""Real chart questions with exact strings or declared5% relative numeric scoring."""

import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from frameworks.vllm.evaluation.gpqa import DatasetUnavailable


def matches(expected, prediction):
    expected, prediction = expected.strip().lower(), prediction.strip().lower()
    if expected == prediction:
        return True
    try:
        a, b = (
            Decimal(expected.strip("%").replace(",", "")),
            Decimal(prediction.strip("%").replace(",", "")),
        )
    except InvalidOperation:
        return False
    if not a.is_finite() or not b.is_finite():
        return False
    return abs(a - b) <= abs(a) * Decimal("0.05") if a else b == 0


def load_rows(directory, *, path=None):
    fixture = json.loads((Path(__file__).parent / "fixtures/chartqa.json").read_text())
    source, selection = fixture["source"], fixture["selection"]
    if path is None:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            path = hf_hub_download(
                source["repository"],
                source["file"],
                repo_type="dataset",
                revision=source["revision"],
                local_files_only=True,
            )
        except LocalEntryNotFoundError as error:
            raise DatasetUnavailable(
                "Provision the pinned ChartQA test Parquet before selecting this profile"
            ) from error
    path = Path(path)
    if (
        path.stat().st_size != source["size"]
        or hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]
    ):
        raise ValueError("ChartQA source bytes changed")
    import pyarrow.parquet as pq

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, row in enumerate(pq.read_table(path).to_pylist()):
        if row["human_or_machine"] != selection["human_or_machine"]:
            continue
        data = row["image"]["bytes"]
        image = directory / f"chart-{index}.png"
        image.write_bytes(data)
        rows.append(
            {
                "id": f"test:{index}",
                "question": row["query"],
                "answers": row["label"],
                "image_path": str(image.resolve()),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        if len(rows) == selection["count"]:
            break
    if len(rows) != selection["count"]:
        raise ValueError("ChartQA selection is incomplete")
    return fixture, rows


def score(rows, predictions):
    if (
        not rows
        or len(rows) != len(predictions)
        or len({row["id"] for row in rows}) != len(rows)
    ):
        raise ValueError(
            "All distinct selected charts must remain in accuracy denominator"
        )
    records = [
        {
            "id": row["id"],
            "prediction": prediction,
            "answers": row["answers"],
            "correct": any(matches(answer, prediction) for answer in row["answers"]),
        }
        for row, prediction in zip(rows, predictions, strict=True)
    ]
    correct = sum(row["correct"] for row in records)
    return {
        "correct": correct,
        "total": len(records),
        "accuracy": correct / len(records),
        "questions": records,
    }

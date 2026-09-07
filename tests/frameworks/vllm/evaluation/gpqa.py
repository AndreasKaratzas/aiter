# SPDX-License-Identifier: MIT
"""Exact-byte GPQA Diamond admission and redacted multiple-choice evaluation."""

import csv
import hashlib
import io
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class DatasetUnavailable(RuntimeError):
    """Provision publisher-approved GPQA bytes before executing this profile."""


def load_questions(count, *, path=None):
    fixture = json.loads((Path(__file__).parent / "fixtures/gpqa.json").read_text())
    source, selection = fixture["source"], fixture["selection"]
    if count not in (selection["smoke_rows"], selection["total_rows"]):
        raise ValueError("Select the declared GPQA smoke32 or complete Diamond198")
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
                "GPQA Diamond is not provisioned. Obtain publisher-approved access and cache the exact declared revision; this is not a passing evaluation."
            ) from error
    data = Path(path).read_bytes()
    digest = hashlib.sha1(
        f"blob {len(data)}\0".encode() + data, usedforsecurity=False
    ).hexdigest()
    if len(data) != source["size"] or digest != source["git_blob_sha1"]:
        raise ValueError("GPQA bytes differ from the pinned publisher revision")
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
    if len(rows) != selection["total_rows"]:
        raise ValueError("GPQA row inventory differs")
    questions = [
        make_question(row, index, selection["shuffle_seed"])
        for index, row in enumerate(rows)
    ]
    if len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Duplicate GPQA questions")
    return fixture, questions[:count], hashlib.sha256(data).hexdigest()


def make_question(row, index, seed):
    question = row["Question"].strip()
    answers = [
        row["Correct Answer"].strip(),
        *(row[f"Incorrect Answer {i}"].strip() for i in range(1, 4)),
    ]
    if not question or any(not answer for answer in answers) or len(set(answers)) != 4:
        raise ValueError("GPQA question needs four distinct nonempty choices")
    order = list(range(4))
    random.Random(seed + index).shuffle(order)
    return {
        "id": hashlib.sha256(question.encode()).hexdigest(),
        "row": index,
        "question": question,
        "choices": [answers[i] for i in order],
        "answer": "ABCD"[order.index(0)],
    }


def final_choice(text):
    if not isinstance(text, str):
        return None
    if len(re.findall(r"FINAL:", text)) != 1:
        return None
    match = re.search(r"FINAL:\s*([ABCD])\s*\.?\s*$", text)
    return match.group(1) if match else None


def score(questions, predictions):
    if (
        not questions
        or len(questions) != len(predictions)
        or len({q["id"] for q in questions}) != len(questions)
    ):
        raise ValueError("Every unique GPQA row needs exactly one response")
    records = [
        {
            "id": q["id"],
            "row": q["row"],
            "selected": final_choice(p),
            "correct": final_choice(p) == q["answer"],
        }
        for q, p in zip(questions, predictions, strict=True)
    ]
    correct = sum(row["correct"] for row in records)
    return {
        "correct": correct,
        "total": len(records),
        "accuracy": correct / len(records),
        "questions": records,
    }


def evaluate(server, fixture, questions, dataset_sha256, *, output):
    selection = fixture["selection"]

    def answer(item):
        text = (
            item["question"]
            + "\n\n"
            + "\n".join(
                f"{label}. {value}"
                for label, value in zip("ABCD", item["choices"], strict=True)
            )
        )
        response = server.request(
            "/v1/chat/completions",
            {
                "model": "aiter-fixture",
                "messages": [
                    {
                        "role": "system",
                        "content": "Solve the multiple-choice question. End the final response with exactly FINAL: followed by the single letter A, B, C, or D.",
                    },
                    {"role": "user", "content": text},
                ],
                "temperature": 0,
                "seed": 0,
                "reasoning_effort": selection["reasoning_effort"],
                "max_tokens": selection["max_tokens"],
            },
            retain=False,
        )
        # Requests and free-form responses are intentionally excluded from retained public records.
        return response["choices"][0]["message"].get("content")

    with ThreadPoolExecutor(max_workers=4) as pool:
        predictions = list(pool.map(answer, questions))
    result = score(questions, predictions)
    result.update(
        source=fixture["source"],
        dataset_sha256=dataset_sha256,
        selection=selection,
        model=server.model,
    )
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    if result["accuracy"] < selection["minimum_accuracy"]:
        raise AssertionError(
            f"GPQA accuracy {result['correct']}/{result['total']} below predeclared {selection['minimum_accuracy']}"
        )
    return result

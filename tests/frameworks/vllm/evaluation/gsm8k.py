# SPDX-License-Identifier: MIT
"""Pinned GSM8K exact-answer evaluation; all selected questions stay in the denominator."""

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation
from pathlib import Path


def final_answer(text):
    if "####" not in text:
        return None
    suffix = text.rsplit("####", 1)[1].strip().replace(",", "")
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?\.?", suffix):
        return None
    try:
        return Decimal(suffix.rstrip("."))
    except InvalidOperation:
        return None


def score_predictions(questions, predictions):
    if len(questions) != len(predictions) or not questions:
        raise ValueError("Every selected question requires exactly one prediction")
    if len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Duplicate evaluation question")
    correct = 0
    results = []
    for question, prediction in zip(questions, predictions, strict=True):
        reference = final_answer(question["answer"])
        if reference is None:
            raise ValueError("Dataset reference has no unambiguous final answer")
        actual = final_answer(prediction)
        match = actual is not None and actual == reference
        correct += match
        results.append(
            {
                "id": question["id"],
                "prediction": prediction,
                "expected": str(reference),
                "actual": str(actual) if actual is not None else None,
                "correct": match,
            }
        )
    return {
        "correct": correct,
        "total": len(questions),
        "accuracy": correct / len(questions),
        "questions": results,
    }


def evaluate(server, *, count, output):
    path = Path(__file__).parent / "fixtures/gsm8k.json"
    fixture = json.loads(path.read_text())
    if count not in (
        fixture["selection"]["daily_count"],
        fixture["selection"]["extended_count"],
    ):
        raise ValueError(
            "Evaluation size must select the declared daily or extended slice"
        )
    offset = (
        fixture["selection"]["extended_offset"]
        if count == fixture["selection"]["extended_count"]
        else 0
    )
    questions = fixture["questions"][offset : offset + count]
    if len(questions) != count or [q["id"] for q in questions] != [
        f"test:{i}" for i in range(offset, offset + count)
    ]:
        raise ValueError("Pinned evaluation rows are missing or out of order")
    prefix = [
        {
            "role": "system",
            "content": "Solve each arithmetic problem. End your answer with #### followed by the final number.",
        }
    ]
    for shot in fixture["fewshot"]:
        prefix.extend(
            [
                {"role": "user", "content": shot["question"]},
                {"role": "assistant", "content": shot["answer"]},
            ]
        )

    def answer(question):
        response = server.request(
            "/v1/chat/completions",
            {
                "model": "aiter-fixture",
                "messages": prefix
                + [{"role": "user", "content": question["question"]}],
                "max_tokens": fixture["selection"]["max_tokens"],
                "temperature": 0,
                "seed": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        return response["choices"][0]["message"]["content"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        predictions = list(pool.map(answer, questions))
    result = score_predictions(questions, predictions)
    result.update(
        source=fixture["source"],
        fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        minimum_accuracy=fixture["selection"]["minimum_accuracy"],
        model=server.model,
        scope="Fixed original GSM8K test slice; deterministic non-thinking Qwen3 with four training demonstrations",
    )
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    if result["accuracy"] < result["minimum_accuracy"]:
        raise AssertionError(
            f"GSM8K accuracy {result['correct']}/{count} below {result['minimum_accuracy']}"
        )
    return result

# SPDX-License-Identifier: MIT
"""Model provisioning is explicit; execution only reads verified cached bytes."""

import json
from pathlib import Path

import pytest
from common.models import (
    ModelUnavailable,
    load_manifest,
    resolve_model,
    verify_snapshot,
)
from common.paths import SUITE_ROOT
from common.prerequisites import require_available


@pytest.fixture(scope="session")
def model_manifest():
    return load_manifest(SUITE_ROOT.parent / "ci/clients/vllm/models.json")


@pytest.fixture(scope="session")
def engine_inputs():
    return json.loads(
        (Path(__file__).parent / "models/fixtures/inputs.json").read_text()
    )


@pytest.fixture(scope="session")
def text_prompts(engine_inputs):
    data = engine_inputs["speculative"]
    return (
        data["instruction"] + data["sequence"] * data["repetitions"],
        data["instruction"] + "one two three four five\n" * data["repetitions"],
    )


@pytest.fixture(scope="session")
def vision_prompts(engine_inputs):
    data = engine_inputs["multimodal"]
    return tuple(
        {"text": data["prompt"], "rgb": item["rgb"], "size": data["image_size"]}
        for item in data["images"]
    )


@pytest.fixture
def model_snapshot(request, model_manifest, tmp_path, e2e_evidence):
    marker = request.node.get_closest_marker("model")
    if marker is None or len(marker.args) != 1:
        pytest.fail("A model test must declare exactly one pinned model fixture.")
    name = marker.args[0]
    if name not in model_manifest:
        pytest.fail(f"Unknown model fixture: {name}")
    try:
        receipt = resolve_model(model_manifest[name], destination=tmp_path / "model")
    except ModelUnavailable as error:
        require_available(False, str(error), request)
    (e2e_evidence / "model-before.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    yield receipt
    # The engine consumed this private, bounded view, never incidental HF cache files.
    observed = verify_snapshot(receipt["snapshot"], model_manifest[name], strict=True)
    (e2e_evidence / "model-after.json").write_text(
        json.dumps(observed, indent=2) + "\n"
    )
    assert observed == receipt, "Model inputs changed during execution."

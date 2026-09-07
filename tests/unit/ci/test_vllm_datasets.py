# SPDX-License-Identifier: MIT
"""Public dataset provisioning stays explicit, exact and separate from gated GPQA."""

import hashlib
import json
from unittest.mock import Mock

import pytest

from ci.clients.vllm.datasets import provision, required_inputs


def declaration(tmp_path):
    raw = b"controlled dataset byte fixture"
    source = {
        "repository": "HuggingFaceM4/ChartQA",
        "revision": "a" * 40,
        "file": "data/test.parquet",
        "size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    path = tmp_path / "tests/frameworks/vllm/evaluation/fixtures/chartqa.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema_version": 1, "source": source}))
    data = tmp_path / "selected.parquet"
    data.write_bytes(raw)
    return source, data


def test_only_explicit_public_group_provisions_and_gated_gpqa_never_downloads():
    download = Mock(side_effect=AssertionError("Unexpected dataset download"))
    assert (
        provision(
            ["vllm-gpqa-smoke", "vllm-gpqa-diamond", "vllm-bridge"], download=download
        )["inputs"]
        == []
    )
    download.assert_not_called()
    inputs = required_inputs(["vllm-chartqa", "vllm-chartqa"])
    assert (
        len(inputs) == 1
        and inputs[0]["source"]["repository"] == "HuggingFaceM4/ChartQA"
    )


def test_provision_binds_revision_repo_type_exact_bytes_and_declaration(tmp_path):
    source, data = declaration(tmp_path)
    download = Mock(return_value=str(data))
    result = provision(
        ["vllm-chartqa"], root=tmp_path, cache_dir="/owned/cache", download=download
    )
    download.assert_called_once_with(
        source["repository"],
        source["file"],
        repo_type="dataset",
        revision=source["revision"],
        cache_dir="/owned/cache",
    )
    record = result["inputs"][0]
    assert record["sha256"] == source["sha256"] and record["size"] == source["size"]
    assert (
        record["declaration_sha256"]
        == hashlib.sha256((tmp_path / record["declaration"]).read_bytes()).hexdigest()
    )
    data.write_bytes(b"x" * source["size"])
    with pytest.raises(ValueError, match="differ"):
        provision(["vllm-chartqa"], root=tmp_path, download=download)


@pytest.mark.parametrize(
    "change",
    [
        {"repository": "unreviewed/other"},
        {"revision": "moving-main"},
        {"file": "../escape"},
        {"file": "/absolute"},
        {"size": True},
        {"sha256": "not-a-digest"},
    ],
)
def test_bad_declaration_fails_before_network(tmp_path, change):
    _, _ = declaration(tmp_path)
    path = tmp_path / "tests/frameworks/vllm/evaluation/fixtures/chartqa.json"
    value = json.loads(path.read_text())
    value["source"].update(change)
    path.write_text(json.dumps(value))
    download = Mock(side_effect=AssertionError("Network before admission"))
    with pytest.raises(ValueError):
        provision(["vllm-chartqa"], root=tmp_path, download=download)
    download.assert_not_called()

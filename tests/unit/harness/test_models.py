# SPDX-License-Identifier: MIT
"""The loader receives only the exact declared model bytes."""

import hashlib
import json

import pytest
from common.models import load_manifest, verify_snapshot


@pytest.fixture
def model(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    data = b"real fixture bytes for admission testing, not model qualification"
    (snapshot / "weights.safetensors").write_bytes(data)
    model = {
        "repository": "owner/model",
        "revision": "a" * 40,
        "files": {
            "weights.safetensors": {
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        },
    }
    return snapshot, model


def test_model_projection_excludes_incidental_weights_and_remote_code(model, tmp_path):
    snapshot, declaration = model
    (snapshot / "extra.safetensors").write_text("undeclared weights")
    (snapshot / "config.json").write_text('{"auto_map": "unreviewed.py"}')
    (snapshot / "unreviewed.py").write_text("raise RuntimeError('must not load')")
    view = tmp_path / "view"
    receipt = verify_snapshot(snapshot, declaration, destination=view)
    assert {p.name for p in view.iterdir()} == {"weights.safetensors"}
    assert not (view / "weights.safetensors").is_symlink()
    assert verify_snapshot(view, declaration, strict=True) == receipt
    (view / "extra.safetensors").write_text("late extra weights")
    with pytest.raises(ValueError, match="undeclared"):
        verify_snapshot(view, declaration, strict=True)


def test_same_size_tamper_fails_and_discards_partial_view(model, tmp_path):
    snapshot, declaration = model
    path = snapshot / "weights.safetensors"
    path.write_bytes(b"x" * path.stat().st_size)
    view = tmp_path / "view"
    with pytest.raises(ValueError, match="differ"):
        verify_snapshot(snapshot, declaration, destination=view)
    assert not view.exists()


@pytest.mark.parametrize(
    "change",
    [
        "boolean_schema",
        "extra_top",
        "extra_model",
        "ambiguous_hash",
        "extra_file_field",
        "path_escape",
        "absolute_path",
    ],
)
def test_manifest_rejects_ambiguous_or_unsafe_declarations(model, tmp_path, change):
    _, declaration = model
    payload = {"schema_version": 1, "models": {"fixture": declaration}}
    record = declaration["files"]["weights.safetensors"]
    if change == "boolean_schema":
        payload["schema_version"] = True
    elif change == "extra_top":
        payload["execute"] = "command"
    elif change == "extra_model":
        declaration["execute"] = "command"
    elif change == "ambiguous_hash":
        record["git_blob_sha1"] = "a" * 40
    elif change == "extra_file_field":
        record["execute"] = "command"
    elif change == "path_escape":
        declaration["files"] = {"../weights": record}
    elif change == "absolute_path":
        declaration["files"] = {"/weights": record}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_manifest(path)


def test_git_blob_metadata_hash_and_normal_hf_blob_symlinks(model, tmp_path):
    snapshot, declaration = model
    contents = b'{"model_type":"llama"}'
    blob = tmp_path / "blob"
    blob.write_bytes(contents)
    (snapshot / "config.json").symlink_to(blob)
    declaration["files"]["config.json"] = {
        "size": len(contents),
        "git_blob_sha1": hashlib.sha1(
            f"blob {len(contents)}\0".encode() + contents, usedforsecurity=False
        ).hexdigest(),
    }
    receipt = verify_snapshot(snapshot, declaration, destination=tmp_path / "view")
    assert (
        receipt["files"]["config.json"]["sha256"]
        == hashlib.sha256(contents).hexdigest()
    )

# SPDX-License-Identifier: MIT
"""Approved selections constrain real preparation and subsequent execution."""

from dataclasses import replace

import pytest


def _fixture():
    import torch

    from aiter.runtime import Runtime
    from aiter.tuning import DispatchManifest, Selection

    assert torch.cuda.is_available() and torch.version.hip, "A ROCm GPU is required."
    torch.manual_seed(47)
    x = torch.randn(16, 1024, device="cuda:0", dtype=torch.bfloat16)
    weight = torch.randn(1024, device="cuda:0", dtype=x.dtype)
    out = torch.full_like(x, 7)
    runtime = Runtime(device=0)
    plan = runtime.prepare_rmsnorm(x, weight, out, backend="triton")
    detail = plan.explain()
    selection = Selection(
        detail["request_id"],
        detail["target"],
        "triton",
        detail["artifact_digest"],
        "e" * 64,
    )
    manifest = DispatchManifest(
        (selection,), runtime.environment_digest, "qa-functional-v1"
    )
    return torch, runtime, {"x": x, "weight": weight, "out": out}, manifest


def test_approved_manifest_selects_exact_artifact_and_survives_reload(tmp_path):
    from aiter.runtime import Runtime
    from aiter.tuning import DispatchManifest

    torch, baseline, bindings, manifest = _fixture()
    path = tmp_path / "dispatch.json"
    manifest.write(path)
    restored = DispatchManifest.read(path)
    runtime = Runtime(device=0, manifest=restored)
    plan = runtime.prepare_rmsnorm(bindings["x"], bindings["weight"], bindings["out"])
    assert plan.explain()["provider"] == "triton"
    assert plan.explain()["manifest_digest"] == manifest.digest
    assert plan.explain()["environment_digest"] == baseline.environment_digest
    plan.execute(bindings)
    x = bindings["x"].float()
    expected = (
        x
        * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        * bindings["weight"].float()
    ).to(bindings["out"].dtype)
    torch.testing.assert_close(bindings["out"], expected, rtol=0.02, atol=0.02)


@pytest.mark.parametrize(
    "mutation", ["artifact", "environment", "request", "backend", "policy"]
)
def test_incompatible_manifest_fails_before_output_changes(mutation):
    from aiter.api import ValidationError
    from aiter.runtime import ExecutionPolicy, Runtime

    torch, _, bindings, manifest = _fixture()
    before = bindings["out"].clone()
    backend = None
    policy = ExecutionPolicy()
    if mutation == "artifact":
        manifest = replace(
            manifest,
            selections=(replace(manifest.selections[0], artifact_digest="f" * 64),),
        )
    elif mutation == "environment":
        manifest = replace(manifest, environment_digest="f" * 64)
    elif mutation == "request":
        manifest = replace(
            manifest, selections=(replace(manifest.selections[0], request_id="f" * 64),)
        )
    elif mutation == "backend":
        backend = "hip"
    elif mutation == "policy":
        policy = ExecutionPolicy(backend_order=("hip",))
    with pytest.raises(ValidationError):
        runtime = Runtime(device=0, manifest=manifest, policy=policy)
        runtime.prepare_rmsnorm(
            bindings["x"], bindings["weight"], bindings["out"], backend=backend
        )
    torch.testing.assert_close(bindings["out"], before, rtol=0, atol=0)

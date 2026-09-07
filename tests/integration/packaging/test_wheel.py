# SPDX-License-Identifier: MIT
"""Native providers execute the installed wheel's exact bytes without compilation."""

import hashlib
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("backend", ["hip", "ck"])
def test_native_wheel_bundle_executes_without_build_tools(backend, monkeypatch):
    import subprocess

    import torch

    import aiter
    import aiter.backends.ck
    import aiter.backends.hip
    from aiter.runtime import ExecutionPolicy, Runtime

    directory = Path(aiter.__file__).parent / "lib"
    assert (
        directory / "manifest.json"
    ).is_file(), "This test requires an installed wheel containing the native SDK"
    manifest = json.loads((directory / "manifest.json").read_text())
    filename = (
        "libaiter_rmsnorm_backend.so" if backend == "hip" else "libaiter_ck_backend.so"
    )
    expected_digest = manifest["libraries"][filename]["sha256"]
    assert (
        hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        == expected_digest
    )
    for variable in (
        "AITER_JIT_DIR",
        "AITER_RMSNORM_LIBRARY",
        "AITER_CK_BLOCKSCALE_LIBRARY",
    ):
        monkeypatch.delenv(variable, raising=False)
    runtime = Runtime(device=0, policy=ExecutionPolicy(allow_compile=False))
    torch.manual_seed(61)
    if backend == "hip":
        x = torch.randn(17, 409, device="cuda:0", dtype=torch.bfloat16)
        weight = torch.randn(409, device="cuda:0", dtype=x.dtype)
        bindings = {"x": x, "weight": weight, "out": torch.empty_like(x)}
        expected = (
            x.float()
            * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
            * weight.float()
        ).to(x.dtype)
    else:
        x = torch.randn(32, 512, device="cuda:0").to(torch.float8_e4m3fn)
        w = torch.randn(128, 512, device="cuda:0").to(torch.float8_e4m3fn)
        xs = torch.rand(32, 4, device="cuda:0")
        ws = torch.rand(1, 4, device="cuda:0")
        bindings = {
            "x": x,
            "w": w,
            "x_scale": xs,
            "w_scale": ws,
            "out": torch.empty(32, 128, device="cuda:0", dtype=torch.bfloat16),
        }
        expected = (
            (x.float() * xs.repeat_interleave(128, 1))
            @ (w.float() * ws.repeat_interleave(128, 0).repeat_interleave(128, 1)).T
        ).to(torch.bfloat16)

    def forbidden(*args, **kwargs):
        raise AssertionError("An installed no-compile plan attempted a source build")

    monkeypatch.setattr(aiter.backends.hip, "compile_rmsnorm", forbidden)
    monkeypatch.setattr(aiter.backends.ck, "compile_ck", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    if backend == "hip":
        plan = runtime.prepare_rmsnorm(x, weight, bindings["out"], backend=backend)
    else:
        plan = runtime.prepare_gemm(x, w, xs, ws, bindings["out"], backend=backend)
    assert plan.explain()["artifact_digest"] == expected_digest
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.execute(bindings)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(bindings["out"], expected, rtol=0.025, atol=0.08)

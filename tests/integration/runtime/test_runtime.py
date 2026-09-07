# SPDX-License-Identifier: MIT
"""Behavioral tests for preparation, borrowed buffers and repeated GPU execution."""

import pytest


def _runtime():
    import torch

    from aiter.runtime import Runtime

    assert torch.cuda.is_available() and torch.version.hip, "A ROCm GPU is required."
    return torch, Runtime(device=0)


def _rms_inputs(torch, dtype_name="bfloat16", rows=32, columns=1024):
    torch.manual_seed(23)
    dtype = getattr(torch, dtype_name)
    return {
        "x": torch.randn(rows, columns, dtype=dtype, device="cuda:0"),
        "weight": torch.randn(columns, dtype=dtype, device="cuda:0"),
        "out": torch.full((rows, columns), 3, dtype=dtype, device="cuda:0"),
    }


def _rms_reference(torch, bindings, epsilon=1e-6):
    x = bindings["x"].float()
    return (
        x
        * torch.rsqrt(x.square().mean(-1, keepdim=True) + epsilon)
        * bindings["weight"].float()
    ).to(bindings["out"].dtype)


def _rms_plan(runtime, bindings, backend):
    return runtime.prepare_rmsnorm(
        bindings["x"], bindings["weight"], bindings["out"], backend=backend
    )


def _gemm_inputs(torch):
    torch.manual_seed(29)
    return {
        "x": torch.randn(32, 1024, device="cuda:0").to(torch.float8_e4m3fn),
        "w": torch.randn(128, 1024, device="cuda:0").to(torch.float8_e4m3fn),
        "x_scale": torch.rand(32, 8, device="cuda:0", dtype=torch.float32),
        "w_scale": torch.rand(1, 8, device="cuda:0", dtype=torch.float32),
        "out": torch.full((32, 128), 3, device="cuda:0", dtype=torch.bfloat16),
    }


def _gemm_plan(runtime, bindings, backend):
    return runtime.prepare_gemm(
        bindings["x"],
        bindings["w"],
        bindings["x_scale"],
        bindings["w_scale"],
        bindings["out"],
        backend=backend,
    )


def _gemm_reference(torch, bindings):
    x = bindings["x"].float() * bindings["x_scale"].repeat_interleave(128, dim=1)
    scale = (
        bindings["w_scale"].repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    )
    w = bindings["w"].float() * scale
    return (x @ w.T).to(bindings["out"].dtype)


@pytest.mark.parametrize("backend", ["hip", "triton"])
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_prepared_rmsnorm_preserves_buffers_and_reuses_code(backend, dtype_name):
    torch, runtime = _runtime()
    bindings = _rms_inputs(torch, dtype_name)
    snapshots = {key: value.clone() for key, value in bindings.items()}
    plan = _rms_plan(runtime, bindings, backend)
    for key in bindings:
        torch.testing.assert_close(bindings[key], snapshots[key], rtol=0, atol=0)
    for _ in range(3):
        assert plan.execute(bindings) is bindings["out"]
    torch.testing.assert_close(
        bindings["out"], _rms_reference(torch, bindings), rtol=0.02, atol=0.02
    )
    detail = plan.explain()
    assert detail["provider"] == backend
    assert detail["target"].startswith("gfx")
    assert len(detail["artifact_digest"]) == 64
    assert len(detail["request_id"]) == 64
    assert detail["kernel"] and detail["capture_safe"]
    assert detail["workspace_bytes"] == 0


@pytest.mark.parametrize("backend", ["triton", "gluon", "ck"])
def test_prepared_fp8_gemm_matches_scaled_reference(backend):
    torch, runtime = _runtime()
    bindings = _gemm_inputs(torch)
    expected = _gemm_reference(torch, bindings)
    before = bindings["out"].clone()
    plan = _gemm_plan(runtime, bindings, backend)
    torch.testing.assert_close(bindings["out"], before, rtol=0, atol=0)
    assert plan.execute(bindings) is bindings["out"]
    torch.testing.assert_close(bindings["out"], expected, rtol=0.02, atol=0.08)


@pytest.mark.parametrize(
    "operation,backend",
    [
        ("rms", "hip"),
        ("rms", "triton"),
        ("gemm", "triton"),
        ("gemm", "gluon"),
        ("gemm", "ck"),
    ],
)
def test_capture_replay_uses_prepared_code(operation, backend, monkeypatch):
    torch, runtime = _runtime()
    if operation == "rms":
        bindings = _rms_inputs(torch)
        plan = _rms_plan(runtime, bindings, backend)
        reference = _rms_reference
    else:
        bindings = _gemm_inputs(torch)
        plan = _gemm_plan(runtime, bindings, backend)
        reference = _gemm_reference
    plan.execute(bindings)
    torch.cuda.synchronize()

    def forbidden(*args, **kwargs):
        raise AssertionError("Prepared execution requested compilation or allocation")

    import triton.runtime.jit

    monkeypatch.setattr(triton.runtime.jit.JITFunction, "run", forbidden)
    monkeypatch.setattr(torch, "empty", forbidden)
    monkeypatch.setattr(torch, "empty_like", forbidden)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.execute(bindings)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    monkeypatch.undo()
    expected = reference(torch, bindings)
    torch.testing.assert_close(bindings["out"], expected, rtol=0.02, atol=0.08)


@pytest.mark.parametrize("backend", ["hip", "triton"])
def test_one_plan_runs_on_two_independent_streams(backend):
    torch, runtime = _runtime()
    first = _rms_inputs(torch)
    second = _rms_inputs(torch)
    second["x"].mul_(2)
    second["weight"].mul_(0.5)
    plan = _rms_plan(runtime, first, backend)
    streams = [torch.cuda.Stream(device=0), torch.cuda.Stream(device=0)]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    for _ in range(4):
        plan.execute(first, stream=streams[0])
        plan.execute(second, stream=streams[1])
    for stream in streams:
        stream.synchronize()
    for bindings in (first, second):
        torch.testing.assert_close(
            bindings["out"], _rms_reference(torch, bindings), rtol=0.02, atol=0.02
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "shape",
        "dtype",
        "stride",
        "gradient",
        "alias",
        "missing",
        "extra",
        "cpu",
        "unaligned",
    ],
)
def test_invalid_bindings_fail_before_output_changes(mutation):
    from aiter.api import ValidationError

    torch, runtime = _runtime()
    bindings = _rms_inputs(torch)
    plan = _rms_plan(runtime, bindings, "triton")
    original_out = bindings["out"]
    before = original_out.clone()
    changed = dict(bindings)
    if mutation == "shape":
        changed["x"] = bindings["x"][:16]
    elif mutation == "dtype":
        changed["x"] = bindings["x"].float()
    elif mutation == "stride":
        changed["x"] = bindings["x"].T.contiguous().T
    elif mutation == "gradient":
        changed["x"] = bindings["x"].clone().requires_grad_(True)
    elif mutation == "alias":
        changed["out"] = bindings["x"]
    elif mutation == "missing":
        changed.pop("weight")
    elif mutation == "extra":
        changed["unused"] = bindings["x"]
    elif mutation == "cpu":
        changed["x"] = bindings["x"].cpu()
    elif mutation == "unaligned":
        storage = torch.empty(
            bindings["x"].numel() + 1, dtype=bindings["x"].dtype, device="cuda:0"
        )
        changed["x"] = storage[1:].reshape(bindings["x"].shape)
    with pytest.raises(ValidationError):
        plan.execute(changed)
    torch.testing.assert_close(original_out, before, rtol=0, atol=0)


def test_cross_device_binding_and_stream_are_rejected():
    from aiter.api import ValidationError

    torch, runtime = _runtime()
    assert torch.cuda.device_count() >= 2, "This group requires two visible GPUs."
    bindings = _rms_inputs(torch)
    plan = _rms_plan(runtime, bindings, "triton")
    changed = dict(bindings, x=bindings["x"].to("cuda:1"))
    with pytest.raises(ValidationError):
        plan.execute(changed)
    with pytest.raises(ValidationError):
        plan.execute(bindings, stream=torch.cuda.Stream(device=1))


@pytest.mark.parametrize("backend", ["hip", "triton"])
@pytest.mark.parametrize("rows,columns", [(1, 4), (7, 409), (17, 4096)])
def test_prepared_rmsnorm_fp32_handles_small_and_masked_rows(backend, rows, columns):
    torch, runtime = _runtime()
    bindings = _rms_inputs(torch, "float32", rows, columns)
    plan = _rms_plan(runtime, bindings, backend)
    plan.execute(bindings)
    torch.testing.assert_close(
        bindings["out"], _rms_reference(torch, bindings), rtol=2e-5, atol=2e-5
    )


@pytest.mark.parametrize("backend", ["triton", "gluon"])
def test_prepared_fp8_gemm_handles_all_dimension_tails(backend):
    torch, runtime = _runtime()
    torch.manual_seed(31)
    m, n, k = 17, 129, 192
    bindings = {
        "x": torch.randn(m, k, device="cuda:0").to(torch.float8_e4m3fn),
        "w": torch.randn(n, k, device="cuda:0").to(torch.float8_e4m3fn),
        "x_scale": torch.rand(m, 2, device="cuda:0", dtype=torch.float32),
        "w_scale": torch.rand(2, 2, device="cuda:0", dtype=torch.float32),
        "out": torch.empty(m, n, device="cuda:0", dtype=torch.bfloat16),
    }
    scaled_x = (
        bindings["x"].float() * bindings["x_scale"].repeat_interleave(128, dim=1)[:, :k]
    )
    scale = (
        bindings["w_scale"]
        .repeat_interleave(128, dim=0)
        .repeat_interleave(128, dim=1)[:n, :k]
    )
    expected = (scaled_x @ (bindings["w"].float() * scale).T).to(torch.bfloat16)
    plan = _gemm_plan(runtime, bindings, backend)
    plan.execute(bindings)
    torch.testing.assert_close(bindings["out"], expected, rtol=0.02, atol=0.08)


def test_no_compile_policy_rejects_empty_native_cache(tmp_path, monkeypatch):
    import aiter.backends.native.artifacts
    from aiter.runtime import ExecutionPolicy, Runtime, UnsupportedOperation

    torch, _ = _runtime()
    bindings = _rms_inputs(torch)
    # Isolate the missing-artifact case even when this suite runs against a
    # wheel that correctly supplies a native SDK. Its positive case is in test_wheel.
    monkeypatch.setattr(
        aiter.backends.native.artifacts, "bundled_library", lambda filename: None
    )
    monkeypatch.delenv("AITER_RMSNORM_LIBRARY", raising=False)
    monkeypatch.setenv("AITER_JIT_DIR", str(tmp_path))
    runtime = Runtime(device=0, policy=ExecutionPolicy(allow_compile=False))
    with pytest.raises(UnsupportedOperation):
        _rms_plan(runtime, bindings, "hip")
    assert not list(tmp_path.iterdir()), "No-compile preparation created cache files."


def test_independent_operation_and_backend_execute_without_runtime_changes(monkeypatch):
    from dataclasses import dataclass

    from aiter.api import RMSNorm
    from aiter.backends.hip import HipBackend
    from aiter.runtime import ExecutionPolicy, Runtime, UnsupportedOperation
    from aiter.runtime.plan import tensor_spec
    from aiter.runtime.provider import Support
    from aiter.tuning import DispatchManifest, Selection

    @dataclass(frozen=True)
    class ExternalRMS:
        description: RMSNorm

        def inputs(self):
            return self.description.inputs()

        def outputs(self):
            return self.description.outputs()

        def to_dict(self):
            return {**self.description.to_dict(), "extension": "independent-example"}

        def fingerprint(self):
            from aiter._validation import canonical_digest

            return canonical_digest(self.to_dict())

    class ExternalBackend:
        name = "laboratory.hip"

        def supports(self, operation, target):
            if not isinstance(operation, ExternalRMS):
                return Support(False, "expected ExternalRMS")
            return HipBackend().supports(operation.description, target)

        def prepare(self, operation, bindings, target, policy):
            return HipBackend().prepare(operation.description, bindings, target, policy)

    torch, ordinary = _runtime()
    bindings = _rms_inputs(torch)
    operation = ExternalRMS(
        RMSNorm(tensor_spec(bindings["x"]), tensor_spec(bindings["weight"]))
    )
    runtime = Runtime(backends=(ExternalBackend(),))
    assert tuple(runtime.capabilities(operation)) == ("laboratory.hip",)
    assert "laboratory.hip" not in ordinary.capabilities(operation)
    plan = runtime.prepare(operation, bindings)
    assert plan.explain()["provider"] == "laboratory.hip"
    manifest = DispatchManifest(
        (
            Selection(
                operation.fingerprint(),
                runtime.target,
                "laboratory.hip",
                plan.explain()["artifact_digest"],
                "e" * 64,
            ),
        ),
        runtime.environment_digest,
        "extension-test.v1",
    )
    pinned = Runtime(backends=(ExternalBackend(),), manifest=manifest)
    plan = pinned.prepare(operation, bindings)
    with pytest.raises(UnsupportedOperation):
        ordinary.prepare(operation, bindings)

    def forbidden(*args, **kwargs):
        raise AssertionError("execution must retain its prepared extension")

    expected = _rms_reference(torch, bindings)
    torch.cuda.synchronize()
    monkeypatch.setattr(ExternalBackend, "prepare", forbidden)
    monkeypatch.setattr(ExternalBackend, "supports", forbidden)
    monkeypatch.setattr(torch, "empty", forbidden)
    monkeypatch.setattr(torch, "empty_like", forbidden)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.execute(bindings)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(bindings["out"], expected, rtol=0.02, atol=0.02)
    with pytest.raises(ValueError, match="unavailable backends"):
        Runtime(backends=(ExternalBackend(),), policy=ExecutionPolicy())


@pytest.mark.parametrize("overlap", ["strided-input", "broadcast-output"])
def test_generic_operation_binding_guards_cover_storage_spans(overlap):
    from aiter.api import ValidationError
    from aiter.runtime.plan import tensor_spec, validate_bindings

    torch, _ = _runtime()
    storage = torch.empty(96, device="cuda", dtype=torch.float32)
    if overlap == "strided-input":
        bindings = {
            "x": storage[:64:2].reshape(4, 8),
            "out": storage[48:80].reshape(4, 8),
        }
    else:
        bindings = {"x": storage[:32].reshape(4, 8), "out": storage[64:72].expand(4, 8)}
    specs = tuple((name, tensor_spec(tensor)) for name, tensor in bindings.items())
    with pytest.raises(ValidationError, match="overlap"):
        validate_bindings(bindings, specs, 0, torch)


@pytest.mark.parametrize("explicit_stream", [False, True])
def test_provider_capture_decline_is_checked_on_the_launch_stream(explicit_stream):
    from dataclasses import replace

    from aiter.api import ValidationError

    torch, runtime = _runtime()
    bindings = _rms_inputs(torch)
    prepared = _rms_plan(runtime, bindings, "hip")
    calls = []
    kernel = replace(
        prepared._kernel,
        capture_safe=False,
        launch=lambda *args: calls.append("launched"),
    )
    plan = replace(prepared, _kernel=kernel)
    capturing = torch.cuda.Stream()
    ordinary = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph, stream=capturing):
        if explicit_stream:
            with torch.cuda.stream(ordinary), pytest.raises(
                ValidationError, match="capture"
            ):
                plan.execute(bindings, stream=capturing)
        else:
            with pytest.raises(ValidationError, match="capture"):
                plan.execute(bindings)
    assert calls == []
    plan.execute(bindings, stream=ordinary)
    assert calls == ["launched"]

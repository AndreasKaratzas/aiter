# SPDX-License-Identifier: MIT
"""Pair prepared and legacy native RMSNorm on one GPU; this is not model speed."""

import hashlib
import random
from pathlib import Path

from benchmarks.common.comparison import compare_pairs


def measure(rows, columns, dtype_name):
    """Return numerical and paired timing evidence for one exact RMSNorm shape."""
    import torch

    from aiter.ops.rmsnorm import rms_norm_opus
    from aiter.runtime import Runtime

    assert torch.version.hip and torch.cuda.is_available(), "ROCm GPU required"
    torch.cuda.set_device(0)
    torch.manual_seed(101)
    dtype = getattr(torch, dtype_name)
    x = torch.randn(rows, columns, device="cuda", dtype=dtype)
    weight = torch.randn(columns, device="cuda", dtype=dtype)
    baseline_out, candidate_out = torch.empty_like(x), torch.empty_like(x)
    reference = (
        x.float()
        * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        * weight.float()
    ).to(dtype)
    runtime = Runtime()
    plan = runtime.prepare_rmsnorm(x, weight, candidate_out, backend="hip")
    bindings = {"x": x, "weight": weight, "out": candidate_out}
    callbacks = {
        "baseline": lambda: rms_norm_opus(baseline_out, x, weight, 1e-6),
        "candidate": lambda: plan.execute(bindings),
    }
    for callback in callbacks.values():
        callback()
    from aiter.jit.core import get_user_jit_dir

    baseline_library = Path(get_user_jit_dir()) / "module_rmsnorm.so"
    baseline_digest = hashlib.sha256(baseline_library.read_bytes()).hexdigest()
    for out in (baseline_out, candidate_out):
        torch.testing.assert_close(out, reference, rtol=0.02, atol=0.02)

    # Repeat enough work per graph to avoid timing isolated microsecond launches.
    repeats = 256
    graphs = {}
    torch.cuda.synchronize()
    for name, callback in callbacks.items():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(repeats):
                callback()
        graphs[name] = graph
    for _ in range(5):
        for graph in graphs.values():
            graph.replay()
    torch.cuda.synchronize()
    generator = random.Random(103)
    samples = {name: [] for name in graphs}
    orders = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    for _ in range(21):
        order = list(graphs)
        generator.shuffle(order)
        orders.append(order)
        for name in order:
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1_000_000 / repeats)
    result = compare_pairs(samples["baseline"], samples["candidate"])
    # Verify again after measurement rather than assuming timed code was correct.
    for out in (baseline_out, candidate_out):
        torch.testing.assert_close(out, reference, rtol=0.02, atol=0.02)
    module = __import__(rms_norm_opus.__module__, fromlist=["__file__"])
    assert hashlib.sha256(baseline_library.read_bytes()).hexdigest() == baseline_digest
    evidence = {
        "schema_version": 1,
        "scope": "prepared-versus-legacy-native-rmsnorm-graph-overhead",
        "protocol": "interleaved-21pairs-256calls-events-v1",
        "seed": 101,
        "order_seed": 103,
        "shape": [rows, columns],
        "dtype": dtype_name,
        "environment_digest": runtime.environment_digest,
        "baseline": {
            "entrypoint": "aiter.ops.rmsnorm.rms_norm_opus",
            "library_path": str(baseline_library),
            "artifact_digest": baseline_digest,
            "source_sha256": hashlib.sha256(
                Path(module.__file__).read_bytes()
            ).hexdigest(),
        },
        "candidate": plan.explain(),
        "correctness": {"passed": True, "rtol": 0.02, "atol": 0.02},
        "orders": orders,
        "samples_ns": samples,
        "comparison": result,
    }
    return evidence

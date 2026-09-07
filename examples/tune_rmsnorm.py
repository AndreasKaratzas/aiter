# SPDX-License-Identifier: MIT
"""Measure prepared RMSNorm implementations and use the resulting selection."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--columns", type=int, default=4096)
    args = parser.parse_args()
    # A fresh directory retains all raw evidence without replacing a prior run.
    args.output.mkdir(parents=True, exist_ok=False)

    import torch

    from aiter.runtime import Runtime
    from aiter.tuning import Trial, promote

    torch.cuda.set_device(args.device)
    torch.manual_seed(67)
    runtime = Runtime(device=args.device)
    x = torch.randn(args.rows, args.columns, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(args.columns, device="cuda", dtype=x.dtype)
    out = torch.empty_like(x)
    bindings = {"x": x, "weight": weight, "out": out}
    reference = (
        x.float()
        * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        * weight.float()
    ).to(out.dtype)
    protocol = "rmsnorm-bf16-graph100-events7-v1"
    trials = []
    for backend in ("hip", "triton"):
        plan = runtime.prepare_rmsnorm(x, weight, out, backend=backend)
        plan.execute(bindings)
        torch.testing.assert_close(out, reference, rtol=0.02, atol=0.02)
        max_error = (out.float() - reference.float()).abs().max().item()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(100):
                plan.execute(bindings)
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        samples = []
        for _ in range(7):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(max(1, round(start.elapsed_time(end) * 1_000_000 / 100)))
        detail = plan.explain()
        evidence = {
            "schema_version": 1,
            "protocol": protocol,
            "plan": detail,
            "seed": 67,
            "correctness": {
                "passed": True,
                "rtol": 0.02,
                "atol": 0.02,
                "max_error": max_error,
            },
            "samples_ns": samples,
        }
        content = (json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode()
        (args.output / f"{backend}.json").write_bytes(content)
        trials.append(
            Trial(
                detail["request_id"],
                detail["target"],
                backend,
                detail["artifact_digest"],
                runtime.environment_digest,
                protocol,
                tuple(samples),
                True,
                hashlib.sha256(content).hexdigest(),
            )
        )
    (args.output / "trials.json").write_text(
        json.dumps([asdict(t) for t in trials], indent=2) + "\n"
    )
    manifest = promote(
        trials, environment_digest=runtime.environment_digest, protocol=protocol
    )
    manifest.write(args.output / "dispatch.json")
    selected = Runtime(device=args.device, manifest=manifest).prepare_rmsnorm(
        x, weight, out
    )
    selected.execute(bindings)
    torch.testing.assert_close(out, reference, rtol=0.02, atol=0.02)
    print(json.dumps(selected.explain(), indent=2))


if __name__ == "__main__":
    main()

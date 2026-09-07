# SPDX-License-Identifier: MIT
"""Prepare once, inspect the selected code, then reuse caller-owned buffers."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("hip", "triton"), default="hip")
    parser.add_argument(
        "--no-compile", action="store_true", help="require an existing native library"
    )
    args = parser.parse_args()
    import torch

    from aiter.runtime import ExecutionPolicy, Runtime

    torch.manual_seed(11)
    runtime = Runtime(
        device=0, policy=ExecutionPolicy(allow_compile=not args.no_compile)
    )
    x = torch.randn(17, 1024, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1024, device=x.device, dtype=x.dtype)
    out = torch.empty_like(x)
    plan = runtime.prepare_rmsnorm(x, weight, out, backend=args.backend)
    bindings = {"x": x, "weight": weight, "out": out}
    for factor in (1.0, -0.5, 2.0):
        x.mul_(factor)
        plan.execute(bindings)
        expected = x.float() * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + 1e-6
        )
        torch.testing.assert_close(
            out, (expected * weight.float()).to(x.dtype), atol=0.02, rtol=0.01
        )
    print(json.dumps(plan.explain(), indent=2))
    print("Prepared RMSNorm passed three input updates.")


if __name__ == "__main__":
    main()

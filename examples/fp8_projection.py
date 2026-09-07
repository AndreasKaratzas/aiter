# SPDX-License-Identifier: MIT
"""Quantize activations and project using ordinary FP8 block scale buffers."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("ck", "triton", "gluon"), default="triton"
    )
    args = parser.parse_args()
    import torch

    from aiter.runtime import Runtime

    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx950":
        raise RuntimeError("This example uses the gfx950 E4M3FN encoding.")
    torch.manual_seed(17)
    runtime = Runtime(device=0)
    m, n, k = 16, 128, 256
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    packed = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    xs = torch.empty(m, k // 128, device="cuda")
    # Stand-in for checkpoint weights and one scale per 128x128 weight block.
    w = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
    ws = torch.rand(n // 128, k // 128, device="cuda") + 0.25
    out = torch.empty(m, n, device="cuda", dtype=x.dtype)
    quant = runtime.prepare_quantize(x, packed, xs, backend="triton")
    gemm = runtime.prepare_gemm(packed, w, xs, ws, out, backend=args.backend)
    quant.execute({"x": x, "out": packed, "scales": xs})
    gemm.execute({"x": packed, "w": w, "x_scale": xs, "w_scale": ws, "out": out})
    expected_scales = (
        x.float().reshape(m, k // 128, 128).abs().amax(-1).clamp(min=1e-10) / 448
    )
    torch.testing.assert_close(xs, expected_scales, atol=1e-12, rtol=2e-6)
    restored_x = packed.float() * xs.repeat_interleave(128, dim=1)
    restored_w = w.float() * ws.repeat_interleave(128, dim=0).repeat_interleave(
        128, dim=1
    )
    torch.testing.assert_close(
        out, (restored_x @ restored_w.T).to(out.dtype), atol=0.08, rtol=0.02
    )
    print(
        f"FP8 quantization → {args.backend} projection passed; output {tuple(out.shape)}."
    )


if __name__ == "__main__":
    main()

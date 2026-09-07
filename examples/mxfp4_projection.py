# SPDX-License-Identifier: MIT
"""Pack ordinary MXFP4 operands and verify GEMM by independent decoding."""


def main():
    import torch

    from aiter.runtime import Runtime

    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx950":
        raise RuntimeError("Prepared ordinary MXFP4 requires gfx950.")
    torch.manual_seed(19)
    runtime = Runtime(device=0)
    m, n, k = 17, 35, 96
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=x.dtype)
    xp = torch.empty(m, k // 2, device="cuda", dtype=torch.uint8)
    wp = torch.empty(n, k // 2, device="cuda", dtype=torch.uint8)
    xs = torch.empty(m, k // 32, device="cuda", dtype=torch.uint8)
    ws = torch.empty(n, k // 32, device="cuda", dtype=torch.uint8)
    out = torch.empty(m, n, device="cuda", dtype=x.dtype)
    qx = runtime.prepare_mxfp4_quantize(x, xp, xs, backend="triton")
    qw = runtime.prepare_mxfp4_quantize(w, wp, ws, backend="triton")
    gemm = runtime.prepare_mxfp4_gemm(xp, wp, xs, ws, out, backend="triton")
    qx.execute({"x": x, "out": xp, "scales": xs})
    qw.execute({"x": w, "out": wp, "scales": ws})
    gemm.execute({"x": xp, "w": wp, "x_scale": xs, "w_scale": ws, "out": out})

    def decode(packed, scales):
        # Even element: low nibble; odd element: high nibble. Bit3 is the sign.
        magnitudes = torch.tensor((0, 0.5, 1, 1.5, 2, 3, 4, 6), device=packed.device)
        codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
        values = magnitudes[(codes & 7).long()] * torch.where(codes & 8 != 0, -1.0, 1.0)
        return values * torch.exp2(scales.float() - 127).repeat_interleave(32, dim=1)

    expected = (decode(xp, xs) @ decode(wp, ws).T).to(out.dtype)
    torch.testing.assert_close(out, expected, atol=0.08, rtol=0.02)
    print(f"Ordinary MXFP4 quantization → GEMM passed; {tuple(out.shape)} output.")


if __name__ == "__main__":
    main()

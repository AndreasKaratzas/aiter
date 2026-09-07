# SPDX-License-Identifier: MIT
"""Call the public operator and compare every output element with Torch."""


def main():
    import torch

    import aiter

    torch.manual_seed(7)
    x = torch.randn(17, 1024, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1024, device=x.device, dtype=x.dtype)
    actual = aiter.rms_norm(x, weight, 1e-6)
    reference = x.float() * torch.rsqrt(
        x.float().square().mean(-1, keepdim=True) + 1e-6
    )
    reference = (reference * weight.float()).to(x.dtype)
    torch.testing.assert_close(actual, reference, atol=0.02, rtol=0.01)
    print(f"Public RMSNorm passed: {tuple(x.shape)}, {x.dtype}, {aiter.__file__}")


if __name__ == "__main__":
    main()

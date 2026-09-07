# SPDX-License-Identifier: MIT
"""PyTorch tensors execute the public normalization entry point."""

import pytest

from frameworks.common.runtime import rmsnorm_reference, rocm

pytestmark = [
    pytest.mark.gpu(),
    pytest.mark.framework("torch"),
    pytest.mark.requires_arch("gfx942", "gfx950"),
]


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_pytorch_rmsnorm(dtype_name):
    torch = rocm()
    import aiter

    torch.manual_seed(13)
    x = torch.randn(16, 1024, device="cuda", dtype=getattr(torch, dtype_name))
    weight = torch.randn(1024, device="cuda", dtype=x.dtype)
    result = aiter.rms_norm(x, weight, 1e-6)
    torch.testing.assert_close(
        result, rmsnorm_reference(x, weight, 1e-6), rtol=0.02, atol=0.02
    )

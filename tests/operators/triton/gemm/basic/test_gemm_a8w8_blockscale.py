# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import importlib
import sys

import pytest
import torch

from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
    gemm_a8w8_blockscale,
    gemm_a8w8_blockscale_preshuffle,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.types import str_to_torch_dtype
from aiter.tuning.search.workloads.gemm.basic.gemm_a8w8_blockscale import (
    block_shape,
    e4m3_type,
    e5m2_type,
    generate_gemm_a8w8_blockscale_inputs,
    run_torch,
)

DEVICE_ARCH = arch_info.get_arch()


def run_triton(x, weight, x_scale, w_scale, dtype=torch.bfloat16, y=None, impl=None):
    return impl(x, weight, x_scale, w_scale, dtype, y)


def get_x_vals():
    x_vals = [(1024 * v, 1024 * v, 1024 * v) for v in (1, 2, 4, 5, 8)]
    # GPT-OSS-120B attention projections
    x_vals += [(v, 106496, 16384) for v in (256, 4096)]  # LL3 405B FC1
    x_vals += [(v, 9216, 7168) for v in (128, 192, 4096, 8000)]
    x_vals += [(v, 7168, 4608) for v in (128, 192, 4096, 8000)]
    x_vals += [(v, 8192, 512) for v in (128, 192, 4096, 8000)]
    # Small-K shapes that exercise the gluon wind-down's num_k_iter guards
    # (BLOCK_SIZE_K=128; K in {128,192,256,320} -> num_k_iter in {1,2,2,3}).
    # K<BLOCK_SIZE_K isn't supported by the gluon wrapper (GROUP_K assert).
    x_vals += [(512, 512, K) for K in (128, 192, 256, 320)]
    x_vals += [(v, 8192, 1024) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 4096, 8192) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 4096, 4096) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 4096, 2048) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 1536, 4096) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 32768, 1024) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 8192, 1536) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 7168, 4096) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 1536, 7168) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 7168, 768) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 2048, 7168) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 16384, 1536) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 65536, 1536) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 7168, 16384) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 6144, 7168) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 7168, 3072) for v in (1, 32, 64, 128, 256, 1024)]
    return x_vals


@pytest.mark.parametrize(
    "dtype, M, N, K, layout, output",
    [
        (dtype, *shape, layout, output)
        for output in [True]
        for dtype in ["bf16"]
        for layout in ["TN"]
        for shape in get_x_vals()
    ],
)
@pytest.mark.parametrize("backend", ["gluon", "triton"])
@pytest.mark.parametrize("shuffle", [True, False])
def test_gemm(dtype, M, N, K, layout, output, backend, shuffle):
    torch.cuda.empty_cache()  # Helps avoid hangs in large tests
    torch.cuda.synchronize()

    block_shape_n, block_shape_k = block_shape

    if backend == "gluon":
        if shuffle:
            if DEVICE_ARCH not in ("gfx1250"):
                pytest.skip("Gluon + shuffle implementation requires gfx1250.")
        elif DEVICE_ARCH not in ("gfx950", "gfx1250"):
            pytest.skip("Gluon implementation requires gfx950 or gfx1250.")

    if shuffle and (N % 16 > 0 or K % 32 > 0):
        pytest.skip(
            "N has to be multiple of 16 and K has to be multiple of 32 for preshuffle cases"
        )

    if backend not in ("gluon",) and K < 512:
        pytest.skip("Small-K shapes exercise gluon-only paths.")

    dtype = str_to_torch_dtype[dtype]
    x, weight, weight_triton, x_scale, x_scale_shuffled, w_scale, y = (
        generate_gemm_a8w8_blockscale_inputs(
            M,
            N,
            K,
            block_shape_n,
            block_shape_k,
            dtype=dtype,
            layout=layout,
            output=output,
            shuffle=shuffle,
        )
    )

    a = run_torch(x, weight, x_scale, w_scale, dtype)

    if shuffle:

        def impl(x, w, xs, ws, dt, y):
            return gemm_a8w8_blockscale_preshuffle(x, w, xs, ws, dt, y, backend=backend)

    else:

        def impl(x, w, xs, ws, dt, y):
            return gemm_a8w8_blockscale(x, w, xs, ws, dt, y, backend=backend)

    b = run_triton(x, weight_triton, x_scale_shuffled, w_scale, dtype, y, impl)

    torch.testing.assert_close(a, b, atol=0.01, rtol=1e-2)


def test_legacy_gluon_import_path_warns():
    """The pre-move path still resolves here, but tells callers to move on."""
    legacy = "aiter.ops.triton.gluon.gemm_a8w8_blockscale"
    sys.modules.pop(legacy, None)

    with pytest.warns(DeprecationWarning, match="has moved to"):
        mod = importlib.import_module(legacy)

    assert (
        mod.gemm_a8w8_blockscale.__module__
        == "aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale"
    )


__all__ = [
    "block_shape",
    "e4m3_type",
    "e5m2_type",
    "generate_gemm_a8w8_blockscale_inputs",
    "get_x_vals",
    "run_torch",
    "run_triton",
    "test_gemm",
    "test_legacy_gluon_import_path_warns",
]

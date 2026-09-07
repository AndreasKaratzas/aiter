# SPDX-License-Identifier: MIT
"""The integrated native BLAS bridges execute real selected solutions."""

import pytest
import torch
from common.paths import assert_package_origin

import aiter
from aiter.codegen import BuildContext


@pytest.mark.parametrize("provider", ("hipb", "rocb"))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_blas_solution_discovery_and_execution(provider, dtype):
    assert_package_origin()
    context = BuildContext.load()
    native_file = {"hipb": "hipbsolgemm.cu", "rocb": "rocsolgemm.cu"}[provider]
    assert context.resource("native", "blas", native_file).is_file()
    torch.manual_seed(602)
    a = torch.randn(17, 128, device="cuda", dtype=dtype)
    b = torch.randn(33, 128, device="cuda", dtype=dtype).T
    create = getattr(aiter, f"{provider}_create_extension")
    destroy = getattr(aiter, f"{provider}_destroy_extension")
    discover = getattr(aiter, f"{provider}_findallsols")
    execute = getattr(aiter, f"{provider}_mm")
    create()
    try:
        solutions = discover(a, b)
        assert solutions, f"{provider} returned no solution for a supported small GEMM"
        actual = execute(a, b, int(solutions[0]))
        torch.testing.assert_close(
            actual, (a.double() @ b.double()).to(dtype), rtol=0.02, atol=0.04
        )
    finally:
        destroy()

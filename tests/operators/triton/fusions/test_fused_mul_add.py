import pytest
import torch

from aiter.ops.triton.fusions.fused_mul_add import fused_mul_add
from aiter.tuning.search.workloads.fusions.fused_mul_add import (
    generate_fused_mul_add_inputs,
)


def run_torch(x, a, b):
    return (a * x.to(torch.float32) + b).to(x.dtype)


@pytest.mark.parametrize(
    "shape", [(1,), (8,), (500,), (10000,), (32, 7168), (16, 50, 4186)]
)
@pytest.mark.parametrize(
    "a_type_is_scalar",
    [(float, True), (int, True), (torch.Tensor, True), (torch.Tensor, False)],
)
@pytest.mark.parametrize(
    "b_type_is_scalar",
    [(float, True), (int, True), (torch.Tensor, True), (torch.Tensor, False)],
)
@pytest.mark.parametrize("output", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_mul_add(shape, a_type_is_scalar, b_type_is_scalar, output: bool, dtype):

    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    x, a, b = generate_fused_mul_add_inputs(
        shape, a_type_is_scalar, b_type_is_scalar, dtype
    )

    x_torch = run_torch(x, a, b).clone()
    if output:
        x_triton = torch.empty_like(x)
        fused_mul_add(x, a, b, x_triton)
    else:
        x_triton = x
        fused_mul_add(x_triton, a, b)

    torch.testing.assert_close(x_torch, x_triton)


__all__ = ["generate_fused_mul_add_inputs", "run_torch", "test_mul_add"]

# SPDX-License-Identifier: MIT
"""The legacy assembly bridge keeps split-K accumulation and buffer use correct."""

import pytest
import torch
from common.paths import assert_package_origin

from aiter.ops.gemm_op_a16w16 import _gemm_a16w16_asm, gemm_a16w16_asm
from aiter.ops.shuffle import shuffle_weight


def inputs(shuffled=False, output_dtype=torch.bfloat16):
    assert_package_origin()
    torch.manual_seed(101)
    # Padded rows exercise the input row-stride ABI without allowing overlap.
    a = torch.randn(64, 5136, device="cuda", dtype=torch.bfloat16)[:, :5120]
    b = torch.randn(256, 5120, device="cuda", dtype=torch.bfloat16)
    expected = a.double() @ b.double().T
    if shuffled:
        b = shuffle_weight(b, layout=(16, 16))
    return a, b, torch.full((64, 256), 7, device="cuda", dtype=output_dtype), expected


@pytest.mark.parametrize("shuffled", (False, True))
@pytest.mark.parametrize("output_dtype", (torch.bfloat16, torch.float32))
def test_split_accumulation_rounds_once(shuffled, output_dtype):
    a, b, out, expected = inputs(shuffled, output_dtype)
    for split in (None, 1, 2, 3, 4, 8, 16):
        actual = gemm_a16w16_asm(a, b, out, splitK=split, bpreshuffle=shuffled)
        assert actual is out
        torch.testing.assert_close(out, expected.to(output_dtype), atol=0.01, rtol=0.01)


@pytest.mark.parametrize("output_dtype", (torch.bfloat16, torch.float32))
@pytest.mark.parametrize("bias_dtype", (torch.bfloat16, torch.float32))
def test_bias_is_added_in_fp32(output_dtype, bias_dtype):
    a, b, out, expected = inputs(False, output_dtype)
    bias = torch.randn(256, device="cuda", dtype=bias_dtype) * 3
    gemm_a16w16_asm(a, b, out, bias=bias, splitK=4)
    torch.testing.assert_close(
        out, (expected + bias.double()).to(output_dtype), atol=0.01, rtol=0.01
    )


@pytest.mark.parametrize("split", (0, -1, 17, True, 1.5))
def test_invalid_split_is_rejected_before_output_changes(split):
    a, b, out, _ = inputs()
    before = out.clone()
    with pytest.raises(ValueError, match="splitK"):
        gemm_a16w16_asm(a, b, out, splitK=split)
    torch.testing.assert_close(out, before, atol=0, rtol=0)


def test_native_missing_semaphore_cannot_launch_split_kernel():
    a, b, out, _ = inputs()
    workspace = torch.empty_like(out, dtype=torch.float32)
    empty = torch.empty(0, device="cuda", dtype=torch.uint32)
    with pytest.raises(RuntimeError, match="nonempty GPU tensor|semaphore"):
        _gemm_a16w16_asm(a, b, out, empty, workspace, splitK=4)
    assert torch.all(out == 7)


def test_split_request_is_not_silently_clamped():
    a = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    out = torch.full((4, 64), 7, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="exact shape"):
        gemm_a16w16_asm(a, b, out, splitK=16)
    assert torch.all(out == 7)


@pytest.mark.parametrize("mixed", (False, True))
def test_fp16_inputs_are_rejected(mixed):
    a, b, out, _ = inputs()
    a = a.half()
    if not mixed:
        b = b.half()
    with pytest.raises(ValueError, match="BF16 inputs"):
        gemm_a16w16_asm(a, b, out)
    assert torch.all(out == 7)


def test_historical_cache_cannot_supply_the_workspace_abi(tmp_path):
    import subprocess
    import sys

    from aiter.codegen import BuildContext

    cache = tmp_path / "jit"
    cache.mkdir()
    old = cache / "module_gemm_a16w16_asm.so"
    old.write_bytes(b"historical binary: must not be loaded by the new workspace ABI")
    environment = BuildContext.load().child_environment()
    environment.update(AITER_JIT_DIR=str(cache), AITER_REBUILD="0", MAX_JOBS="2")
    program = """
import torch
from pathlib import Path
from aiter.ops.gemm_op_a16w16 import gemm_a16w16_asm
x = torch.randn(4, 64, device='cuda', dtype=torch.bfloat16)
w = torch.randn(64, 64, device='cuda', dtype=torch.bfloat16)
out = torch.empty(4, 64, device='cuda', dtype=torch.bfloat16)
gemm_a16w16_asm(x, w, out)
torch.testing.assert_close(out, (x.double() @ w.double().T).to(out.dtype), atol=.01, rtol=.01)
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert old.read_bytes().startswith(b"historical binary:")
    assert (cache / "module_gemm_a16w16_asm_workspace.so").is_file()

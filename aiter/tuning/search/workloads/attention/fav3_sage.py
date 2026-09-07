# SPDX-License-Identifier: MIT
"""Shared attention.fav3_sage workload generation and numerical references."""

import numpy as np
import torch

ATOL_fp8 = 3.0e-1


RTOL_fp8 = 2.5e-1


def compare_accuracy(current, reference):
    """Print quick statistics comparing FP8 and SageAttn tensors."""
    current_f = current.float()
    reference_f = reference.float()
    abs_diff = torch.abs(reference_f - current_f)

    print("Output Tensor Stats:")
    print(
        f"  Reference ({tuple(reference_f.shape)}): min={reference_f.min().item():.6f}, max={reference_f.max().item():.6f}, "
        f"mean={reference_f.mean().item():.6f}, std={reference_f.std().item():.6f}"
    )
    print(
        f"  Test      ({tuple(current_f.shape)}): min={current_f.min().item():.6f}, max={current_f.max().item():.6f}, "
        f"mean={current_f.mean().item():.6f}, std={current_f.std().item():.6f}"
    )

    print("Correctness Comparison:")
    print(f"  Mean Absolute Error: {abs_diff.mean().item():.6e}")
    print(f"  Max Absolute Error: {abs_diff.max().item():.6e}")
    print(f"  Std Absolute Error: {abs_diff.std().item():.6e}")
    ref_flat = reference_f.reshape(-1)
    test_flat = current_f.reshape(-1)
    cos_sim = torch.nn.functional.cosine_similarity(
        ref_flat.unsqueeze(0), test_flat.unsqueeze(0)
    )
    print(f"  Cosine Similarity: {cos_sim.item():.8f}")
    # Per-row (per-query) cosine over the head-dim D (last axis). Robust to a few outlier
    # rows: the global flatten cosine is dominated by the largest-magnitude elements, so an
    # aligned kernel with a handful of blown-up rows still reports a low global cosine. The
    # median over rows is the alignment gate we trust.
    rc = torch.nn.functional.cosine_similarity(current_f, reference_f, dim=-1).reshape(
        -1
    )
    print(
        f"  Per-row cosine (over D): mean={rc.mean().item():.6f} "
        f"median={rc.median().item():.6f} p10={rc.quantile(0.10).item():.6f} "
        f"frac>0.99={(rc > 0.99).float().mean().item():.4f}"
    )


def fp8_assert_close(
    tensor_a, tensor_b, atol=ATOL_fp8, rtol=RTOL_fp8, max_diff_percentage=0.5
):
    """Assert tensors are close with tolerance for small percentage of elements"""
    # standard comparison
    abs_diff = torch.abs(tensor_a - tensor_b)
    rel_diff = abs_diff / torch.abs(tensor_b.clamp(min=1e-6))

    # calculate elements that exceed tolerance
    abs_check = abs_diff > atol
    rel_check = rel_diff > rtol
    failed_check = torch.logical_and(abs_check, rel_check)

    # calculate percentage of failed elements
    failed_percentage = failed_check.sum().item() / failed_check.numel() * 100

    # if percentage is small enough, test passes
    if failed_percentage <= max_diff_percentage:
        return True

    # Otherwise, provide diagnostic information
    max_abs_idx = torch.argmax(abs_diff).item()
    max_rel_idx = torch.argmax(rel_diff).item()

    def flat_to_idx(flat_idx, shape):
        return np.unravel_index(flat_idx, shape)

    max_abs_pos = flat_to_idx(max_abs_idx, tensor_a.shape)
    max_rel_pos = flat_to_idx(max_rel_idx, tensor_a.shape)

    max_abs_diff = abs_diff.flatten()[max_abs_idx].item()
    max_rel_diff = rel_diff.flatten()[max_rel_idx].item()

    raise AssertionError(
        f"Tensors not close enough! {failed_percentage:.6f}% elements exceed tolerance.\n"
        f"Greatest absolute difference: {max_abs_diff} at index {max_abs_pos} (up to {atol} allowed)\n"
        f"Greatest relative difference: {max_rel_diff} at index {max_rel_pos} (up to {rtol} allowed)"
    )


def _tensor_from_result(result):
    if isinstance(result, torch.Tensor):
        return result
    if isinstance(result, (list, tuple)) and result:
        return _tensor_from_result(result[0])
    raise TypeError(f"Unsupported result type for comparison: {type(result)}")


def check_attention_outputs(
    current,
    reference,
    fp8=False,
    atol=None,
    rtol=None,
    max_diff_percentage=0.5,
):
    current_tensor = _tensor_from_result(current)
    reference_tensor = _tensor_from_result(reference).to(current_tensor.dtype)

    if fp8:
        fp8_assert_close(
            current_tensor,
            reference_tensor,
            atol=atol or ATOL_fp8,
            rtol=rtol or RTOL_fp8,
            max_diff_percentage=max_diff_percentage,
        )
    else:
        torch.testing.assert_close(
            current_tensor,
            reference_tensor,
            atol=atol or 1e-2,
            rtol=rtol or 1e-2,
        )


def input_helper(
    BATCH,
    HQ,
    HK,
    N_CTX_Q,
    N_CTX_K,
    D_HEAD,
    D_HEAD_V,
    dtype,
    layout,
):
    # Set up tensor shapes based on layout
    if layout == "bhsd":
        q_shape = (BATCH, HQ, N_CTX_Q, D_HEAD)
        k_shape = (BATCH, HK, N_CTX_K, D_HEAD)
        v_shape = (BATCH, HK, N_CTX_K, D_HEAD_V)
    else:  # bshd
        q_shape = (BATCH, N_CTX_Q, HQ, D_HEAD)
        k_shape = (BATCH, N_CTX_K, HK, D_HEAD)
        v_shape = (BATCH, N_CTX_K, HK, D_HEAD_V)

    torch.manual_seed(20)
    q = torch.randn(q_shape, device="cuda", dtype=dtype)
    k = torch.randn(k_shape, device="cuda", dtype=dtype)
    v = torch.randn(v_shape, device="cuda", dtype=dtype)
    q.requires_grad = False
    k.requires_grad = False
    v.requires_grad = False

    return q, k, v

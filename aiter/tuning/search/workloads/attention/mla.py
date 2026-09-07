# SPDX-License-Identifier: MIT
"""Shared attention.mla workload generation and numerical references."""

import torch

from aiter.ops.triton.utils.types import e4m3_dtype


def shuffle_kv_buffer(
    kv_buffer: torch.Tensor,
    kv_lora_rank: int,
):
    """
    Shuffle key and value cache layout for optimized memory access.

        layout: (num_lanes, num_elements_per_thread)
            gfx1250: (16, 8) for BF16 and FP8.
            gfx950: (16, 8) for BF16 and (16, 16) for FP8.

        WMMA/MFMA instruction shape:
            BF16: 16x16x32
            FP8: 16x16x64
    """

    dtype = kv_buffer.dtype
    assert dtype in (torch.bfloat16, e4m3_dtype)

    if dtype == torch.bfloat16:
        layout = (16, 8)
    else:
        # Caution: in gfx1250, the 16-bit and 8-bit layout should both be (16, 8), however, in order to enable ds_load_b128 for 8-bit WMMA,
        # we use (16, 16) here, noted that you must set k_width to 16 in the corresponding DotOperandLayout, the math will be equivalent.
        layout = (16, 16)

    _num_blocks, block_size, num_kv_heads, head_size = kv_buffer.shape

    assert block_size >= 16

    num_lanes, num_elements_per_thread = layout

    def shuffle(kv_lora_or_rope_buffer):
        d = kv_lora_or_rope_buffer.shape[-1]
        kv_lora_or_rope_buffer = kv_lora_or_rope_buffer.view(
            -1,
            num_kv_heads,
            block_size // num_lanes,
            num_lanes,
            d // (2 * num_elements_per_thread),
            2,  # there are 2 groups of threads, t0 ~ t15 and t16 ~ t31
            num_elements_per_thread,
        )
        kv_lora_or_rope_buffer = kv_lora_or_rope_buffer.permute(
            0, 1, 2, 4, 5, 3, 6
        ).contiguous()
        kv_lora_or_rope_buffer = kv_lora_or_rope_buffer.view(
            -1, num_kv_heads, block_size // 16, d * 16
        )
        return kv_lora_or_rope_buffer

    kv_buffer_shuffled = kv_buffer.view(
        -1, block_size, num_kv_heads, head_size
    ).permute(0, 2, 1, 3)
    kv_buffer_shuffled_lora = shuffle(kv_buffer_shuffled[..., :kv_lora_rank])
    kv_buffer_shuffled_rope = shuffle(kv_buffer_shuffled[..., kv_lora_rank:])
    kv_buffer_shuffled_lora = kv_buffer_shuffled_lora.view(
        -1, num_kv_heads, block_size * kv_lora_rank
    )
    kv_buffer_shuffled_rope = kv_buffer_shuffled_rope.view(
        -1, num_kv_heads, block_size * (head_size - kv_lora_rank)
    )
    kv_buffer_shuffled = torch.cat(
        [kv_buffer_shuffled_lora, kv_buffer_shuffled_rope], dim=-1
    ).contiguous()
    kv_buffer_shuffled = kv_buffer_shuffled.view(
        -1, num_kv_heads, block_size, head_size
    )

    return kv_buffer_shuffled

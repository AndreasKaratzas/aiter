# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness + perf sweep for the GDN prefill K5 hidden-state scan.

Backends per shape: FlyDSL opt, Triton ``opt_vk``, and the HIP/C++ K5 kernel
where the shape allows it. The pure-torch fp32 reference is never timed.

Shapes: Qwen3.5-35B (Hv=32) and -397B (Hv=64), TP 1/2/4/8, dense prompts and
varlen batches. ``PREFILL_PARAMS`` / ``PREFILL_TEST_IDS`` hold the full catalog
consumed by ``aiter/tuning/search/recurrent/gdn_k5.py``.

Run the default sweep::

    python tests/operators/flydsl/test_flydsl_linear_attention_prefill.py

Filter to one model / TP / shape family::

    python tests/operators/flydsl/test_flydsl_linear_attention_prefill.py \\
        --model 397b --tp 4 --seqlen 8192 --total-tokens 32768 \\
        --mode varlen --snapshot-dtype bf16 fp32

Reproduce the whole tuner catalog (slow -- the torch reference is O(T*H))::

    python tests/operators/flydsl/test_flydsl_linear_attention_prefill.py \\
        --model 35b 397b --tp 1 2 4 8 --mode dense varlen \\
        --seqlen 1024 2048 4096 8192 16384 32768 65536 \\
        --total-tokens 8192 16384 32768 65536 --snapshot-dtype bf16 fp32
"""

from __future__ import annotations

import argparse
import itertools
import zlib

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.linear_attention_prefill_kernels import (
    chunk_gated_delta_rule_fwd_h_flydsl_opt,
)
from aiter.ops.prefill_batch_metadata import (
    build_gated_delta_rule_prefill_metadata,
)
from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h_opt_vk,
)
from aiter.testing import benchmark, checkAllclose, run_perftest

# HIP/C++ K5 (chunk_gated_delta_rule_fwd_h.cu), JIT-compiled on first call.
try:
    from aiter.ops.chunk_gated_delta_rule_fwd_h import (
        chunk_gated_delta_rule_fwd_h_hip_fn,
    )

    _HAS_HIP_K5 = True
except ImportError:
    chunk_gated_delta_rule_fwd_h_hip_fn = None
    _HAS_HIP_K5 = False

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]


# -- Case catalog (imported by the tuner / block benchmark) --------------


from aiter.tuning.search.workloads.prefill import (
    _DENSE_PROMPT_LENS,
    _K5_TPS,
    _SNAPSHOT_DTYPES,
    _STATE_DTYPES,
    _VARLEN_SEQLENS,
    _VARLEN_TOTAL_T,
    K5_MODELS,
    PrefillArgs,
    _build_context_lens,
)

# -- Helpers -------------------------------------------------------------


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _dtype_size(dtype: torch.dtype) -> int:
    return torch.empty(0, dtype=dtype, device="cpu").element_size()


def _build_cu_seqlens(context_lens, device="cuda"):
    return torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(context_lens), 0).tolist()),
        dtype=torch.int32,
        device=device,
    )


def _case_seed(context_lens, case: PrefillArgs) -> int:
    """Per-case seed, so inputs do not depend on how many cases ran first.

    ``crc32`` rather than the builtin ``hash`` keeps it stable across
    processes regardless of ``PYTHONHASHSEED``.
    """
    return zlib.crc32(f"{case!r}|{list(context_lens)}".encode()) & 0x7FFFFFFF


def _make_inputs(case: PrefillArgs, context_lens):
    """Build the K5 operands as the serving stack hands them over.

    ``k`` is GQA token-major ``[B, T, Hg, K]``; ``w``/``u`` are the head-major
    ``[B, H, T, K/V]`` contiguous buffers K4 produces (the un-permuted
    ``w_orig``/``u_orig`` views go to the torch reference only).
    """
    torch.manual_seed(_case_seed(context_lens, case))
    Hg, H = case.Hg, case.H
    device = "cuda"

    if case.is_varlen:
        cu_seqlens = _build_cu_seqlens(context_lens, device=device)
        T_total = int(cu_seqlens[-1].item())
        B, N = 1, len(context_lens)
    else:
        cu_seqlens = None
        T_total = sum(context_lens)
        B = case.dense_batch
        N = B

    dtype = case.dtype
    k = torch.randn(B, T_total, Hg, case.K, dtype=dtype, device=device) * 0.1
    w_orig = torch.randn(B, T_total, H, case.K, dtype=dtype, device=device) * 0.1
    u_orig = torch.randn(B, T_total, H, case.V, dtype=dtype, device=device) * 0.1

    # g is always 3-D, matching the wrapper/HIP contract, with cumsum along T.
    # Generate head-major first (cumsum on the last dim), then transpose, so
    # both layouts hold identical values.
    if not case.use_g:
        g = None
    else:
        gh = torch.randn(B, H, T_total, dtype=torch.float32, device=device).abs() * -0.5
        gh = gh.cumsum(dim=-1)
        g = gh.contiguous() if case.g_head_major else gh.transpose(1, 2).contiguous()

    w_c = w_orig.permute(0, 2, 1, 3).contiguous()
    u_c = u_orig.permute(0, 2, 1, 3).contiguous()

    # Allocate in f32 first so the reference built off this tensor stays clean,
    # then cast down when a bf16 state is asked for.
    h0 = torch.randn(N, H, case.V, case.K, dtype=torch.float32, device=device) * 0.01
    if case.ssm_state_dtype != torch.float32:
        h0 = h0.to(case.ssm_state_dtype)

    return k, w_orig, u_orig, w_c, u_c, g, h0, cu_seqlens


def _build_prefill_metadata(context_lens, cu_seqlens, chunk_size: int = 64):
    """Prebuild the GDR chunk schedule a serving stack builds once per forward
    pass. Skipping it makes each wrapper rediscover the chunk counts with a
    blocking D2H copy. None for dense shapes, where the wrappers read the batch
    layout straight off the tensor shapes.
    """
    if cu_seqlens is None:
        return None
    return build_gated_delta_rule_prefill_metadata(
        list(context_lens),
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )


def _hip_k5_supported(case: PrefillArgs) -> bool:
    """The HIP K5 kernel only handles K=V=128, bf16 inputs, chunk_size=64."""
    return (
        _HAS_HIP_K5
        and case.K == 128
        and case.V == 128
        and case.dtype == torch.bfloat16
        and case.BT == 64
    )


def _normalize_opt_v_new(vn_opt):
    """Convert the kernels' v_new layout [B, H, T, V] back to [B, T, H, V]."""
    return vn_opt.permute(0, 2, 1, 3).contiguous()


# -- Pure-PyTorch reference ----------------------------------------------


def ref_chunk_gated_delta_rule_fwd_h(
    k,
    w,
    u,
    g,
    initial_state=None,
    output_final_state=False,
    chunk_size=64,
    cu_seqlens=None,
    g_head_major=False,
):
    """Reference in FP32 for correctness checking. Never timed."""
    B, T, Hg_dim, K_dim = k.shape
    H_dim, V_dim = u.shape[-2], u.shape[-1]
    BT_dim = chunk_size
    if cu_seqlens is None:
        NT = _cdiv(T, BT_dim)
    else:
        seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        NT = sum(_cdiv(int(seq_len), BT_dim) for seq_len in seq_lens)
    gqa_ratio = H_dim // Hg_dim

    h_out = k.new_zeros(B, NT, H_dim, V_dim, K_dim, dtype=torch.float32)
    v_new_out = torch.zeros_like(u, dtype=torch.float32)

    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    final_state = (
        torch.zeros(N, H_dim, V_dim, K_dim, dtype=torch.float32, device=k.device)
        if output_final_state
        else None
    )

    for b_idx in range(B):
        if cu_seqlens is not None:
            seqs = [
                (s, cu_seqlens[s].item(), cu_seqlens[s + 1].item()) for s in range(N)
            ]
        else:
            seqs = [(b_idx, 0, T)]

        chunk_offset = 0
        for seq_idx, bos, eos in seqs:
            seq_len = eos - bos
            seq_nt = _cdiv(seq_len, BT_dim)

            for i_h in range(H_dim):
                i_hg = i_h // gqa_ratio
                h_state = torch.zeros(
                    V_dim, K_dim, dtype=torch.float32, device=k.device
                )
                if initial_state is not None:
                    h_state = initial_state[seq_idx, i_h].float().clone()

                for i_t in range(seq_nt):
                    t_start = i_t * BT_dim
                    t_end = min(t_start + BT_dim, seq_len)
                    actual_bt = t_end - t_start

                    h_out[b_idx, chunk_offset + i_t, i_h] = h_state.clone()

                    w_chunk = w[b_idx, bos + t_start : bos + t_end, i_h].float()
                    u_chunk = u[b_idx, bos + t_start : bos + t_end, i_h].float()
                    b_v = u_chunk - w_chunk @ h_state.T
                    v_new_out[b_idx, bos + t_start : bos + t_end, i_h] = b_v

                    # g sequence for (b_idx, i_h) under either 3-D layout.
                    if g is None:
                        g_seq = None
                    elif g_head_major:
                        g_seq = g[b_idx, i_h]
                    else:
                        g_seq = g[b_idx, :, i_h]

                    mask = torch.zeros(BT_dim, device=k.device)
                    mask[:actual_bt] = 1.0
                    if g_seq is None:
                        # No decay: valid rows gate to 1, matching the kernel's
                        # pure padding masking under USE_G=False.
                        gate = mask[:actual_bt]
                    else:
                        last_idx = bos + t_end - 1
                        g_last = g_seq[last_idx].float()
                        g_chunk = g_seq[bos + t_start : bos + t_end].float()
                        gate = torch.where(
                            mask[:actual_bt].bool(),
                            torch.exp(g_last - g_chunk),
                            torch.zeros_like(g_chunk),
                        )
                        h_state = h_state * torch.exp(g_last)
                    b_v_gated = b_v * gate.unsqueeze(-1)

                    k_chunk = k[b_idx, bos + t_start : bos + t_end, i_hg].float()
                    b_v_gated_cast = b_v_gated.to(k.dtype).float()
                    h_state = h_state + b_v_gated_cast.T @ k_chunk

                if output_final_state:
                    final_state[seq_idx, i_h] = h_state

            chunk_offset += seq_nt

    return h_out, v_new_out.to(u.dtype), final_state


# -- Benchmark -----------------------------------------------------------


def _build_case(model, tp, seqlen, total_tokens, mode, snapshot_dtype, state_dtype):
    """Materialise the ``PrefillArgs`` for one sweep row.

    Dense rows mirror ``_k5_dense_groups`` (single segment, no final state);
    varlen rows mirror ``_k5_varlen_groups`` (``total_tokens // seqlen`` equal
    segments, final state written back).
    """
    spec = K5_MODELS[model]
    return PrefillArgs(
        K=128,
        V=128,
        Hk=16,
        Hv=spec["Hv"],
        tp=tp,
        full_prompt_len=seqlen,
        model_name=f"{spec['label']}-{mode}",
        max_num_batched_tokens=total_tokens,
        is_varlen=mode == "varlen",
        output_final_state=mode == "varlen",
        ssm_state_dtype=_STATE_DTYPES[state_dtype],
        snapshot_dtype=_SNAPSHOT_DTYPES[snapshot_dtype],
    )


@benchmark()
def test_chunk_gdn_prefill_h(
    model, tp, seqlen, total_tokens, mode, snapshot_dtype, state_dtype
):
    case = _build_case(
        model, tp, seqlen, total_tokens, mode, snapshot_dtype, state_dtype
    )
    context_lens = case.resolve_context_lens()
    k, w_orig, u_orig, w_c, u_c, g, h0, cu = _make_inputs(case, context_lens)
    ofs = case.output_final_state
    H, Hg, K, V, BT = case.H, case.Hg, case.K, case.V, case.BT

    # Triton/HIP consume a head-major g; FlyDSL takes the layout flag directly.
    g_hm = None
    if g is not None:
        g_hm = g if case.g_head_major else g.transpose(1, 2).contiguous()

    # Shared by every backend; without it the D2H stall would show up in the
    # measurement as host behaviour the production path does not have.
    metadata = _build_prefill_metadata(context_lens, cu)

    if case.is_varlen:
        B, N = 1, len(context_lens)
        total_chunks = sum(_cdiv(n, BT) for n in context_lens)
    else:
        B = N = case.dense_batch
        total_chunks = B * _cdiv(sum(context_lens), BT)
    T_flat = int(cu[-1].item()) if cu is not None else sum(context_lens)

    ref_h, ref_vn, ref_fs = ref_chunk_gated_delta_rule_fwd_h(
        k,
        w_orig,
        u_orig,
        g=g,
        initial_state=h0,
        output_final_state=ofs,
        chunk_size=BT,
        cu_seqlens=cu,
        g_head_major=case.g_head_major,
    )

    common = {
        "initial_state": h0,
        "output_final_state": ofs,
        "chunk_size": BT,
        "cu_seqlens": cu,
        "state_dtype": case.ssm_state_dtype,
        "snapshot_dtype": case.snapshot_dtype,
        "prefill_metadata": metadata,
        # ``g`` is natural-log space and the reference decays with ``exp``, so
        # the kernels must apply the LOG2E scale (exp2(x*LOG2E) == exp(x)). The
        # default True would read ``g`` as log2-space, a mismatch masked only by
        # gates decaying to 0.
        "use_exp2": False,
    }
    candidates = {
        "flydsl": lambda: chunk_gated_delta_rule_fwd_h_flydsl_opt(
            k, w_c, u_c, g=g, g_head_major=case.g_head_major, **common
        ),
        "triton": lambda: chunk_gated_delta_rule_fwd_h_opt_vk(
            k, w_c, u_c, g=g_hm, **common
        ),
    }
    # Unsupported shapes leave the HIP cells nan rather than dropping the row.
    if _hip_k5_supported(case):
        candidates["hip"] = lambda: chunk_gated_delta_rule_fwd_h_hip_fn(
            k, w_c, u_c, g=g_hm, g_head_major=True, **common
        )

    # Two bf16 MFMA GEMMs against the [V, K] state per (chunk, head):
    #   v_new = u - w @ h^T   ([BT,K] @ [K,V])  -> 2*BT*K*V
    #   h    += v_gated^T @ k ([V,BT] @ [BT,K]) -> 2*BT*V*K
    # Elementwise gate work is negligible. Padded tokens count, since a partial
    # chunk costs a full one.
    flops = 4 * (total_chunks * BT) * H * K * V
    esz = k.element_size()
    snap_esz = _dtype_size(case.snapshot_dtype or case.dtype)
    state_esz = _dtype_size(case.ssm_state_dtype)
    nbytes = (
        T_flat * Hg * K * esz  # k in
        + T_flat * H * K * esz  # w in
        + T_flat * H * V * esz  # u in
        + (T_flat * H * 4 if g is not None else 0)  # g in (fp32 cumulative gate)
        + T_flat * H * V * esz  # v_new out
        # the per-chunk h snapshots dominate: one [V, K] tile per chunk and head
        + total_chunks * H * V * K * snap_esz
        + N * H * V * K * state_esz  # initial state in
        + (N * H * V * K * state_esz if ofs else 0)  # final state out
    )

    ret = {"gfx": get_gfx(), "B": B, "N": N, "H": H, "T_flat": T_flat}
    for name, fn in candidates.items():
        (h, vn, fs), us = run_perftest(fn)

        # Output contract shared by all three backends.
        assert h.shape == (B, total_chunks // B, H, V, K), f"{name}: h shape {h.shape}"
        assert h.dtype == (case.snapshot_dtype or k.dtype), f"{name}: h dtype {h.dtype}"
        assert vn.shape == (B, H, T_flat, V), f"{name}: v_new shape {vn.shape}"
        if ofs:
            assert fs.shape == (N, H, V, K), f"{name}: final_state shape {fs.shape}"
            assert fs.dtype == case.ssm_state_dtype, f"{name}: fs dtype {fs.dtype}"
        else:
            assert fs is None, f"{name}: expected no final_state"

        err = checkAllclose(
            ref_h.to(dtypes.fp32),
            h.to(dtypes.fp32),
            rtol=2e-2,
            atol=2e-2,
            msg=f"{name}: K5 h snapshots",
        )
        err = max(
            err,
            checkAllclose(
                ref_vn.to(dtypes.fp32),
                _normalize_opt_v_new(vn).to(dtypes.fp32),
                rtol=2e-2,
                atol=2e-2,
                msg=f"{name}: K5 v_new",
            ),
        )
        if ofs:
            err = max(
                err,
                checkAllclose(
                    ref_fs.to(dtypes.fp32),
                    fs.to(dtypes.fp32),
                    rtol=2e-2,
                    atol=2e-2,
                    msg=f"{name}: K5 final_state",
                ),
            )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err

    return ret


def _sweep_rows(args, model):
    """Cartesian sweep of the CLI axes, minus the combinations K5 cannot run.

    Dense prefill is one contiguous prompt, so ``total_tokens`` is pinned to
    ``seqlen`` there and only shapes varlen batches (N = total_tokens //
    seqlen). Rows that collapse onto the same shape are dropped.
    """
    rows, seen = [], set()
    for tp, mode, seqlen, total, snap, state in itertools.product(
        args.tp,
        args.mode,
        args.seqlen,
        args.total_tokens,
        args.snapshot_dtype,
        args.state_dtype,
    ):
        if mode == "dense":
            total = seqlen
        elif total < seqlen:
            continue  # a varlen batch needs at least one full segment
        row = (model, tp, seqlen, total, mode, snap, state)
        if row not in seen:
            seen.add(row)
            rows.append(row)
    return rows


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("GDN prefill K5 unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(K5_MODELS),
        nargs="*",
        default=["35b"],
        help=f"""Qwen3.5 GDN variant (sets Hv).
        {", ".join(f"{n}: Hv={s['Hv']}" for n, s in K5_MODELS.items())}
        e.g.: --model 35b 397b""",
    )
    parser.add_argument(
        "--tp",
        type=int,
        choices=_K5_TPS,
        nargs="*",
        default=[1, 8],
        help="""Tensor-parallel degree; H = Hv // tp, Hg = 16 // tp.
        e.g.: --tp 1 2 4 8""",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dense", "varlen"],
        nargs="*",
        default=["dense", "varlen"],
        help="""Launch path. dense = one prompt, no final state;
        varlen = cu_seqlens batch, final state written back.
        e.g.: --mode varlen""",
    )
    parser.add_argument(
        "-t",
        "--seqlen",
        type=int,
        nargs="*",
        default=[1024, 4096],
        help=f"""Per-sequence prompt length (the dense prompt length, or the
        varlen segment length). Catalog values: dense {_DENSE_PROMPT_LENS},
        varlen {_VARLEN_SEQLENS}.
        e.g.: --seqlen 1024 4096""",
    )
    parser.add_argument(
        "--total-tokens",
        type=int,
        nargs="*",
        default=[8192],
        help=f"""Scheduler token budget for the varlen path (batch size
        N = total_tokens // seqlen); ignored by dense, which always runs one
        seqlen-long prompt. Catalog values: {_VARLEN_TOTAL_T}.
        e.g.: --total-tokens 8192 32768""",
    )
    parser.add_argument(
        "--snapshot-dtype",
        type=str,
        choices=list(_SNAPSHOT_DTYPES),
        nargs="*",
        default=["bf16", "fp32"],
        help="""Per-chunk h snapshot store dtype (bf16 = k.dtype default).
        e.g.: --snapshot-dtype bf16 fp32""",
    )
    parser.add_argument(
        "--state-dtype",
        type=str,
        choices=list(_STATE_DTYPES),
        nargs="*",
        default=["fp32"],
        help="""Persistent SSM initial/final state dtype.
        e.g.: --state-dtype fp32 bf16""",
    )
    args = parser.parse_args()

    for model in args.model:  # one table per model (Hv differs -> different shapes)
        df = [test_chunk_gdn_prefill_h(*row) for row in _sweep_rows(args, model)]
        df = pd.DataFrame(df)
        aiter.logger.info(
            "chunk_gdn_prefill_h (%s) summary (markdown):\n%s",
            K5_MODELS[model]["label"],
            df.to_markdown(index=False),
        )


if __name__ == "__main__":
    main()

__all__ = [
    "_SNAPSHOT_DTYPES",
    "_STATE_DTYPES",
    "PrefillArgs",
    "_build_case",
    "_build_context_lens",
    "_build_cu_seqlens",
    "_build_prefill_metadata",
    "_case_seed",
    "_cdiv",
    "_dtype_size",
    "_hip_k5_supported",
    "_make_inputs",
    "_normalize_opt_v_new",
    "_sweep_rows",
    "main",
    "ref_chunk_gated_delta_rule_fwd_h",
    "test_chunk_gdn_prefill_h",
]

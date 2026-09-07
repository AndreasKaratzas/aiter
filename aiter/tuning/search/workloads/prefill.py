# SPDX-License-Identifier: MIT
"""Recurrent prefill shape catalog used by tuning and regression tests."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _build_context_lens(full_prompt_len, max_tokens=32768):
    context_lens = []
    remaining = max_tokens
    while remaining > 0:
        cur = min(full_prompt_len, remaining)
        context_lens.append(cur)
        remaining -= cur
    return context_lens


@dataclass
class PrefillArgs:
    K: int
    V: int
    Hk: int
    Hv: int
    tp: int
    full_prompt_len: int
    model_name: str = ""
    BT: int = 64
    max_num_batched_tokens: int = 32768
    dtype: torch.dtype = torch.bfloat16
    is_varlen: bool = True
    output_final_state: bool = True
    # SSM-state dtype for h0 / final_state; the f32 accumulator is unchanged
    # either way, so bf16 only trims HBM traffic and footprint.
    ssm_state_dtype: torch.dtype = torch.float32
    # Per-chunk h-snapshot dtype, independent of the SSM state dtype.
    # None -> k.dtype (bf16 store specialization); torch.float32 -> fp32 one.
    snapshot_dtype: object = None  # torch.dtype | None
    # Explicit segment lengths that override ``_build_context_lens``, for
    # trace-derived ragged batches the "equal segments + remainder" recipe
    # cannot express.
    context_lens: object = None  # list[int] | None
    # Disambiguates ids when many trace shapes share the same (T, num_seqs).
    trace_tag: str = ""
    # Disambiguates ids when a group sweeps ``max_num_batched_tokens``.
    bt_tag: str = ""
    # Dense-path batch size. B>1 builds ``g`` as [B, H, T_flat], exercising the
    # ``i_n*H*T_flat`` batch stride in the kernel's ``g_head_base``. The varlen
    # path ignores this (always B=1, N segments).
    dense_batch: int = 1
    # False takes the ``g=None`` (USE_G=False) path, where the last chunk's
    # padding rows must be masked or their v_new corrupts the state update.
    use_g: bool = True
    # False (default) -> token-major [B, T_flat, H] (== HIP default);
    # True -> head-major [B, H, T_flat].
    g_head_major: bool = False

    @property
    def Hg(self):
        return self.Hk // self.tp

    @property
    def H(self):
        return self.Hv // self.tp

    def resolve_context_lens(self):
        """Per-segment token counts: the captured trace list, or the
        equal-length recipe ``_build_context_lens`` emits."""
        if self.context_lens is not None:
            return list(self.context_lens)
        return _build_context_lens(self.full_prompt_len, self.max_num_batched_tokens)

    def __repr__(self):
        # Elide a trace case's cu_seqlens; they run to 64+ entries.
        if self.context_lens is not None:
            n = len(self.context_lens)
            T = sum(self.context_lens)
            tag = self.model_name or "trace"
            tag += f"_T{T}_n{n}"
            if self.trace_tag:
                tag += f"_{self.trace_tag}"
            if not self.use_g:
                tag += "_nog"
            if self.g_head_major:
                tag += "_ghm"
            return tag
        tag = self.model_name + "_" if self.model_name else ""
        tag += f"K{self.K}_V{self.V}_Hk{self.Hk}_Hv{self.Hv}"
        tag += f"_TP{self.tp}_T{self.full_prompt_len}"
        if self.bt_tag:
            tag += f"_{self.bt_tag}"
        if not self.is_varlen:
            tag += "_novarlen"
        if self.dense_batch != 1:
            tag += f"_B{self.dense_batch}"
        if not self.use_g:
            tag += "_nog"
        if self.g_head_major:
            tag += "_ghm"
        if not self.output_final_state:
            tag += "_nofs"
        if self.ssm_state_dtype == torch.bfloat16:
            tag += "_stateBF16"
        if self.snapshot_dtype == torch.float32:
            tag += "_snapFP32"
        return tag


@dataclass
class PrefillGroup:
    """A family of ``PrefillArgs`` cases sharing every field except ``tp`` and
    ``full_prompt_len``.

    ``expand_groups`` materialises the (tps x full_prompt_lens) Cartesian
    product into the flat list ``PREFILL_PARAMS`` exposes. ``PrefillArgs.__repr__``
    encodes (tp, full_prompt_len), so ids stay unique within a ``model_name``.
    """

    model_name: str
    Hv: int
    tps: list
    full_prompt_lens: list
    Hk: int = 16
    K: int = 128
    V: int = 128
    BT: int = 64
    dtype: torch.dtype = torch.bfloat16
    is_varlen: bool = True
    output_final_state: bool = True
    ssm_state_dtype: torch.dtype = torch.float32
    # Per-chunk h-snapshot dtype; None -> k.dtype (bf16). See PrefillArgs.
    snapshot_dtype: object = None  # torch.dtype | None
    # Semantics for ``max_num_batched_tokens``:
    #   - list/tuple : sweep one case per element (Cartesian with the rest),
    #           each element being one of the specs below. For varlen this
    #           sweeps the batch size N = mnbt // full_prompt_len, and ids gain
    #           an ``mnbt{value}`` suffix.
    #   - int : a fixed scheduler budget across the full_prompt_len sweep.
    #   - "full_prompt_len" : tie it to each case, so ``_build_context_lens``
    #           returns exactly one segment (the dense rows).
    #   - None (default) : the ``PrefillArgs`` default of 32768, e.g.
    #           ``_build_context_lens(1024, 32768)`` -> 32 segments of 1024.
    max_num_batched_tokens: object = None
    # Trace-derived ragged expansion. When set, the group materialises the
    # (tps x full_prompt_lens x head_seqlens) product with an explicit
    # cu_seqlens instead of the equal-split recipe, which probes how sensitive
    # a backend is to where the segment boundary falls.
    head_seqlens: object = None  # list[int] | None
    mid_seqlen: int = 10000
    # Segments per case when ``head_seqlens`` is set:
    #   3 (default): [head, mid_seqlen, full_len - head - mid_seqlen]
    #   2          : [head, full_len - head]; ``mid_seqlen`` is ignored
    num_segments: int = 3
    # See PrefillArgs for the three fields below.
    dense_batch: int = 1
    use_g: bool = True
    g_head_major: bool = False


def expand_groups(groups):
    out = []
    for g in groups:
        # A list of specs sweeps one case per value; ids only gain the
        # ``mnbt{value}`` suffix when there is more than one.
        mnbt_specs = g.max_num_batched_tokens
        if not isinstance(mnbt_specs, (list, tuple)):
            mnbt_specs = [mnbt_specs]
        _sweep_mnbt = len(mnbt_specs) > 1
        for tp in g.tps:
            for full_len in g.full_prompt_lens:
                for mnbt_spec in mnbt_specs:
                    if mnbt_spec == "full_prompt_len":
                        mnbt = full_len
                    elif mnbt_spec is None:
                        mnbt = 32768  # PrefillArgs dataclass default
                    else:
                        mnbt = mnbt_spec
                    bt_tag = f"mnbt{mnbt}" if _sweep_mnbt else ""

                    # No head_seqlens means the equal split _build_context_lens
                    # produces; otherwise one case per (tp, full_len, head).
                    if g.head_seqlens is None:
                        out.append(
                            PrefillArgs(
                                K=g.K,
                                V=g.V,
                                Hk=g.Hk,
                                Hv=g.Hv,
                                tp=tp,
                                full_prompt_len=full_len,
                                model_name=g.model_name,
                                BT=g.BT,
                                max_num_batched_tokens=mnbt,
                                dtype=g.dtype,
                                is_varlen=g.is_varlen,
                                output_final_state=g.output_final_state,
                                ssm_state_dtype=g.ssm_state_dtype,
                                snapshot_dtype=g.snapshot_dtype,
                                bt_tag=bt_tag,
                                dense_batch=g.dense_batch,
                                use_g=g.use_g,
                                g_head_major=g.g_head_major,
                            )
                        )
                    else:
                        for head in g.head_seqlens:
                            if g.num_segments == 2:
                                tail = full_len - head
                                if tail <= 0:
                                    raise ValueError(
                                        f"head_seqlens (num_segments=2) produced "
                                        f"non-positive tail ({tail}) for "
                                        f"group={g.model_name!r} "
                                        f"full_prompt_len={full_len} head={head}."
                                    )
                                context_lens = [head, tail]
                                tag = f"head{head}_tail{tail}"
                            elif g.num_segments == 3:
                                tail = full_len - head - g.mid_seqlen
                                if tail <= 0:
                                    raise ValueError(
                                        f"head_seqlens (num_segments=3) produced "
                                        f"non-positive tail ({tail}) for "
                                        f"group={g.model_name!r} "
                                        f"full_prompt_len={full_len} head={head} "
                                        f"mid_seqlen={g.mid_seqlen}. Drop this "
                                        f"(full_len, head) combo or raise "
                                        f"full_prompt_len."
                                    )
                                context_lens = [head, g.mid_seqlen, tail]
                                tag = f"head{head}_mid{g.mid_seqlen}"
                            else:
                                raise ValueError(
                                    f"num_segments={g.num_segments} unsupported; "
                                    f"only 2 or 3 are implemented."
                                )
                            if _sweep_mnbt:
                                tag = f"{tag}_mnbt{mnbt}"
                            out.append(
                                PrefillArgs(
                                    K=g.K,
                                    V=g.V,
                                    Hk=g.Hk,
                                    Hv=g.Hv,
                                    tp=tp,
                                    full_prompt_len=full_len,
                                    model_name=g.model_name,
                                    BT=g.BT,
                                    max_num_batched_tokens=mnbt,
                                    dtype=g.dtype,
                                    is_varlen=g.is_varlen,
                                    output_final_state=g.output_final_state,
                                    ssm_state_dtype=g.ssm_state_dtype,
                                    snapshot_dtype=g.snapshot_dtype,
                                    context_lens=context_lens,
                                    trace_tag=tag,
                                    dense_batch=g.dense_batch,
                                    use_g=g.use_g,
                                    g_head_major=g.g_head_major,
                                )
                            )
    return out


# Qwen3.5 GDN prefill deployments: Hk=16 KV heads, K=V=128, chunk BT=64.
K5_MODELS = {
    "35b": {"label": "Qwen3.5-35B", "Hv": 32},
    "397b": {"label": "Qwen3.5-397B", "Hv": 64},
}
_DENSE_PROMPT_LENS = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
_VARLEN_SEQLENS = [1024, 2048, 4096, 8192]
_VARLEN_TOTAL_T = [8192, 16384, 32768, 65536]
_K5_TPS = [1, 2, 4, 8]
_SNAPSHOT_DTYPES = {"bf16": None, "fp32": torch.float32}  # None -> k.dtype (bf16)
_STATE_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def _k5_dense_groups(model_name: str, hv: int) -> list[PrefillGroup]:
    """Dense prefill: TP x T sweep x bf16/fp32 per-chunk snapshot."""
    groups: list[PrefillGroup] = []
    for tp in _K5_TPS:
        groups.append(
            PrefillGroup(
                model_name=f"{model_name}-dense-tp{tp}-bf16snap",
                Hv=hv,
                tps=[tp],
                full_prompt_lens=_DENSE_PROMPT_LENS,
                is_varlen=False,
                output_final_state=False,
                max_num_batched_tokens="full_prompt_len",
            )
        )
        groups.append(
            PrefillGroup(
                model_name=f"{model_name}-dense-tp{tp}-fp32snap",
                Hv=hv,
                tps=[tp],
                full_prompt_lens=_DENSE_PROMPT_LENS,
                is_varlen=False,
                output_final_state=False,
                max_num_batched_tokens="full_prompt_len",
                snapshot_dtype=torch.float32,
            )
        )
    return groups


def _k5_varlen_groups(model_name: str, hv: int) -> list[PrefillGroup]:
    """Varlen prefill: TP x seqlen x total T x bf16/fp32 snapshot."""
    groups: list[PrefillGroup] = []
    for tp in _K5_TPS:
        groups.append(
            PrefillGroup(
                model_name=f"{model_name}-varlen-tp{tp}-bf16snap",
                Hv=hv,
                tps=[tp],
                full_prompt_lens=_VARLEN_SEQLENS,
                max_num_batched_tokens=_VARLEN_TOTAL_T,
            )
        )
        groups.append(
            PrefillGroup(
                model_name=f"{model_name}-varlen-tp{tp}-fp32snap",
                Hv=hv,
                tps=[tp],
                full_prompt_lens=_VARLEN_SEQLENS,
                max_num_batched_tokens=_VARLEN_TOTAL_T,
                snapshot_dtype=torch.float32,
            )
        )
    return groups


_PREFILL_GROUPS = [
    *_k5_dense_groups(K5_MODELS["35b"]["label"], K5_MODELS["35b"]["Hv"]),
    *_k5_dense_groups(K5_MODELS["397b"]["label"], K5_MODELS["397b"]["Hv"]),
    *_k5_varlen_groups(K5_MODELS["35b"]["label"], K5_MODELS["35b"]["Hv"]),
    *_k5_varlen_groups(K5_MODELS["397b"]["label"], K5_MODELS["397b"]["Hv"]),
]

# Full tuner catalog. ``aiter/tuning/search/recurrent/gdn_k5.py`` loads this module
# by path and zips these two lists, so both must stay module-level and aligned.
PREFILL_PARAMS = expand_groups(_PREFILL_GROUPS)
PREFILL_TEST_IDS = [repr(p) for p in PREFILL_PARAMS]

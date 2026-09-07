# Unquantized CK MoE requires prepared weights

The CK two-stage FP16/BF16 kernels expect preshuffled weights. The Python path
previously accepted tensors with `is_shuffled=False`, built an extension whose
name contained `preshuffle_off`, and passed those bytes to the same fixed-layout
kernel family. Both expert stages could return incorrect values without an
error. The generator's raw/preshuffled alternatives belong to other quantized
families; the extension name did not establish unquantized raw-layout support.

Requests with `QuantType.No`, FP16/BF16 weights and an unprepared weight now raise
`ValueError` before the selected CK adapter loads or launches. The public fused
path checks both selected stages before sorting. This applies to
`aiter.fused_moe.fused_moe`, its two-stage path, and direct CK stage preparation
through `ck_moe_stage1_fwd` / `ck_moe_stage2_fwd`. The direct low-level stage
functions also reject an explicit `is_shuffled=False` request for this family.
Other backends and quantized raw-capable formats retain their existing behavior.

Prepare weights once when loading a model, preserving the layout attribute:

```python
from aiter.ops.shuffle import shuffle_weight

prepared_w1 = shuffle_weight(w1)
prepared_w2 = shuffle_weight(w2)
prepared_w1.is_shuffled = True
prepared_w2.is_shuffled = True
```

vLLM already performs this conversion in
`vllm/model_executor/layers/fused_moe/oracle/unquantized.py` using
`rocm_aiter_ops.shuffle_weights`. Its runtime call remains supported. The new
framework tests exercise that actual preprocessing and compare expert execution
and weighted combination with independent FP64 matrix operations. They also
reject either one or both unprepared weights while native loading and sorting
are forbidden.

This correction changes previously accepted **incorrect** raw-weight calls into
an explicit error. It does not add a raw CK kernel or silently copy and shuffle
weights on each invocation. Numerical evidence for this change is from gfx950;
no additional hardware qualification is implied.

# Fused activation reference precision

The nightly activation driver compared two different calculations. The native
`silu_and_mul_quant` operation keeps SiLU and multiplication in FP32 before
choosing a quantization scale. Its old test reference rounded the SiLU output
to the input dtype, multiplied and rounded again, and only then converted to
FP32. Those extra rounding steps can change an MXFP4 scale by a factor of two.

For example, BF16 gate `0.7421875` and up value `5.96875` produce approximately
`3.00116358` in an independent FP64 calculation. The fused calculation therefore
needs scale `1`, encoded as E8M0 byte `127`. The old reference rounded the result
to exactly `3`, choosing scale `0.5`, encoded as byte `126`.

The quantized activation driver now uses an explicit FP32 reference in
`tests/operators/hip/references/activation.py`. The reference also preserves the
native limit rule: a clamped gate is converted back to the input representation,
while the clamped up branch stays FP32. Unquantized activation references are
unchanged. The native activation algorithm and comparison tolerances are
unchanged.

Regression tests check the exact positive and negative boundary encodings,
FP16 and BF16 inputs, zero groups, FP8 and MXFP4 scales, and representable and
nonrepresentable clamp limits. Scale expectations use an independent FP64
calculation rather than AITER's scale conversion helper. This fused operation's
round-up scale policy remains separate from the ordinary prepared MXFP4
quantizer's documented rounding policy.

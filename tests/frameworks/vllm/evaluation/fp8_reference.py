# SPDX-License-Identifier: MIT
"""Independent FP32 block scaling for publisher FP8 checkpoint reference weights."""


def dequantize_weight(weight, scales, block_size):
    import torch

    if weight.ndim != 2 or scales.ndim != 2 or block_size != (128, 128):
        raise ValueError("Reference requires matrix weights and declared128x128 blocks")
    rows, cols = weight.shape
    if tuple(scales.shape) != ((rows + 127) // 128, (cols + 127) // 128):
        raise ValueError("Checkpoint FP8 scales do not cover every weight")
    if not torch.isfinite(scales).all() or not (scales > 0).all():
        raise ValueError("Checkpoint FP8 scales must be finite and positive")
    return (
        weight.float()
        * scales.float()
        .repeat_interleave(128, 0)
        .repeat_interleave(128, 1)[:rows, :cols]
    )


def load_reference(snapshot, dtype):
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM
    from pathlib import Path

    config = AutoConfig.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    quant = config.quantization_config
    if quant.get("quant_method") != "fp8" or quant.get("weight_block_size") != [
        128,
        128,
    ]:
        raise ValueError("Unsupported reference checkpoint FP8 encoding")
    del config.quantization_config
    state = load_file(str(Path(snapshot) / "model.safetensors"), device="cpu")
    for key in tuple(state):
        if not key.endswith(".weight_scale_inv"):
            continue
        weight_key = key.removesuffix("_scale_inv")
        state[weight_key] = dequantize_weight(
            state[weight_key], state.pop(key), (128, 128)
        ).to(dtype)
    model = AutoModelForCausalLM.from_config(
        config, torch_dtype=dtype, attn_implementation="eager"
    )
    model.load_state_dict(state, strict=True, assign=True)
    if any(parameter.is_meta for parameter in model.parameters()):
        raise ValueError("Reference checkpoint did not initialize all parameters")
    return model.to(device="cuda", dtype=dtype).eval()

# SPDX-License-Identifier: MIT
"""Independent Transformers eager attention oracle using the same admitted weights."""

import argparse
import hashlib
import json
from pathlib import Path


def evaluate(request):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = request.get("dtype", "bfloat16")
    if dtype not in ("bfloat16", "float16"):
        raise ValueError("Reference dtype must be BF16 or FP16")

    tokenizer = AutoTokenizer.from_pretrained(
        request["model"]["snapshot"], local_files_only=True
    )
    extra = {}
    if request.get("dequantize_mxfp4", False):
        from transformers import Mxfp4Config

        extra["quantization_config"] = Mxfp4Config(dequantize=True)
    if request.get("dequantize_fp8", False):
        from frameworks.vllm.evaluation.fp8_reference import load_reference

        if extra:
            raise ValueError("Reference quantization formats are mutually exclusive")
        model = load_reference(request["model"]["snapshot"], getattr(torch, dtype))
    else:
        model = (
            AutoModelForCausalLM.from_pretrained(
                request["model"]["snapshot"],
                local_files_only=True,
                trust_remote_code=False,
                dtype=getattr(torch, dtype),
                attn_implementation="eager",
                **extra,
            )
            .to("cuda")
            .eval()
        )
    outputs = []
    with torch.inference_mode():
        for prompt in request["prompts"]:
            encoded = tokenizer(prompt, return_tensors="pt").to("cuda")
            ids = encoded["input_ids"]
            logits = model(**encoded).logits[0, :-1].float()
            logprobs = logits.log_softmax(-1).gather(-1, ids[0, 1:, None]).flatten()
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=request["max_tokens"],
                pad_token_id=tokenizer.eos_token_id,
            )
            outputs.append(
                {
                    "prompt_token_ids": ids[0].tolist(),
                    "prompt_logprobs": [None, *logprobs.tolist()],
                    "token_ids": generated[0, ids.shape[1] :].tolist(),
                }
            )
    return {
        "outputs": outputs,
        "model": request["model"],
        "dtype": str(model.dtype),
        "attention_implementation": model.config._attn_implementation,
        "environment": {
            "torch": {"version": torch.__version__, "path": torch.__file__},
            "transformers": {
                "version": transformers.__version__,
                "path": transformers.__file__,
            },
        },
        "model_class": type(model).__module__ + "." + type(model).__name__,
        "dequantized_mxfp4": bool(request.get("dequantize_mxfp4", False)),
        "dequantized_fp8": bool(request.get("dequantize_fp8", False)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    result = evaluate(request)
    result["request_sha256"] = hashlib.sha256(args.request.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

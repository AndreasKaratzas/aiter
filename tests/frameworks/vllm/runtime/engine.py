# SPDX-License-Identifier: MIT
"""Execute bounded batches through a real vLLM engine and all its workers."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from .protocol import parse_request


def options_for(settings, model):
    options = {
        "model": model["snapshot"],
        "dtype": settings.dtype,
        "seed": 0,
        "trust_remote_code": False,
        "enforce_eager": not settings.cuda_graph,
        "max_model_len": settings.max_model_len,
        "max_num_seqs": 4,
        "max_num_batched_tokens": settings.prefill_budget,
        "enable_chunked_prefill": settings.chunked_prefill,
        "kv_cache_memory_bytes": 512 * 1024**2,
        "gpu_memory_utilization": 0.1,
        "enable_prefix_caching": settings.prefix_cache,
        "disable_log_stats": False,
        "async_scheduling": False,
        "tensor_parallel_size": settings.tensor_parallel,
        "worker_cls": "frameworks.vllm.runtime.worker.ObservedWorker",
        "kernel_config": {
            "ir_op_priority": {"rms_norm": ["aiter"], "fused_add_rms_norm": ["aiter"]}
        },
        "attention_config": {"backend": "ROCM_AITER_UNIFIED_ATTN"},
    }
    if settings.tensor_parallel > 1:
        options["distributed_executor_backend"] = "mp"
    if settings.online_fp8:
        options["quantization"] = "fp8_per_channel"
    if settings.speculative:
        options["speculative_config"] = {
            "method": "ngram_gpu",
            "prompt_lookup_max": 3,
            "prompt_lookup_min": 2,
            "num_speculative_tokens": 3,
        }
    if settings.cuda_graph:
        options["compilation_config"] = {
            "mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 4],
            "max_cudagraph_capture_size": 4,
        }
    if settings.multimodal:
        options["limit_mm_per_prompt"] = {"image": 1, "video": 0}
        options["mm_processor_kwargs"] = {
            "min_pixels": 224 * 224,
            "max_pixels": 224 * 224,
        }
    return options


def materialize(prompts):
    if isinstance(prompts[0], str) or "prompt_token_ids" in prompts[0]:
        return list(prompts), []
    from PIL import Image

    values, hashes = [], []
    for item in prompts:
        image = Image.new("RGB", tuple(item["size"]), tuple(item["rgb"]))
        text = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            + item["text"]
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
        values.append({"prompt": text, "multi_modal_data": {"image": image}})
        hashes.append(hashlib.sha256(image.tobytes()).hexdigest())
    return values, hashes


def generate(request):
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    settings, batches = parse_request(request)
    options = options_for(settings, request["model"])
    engine = None
    try:
        engine = LLM(**options)
        records = []
        for batch in batches:
            if batch.reset_prefix_cache:
                engine.reset_prefix_cache()
            engine.collective_rpc("aiter_test_reset", timeout=120)
            prompts, image_hashes = materialize(batch.prompts)
            outputs = engine.generate(
                prompts,
                SamplingParams(
                    temperature=0,
                    max_tokens=batch.max_tokens,
                    ignore_eos=batch.ignore_eos,
                    prompt_logprobs=1 if batch.prompt_logprobs else None,
                ),
                use_tqdm=False,
            )
            observations = engine.collective_rpc("aiter_test_snapshot", timeout=120)
            records.append(
                {
                    "name": batch.name,
                    "outputs": [
                        {
                            "text": output.outputs[0].text,
                            "token_ids": output.outputs[0].token_ids,
                            "finish_reason": output.outputs[0].finish_reason,
                            "prompt_token_ids": output.prompt_token_ids,
                            "prompt_logprobs": (
                                [
                                    (
                                        values[token].logprob
                                        if values is not None
                                        else None
                                    )
                                    for token, values in zip(
                                        output.prompt_token_ids,
                                        output.prompt_logprobs,
                                        strict=True,
                                    )
                                ]
                                if output.prompt_logprobs is not None
                                else None
                            ),
                            "num_cached_tokens": getattr(
                                output, "num_cached_tokens", None
                            ),
                        }
                        for output in outputs
                    ],
                    "workers": observations,
                    "metrics": {
                        metric.name: float(metric.value)
                        for metric in engine.get_metrics()
                        if isinstance(getattr(metric, "value", None), (int, float))
                    },
                    "image_pixel_sha256": image_hashes,
                }
            )
        return {
            "name": request["name"],
            "engine_options": options,
            "batches": records,
            "workers": engine.collective_rpc("aiter_test_identity", timeout=120),
        }
    finally:
        if engine is not None:
            engine.llm_engine.engine_core.shutdown()
            del engine
        cleanup_dist_env_and_memory()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    parse_request(request)
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        raise RuntimeError("The reviewed controller uses an in-process engine core.")
    from common.paths import assert_package_origin

    assert_package_origin()
    result = generate(request)
    result["request_sha256"] = hashlib.sha256(args.request.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

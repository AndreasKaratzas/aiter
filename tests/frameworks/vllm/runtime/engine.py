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
        "gpu_memory_utilization": settings.gpu_memory_utilization,
        "enable_prefix_caching": settings.prefix_cache,
        "disable_log_stats": False,
        "async_scheduling": False,
        "tensor_parallel_size": settings.tensor_parallel,
        "worker_cls": "frameworks.vllm.runtime.worker.ObservedWorker",
        "kernel_config": {
            "ir_op_priority": {"rms_norm": ["aiter"], "fused_add_rms_norm": ["aiter"]}
        },
        "attention_config": {
            "backend": {
                "unified": "ROCM_AITER_UNIFIED_ATTN",
                "flash": "ROCM_AITER_FA",
                "mla": "ROCM_AITER_MLA",
            }[settings.attention_backend]
        },
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
            "max_pixels": settings.vision_max_pixels,
        }
    if settings.moe:
        options["kernel_config"]["moe_backend"] = settings.moe_backend
    if settings.audio:
        options["limit_mm_per_prompt"] = {"audio": 1}
    return options


def materialize(prompts):
    if isinstance(prompts[0], str) or "prompt_token_ids" in prompts[0]:
        return list(prompts), []
    if "audio_path" in prompts[0]:
        import soundfile

        values, hashes = [], []
        for item in prompts:
            path = Path(item["audio_path"])
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != item["sha256"]:
                raise ValueError("Speech input bytes changed")
            samples, rate = soundfile.read(path, dtype="float32")
            if (
                rate != item["sample_rate"]
                or samples.ndim != 1
                or not 0 < len(samples) <= rate * 30
            ):
                raise ValueError(
                    "Speech input must be mono, 16kHz and at most30 seconds"
                )
            prompt = "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>\n<|im_end|>\n<|im_start|>assistant\n"
            values.append(
                {"prompt": prompt, "multi_modal_data": {"audio": (samples, rate)}}
            )
            hashes.append(
                {
                    "kind": "audio",
                    "source_sha256": digest,
                    "samples": len(samples),
                    "sample_rate": rate,
                }
            )
        return values, hashes
    from PIL import Image

    values, hashes = [], []
    for item in prompts:
        source_digest = None
        if "image_path" in item:
            path = Path(item["image_path"])
            source_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if source_digest != item["sha256"]:
                raise ValueError("Image source bytes changed")
            with Image.open(path) as original:
                image = original.convert("RGB")
        else:
            image = Image.new("RGB", tuple(item["size"]), tuple(item["rgb"]))
        text = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            + item["text"]
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
        values.append({"prompt": text, "multi_modal_data": {"image": image}})
        hashes.append(
            {
                "kind": "image",
                "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                "source_sha256": source_digest,
            }
        )
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
                    "image_pixel_sha256": [
                        record["pixel_sha256"]
                        for record in image_hashes
                        if record["kind"] == "image"
                    ],
                    "media_inputs": image_hashes,
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

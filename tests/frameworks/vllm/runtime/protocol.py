# SPDX-License-Identifier: MIT
"""Bounded engine requests shared by feature tests and the isolated process."""

from dataclasses import asdict, dataclass


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class EngineSettings:
    dtype: str = "bfloat16"
    attention_backend: str = "unified"
    audio: bool = False
    moe: bool = False
    moe_backend: str = "auto"
    gpu_memory_utilization: float = 0.1
    vision_max_pixels: int = 224 * 224
    max_model_len: int = 1024
    tensor_parallel: int = 1
    prefix_cache: bool = False
    cuda_graph: bool = False
    speculative: bool = False
    multimodal: bool = False
    online_fp8: bool = False
    chunked_prefill: bool = True
    prefill_budget: int = 1024

    def __post_init__(self):
        require(
            self.moe_backend in ("auto", "aiter", "aiter_triton_mxfp4_bf16")
            and (self.moe or self.moe_backend == "auto"),
            "Explicit expert backend requires a reviewed MoE scenario",
        )
        require(
            type(self.vision_max_pixels) is int
            and self.vision_max_pixels in (224 * 224, 1024 * 1024),
            "Unknown reviewed vision resolution budget",
        )
        require(
            self.attention_backend in ("unified", "flash", "mla"),
            "Unknown reviewed attention backend",
        )
        require(
            type(self.gpu_memory_utilization) is float
            and 0.1 <= self.gpu_memory_utilization <= 0.8,
            "Invalid bounded GPU memory allocation",
        )
        require(
            self.dtype in ("bfloat16", "float16"),
            "Reviewed model dtype is BF16 or FP16",
        )
        require(
            type(self.max_model_len) is int
            and self.max_model_len in (1024, 4096, 16384),
            "Reviewed context capacity is 1024, 4096 or 16384",
        )
        require(
            type(self.tensor_parallel) is int and self.tensor_parallel in (1, 2),
            "The reviewed engine scenarios use one or two ranks.",
        )
        for name in (
            "prefix_cache",
            "cuda_graph",
            "speculative",
            "multimodal",
            "online_fp8",
            "chunked_prefill",
            "audio",
            "moe",
        ):
            require(type(getattr(self, name)) is bool, f"{name} must be a boolean")
        require(
            type(self.prefill_budget) is int
            and self.prefill_budget in (128, 1024, 4096),
            "Prefill budget must select the reviewed 128, 1024 or 4096 token schedule",
        )
        require(
            self.chunked_prefill or self.prefill_budget >= self.max_model_len,
            "Unchunked prefill must fit the complete context",
        )
        require(
            not (self.multimodal and (self.speculative or self.cuda_graph)),
            "Vision/speculation/graph combinations need a separate reviewed scenario.",
        )
        require(
            not (
                self.audio and (self.multimodal or self.speculative or self.cuda_graph)
            ),
            "Audio is a separate reviewed modality",
        )
        require(
            not (
                self.online_fp8
                and (self.multimodal or self.speculative or self.cuda_graph)
            ),
            "Online FP8 is qualified separately from vision/speculation/graphs",
        )

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Batch:
    name: str
    prompts: tuple
    max_tokens: int = 64
    ignore_eos: bool = True
    reset_prefix_cache: bool = False
    prompt_logprobs: bool = False

    def __post_init__(self):
        require(
            isinstance(self.name, str) and self.name.isidentifier(),
            "Invalid batch name",
        )
        require(
            type(self.prompts) is tuple and 1 <= len(self.prompts) <= 4,
            "A reviewed batch has one to four prompts",
        )
        require(
            type(self.max_tokens) is int and 1 <= self.max_tokens <= 128,
            "Generation must be bounded to 1..128 tokens",
        )
        require(
            type(self.ignore_eos) is bool
            and type(self.reset_prefix_cache) is bool
            and type(self.prompt_logprobs) is bool,
            "Batch options must be booleans",
        )
        for prompt in self.prompts:
            if isinstance(prompt, str):
                require(0 < len(prompt) <= 16384, "Prompt text is empty or too large")
                continue
            if type(prompt) is dict and set(prompt) == {
                "audio_path",
                "sha256",
                "sample_rate",
            }:
                from pathlib import Path
                import re

                require(
                    type(prompt["audio_path"]) is str
                    and Path(prompt["audio_path"]).is_absolute(),
                    "Audio needs an explicit absolute path",
                )
                require(
                    type(prompt["sha256"]) is str
                    and re.fullmatch(r"[a-f0-9]{64}", prompt["sha256"]),
                    "Audio needs an exact byte identity",
                )
                require(
                    prompt["sample_rate"] == 16000
                    and type(prompt["sample_rate"]) is int,
                    "Reviewed speech sampling rate is 16kHz",
                )
                continue
            if type(prompt) is dict and set(prompt) == {"prompt_token_ids"}:
                values = prompt["prompt_token_ids"]
                require(
                    type(values) is list
                    and 1 <= len(values) <= 4096
                    and all(
                        type(value) is int and 0 <= value < 2**31 for value in values
                    ),
                    "Invalid bounded token prompt",
                )
                continue
            if type(prompt) is dict and set(prompt) == {"text", "image_path", "sha256"}:
                from pathlib import Path
                import re

                require(
                    type(prompt["text"]) is str and 0 < len(prompt["text"]) <= 16384,
                    "Invalid image prompt text",
                )
                require(
                    type(prompt["image_path"]) is str
                    and Path(prompt["image_path"]).is_absolute(),
                    "Image needs explicit absolute path",
                )
                require(
                    type(prompt["sha256"]) is str
                    and re.fullmatch(r"[a-f0-9]{64}", prompt["sha256"]),
                    "Image requires exact byte identity",
                )
                continue
            require(
                type(prompt) is dict and set(prompt) == {"text", "rgb", "size"},
                "Image prompts declare exactly text, rgb and size",
            )
            require(
                type(prompt["text"]) is str and 0 < len(prompt["text"]) <= 16384,
                "Invalid image prompt text",
            )
            require(
                type(prompt["rgb"]) is list
                and len(prompt["rgb"]) == 3
                and all(type(v) is int and 0 <= v <= 255 for v in prompt["rgb"]),
                "Image RGB values must be bytes",
            )
            require(prompt["size"] == [224, 224], "Reviewed vision inputs are224x224")

    def to_dict(self):
        return {**asdict(self), "prompts": list(self.prompts)}


def parse_request(record):
    require(
        type(record) is dict
        and set(record) == {"name", "model", "settings", "batches"},
        "Unknown or missing engine request fields",
    )
    require(
        type(record["name"]) is str and record["name"].isidentifier(),
        "Engine request name must be an identifier",
    )
    require(type(record["settings"]) is dict, "Engine settings must be an object")
    settings = EngineSettings(**record["settings"])
    require(
        type(record["batches"]) is list and 1 <= len(record["batches"]) <= 4,
        "Engine request has one to four batches",
    )
    batches = [
        Batch(**{**batch, "prompts": tuple(batch["prompts"])})
        for batch in record["batches"]
    ]
    for batch in batches:
        kinds = {
            "text"
            if isinstance(prompt, str)
            else "tokens"
            if "prompt_token_ids" in prompt
            else "audio"
            if "audio_path" in prompt
            else "image"
            for prompt in batch.prompts
        }
        require(len(kinds) == 1, "A batch must use one prompt representation")
        require(
            all(
                not isinstance(prompt, dict)
                or "prompt_token_ids" not in prompt
                or len(prompt["prompt_token_ids"]) + batch.max_tokens
                <= settings.max_model_len
                for prompt in batch.prompts
            ),
            "Token prompt and output exceed context capacity",
        )
    require(
        len({batch.name for batch in batches}) == len(batches), "Duplicate batch names"
    )
    require(
        all(
            (isinstance(prompt, dict) and ("rgb" in prompt or "image_path" in prompt))
            == settings.multimodal
            for batch in batches
            for prompt in batch.prompts
        ),
        "Prompt modality differs from engine configuration",
    )
    require(
        all(
            (isinstance(prompt, dict) and "audio_path" in prompt) == settings.audio
            for batch in batches
            for prompt in batch.prompts
        ),
        "Audio modality differs from engine configuration",
    )
    require(
        all(not batch.reset_prefix_cache or settings.prefix_cache for batch in batches),
        "Cache reset requires prefix caching",
    )
    return settings, batches

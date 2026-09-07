# SPDX-License-Identifier: MIT
"""Bounded engine requests shared by feature tests and the isolated process."""

from dataclasses import asdict, dataclass


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class EngineSettings:
    dtype: str = "bfloat16"
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
            self.dtype in ("bfloat16", "float16"),
            "Reviewed model dtype is BF16 or FP16",
        )
        require(
            type(self.max_model_len) is int and self.max_model_len in (1024, 4096),
            "Reviewed context capacity is 1024 or 4096",
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
            (isinstance(prompt, dict) and "rgb" in prompt) == settings.multimodal
            for batch in batches
            for prompt in batch.prompts
        ),
        "Prompt modality differs from engine configuration",
    )
    require(
        all(not batch.reset_prefix_cache or settings.prefix_cache for batch in batches),
        "Cache reset requires prefix caching",
    )
    return settings, batches

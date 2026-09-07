# SPDX-License-Identifier: MIT
"""Image content must survive the HTTP API and affect the model's answer."""

import base64
import io

import pytest

from frameworks.vllm.runtime.protocol import EngineSettings
from frameworks.vllm.runtime.server import running_server

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu(min_memory_gib=16),
    pytest.mark.requires_arch("gfx950"),
    pytest.mark.requires_capability("bf16"),
    pytest.mark.framework("vllm"),
    pytest.mark.model("qwen25_vl_3b"),
]


def test_image_chat_uses_pixels_and_aiter(model_snapshot, e2e_evidence, engine_inputs):
    from PIL import Image

    fixture = engine_inputs["multimodal"]
    with running_server(
        model_snapshot,
        e2e_evidence / "vision-server",
        settings=EngineSettings(multimodal=True),
    ) as server:
        answers = []
        for item in fixture["images"]:
            with io.BytesIO() as encoded:
                Image.new(
                    "RGB", tuple(fixture["image_size"]), tuple(item["rgb"])
                ).save(encoded, format="PNG")
                image_url = "data:image/png;base64," + base64.b64encode(
                    encoded.getvalue()
                ).decode("ascii")
            server.reset()
            response = server.request(
                "/v1/chat/completions",
                {
                    "model": "aiter-fixture",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": fixture["prompt"]},
                                {"type": "image_url", "image_url": {"url": image_url}},
                            ],
                        }
                    ],
                    "temperature": 0,
                    "seed": 0,
                    "max_tokens": fixture["max_tokens"],
                },
            )
            answer = response["choices"][0]["message"]["content"]
            answer = answer.strip().rstrip(".! ").lower()
            assert answer == item["name"]
            assert 0 < response["usage"]["completion_tokens"] <= fixture["max_tokens"]
            answers.append(answer)
            observation = server.observe(item["name"] + "_image_chat")
            assert (
                observation["workers"][0]["operations"].get("flash_attn_varlen_func", 0)
                > 0
            ), "Image encoding bypassed AITER vision attention"
        assert len(set(answers)) == len(fixture["images"])

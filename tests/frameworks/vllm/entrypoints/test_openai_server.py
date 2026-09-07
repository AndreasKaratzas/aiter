# SPDX-License-Identifier: MIT
"""The public server transports must preserve generation and recover from bad input."""

import pytest

from frameworks.vllm.runtime.server import running_server, stream_events

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu(min_memory_gib=8),
    pytest.mark.requires_arch("gfx950"),
    pytest.mark.requires_capability("bf16"),
    pytest.mark.framework("vllm"),
    pytest.mark.model("llama32_1b"),
]


def test_completion_stream_matches_complete_response(
    model_snapshot, e2e_evidence, text_prompts
):
    with running_server(model_snapshot, e2e_evidence / "completion-server") as server:
        models = server.request("/v1/models")
        assert [entry["id"] for entry in models["data"]] == ["aiter-fixture"]
        request = {
            "model": "aiter-fixture",
            "prompt": text_prompts[0],
            "max_tokens": 32,
            "temperature": 0,
            "seed": 0,
            "ignore_eos": True,
        }
        complete = server.request("/v1/completions", request)
        chunks = stream_events(
            server.request("/v1/completions", {**request, "stream": True})
        )
        streamed = "".join(
            chunk["choices"][0]["text"] for chunk in chunks if chunk["choices"]
        )
        assert streamed == complete["choices"][0]["text"]
        assert streamed.startswith("red blue green yellow orange\n")
        assert complete["usage"]["completion_tokens"] == 32
        assert chunks[-1]["choices"][0]["finish_reason"] == "length"
        assert len({chunk["id"] for chunk in chunks}) == 1
        server.observe("completion_stream")


def test_chat_stream_and_invalid_request_recovery(model_snapshot, e2e_evidence):
    with running_server(model_snapshot, e2e_evidence / "chat-server") as server:
        server.request(
            "/v1/chat/completions",
            {"model": "aiter-fixture", "messages": []},
            expected_status=400,
        )
        request = {
            "model": "aiter-fixture",
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with only the word Paris: What is the capital of France?",
                }
            ],
            "max_tokens": 12,
            "temperature": 0,
            "seed": 0,
        }
        complete = server.request("/v1/chat/completions", request)
        chunks = stream_events(
            server.request("/v1/chat/completions", {**request, "stream": True})
        )
        text = "".join(
            chunk["choices"][0]["delta"].get("content") or ""
            for chunk in chunks
            if chunk["choices"]
        )
        assert text == complete["choices"][0]["message"]["content"]
        assert text.strip().rstrip(".! ").lower() == "paris"
        assert complete["usage"]["completion_tokens"] > 0
        assert (
            chunks[-1]["choices"][0]["finish_reason"]
            == complete["choices"][0]["finish_reason"]
        )
        server.request("/health")
        server.observe("chat_recovery")

# SPDX-License-Identifier: MIT
"""The public server transports must preserve generation and recover from bad input."""

import math

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


def test_batched_completion_accounting_and_logprobs(
    model_snapshot, e2e_evidence, text_prompts
):
    """A batched API call keeps prompt order, token totals and per-token scores."""
    with running_server(model_snapshot, e2e_evidence / "batch-server") as server:
        server.request(
            "/v1/completions",
            {"model": "unknown-model", "prompt": text_prompts[0]},
            expected_status=404,
        )
        prompts = list(text_prompts[:2])
        assert len(prompts) == 2 and prompts[0] != prompts[1]
        options = {
            "model": "aiter-fixture",
            "max_tokens": 16,
            "temperature": 0,
            "seed": 0,
            "ignore_eos": True,
            "logprobs": 2,
        }
        singles = [
            server.request("/v1/completions", {**options, "prompt": prompt})
            for prompt in prompts
        ]
        server.reset()
        batched = server.request("/v1/completions", {**options, "prompt": prompts})
        assert [choice["index"] for choice in batched["choices"]] == [0, 1]
        for choice, single in zip(batched["choices"], singles, strict=True):
            assert choice["text"] == single["choices"][0]["text"]
            assert choice["finish_reason"] == "length"
            scores = choice["logprobs"]
            assert len(scores["tokens"]) == 16
            assert len(scores["token_logprobs"]) == len(scores["top_logprobs"]) == 16
            assert all(math.isfinite(value) for value in scores["token_logprobs"])
            assert all(
                candidates and all(math.isfinite(value) for value in candidates.values())
                for candidates in scores["top_logprobs"]
            )
        usage = batched["usage"]
        assert usage["completion_tokens"] == 32
        assert usage["prompt_tokens"] == sum(
            single["usage"]["prompt_tokens"] for single in singles
        )
        assert usage["total_tokens"] == usage["prompt_tokens"] + 32
        server.observe("batched_logprobs")

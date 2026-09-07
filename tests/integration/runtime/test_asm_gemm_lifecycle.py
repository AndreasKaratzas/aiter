# SPDX-License-Identifier: MIT
"""Independent stream, large-tile and storage checks for assembly GEMM."""

import pytest
from common.paths import assert_package_origin


def tensors(m, n, k):
    import torch

    assert_package_origin()
    torch.manual_seed(917)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    return x, weight


@pytest.mark.parametrize("shuffled", (False, True))
def test_assembly_gemm_independent_streams_and_graph_replays(shuffled):
    import torch

    import aiter
    from aiter.ops.shuffle import shuffle_weight

    entries = []
    for index in range(2):
        x, weight = tensors(64, 256, 5120)
        x.mul_(index + 1)
        bias = torch.randn(256, device="cuda", dtype=torch.bfloat16)
        selected = shuffle_weight(weight, layout=(16, 16)) if shuffled else weight
        out = torch.empty(64, 256, device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            aiter.gemm_a16w16_asm(x, selected, out, bias, 4, None, shuffled)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            aiter.gemm_a16w16_asm(x, selected, out, bias, 4, None, shuffled)
        # Graphs retain device addresses; keep the caller's shuffled weight
        # alive through every replay as well as the unshuffled reference.
        entries.append((x, weight, bias, out, stream, graph, selected))

    for _ in range(3):
        for x, _, bias, _, _, _, _ in entries:
            x.mul_(0.5)
            bias.add_(0.125)
        for _, _, _, _, stream, graph, _ in entries:
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graph.replay()
        torch.cuda.synchronize()
        for x, weight, bias, out, _, _, _ in entries:
            expected = (x.double() @ weight.double().T + bias.double()).to(out.dtype)
            torch.testing.assert_close(out, expected, rtol=0.01, atol=0.01)


@pytest.mark.parametrize("shuffled", (False, True))
def test_large_bf16_only_tile_cannot_write_bf16_into_fp32_output(shuffled):
    import torch

    import aiter
    from aiter.ops.shuffle import shuffle_weight

    x, weight = tensors(256, 256, 512)
    selected = shuffle_weight(weight, layout=(16, 16)) if shuffled else weight
    name = (
        "_ZN5aiter36bf16gemm_bf16_tn_256x256_bpreshuffleE"
        if shuffled
        else "_ZN5aiter24bf16gemm_bf16_tn_256x256E"
    )
    expected = x.double() @ weight.double().T
    out = torch.full((256, 256), float("nan"), device="cuda", dtype=torch.bfloat16)
    aiter.gemm_a16w16_asm(x, selected, out, None, 1, name, shuffled)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected.to(out.dtype), rtol=0.01, atol=0.01)

    precise = torch.full_like(out, float("nan"), dtype=torch.float32)
    with pytest.raises((RuntimeError, ValueError)):
        aiter.gemm_a16w16_asm(x, selected, precise, None, 1, name, shuffled)
    assert torch.isnan(precise).all(), "An incompatible explicit kernel wrote output."
    aiter.gemm_a16w16_asm(x, selected, precise, bpreshuffle=shuffled)
    torch.cuda.synchronize()
    torch.testing.assert_close(precise, expected.float(), rtol=0.0002, atol=0.0005)


@pytest.mark.parametrize("invalid", ("transposed-output", "input-alias", "bias-alias"))
def test_assembly_gemm_rejects_invalid_storage_before_writing(invalid):
    import torch

    import aiter

    x, weight = tensors(64, 256, 512)
    bias = None
    if invalid == "transposed-output":
        out = torch.randn(256, 64, device="cuda", dtype=torch.float32).T
    elif invalid == "input-alias":
        out = x.view(torch.float32)
    else:
        out = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
        bias = out[0]
    original = out.clone()
    with pytest.raises((RuntimeError, ValueError)):
        aiter.gemm_a16w16_asm(x, weight, out, bias)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, original, rtol=0, atol=0, equal_nan=True)

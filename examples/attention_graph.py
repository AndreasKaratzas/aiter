# SPDX-License-Identifier: MIT
"""Capture rotary Q/K and causal grouped-query attention as one graph."""


def main():
    import torch

    from aiter.runtime import Runtime

    torch.manual_seed(13)
    runtime = Runtime(device=0)
    q = torch.randn(33, 1, 4, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(33, 1, 2, 64, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    angles = torch.randn(33, 1, 1, 32, device="cuda")
    qr, kr, out = torch.empty_like(q), torch.empty_like(k), torch.empty_like(q)
    lse = torch.empty(1, 4, 33, device="cuda")
    rotate_q = runtime.prepare_rope(q, angles, qr, backend="triton")
    rotate_k = runtime.prepare_rope(k, angles, kr, backend="triton")
    attention = runtime.prepare_attention(
        qr, kr, v, out, lse, causal=True, backend="triton"
    )
    # Preparation compiles and warms the leaf. These are the first executions
    # of these plans; graph capture records their fixed launches.
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rotate_q.execute({"x": q, "freqs": angles, "out": qr})
        rotate_k.execute({"x": k, "freqs": angles, "out": kr})
        attention.execute({"q": qr, "k": kr, "v": v, "out": out, "lse": lse})

    def rotate(x):
        left, right = x.float().chunk(2, dim=-1)
        return torch.cat(
            (
                left * angles.cos() - right * angles.sin(),
                right * angles.cos() + left * angles.sin(),
            ),
            dim=-1,
        ).to(x.dtype)

    for factor in (1.0, 0.5, -1.0):
        q.mul_(factor)
        graph.replay()
        expected_q, expected_k = rotate(q), rotate(k)
        torch.testing.assert_close(qr, expected_q, atol=0.02, rtol=0.01)
        torch.testing.assert_close(kr, expected_k, atol=0.02, rtol=0.01)
        query = expected_q.permute(1, 2, 0, 3).float()
        key = expected_k.permute(1, 2, 0, 3).repeat_interleave(2, dim=1).float()
        value = v.permute(1, 2, 0, 3).repeat_interleave(2, dim=1).float()
        scores = query @ key.transpose(-1, -2) / 8
        scores.masked_fill_(
            torch.ones(33, 33, device=q.device, dtype=torch.bool).triu(1), -float("inf")
        )
        expected = (scores.softmax(-1) @ value).permute(2, 0, 1, 3).to(out.dtype)
        torch.testing.assert_close(out, expected, atol=0.03, rtol=0.02)
        torch.testing.assert_close(lse, scores.logsumexp(-1), atol=0.02, rtol=0.01)
    print("RoPE → causal GQA passed three changing-input graph replays.")


if __name__ == "__main__":
    main()

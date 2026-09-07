# SPDX-License-Identifier: MIT
"""Shared utils.mhc_ref workload generation and numerical references."""

import torch


def mhc_torch(
    x: torch.Tensor,
    phi: torch.Tensor,  # (K, n + n + n_res)
    alpha_pre: float,
    alpha_post: float,
    alpha_res: float,
    bias: torch.Tensor,
    n: int,
    eps: float = 1e-6,
    hc_pre_eps: float = 0.0,
    sinkhorn_iters: int = 20,
    return_with_sinkhorn: bool = True,
    asymmetric_exp_domain: bool = False,
    hc_sinkhorn_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single PyTorch reference for mHC (Eq 14-19) plus the apply-pre fusion step,
    exposing every meaningful intermediate so each test picks what it needs.

    The Triton ``mhc()`` wrapper (and HIP ``aiter.mhc_pre``) only return three
    of the four outputs (``hpost``, ``hres``, ``layer_input``); ``H^pre`` is
    consumed in-kernel. This ref additionally returns the raw pre-sigmoid
    ``hpre`` so tests that want to assert on it (or re-derive ``pre_mix``
    via ``σ(hpre) + hc_pre_eps``) can do so

    Implements:
    - Eq 14: H̃ = x̃φ (matrix multiplication)
    - Eq 15: r = ||x̃||₂ / √(nC) (RMS normalization)
    - Eq 16: [H^pre, H^post, H^res] = 1/r [α^pre·H̃^pre, α^post·H̃^post, α^res·H̃^res] + b (scaling)
    - Eq 17: H^pre = σ(H^pre) + hc_pre_eps; folded into
      ``layer_input[m, c] = Σᵢ pre_mix[m, i] · x[m, i*C + c]`` (apply-pre).
    - Eq 18: H^post = 2σ(H^post) (scaled sigmoid activation for post-stream)
    - Eq 19: H^res = Sinkhorn(H^res) (project residual stream onto doubly stochastic manifold)

    Args:
        x: (M, K) input tensor where K = n × C (flattened n-stream residual)
        phi: (K, N) unified projection matrix where N = n + n + n²
            Layout: [pre: 0..n-1, post: n..2n-1, res: 2n..2n+n²-1]
        alpha_pre: α^pre scaling factor for pre-stream (Eq 12)
        alpha_post: α^post scaling factor for post-stream (Eq 12)
        alpha_res: α^res scaling factor for residual stream (Eq 12)
        bias: (N,) bias vector where N = n + n + n²
        n: stream parameter (manifold dimension controller)
        eps: epsilon for RMS normalization stability
        hc_pre_eps: additive epsilon on σ(H^pre) before the apply-pre fold.
            Note: ``hpre`` is returned **before** sigmoid and **before** this
            additive eps; the eps only affects ``layer_input``.
        sinkhorn_iters: number of Sinkhorn-Knopp iterations for H^res (Eq 19)
        return_with_sinkhorn: if True, apply Sinkhorn-Knopp to H_res; else
            return raw H^res logits as ``hres`` (matches ``mhc(..., sinkhorn_iters=0)``
            which skips Eq 19).
        asymmetric_exp_domain: select the Sinkhorn variant matching Triton
            ``mhc_post_pre(..., asymmetric_exp_domain=...)``. When False
            (default), uses canonical log-domain Sinkhorn-Knopp. When True,
            uses the HIP-compatible asymmetric exp-domain variant (first iter
            ``softmax(row) + eps`` then ``div(col + eps)``, followed by
            ``sinkhorn_iters - 1`` symmetric ``div(row + eps)/div(col + eps)``
            iters); see ``sinkhorn_knopp_asymmetric_exp_domain_torch``.
        hc_sinkhorn_eps: eps used by the asymmetric exp-domain Sinkhorn;
            ignored when ``asymmetric_exp_domain`` is False.

    Returns:
        Tuple ``(hpost, hres, hpre, layer_input)`` all in fp32:
        - ``hpost``: (M, n, 1) — ``2σ(H^post)`` (Eq 18, unsqueezed to match ``mhc()``).
        - ``hres``: (M, n, n) — Sinkhorn-projected H^res (or raw logits when
          ``return_with_sinkhorn=False``).
        - ``hpre``: (M, n) — **raw H^pre logits**, post-Eq-16, pre-Eq-17. Apply
          ``σ`` and ``+hc_pre_eps`` to recover ``pre_mix`` if needed.
        - ``layer_input``: (M, C) — apply-pre fused output, Σᵢ pre_mix[i] · x_i.
    """
    # Extract individual phi components from unified tensor
    phi_pre = phi[:, :n]
    phi_post = phi[:, n : 2 * n]
    phi_res = phi[:, 2 * n :]
    x_f32 = x.to(torch.float32)

    # Eq 15: r = ||x̃||₂ / √(nC)
    mean_sq = torch.mean(x_f32**2, dim=-1, keepdim=True)
    rms = torch.sqrt(mean_sq + eps)
    x_norm = x_f32 / rms

    # Eq 14: H̃ = x̃φ - compute each stream separately
    phi_pre_f32 = phi_pre.to(torch.float32)
    phi_post_f32 = phi_post.to(torch.float32)
    phi_res_f32 = phi_res.to(torch.float32)

    H_tilde_pre = x_norm @ phi_pre_f32  # (M, n)
    H_tilde_post = x_norm @ phi_post_f32  # (M, n)
    H_tilde_res = x_norm @ phi_res_f32  # (M, n²)

    # Split bias
    bias_f32 = bias.to(torch.float32)
    bias_pre = bias_f32[:n]
    bias_post = bias_f32[n : 2 * n]
    bias_res = bias_f32[2 * n :]

    # Eq 16: Apply stream-specific scaling and bias
    H_pre = alpha_pre * H_tilde_pre + bias_pre
    H_post = alpha_post * H_tilde_post + bias_post
    H_res = alpha_res * H_tilde_res + bias_res

    # Eq 17 + apply-pre fold: pre_mix = σ(H_pre) + hc_pre_eps, then
    # layer_input[m, c] = Σᵢ pre_mix[m, i] * x[m, i*C + c].
    pre_mix = torch.sigmoid(H_pre) + hc_pre_eps  # (M, n)
    M, K = x.shape
    C = K // n
    x_3d = x_f32.view(M, n, C)  # (M, n, C)
    layer_input = torch.einsum("mn,mnc->mc", pre_mix, x_3d)  # (M, C)

    # Eq 18: hpost = 2σ(H^post), reshaped to (M, n, 1) to match mhc()'s output.
    hpost = (2.0 * torch.sigmoid(H_post)).unsqueeze(-1)  # (M, n, 1) fp32

    # Eq 19: Apply Sinkhorn-Knopp to H^res for doubly stochastic constraint.
    H_res_3d = H_res.view(M, n, n)
    if return_with_sinkhorn:
        if asymmetric_exp_domain:
            hres = sinkhorn_knopp_asymmetric_exp_domain_torch(
                H_res_3d, num_iters=sinkhorn_iters, eps=hc_sinkhorn_eps
            )
        else:
            hres = sinkhorn_knopp_log_domain_torch(H_res_3d, num_iters=sinkhorn_iters)
    else:
        hres = H_res_3d  # already fp32

    return hpost, hres, H_pre, layer_input


def sinkhorn_knopp_asymmetric_exp_domain_torch(
    logits: torch.Tensor,
    num_iters: int = 20,
    eps: float = 1e-6,
) -> torch.Tensor:
    """HIP-compatible asymmetric exp-domain Sinkhorn-Knopp reference.

    Bit-for-bit mirror of the Triton ``ASYMMETRIC_EXP_DOMAIN`` branch
    (``_triton_kernels/fusions/mhc.py``: ``_mhc_post_pre_reduce_apply_res_block``)
    and the HIP kernel (``csrc/kernels/mhc_kernels.cu``):

      - First iteration is asymmetric: ``P = softmax_row(logits)`` (i.e.
        ``exp(x - row_max) / row_sum``) then ``+ eps``, followed by
        ``P /= (col_sum + eps)``.
      - The remaining ``num_iters - 1`` iterations are symmetric:
        ``P /= (row_sum + eps)`` then ``P /= (col_sum + eps)``.

    Row direction is the last dim (sum over columns of a row); column
    direction is dim ``-2`` (sum over rows of a column), matching the kernels'
    ``axis=2`` / ``axis=1`` reductions on the ``(M, n, n)`` tile.

    Args:
        logits: (M, N, N) raw H^res logits.
        num_iters: total Sinkhorn iterations (>= 1 expected; ``0`` returns the
            softmax-free input unchanged, matching the kernels' skip path).
        eps: ``hc_sinkhorn_eps`` perturbation.

    Returns:
        (M, N, N) tensor cast back to ``logits.dtype``.
    """
    num_iters = int(num_iters)
    if num_iters <= 0:
        return logits

    A = logits.to(torch.float32)

    # First iter (asymmetric): softmax over the row (last dim) + eps, then
    # divide by (col_sum + eps).
    row_max = A.amax(dim=-1, keepdim=True)  # (M, N, 1)
    P = torch.exp(A - row_max)
    row_sum = P.sum(dim=-1, keepdim=True)  # (M, N, 1)
    P = P / row_sum + eps
    col_sum = P.sum(dim=-2, keepdim=True)  # (M, 1, N)
    P = P / (col_sum + eps)

    # Remaining iters (symmetric): div(row + eps) then div(col + eps).
    for _ in range(num_iters - 1):
        row_sum = P.sum(dim=-1, keepdim=True)
        P = P / (row_sum + eps)
        col_sum = P.sum(dim=-2, keepdim=True)
        P = P / (col_sum + eps)

    return P.to(logits.dtype)


def sinkhorn_knopp_log_domain_torch(
    logits: torch.Tensor,
    num_iters: int = 10,
) -> torch.Tensor:
    """
    PyTorch reference implementation of Sinkhorn-Knopp in log domain.

    Returns:
        Doubly stochastic matrices with shape (M, N, N)
    """
    num_iters = int(num_iters)  # Ensure int for range()
    M, N, _ = logits.shape

    log_A = logits.to(torch.float32)

    # Initialize log scaling factors (log(1) = 0, so no initial scaling)
    log_u = torch.zeros(M, N, device=logits.device, dtype=torch.float32)
    log_v = torch.zeros(M, N, device=logits.device, dtype=torch.float32)

    for _ in range(num_iters):
        # Row normalization in log domain:
        # log_u[i] = -logsumexp_j(log_A[i,j] + log_v[j])
        scaled = log_A + log_v.unsqueeze(1)  # (M, N, N)
        log_row_sums = torch.logsumexp(scaled, dim=2)  # (M, N)
        log_u = -log_row_sums

        # Column normalization in log domain:
        # log_v[j] = -logsumexp_i(log_A[i,j] + log_u[i])
        scaled = log_A + log_u.unsqueeze(2)  # (M, N, N)
        log_col_sums = torch.logsumexp(scaled, dim=1)  # (M, N)
        log_v = -log_col_sums

    # Compute final matrix: P = exp(log_A + log_u + log_v)
    log_P = log_A + log_u.unsqueeze(2) + log_v.unsqueeze(1)
    P = torch.exp(log_P)

    return P.to(logits.dtype)


def generate_mhc_inputs(
    M: int,
    n: int,
    C: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """
    Generate test inputs for mHC mapping.

    Returns:
        Tuple of (x, phi, alpha_pre, alpha_post, alpha_res, bias, n) where:
        - x: (M, nC) flattened n-stream residual input
        - phi: (nC, n + n + n²) unified projection matrix [pre | post | res]
        - alpha_pre: α^pre scaling factor for pre-stream (Eq 12)
        - alpha_post: α^post scaling factor for post-stream (Eq 12)
        - alpha_res: α^res scaling factor for residual stream (Eq 12)
        - bias: (n² + 2n,) bias vector
        - n: stream parameter (returned for convenience)
    """

    nC = n * C  # Total flattened dimension

    n_res = n * n  # n² columns

    N_total = n_res + 2 * n

    # flattened n-stream residual
    x = torch.randn(M, nC, dtype=dtype, device=device)

    # Unified projection matrix (nC, n + n + n_res)
    # Layout: [pre: 0..n-1, post: n..2n-1, res: 2n..2n+n_res-1]
    phi = torch.randn(nC, N_total, dtype=dtype, device=device) * 0.1

    # scaling factors (Eq 12)
    alpha_pre = 0.5 + torch.rand(1).item()
    alpha_post = 0.5 + torch.rand(1).item()
    alpha_res = 0.5 + torch.rand(1).item()

    # bias (Eq 13)
    bias = torch.randn(N_total, dtype=torch.float32, device=device) * 0.1

    return x, phi, alpha_pre, alpha_post, alpha_res, bias, n


def mhc_post_torch(
    layer_input: torch.Tensor,  # (M, C)
    residual: torch.Tensor,  # (M, n, C)
    post_mix: torch.Tensor,  # (M, n) or (M, n, 1)
    comb_mix: torch.Tensor,  # (M, n, n)  [src, dst]
) -> torch.Tensor:
    """PyTorch reference for the mhc_post step.

    Computes the updated multi-stream residual by mixing the transformer
    output ``layer_input`` with the previous-layer residual streams:

        out[m, j, c] = post_mix[m, j] * layer_input[m, c]
                     + sum_h comb_mix[m, h, j] * residual[m, h, c]

    ``post_mix`` may be passed as ``(M, n, 1)`` (the layout produced by
    ``mhc()``) and is squeezed internally.

    Reference: arXiv:2512.24880.
    """
    post_mix = post_mix.squeeze(-1) if post_mix.ndim == 3 else post_mix
    x_f32 = layer_input.float()
    res_f32 = residual.float()

    # Term 1: post_mix * layer_input broadcast across dst stream j -> (M, n, C)
    out = post_mix[:, :, None] * x_f32[:, None, :]

    # Term 2: contract over src dim h: einsum "mhc,mhj -> mjc"
    out = out + torch.einsum("mhc,mhj->mjc", res_f32, comb_mix.float())

    return out.to(layer_input.dtype)


def mhc_e2e_ref(
    x_l,
    phi,
    alpha_pre,
    alpha_post,
    alpha_res,
    bias,
    n,
    eps=1e-6,
    hc_pre_eps=0.0,
    hc_post_mult_value=2.0,
    sinkhorn_iters=20,
):
    """
    Reference implementation using PyTorch

    # =============================================================================
    # End-to-End Pipeline Tests (mhc → mhc_post)
    # =============================================================================
    #
    # Pipeline Overview:
    #
    # x_l (M, n, C)  ──┐
    #                  │
    #                  ├──> mhc() ──> (h_post, h_res, layer_input)
    #                  │                                 │
    #                  │                                 ├──> layer_input (M, C)
    #                  │                                 │
    #                  │                                 v
    #                  └──────────────────────────> mhc_post() ──> x_l+1 (M, n, C)
    #                                                   │
    #                                                   └──> Uses (layer_input, x_l, h_post, h_res)

    Pipeline:
    x_l (M, n, C) → flatten → mhc → (h_post, h_res, layer_input) → mhc_post → x_l+1 (M, n, C)
    """
    sinkhorn_iters = int(sinkhorn_iters)
    M, n_check, C = x_l.shape
    assert n_check == n, f"Stream count mismatch: {n_check} != {n}"
    assert hc_post_mult_value == 2.0, (
        "mhc_torch hardcodes 2.0 * sigmoid(H_post); "
        "non-default hc_post_mult_value is not supported in the reference"
    )

    x_l_flat = x_l.view(M, n * C)

    # Step 1: mhc - compute coefficients and layer_input
    h_post, h_res, _h_pre, layer_input = mhc_torch(
        x_l_flat,
        phi,
        alpha_pre,
        alpha_post,
        alpha_res,
        bias,
        n,
        eps=eps,
        hc_pre_eps=hc_pre_eps,
        sinkhorn_iters=sinkhorn_iters,
    )

    # Step 2: mhc_post - merge layer_input back to multi-stream
    x_l_plus_1 = mhc_post_torch(layer_input, x_l, h_post, h_res)

    return layer_input, x_l_plus_1, h_post, h_res

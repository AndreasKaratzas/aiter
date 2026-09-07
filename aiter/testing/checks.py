# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import torch

from aiter import logger

_CATASTROPHIC_REL_THRESHOLD = 0.5


def _relmag_catastrophic(actual_max_delta, b):
    """Relative-magnitude catastrophic heuristic.

    Triggers when ``max(|a - b|) > ref_abs_max * 0.5`` -- i.e. a single
    element diverges by more than half of the reference tensor's peak
    magnitude. Designed to catch real precision regressions in kernels that
    write plausible-looking but wrong values to specific positions (e.g.
    bpreshuffle precision drift, wrong scale/quant, half-broken pipeline).

    By contract, ``catastrophic_check=True`` is opt-in: the caller asserts
    the comparison is *position-sensitive* (no sort/ties permutation
    semantics). For position-insensitive data (sorted topk_ids, sort+gather
    weights with degenerate scores, byte-viewed fp4) this heuristic would
    misfire, so callers MUST NOT enable it there.

    For non-floating-point tensors this returns False -- there is no
    meaningful "magnitude" notion for integer indices/IDs. Callers who want
    a hard cap on integer deltas can still use explicit ``max_abs_delta``.
    """
    if not b.is_floating_point():
        return False
    ref_abs_max = max(b.abs().max().item(), 1.0)
    return actual_max_delta > ref_abs_max * _CATASTROPHIC_REL_THRESHOLD


def _check_catastrophic(actual_max_delta, a, b, max_abs_delta, catastrophic_check):
    """Decide whether a checkAllclose mismatch is "catastrophic" (fail-fast).

    Priority order (returns True at the first hit):

    1. Explicit ``max_abs_delta`` -- opt-in hard cap, takes precedence over
       the relative heuristic for callers that know the acceptable absolute
       magnitude.
    2. ``catastrophic_check=True`` -- enables NaN/Inf detection and the
       relative-magnitude heuristic (delta > ref_max * 0.5). NaN/Inf in
       either tensor is catastrophic (covers tuner NaN sentinel and
       numerically blown-up kernels). Do NOT enable on data that may
       legitimately contain NaN in padding regions.
    3. Otherwise: not catastrophic. The caller gets ``err_ratio`` back via
       the normal return value and decides what to do with it.

    ``torch.isfinite`` is safe on integer / byte tensors (returns all True),
    so this function works uniformly across dtypes.
    """
    if max_abs_delta is not None:
        return actual_max_delta > max_abs_delta
    if catastrophic_check:
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            return True
        return _relmag_catastrophic(actual_max_delta, b)
    return False


def _catastrophic_check_silent(a, b, max_abs_delta, catastrophic_check):
    """Same policy as ``_check_catastrophic`` but without an already-computed
    ``actual_max_delta``. Used by the not-printLog (tuner) fast path so we
    avoid materialising masked tensors when ``isclose`` already failed."""
    if max_abs_delta is not None:
        return (a - b).abs().max().item() > max_abs_delta
    if catastrophic_check:
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            return True
        return _relmag_catastrophic((a - b).abs().max().item(), b)
    return False


def checkAllclose(
    a,
    b,
    rtol=1e-2,
    atol=1e-2,
    tol_err_ratio=0.05,
    msg="",
    printNum=8,
    printLog=True,
    max_abs_delta=None,
    catastrophic_check=False,
    mask=None,
):
    isClose = torch.isclose(a, b, rtol=rtol, atol=atol)
    # mask (bool, broadcastable to a/b): True = compare, False = ignore.
    # Error ratio is taken over the checked elements only.
    if mask is not None:
        mask = mask.to(device=isClose.device, dtype=torch.bool).broadcast_to(
            isClose.shape
        )
        isClose = isClose | ~mask
        denom = int(mask.sum().item())
        if denom == 0:
            if printLog:
                logger.info(
                    f"{msg}[checkAllclose {atol=} {rtol=} "
                    f"\033[33mskipped: empty mask\033[0m]"
                )
            return 0
    else:
        denom = a.numel()

    if isClose.all():
        if printLog:
            logger.info(f"{msg}[checkAllclose {atol=} {rtol=} \033[32mpassed~\033[0m]")
        return 0
    else:
        try:
            mismatch = ~isClose
            num = int(mismatch.sum().item())
            printNum = min(printNum, num)
            percent = num / denom
            if not printLog:
                if percent >= tol_err_ratio:
                    return percent
                is_cat = _catastrophic_check_silent(
                    a, b, max_abs_delta, catastrophic_check
                )
                return 1.0 if is_cat else percent
            a_msked = a[mismatch]
            b_msked = b[mismatch]
            delta = (a_msked - b_msked).abs()
        except RuntimeError:
            a, b = a.to("cpu"), b.to("cpu")
            mismatch = ~isClose.to("cpu")
            num = int(mismatch.sum().item())
            printNum = min(printNum, num)
            percent = num / denom
            if not printLog:
                if percent >= tol_err_ratio:
                    return percent
                is_cat = _catastrophic_check_silent(
                    a, b, max_abs_delta, catastrophic_check
                )
                return 1.0 if is_cat else percent
            a_msked = a[mismatch]
            b_msked = b[mismatch]
            delta = (a_msked - b_msked).abs()

        actual_max_delta = delta.max().item()
        is_catastrophic = _check_catastrophic(
            actual_max_delta, a, b, max_abs_delta, catastrophic_check
        )

        if is_catastrophic:
            logger.info(
                f"""{msg}[checkAllclose {atol=} {rtol=} \033[31mcatastrophic!\033[0m] max abs delta {actual_max_delta:.4f}
    a    : {a.shape}
           {a_msked[:printNum]}
    b    : {b.shape}
           {b_msked[:printNum]}
    delta:
           {delta[:printNum]}"""
            )
        elif percent > tol_err_ratio:
            logger.info(f"""{msg}[checkAllclose {atol=} {rtol=} \033[31mfailed!\033[0m]
    a    : {a.shape}
           {a_msked[:printNum]}
    b    : {b.shape}
           {b_msked[:printNum]}
    delta:
           {delta[:printNum]}""")
        else:
            logger.info(
                f"""{msg}[checkAllclose {atol=} {rtol=} \033[33mwarning!\033[0m] a and b results are not all close"""
            )
        logger.info(
            f"-->max abs delta:{delta.max()}, delta details: {percent:.1%} ({num} of {denom}) elements"
        )
        if is_catastrophic:
            raise AssertionError(
                f"{msg}catastrophic error: max abs delta {actual_max_delta:.4f}, "
                f"{percent:.1%} ({num} of {denom}) elements mismatch"
            )
        return percent

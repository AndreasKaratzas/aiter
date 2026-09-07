# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared compilation lifecycle for providers using Triton's compiled launcher."""

import hashlib


def compile_kernel(leaf, args, constants, grid, *, num_warps=4):
    # All specialization and launch constants are fixed during preparation.
    # The returned launcher bypasses JITFunction.run and heuristic decorators.
    while not hasattr(leaf, "arg_names"):
        leaf = leaf.fn
    compiled = leaf.warmup(*args, grid=grid, **constants, num_warps=num_warps)
    runner = compiled[grid]
    values = dict(zip(leaf.arg_names, args))
    values.update(constants)
    values["num_warps"] = num_warps
    tail = tuple(values[name] for name in leaf.arg_names[len(args) :])
    digest = hashlib.sha256(compiled.asm["hsaco"]).hexdigest()
    return compiled, runner, tail, digest

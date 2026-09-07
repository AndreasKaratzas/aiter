# SPDX-License-Identifier: MIT
"""Profile explicit configurations for batched_gemm_a16w16."""


def main():
    import sys

    from ..parameters import driver_help

    if driver_help(sys.argv):
        return 0

    import torch
    import triton

    from aiter.ops.triton.gemm.batched.batched_gemm_bf16 import batched_gemm_bf16
    from aiter.tuning.search.workloads.gemm.batched.batched_gemm_bf16 import (
        generate_batched_gemm_a16w16_inputs,
    )

    from ..measurement import run_profile
    from ..parameters import (
        get_input_shape_and_config_list,
    )

    input_shape, config_list = get_input_shape_and_config_list(sys.argv, shape_size=3)

    dtype = torch.bfloat16
    M, N, K = input_shape
    # Batch size is hard coded for now.
    B = 8 if K == 4096 else 16
    x, weight, bias, y = generate_batched_gemm_a16w16_inputs(
        B,
        M,
        N,
        K,
        dtype,
        output=True,
    )

    for config in config_list:
        if config is not None:
            config = config.copy()
            config["SPLITK_BLOCK_SIZE"] = triton.cdiv(
                input_shape[2], config["NUM_KSPLIT"]
            )

        def fn(config=config):
            ############################################################
            # <run API>
            batched_gemm_bf16(x, weight, bias, dtype, YQ=y, config=config)
            ############################################################

        run_profile(fn)


if __name__ == "__main__":
    main()

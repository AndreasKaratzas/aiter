# SPDX-License-Identifier: MIT
"""Profile explicit configurations for gemm_a8wfp4."""


def main():
    import sys

    from ..parameters import driver_help

    if driver_help(sys.argv):
        return 0

    import torch
    import triton

    from aiter.ops.triton.gemm.basic.gemm_a8wfp4 import gemm_a8wfp4
    from aiter.ops.triton.utils.types import get_fp8_dtypes
    from aiter.tuning.search.workloads.gemm.basic.gemm_a8wfp4 import (
        generate_gemm_a8wfp4_inputs,
    )

    from ..measurement import run_profile
    from ..parameters import (
        get_input_shape_and_config_list,
    )

    input_shape, config_list = get_input_shape_and_config_list(sys.argv, shape_size=3)
    M, N, K = input_shape

    _, e4m3_type = get_fp8_dtypes()
    dtype = torch.float16
    # Returns: (x, w, x_scales, w_scales, x_fp32, w_fp32, y)
    x, w, x_scales, w_scales, _, _, y = generate_gemm_a8wfp4_inputs(
        M,
        N,
        K,
        e4m3_type,
        dtype,
        layout="TN",
        output=True,
    )

    for config in config_list:
        if config is not None:
            config = config.copy()
            config["SPLITK_BLOCK_SIZE"] = triton.cdiv(K, config["NUM_KSPLIT"])

        def fn(config=config):
            ############################################################
            # <run API>
            gemm_a8wfp4(x, w, y, x_scales, w_scales, dtype, config=config)
            ############################################################

        run_profile(fn)


if __name__ == "__main__":
    main()

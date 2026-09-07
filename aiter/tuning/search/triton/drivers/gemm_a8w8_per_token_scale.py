# SPDX-License-Identifier: MIT
"""Profile explicit configurations for gemm_a8w8_per_token_scale."""


def main():
    import sys

    from ..parameters import driver_help

    if driver_help(sys.argv):
        return 0

    import torch

    from aiter.ops.triton.gemm.basic.gemm_a8w8_per_token_scale import (
        gemm_a8w8_per_token_scale,
    )
    from aiter.tuning.search.workloads.gemm.basic.gemm_a8w8_per_token_scale import (
        generate_gemm_a8w8_per_token_scale_inputs,
    )

    from ..measurement import run_profile
    from ..parameters import (
        get_input_shape_and_config_list,
    )

    input_shape, config_list = get_input_shape_and_config_list(sys.argv, shape_size=3)

    dtype = torch.bfloat16
    x, weight, x_scale, w_scale, y = generate_gemm_a8w8_per_token_scale_inputs(
        *input_shape,
        dtype=dtype,
        layout="TN",
        output=True,
    )

    for config in config_list:

        def fn(config=config):
            ############################################################
            # <run API>
            gemm_a8w8_per_token_scale(
                x, weight, x_scale, w_scale, dtype, y, config=config
            )
            ############################################################

        run_profile(fn)


if __name__ == "__main__":
    main()

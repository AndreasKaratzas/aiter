# SPDX-License-Identifier: MIT
"""Profile explicit configurations for gemm_a8w8_blockscale_preshuffle."""


def main():
    import sys

    from ..parameters import driver_help

    if driver_help(sys.argv):
        return 0

    import torch

    from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
        gemm_a8w8_blockscale_preshuffle,
    )
    from aiter.tuning.search.workloads.gemm.basic.gemm_a8w8_blockscale import (
        generate_gemm_a8w8_blockscale_inputs,
    )

    from ..measurement import run_profile
    from ..parameters import (
        get_input_shape_and_config_list,
    )

    input_shape, config_list = get_input_shape_and_config_list(sys.argv, shape_size=3)

    dtype = torch.bfloat16
    shuffle = True
    block_shape_n, block_shape_k = 128, 128
    x, _weight, weight_triton, _x_scale, x_scale_shuffled, w_scale, y = (
        generate_gemm_a8w8_blockscale_inputs(
            *input_shape,
            block_shape_n,
            block_shape_k,
            dtype=dtype,
            layout="TN",
            output=True,
            shuffle=shuffle,
        )
    )

    for config in config_list:
        assert config is None or config["BLOCK_SIZE_K"] == 128

        def fn(config=config):
            ############################################################
            # <run API>
            gemm_a8w8_blockscale_preshuffle(
                x, weight_triton, x_scale_shuffled, w_scale, dtype, y, config=config
            )
            ############################################################

        run_profile(fn)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: MIT
"""Profile explicit configurations for gemm_afp4wfp4."""


def main():
    import sys

    from ..parameters import driver_help

    if driver_help(sys.argv):
        return 0

    import torch

    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
    from aiter.tuning.search.workloads.gemm.basic.gemm_afp4wfp4 import (
        generate_gemm_afp4wfp4_inputs,
    )

    from ..measurement import run_profile
    from ..parameters import (
        get_input_shape_and_config_list,
    )

    input_shape, config_list = get_input_shape_and_config_list(sys.argv, shape_size=3)

    dtype = torch.bfloat16
    shuffle = False
    (
        x,
        _w,
        w_triton,
        _x_scales,
        _w_scales,
        x_scales_triton,
        w_scales_triton,
        _out_dtype,
        y,
    ) = generate_gemm_afp4wfp4_inputs(
        *input_shape,
        dtype,
        output=True,
        shuffle_scales_fg=shuffle,
        shuffle_weight_fg=shuffle
    )

    for config in config_list:

        def fn(config=config):
            ############################################################
            # <run API>
            gemm_afp4wfp4(
                x, w_triton, x_scales_triton, w_scales_triton, dtype, y, config=config
            )
            ############################################################

        run_profile(fn)


if __name__ == "__main__":
    main()

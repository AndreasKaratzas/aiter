# SPDX-License-Identifier: MIT
"""Profile explicit configurations for gemm_a8w8."""


def main():
    import sys

    from ..parameters import driver_help

    if driver_help(sys.argv):
        return 0

    import torch

    from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8
    from aiter.ops.triton.utils.gemm_config_utils import compute_splitk_params
    from aiter.ops.triton.utils.types import get_fp8_dtypes
    from aiter.tuning.search.workloads.gemm.basic.gemm_a8w8 import (
        generate_gemm_a8w8_inputs,
    )

    from ..measurement import run_profile
    from ..parameters import (
        get_input_shape_and_config_list,
    )

    input_shape, config_list = get_input_shape_and_config_list(sys.argv, shape_size=3)
    _M, _N, K = input_shape

    _, e4m3_type = get_fp8_dtypes()
    dtype = torch.bfloat16
    x, _weight, weight_triton, x_scale, w_scale, _bias, y = generate_gemm_a8w8_inputs(
        *input_shape,
        in_dtype=e4m3_type,
        out_dtype=dtype,
        layout="TN",
        output=True,
    )

    for config in config_list:
        if config is not None:
            compute_splitk_params(config, K)

        def fn(config=config):
            ############################################################
            # <run API>
            gemm_a8w8(x, weight_triton, x_scale, w_scale, None, dtype, y, config=config)
            ############################################################

        run_profile(fn)


if __name__ == "__main__":
    main()

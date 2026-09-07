#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM

#undef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_CONVERSIONS__

#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <numeric>

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <torch/extension.h>

#include "aiter_ck_blockscale.h"

template <typename DDataType, typename EDataType, typename GemmInstance>
__forceinline__ torch::Tensor gemm_a8w8_blockscale_impl(torch::Tensor& XQ,
                                                        torch::Tensor& WQ,
                                                        torch::Tensor& x_scale,
                                                        torch::Tensor& w_scale,
                                                        torch::Tensor& Y,
                                                        int KBatch = 1)
{
    int M = XQ.size(0);
    int N = WQ.size(0);
    int K = XQ.size(1);

    int StrideA = XQ.stride(-2);
    int StrideB = WQ.stride(-2);
    int StrideE = N;

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};

    constexpr ck::index_t NumDTensor = DsDataType::Size();

    // do legacy GEMM
    auto device_gemm = GemmInstance{};
    auto invoker     = device_gemm.MakeInvoker();
    auto argument    = device_gemm.MakeArgument(XQ.data_ptr(),
                                             WQ.data_ptr(),
                                             std::array<const void*, NumDTensor>{},
                                             reinterpret_cast<EDataType*>(Y.data_ptr()),
                                             M,
                                             N,
                                             K,
                                             StrideA,
                                             StrideB,
                                             std::array<ck::index_t, NumDTensor>{},
                                             StrideE,
                                             reinterpret_cast<DDataType*>(x_scale.data_ptr()),
                                             reinterpret_cast<DDataType*>(w_scale.data_ptr()),
                                             a_element_op,
                                             b_element_op,
                                             cde_element_op);

    TORCH_CHECK(KBatch >= 1, "KBatch must be >= 1, got ", KBatch);

    if(KBatch > 1)
    {
        device_gemm.SetKBatch(&argument, KBatch);
    }

    TORCH_CHECK(device_gemm.IsSupportedArgument(argument), "This GEMM is not supported!");

    invoker.Run(argument, StreamConfig{at::hip::getCurrentHIPStream()});
    return Y;
}

#endif // USE_ROCM

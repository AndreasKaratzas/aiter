// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_ck_blockscale.h"
#include <hip/hip_runtime.h>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string error_message;
struct Plan {
    virtual ~Plan() = default;
    virtual void launch(const void*, const void*, const float*, const float*, void*, hipStream_t) const = 0;
};
template <class Output>
struct TypedPlan final : Plan {
    using Kernel = DeviceLegacyGemmHelperF8BlockScale<
        FP32, Output, 256, 1, 128, 128, 16, 128, 256, 16, 16, 16, 16, 1, 2,
        S<16, 16, 1>, S<16, 16, 1>, 1, 2, S<1, 16, 1, 16>, S<8>>;
    typename Kernel::Argument argument;
    TypedPlan(int m, int n, int k)
        : argument(Kernel::MakeArgument(nullptr, nullptr, {}, nullptr, m, n, k,
                                        k, k, {}, n, nullptr, nullptr,
                                        PassThrough{}, PassThrough{}, PassThrough{})) {
        if (!Kernel::IsSupportedArgument(argument))
            throw std::invalid_argument("CK rejected the prepared shape or device");
    }
    void launch(const void* x, const void* w, const float* xs, const float* ws,
                void* out, hipStream_t stream) const override {
        // Stack invocation state permits concurrent use of an immutable plan.
        auto call = argument;
        call.p_a_grid = static_cast<const FP8*>(x);
        call.p_b_grid = static_cast<const FP8*>(w);
        call.p_a_scale_grid = xs;
        call.p_b_scale_grid = ws;
        call.p_c_grid = static_cast<Output*>(out);
        typename Kernel::Invoker{}.Run(call, StreamConfig{stream});
    }
};
int failure(int code, const char* message) { error_message = message; return code; }
}
extern "C" {
int aiter_ck_blockscale_abi_version() { return 1; }
const char* aiter_ck_blockscale_last_error() { return error_message.c_str(); }
int aiter_ck_blockscale_prepare(int m, int n, int k, int output_dtype, void** handle) {
    error_message.clear();
    if (!handle) return failure(1, "handle output is null");
    *handle = nullptr;
    if (m <= 0 || n <= 0 || k <= 0 || m % 16 || n % 128 || k % 256)
        return failure(2, "CK requires positive M%16=0, N%128=0, K%256=0");
    if (output_dtype != 0 && output_dtype != 1)
        return failure(1, "output dtype must be 0 (FP16) or 1 (BF16)");
    try {
        std::unique_ptr<Plan> plan;
        if (output_dtype == 0) plan = std::make_unique<TypedPlan<FP16>>(m, n, k);
        else plan = std::make_unique<TypedPlan<BF16>>(m, n, k);
        *handle = plan.release();
        return 0;
    } catch (const std::invalid_argument& e) { return failure(2, e.what()); }
      catch (const std::exception& e) { return failure(3, e.what()); }
      catch (...) { return failure(3, "unknown CK preparation error"); }
}
int aiter_ck_blockscale_launch(const void* handle, const void* x, const void* w,
                              const float* x_scale, const float* w_scale,
                              void* out, void* stream) {
    error_message.clear();
    if (!handle || !x || !w || !x_scale || !w_scale || !out)
        return failure(1, "plan and tensor pointers must not be null");
    try {
        static_cast<const Plan*>(handle)->launch(x, w, x_scale, w_scale, out,
                                                static_cast<hipStream_t>(stream));
        const hipError_t status = hipGetLastError();
        return status == hipSuccess ? 0 : failure(3, hipGetErrorString(status));
    } catch (const std::exception& e) { return failure(3, e.what()); }
      catch (...) { return failure(3, "unknown CK enqueue error"); }
}
void aiter_ck_blockscale_destroy(void* handle) { delete static_cast<Plan*>(handle); }
}

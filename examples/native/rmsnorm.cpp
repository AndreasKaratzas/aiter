// SPDX-License-Identifier: MIT
#include <aiter/aiter.h>
#include <cmath>
#include <cstdio>
#include <hip/hip_runtime_api.h>
#include <vector>

// Run: aiter_rmsnorm_example /absolute/path/to/libaiter_rmsnorm_backend.so
int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::fprintf(stderr, "usage: %s RMSNORM_BACKEND_LIBRARY\n", argv[0]);
        return 2;
    }
    constexpr int rows = 16, hidden = 1024;
    std::vector<float> input(rows * hidden, 1.0f), weight(hidden, 2.0f), output(rows * hidden);
    float *x = nullptr, *w = nullptr, *y = nullptr;
    auto hip_check = [](hipError_t result) {
        if(result != hipSuccess)
        {
            std::fprintf(stderr, "%s\n", hipGetErrorString(result));
            return false;
        }
        return true;
    };
    if(!hip_check(hipSetDevice(0)) || !hip_check(hipMalloc(&x, input.size() * sizeof(float))) ||
       !hip_check(hipMalloc(&w, weight.size() * sizeof(float))) ||
       !hip_check(hipMalloc(&y, output.size() * sizeof(float))))
        return 1;
    if(!hip_check(
           hipMemcpy(x, input.data(), input.size() * sizeof(float), hipMemcpyHostToDevice)) ||
       !hip_check(
           hipMemcpy(w, weight.data(), weight.size() * sizeof(float), hipMemcpyHostToDevice)))
        return 1;
    aiter_rmsnorm_desc description{sizeof(aiter_rmsnorm_desc),
                                   AITER_ABI_VERSION,
                                   0,
                                   AITER_FP32,
                                   rows,
                                   hidden,
                                   hidden,
                                   1e-6f,
                                   0};
    aiter_plan* plan = nullptr;
    if(aiter_rmsnorm_prepare(&description, argv[1], &plan) != AITER_SUCCESS)
    {
        std::fprintf(stderr, "prepare: %s\n", aiter_last_error());
        return 1;
    }
    aiter_buffer buffers[] = {{x, input.size() * sizeof(float)},
                              {w, weight.size() * sizeof(float)},
                              {y, output.size() * sizeof(float)}};
    if(aiter_execute(plan, buffers, 3, nullptr) != AITER_SUCCESS)
    {
        std::fprintf(stderr, "execute: %s\n", aiter_last_error());
        return 1;
    }
    if(!hip_check(
           hipMemcpy(output.data(), y, output.size() * sizeof(float), hipMemcpyDeviceToHost)))
        return 1;
    for(float value : output)
        if(std::abs(value - 2.0f / std::sqrt(1.0f + 1e-6f)) > 1e-5f)
            return 1;
    aiter_plan_destroy(plan);
    if(!hip_check(hipFree(x)) || !hip_check(hipFree(w)) || !hip_check(hipFree(y)))
        return 1;
    std::puts("AITER native RMSNorm passed without Python or Torch.");
}

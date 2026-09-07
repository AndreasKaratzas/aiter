// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include AITER_TEST_HEADER
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>

static void check(hipError_t status)
{
    if(status != hipSuccess)
        throw std::runtime_error(hipGetErrorString(status));
}

static void run(int m, int n, int k, hipStream_t stream)
{
    std::vector<__half> a(m * k), b(k * n), c(m * n);
    for(size_t i = 0; i < a.size(); ++i)
        a[i] = __float2half((static_cast<int>((i * 17 + 3) % 31) - 15) / 16.0f);
    for(size_t i = 0; i < b.size(); ++i)
        b[i] = __float2half((static_cast<int>((i * 7 + 1) % 23) - 11) / 16.0f);

    __half *da, *db, *dc;
    check(hipMalloc(&da, a.size() * sizeof(__half)));
    check(hipMalloc(&db, b.size() * sizeof(__half)));
    check(hipMalloc(&dc, c.size() * sizeof(__half)));
    check(hipMemcpyAsync(da, a.data(), a.size() * sizeof(__half), hipMemcpyHostToDevice, stream));
    check(hipMemcpyAsync(db, b.data(), b.size() * sizeof(__half), hipMemcpyHostToDevice, stream));
    check(AITER_TEST_KERNEL(stream, dc, da, db, m, n, k, n, k, n));
    check(hipMemcpyAsync(c.data(), dc, c.size() * sizeof(__half), hipMemcpyDeviceToHost, stream));
    check(hipStreamSynchronize(stream));

    // These small dyadic inputs accumulate exactly in FP32. Only the final
    // FP16 rounding remains, so the comparison needs no numerical tolerance.
    for(int row = 0; row < m; ++row)
        for(int col = 0; col < n; ++col)
        {
            float expected = 0;
            for(int inner = 0; inner < k; ++inner)
                expected += __half2float(a[row * k + inner]) * __half2float(b[inner * n + col]);
            if(__half2float(c[row * n + col]) != __half2float(__float2half(expected)))
                throw std::runtime_error(
                    "generated GEMM differs from the independent CPU reference");
        }
    check(hipFree(da));
    check(hipFree(db));
    check(hipFree(dc));
}

int main()
{
    try
    {
        check(hipSetDevice(0));
        hipStream_t stream;
        check(hipStreamCreate(&stream));
        int cases = 0;
        for(int m : {16, 32, 64})
            for(int n : {16, 32, 64})
                for(int k : {16, 32})
                {
                    run(m, n, k, stream);
                    ++cases;
                }
        run(19, 23, 21, stream);
        check(hipStreamDestroy(stream));
        std::cout << "validated_shapes=" << cases + 1 << '\n';
        return 0;
    }
    catch(const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

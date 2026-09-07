// SPDX-License-Identifier: MIT
// Independent C ABI acceptance checks. This executable has no Python dependency.
#include "aiter/aiter.h"
#include <hip/hip_runtime_api.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
int checks = 0;
void expect(bool value, const char* message) {
    ++checks;
    if (!value) {
        std::fprintf(stderr, "FAIL: %s (%s)\n", message, aiter_last_error());
        std::exit(1);
    }
}
void hip_ok(hipError_t status) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "HIP: %s\n", hipGetErrorString(status));
        std::exit(1);
    }
}
void check_values(const std::vector<float>& actual, const std::vector<float>& x,
                  const std::vector<float>& weight, int rows, int hidden) {
    for (int row = 0; row < rows; ++row) {
        float square = 0;
        for (int col = 0; col < hidden; ++col) {
            float value = x[row * hidden + col];
            square += value * value;
        }
        float factor = 1 / std::sqrt(square / hidden + 1e-6f);
        for (int col = 0; col < hidden; ++col) {
            float expected = x[row * hidden + col] * factor * weight[col];
            if (std::abs(actual[row * hidden + col] - expected) > 2e-5f) {
                expect(false, "native RMSNorm numerical reference");
            }
        }
    }
    expect(true, "native RMSNorm numerical reference");
}

void check_rms_width(const char* backend, int hidden, int stride) {
    constexpr int rows = 7;
    aiter_rmsnorm_desc desc{sizeof(aiter_rmsnorm_desc), AITER_ABI_VERSION, 0,
                            AITER_FP32, rows, hidden, stride, 1e-6f, 0};
    aiter_plan* plan = nullptr;
    expect(aiter_rmsnorm_prepare(&desc, backend, &plan) == AITER_SUCCESS,
           "native RMSNorm small or row-strided preparation");
    const size_t input_elements = size_t(rows - 1) * stride + hidden;
    std::vector<float> x(input_elements, -99), dense(rows * hidden), weight(hidden), out(rows * hidden);
    for (int row = 0; row < rows; ++row)
        for (int col = 0; col < hidden; ++col) {
            float value = std::sin(float(row * hidden + col) * 0.03f);
            x[row * stride + col] = value;
            dense[row * hidden + col] = value;
        }
    for (int col = 0; col < hidden; ++col) weight[col] = 0.5f + float(col % 11) * 0.1f;
    float *dx, *dw, *dy;
    hip_ok(hipMalloc(&dx, x.size() * sizeof(float)));
    hip_ok(hipMalloc(&dw, weight.size() * sizeof(float)));
    hip_ok(hipMalloc(&dy, out.size() * sizeof(float)));
    hip_ok(hipMemcpy(dx, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice));
    hip_ok(hipMemcpy(dw, weight.data(), weight.size() * sizeof(float), hipMemcpyHostToDevice));
    aiter_buffer buffers[] = {{dx, x.size() * sizeof(float)}, {dw, weight.size() * sizeof(float)},
                              {dy, out.size() * sizeof(float)}};
    expect(aiter_execute(plan, buffers, 3, nullptr) == AITER_SUCCESS,
           "native RMSNorm small or row-strided execution");
    hip_ok(hipMemcpy(out.data(), dy, out.size() * sizeof(float), hipMemcpyDeviceToHost));
    check_values(out, dense, weight, rows, hidden);
    aiter_plan_destroy(plan);
    hip_ok(hipFree(dx));
    hip_ok(hipFree(dw));
    hip_ok(hipFree(dy));
}

void check_gemm(const char* backend) {
    constexpr int m = 32, n = 256, k = 512;
    aiter_gemm_desc desc{sizeof(aiter_gemm_desc), AITER_ABI_VERSION, 0, AITER_FP16, m, n, k};
    aiter_plan* plan = nullptr;
    auto bad = desc;
    bad.m = 17;
    expect(aiter_gemm_prepare(&bad, backend, &plan) == AITER_UNSUPPORTED,
           "native CK rejects unsupported tile shape");
    expect(aiter_gemm_prepare(&desc, backend, &plan) == AITER_SUCCESS,
           "native CK preparation through public C ABI");
    std::vector<unsigned char> x(m * k), w(n * k);
    std::vector<float> xs(m * (k / 128)), ws((n / 128) * (k / 128));
    std::vector<__half> out(m * n);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < k; ++col) x[row * k + col] = row % 2 ? 0x40 : 0x38;
        for (int block = 0; block < k / 128; ++block)
            xs[row * (k / 128) + block] = float((row % 3 + 1) * (block + 1)) / 16;
    }
    for (int row = 0; row < n; ++row)
        for (int col = 0; col < k; ++col) w[row * k + col] = row % 2 ? 0x38 : 0x40;
    for (int blockn = 0; blockn < n / 128; ++blockn)
        for (int blockk = 0; blockk < k / 128; ++blockk)
            ws[blockn * (k / 128) + blockk] = float((blockn + 1) * (blockk + 1)) / 32;
    void *dx, *dw, *dxs, *dws, *dy;
    hip_ok(hipMalloc(&dx, x.size()));
    hip_ok(hipMalloc(&dw, w.size()));
    hip_ok(hipMalloc(&dxs, xs.size() * sizeof(float)));
    hip_ok(hipMalloc(&dws, ws.size() * sizeof(float)));
    hip_ok(hipMalloc(&dy, out.size() * sizeof(__half)));
    hip_ok(hipMemcpy(dx, x.data(), x.size(), hipMemcpyHostToDevice));
    hip_ok(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice));
    hip_ok(hipMemcpy(dxs, xs.data(), xs.size() * sizeof(float), hipMemcpyHostToDevice));
    hip_ok(hipMemcpy(dws, ws.data(), ws.size() * sizeof(float), hipMemcpyHostToDevice));
    aiter_buffer buffers[] = {{dx, x.size()}, {dw, w.size()},
                              {dxs, xs.size() * sizeof(float)},
                              {dws, ws.size() * sizeof(float)},
                              {dy, out.size() * sizeof(__half)}};
    expect(aiter_execute(plan, buffers, 5, nullptr) == AITER_SUCCESS,
           "native CK public C ABI execution");
    hipStream_t stream;
    hipGraph_t graph;
    hipGraphExec_t executable;
    hip_ok(hipStreamCreate(&stream));
    hip_ok(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    expect(aiter_execute(plan, buffers, 5, stream) == AITER_SUCCESS,
           "native CK graph capture");
    hip_ok(hipStreamEndCapture(stream, &graph));
    hip_ok(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    aiter_plan_destroy(plan);
    for (int repeat = 0; repeat < 3; ++repeat) hip_ok(hipGraphLaunch(executable, stream));
    hip_ok(hipStreamSynchronize(stream));
    hip_ok(hipMemcpy(out.data(), dy, out.size() * sizeof(__half), hipMemcpyDeviceToHost));
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float expected = 0;
            for (int block = 0; block < k / 128; ++block)
                expected += 128 * float(row % 2 + 1) * float(2 - col % 2)
                            * xs[row * (k / 128) + block]
                            * ws[(col / 128) * (k / 128) + block];
            if (std::abs(float(out[row * n + col]) - expected) > 0.02f)
                expect(false, "native CK scaled FP8 numerical reference");
        }
    }
    expect(true, "native CK scaled FP8 numerical reference");
    hip_ok(hipGraphExecDestroy(executable));
    hip_ok(hipGraphDestroy(graph));
    hip_ok(hipStreamDestroy(stream));
    for (void* pointer : {dx, dw, dxs, dws, dy}) hip_ok(hipFree(pointer));
}
}

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        std::fprintf(stderr, "usage: test_runtime /absolute/path/to/rmsnorm_backend.so [ck_backend.so]\n");
        return 2;
    }
    hip_ok(hipSetDevice(0));
    expect(aiter_abi_version() == AITER_ABI_VERSION, "public ABI version");
    expect(std::strcmp(aiter_status_string(AITER_SUCCESS), "success") == 0,
           "public status text");
    aiter_rmsnorm_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.abi_version = AITER_ABI_VERSION;
    desc.device = 0;
    desc.dtype = AITER_FP32;
    desc.rows = 16;
    desc.hidden = 128;
    desc.input_row_stride = 128;
    desc.epsilon = 1e-6f;
    aiter_plan* plan = nullptr;
    expect(aiter_rmsnorm_prepare(nullptr, argv[1], &plan) == AITER_INVALID_ARGUMENT,
           "null descriptor rejected");
    expect(plan == nullptr, "failed preparation clears output handle");
    expect(aiter_rmsnorm_prepare(&desc, argv[1], nullptr) == AITER_INVALID_ARGUMENT,
           "null output handle rejected");
    auto bad = desc;
    bad.abi_version += 1;
    expect(aiter_rmsnorm_prepare(&bad, argv[1], &plan) == AITER_INVALID_ARGUMENT,
           "unknown ABI rejected");
    bad = desc;
    bad.struct_size = 4;
    expect(aiter_rmsnorm_prepare(&bad, argv[1], &plan) == AITER_INVALID_ARGUMENT,
           "short descriptor rejected");
    bad = desc;
    bad.epsilon = NAN;
    expect(aiter_rmsnorm_prepare(&bad, argv[1], &plan) == AITER_INVALID_ARGUMENT,
           "nonfinite epsilon rejected");
    bad = desc;
    bad.reserved = 1;
    expect(aiter_rmsnorm_prepare(&bad, argv[1], &plan) == AITER_INVALID_ARGUMENT,
           "unknown reserved flag rejected");
    expect(aiter_rmsnorm_prepare(&desc, "/missing/aiter-backend.so", &plan) == AITER_INVALID_ARGUMENT,
           "missing library reports an error");
    expect(std::strlen(aiter_last_error()) > 0, "load error has a diagnostic");
    expect(aiter_rmsnorm_prepare(&desc, argv[1], &plan) == AITER_SUCCESS,
           "native RMSNorm preparation");
    expect(std::strlen(aiter_last_error()) == 0, "successful preparation clears last error");

    const size_t elements = desc.rows * desc.hidden;
    std::vector<float> x(elements), weight(desc.hidden), actual(elements);
    for (size_t i = 0; i < elements; ++i) x[i] = std::sin(float(i) * 0.03f);
    for (int i = 0; i < desc.hidden; ++i) weight[i] = 0.5f + float(i % 11) * 0.1f;
    float *dx, *dw, *dy;
    hip_ok(hipMalloc(&dx, elements * sizeof(float)));
    hip_ok(hipMalloc(&dw, weight.size() * sizeof(float)));
    hip_ok(hipMalloc(&dy, elements * sizeof(float)));
    hip_ok(hipMemcpy(dx, x.data(), elements * sizeof(float), hipMemcpyHostToDevice));
    hip_ok(hipMemcpy(dw, weight.data(), weight.size() * sizeof(float), hipMemcpyHostToDevice));
    aiter_buffer buffers[] = {{dx, elements * sizeof(float)},
                              {dw, weight.size() * sizeof(float)},
                              {dy, elements * sizeof(float)}};
    expect(aiter_execute(plan, buffers, 2, nullptr) == AITER_INVALID_ARGUMENT,
           "wrong buffer count rejected");
    buffers[2].size_bytes = 4;
    expect(aiter_execute(plan, buffers, 3, nullptr) == AITER_INVALID_ARGUMENT,
           "short output rejected");
    buffers[2] = buffers[0];
    expect(aiter_execute(plan, buffers, 3, nullptr) == AITER_INVALID_ARGUMENT,
           "output alias rejected");
    buffers[2] = {dy, elements * sizeof(float)};
    buffers[0].data = reinterpret_cast<char*>(dx) + 4;
    expect(aiter_execute(plan, buffers, 3, nullptr) == AITER_INVALID_ARGUMENT,
           "unaligned input rejected");
    buffers[0].data = dx;
    expect(aiter_execute(plan, buffers, 3, nullptr) == AITER_SUCCESS,
           "native execution accepts valid bindings");
    hip_ok(hipMemcpy(actual.data(), dy, elements * sizeof(float), hipMemcpyDeviceToHost));
    check_values(actual, x, weight, desc.rows, desc.hidden);

    int devices = 0;
    hip_ok(hipGetDeviceCount(&devices));
    expect(devices >= 2, "cross-device tests require two visible GPUs");
    hip_ok(hipSetDevice(1));
    hipStream_t foreign_stream;
    float* foreign_buffer;
    hip_ok(hipStreamCreate(&foreign_stream));
    hip_ok(hipMalloc(&foreign_buffer, elements * sizeof(float)));
    expect(aiter_execute(plan, buffers, 3, nullptr) == AITER_INVALID_ARGUMENT,
           "wrong calling device rejected");
    hip_ok(hipSetDevice(0));
    expect(aiter_execute(plan, buffers, 3, foreign_stream) == AITER_INVALID_ARGUMENT,
           "foreign stream rejected before launch");
    buffers[0].data = foreign_buffer;
    expect(aiter_execute(plan, buffers, 3, nullptr) == AITER_INVALID_ARGUMENT,
           "foreign device buffer rejected before launch");
    buffers[0].data = dx;
    hip_ok(hipSetDevice(1));
    hip_ok(hipFree(foreign_buffer));
    hip_ok(hipStreamDestroy(foreign_stream));
    hip_ok(hipSetDevice(0));

    hipStream_t stream;
    hipGraph_t graph;
    hipGraphExec_t executable;
    hip_ok(hipStreamCreate(&stream));
    hip_ok(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    expect(aiter_execute(plan, buffers, 3, stream) == AITER_SUCCESS,
           "native execution during capture");
    hip_ok(hipStreamEndCapture(stream, &graph));
    hip_ok(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    aiter_plan_destroy(plan);
    plan = nullptr;
    for (int repeat = 0; repeat < 3; ++repeat) hip_ok(hipGraphLaunch(executable, stream));
    hip_ok(hipStreamSynchronize(stream));
    hip_ok(hipMemcpy(actual.data(), dy, elements * sizeof(float), hipMemcpyDeviceToHost));
    check_values(actual, x, weight, desc.rows, desc.hidden);
    hip_ok(hipGraphExecDestroy(executable));
    hip_ok(hipGraphDestroy(graph));
    hip_ok(hipStreamDestroy(stream));
    hip_ok(hipFree(dx));
    hip_ok(hipFree(dw));
    hip_ok(hipFree(dy));
    aiter_plan_destroy(nullptr);
    check_rms_width(argv[1], 4, 4);
    check_rms_width(argv[1], 409, 413);
    if (argc == 3) check_gemm(argv[2]);
    std::printf("%d native API checks passed\n", checks);
    return 0;
}

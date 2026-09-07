// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_ctypes_error.h"
#include "aiter_tensor.h"
#include "asm_bf16gemm_configs.hpp"
#include <climits>
#include <cmath>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <optional>

struct __attribute__((packed)) KernelArgs
{
    void* ptr_D;
    p2 _p0;
    void* ptr_C;
    p2 _p1;
    void* ptr_A;
    p2 _p2;
    void* ptr_B;
    p2 _p3;
    float alpha;
    p3 _p4;
    float beta;
    p3 _p5;
    unsigned int stride_D0;
    p3 _p6;
    unsigned int stride_D1;
    p3 _p7;
    unsigned int stride_C0;
    p3 _p8;
    unsigned int stride_C1;
    p3 _p9;
    unsigned int stride_A0;
    p3 _p10;
    unsigned int stride_A1;
    p3 _p11;
    unsigned int stride_B0;
    p3 _p12;
    unsigned int stride_B1;
    p3 _p13;
    unsigned int M;
    p3 _p14;
    unsigned int N;
    p3 _p15;
    unsigned int K;
    p3 _p16;
    unsigned int splitk;
    p3 _p17;
    unsigned int is_out_b16;
    p3 _p18;
    void* ptr_Bias;
    p2 _p19;
    unsigned int add_bias;
    p3 _p20;
    void* ptr_semaphore;
    p2 _p21;
};

std::tuple<std::string, int> get_heuristic_kernel(int M,
                                                  int N,
                                                  int K,
                                                  CFG* cfgs,
                                                  std::string arch_id,
                                                  bool bpreshuffle,
                                                  bool require_fp32,
                                                  int splitk             = -1,
                                                  const char* kernelName = nullptr)
{
    AITER_CHECK(splitk == -1 || (splitk >= 1 && splitk <= 16),
                "splitK must be automatic or an integer from 1 through 16");
    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
    const uint32_t num_cu = dev_prop.multiProcessorCount;
    uint32_t empty_cu     = num_cu;
    uint32_t round        = UINT32_MAX;
    float efficiency      = 1.0;
    int oob               = M;
    std::string selected;
    int selected_split = 1;
    for(const auto& [name, cfg] : *cfgs)
    {
        if(name.find(arch_id) != 0 || (kernelName && name != arch_id + kernelName))
            continue;
        if(require_fp32 && cfg.supports_fp32 == 0)
            continue;
        if(cfg.tileM <= 0 || cfg.tileN <= 0 || cfg.subK <= 0 || N % cfg.tileN != 0 ||
           cfg.bPreshuffle != (bpreshuffle ? 1 : 0))
            continue;
        const uint32_t tiles = ((M + cfg.tileM - 1) / cfg.tileM) * (N / cfg.tileN);
        int split            = 1;
        if(splitk >= 1)
        {
            if((splitk > 1) != (cfg.splitK == 1) || splitk > K / cfg.subK)
                continue;
            split = splitk;
        }
        else if(cfg.splitK == 1)
        {
            if(K / cfg.subK < 2)
                continue;
            split = std::max(2, std::min({static_cast<int>(num_cu / tiles), 16, K / cfg.subK}));
        }
        if(split > 1 && tiles > 1024)
            continue;
        const uint32_t groups      = tiles * split;
        const uint32_t local_round = (groups + num_cu - 1) / num_cu;
        const uint32_t local_empty = local_round * num_cu - groups;
        const float local_efficiency =
            static_cast<float>(cfg.tileM * cfg.tileN) / (cfg.tileM + cfg.tileN);
        const int local_oob = M % cfg.tileM == 0 ? 0 : cfg.tileM - M % cfg.tileM;
        if(kernelName || local_round < round ||
           (local_round == round && (local_empty < empty_cu || local_oob < oob)) ||
           (local_round == round && local_empty == empty_cu && local_oob == oob &&
            local_efficiency > efficiency))
        {
            selected       = name;
            selected_split = split;
            round          = local_round;
            empty_cu       = local_empty;
            efficiency     = local_efficiency;
            oob            = local_oob;
            if(kernelName)
                break;
        }
    }
    AITER_CHECK(
        !selected.empty(),
        "No assembly kernel supports the exact shape, output, "
        "preshuffle and splitK request (BF16-only kernels cannot produce FP32 or fuse bias)");
    return {selected, selected_split};
}

AiterAsmKernel* get_or_load_kernel(const std::string& selectedKernelName,
                                   CFG* config_map,
                                   unsigned int& SUBM,
                                   unsigned int& SUBN)
{
    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;

    auto it_kl = config_map->find(selectedKernelName);
    AITER_CHECK(it_kl != config_map->end(), __func__, " not find kernel~ " + selectedKernelName);

    const auto& cfg     = it_kl->second;
    const char* name    = cfg.knl_name.c_str();
    const char* co_name = cfg.co_name.c_str();
    SUBM                = cfg.tileM;
    SUBN                = cfg.tileN;

    return &impl_ptr_map.get_or_create(name, [&]() { return AiterAsmKernel(name, co_name); });
}

// Descriptor arithmetic is bounded by this assembly ABI's 32-bit offsets.
// Python additionally checks the real storage capacity before creating descriptors.
size_t checked_span(const aiter_tensor_t* tensor)
{
    AITER_CHECK(tensor && tensor->ptr && tensor->is_gpu(), "Expected a nonempty GPU tensor");
    AITER_CHECK(tensor->ndim > 0 && tensor->ndim <= 2, "Expected a vector or matrix");
    uint64_t span  = 1;
    uint64_t count = 1;
    for(int i = 0; i < tensor->ndim; ++i)
    {
        AITER_CHECK(tensor->shape[i] > 0 && tensor->shape[i] <= INT_MAX &&
                        tensor->strides[i] >= 0 && tensor->strides[i] <= UINT32_MAX,
                    "Invalid tensor shape or stride");
        count *= tensor->shape[i];
        span += static_cast<uint64_t>(tensor->shape[i] - 1) * tensor->strides[i];
        AITER_CHECK(span <= UINT32_MAX && count <= UINT32_MAX, "Tensor span is too large");
    }
    AITER_CHECK(count == tensor->numel(), "Tensor shape and element count disagree");
    span *= tensor->element_size();
    AITER_CHECK(span <= UINT32_MAX &&
                    reinterpret_cast<uintptr_t>(tensor->ptr) <= UINTPTR_MAX - span,
                "Tensor exceeds assembly address range");
    return span;
}

void disjoint(const aiter_tensor_t* output, const aiter_tensor_t* input)
{
    if(!input || !input->numel())
        return;
    const uintptr_t a = reinterpret_cast<uintptr_t>(output->ptr);
    const uintptr_t b = reinterpret_cast<uintptr_t>(input->ptr);
    const size_t as = checked_span(output), bs = checked_span(input);
    AITER_CHECK(a + as <= b || b + bs <= a, "GEMM writable storage overlaps another tensor");
}

__global__ void finish_gemm(const float* accumulated,
                            void* output,
                            const void* bias,
                            size_t elements,
                            int columns,
                            bool bf16_output,
                            bool bf16_bias)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(index >= elements)
        return;
    float value = accumulated[index];
    if(bias)
        value += bf16_bias
                     ? static_cast<float>(static_cast<const hip_bfloat16*>(bias)[index % columns])
                     : static_cast<const float*>(bias)[index % columns];
    if(bf16_output)
        static_cast<hip_bfloat16*>(output)[index] = hip_bfloat16(value);
    else
        static_cast<float*>(output)[index] = value;
}

AITER_CTYPES_ERROR_DEF

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    gemm_a16w16_asm_with_workspace,
    (aiter_tensor_t * A,
     aiter_tensor_t* B,
     aiter_tensor_t* out,
     aiter_tensor_t* semaphore,
     aiter_tensor_t* workspace,
     aiter_tensor_t* bias,
     int splitK,
     const char* kernelName,
     int bpreshuffle,
     hipStream_t stream),
    (A, B, out, semaphore, workspace, bias, splitK, kernelName, bpreshuffle, stream))
{
    checked_span(A);
    checked_span(B);
    checked_span(out);
    AITER_CHECK(A->dtype() == AITER_DTYPE_bf16 && B->dtype() == AITER_DTYPE_bf16,
                "These assembly kernels support BF16 inputs only");
    AITER_CHECK(out->dtype() == AITER_DTYPE_fp32 || out->dtype() == AITER_DTYPE_bf16,
                "out must have FP32 or BF16 dtype");
    AITER_CHECK(A->ndim == 2 && B->ndim == 2 && out->ndim == 2, "A, B and out must be matrices");
    AITER_CHECK(A->device_id == B->device_id && A->device_id == out->device_id,
                "GEMM tensors must share a GPU");
    const int M = A->size(0), N = B->size(0), K = A->size(1);
    AITER_CHECK(B->size(1) == K && out->size(0) == M && out->size(1) == N && K % 64 == 0 &&
                    N % 64 == 0,
                "Invalid GEMM dimensions or alignment");
    AITER_CHECK(A->stride(1) == 1 && B->stride(1) == 1 && A->stride(0) >= K && B->stride(0) >= K &&
                    out->is_contiguous(),
                "GEMM inputs need contiguous rows and output must be contiguous");
    AITER_CHECK(bpreshuffle == 0 || (bpreshuffle == 1 && B->is_contiguous()),
                "Preshuffled weight must be contiguous");
    disjoint(out, A);
    disjoint(out, B);
    if(bias)
    {
        checked_span(bias);
        AITER_CHECK(bias->device_id == A->device_id && bias->ndim == 1 && bias->size(0) == N &&
                        bias->is_contiguous() &&
                        (bias->dtype() == AITER_DTYPE_bf16 || bias->dtype() == AITER_DTYPE_fp32),
                    "bias must be a BF16 or FP32 vector of length N on the same GPU");
        disjoint(out, bias);
    }
    const bool bf16_output = out->dtype() == AITER_DTYPE_bf16;
    if(bf16_output)
    {
        checked_span(workspace);
        AITER_CHECK(workspace->device_id == A->device_id && workspace->ndim == 2 &&
                        workspace->size(0) == M && workspace->size(1) == N &&
                        workspace->dtype() == AITER_DTYPE_fp32 && workspace->is_contiguous(),
                    "BF16 output requires a contiguous FP32 workspace of shape M,N");
        for(const auto* tensor : {A, B, out, bias})
            disjoint(workspace, tensor);
    }
    const HipDeviceGuard device_guard(A->device_id);
    CFG* config_map        = &cfg_bf16gemm_fp32bf16;
    auto [name, split]     = get_heuristic_kernel(M,
                                                  N,
                                                  K,
                                                  config_map,
                                                  get_gpu_arch(),
                                                  bpreshuffle,
                                                  !bf16_output || bias != nullptr,
                                                  splitK,
                                                  kernelName);
    const auto& config     = config_map->at(name);
    const bool direct_bf16 = config.supports_fp32 == 0;
    AITER_CHECK(!direct_bf16 || (bf16_output && split == 1 && !bias),
                "BF16-only kernel requires single-pass BF16 output without bias");
    const int gdx = N / config.tileN;
    const int gdy = (M + config.tileM - 1) / config.tileM;
    if(split > 1)
    {
        checked_span(semaphore);
        AITER_CHECK(semaphore->device_id == A->device_id && semaphore->dtype() == AITER_DTYPE_u32 &&
                        semaphore->is_contiguous() && semaphore->numel() >= gdx * gdy,
                    "Split-K requires an adequate same-GPU uint32 semaphore");
        for(const auto* tensor : {A, B, out, bias, bf16_output ? workspace : nullptr})
            disjoint(semaphore, tensor);
    }
    aiter_tensor_t* accumulated = bf16_output && !direct_bf16 ? workspace : out;
    KernelArgs args             = {};
    args.ptr_D                  = accumulated->ptr;
    args.ptr_A                  = A->ptr;
    args.ptr_B                  = B->ptr;
    args.alpha                  = 1.0f;
    args.stride_A0              = A->stride(0) * A->element_size();
    args.stride_B0              = B->stride(0) * B->element_size();
    args.stride_C0 = args.stride_D0 = N * accumulated->element_size();
    args.M                          = M;
    args.N                          = N;
    args.K                          = K;
    args.is_out_b16                 = direct_bf16 ? 1 : 0;
    args.splitk                     = split;
    args.ptr_semaphore              = split > 1 ? semaphore->ptr : nullptr;
    unsigned int tile_m, tile_n;
    AiterAsmKernel* kernel = get_or_load_kernel(name, config_map, tile_m, tile_n);
    size_t arg_size        = sizeof(args);
    kernel->launch_kernel({&args, &arg_size, gdx, gdy, split, 256, 1, 1, stream});
    if((bf16_output && !direct_bf16) || bias)
    {
        const size_t elements = static_cast<size_t>(M) * N;
        finish_gemm<<<(elements + 255) / 256, 256, 0, stream>>>(
            static_cast<const float*>(accumulated->ptr),
            out->ptr,
            bias ? bias->ptr : nullptr,
            elements,
            N,
            bf16_output,
            bias && bias->dtype() == AITER_DTYPE_bf16);
        HIP_CALL(hipGetLastError());
    }
}

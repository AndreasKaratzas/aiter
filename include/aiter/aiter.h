/* SPDX-License-Identifier: MIT */
#ifndef AITER_PUBLIC_API_H
#define AITER_PUBLIC_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AITER_ABI_VERSION 1u
#if defined(__GNUC__)
#define AITER_API __attribute__((visibility("default")))
#else
#define AITER_API
#endif

typedef enum aiter_status
{
    AITER_SUCCESS          = 0,
    AITER_INVALID_ARGUMENT = 1,
    AITER_UNSUPPORTED      = 2,
    AITER_RUNTIME_ERROR    = 3
} aiter_status;

typedef enum aiter_dtype
{
    AITER_FP16 = 0,
    AITER_BF16 = 1,
    AITER_FP32 = 2
} aiter_dtype;

typedef struct aiter_plan aiter_plan;

/* Descriptors are copied during preparation. No caller descriptor is retained. */
/* ABI 1 native providers currently qualify gfx950. backend_library is a file
 * containing the matching versioned provider ABI and GPU code object. AITER
 * retains executable code for the process lifetime so captured graphs remain
 * valid after plan destruction. It does not own the caller's device buffers.
 */
typedef struct aiter_rmsnorm_desc
{
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t device;
    aiter_dtype dtype;
    int64_t rows;
    int64_t hidden;
    int64_t input_row_stride;
    float epsilon;
    uint32_t reserved;
} aiter_rmsnorm_desc;

/* Ordinary FP8 E4M3FN values on gfx950, FP32 scales in 128x128 blocks. */
typedef struct aiter_gemm_desc
{
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t device;
    aiter_dtype output_dtype;
    int64_t m;
    int64_t n;
    int64_t k;
} aiter_gemm_desc;

typedef struct aiter_buffer
{
    void* data;
    size_t size_bytes;
} aiter_buffer;

/* Buffer order: RMSNorm input, weight, output; GEMM x, w, x_scale, w_scale, output. */
AITER_API uint32_t aiter_abi_version(void);
AITER_API const char* aiter_status_string(aiter_status status);
AITER_API const char* aiter_last_error(void);

AITER_API aiter_status aiter_rmsnorm_prepare(const aiter_rmsnorm_desc* desc,
                                             const char* backend_library,
                                             aiter_plan** output);
AITER_API aiter_status aiter_gemm_prepare(const aiter_gemm_desc* desc,
                                          const char* backend_library,
                                          aiter_plan** output);

/* Success means accepted for enqueue, not GPU completion. The caller owns
 * buffers/stream and retains them until completion, including graph replays.
 * Execution does not allocate GPU memory, compile, search or synchronize.
 * Plans are immutable and reusable with independent output bindings/streams.
 * The calling thread must select the descriptor's HIP device before execution.
 */
AITER_API aiter_status aiter_execute(const aiter_plan* plan,
                                     const aiter_buffer* buffers,
                                     size_t buffer_count,
                                     void* hip_stream);
AITER_API void aiter_plan_destroy(aiter_plan* plan);

#ifdef __cplusplus
}
#endif
#endif
